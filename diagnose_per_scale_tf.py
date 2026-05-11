"""Per-scale teacher-forcing diagnostic for the SkelVQ-FSQ AR.

Sweeps `gt_prefix_count = 0..4`: for each k, the AR generates with the first k
scales forced to GT indices and scales k..3 sampled. k=0 is pure AR (matches
production); k=4 is full TF (matches encode->decode). FID at each k localizes
where the cascade drift takes off.

The encoder produces GT indices from the ground-truth motion in the batch.

Run:
    python diagnose_per_scale_tf.py --name skelvq_ar_cluster \\
        --tokenizer_ckpt checkpoints/t2m/skelvq_fsq/model/net_best_fid.tar \\
        --ar_ckpt /scratch/zhiweiz/MoScale++/ckpt/t2m/skelvq_ar_cluster/model/net_best_loss.tar \\
        --repeat_times 3
"""
from __future__ import annotations

import argparse
import os
import sys
from os.path import join as pjoin

HERE = os.path.dirname(os.path.abspath(__file__))
# Same sys.path shape as eval_skelvq_ar.py: SALAD root first (default), moscale
# appended (lower precedence) — keeps SALAD's utils.* visible while still
# allowing `from model.transformer.moscale_fsq import ...`.
sys.path.append(os.path.join(HERE, "moscale"))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.t2m_dataset import Text2MotionDatasetEval, collate_fn as eval_collate
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.get_opt import get_opt
from utils.skelvq_ar_eval import evaluate_once
from utils.word_vectorizer import WordVectorizer

from model.vq.skelvq_wrapper import SkelVQWrapper       # type: ignore
from model.transformer.moscale_fsq import MoScaleFSQ    # type: ignore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--tokenizer_ckpt", required=True)
    p.add_argument("--ar_ckpt", required=True)
    p.add_argument("--dataset_name", default="t2m")
    p.add_argument("--data_root", default="./dataset/humanml3d/")
    p.add_argument("--dataset_opt_path", default="./checkpoints/t2m/Comp_v6_KLD005/opt.txt")
    p.add_argument("--glove_dir", default="./glove")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cond_scale", type=float, default=4.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repeat_times", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading tokenizer: {args.tokenizer_ckpt}")
    vq_model = SkelVQWrapper(args.tokenizer_ckpt, device=args.device)
    n_scales = len(vq_model.scales)
    print(f"  scales={vq_model.scales}  J_b={vq_model.J_b}  eff_levels={vq_model.effective_levels}")

    print(f"Loading AR: {args.ar_ckpt}")
    ar_state = torch.load(args.ar_ckpt, map_location=args.device, weights_only=False)
    ar_opt = ar_state["opt"]
    cfg = OmegaConf.create({
        "model": dict(latent_dim=ar_opt["latent_dim"],
                      head_latent_dim=ar_opt.get("head_latent_dim", -1) if ar_opt.get("head_latent_dim", -1) > 0 else ar_opt["latent_dim"],
                      num_layers=ar_opt["num_layers"], n_heads=ar_opt["num_heads"],
                      mlp_ratio=ar_opt["mlp_ratio"], dropout=ar_opt["dropout"],
                      attn_drop_rate=0.0, use_crossattn=True, attn_l2_norm=False,
                      infer_use_kvcache=False),
        "text_embedder": dict(dim_embed=768, version=ar_opt["t5_model"]),
        "data": dict(dim_pose=ar_opt["pose_dim"], max_motion_length=ar_opt["window_size"],
                     max_text_length=ar_opt["t5_max_len"]),
        "training": dict(perturb_rate=[0.0, 0.0], cond_drop_prob=ar_opt["cond_drop_prob"],
                         sample_level_times=[1, 1, 1, 1]),
        "vq": dict(code_dim=32, scales=[8, 4, 2, 1]),
        "exp": dict(seed=ar_opt["seed"]),
    })
    full_length = (ar_opt["window_size"] // (2 ** vq_model.n_layers)) * vq_model.J_b
    ar = MoScaleFSQ(
        code_dim=vq_model.code_dim,
        latent_dim=ar_opt["latent_dim"], num_heads=ar_opt["num_heads"], dropout=ar_opt["dropout"],
        text_dim=cfg.text_embedder.dim_embed, cond_drop_prob=ar_opt["cond_drop_prob"],
        mlp_ratio=ar_opt["mlp_ratio"], device=torch.device(args.device), cfg=cfg,
        full_length=full_length, scales=list(vq_model.scales),
        effective_levels=vq_model.effective_levels,
    ).to(args.device)
    ar.load_state_dict(ar_state["ar"])
    ar.train(False)

    wrapper_opt = get_opt(args.dataset_opt_path, torch.device(args.device))
    mean = np.load(pjoin(wrapper_opt.meta_dir, "mean.npy"))
    std = np.load(pjoin(wrapper_opt.meta_dir, "std.npy"))
    w_vec = WordVectorizer(args.glove_dir, "our_vab")
    test_split = pjoin(args.data_root, "test.txt")
    eval_ds = Text2MotionDatasetEval(wrapper_opt, mean, std, test_split, w_vec)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=False, pin_memory=True,
                             collate_fn=eval_collate)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    def make_gen_func(gt_prefix_count: int):
        @torch.no_grad()
        def gen_func(batch):
            caption, motion, m_length = batch
            if not isinstance(caption, list):
                caption = list(caption)
            motion_dev = motion.to(args.device, dtype=torch.float32)
            m_length_dev = m_length.to(args.device, dtype=torch.long)
            # Encode the GT motion → per-scale indices.
            gt_idx_list = None
            if gt_prefix_count > 0:
                gt_idx_list, _, _ = vq_model.encode(motion_dev, m_lens=m_length_dev, train=False)
                gt_idx_list = [t.clamp(min=0) for t in gt_idx_list]
            return_list = ar.generate(
                caption, m_length_dev,
                cond_scale=args.cond_scale, temperature=args.temperature, top_p_thres=args.top_p,
                vq_model=vq_model,
                gt_prefix_indices=gt_idx_list, gt_prefix_count=gt_prefix_count,
            )
            # ar.generate now returns raw indices at padded positions (no -1
            # mask). decode_from_indices treats them as valid FSQ idx; the
            # caller (evaluate_once) masks pred_motion past m_length.
            pred = vq_model.decode_from_indices(return_list)
            return pred, None
        return gen_func

    summary = []
    for k in range(0, n_scales + 1):
        gen_func = make_gen_func(k)
        fids = []
        top1s = []
        for r in range(args.repeat_times):
            out = evaluate_once(ar, vq_model, eval_loader, eval_wrapper, gen_func, args.device)
            fids.append(out["fid"]); top1s.append(out["top1"])
            print(f"[k={k} rep={r}] FID={out['fid']:.4f}  top1={out['top1']:.4f}  Div={out['diversity']:.4f}")
        mean_fid = float(np.mean(fids))
        mean_top1 = float(np.mean(top1s))
        summary.append((k, mean_fid, mean_top1))

    print("\n=== Per-scale teacher-forcing sweep ===")
    print(f"  k=0    pure AR sampling (production)")
    print(f"  k={n_scales}    full teacher-forcing (encode->decode)")
    print()
    print(f"  {'k':>3}  {'FID':>10}  {'top1':>8}")
    for k, fid, top1 in summary:
        print(f"  {k:>3}  {fid:>10.4f}  {top1:>8.4f}")


if __name__ == "__main__":
    main()
