"""Generation FID / R-precision / Diversity eval for the trained MoScaleFSQ AR.

Pipeline:
  caption -> MoScaleFSQ.generate -> per-scale FSQ indices
            -> SkelVQWrapper.decode_from_indices -> motion (B, T, 263)
  motion + caption -> SALAD's t2m EvaluatorWrapper -> FID + R-precision + diversity

Run:
    python eval_skelvq_ar.py --name skelvq_ar_v1 \
        --tokenizer_ckpt checkpoints/t2m/skelvq_fsq/model/net_best_fid.tar \
        --ar_ckpt checkpoints/t2m/skelvq_ar_v1/model/net_best_loss.tar \
        --cond_scale 4.0 --top_p 0.9
"""
from __future__ import annotations

import argparse
import os
import sys
from os.path import join as pjoin

# Make moscale/ importable.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "moscale"))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.t2m_dataset import Text2MotionDatasetEval
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.get_opt import get_opt
from utils.skelvq_ar_eval import make_gen_func, evaluate_once, aggregate_repeats
from utils.word_vectorizer import WordVectorizer

from model.vq.skelvq_wrapper import SkelVQWrapper             # type: ignore
from model.transformer.moscale_fsq import MoScaleFSQ          # type: ignore


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
    p.add_argument("--repeat_times", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


# Shared helpers `make_gen_func`, `evaluate_once`, and `aggregate_repeats`
# now live in utils/skelvq_ar_eval.py and are also called from
# train_skelvq_ar.py for periodic in-training gen-eval. Keeps the eval logic
# in one place.


def main():
    args = parse_args()

    # Tokenizer (frozen) + AR (load trained weights)
    print(f"Loading tokenizer ckpt: {args.tokenizer_ckpt}")
    vq_model = SkelVQWrapper(args.tokenizer_ckpt, device=args.device)
    print(f"  J_b={vq_model.J_b}  effective_levels={vq_model.effective_levels}")

    print(f"Loading AR ckpt: {args.ar_ckpt}")
    ar_state = torch.load(args.ar_ckpt, map_location=args.device, weights_only=False)
    ar_opt = ar_state.get("opt", None)
    if ar_opt is None:
        raise RuntimeError(f"AR checkpoint at {args.ar_ckpt} has no `opt` field; was it saved by train_skelvq_ar.py?")
    cfg = OmegaConf.create({
        "model": dict(latent_dim=ar_opt["latent_dim"], head_latent_dim=ar_opt["latent_dim"],
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

    # Eval data loader
    wrapper_opt = get_opt(args.dataset_opt_path, torch.device(args.device))
    mean = np.load(pjoin(wrapper_opt.meta_dir, "mean.npy"))
    std = np.load(pjoin(wrapper_opt.meta_dir, "std.npy"))
    w_vec = WordVectorizer(args.glove_dir, "our_vab")
    test_split = pjoin(args.data_root, "test.txt")
    eval_ds = Text2MotionDatasetEval(wrapper_opt, mean, std, test_split, w_vec)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=False, pin_memory=True)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    gen_func = make_gen_func(ar, vq_model, args.cond_scale, args.temperature, args.top_p, args.device)

    runs = []
    for r in range(args.repeat_times):
        out = evaluate_once(ar, vq_model, eval_loader, eval_wrapper, gen_func, args.device)
        msg = (
            f"[repeat {r}] FID {out['fid']:.4f}  Div {out['diversity']:.4f}  "
            f"R@1/2/3 {out['top1']:.4f}/{out['top2']:.4f}/{out['top3']:.4f}  "
            f"Match {out['matching']:.4f}"
        )
        print(msg)
        runs.append(out)

    # Aggregate (mean ± 95% CI per metric)
    agg = aggregate_repeats(runs)
    print()
    print(f"=== {args.name} | cfg_scale={args.cond_scale} top_p={args.top_p} | {args.repeat_times} repeats ===")
    for k in ("fid", "diversity", "top1", "top2", "top3", "matching"):
        print(f"  {k}: {agg[k]:.4f} ± {agg[f'{k}_conf95']:.4f}")


if __name__ == "__main__":
    main()
