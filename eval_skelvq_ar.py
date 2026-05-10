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
from utils.metrics import (
    calculate_activation_statistics, calculate_diversity, calculate_frechet_distance,
    calculate_R_precision, euclidean_distance_matrix,
)
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


def make_gen_func(ar, vq_model, cond_scale, temperature, top_p, device):
    """Returns a function gen_func((caption, motion, m_length)) -> (pred_motion, _).

    pred_motion: (B, T, pose_dim) — generated and decoded via the wrapper.
    Note: m_length here is at original-frame resolution. We pass it through.
    """
    @torch.no_grad()
    def gen_func(batch):
        caption, motion, m_length = batch
        if not isinstance(caption, list):
            caption = list(caption)
        m_length = m_length.to(device, dtype=torch.long)
        return_list = ar.generate(
            caption, m_length, cond_scale=cond_scale,
            temperature=temperature, top_p_thres=top_p, vq_model=vq_model,
        )
        # Replace any -1 padded indices with 0 before dequantizing.
        return_list = [r.clamp(min=0) for r in return_list]
        pred = vq_model.decode_from_indices(return_list)
        return pred, None
    return gen_func


def evaluate_once(args, ar, vq_model, eval_loader, eval_wrapper, gen_func, repeat_id):
    motion_annotation_list, motion_pred_list = [], []
    motion_gt_list, motion_gen_list = [], []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    nb_sample = 0

    for batch in eval_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch
        motion = motion.to(args.device, dtype=torch.float32)
        m_length = m_length.to(args.device, dtype=torch.long)
        bs = motion.shape[0]

        # Real motion side
        motion_gt_list.append(motion)
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        R_precision_real += calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        matching_score_real += euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()

        # Generated motion side
        pred_pose_eval, _ = gen_func((caption, motion, m_length))
        mask = torch.arange(motion.shape[1]).unsqueeze(0).expand(motion.shape[0], -1).to(args.device) >= m_length.unsqueeze(1)
        pred_pose_eval = pred_pose_eval.masked_fill(mask.unsqueeze(-1), 0)
        motion_gen_list.append(pred_pose_eval)
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)
        R_precision += calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        matching_score_pred += euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        f"[repeat {repeat_id}] FID {fid:.4f}  Div {diversity:.4f}  "
        f"R@1/2/3 {R_precision[0]:.4f}/{R_precision[1]:.4f}/{R_precision[2]:.4f}  "
        f"Match {matching_score_pred:.4f}"
    )
    print(msg)
    return dict(fid=fid, diversity=diversity, top1=R_precision[0], top2=R_precision[1],
                top3=R_precision[2], matching=matching_score_pred,
                diversity_real=diversity_real, matching_real=matching_score_real)


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

    # Repeat
    metrics = {k: [] for k in ["fid", "diversity", "top1", "top2", "top3", "matching"]}
    for r in range(args.repeat_times):
        out = evaluate_once(args, ar, vq_model, eval_loader, eval_wrapper, gen_func, r)
        for k in metrics:
            metrics[k].append(out[k])

    # Aggregate
    print()
    print(f"=== {args.name} | cfg_scale={args.cond_scale} top_p={args.top_p} | {args.repeat_times} repeats ===")
    for k, vs in metrics.items():
        a = np.array(vs)
        ci = a.std() * 1.96 / np.sqrt(len(a))
        print(f"  {k}: {a.mean():.4f} ± {ci:.4f}")


if __name__ == "__main__":
    main()
