"""Shared generation-quality eval helpers for the SkelVQ-FSQ AR pipeline.

Used by train_skelvq_ar.py (periodic in-training eval, fewer repeats) and
eval_skelvq_ar.py (post-training paper-protocol eval, 20 repeats).

The eval pipeline:
    caption  -> MoScaleFSQ.generate -> per-scale FSQ indices
             -> SkelVQWrapper.decode_from_indices -> motion (B, T, 263)
    motion   -> SALAD's t2m EvaluatorWrapper -> FID + R-precision + Diversity
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from utils.metrics import (
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_R_precision,
    euclidean_distance_matrix,
)


def make_gen_func(ar, vq_model, cond_scale: float, temperature: float, top_p: float, device: str):
    """Returns a callable `(caption, motion, m_length) -> (pred_motion, _)`
    matching the signature SALAD's evaluation_denoiser expects.
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
        return_list = [r.clamp(min=0) for r in return_list]
        pred = vq_model.decode_from_indices(return_list)
        return pred, None
    return gen_func


def evaluate_once(
    ar, vq_model, eval_loader, eval_wrapper, gen_func, device: str,
) -> Dict[str, float]:
    """One full pass over `eval_loader`. Returns FID + R@1/2/3 + Diversity +
    Matching score, plus the real-data references for sanity-checking.
    """
    ar.train(False)
    motion_annotation_list, motion_pred_list = [], []
    R_precision_real = 0.0
    R_precision = 0.0
    matching_score_real = 0.0
    matching_score_pred = 0.0
    nb_sample = 0

    for batch in eval_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch
        motion = motion.to(device, dtype=torch.float32)
        m_length = m_length.to(device, dtype=torch.long)
        bs = motion.shape[0]

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        R_precision_real += calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        matching_score_real += euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()

        pred_motion, _ = gen_func((caption, motion, m_length))
        mask = torch.arange(motion.shape[1]).unsqueeze(0).expand(bs, -1).to(device) >= m_length.unsqueeze(1)
        pred_motion = pred_motion.masked_fill(mask.unsqueeze(-1), 0)
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motion, m_length)

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
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    matching_score_real /= nb_sample
    matching_score_pred /= nb_sample

    return dict(
        fid=float(fid),
        diversity=float(diversity),
        diversity_real=float(diversity_real),
        top1=float(R_precision[0]), top2=float(R_precision[1]), top3=float(R_precision[2]),
        top1_real=float(R_precision_real[0]),
        top2_real=float(R_precision_real[1]),
        top3_real=float(R_precision_real[2]),
        matching=float(matching_score_pred),
        matching_real=float(matching_score_real),
    )


def aggregate_repeats(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """mean ± 95% CI per metric across multiple repeat runs."""
    keys = metrics_list[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metrics_list])
        out[k] = float(vals.mean())
        out[f"{k}_conf95"] = float(vals.std() * 1.96 / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return out
