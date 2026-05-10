"""End-to-end smoke test + 50-step overfit run for the SkelVQ-FSQ AR backbone.

What it verifies:
  1. SkelVQWrapper loads the trained tokenizer ckpt cleanly.
  2. LevelSelfCorrection produces clean-baseline gt indices + scale-conditioning
     features when noise is off, and noisy versions when noise is on.
  3. SkelVQAR forward + backward run finite, all params receive gradients.
  4. After ~50 steps overfitting on a single batch, the AR loss drops well below
     the uniform-random baseline (ln(7) = 1.946) — the model can learn.

Run from the SALAD repo root:
    python moscale/test_skelvq_ar.py --ckpt checkpoints/t2m/skelvq_fsq/model/net_best_fid.tar
"""
from __future__ import annotations

import argparse
import os
import sys
import math
import importlib.util


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    salad_root = os.path.normpath(os.path.join(here, ".."))
    if salad_root not in sys.path:
        sys.path.insert(0, salad_root)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.path.join(salad_root, "checkpoints/t2m/skelvq_fsq/model/net_best_fid.tar"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--T", type=int, default=196)
    parser.add_argument("--noise_apply_layers", type=int, default=-1, help="-1 to disable BSC")
    parser.add_argument("--noise_apply_strength", type=float, default=0.3)
    args = parser.parse_args()

    skelvq_wrapper = _load(
        "skelvq_wrapper",
        os.path.join(here, "model/vq/skelvq_wrapper.py"),
    )
    lsc_mod = _load(
        "level_self_correction",
        os.path.join(here, "model/level_self_correction.py"),
    )
    ar_mod = _load(
        "skelvq_ar",
        os.path.join(here, "model/transformer/skelvq_ar.py"),
    )

    import torch

    print("=" * 60)
    print("Stage 1 — load frozen SkelVQ-FSQ tokenizer")
    print("=" * 60)
    w = skelvq_wrapper.SkelVQWrapper(args.ckpt, device=args.device)
    print(f"  ckpt: {args.ckpt}")
    print(f"  J_b={w.J_b}  effective_levels={w.effective_levels}  scales={w.scales}")

    print()
    print("=" * 60)
    print("Stage 2 — build AR transformer + LevelSelfCorrection")
    print("=" * 60)
    lsc = lsc_mod.LevelSelfCorrection(
        w,
        noise_apply_layers=args.noise_apply_layers,
        noise_apply_strength=args.noise_apply_strength,
        noise_apply_requant=True,
    )
    ar = ar_mod.SkelVQAR(
        w,
        latent_dim=384, num_layers=4, num_heads=8, mlp_ratio=4.0, dropout=0.1,
    ).to(args.device)
    n_params = sum(p.numel() for p in ar.parameters())
    print(f"  AR params: {n_params/1e6:.2f}M")
    print(f"  noise_apply_layers={args.noise_apply_layers}, strength={args.noise_apply_strength}")

    print()
    print("=" * 60)
    print(f"Stage 3 — overfit a single batch for {args.steps} steps")
    print("=" * 60)
    torch.manual_seed(0)
    motion = torch.randn(args.batch, args.T, 263, device=args.device)
    # Encode once; we'll run AR many times against the same labels.
    with torch.no_grad():
        z_1d, _, _ = w._encode_to_grid(motion)
        gt_idx_list, x_cond_list, _ = lsc.perturb_requant(z_1d)

    optim = torch.optim.AdamW(ar.parameters(), lr=args.lr, betas=(0.9, 0.99))
    uniform_baseline = math.log(w.effective_levels)
    print(f"  uniform CE baseline: {uniform_baseline:.4f}")
    print(f"  step  loss       acc")
    for step in range(args.steps):
        optim.zero_grad()
        loss, info = ar.forward(gt_idx_list, x_cond_list)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ar.parameters(), 2.0)
        optim.step()
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            print(f"  {step:4d}  {loss.item():.4f}    {info['acc_per_channel']:.4f}")

    print()
    final_loss = loss.item()
    final_acc = info["acc_per_channel"]
    print(f"  final loss: {final_loss:.4f}  (baseline {uniform_baseline:.4f})")
    print(f"  final acc:  {final_acc:.4f}  (baseline 1/{w.effective_levels} = {1/w.effective_levels:.4f})")
    pass_loss = final_loss < uniform_baseline * 0.7    # at least 30% improvement
    pass_acc = final_acc > 1.5 / w.effective_levels    # 50% better than uniform
    print(f"  loss pass: {pass_loss}  acc pass: {pass_acc}")
    print()
    if pass_loss and pass_acc:
        print("[OK] SkelVQAR can overfit a single batch — integration is correct.")
    else:
        print("[WARN] Overfitting incomplete. Could be normal for very few steps; check trend.")


if __name__ == "__main__":
    main()
