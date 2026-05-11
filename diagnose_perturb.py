"""Diagnostic: does perturb_rate actually corrupt the propagated `f_hat`
during SkelVQWrapper.encode at train time, or only the returned CE targets?

The wrapper docstring says:
    "The cascade re-quantizes with the perturbed codes so the propagated
     q_cum reflects corruption, while the returned idx_list stays the clean GT"

We verify by calling .encode() three ways on the same motion and comparing
both the returned indices and the cumulative q_cum / f_hat tensors.
"""
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "moscale"))

from model.vq.skelvq_wrapper import SkelVQWrapper  # type: ignore


def diff(a: torch.Tensor, b: torch.Tensor, name: str):
    d = (a - b).abs()
    n_diff = (d > 1e-6).sum().item()
    print(f"  {name:>18}: max-abs-diff={d.max().item():.6e}  "
          f"mean-abs-diff={d.mean().item():.6e}  "
          f"#elem-differ={n_diff}/{d.numel()}")


def main():
    device = "cpu"
    ckpt = "/workspace/MoScale-plusplus-dev/checkpoints/t2m/skelvq_fsq/model/net_best_fid.tar"
    vq = SkelVQWrapper(ckpt, device=device)
    print(f"loaded; J_b={vq.J_b} effective_levels={vq.effective_levels} scales={vq.scales}")

    # Synthetic batch: B=2, T=64, D=263 (HumanML3D pose dim).
    torch.manual_seed(0)
    motion = torch.randn(2, 64, 263, device=device)

    print("\n=== Pass 1: clean (perturb_rate=None, train=False) ===")
    idx_clean, qcum_clean, fhat_clean = vq.encode(motion, perturb_rate=None, train=False)
    print(f"  idx_list shapes: {[t.shape for t in idx_clean]}")
    print(f"  q_cum_list shapes: {[t.shape for t in qcum_clean]}")
    print(f"  f_hat shape: {fhat_clean.shape}")

    print("\n=== Pass 2: perturb_rate=[0.0, 0.0] train=True (off via knob) ===")
    idx_off, qcum_off, fhat_off = vq.encode(motion, perturb_rate=[0.0, 0.0], train=True)
    print(f"  vs Pass 1:")
    for s, (a, b) in enumerate(zip(idx_clean, idx_off)):
        diff(a.float(), b.float(), f"idx scale {s}")
    for s, (a, b) in enumerate(zip(qcum_clean, qcum_off)):
        diff(a, b, f"q_cum scale {s}")
    diff(fhat_clean, fhat_off, "f_hat")

    print("\n=== Pass 3: perturb_rate=[0.6, 0.6] train=True (max perturbation) ===")
    torch.manual_seed(123)
    idx_max, qcum_max, fhat_max = vq.encode(motion, perturb_rate=[0.6, 0.6], train=True)
    print(f"  vs Pass 1 (clean):")
    for s, (a, b) in enumerate(zip(idx_clean, idx_max)):
        diff(a.float(), b.float(), f"idx scale {s}")
    for s, (a, b) in enumerate(zip(qcum_clean, qcum_max)):
        diff(a, b, f"q_cum scale {s}")
    diff(fhat_clean, fhat_max, "f_hat")

    print("\n=== Pass 4: perturb_rate=[0.6, 0.6] train=False (perturb ignored?) ===")
    torch.manual_seed(123)
    idx_off2, qcum_off2, fhat_off2 = vq.encode(motion, perturb_rate=[0.6, 0.6], train=False)
    print(f"  vs Pass 1 (clean) — should be identical if train=False disables perturb:")
    for s, (a, b) in enumerate(zip(idx_clean, idx_off2)):
        diff(a.float(), b.float(), f"idx scale {s}")
    for s, (a, b) in enumerate(zip(qcum_clean, qcum_off2)):
        diff(a, b, f"q_cum scale {s}")
    diff(fhat_clean, fhat_off2, "f_hat")

    print("\n=== Interpretation ===")
    # Verdict: if Pass 3 q_cum differs from Pass 1 but idx_list matches,
    # perturb is propagating into f_hat correctly (the wrapper docstring's
    # claim). If Pass 3 q_cum == Pass 1 q_cum, perturb is only affecting the
    # CE target (idx_list), NOT propagating into the residual cascade.
    qcum_perturbed = any(
        (a - b).abs().max().item() > 1e-6 for a, b in zip(qcum_clean, qcum_max)
    )
    fhat_perturbed = (fhat_clean - fhat_max).abs().max().item() > 1e-6
    idx_perturbed = any(
        (a - b).abs().sum().item() > 0 for a, b in zip(idx_clean, idx_max)
    )

    print(f"  Pass 3 vs clean — idx differ: {idx_perturbed}")
    print(f"  Pass 3 vs clean — q_cum differ: {qcum_perturbed}")
    print(f"  Pass 3 vs clean — f_hat differ: {fhat_perturbed}")
    if not qcum_perturbed and not fhat_perturbed:
        print("\n  >>> BUG CONFIRMED: perturb_rate does NOT propagate into f_hat.")
        print("      Training forward sees CLEAN residuals; inference sees SAMPLED.")
        print("      Classic train-test gap → cascade drift at gen time.")
    elif qcum_perturbed and not idx_perturbed:
        print("\n  Perturb propagates into f_hat AND CE targets stay clean.")
        print("  Docstring behavior matches; this hypothesis is NOT the bug.")
    else:
        print("\n  Mixed result; manual inspection needed.")


if __name__ == "__main__":
    main()
