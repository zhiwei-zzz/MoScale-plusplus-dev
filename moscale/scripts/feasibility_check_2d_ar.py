"""AR-side feasibility check for the 2D / RoPE2D port.

Runs from the moscale/ root so the AR's `from model.transformer...` imports
resolve correctly without colliding with SALAD's package layout.

Mocks the SkelVQWrapper with a tiny in-memory nn.Module that returns random
per-scale FSQ indices at the right 2D shapes. We're checking the AR's
forward/generate plumbing, not tokenizer correctness (that's covered by
SALAD/scripts/feasibility_check_2d.py).

Run:
    cd <SALAD>/moscale
    python scripts/feasibility_check_2d_ar.py
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
MOSCALE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, MOSCALE_ROOT)


def step1_rope2d():
    print("=" * 70)
    print("Step 1: precompute_rope2d_for_batch — shape + spot-check values")
    print("=" * 70)
    from model.transformer.moscale_fsq import precompute_rope2d_for_batch

    head_dim = 96
    scale_shapes = [(6, 7), (12, 7), (24, 7), (49, 7)]
    total = sum(t * j for t, j in scale_shapes)
    rope, offsets, total_lens = precompute_rope2d_for_batch(
        head_dim, scale_shapes, max_t=49, max_j=7, device="cuda"
    )
    expected_shape = (1, 2, 1, 1, 1, total, head_dim // 2)
    print(f"  rope.shape:          {tuple(rope.shape)} (expected {expected_shape})")
    print(f"  offsets:             {offsets.cpu().tolist()}")
    print(f"  total_lens:          {total_lens.cpu().tolist()}")
    assert rope.shape == expected_shape
    assert offsets[0].cpu().tolist() == [0, 42, 126, 294]
    assert total_lens.item() == 637

    # cos(0)=1 and sin(0)=0 at position 0 (t_canon=0, j_canon=0).
    cos0 = rope[0, 0, 0, 0, 0, 0, :]
    sin0 = rope[0, 1, 0, 0, 0, 0, :]
    print(f"  pos 0 cos[0]:        {cos0[0].item():.4f}  (should be 1.0)")
    print(f"  pos 0 cos[24]:       {cos0[24].item():.4f}  (j-axis at j=0, should be 1.0)")
    print(f"  pos 0 sin[0]:        {sin0[0].item():.4f}  (should be 0.0)")
    assert abs(cos0[0].item() - 1.0) < 1e-5
    assert abs(cos0[24].item() - 1.0) < 1e-5
    assert abs(sin0[0].item()) < 1e-5
    print("  [OK] RoPE2D base tensor structurally correct.")
    print()


class _MockFSQ(nn.Module):
    """Mocks MultiScaleFSQ enough for MoScaleFSQ to consume during forward/generate."""

    def __init__(self, code_dim=32, scales=(8, 4, 2, 1)):
        super().__init__()
        self.code_dim = code_dim
        self.scales = list(scales)
        self.effective_levels = 7
        self.int_half = 3

    def indices_to_codes(self, idx):
        return (idx.to(torch.float32) - self.int_half) / 3.5

    def dequantize(self, idx):
        return self.indices_to_codes(idx)

    class _IdentityIndexed:
        def __getitem__(self, _):
            return lambda x: x

    @property
    def quant_resi(self):
        return self._IdentityIndexed()


class _MockWrapper(nn.Module):
    """Returns random per-scale level indices + cumulative dequant at full (T_b, J_b).

    Same shape contract as the real SkelVQWrapper in 2D mode:
      idx_list[s]:  (B, T_s*J, code_dim) int64 in [0, 7)
      q_cum_list[s]: (B, code_dim, T_b, J_b) float
      f_hat:        q_cum_list[-1]
    """

    def __init__(self, T_b=49, J_b=7, code_dim=32, scales=(8, 4, 2, 1)):
        super().__init__()
        self.T_b = T_b
        self.J_b = J_b
        self.code_dim = code_dim
        self.scales = list(scales)
        self.effective_levels = 7
        self.n_layers = 2
        self.down_t = 2
        self.cascade_mode = "2d"
        self.quantizer = _MockFSQ(code_dim=code_dim, scales=scales)

    @torch.no_grad()
    def encode(self, motion, m_lens=None, perturb_rate=None, train=False, codebook=None):
        B = motion.shape[0]
        device = motion.device
        idx_list, q_cum_list = [], []
        cum = torch.zeros(B, self.code_dim, self.T_b, self.J_b, device=device)
        for s in self.scales:
            T_s = max(1, self.T_b // s)
            L_s = T_s * self.J_b
            idx = torch.randint(0, 7, (B, L_s, self.code_dim), device=device, dtype=torch.int64)
            idx_list.append(idx)
            q_flat = self.quantizer.indices_to_codes(idx).permute(0, 2, 1).contiguous()  # (B, D, L_s)
            q_2d = q_flat.reshape(B, self.code_dim, T_s, self.J_b)
            if (T_s, self.J_b) != (self.T_b, self.J_b):
                q_up = F.interpolate(q_2d, size=(self.T_b, self.J_b), mode="bilinear", align_corners=False)
            else:
                q_up = q_2d
            cum = cum + q_up
            q_cum_list.append(cum.clone())
        f_hat = q_cum_list[-1]
        return idx_list, q_cum_list, f_hat


def step2_ar_forward_backward():
    print("=" * 70)
    print("Step 2: MoScaleFSQ with use_rope2d=True — forward + backward")
    print("=" * 70)
    from model.transformer.moscale_fsq import MoScaleFSQ

    wrapper = _MockWrapper().cuda()

    cfg = OmegaConf.create({
        "model": dict(
            latent_dim=384, head_latent_dim=768,
            num_layers=2, n_heads=8,
            mlp_ratio=2.0, dropout=0.1, attn_drop_rate=0.0,
            use_crossattn=True, attn_l2_norm=False, infer_use_kvcache=False,
            use_rope2d=True, rope2d_J=7,
        ),
        "text_embedder": dict(dim_embed=768, version="google/t5-v1_1-base"),
        "data": dict(dim_pose=263, max_motion_length=196, max_text_length=20),
        "training": dict(
            perturb_rate=[0.0, 0.1], cond_drop_prob=0.1,
            sample_level_times=[1, 1, 1, 1],
        ),
        "vq": dict(code_dim=32, scales=[8, 4, 2, 1]),
        "exp": dict(seed=0),
    })

    full_length = 49 * 7  # T_b * J_b
    ar = MoScaleFSQ(
        code_dim=32, latent_dim=384, num_heads=8, dropout=0.1,
        text_dim=768, cond_drop_prob=0.1, mlp_ratio=2.0,
        device="cuda:0", cfg=cfg, full_length=full_length,
        scales=[8, 4, 2, 1],
        shared_aln=False, norm_eps=1e-6,
        flash_if_available=False, fused_if_available=False,
        effective_levels=7,
    ).cuda()

    print(f"  ar.use_rope2d:       {ar.use_rope2d}")
    print(f"  ar.patch_sizes:      {ar.patch_sizes}")
    print(f"  ar.patch_sizes_2d:   {ar.patch_sizes_2d}")
    print(f"  ar.full_(T,J):       ({ar.full_T},{ar.full_J})")
    print(f"  ar.L:                {ar.L}")
    print(f"  ar.rope_base.shape:  {tuple(ar.rope_base.shape)}")
    assert ar.patch_sizes == [42, 84, 168, 343]
    assert ar.patch_sizes_2d == [(6, 7), (12, 7), (24, 7), (49, 7)]
    assert ar.L == 637

    B, T = 2, 196
    motion = torch.randn(B, T, 263, device="cuda")
    m_lens = torch.full((B,), T, device="cuda", dtype=torch.long)
    texts = ["a person walks forward", "a person waves their hand"]
    ar.train(True)
    loss, _pred, acc, ps_acc = ar.forward(motion, texts, m_lens, wrapper, train=True)
    print(f"  AR loss:             {loss.item():.4f}")
    print(f"  AR overall acc:      {acc:.4f}")
    print(f"  per-scale acc:       {ps_acc}")
    assert torch.isfinite(loss), "AR loss non-finite"

    loss.backward()
    n_grad = sum(1 for p in ar.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_tot = sum(1 for p in ar.parameters() if p.requires_grad)
    print(f"  [OK] backward — params with non-zero grad: {n_grad}/{n_tot}")
    print()
    return ar, wrapper


def step3_ar_generate(ar, wrapper):
    print("=" * 70)
    print("Step 3: MoScaleFSQ.generate (CFG sampling) — 2D path")
    print("=" * 70)
    ar.train(False)
    B = 2
    m_lens = torch.full((B,), 196, device="cuda", dtype=torch.long)
    texts = ["a person walks forward", "a person waves their hand"]
    return_list = ar.generate(texts, m_lens, cond_scale=4.0,
                              temperature=1.0, top_p_thres=0.9,
                              vq_model=wrapper)
    print(f"  generate returned {len(return_list)} scale tensors:")
    for s, idx in enumerate(return_list):
        print(f"    scale {s}: {tuple(idx.shape)} dtype={idx.dtype}"
              f" range=[{idx.min().item()}, {idx.max().item()}]")
    assert len(return_list) == 4
    assert return_list[0].shape == (B, 42, 32)
    assert return_list[1].shape == (B, 84, 32)
    assert return_list[2].shape == (B, 168, 32)
    assert return_list[3].shape == (B, 343, 32)
    print("  [OK] generate produces 2D-shaped per-scale indices.")
    print()


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; this script needs a GPU.")
        sys.exit(1)
    torch.manual_seed(0)

    step1_rope2d()
    ar, wrapper = step2_ar_forward_backward()
    step3_ar_generate(ar, wrapper)

    print("=" * 70)
    print("ALL AR STEPS PASSED — 2D / RoPE2D AR is wired correctly.")
    print("=" * 70)


if __name__ == "__main__":
    main()
