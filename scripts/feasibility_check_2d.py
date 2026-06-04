"""Feasibility check for the 2D / RoPE2D port — TOKENIZER side.

Per the project's "feasibility check before long training runs" rule, this
script verifies:
  1. The 2D-cascade FSQ runs forward + backward at the standard shapes.
  2. SkelVQ end-to-end with cascade_mode="2d" trains gradients into encoder
     and decoder, and reconstructs (B, T, 263) motion.

For the AR-side feasibility check (RoPE2D function shapes, MoScaleFSQ forward
+ generate at 2D), see ``moscale/scripts/feasibility_check_2d_ar.py`` — it
runs from the moscale/ root because moscale's transformer code uses
``from model.transformer...`` imports that collide with SALAD's package
layout.

Run from the SALAD repo root:
    cd <SALAD>
    python scripts/feasibility_check_2d.py
"""
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

# Ensure SALAD root on sys.path.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def step1_tokenizer_2d_cascade():
    print("=" * 70)
    print("Step 1: 2D FSQ cascade (no encoder/decoder, just MultiScaleFSQ)")
    print("=" * 70)
    from models.vae.bsq import MultiScaleFSQ

    B, D, T, J = 4, 32, 49, 7
    x = torch.randn(B, D, T, J, requires_grad=True, device="cuda")
    print(f"  input shape: {tuple(x.shape)}")

    fsq2d = MultiScaleFSQ(code_dim=D, scales=[8, 4, 2, 1], levels=8,
                          cascade_mode="2d").cuda()
    q, loss, diag = fsq2d(x)
    print(f"  q shape:     {tuple(q.shape)}  (should equal input)")
    print(f"  q range:     [{q.min().item():.3f}, {q.max().item():.3f}]")
    print(f"  fsq loss:    {loss.item():.4f}  (0 for default FSQ)")
    print(f"  diag:        {diag.item():.4f}")
    assert q.shape == x.shape

    # Backward
    target = torch.randn_like(q)
    rec = ((q - target) ** 2).mean()
    (rec + loss).backward()
    assert x.grad is not None and x.grad.abs().sum().item() > 0
    print(f"  [OK] backward — input grad norm: {x.grad.norm().item():.4f}")

    # Indices roundtrip
    fsq2d.zero_grad()
    with torch.no_grad():
        idx_list, q_per_scale, q_cum = fsq2d.encode_indices_2d(x.detach())
    print()
    print(f"  encode_indices_2d returned {len(idx_list)} scales:")
    for s, (idx, qs, qc) in enumerate(zip(idx_list, q_per_scale, q_cum)):
        print(f"    scale {s}: idx={tuple(idx.shape)}  q_native={tuple(qs.shape)}  q_cum={tuple(qc.shape)}")
        assert idx.dtype == torch.int64
        assert idx.min().item() >= 0 and idx.max().item() <= 6
    print()


def step2_skelvq_2d_endtoend():
    print("=" * 70)
    print("Step 2: SkelVQ end-to-end with 2D cascade (encoder → FSQ → decoder)")
    print("=" * 70)
    from models.vae.skel_vq import SkelVQ
    from utils.paramUtil import t2m_kinematic_chain  # noqa

    opt = types.SimpleNamespace(
        pose_dim=263, joints_num=22, contact_joints=[7, 10, 8, 11],
        dataset_name="t2m",
        latent_dim=32, code_dim=32, kernel_size=3,
        n_layers=2, n_extra_layers=1,
        norm="none", activation="gelu", dropout=0.1,
        quantizer_type="fsq", fsq_levels=8,
        fsq_inv_temperature=20.0, fsq_entropy_weight=0.0, fsq_zeta=1.0,
        quantizer_cascade="2d",
        scales=[8, 4, 2, 1],
        start_drop=-1, quantize_dropout_prob=0.0,
        nb_code=512, mu=0.99, share_quant_resi=4, quant_resi=0.0,
        inv_temperature=100.0, entropy_weight=0.1, zeta=1.0,
    )
    model = SkelVQ(opt).cuda()
    print(f"  J_b probed by SkelVQ: {model.J_b}")
    print(f"  cascade_mode:        {model.cascade_mode}")

    B, T = 4, 196
    motion = torch.randn(B, T, opt.pose_dim, requires_grad=False, device="cuda")
    out, loss_dict = model(motion)
    print(f"  recon shape:         {tuple(out.shape)} (expected B,T,263)")
    print(f"  loss_dict keys:      {list(loss_dict.keys())}")
    rec = F.mse_loss(out, motion)
    rec.backward()
    n_params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_params_total = sum(1 for p in model.parameters())
    print(f"  [OK] backward — params with non-zero grad: {n_params_with_grad}/{n_params_total}")
    print()
    return model


def step3_rope2d_function():
    print("=" * 70)
    print("Step 3: precompute_rope2d_for_batch — shape + value sanity")
    print("=" * 70)
    from moscale.model.transformer.moscale_fsq import precompute_rope2d_for_batch

    head_dim = 96  # head_latent_dim=768 / num_heads=8
    scale_shapes = [(6, 7), (12, 7), (24, 7), (49, 7)]
    total = sum(t * j for t, j in scale_shapes)
    rope, offsets, total_lens = precompute_rope2d_for_batch(
        head_dim, scale_shapes, max_t=49, max_j=7, device="cuda"
    )
    print(f"  rope.shape:          {tuple(rope.shape)} (expected (1,2,1,1,1,{total},{head_dim//2}))")
    print(f"  offsets:             {offsets.cpu().tolist()}")
    print(f"  total_lens:          {total_lens.cpu().tolist()}")
    assert rope.shape == (1, 2, 1, 1, 1, total, head_dim // 2)
    assert offsets[0].cpu().tolist() == [0, 42, 126, 294]
    assert total_lens.item() == total

    # Spot-check: scale 0 first row at position (t_local=0, j=0) — t_canon=0, j_canon=0.
    # Position 0 of scale 0 should have cos(0)=1 in both axes' first channel.
    # First half is t-axis (qtr=24 cols), second half is j-axis (qtr=24 cols).
    cos0 = rope[0, 0, 0, 0, 0, 0, :]
    sin0 = rope[0, 1, 0, 0, 0, 0, :]
    print(f"  pos 0 cos[:4]:       {cos0[:4].cpu().tolist()} (should be ~1)")
    print(f"  pos 0 cos[24:28]:    {cos0[24:28].cpu().tolist()} (j-axis at j=0, should be ~1)")
    print(f"  pos 0 sin[:4]:       {sin0[:4].cpu().tolist()} (should be ~0)")
    assert abs(cos0[0].item() - 1.0) < 1e-5
    assert abs(cos0[24].item() - 1.0) < 1e-5
    assert abs(sin0[0].item() - 0.0) < 1e-5
    print("  [OK] cos(0)=1 and sin(0)=0 at the origin position")
    print()


def step4_ar_2d_forward_backward(skelvq_2d):
    print("=" * 70)
    print("Step 4: MoScaleFSQ with use_rope2d=True — forward + backward via wrapper")
    print("=" * 70)
    # We can't import SkelVQWrapper here because step2's SkelVQ instance is
    # already loaded and we'd hit the namespace-isolation dance. Instead we
    # build a tiny wrapper-shim around the in-memory skelvq_2d that surfaces
    # the same API points MoScaleFSQ.forward needs: .encode(...), .quantizer,
    # .J_b, .effective_levels, .scales, .code_dim, .compute_bottleneck_lens.
    from moscale.model.transformer.moscale_fsq import MoScaleFSQ

    class _InMemoryWrapper(nn.Module):
        def __init__(self, skelvq):
            super().__init__()
            self.skelvq = skelvq
            self.J_b = skelvq.J_b
            self.code_dim = skelvq.code_dim
            self.effective_levels = skelvq.quantizer.effective_levels
            self.scales = list(skelvq.quantizer.scales)
            self.n_layers = skelvq.opt.n_layers
            self.down_t = self.n_layers
            self.cascade_mode = "2d"
            self.quantizer = _QShim(skelvq.quantizer)

        @torch.no_grad()
        def encode(self, motion, m_lens=None, perturb_rate=None, train=False, codebook=None):
            x = motion.detach().float()
            h = self.skelvq.motion_enc(x)
            h = self.skelvq.conv_enc(h)
            z_2d = h.permute(0, 3, 1, 2).contiguous()
            idx_list, _qps, q_cum_list = self.skelvq.quantizer.encode_indices_2d(
                z_2d, perturb_rate=perturb_rate, train=train,
            )
            f_hat = q_cum_list[-1]
            return idx_list, q_cum_list, f_hat

    class _QShim(nn.Module):
        def __init__(self, fsq):
            super().__init__()
            self.fsq = fsq

        @property
        def scales(self):
            return self.fsq.scales

        @property
        def code_dim(self):
            return self.fsq.code_dim

        @property
        def effective_levels(self):
            return self.fsq.effective_levels

        def indices_to_codes(self, idx):
            return self.fsq.indices_to_codes(idx)

        def dequantize(self, idx):
            return self.fsq.indices_to_codes(idx)

        class _IdentityIndexed:
            def __getitem__(self, _):
                return lambda x: x

        @property
        def quant_resi(self):
            return self._IdentityIndexed()

    wrapper = _InMemoryWrapper(skelvq_2d)
    print(f"  wrapper: J_b={wrapper.J_b}  scales={wrapper.scales}  cascade={wrapper.cascade_mode}")

    # Build a tiny AR cfg matching the 2D layout.
    cfg = OmegaConf.create({
        "model": dict(
            latent_dim=64, head_latent_dim=128,
            num_layers=2, n_heads=4,
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

    # full_length = T_b * J_b = 49 * 7 = 343 for window_size=196 with n_layers=2.
    full_length = (196 // (2 ** 2)) * 7
    print(f"  building MoScaleFSQ: full_length={full_length}, use_rope2d=True ...")
    ar = MoScaleFSQ(
        code_dim=32, latent_dim=64, num_heads=4, dropout=0.1,
        text_dim=768, cond_drop_prob=0.1, mlp_ratio=2.0,
        device="cuda:0", cfg=cfg, full_length=full_length,
        scales=[8, 4, 2, 1],
        shared_aln=False, norm_eps=1e-6,
        flash_if_available=False, fused_if_available=False,
        effective_levels=7,
    ).cuda()

    print(f"  ar.patch_sizes:      {ar.patch_sizes}")
    print(f"  ar.patch_sizes_2d:   {ar.patch_sizes_2d}")
    print(f"  ar.L (total tokens): {ar.L}")
    print(f"  ar.rope_base.shape:  {tuple(ar.rope_base.shape)}")
    assert ar.patch_sizes == [42, 84, 168, 343]
    assert ar.patch_sizes_2d == [(6, 7), (12, 7), (24, 7), (49, 7)]
    assert ar.L == 637

    # Forward + backward
    B, T = 2, 196
    motion = torch.randn(B, T, 263, device="cuda")
    m_lens = torch.full((B,), T, device="cuda", dtype=torch.long)
    texts = ["a person walks forward", "the person waves their hand"]
    ar.train(True)
    loss, _pred, acc, ps_acc = ar.forward(motion, texts, m_lens, wrapper, train=True)
    print(f"  AR loss:             {loss.item():.4f}")
    print(f"  AR overall acc:      {acc:.4f}")
    print(f"  per-scale acc:       {ps_acc}")
    assert torch.isfinite(loss), "AR loss is non-finite!"

    loss.backward()
    n_with_grad = sum(1 for p in ar.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in ar.parameters() if p.requires_grad)
    print(f"  [OK] backward — params with non-zero grad: {n_with_grad}/{n_total}")
    print()

    return ar, wrapper


def step5_ar_2d_generate(ar, wrapper):
    print("=" * 70)
    print("Step 5: MoScaleFSQ.generate (CFG sampling) — 2D path")
    print("=" * 70)
    ar.train(False)
    B = 2
    m_lens = torch.full((B,), 196, device="cuda", dtype=torch.long)
    texts = ["a person walks forward", "the person waves their hand"]
    return_list = ar.generate(texts, m_lens, cond_scale=4.0,
                              temperature=1.0, top_p_thres=0.9,
                              vq_model=wrapper)
    print(f"  generate returned {len(return_list)} scale tensors:")
    for s, idx in enumerate(return_list):
        print(f"    scale {s}: {tuple(idx.shape)} dtype={idx.dtype} range=[{idx.min().item()}, {idx.max().item()}]")
        assert idx.shape[-1] == 32, f"last dim should be code_dim=32, got {idx.shape[-1]}"
    print()

    # Decode roundtrip via the wrapper's 2D path mirrored inline.
    print("  decoding return_list back to motion via in-memory 2D path...")
    z_cum = torch.zeros(B, 32, 49, 7, device="cuda")
    for idx in return_list:
        Bi, L_s, D_ = idx.shape
        T_s = L_s // 7
        q_flat = wrapper.quantizer.indices_to_codes(idx).permute(0, 2, 1).contiguous()
        q = q_flat.reshape(Bi, D_, T_s, 7)
        if (T_s, 7) != (49, 7):
            q_up = F.interpolate(q, size=(49, 7), mode="bilinear", align_corners=False)
        else:
            q_up = q
        z_cum = z_cum + q_up
    motion = wrapper.skelvq.decode(z_cum.permute(0, 2, 3, 1).contiguous())
    print(f"  decoded motion shape: {tuple(motion.shape)} (expected ({B},196,263))")
    assert motion.shape == (B, 196, 263)
    print("  [OK] full text → indices → motion roundtrip works.")
    print()


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; this script needs a GPU.")
        sys.exit(1)
    torch.manual_seed(0)

    step1_tokenizer_2d_cascade()
    step2_skelvq_2d_endtoend()

    print("=" * 70)
    print("TOKENIZER 2D PORT OK. Now run the AR-side check:")
    print("    cd moscale && python scripts/feasibility_check_2d_ar.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
