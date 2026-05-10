"""Smoke tests for the FSQ index API and the trained SkelVQ-FSQ at the AR
sequence length MoScale will use (T=196 -> L=343)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import types
import torch
import torch.nn.functional as F

from models.vae.bsq import MultiScaleFSQ
from models.vae.skel_vq import SkelVQ


def test_encode_indices_roundtrip():
    print("=" * 60)
    print("Test 1: MultiScaleFSQ.encode_indices roundtrip")
    print("=" * 60)
    torch.manual_seed(0)
    B, D, L = 2, 32, 112
    m = MultiScaleFSQ(code_dim=D, scales=[8, 4, 2, 1], levels=8).cuda()
    m.train(False)

    x = torch.randn(B, D, L, device="cuda")
    q_forward, _, _ = m(x)

    idx_list, q_per_scale, q_cum = m.encode_indices(x)

    print(f"effective_levels = {m.effective_levels}")
    assert m.effective_levels == 7, f"expected 7, got {m.effective_levels}"

    for si, (idx, qs, qc) in enumerate(zip(idx_list, q_per_scale, q_cum)):
        L_s = max(1, L // m.scales[si])
        assert idx.shape == (B, L_s, D), f"scale {si}: idx shape {idx.shape}"
        assert qs.shape == (B, D, L_s), f"scale {si}: q_per_scale shape {qs.shape}"
        assert qc.shape == (B, D, L), f"scale {si}: q_cum shape {qc.shape}"
        assert idx.min() >= 0 and idx.max() < m.effective_levels, \
            f"scale {si}: idx range [{idx.min()}, {idx.max()}]"
        # Dequantize and compare with q_per_scale
        deq = m.indices_to_codes(idx).permute(0, 2, 1)  # (B, D, L_s)
        diff = (deq - qs).abs().max().item()
        assert diff < 1e-6, f"scale {si}: indices_to_codes vs q_per_scale diff {diff}"
        print(f"  scale {m.scales[si]}: idx{tuple(idx.shape)} in [{idx.min()}, {idx.max()}]  q{tuple(qs.shape)}  cum{tuple(qc.shape)}  roundtrip ok")

    # Final cumulative should match forward output
    diff = (q_cum[-1] - q_forward).abs().max().item()
    print(f"  q_cum[-1] vs MultiScaleFSQ.forward(x)[0]:  max diff = {diff:.2e}")
    assert diff < 1e-5, f"final cumulative diverges from forward: {diff}"
    print("[OK] roundtrip clean")


def _build_opt(window_size, joints_num=22, latent_dim=32):
    """Reconstruct a minimal opt for SkelVQ that matches the trained ckpt."""
    opt = types.SimpleNamespace()
    opt.batch_size = 1
    opt.window_size = window_size
    opt.pose_dim = 263
    opt.joints_num = joints_num
    opt.latent_dim = latent_dim
    opt.code_dim = 32
    opt.kernel_size = 3
    opt.n_layers = 2
    opt.n_extra_layers = 1
    opt.norm = "none"
    opt.activation = "gelu"
    opt.dropout = 0.1
    opt.quantizer_type = "fsq"
    opt.fsq_levels = 8
    opt.fsq_inv_temperature = 20.0
    opt.fsq_entropy_weight = 0.0
    opt.fsq_zeta = 1.0
    opt.scales = [8, 4, 2, 1]
    opt.start_drop = -1
    opt.quantize_dropout_prob = 0.0
    # Required by encdec.MotionEncoder/Decoder
    opt.contact_joints = [7, 10, 8, 11]   # t2m kinematic chain feet/toes
    opt.dataset_name = "t2m"
    return opt


def test_trained_ckpt_at_T196(ckpt_path, T=196):
    print()
    print("=" * 60)
    print(f"Test 2: trained SkelVQ-FSQ ckpt at T={T} (MoScale's max length)")
    print("=" * 60)
    opt = _build_opt(window_size=64)
    model = SkelVQ(opt).cuda()
    model.train(False)
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    state = ckpt["vae"] if "vae" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"loaded checkpoint: {ckpt_path}")
    print(f"J_b (joints after STPool) = {model.J_b}")

    torch.manual_seed(0)
    motion = torch.randn(2, T, 263, device="cuda")
    with torch.no_grad():
        recon, loss_dict = model(motion)
    print(f"input motion: {tuple(motion.shape)}")
    print(f"recon:        {tuple(recon.shape)}")
    assert recon.shape == motion.shape, f"shape mismatch: {recon.shape} vs {motion.shape}"
    smooth_l1 = F.smooth_l1_loss(recon, motion).item()
    print(f"recon smooth-l1 (random N(0,1) input): {smooth_l1:.4f}  (sanity check, not a real metric)")
    print(f"fsq_diag at T={T}: {loss_dict.get('fsq_diag', torch.tensor(0.0)).item():.4f}")

    # Now exercise encode_indices on a real input shape.
    x = motion.detach().float()
    h = model.motion_enc(x)            # (B, T, J=22, D)
    h = model.conv_enc(h)              # (B, T_b, J_b, D)
    Bn, T_b, J_b, D = h.shape
    L_actual = T_b * J_b
    print(f"after conv_enc: T_b={T_b}, J_b={J_b}, L = T_b*J_b = {L_actual}")

    z_1d = h.reshape(Bn, T_b * J_b, D).transpose(1, 2).contiguous()  # (B, D, L)
    idx_list, q_per_scale, q_cum = model.quantizer.encode_indices(z_1d)
    print(f"per-scale indices shapes (B, L_s, D):")
    for si, idx in enumerate(idx_list):
        print(f"  scale {model.quantizer.scales[si]}: {tuple(idx.shape)}  range [{idx.min()}, {idx.max()}]  unique levels used: {idx.unique().numel()}/{model.quantizer.effective_levels}")
    total_token_positions = sum(idx.shape[1] for idx in idx_list)
    print(f"total AR sequence length (sum L_s over scales) = {total_token_positions}")
    print(f"per-channel categorical: {model.quantizer.effective_levels}-way x {D} channels = {model.quantizer.effective_levels * D} logits per position")
    print("[OK] T=196 forward + encode_indices working")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/t2m/skelvq_fsq/model/net_best_fid.tar")
    p.add_argument("--T", type=int, default=196)
    args = p.parse_args()

    test_encode_indices_roundtrip()
    test_trained_ckpt_at_T196(args.ckpt, T=args.T)
    print()
    print("All smoke tests passed.")
