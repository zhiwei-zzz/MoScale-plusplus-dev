"""Smoke test SkelVQ: build the model with default opts, forward + backward on a fake batch."""
import sys
from types import SimpleNamespace
import torch

sys.path.insert(0, ".")  # so models/, utils/ resolve when run from SALAD repo root

# minimal opt namespace mimicking what skel_vq_option.arg_parse would produce
opt = SimpleNamespace(
    pose_dim=263, joints_num=22, contact_joints=[7, 10, 8, 11],
    latent_dim=32, kernel_size=3, n_layers=2, n_extra_layers=1,
    norm="none", activation="gelu", dropout=0.1,
    code_dim=32, nb_code=512, scales=[8, 4, 2, 1],
    mu=0.99, share_quant_resi=4, quant_resi=0.0,
    start_drop=-1, quantize_dropout_prob=0.0,
    dataset_name="t2m",
    window_size=64,
)

from models.vae.skel_vq import SkelVQ

torch.manual_seed(0)
model = SkelVQ(opt).cuda().train()

n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {n/1e6:.3f}M")
print(f"Bottleneck J_b={model.J_b} (expected 7), T_b={opt.window_size // (2 ** opt.n_layers)} (expected 16)")

x = torch.randn(4, opt.window_size, opt.pose_dim, device="cuda")
out, loss_dict = model(x)
print(f"input  shape: {tuple(x.shape)}")
print(f"output shape: {tuple(out.shape)}")
print(f"loss_dict keys: {list(loss_dict.keys())}")
print(f"loss_commit: {loss_dict['loss_commit'].item():.4f}")
print(f"perplexity:  {loss_dict['perplexity'].mean().item():.2f}")

# backward
recon_loss = ((out - x) ** 2).mean()
total = recon_loss + 0.02 * loss_dict["loss_commit"]
total.backward()

n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
n_total = sum(1 for p in model.parameters() if p.requires_grad)
print(f"grad: {n_grad}/{n_total} trainable params got nonzero grad")
print(f"recon_loss: {recon_loss.item():.4f}, total: {total.item():.4f}")

assert torch.isfinite(total).item(), "non-finite loss"
assert n_grad > 0.5 * n_total, f"too few params got grad: {n_grad}/{n_total}"
print("\n[OK] SkelVQ smoke test passed.")
