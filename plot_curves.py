"""Plot training+test curves directly from TensorBoard event logs."""
import glob
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIR = "log/t2m/vae_repro"
OUT_PATH = Path("training_logs/vae_repro_curves.png")

ev_paths = sorted(glob.glob(f"{LOG_DIR}/events.out.tfevents.*"))
if not ev_paths:
    print(f"No event files in {LOG_DIR}")
    sys.exit(1)

ea = EventAccumulator(ev_paths[-1], size_guidance={"scalars": 0})  # 0 = unlimited
ea.Reload()


def get(tag):
    if tag not in ea.Tags()["scalars"]:
        return [], []
    evs = ea.Scalars(tag)
    return [e.step for e in evs], [e.value for e in evs]


fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# Top-left: train losses
ax = axes[0, 0]
for tag, color, label in [
    ("Train/loss", "black", "total"),
    ("Train/loss_recon", "C0", "recon"),
    ("Train/loss_vel", "C1", "vel"),
    ("Train/loss_pos", "C2", "pos"),
]:
    s, v = get(tag)
    if s:
        ax.plot(s, v, color=color, label=label, linewidth=1.2)
ax.set_xlabel("iter")
ax.set_ylabel("loss")
ax.set_title("Train losses")
ax.legend()
ax.grid(alpha=0.3)

# Top-right: KL train + val (log scale)
ax = axes[0, 1]
s, v = get("Train/loss_kl")
if s:
    ax.plot(s, v, color="C3", label="train KL")
ax.set_xlabel("iter")
ax.set_ylabel("KL")
ax.set_title("KL loss")
ax.set_yscale("log")
ax.grid(alpha=0.3)
ax.legend()

# Bottom-left: validation losses
ax = axes[1, 0]
for tag, color, label in [
    ("Val/loss", "black", "total"),
    ("Val/loss_recon", "C0", "recon"),
    ("Val/loss_vel", "C1", "vel"),
    ("Val/loss_pos", "C2", "pos"),
]:
    s, v = get(tag)
    if s:
        ax.plot(s, v, color=color, label=label, marker="o", markersize=3, linewidth=1.0)
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("Validation losses")
ax.legend()
ax.grid(alpha=0.3)

# Bottom-right: Test FID (the headline number)
ax = axes[1, 1]
s, v = get("Test/FID")
if s:
    ax.plot(s, v, color="C4", marker="o", markersize=4, linewidth=1.4)
    best_i = min(range(len(v)), key=lambda i: v[i])
    ax.scatter([s[best_i]], [v[best_i]], color="red", zorder=5, s=80, label=f"best ep {s[best_i]}: {v[best_i]:.4f}")
    ax.axhline(0.003, color="gray", linestyle="--", linewidth=1, label="pretrained ckpt (0.003)")
ax.set_xlabel("epoch")
ax.set_ylabel("FID")
ax.set_title("Test/FID")
ax.set_yscale("log")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=120)
print(f"Wrote {OUT_PATH}")

# Also print the last few FID values
s, v = get("Test/FID")
print("\nLast 10 epoch FID values:")
for ep, fid in list(zip(s, v))[-10:]:
    print(f"  ep {ep:3d}  FID = {fid:.4f}")
