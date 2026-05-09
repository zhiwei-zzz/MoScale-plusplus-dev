"""Parse training log + plot loss curves."""
import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_PATH = Path("training_logs/vae_repro.stdout")
OUT_PATH = Path("training_logs/vae_repro_loss.png")

pat = re.compile(
    r"niter:\s*(?P<it>\d+).*?loss:\s*(?P<loss>[\d.eE\-+]+).*?"
    r"loss_kl:\s*(?P<kl>[\d.eE\-+]+).*?"
    r"loss_recon:\s*(?P<rec>[\d.eE\-+]+).*?"
    r"loss_vel:\s*(?P<vel>[\d.eE\-+]+).*?"
    r"loss_pos:\s*(?P<pos>[\d.eE\-+]+)"
)

iters, total, kl, rec, vel, pos = [], [], [], [], [], []
with LOG_PATH.open() as f:
    for line in f:
        m = pat.search(line)
        if m:
            iters.append(int(m.group("it")))
            total.append(float(m.group("loss")))
            kl.append(float(m.group("kl")))
            rec.append(float(m.group("rec")))
            vel.append(float(m.group("vel")))
            pos.append(float(m.group("pos")))

if not iters:
    print(f"No matches in {LOG_PATH}")
    sys.exit(1)

print(f"Parsed {len(iters)} log entries (iter {iters[0]}-{iters[-1]})")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
ax.plot(iters, total, label="total", color="black", linewidth=1.4)
ax.plot(iters, rec, label="recon", color="C0")
ax.plot(iters, vel, label="vel", color="C1")
ax.plot(iters, pos, label="pos", color="C2")
ax.set_xlabel("iter")
ax.set_ylabel("loss")
ax.set_title("SALAD VAE retrain (vae_repro) - main losses")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(iters, kl, color="C3")
ax.set_xlabel("iter")
ax.set_ylabel("KL")
ax.set_title("KL loss (note: weighted by lambda_kl=0.02 in total)")
ax.set_yscale("log")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=120)
print(f"Wrote {OUT_PATH}")
print(f"\nLatest values (iter {iters[-1]}):  total={total[-1]:.4f}  recon={rec[-1]:.4f}  vel={vel[-1]:.4f}  pos={pos[-1]:.4f}  kl={kl[-1]:.4f}")
