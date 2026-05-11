"""Parse the 5 FSQ-tokenizer 20-rep eval stdouts in training_logs/ and emit
a per-run row CSV with mean±std for every metric.

Mirrors results/build_comparison_csv.py (vae_repro / bsq / fsq) but covers
the new FSQ ablations: L=4, 6, 8 (no reg), 8 (entropy reg), 10.
"""
import csv
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Per-clip bit budgets at T=64 training (cells = sum of patch_sizes-derived
# per-scale lengths * J_b, * channels=32, * log2(L_effective)).
# For the L=8 ckpt, effective_levels=7 (banker's rounding edge case) per
# STAGE2_STATUS.md. For other L we conservatively use L itself as the
# vocabulary; the in-CSV bits column is for human reference only.
def bits(L: int, label: str = None) -> str:
    if label is not None:
        return label
    # 210 cells * 32 channels * log2(L)
    return f"{int(round(210 * 32 * math.log2(L)))}"

RUNS = [
    ("skelvq_fsq_l4_cluster",   "FSQ L=4 (no reg)",                                 bits(4),
     ROOT / "training_logs/skelvq_fsq_l4_cluster_eval.stdout"),
    ("skelvq_fsq_l6_cluster",   "FSQ L=6 (no reg)",                                 bits(6),
     ROOT / "training_logs/skelvq_fsq_l6_cluster_eval.stdout"),
    ("skelvq_fsq_cluster",      "FSQ L=8 (no reg, log2(7) effective)",              bits(7, "18860"),
     ROOT / "training_logs/skelvq_fsq_cluster_eval.stdout"),
    ("skelvq_fsq_reg_cluster",  "FSQ L=8 + entropy reg (w=0.1, inv_temp=20)",       bits(7, "18860"),
     ROOT / "training_logs/skelvq_fsq_reg_cluster_eval.stdout"),
    ("skelvq_fsq_l10_cluster",  "FSQ L=10 (no reg)",                                bits(10),
     ROOT / "training_logs/skelvq_fsq_l10_cluster_eval.stdout"),
]

LINE_RE = re.compile(
    r"Eva\. Repeat \d+ : "
    r"FID\. ([\d.]+), "
    r"Diversity Real\. [\d.]+, "
    r"Diversity\. ([\d.]+), "
    r"R_precision_real\. \([\d., ]+\), "
    r"R_precision\. \(([\d.]+), ([\d.]+), ([\d.]+)\), "
    r"matching_real\. [\d.]+, "
    r"matching_pred\. ([\d.]+), "
    r"MPJPE\. ([\d.]+), "
    r"Multimodality\. ([\d.]+)"
)
METRICS = ["FID", "Diversity", "TOP1", "TOP2", "TOP3", "Matching", "MPJPE", "Multimodality"]


def parse(path: Path):
    out = {m: [] for m in METRICS}
    for line in path.read_text().splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        fid, div, t1, t2, t3, mat, mpjpe, mm = (float(x) for x in m.groups())
        out["FID"].append(fid)
        out["Diversity"].append(div)
        out["TOP1"].append(t1)
        out["TOP2"].append(t2)
        out["TOP3"].append(t3)
        out["Matching"].append(mat)
        out["MPJPE"].append(mpjpe)
        out["Multimodality"].append(mm)
    return out


def stats(xs):
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n  # population variance
    return mean, math.sqrt(var)


PLACES = {"FID": 4, "Diversity": 3, "TOP1": 4, "TOP2": 4, "TOP3": 4,
          "Matching": 4, "MPJPE": 4, "Multimodality": 4}

header = ["run", "bottleneck", "bits_per_clip"] + METRICS
rows = [header]

for name, bottleneck, bcl, path in RUNS:
    if not path.exists():
        raise SystemExit(f"missing eval stdout: {path}")
    data = parse(path)
    if not data["FID"]:
        raise SystemExit(f"no Eva. Repeat lines parsed from: {path}")
    row = [name, bottleneck, bcl]
    for m in METRICS:
        mean, std = stats(data[m])
        p = PLACES[m]
        row.append(f"{mean:.{p}f}±{std:.{p}f}")
    rows.append(row)

out_path = ROOT / "results" / "skelvq_fsq_comparison.csv"
with out_path.open("w", newline="") as f:
    csv.writer(f).writerows(rows)
print(f"wrote {out_path} ({len(rows) - 1} runs, {len(METRICS)} metrics)")
