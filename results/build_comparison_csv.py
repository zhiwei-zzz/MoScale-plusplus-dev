"""Parse the 20-repeat eval stdouts and emit a per-run row CSV with mean,
variance, and 95% confidence interval for every metric.
"""
import re
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    # bits_per_clip is computed for T=64 (training distribution), L=112,
    # residual cascade [8,4,2,1] -> per-scale cells [14,28,56,112] sum 210.
    # BSQ:  210 cells * 32 channels * 1 bit = 6720
    # FSQ:  210 cells * 32 channels * log2(7) ~= 18860  (7 effective levels)
    ("vae_repro",  "Gaussian + KL",                  "continuous (~17000-22000 effective)",
     ROOT / "training_logs/vae_repro_finaleval.stdout"),
    ("skelvq_bsq", "BSQ (1 bit x 32 ch x 4 scales)",  "6720",
     ROOT / "training_logs/skelvq_bsq_finaleval.stdout"),
    ("skelvq_fsq", "FSQ L=8 (log2(7) bit x 32 ch x 4 scales)", "18860",
     ROOT / "training_logs/skelvq_fsq_finaleval.stdout"),
]

# Match the test_skel_vq.py format:
#   --> Eva. Repeat 7 : FID. 0.0067, Diversity Real. 9.5975, Diversity. 9.5762,
#       R_precision_real. (...), R_precision. (0.5142, 0.7045, 0.7944),
#       matching_real. 2.9905, matching_pred. 2.9988, MPJPE. 0.0188, Multimodality. 0.0000
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
    var = sum((x - mean) ** 2 for x in xs) / n               # population variance
    std = math.sqrt(var)
    conf = std * 1.96 / math.sqrt(n)                          # 95% CI half-width
    return mean, var, conf


def fmt(v, places):
    return f"{v:.{places}f}"


# Decimal places per metric (FID/MPJPE need extra precision for variance).
PLACES = {"FID": 4, "Diversity": 3, "TOP1": 4, "TOP2": 4, "TOP3": 4,
          "Matching": 4, "MPJPE": 4, "Multimodality": 4}
VAR_PLACES = {"FID": 8, "Diversity": 4, "TOP1": 6, "TOP2": 6, "TOP3": 6,
              "Matching": 5, "MPJPE": 7, "Multimodality": 6}

header = ["run", "bottleneck", "bits_per_clip"] + METRICS

rows = [header]
for name, bottleneck, bits, path in RUNS:
    if not path.exists():
        raise SystemExit(f"missing eval stdout: {path}")
    data = parse(path)
    row = [name, bottleneck, bits]
    for m in METRICS:
        mean, var, _conf = stats(data[m])
        std = math.sqrt(var)
        row.append(f"{fmt(mean, PLACES[m])}±{fmt(std, PLACES[m])}")
    rows.append(row)

out_path = ROOT / "results" / "skelvq_comparison.csv"
with out_path.open("w", newline="") as f:
    csv.writer(f).writerows(rows)
print(f"wrote {out_path} ({len(rows) - 1} runs, {len(METRICS)} metrics)")
