# SkelVQ — SALAD encoder + discrete tokenization (status)

This branch extends SALAD's VAE with discrete-bottleneck variants for downstream
next-scale autoregressive generation. The encoder/decoder come from SALAD unchanged;
the bottleneck is replaced. See `models/vae/skel_vq.py` for the entry point.

## Goal

Build a tokenizer that:
1. Reuses SALAD's skeleton-aware structured encoder (preserves `(T_b, J_b, D)` lattice).
2. Replaces SALAD's continuous Gaussian + KL bottleneck with a **discrete** bottleneck
   suitable for autoregressive prediction (downstream stage 2, à la VAR / MoScale).
3. Stays within striking distance of the continuous SALAD VAE on motion FID, so the
   codes don't lose so much information that downstream generation suffers.

## Variants implemented

All three live behind a single `--quantizer_type` flag in
`options/skel_vq_option.py`. Each is a single-line config change at training time.

| `--quantizer_type` | Bottleneck | Source | File |
|---|---|---|---|
| `msvq` | MoScale's MSQuantizer (learned codebook, 512 entries × 32 dim) | MoScale repo | `models/vae/quantizer.py` |
| `bsq` | Binary Spherical Quantization (Infinity-style) — `l2norm(z)` then `sign(z)/√D`, with entropy regularizer | Infinity (arxiv 2412.04431) | `models/vae/bsq.py:BSQ` / `MultiScaleBSQ` |
| `fsq` | Finite Scalar Quantization (Mentzer ICLR 2024) — `tanh(z)` then `round` to one of L levels per channel; optional LFQ-style entropy regularizer | Mentzer + our extension | `models/vae/bsq.py:FSQ` / `MultiScaleFSQ` |

The encoder/decoder is SALAD's verbatim — `MotionEncoder` + `STConvEncoder` +
`STConvDecoder` + `MotionDecoder` from `models/vae/encdec.py`. Only the bottleneck
between them is swapped.

## Results so far (HumanML3D, 50 epochs, window_size=64)

All runs use SALAD's exact training config: batch 256, lr 2e-4, warmup 2000 iters,
milestones `[150_000, 250_000]` iters, gamma 0.05, λ_pos = 0.5, λ_vel = 0.5,
window 64, latent_dim 32, n_layers 2.

20-repeat eval on `net_best_fid.tar` via `python test_*.py` (paper-style protocol):

| Run | Bottleneck | bits/clip | FID ↓ | MPJPE ↓ | Diversity | TOP3 |
|---|---|---|---|---|---|---|
| `vae_repro` | continuous Gaussian + KL | (effective ~17-22k) | **0.003 ± 0.000** | 0.016 | 9.510 | 0.797 |
| `skelvq_bsq` | BSQ (1 bit/channel × 32 × 4 scales) | 14,336 | 0.021 ± 0.000 | 0.029 | 9.432 | 0.796 |
| `skelvq_fsq` | FSQ L=8 (3 bits/channel × 32 × 4 scales) | 43,008 | **0.007 ± 0.000** | 0.019 | 9.438 | 0.797 |

Full per-metric breakdown with variance and 95% CI: `results/skelvq_comparison.csv`
(regenerate with `python results/build_comparison_csv.py`).

**Headline:** FSQ closes ~78% of the BSQ→VAE FID gap (BSQ was 7× the VAE; FSQ is 2.3×)
while matching VAE on R-precision and Diversity. The skelvq_fsq_reg regularized variant
was descoped — entropy regularization is queued only if there's a downstream reason to
push level utilization above 0.42.

**Key per-term observations** (validation losses at ep 40, mid-FSQ run):

| | recon | vel | pos |
|---|---|---|---|
| `vae_repro` | 0.0205 | 0.0170 | 0.0050 |
| `skelvq_bsq` | 0.047 | 0.034 | 0.016 |
| `skelvq_fsq` (live) | 0.0211 | 0.0170 | 0.0052 |

So FSQ matches VAE on per-term reconstruction within 1-3% across all three terms, while
BSQ is ~2× worse on each. Per-epoch FID still has a residual ~1.5× gap between FSQ and
VAE, suggesting the gap is in higher-level perceptual quality not captured by SmoothL1.

## Why FSQ over BSQ for our task

Two reasons (neither alone is sufficient explanation):

1. **Capacity per cell**. With short motion clips (112 cells × 4 scales = 448 token
   slots) we get ~30× fewer total cells than Infinity's image setup (1024 × 13 ≈ 13k).
   Per-degree-of-freedom: Infinity 2.16 bits/dof; BSQ on our setup 0.85 bits/dof.
   FSQ L=8 brings us to 2.55 bits/dof — back into Infinity's regime.

2. **Magnitude preservation**. BSQ requires `l2norm(z)` before `sign(z)`, throwing away
   magnitude information. Pose features (especially `loss_pos` — joint position offsets)
   carry meaningful small-magnitude continuous values; this is the term BSQ hurts most
   (3× ratio vs VAE). FSQ uses `tanh` (no L2-norm) and preserves magnitude; pos loss
   matches VAE.

These two effects are confounded in the BSQ→FSQ comparison — a clean attribution would
require ablations:
- BSQ + more residual scales (capacity only).
- FSQ L=4 (intermediate capacity, magnitude preserved).
- BSQ without L2-norm (capacity unchanged, magnitude preserved).

These ablations are TODO; one cluster-friendly experiment per setup.

## Open issues

1. **FSQ level under-utilization.** Training-time `Train/fsq_diag = 0.42` (mean abs(q))
   indicates the encoder is using ~4-5 of the 8 available levels per channel. The
   `skelvq_fsq_reg` queued run will test whether an LFQ-style entropy regularizer pushes
   `fsq_diag` toward 0.55+ and improves FID, or whether the encoder's preference for
   center levels reflects a recon optimum that regularization can't beat.
2. **Confound between capacity and magnitude.** See above.
3. **FID gap remains ~1.5× VAE** even with FSQ. Per-term reconstruction is essentially
   tied; the gap is in perceptual / temporal-smoothness quality the eval encoder picks
   up on. Adversarial loss or a perceptual recon term could close this; not yet tried.
4. **Stage 2 (next-scale AR) not implemented.** All work so far is the tokenizer
   (stage 1). Plan for stage 2 sketched in commentary; concrete code TODO.

## How to reproduce

### Setup

```bash
git clone git@github.com:zhiwei-zzz/MoScale-plusplus-dev.git
cd MoScale-plusplus-dev
pip install -r requirements.txt        # SALAD's deps
pip install 'protobuf<5'                # for TensorBoard 2.14 compat (one-time)

# Symlink HumanML3D into the expected location:
ln -s /path/to/HumanML3D dataset/humanml3d

# Download SALAD's evaluator + glove (needed for the FID eval):
bash prepare/download_t2m.sh
bash prepare/download_glove.sh
```

### Train (stage 1)

```bash
# Continuous baseline (SALAD VAE, ~22 hr on a 3090 Ti):
python train_vae.py --name vae_repro

# BSQ (binary, capacity-limited):
python train_skel_vq.py --name skelvq_bsq --quantizer_type bsq

# FSQ L=8 (recommended discrete variant):
python train_skel_vq.py --name skelvq_fsq --quantizer_type fsq --fsq_levels 8

# FSQ L=8 + entropy regularizer (push level utilization):
python train_skel_vq.py --name skelvq_fsq_reg --quantizer_type fsq --fsq_levels 8 \
    --fsq_entropy_weight 0.1 --fsq_inv_temperature 20.0
```

### Eval (20-repeat, publishable protocol)

```bash
python test_vae.py --name vae_repro
python test_skel_vq.py --name skelvq_bsq --quantizer_type bsq
python test_skel_vq.py --name skelvq_fsq --quantizer_type fsq --fsq_levels 8
```

Each writes `checkpoints/t2m/<name>/eval/eval.log` with FID, Diversity, R-precision,
MPJPE, Multimodality.

### Smoke tests

```bash
python test_skel_vq_smoke.py        # SkelVQ + MSQuantizer (learned codebook)
python test_skel_vq_bsq_smoke.py    # SkelVQ + BSQ
python test_skel_vq_fsq_smoke.py    # SkelVQ + FSQ
```

All three should print "[OK]" within ~5 sec on a GPU.

## Files added in this branch

```
models/vae/quantizer.py            # MoScale MSQuantizer (copied verbatim)
models/vae/skel_vq.py              # SkelVQ class — encoder/decoder + bottleneck switch
models/vae/skel_vq_trainer.py      # Trainer (forked from VAETrainer)
models/vae/bsq.py                  # BSQ + FSQ + their multi-scale wrappers
options/skel_vq_option.py          # CLI options for the new training script
train_skel_vq.py                   # Training entrypoint
test_skel_vq.py                    # Evaluation entrypoint
test_skel_vq_*_smoke.py            # Quick smoke tests for each quantizer
plot_loss.py, plot_curves.py       # Lightweight TB-curve visualizers (optional)
watchdog_*.sh                      # Run-chaining helpers used in dev (optional)
SKELVQ_STATUS.md                   # This file
```

## Files patched (minimal compat fixes for newer numpy/matplotlib)

```
common/quaternion.py               # np.float -> float (deprecated alias)
utils/motion_process.py            # np.float -> float
```

These are the only edits to upstream SALAD code.
