# Stage-2 — MoScale + SkelVQ-FSQ integration

This is the on-ramp document for cluster work on stage 2 (text-to-motion AR
on top of the trained SkelVQ-FSQ tokenizer). **Pipeline is now end-to-end
ready** — training launcher verified on real HumanML3D (loss 0.66, acc 70%
at 7k iters), generate() runs scale-by-scale with CFG, eval script computes
FID via SALAD's evaluator. Architecture: MoScale's transformer adapted with
mask path removed (see `moscale_fsq.py`); recipe matches **Infinity**
(arxiv 2412.04431) for the FSQ side.

## Quick start on the cluster

```bash
# 1. Clone + env (one-time)
git clone git@github.com:zhiwei-zzz/MoScale-plusplus-dev.git
cd MoScale-plusplus-dev
pip install -r requirements.txt           # SALAD's deps
pip install 'protobuf<5' omegaconf huggingface_hub
                                          # protobuf<5 for TensorBoard 2.14 compat
                                          # omegaconf for MoScaleFSQ cfg
                                          # huggingface_hub for ckpt download

# 2. Pull the trained SkelVQ-FSQ tokenizer from the HF private repo
export HF_TOKEN=hf_...                     # read-scope token
python scripts/download_checkpoints.py \
    --repo-id zzwalala/moscale-plusplus \
    --runs skelvq_fsq

# 3. Set up HumanML3D + the SALAD evaluator
ln -s /your/path/to/HumanML3D dataset/humanml3d
bash prepare/download_t2m.sh               # pretrained evaluator
bash prepare/download_glove.sh

# 4. (Optional) Verify the architecture roundtrips before training
python moscale/test_skelvq_ar.py --steps 100 --lr 3e-4
# expect: final loss < 0.05, acc > 0.99 (this overfits a random batch through
# the SkelVQAR minimal backbone; it's an architectural sanity check, not a
# real training run)

# 5. Train the text-to-motion AR (MoScaleFSQ)
python train_skelvq_ar.py \
    --name skelvq_ar_v1 \
    --batch_size 32 --max_epoch 200 \
    --num_layers 8 --num_heads 8 --latent_dim 384 \
    --cond_drop_prob 0.1 \
    --use_wandb

# 6. After training, evaluate generation FID (20-rep, paper-style)
python eval_skelvq_ar.py \
    --name skelvq_ar_v1 \
    --tokenizer_ckpt checkpoints/t2m/skelvq_fsq/model/net_best_fid.tar \
    --ar_ckpt checkpoints/t2m/skelvq_ar_v1/model/net_best_loss.tar \
    --cond_scale 4.0 --top_p 0.9 --repeat_times 20
```

## Other docs

| File | What it covers |
|---|---|
| [`SKELVQ_STATUS.md`](SKELVQ_STATUS.md) | **Stage 1**: the SkelVQ tokenizer (encoder + FSQ bottleneck + decoder). Final 20-rep eval numbers, training config, BSQ→FSQ design rationale. |
| `STAGE2_STATUS.md` (this file) | **Stage 2**: the AR transformer over FSQ tokens. |
| [`results/skelvq_comparison.csv`](results/skelvq_comparison.csv) | Tokenizer comparison table (vae_repro / skelvq_bsq / skelvq_fsq), mean ± std on FID/MPJPE/R-precision/Diversity. |
| [`README.md`](README.md) | Upstream SALAD README (preserved). |
| [`moscale/README.md`](moscale/README.md) | Upstream MoScale README (vendored). |
| [`moscale/VENDORED.md`](moscale/VENDORED.md) | Provenance of the MoScale vendor (commit pin + update procedure). |

## What's done in stage 2

| Component | Path | Notes |
|---|---|---|
| FSQ index API | `models/vae/bsq.py:MultiScaleFSQ.encode_indices`, `indices_to_codes`, `effective_levels` | Per-scale `(B, L_s, D)` int indices in [0, 7). **`effective_levels=7`** (not 8) — banker's rounding on `round(half·tanh(z))` gives 7 distinct levels for the L=8 setting. |
| SkelVQ wrapper | `moscale/model/vq/skelvq_wrapper.py:SkelVQWrapper` | HRVQVAE-shaped API around frozen SkelVQ-FSQ ckpt. Handles the SALAD ↔ moscale `utils/` import collision via sys.path / sys.modules isolation. Verified at T=64 (training) and T=196 (L=343). |
| Level Self-Correction | `moscale/model/level_self_correction.py:LevelSelfCorrection` | Port of Infinity's `BitwiseSelfCorrection`. Uniform Categorical resample over {0..6}. Wired only in the minimal `SkelVQAR` path; `MoScaleFSQ` doesn't use it yet (works without). |
| **MoScaleFSQ (text-to-motion AR)** | `moscale/model/transformer/moscale_fsq.py:MoScaleFSQ` | **Surgical fork of MoScale's transformer**: T5 cross-attention, AdaLN, RoPE preserved verbatim; output head emits `code_dim*effective_levels` logits per position; per-channel CE loss; BERT-style mask augmentation removed; `generate()` is single-pass per scale. ~15M params at depth-4. |
| Caption-aware data loader | `data/t2m_caption_dataset.py:Text2MotionWindowDataset` | Window-based caption pairs (caption + 64-frame motion window) using SALAD's evaluator stats for normalization (matches the tokenizer's training distribution). |
| Training launcher | `train_skelvq_ar.py` | Loads frozen tokenizer, builds `MoScaleFSQ`, trains text-conditional AR with CFG dropout. **Verified on real HumanML3D**: loss 0.66, accuracy 70% at iter 7200 (3 min on a 3090 Ti). |
| Generation eval | `eval_skelvq_ar.py` | Generates motion from captions, decodes via wrapper, computes FID + R-precision + Diversity via SALAD's evaluator. 20-rep paper-style protocol. |
| Minimal backbone (alt) | `moscale/model/transformer/skelvq_ar.py:SkelVQAR` | From-scratch reference: cleaner code, fewer features (no AdaLN, no RoPE). Single-batch overfit verified (loss 2.11 → 0.009). Useful for ablations or as a fallback if MoScaleFSQ has bugs. |
| Smoke test | `moscale/test_skelvq_ar.py` | 100-step single-batch overfit through the minimal `SkelVQAR` — fast architectural sanity check independent of MoScaleFSQ. |
| HF ckpt sync | `scripts/upload_checkpoints.py`, `scripts/download_checkpoints.py` | Push/pull trained checkpoints between local and cluster via a private HF model repo. |

## What's optional / not yet done

| # | Task |
|---|---|
| 1 | KV cache at inference (`infer_use_kvcache=True` would speed up `generate()` substantially; the path exists in `moscale.py` but isn't yet wired into the FSQ generate). |
| 2 | BERT-style mask augmentation (MoScale's parallel-prediction trick) — *intentionally* removed for stage 2; can be added back via `moscale.py` codepath if pure AR underperforms. |
| 3 | LSC (`LevelSelfCorrection`) plumbed into `MoScaleFSQ.preprocess_motion_for_training` — currently only the minimal `SkelVQAR` path uses LSC. |
| 4 | Variable-length training (current loader is window-based with all positions valid). |
| 5 | KIT support — option exists but not exercised. |

## Critical gotchas already handled

1. **`utils/` import collision** between SALAD and moscale: both use bare `from utils.X import Y`. The wrapper isolates SALAD imports via `_import_skelvq_with_isolated_utils()` — saves/restores sys.path + sys.modules around the SkelVQ import chain. **Do not remove this.**

2. **Banker's rounding edge case** for FSQ L=8: `round(half * tanh(z))` clamped to `[-int(half), int(half)]` gives 7 distinct levels. The 8th level (±4 from banker's rounding of 3.5) is rare and dropped for a clean vocab. The trained model uses all 7 reachable levels.

3. **FSQ has no `quant_resi`** — that's MSQuantizer's per-scale residual conv. Wrapper's `quantizer.quant_resi[α]` returns identity (no residual conv applied). FSQ residual cascade is pure interpolate-and-subtract. If you port `Infinity.autoregressive_infer_cfg`, drop the `quant_resi` calls.

4. **Window length:** SkelVQ trained at T=64 (L=112) but the encoder is fully convolutional in time, so it runs cleanly at T=196 (L=343). All 7 levels are reachable at L=343; mild distribution drift but no crash. Scales [8,4,2,1] become per-scale lengths [42, 85, 171, 343], summing to AR sequence length **641**.

## File tree of new code

```
MoScale-plusplus-dev/
├── STAGE2_STATUS.md                                # this file
├── SKELVQ_STATUS.md                                # stage-1 (tokenizer) status
├── results/skelvq_comparison.csv                   # tokenizer comparison table
├── scripts/
│   ├── upload_checkpoints.py                       # push ckpts -> HF
│   └── download_checkpoints.py                     # pull ckpts <- HF
├── models/vae/bsq.py                               # +85 lines: encode_indices, indices_to_codes, effective_levels
├── test_skelvq_fsq_indices.py                      # bsq.py + ckpt smoke tests
├── data/t2m_caption_dataset.py                     # caption-aware HumanML3D loader for the AR training
├── train_skelvq_ar.py                              # text-to-motion AR training launcher (uses MoScaleFSQ)
├── eval_skelvq_ar.py                               # generation FID + R-precision + Diversity eval
├── options/skelvq_ar_option.py                     # CLI options for the AR launcher
├── utils/__init__.py                               # makes SALAD's utils a regular package (resolution priority)
└── moscale/
    ├── test_skelvq_ar.py                           # minimal-backbone single-batch overfit test
    └── model/
        ├── vq/skelvq_wrapper.py                    # HRVQVAE-shaped API around SkelVQ-FSQ
        ├── level_self_correction.py                # FSQ analogue of BitwiseSelfCorrection
        └── transformer/
            ├── moscale_fsq.py                      # ★ text-to-motion AR (MoScale fork; mask path removed)
            └── skelvq_ar.py                        # minimal alternative backbone (no AdaLN/RoPE)
```

Existing files untouched: SALAD's encoder/decoder (`models/vae/encdec.py`),
trainer (`models/vae/skel_vq_trainer.py`), MoScale's transformer
(`moscale/model/transformer/moscale.py`), MoScale's trainer
(`moscale/trainers/transformer_trainer.py`).

## Acceptance criterion

After steps 1–5 above, `python moscale/train_skelvq_ar.py` should run without
crashing for at least one epoch on HumanML3D, and `python
moscale/eval_skelvq_ar.py` should produce a finite generation FID. Beyond
that, the target is a generation FID within ~30% of MoScale's published
HRVQVAE-based number (slack accounts for the ~2× tokenizer FID gap between
MSVQ and FSQ in stage-1 numbers; see `results/skelvq_comparison.csv`).
