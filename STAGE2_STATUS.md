# Stage-2 — MoScale + SkelVQ-FSQ integration

This is the on-ramp document for cluster work on stage 2 (text-to-motion AR
on top of the trained SkelVQ-FSQ tokenizer). The integration backbone is
landed and end-to-end smoke-tested (single-batch overfit: loss 2.11 → 0.009
in 100 steps on 7.35M-param AR). Architectural recipe verified from
**Infinity** (arxiv 2412.04431) and ported to FSQ.

## Quick start on the cluster

```bash
# 1. Clone + env (one-time)
git clone git@github.com:zhiwei-zzz/MoScale-plusplus-dev.git
cd MoScale-plusplus-dev
pip install -r requirements.txt           # SALAD's deps
pip install 'protobuf<5'                   # TensorBoard 2.14 compat
pip install huggingface_hub                # for ckpt download

# 2. Pull the trained SkelVQ-FSQ tokenizer from the HF private repo
export HF_TOKEN=hf_...                     # read-scope token from huggingface.co/settings/tokens
python scripts/download_checkpoints.py \
    --repo-id zzwalala/moscale-plusplus \
    --runs skelvq_fsq
# now at: checkpoints/t2m/skelvq_fsq/model/net_best_fid.tar

# 3. (For full SALAD eval / dataset access) set up data + evaluator
ln -s /your/path/to/HumanML3D dataset/humanml3d
bash prepare/download_t2m.sh               # pretrained evaluator
bash prepare/download_glove.sh

# 4. Verify the integration end-to-end (no dataset needed; uses random motion)
python moscale/test_skelvq_ar.py --steps 100 --lr 3e-4
# expect: final loss < 0.05, acc > 0.99
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
| FSQ index API | `models/vae/bsq.py:MultiScaleFSQ.encode_indices`, `indices_to_codes`, `effective_levels` | Per-scale `(B, L_s, D)` int indices in [0, 7); roundtrip exact vs `MultiScaleFSQ.forward(x)[0]`. **`effective_levels=7`** (not 8) — `round(half·tanh(z))` with banker's rounding gives 7 distinct levels for the L=8 setting; verified empirically. |
| SkelVQ wrapper | `moscale/model/vq/skelvq_wrapper.py:SkelVQWrapper` | HRVQVAE-shaped API around frozen SkelVQ-FSQ ckpt. Handles the SALAD ↔ moscale `utils/` import collision via sys.path / sys.modules isolation at construction time. Tested at T=64 (training distribution) and T=196 (MoScale's max length, L=343). |
| Level Self-Correction | `moscale/model/level_self_correction.py:LevelSelfCorrection` | Port of Infinity's `BitwiseSelfCorrection`. Bit-flip → uniform Categorical resample over {0..6}. Defaults: `noise_apply_strength=0.3`, `noise_apply_layers=-1` (off), `noise_apply_requant=True`. |
| AR transformer | `moscale/model/transformer/skelvq_ar.py:SkelVQAR` | Minimal but correct: `word_embed = Linear(d_vae, C)` over continuous code, block-causal mask, per-scale embedding, learned SOS, `head = Linear(C, d_vae × 7)` with per-channel CE. 7.35M params at default config. |
| Smoke test | `moscale/test_skelvq_ar.py` | Loads ckpt → encodes → LSC → AR forward + 100 backward steps. Verifies `loss < uniform_baseline × 0.7` and `acc > 1.5 / effective_levels`. |
| HF ckpt sync | `scripts/upload_checkpoints.py`, `scripts/download_checkpoints.py` | Push trained tokenizer (or all stage-1 runs) to a private HF model repo + pull on cluster. Defaults to uploading only `model/net_best_fid.tar` + `opt.txt` + `eval/eval.log` (~700 KB per run). |

## What's next (TODO on cluster)

### Required for a real text-to-motion training run

| # | Task | Where |
|---|---|---|
| 1 | Wire HumanML3D dataset into the AR training loop | `moscale/dataset/humanml3d_dataset.py` already loads motion + text; just needs an AR-specific collator that handles fixed-length T=196 padding |
| 2 | Add T5 text cross-attention to `SkelVQAR` | Pull from `moscale/model/transformer/moscale.py:200-220` (text_emb, text_norm, text_proj) and `transformer_helper.py:AdaLNSelfAttn(use_crossattn=True)`. Add to each `_Block`. |
| 3 | Classifier-free guidance | Training: drop text condition with prob `cond_drop_prob` (replace with learned `cfg_uncond` embedding). Inference: sample with `logits = cfg_scale * logits_cond + (1 - cfg_scale) * logits_uncond`. Reference: `moscale.py:485-498`. |
| 4 | Inference: scale-by-scale sampling decoder | Mirror `Infinity.autoregressive_infer_cfg` in `infinity.py:480-660`. At each scale: get logits → reshape `(B, L, D, V)` → top-k/top-p sample per channel → `indices_to_codes` → upsample → accumulate residual → feed to next scale. |
| 5 | Training launcher + config | New file `moscale/train_skelvq_ar.py` patterned on `train_moscale.py`. New config `moscale/config/train_skelvq_ar.yaml` with `vq_ckpt`, `noise_apply_*`, `cond_drop_prob`, etc. |
| 6 | Eval script | `moscale/eval_skelvq_ar.py` — generate text-conditioned motions, decode via SkelVQ wrapper, compute FID-on-generation. Mirror `eval_moscale.py`. |

### Optional (nice-to-have parity)

| # | Task |
|---|---|
| 7 | KV cache for inference (Infinity's `infer_use_kvcache`). |
| 8 | BERT-style mask augmentation (MoScale's parallel-prediction trick from `moscale.py:519-549`) — improves sample quality but isn't required for a working AR. |
| 9 | AdaLN conditioner (replaces our plain `LayerNorm` in `_Block`). |

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
└── moscale/
    ├── test_skelvq_ar.py                           # end-to-end overfit smoke test (cluster start point)
    └── model/
        ├── vq/skelvq_wrapper.py                    # HRVQVAE-shaped API around SkelVQ-FSQ
        ├── level_self_correction.py                # FSQ analogue of BitwiseSelfCorrection
        └── transformer/skelvq_ar.py                # minimal next-scale AR for FSQ
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
