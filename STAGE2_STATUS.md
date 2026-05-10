# Stage-2 status — MoScale + SkelVQ-FSQ integration

Architectural recipe verified from Infinity (arxiv 2412.04431) and ported
to FSQ. End-to-end smoke-tested (single-batch overfit: loss 2.11 → 0.009 in
100 steps).

## What's done

| Component | Path | Notes |
|---|---|---|
| FSQ index API | `models/vae/bsq.py:MultiScaleFSQ.encode_indices`, `indices_to_codes`, `effective_levels` | Per-scale `(B, L_s, D)` int indices in [0, 7); roundtrip exact vs `MultiScaleFSQ.forward(x)[0]`. **Effective_levels=7** (not 8) — round(half·tanh(z)) with banker's rounding gives 7 distinct levels for L=8 setting; verified empirically. |
| SkelVQ wrapper | `moscale/model/vq/skelvq_wrapper.py:SkelVQWrapper` | HRVQVAE-shaped API around frozen SkelVQ-FSQ ckpt. Handles the SALAD/moscale `utils/` import collision via sys.path/sys.modules isolation at construction time. Tested at T=64 (training distribution) and T=196 (MoScale's max length, L=343). |
| Level Self-Correction | `moscale/model/level_self_correction.py:LevelSelfCorrection` | Port of Infinity's `BitwiseSelfCorrection`. Bit-flip → uniform Categorical resample over {0..6}. Defaults: `noise_apply_strength=0.3`, `noise_apply_layers=-1` (off), `noise_apply_requant=True`. |
| AR transformer | `moscale/model/transformer/skelvq_ar.py:SkelVQAR` | Minimal but correct: `word_embed = Linear(d_vae, C)` over continuous code, block-causal mask, per-scale embedding, learned SOS, `head = Linear(C, d_vae × 7)` with per-channel CE. 7.35M params at default config. |
| Smoke test | `moscale/test_skelvq_ar.py` | Loads ckpt → encodes → LSC → AR forward + 100 backward steps. Verifies `loss < uniform_baseline × 0.7` and `acc > 1.5 / effective_levels`. |

Reproduce locally:

```bash
cd MoScale-plusplus-dev
python moscale/test_skelvq_ar.py --steps 100 --lr 3e-4
# expect: final loss < 0.05, acc > 0.99
```

## What's next (cluster work)

### Required for a real training run

| # | Task | Where |
|---|---|---|
| 1 | Wire HumanML3D dataset into the AR training loop | `moscale/dataset/humanml3d_dataset.py` already loads motion + text; just needs the AR-specific collator that handles fixed-length T=196 padding |
| 2 | Add T5 text cross-attention to `SkelVQAR` | Pull from `moscale/model/transformer/moscale.py:200-220` (text_emb, text_norm, text_proj) and `transformer_helper.py:AdaLNSelfAttn(use_crossattn=True)`. Add to each `_Block`. |
| 3 | Classifier-free guidance | At training, drop text condition with prob `cond_drop_prob` (replace with learned `cfg_uncond` embedding). At inference, sample with `cfg = cfg_scale * logits_cond + (1 - cfg_scale) * logits_uncond`. Reference: `moscale.py:485-498`. |
| 4 | Inference: scale-by-scale sampling decoder | Mirror `Infinity.autoregressive_infer_cfg` in `infinity.py:480-660`. At each scale: get logits → reshape `(B, L, D, V)` → top-k/top-p sample per channel → `indices_to_codes` → upsample → accumulate residual → feed to next scale. |
| 5 | Training launcher + config | New file `moscale/train_skelvq_ar.py` patterned on `train_moscale.py`. New config `moscale/config/train_skelvq_ar.yaml` with `vq_ckpt`, `noise_apply_*`, `cond_drop_prob`, etc. |
| 6 | Eval script | `moscale/eval_skelvq_ar.py` — generate text-conditioned motions, decode via SkelVQ wrapper, compute FID-on-generation. Mirror `eval_moscale.py`. |

### Optional (nice-to-have parity)

| # | Task |
|---|---|
| 7 | KV cache for inference (Infinity's `infer_use_kvcache`). |
| 8 | BERT-style mask augmentation (MoScale's parallel-prediction trick from `moscale.py:519-549`) — improves sample quality but isn't required for a working AR. |
| 9 | AdaLN conditioner (replaces our plain LayerNorm in `_Block`). |

### Critical gotchas already handled

1. **utils/ import collision** between SALAD and moscale: both use bare `from utils.X import Y`. The wrapper isolates SALAD imports via `_import_skelvq_with_isolated_utils()` — saves/restores sys.path + sys.modules around the SkelVQ import chain. Don't remove this.

2. **Banker's rounding edge case** for FSQ L=8: `round(half * tanh(z))` clamped to `[-int(half), int(half)]` gives 7 distinct levels. The 8th level (±4 from banker's rounding of 3.5) is rare and dropped for a clean vocab. The trained model uses all 7 reachable levels.

3. **FSQ has no `quant_resi`** — MSQuantizer's per-scale residual conv. Wrapper's `quantizer.quant_resi[α]` returns identity (no residual conv applied). FSQ residual cascade is pure interpolate-and-subtract. If you port `Infinity.autoregressive_infer_cfg`, drop the `quant_resi` calls.

4. **Window length:** SkelVQ trained at T=64 (L=112) but the encoder is fully convolutional in time, so it runs cleanly at T=196 (L=343). All 7 levels are reachable at L=343; mild distribution drift but no crash. Scales [8,4,2,1] become per-scale lengths [42, 85, 171, 343], summing to AR sequence length 641.

## File tree of new code

```
SALAD/
├── models/vae/bsq.py                              # +85 lines: encode_indices, indices_to_codes, effective_levels
├── moscale/
│   ├── model/
│   │   ├── vq/skelvq_wrapper.py                   # NEW: HRVQVAE-shaped API around SkelVQ-FSQ
│   │   ├── level_self_correction.py               # NEW: FSQ analogue of BitwiseSelfCorrection
│   │   └── transformer/skelvq_ar.py               # NEW: minimal next-scale AR for FSQ
│   └── test_skelvq_ar.py                          # NEW: end-to-end overfit smoke test
└── test_skelvq_fsq_indices.py                     # NEW: bsq.py + ckpt smoke tests
```

Existing files untouched: SALAD's encoder/decoder (`models/vae/encdec.py`),
trainer (`models/vae/skel_vq_trainer.py`), MoScale's transformer
(`moscale/model/transformer/moscale.py`), MoScale's trainer
(`moscale/trainers/transformer_trainer.py`).
