# MoScaleFSQ AR — Next Steps

Baseline (run `90ds465n`): L=16, H=768, perturb_hi=0.8, 120 epochs, SALAD-aligned no-masking pipeline.
- Best FID **0.396** at epoch ~104, final 0.501.
- Val acc per scale: `[0.81, 0.88, 0.88, 0.88]` — scale 0 lags.
- MoScale published target: FID 0.06 (codebook tokenizer). Stage-1 FSQ recon: 0.0071. Gap is on the AR side.

Three things to try, in order. Each step is gated on the previous result — don't run #2 before knowing #1's outcome.

---

## 1. Perturbation rate sweep (cheapest, run first)

The old default `perturb_hi=0.1` and "sweet spot 0.1" comment in `options/skelvq_ar_option.py:74-78` was made under the **buggy masking pipeline**. With the SALAD-aligned fix + epoch-based scheduler, run `90ds465n` already hit FID 0.40 at `perturb_hi=0.8`. The optimum under the new pipeline is unknown.

**Cost**: 5 × ~30-epoch runs on cluster ≈ 5 × 8h = 40 GPU-hours.

**Action**:
- [ ] Run 30-epoch sweeps at `perturb_hi ∈ {0.0, 0.1, 0.3, 0.5, 0.8}` with everything else fixed to the `90ds465n` config (L=16, H=768, max_epoch=30, milestones=[15, 22], gamma=0.05 — proportional milestone scaling).
- [ ] Pick the best and run it full 120 epochs.
- [ ] Update the stale comment in `options/skelvq_ar_option.py` with the new ablation table.

**Acceptance**: a tightened perturb_hi ablation table (best FID per setting at 30 ep + 1 full 120 ep at the winner). Pass if winner ≤ 0.40 at 120 ep.

---

## 2. CFG / sampling-knob scan (inference-only, no retrain)

Free wins on the existing `90ds465n` ckpt. Knobs currently hardcoded:
- `cond_scale = 4.0`
- `top_p = 0.9`, `temperature = 1.0`
- Per-scale CFG decay `t = cond_scale + (i-1)*(-0.25)` → `[4.00, 3.75, 3.50, 3.25]`

**Cost**: ~2-4h of cluster eval, no training. Each FID eval is ~30-60s × 32 settings × 3 repeats ≈ 1-2h.

**Action**:
- [ ] **CFG scale**: sweep flat `cond_scale ∈ {2, 3, 4, 5, 6, 8}` with decay disabled. Pick winner.
- [ ] **top-p**: at the winning CFG, sweep `top_p ∈ {0.5, 0.7, 0.85, 0.95, 1.0}`.
- [ ] **Temperature**: at the winning (CFG, top-p), sweep `temperature ∈ {0.7, 0.9, 1.0, 1.1}`.
- [ ] **Per-scale CFG profile**: try `[6, 4, 3, 2]` (high at scale 0, low at scale 3) and `[3, 4, 5, 6]` (the opposite). Scale 0 acc lags, so giving it stronger text guidance may help.
- [ ] Build a small `sweep_sampling.py` that reuses `evaluate_once` with overridable cond_scale/top_p/temperature args. Log results to a CSV.

**Acceptance**: a CSV table of FID per setting; report the best combo. Target: FID ≤ 0.35 with no training change.

---

## 3. Joint per-position prediction (architectural change, retrain needed)

The current head emits 384 independent 7-way categoricals per position. With per-channel accuracy ~0.87, joint-position accuracy is ~`0.87^384 ≈ 10^-23` — the independence assumption is leaving a lot on the table. Add a lightweight intra-position channel-coupling module.

**Approach** (cheapest variant): add a 2-3 layer per-position channel transformer on top of the existing block output, before the final 7-way head.

```
block output  (B, L, 768)                                          # current
   ↓ Linear(768 → 384·d_ch) reshape → (B, L, 384, d_ch)           # d_ch = 32 or 64
   ↓ within-position channel-attention: 2-3 transformer layers acting over the 384-channel axis
   ↓ Linear(d_ch → 7) head                                          # final logits per channel
```

Within-position attention has cost `O(B·L · 384²·d_ch) = O(B·641·150k·d_ch)` — manageable if `d_ch ≤ 64`. The 384-channel axis is the "sequence" inside this mini-transformer; each position runs an independent stack.

**Action**:
- [ ] Implement `ChannelCouplingHead` in `moscale/model/transformer/moscale_fsq.py` as a swap-in replacement for `self.head = nn.Linear(head_latent_dim, fsq_V)`.
- [ ] Gate it behind a config flag `cfg.model.channel_coupling.enabled` so the old path still works.
- [ ] Feasibility check first: forward + 1 backward step on a `(B=2, L=641, 768)` tensor — make sure shapes line up, no NaN, gradient flows. (Per memory: feasibility check before long training runs.)
- [ ] Train at L=16/H=768 for 60 epochs (half budget) to see if `Val/acc_scale_*` improves over baseline at the same epoch.
- [ ] If acc improves and CE drops, train to full 120 epochs.

**Acceptance**: per-channel val acc ≥ baseline at every scale; FID strictly improves over the best from steps 1 + 2. Target: FID ≤ 0.25.

---

## 4. 2D-cascade tokenizer + RoPE2D AR — ScaleMoGen port (Option A)

ScaleMoGen (arxiv 2605.11704) reports FID 0.030 vs MoMask's 0.045 on HumanML3D
by (a) arranging per-scale tokens as 2D (temporal × skeletal) grids instead of
1D flattening, and (b) using RoPE2D over those grids. This is a direct test of
whether the same wins apply to our FSQ tokenizer.

**Scope** (Option A, "time-only downsampling"):
- Per-scale token maps: `(T_s, J=7)` for T_s in `[6, 12, 24, 49]`. J fixed.
- Per-scale token counts: `[42, 84, 168, 343]` (total 637, was 641).
- Tokenizer cascade: 2D `area` downsample + `bilinear` upsample.
- AR positional encoding: RoPE2D, head_dim split half→t, half→j.
- J subdivision per scale (a la ScaleMoGen's skeletal partition) is deferred.

**Code** (landed on `main`):
- [x] `models/vae/bsq.py`: `MultiScaleFSQ` gains `cascade_mode={"1d","2d"}`;
       new `_forward_2d` and `encode_indices_2d` methods.
- [x] `models/vae/skel_vq.py`: when `quantizer_cascade="2d"`, passes
       `(B, D, T_b, J_b)` to the FSQ directly (no flatten).
- [x] `options/skel_vq_option.py`: `--quantizer_cascade {1d,2d}` flag.
- [x] `moscale/model/transformer/moscale_fsq.py`: new
       `precompute_rope2d_for_batch`; `MoScaleFSQ` reads
       `cfg.model.use_rope2d`; `create_PE`, `preprocess_motion_for_training`,
       `get_next_autoregressive_input`, and `generate` all branch on the flag.
- [x] `moscale/model/vq/skelvq_wrapper.py`: auto-detects `cascade_mode` from
       the sibling `opt.txt`; encode/decode-from-indices route to 2D paths.
- [x] `options/skelvq_ar_option.py`: `--use_rope2d` flag.
- [x] `train_skelvq_ar.py`: threads `use_rope2d`/`rope2d_J` into cfg; asserts
       tokenizer-cascade vs AR-flag consistency at startup.

**Feasibility checks** (passing locally):
- [x] `scripts/feasibility_check_2d.py` (run from SALAD root): 2D FSQ
       cascade shapes + gradients OK; SkelVQ end-to-end recon at 2D OK
       (272/292 params with non-zero grad).
- [x] `moscale/scripts/feasibility_check_2d_ar.py` (run from `moscale/`):
       RoPE2D base tensor structure OK; MoScaleFSQ with `use_rope2d=True`
       forward+backward OK (loss=1.946=ln(7), acc=0.142=1/7 at init — exactly
       random-init baseline); generate produces `[(42,32), (84,32), (168,32),
       (343,32)]` per-scale shapes.

**Cluster handoff (still to do)**:
- [ ] Retrain SkelVQ-FSQ tokenizer at 2D for the same 50-epoch budget:
       ```
       python train_skel_hrvqvae.py --name skelvq_fsq_2d \
         --quantizer_type fsq --quantizer_cascade 2d \
         <same hyperparams as the 1D ckpt>
       ```
       Estimated ~6h on a 3090. Acceptance: recon FID within 2× of the 1D
       ckpt's 0.0071 (margin allows for the cascade-math change; tokenizer
       isn't the bottleneck so this is a sanity floor, not a target).
- [ ] Retrain MoScaleFSQ AR at 2D with the new tokenizer, same L=16/H=768 +
       perturb=0.8 + scheduler config as `90ds465n`:
       ```
       python train_skelvq_ar.py --name skelvq_ar_l16_h768_perturb080_2d \
         --use_rope2d --tokenizer_ckpt checkpoints/t2m/skelvq_fsq_2d/model/net_best_fid.tar \
         <rest of 90ds465n config>
       ```
       Estimated ~30h. Acceptance: FID strictly better than `90ds465n`'s 0.40.
       Target: ScaleMoGen's reported 0.030 ballpark.
- [ ] If FID improves: keep 2D as the new baseline and re-run Tier 1 perturb
       sweep + Tier 2 CFG scan on top. If it doesn't: investigate whether the
       additional skeletal-partition piece from ScaleMoGen (Option B) is
       carrying the wins, not the 2D arrangement alone.

---

## Out of scope for now

These came up in the discussion but are deferred until 1-3 are explored:
- Per-scale loss reweighting (would compete with #3 for attribution credit — pick one architectural change at a time)
- Iterative MaskGIT-style refinement on scale 0
- Narrower FSQ bottleneck (`code_dim=64`) — requires tokenizer retrain
- Discrete diffusion on the index grid as an alternative to next-scale AR

---

## Reference numbers

| | tokenizer recon FID | AR-gen FID |
|---|---|---|
| MoScale (published, HRVQVAE codebook) | — | 0.06 |
| Us — `90ds465n` (L=16/H=768, perturb=0.8, FSQ) | 0.0071 | **0.396 best, 0.501 final** |
| Us — prior (L=8/H=384, perturb=0.1, buggy pipeline) | 0.0071 | 1.38 |

Stage-1 recon FID = 0.0071 is the irreducible floor; current AR gap = ~56× above it.
