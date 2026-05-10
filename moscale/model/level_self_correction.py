"""LevelSelfCorrection: FSQ analogue of Infinity's BitwiseSelfCorrection.

Reference: Infinity (arxiv 2412.04431),
    github.com/FoundationVision/Infinity/blob/main/infinity/models/bitwise_self_correction.py

What it does (training-time only):
  At each residual scale s = 0, 1, ..., S-1:
    1. Compute the GT FSQ level indices for this scale from the encoder's
       residual.
    2. With probability p (uniform per step in [0, noise_apply_strength]),
       pick a random fraction of cell-channel positions and replace their
       GT indices with a uniform Categorical resample over {0, ..., L-1}.
       (For binary BSQ, Infinity's flip = `1 - bit`. The L-way generalization
       — and the user's chosen variant — is uniform resample, which is the
       max-distance perturbation on the ordered grid.)
    3. If `noise_apply_requant`, re-quantize the residual using the *corrupted*
       indices and pass that forward as conditioning input for the next
       (finer) scale. CE targets always stay the *clean* GT indices.
       This trains the AR transformer to recover from coarse-scale prediction
       errors — fixing the train-test exposure-bias gap.

Defaults match Infinity's `scripts/train.sh`:
  noise_apply_strength = 0.3
  noise_apply_layers   = all S scales
  noise_apply_requant  = 1
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class LevelSelfCorrection:
    """Stateless utility (no learnable parameters; just args + math).

    Args:
      vq_model:               SkelVQWrapper instance (provides .quantizer,
                              .scales, .effective_levels, encoder access).
      noise_apply_layers:     int. Apply noise to the first N scales. -1 = none.
      noise_apply_strength:   float in [0, 1]. Max fraction of positions to perturb
                              per step. Per-step strength uniform in [0, this].
      noise_apply_requant:    bool. If True, the post-noise residual cascade uses
                              the corrupted indices; if False, only the labels
                              fed to the input embedding are corrupted but the
                              residual stays clean (no exposure-bias mitigation
                              effect — useful only as a debug ablation).
    """

    def __init__(
        self,
        vq_model,
        noise_apply_layers: int = -1,
        noise_apply_strength: float = 0.3,
        noise_apply_requant: bool = True,
    ):
        self.vq_model = vq_model
        self.noise_apply_layers = noise_apply_layers
        self.noise_apply_strength = float(noise_apply_strength)
        self.noise_apply_requant = bool(noise_apply_requant)
        self.effective_levels = vq_model.effective_levels
        self.scales = list(vq_model.scales)

    @torch.no_grad()
    def perturb_requant(
        self,
        z_1d: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Run the residual FSQ cascade with optional level-perturbation noise.

        Args:
          z_1d: (B, D, L) — the encoder's pre-quantize 1D feature grid.

        Returns:
          gt_idx_list:  list[(B, L_s, D) int64]. Clean ground-truth indices
                        per scale — used as CE training targets.
          x_BLC_per_scale: list of (B, D, L_{s+1}) tensors representing the
                        cumulative-quantized features after scale s, downsampled
                        to scale s+1's native length (i.e., the conditioning
                        input the AR head sees when predicting scale s+1).
                        Length S-1 (no entry for the final scale, which has no
                        successor). When perturbation is active and
                        noise_apply_requant=True, these tensors reflect the
                        *corrupted* residual cascade.
          f_hat:        (B, D, L) — final cumulative feature (clean if no
                        perturbation, corrupted otherwise).
        """
        quantizer = self.vq_model.quantizer
        B, D, L = z_1d.shape
        assert D == quantizer.code_dim, f"D={D} vs quantizer.code_dim={quantizer.code_dim}"
        scales = self.scales
        S = len(scales)

        residual = z_1d.clone()
        cum = torch.zeros_like(z_1d)
        gt_idx_list: List[torch.Tensor] = []
        x_BLC_per_scale: List[torch.Tensor] = []

        for si, scale in enumerate(scales):
            if scale > 1:
                target_L = max(1, L // scale)
                interp = F.interpolate(residual, size=target_L, mode="area")
            else:
                target_L = L
                interp = residual

            # FSQ with clamp (matches encode_indices).
            int_half = int(quantizer.fsq.fsq.half) if hasattr(quantizer, "fsq") else int(quantizer.half)
            half = float(int_half) + 0.5  # restore half = 3.5 for L=8 setting
            z_T = interp.transpose(1, 2).contiguous()  # (B, L_s, D)
            bounded = half * torch.tanh(z_T)
            rounded = torch.round(bounded).clamp(-int_half, int_half)
            gt_idx = (rounded + int_half).to(dtype=torch.int64)  # (B, L_s, D)
            gt_idx_list.append(gt_idx)

            # Decide whether this scale gets noise.
            apply_noise = (si < self.noise_apply_layers) and self.noise_apply_strength > 0
            if apply_noise:
                # Per-step strength sampled uniformly in [0, noise_apply_strength]
                # (mirrors Infinity's np.random.randint(0, 100*strength+1)*0.01).
                strength = random.uniform(0.0, self.noise_apply_strength)
                if strength > 0:
                    mask = torch.rand_like(gt_idx, dtype=torch.float32) < strength
                    if mask.any():
                        # Uniform Categorical resample over {0..effective_levels-1}.
                        rand_levels = torch.randint(
                            0, self.effective_levels, gt_idx.shape,
                            dtype=torch.int64, device=gt_idx.device,
                        )
                        eff_idx = torch.where(mask, rand_levels, gt_idx)
                    else:
                        eff_idx = gt_idx
                else:
                    eff_idx = gt_idx
            else:
                eff_idx = gt_idx

            # Residual cascade input uses eff_idx (corrupted iff requant + apply_noise).
            if self.noise_apply_requant and apply_noise:
                q_native = quantizer.indices_to_codes(eff_idx).permute(0, 2, 1).contiguous()
            else:
                # Use clean indices for the cascade.
                q_native = quantizer.indices_to_codes(gt_idx).permute(0, 2, 1).contiguous()

            if target_L != L:
                q_up = F.interpolate(q_native, size=L, mode="linear", align_corners=False)
            else:
                q_up = q_native

            residual = residual - q_up
            cum = cum + q_up

            # Conditioning input for scale (si+1): downsample cum to next scale's L.
            if si < S - 1:
                next_L = max(1, L // scales[si + 1])
                if next_L != L:
                    cond = F.interpolate(cum, size=next_L, mode="area")
                else:
                    cond = cum
                x_BLC_per_scale.append(cond)

        return gt_idx_list, x_BLC_per_scale, cum
