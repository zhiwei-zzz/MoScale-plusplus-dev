"""Binary Spherical Quantization (BSQ) ported from Infinity (arxiv 2412.04431).

Reference:
  github.com/FoundationVision/Infinity/blob/main/infinity/models/bsq_vae/multiscale_bsq.py

Adapted for 1D temporal sequences (shape (B, D, L)) instead of Infinity's
3D image/video case. Drop-in API replacement for MoScale's MSQuantizer.

Key facts:
  * Quantizer is parameter-free: q = sign(z) / sqrt(D), with straight-through
    gradient z + (q - z).detach().
  * Pre-quantize: l2-normalize z to the unit sphere along the channel axis (critical;
    without it sign() saturates and the model collapses).
  * Auxiliary loss is a soft entropy regularizer over per-bit Bernoulli probability,
    computed analytically via sigmoid. Replaces commitment + KL.
  * Multi-scale residual: at each scale, downsample residual along L by an integer
    factor, quantize, upsample back, subtract from residual, accumulate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _entropy(prob, eps: float = 1e-8) -> torch.Tensor:
    return -(prob * prob.clamp(min=eps).log()).sum(dim=-1)


class BSQ(nn.Module):
    """Per-scale binary spherical quantizer.

    Forward:
      z: (B, D, L) channel-first
      returns: (q, loss, bit_balance)
        - q: (B, D, L) quantized features (on unit sphere, scaled by 1/sqrt(D))
        - loss: scalar — entropy regularizer (already weighted by zeta * entropy_weight / inv_t)
        - bit_balance: scalar — average p(+1) across the batch (~0.5 means well-distributed bits)
    """

    def __init__(
        self,
        code_dim: int,
        inv_temperature: float = 100.0,
        zeta: float = 1.0,
        entropy_weight: float = 0.1,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.inv_temperature = inv_temperature
        self.zeta = zeta
        self.entropy_weight = entropy_weight
        self.q_scale = 1.0 / (code_dim ** 0.5)

    def _quantize(self, z: torch.Tensor) -> torch.Tensor:
        # z: (..., D)  unit-sphere normalized
        zhat = torch.where(z > 0, torch.ones_like(z), -torch.ones_like(z))
        zhat = self.q_scale * zhat
        return z + (zhat - z).detach()

    def _soft_entropy_loss(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Infinity's analytical per-bit Bernoulli prob:
        #   p(b_i = -1 | z_i) = sigmoid(-4 * z_i / sqrt(D) * inv_temperature)
        # then per-sample entropy of each bit, summed across bits, averaged across batch.
        p_neg = torch.sigmoid(-4.0 * z * self.inv_temperature / (self.code_dim ** 0.5))
        prob = torch.stack([p_neg, 1.0 - p_neg], dim=-1)  # (..., D, 2)
        per_sample_H = _entropy(prob).sum(dim=-1).mean()  # encourage saturated bits per sample
        # marginal across all leading dims
        avg_prob = prob.reshape(-1, prob.shape[-2], prob.shape[-1]).mean(dim=0)  # (D, 2)
        codebook_H = _entropy(avg_prob).sum()  # encourage uniform per-bit marginal
        return per_sample_H, codebook_H, avg_prob

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, D, L) channel-first
        z = z.transpose(1, 2).contiguous()  # (B, L, D)
        assert z.shape[-1] == self.code_dim, f"expected D={self.code_dim}, got {z.shape[-1]}"
        z = F.normalize(z, dim=-1)  # unit sphere

        q = self._quantize(z)  # (B, L, D)

        per_sample_H, codebook_H, avg_prob = self._soft_entropy_loss(z)
        # entropy_penalty: high per-sample entropy = bits not committed; low codebook
        # entropy = collapsed marginal. Infinity uses (per_sample - codebook) and minimizes
        # it (push per_sample down, push codebook up).
        entropy_penalty = per_sample_H - codebook_H
        loss = self.zeta * entropy_penalty / self.inv_temperature * self.entropy_weight

        # bit_balance diagnostic: ideal value 0.5 (equal +1 / -1 marginal across batch).
        bit_balance = avg_prob[..., 1].mean()

        q = q.transpose(1, 2).contiguous()  # back to (B, D, L)
        return q, loss, bit_balance


class MultiScaleBSQ(nn.Module):
    """Residual BSQ over a 1D temporal axis.

    Mirrors the loop in Infinity's MultiScaleBSQ but for 1D sequences. API is
    drop-in compatible with `MSQuantizer.forward(z, ...)` in MoScale.
    """

    def __init__(
        self,
        code_dim: int,
        scales: list[int] | tuple[int, ...],
        inv_temperature: float = 100.0,
        zeta: float = 1.0,
        entropy_weight: float = 0.1,
        use_decay_factor: bool = False,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.scales = list(scales)
        self.use_decay_factor = use_decay_factor
        self.bsq = BSQ(
            code_dim=code_dim,
            inv_temperature=inv_temperature,
            zeta=zeta,
            entropy_weight=entropy_weight,
        )

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 0.0,        # unused; kept for API parity with MSQuantizer
        m_lens=None,                     # unused; fixed-length training
        start_drop: int = -1,
        quantize_dropout_prob: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, D, L)
        del temperature, m_lens, start_drop, quantize_dropout_prob  # quiet unused warnings
        B, D, L = x.shape
        assert D == self.code_dim, f"expected D={self.code_dim}, got {D}"

        residual = x
        quantized_out = torch.zeros_like(x)
        losses: list[torch.Tensor] = []
        bit_balances: list[torch.Tensor] = []
        out_fact = 1.0

        for si, scale in enumerate(self.scales):
            if scale > 1:
                target_L = max(1, L // scale)
                interp = F.interpolate(residual, size=target_L, mode="area")
            else:
                target_L = L
                interp = residual

            q, loss, bit_balance = self.bsq(interp)
            if self.use_decay_factor:
                q = q * max(0.1, out_fact)
                out_fact -= 0.1

            if target_L != L:
                q_up = F.interpolate(q, size=L, mode="linear", align_corners=False)
            else:
                q_up = q

            residual = residual - q_up.detach()
            quantized_out = quantized_out + q_up

            losses.append(loss)
            bit_balances.append(bit_balance)

        total_loss = torch.stack(losses).sum()
        avg_bit_balance = torch.stack(bit_balances).mean()
        return quantized_out, total_loss, avg_bit_balance


class FSQ(nn.Module):
    """Per-scale Finite Scalar Quantization (Mentzer et al., ICLR 2024).

    Forward:
      z: (B, D, L) channel-first
      returns: (q, loss, diag)
        - q: (B, D, L) quantized features in [-1, 1] (one of L levels per channel)
        - loss: scalar — entropy regularizer (zero by default; opt-in via entropy_weight>0)
        - diag: average abs(q) across batch — codes-spread diagnostic
                (approaches 1 - 1/half as channels saturate the outer levels)

    Math, per-channel:
        bounded = ((L-1)/2) * tanh(z)         # squash to [-half, +half]
        rounded = round(bounded)              # snap to integer grid
        q = rounded + (bounded - bounded.detach())  # straight-through
        q /= half                             # rescale to [-1, +1]

    Original FSQ (Mentzer 2024) has no auxiliary loss. We optionally add an
    LFQ/BSQ-style entropy regularizer (generalized to L>2 via Categorical over the
    grid levels) — set entropy_weight>0 to enable. The regularizer encourages:
      * low per-sample entropy (each cell-channel commits to a single level)
      * high batch-marginal entropy (all L levels are used roughly uniformly across
        the dataset, fixing the under-utilization that plain FSQ exhibits)
    """

    def __init__(
        self,
        code_dim: int,
        levels: int = 8,
        inv_temperature: float = 20.0,
        entropy_weight: float = 0.0,
        zeta: float = 1.0,
    ):
        super().__init__()
        assert levels >= 2, f"FSQ needs >=2 levels, got {levels}"
        self.code_dim = code_dim
        self.levels = levels
        self.half = (levels - 1) / 2.0  # e.g. L=8 -> half=3.5
        self.inv_temperature = inv_temperature
        self.entropy_weight = entropy_weight
        self.zeta = zeta

        # Pre-build the grid level positions in the rescaled [-1, +1] space.
        # When half=3.5, integer levels are {-3,...,3}; rescaled they are {-3,...,3}/3.5.
        # (We use this for the soft-assignment regularizer only.)
        int_levels = torch.arange(-int(self.half), int(self.half) + 1, dtype=torch.float32)
        # If levels is even (L=8), int_levels has L-1 entries. The remaining odd-half
        # value (-3.5 / +3.5) is asymptotically reached by tanh but never exactly hit
        # by round(), so the effective grid is the integer set above.
        rescaled_levels = int_levels / self.half  # (L_eff,) values in [-1, +1]
        self.register_buffer("grid_levels", rescaled_levels, persistent=False)

    def _entropy_regularizer(self, q: torch.Tensor) -> torch.Tensor:
        """Soft-assignment entropy regularizer over per-channel grid levels.
        q: (..., D) values in [-1, +1] after rescaling.
        Returns scalar loss in the same sign convention as Infinity's BSQ:
          (per_sample_H - codebook_H) — minimize this.
        """
        # Distance from each cell-channel's q to each grid level: (..., D, L_eff)
        diff = q.unsqueeze(-1) - self.grid_levels  # (..., D, L_eff)
        # Soft assignment: -dist² scaled by inv_temperature, then softmax.
        soft_logits = -(diff ** 2) * self.inv_temperature  # (..., D, L_eff)
        soft_prob = F.softmax(soft_logits, dim=-1)  # (..., D, L_eff)

        # Per-sample entropy: -Σ p log p, summed over levels, mean over (sample, channel).
        per_sample_H = -(soft_prob * soft_prob.clamp(min=1e-8).log()).sum(dim=-1).mean()

        # Batch-marginal entropy: avg soft_prob across all leading dims, then entropy
        # summed over channels.
        avg_prob = soft_prob.reshape(-1, self.code_dim, self.grid_levels.shape[0]).mean(dim=0)
        # avg_prob: (D, L_eff)
        codebook_H = -(avg_prob * avg_prob.clamp(min=1e-8).log()).sum(dim=-1).sum()

        return per_sample_H - codebook_H  # minimize

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, D, L) channel-first
        z = z.transpose(1, 2).contiguous()  # (B, L, D)
        assert z.shape[-1] == self.code_dim, f"expected D={self.code_dim}, got {z.shape[-1]}"

        bounded = self.half * torch.tanh(z)             # [-half, +half], differentiable
        rounded = torch.round(bounded)                  # snap to grid
        q = rounded + (bounded - bounded.detach())       # straight-through gradient
        q = q / self.half                                # rescale to [-1, +1]

        if self.entropy_weight > 0:
            # Use the *continuous* (post-tanh, pre-round, but rescaled) value for soft
            # assignment, so gradients flow through the encoder.
            cont_q = bounded / self.half  # (B, L, D), in [-1, +1]
            entropy_penalty = self._entropy_regularizer(cont_q)
            loss = self.zeta * entropy_penalty / self.inv_temperature * self.entropy_weight
        else:
            loss = torch.zeros((), device=z.device, dtype=z.dtype)

        # Diagnostic: how saturated are the codes across this batch.
        diag = q.abs().mean()

        q = q.transpose(1, 2).contiguous()  # back to (B, D, L)
        return q, loss, diag


class MultiScaleFSQ(nn.Module):
    """Residual FSQ over a 1D temporal axis. Drop-in for MultiScaleBSQ.

    Same residual loop as MultiScaleBSQ, only the per-scale quantizer changes:
      r_s = downsample(residual, L/scale)
      q_s = FSQ(r_s)
      residual <- residual - upsample(q_s).detach()
      quantized_out += upsample(q_s)

    Effective level count for stage-2 AR (encode_indices / indices_to_codes):
      For L=8 setting, half=(L-1)/2=3.5; clamped reachable rounded ∈ {-3..3}, so
      effective_levels = 7. (The banker's-rounding extreme ±4 is reachable only
      when tanh saturates to exactly 1.0 in float — rare for in-distribution
      encoder outputs and discarded by the clamp for a clean vocab.)
    """

    def __init__(
        self,
        code_dim: int,
        scales: list[int] | tuple[int, ...],
        levels: int = 8,
        use_decay_factor: bool = False,
        inv_temperature: float = 20.0,
        entropy_weight: float = 0.0,
        zeta: float = 1.0,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.scales = list(scales)
        self.levels = levels
        self.use_decay_factor = use_decay_factor
        self.fsq = FSQ(
            code_dim=code_dim,
            levels=levels,
            inv_temperature=inv_temperature,
            entropy_weight=entropy_weight,
            zeta=zeta,
        )

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 0.0,        # unused; API parity with MSQuantizer/MultiScaleBSQ
        m_lens=None,                     # unused; fixed-length training
        start_drop: int = -1,
        quantize_dropout_prob: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del temperature, m_lens, start_drop, quantize_dropout_prob
        B, D, L = x.shape
        assert D == self.code_dim, f"expected D={self.code_dim}, got {D}"

        residual = x
        quantized_out = torch.zeros_like(x)
        losses: list[torch.Tensor] = []
        diags: list[torch.Tensor] = []
        out_fact = 1.0

        for si, scale in enumerate(self.scales):
            if scale > 1:
                target_L = max(1, L // scale)
                interp = F.interpolate(residual, size=target_L, mode="area")
            else:
                target_L = L
                interp = residual

            q, loss, diag = self.fsq(interp)
            if self.use_decay_factor:
                q = q * max(0.1, out_fact)
                out_fact -= 0.1

            if target_L != L:
                q_up = F.interpolate(q, size=L, mode="linear", align_corners=False)
            else:
                q_up = q

            residual = residual - q_up.detach()
            quantized_out = quantized_out + q_up

            losses.append(loss)
            diags.append(diag)

        total_loss = torch.stack(losses).sum()
        avg_diag = torch.stack(diags).mean()
        return quantized_out, total_loss, avg_diag

    @property
    def effective_levels(self) -> int:
        """Number of distinct discrete level indices per channel.

        For L=8 setting (half=3.5), this is 7 — the inner range {-3..3} reachable
        without relying on banker's-rounding edge cases.
        """
        return 2 * int(self.fsq.half) + 1

    def indices_to_codes(self, idx: torch.Tensor) -> torch.Tensor:
        """Inverse of the integer-index emission in `encode_indices`.

        idx: integer tensor with values in [0, effective_levels).
        Returns: float tensor of the same shape, with values in [-1, +1] on the
        FSQ grid (rounded / half).
        """
        int_half = int(self.fsq.half)
        rounded = idx.to(dtype=torch.float32) - int_half
        return rounded / self.fsq.half

    @torch.no_grad()
    def encode_indices(
        self,
        x: torch.Tensor,
        perturb_rate=None,
        train: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Run the residual FSQ cascade and return per-scale integer indices.

        Args:
          x: (B, D, L) — pre-quantize 1D features.
          perturb_rate: optional [lo, hi] pair (or scalar). When non-zero AND
              `train=True`, mimic MoScale's MSQuantizer perturb / Infinity's
              Bitwise Self-Correction: at each scale, sample a per-step rate
              uniformly in [lo, hi], then with that probability replace each
              (cell, channel) GT level index with a uniform Categorical
              resample over {0..effective_levels-1} **excluding the true
              level**. The cascade then re-quantizes the residual with the
              *perturbed* indices so subsequent scales see the corruption
              (matches Infinity's `noise_apply_requant=1` default).
          train: must be True for perturbation to fire.

        Returns three same-length lists, one entry per residual scale:
          idx_per_scale[s]: (B, L_s, D) int64 — GT level indices computed
              from the residual that the encoder *would* have seen entering
              scale s. When perturbation is active, the residual at s>=1
              already incorporates the corruption from earlier scales, so
              `idx_per_scale[s]` is "the optimal code given the imperfect
              partial reconstruction so far" — exactly the AR's CE target.
              Matches Infinity's BSC semantics: at scale 0 these are equal
              to the clean encoder output; at later scales they differ
              proportionally to the accumulated corruption.
          q_per_scale[s]:   (B, D, L_s) float — dequantized at native scale L_s.
              When perturbation is active, this reflects the *perturbed* code
              (so it lines up with the corrupted residual cascade fed forward
              through `q_cum`).
          q_cum[s]:         (B, D, L)  float — cumulative dequantized after
              scale s, upsampled back to full L. Reflects the perturbed
              cascade when perturbation is active.

        Where L_s = max(1, L // scale_s) for scale_s in self.scales.
        """
        B, D, L = x.shape
        assert D == self.code_dim, f"expected D={self.code_dim}, got {D}"
        int_half = int(self.fsq.half)
        half = self.fsq.half
        eff_lvl = 2 * int_half + 1

        # Resolve perturb rate — mirror MSQuantizer's [lo, hi] pair convention.
        if perturb_rate is None:
            lo = hi = 0.0
        elif isinstance(perturb_rate, (int, float)):
            lo, hi = 0.0, float(perturb_rate)
        else:
            lo, hi = float(perturb_rate[0]), float(perturb_rate[1])
        do_perturb = train and hi > 0.0

        residual = x
        cum = torch.zeros_like(x)
        idx_per_scale: list[torch.Tensor] = []
        q_per_scale: list[torch.Tensor] = []
        q_cum: list[torch.Tensor] = []

        for scale in self.scales:
            if scale > 1:
                target_L = max(1, L // scale)
                interp = F.interpolate(residual, size=target_L, mode="area")
            else:
                target_L = L
                interp = residual

            # Per-channel FSQ at this scale (clean GT indices).
            z_T = interp.transpose(1, 2).contiguous()                    # (B, L_s, D)
            bounded = half * torch.tanh(z_T)
            rounded = torch.round(bounded).clamp(-int_half, int_half)    # (B, L_s, D)
            idx_gt = (rounded + int_half).to(dtype=torch.int64)           # (B, L_s, D), ∈ [0, eff_lvl)
            idx_per_scale.append(idx_gt)

            if do_perturb:
                # Per-step strength uniform in [lo, hi] — matches Infinity's
                # `np.random.randint(0, 100*strength+1) * 0.01` discretized
                # uniform sampling and MoScale's `random.uniform(lo, hi)`.
                import random as _random
                rate = _random.uniform(lo, hi)
                if rate > 0:
                    mask = torch.rand_like(idx_gt, dtype=torch.float32) < rate
                    if mask.any():
                        # Resample uniformly over {0..eff_lvl-2}, then bump up
                        # by 1 wherever the resample matches the true level —
                        # this gives a uniform Categorical over the (eff_lvl-1)
                        # *non-true* levels, matching MSQuantizer's
                        # `r + (r >= true)` trick.
                        flat_true = idx_gt[mask]
                        r = torch.randint(0, eff_lvl - 1, (flat_true.numel(),),
                                          dtype=torch.int64, device=idx_gt.device)
                        r = r + (r >= flat_true).to(torch.int64)
                        idx_eff = idx_gt.clone()
                        idx_eff[mask] = r
                    else:
                        idx_eff = idx_gt
                else:
                    idx_eff = idx_gt
            else:
                idx_eff = idx_gt

            # Convert (perturbed if active, else clean) indices to continuous codes.
            q_native = self.indices_to_codes(idx_eff).permute(0, 2, 1).contiguous()  # (B, D, L_s)
            q_per_scale.append(q_native)

            if target_L != L:
                q_up = F.interpolate(q_native, size=L, mode="linear", align_corners=False)
            else:
                q_up = q_native

            residual = residual - q_up
            cum = cum + q_up
            q_cum.append(cum.clone())

        return idx_per_scale, q_per_scale, q_cum


if __name__ == "__main__":
    # Smoke tests for both BSQ and FSQ
    torch.manual_seed(0)
    B, D, L = 4, 32, 112

    print("=" * 60)
    print("BSQ smoke test")
    print("=" * 60)
    x = torch.randn(B, D, L, requires_grad=True, device="cuda")
    m = MultiScaleBSQ(code_dim=D, scales=[8, 4, 2, 1]).cuda()
    q, loss, bal = m(x)
    print(f"in:  {tuple(x.shape)}")
    print(f"q:   {tuple(q.shape)}")
    print(f"loss: {loss.item():.4f}")
    print(f"bit balance (~0.5 ideal): {bal.item():.4f}")
    print(f"q range: [{q.min().item():.4f}, {q.max().item():.4f}]")
    target = torch.randn_like(q)
    rec_loss = ((q - target) ** 2).mean()
    (rec_loss + loss).backward()
    print("[OK] BSQ backward ran")

    print()
    print("=" * 60)
    print("FSQ L=8 smoke test")
    print("=" * 60)
    x = torch.randn(B, D, L, requires_grad=True, device="cuda")
    m = MultiScaleFSQ(code_dim=D, scales=[8, 4, 2, 1], levels=8).cuda()
    q, loss, diag = m(x)
    print(f"in:  {tuple(x.shape)}")
    print(f"q:   {tuple(q.shape)}")
    print(f"loss: {loss.item():.4f}  (FSQ has no aux loss)")
    print(f"abs(q) avg diag (~0.5 means evenly spread across grid): {diag.item():.4f}")
    print(f"q range: [{q.min().item():.4f}, {q.max().item():.4f}]")
    # Verify each cell of q at scale=1 lies on the L=8 grid: {-1, -5/7, -3/7, -1/7, 1/7, 3/7, 5/7, 1}
    target = torch.randn_like(q)
    rec_loss = ((q - target) ** 2).mean()
    (rec_loss + loss).backward()
    print("[OK] FSQ backward ran")
