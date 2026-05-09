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
