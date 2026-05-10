"""Minimal next-scale AR transformer for the SkelVQ-FSQ tokenizer.

Reference architecture: Infinity (arxiv 2412.04431). We deliberately keep this
minimal — text conditioning, classifier-free guidance, KV cache, and sampling
top-k/top-p are all OUT of scope for this file. The goal is a clean,
verifiable backbone that demonstrates:

  1. Per-scale residual encoding via SkelVQWrapper (frozen tokenizer).
  2. Block-wise causal AR over the concatenated multi-scale token sequence
     (sum_s L_s = 641 positions for our T=196, scales=[8,4,2,1] config).
  3. FSQ-factorized output: per-position the head emits
     (code_dim * effective_levels) logits, reshaped to (B, L, D, V) and scored
     with per-channel cross-entropy.
  4. Continuous-code input embedding (linear projection of dequantized FSQ
     codes) — matches Infinity's `word_embed = nn.Linear(d_vae, C)` recipe;
     no token-lookup table.

For full MoScale features (BERT-style mask augmentation, learned token map,
text cross-attention with T5, AdaLN with classifier-free guidance, KV cache),
adapt the moscale.py path; this file is the *correctness-first* reference.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_block_causal_mask(scale_lens: List[int], device) -> torch.Tensor:
    """Block-wise causal mask. Within a scale, full attention. Across scales,
    later scales attend to all earlier scales but earlier scales cannot see
    later ones (MoScale / VAR / Infinity convention).

    Returns: bool mask of shape (L, L) where True == "this query position can
    attend to this key position".
    """
    L = sum(scale_lens)
    starts = []
    cum = 0
    for ls in scale_lens:
        starts.append(cum)
        cum += ls
    # scale_id_at_pos[i] = which scale index position i belongs to
    scale_id_at_pos = torch.zeros(L, dtype=torch.long, device=device)
    for si, ls in enumerate(scale_lens):
        scale_id_at_pos[starts[si] : starts[si] + ls] = si
    # Allow attention if key_scale <= query_scale.
    q_scale = scale_id_at_pos.unsqueeze(1)   # (L, 1)
    k_scale = scale_id_at_pos.unsqueeze(0)   # (1, L)
    return k_scale <= q_scale                 # (L, L) bool


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask):
        # nn.MultiheadAttention expects ~"True means do not attend" for `attn_mask`
        # when it is bool. We pass the inverse of our "can attend" mask.
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=~attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SkelVQAR(nn.Module):
    """Next-scale AR over SkelVQ-FSQ codes.

    Args:
      vq_model:        SkelVQWrapper instance (frozen tokenizer).
      latent_dim:      transformer hidden dim (a.k.a. C).
      num_layers:      transformer block depth.
      num_heads:       attention heads per block.
      mlp_ratio:       FFN expansion ratio.
      dropout:         dropout in attention + FFN.
      use_lvl_embed:   if True, add a per-scale learned embedding (Infinity-style).
    """

    def __init__(
        self,
        vq_model,
        latent_dim: int = 384,
        num_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_lvl_embed: bool = True,
    ):
        super().__init__()
        self.vq_model = vq_model
        self.code_dim = vq_model.code_dim                     # 32
        self.effective_levels = vq_model.effective_levels      # 7
        self.scales = list(vq_model.scales)                    # [8, 4, 2, 1]
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Input: linear projection of the continuous dequantized FSQ code.
        # See Infinity infinity.py:228 — `word_embed = nn.Linear(d_vae, C)`.
        self.word_embed = nn.Linear(self.code_dim, latent_dim)

        # Optional per-scale embedding added to every position in that scale.
        self.use_lvl_embed = use_lvl_embed
        if use_lvl_embed:
            self.lvl_embed = nn.Embedding(len(self.scales), latent_dim)

        # Start-of-sequence token (used as the "input" for the first scale,
        # since there's no prior partial reconstruction at scale 0).
        self.sos = nn.Parameter(torch.zeros(1, 1, latent_dim))
        nn.init.trunc_normal_(self.sos, std=0.02)

        self.blocks = nn.ModuleList([
            _Block(latent_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(latent_dim)

        # Per-channel output head: emits (code_dim * effective_levels) logits per
        # position. Reshape to (B, L, code_dim, effective_levels) and CE per channel.
        self.head = nn.Linear(latent_dim, self.code_dim * self.effective_levels)

        self._init_weights()

    def _init_weights(self):
        std = (1.0 / self.latent_dim / 3.0) ** 0.5
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=std)

    @property
    def scale_lens(self) -> List[int]:
        """Per-scale token-position counts at the model's expected input resolution.

        Note: this depends on the bottleneck-space length L of the input, so we
        compute it dynamically per-batch in `forward` — this property exists only
        for documentation purposes assuming default L=343 (T=196 setting).
        """
        L = self.scales[-1] * (343 // max(self.scales))  # placeholder
        return [max(1, L // s) for s in self.scales]

    @staticmethod
    def _per_scale_lens(L: int, scales: List[int]) -> List[int]:
        return [max(1, L // s) for s in scales]

    def _build_lvl_ids(self, scale_lens: List[int], device) -> torch.Tensor:
        """(L_total,) tensor mapping each AR position to its scale id."""
        return torch.cat([
            torch.full((ls,), si, dtype=torch.long, device=device)
            for si, ls in enumerate(scale_lens)
        ], dim=0)

    def _embed_input_sequence(
        self,
        x_BLC_per_scale: List[torch.Tensor],
        scale_lens: List[int],
        device,
    ) -> torch.Tensor:
        """Build the AR input sequence x_BLC of shape (B, L_total, latent_dim).

        Layout (Infinity-style):
          [SOS (first_l = scale_lens[0] copies)] +
          [downsample(cum_after_scale_0, scale_lens[1])] +
          [downsample(cum_after_scale_1, scale_lens[2])] +
          ... +
          [downsample(cum_after_scale_{S-2}, scale_lens[S-1])]

        That is: the *input* at AR positions belonging to scale s is the partial
        reconstruction up through scale s-1, downsampled to scale s's native
        length. For scale 0 (no prior reconstruction) we use the learned SOS
        embedding broadcast over scale_lens[0] positions.

        x_BLC_per_scale[s]: (B, code_dim, scale_lens[s+1]) for s in 0..S-2.
        """
        B = x_BLC_per_scale[0].shape[0]
        first_l = scale_lens[0]
        sos_block = self.sos.expand(B, first_l, -1)   # (B, first_l, latent_dim)

        embedded_chunks = [sos_block]
        for s_input in x_BLC_per_scale:               # (B, code_dim, L_{s+1})
            embedded = self.word_embed(s_input.transpose(1, 2))  # (B, L_{s+1}, latent_dim)
            embedded_chunks.append(embedded)
        x_BLC = torch.cat(embedded_chunks, dim=1)     # (B, L_total, latent_dim)

        if self.use_lvl_embed:
            lvl_ids = self._build_lvl_ids(scale_lens, device)         # (L_total,)
            x_BLC = x_BLC + self.lvl_embed(lvl_ids).unsqueeze(0)      # broadcast over B

        return x_BLC

    def forward(
        self,
        gt_idx_list: List[torch.Tensor],
        x_BLC_per_scale: List[torch.Tensor],
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute AR loss given the LSC outputs.

        Args:
          gt_idx_list:     S-long list of (B, L_s, code_dim) int64. CE targets.
          x_BLC_per_scale: (S-1)-long list of (B, code_dim, L_{s+1}) float.
                           Conditioning input for scale 1..S-1.
                           (Scale 0's input is the learned SOS embedding.)

        Returns:
          loss: scalar mean CE over all (position, channel) pairs.
          info: dict with per-scale-and-channel acc/ce diagnostics.
        """
        device = gt_idx_list[0].device
        scale_lens = [g.shape[1] for g in gt_idx_list]
        L_total = sum(scale_lens)
        B = gt_idx_list[0].shape[0]

        # 1) Embed the AR input sequence.
        x_BLC = self._embed_input_sequence(x_BLC_per_scale, scale_lens, device)  # (B, L_total, latent_dim)
        assert x_BLC.shape == (B, L_total, self.latent_dim), \
            f"x_BLC shape {x_BLC.shape} vs expected ({B}, {L_total}, {self.latent_dim})"

        # 2) Run blocks under block-causal mask.
        attn_mask = _build_block_causal_mask(scale_lens, device)
        for block in self.blocks:
            x_BLC = block(x_BLC, attn_mask)
        x_BLC = self.out_norm(x_BLC)

        # 3) Output head: (B, L_total, code_dim * effective_levels) -> reshape.
        logits_BLDV = self.head(x_BLC).reshape(B, L_total, self.code_dim, self.effective_levels)

        # 4) Compute per-channel CE per scale and aggregate.
        gt_BLD = torch.cat(gt_idx_list, dim=1)  # (B, L_total, code_dim)
        # Permute logits to (B, V, L_total, code_dim) for F.cross_entropy.
        logits_for_ce = logits_BLDV.permute(0, 3, 1, 2).contiguous()  # (B, V, L_total, code_dim)
        loss = F.cross_entropy(logits_for_ce, gt_BLD)

        info = {}
        with torch.no_grad():
            pred = logits_BLDV.argmax(dim=-1)  # (B, L_total, code_dim)
            info["acc_per_channel"] = (pred == gt_BLD).float().mean().item()
            # per-scale accuracy
            offset = 0
            for si, ls in enumerate(scale_lens):
                pred_s = pred[:, offset:offset + ls]
                gt_s = gt_BLD[:, offset:offset + ls]
                info[f"acc_scale_{si}"] = (pred_s == gt_s).float().mean().item()
                offset += ls

        if return_logits:
            return loss, logits_BLDV, info
        return loss, info
