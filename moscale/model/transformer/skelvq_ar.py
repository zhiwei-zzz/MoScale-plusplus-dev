"""Next-scale AR transformer for the SkelVQ-FSQ tokenizer, with optional T5
text cross-attention + classifier-free guidance + scale-by-scale sampling.

Reference architecture: Infinity (arxiv 2412.04431). Adapted from binary BSQ
to L-way (here L=7 effective) FSQ codes per channel. The same network can be
trained:

  * unconditionally:  ar.forward(gt_idx_list, x_BLC_per_scale)
  * text-conditionally: ar.forward(gt_idx_list, x_BLC_per_scale, text_strs=...,
                                   cond_drop_prob=...)

and sampled at inference via:

  * ar.generate(text_strs, cfg_scale=..., temperature=..., top_p=...)

Design choices:

  * `word_embed = nn.Linear(d_vae, C)` over the *continuous dequantized FSQ
    code* — no token-lookup table (matches Infinity infinity.py:228).
  * `head = nn.Linear(C, d_vae * effective_levels)`, reshape (B, L, D, V),
    score with per-channel cross-entropy (class-dim=V). For our ckpt
    V=effective_levels=7, D=code_dim=32 -> 224 logits/position.
  * Block-causal attention: within a scale full attention; across scales,
    query at scale i sees keys at scale j <= i.
  * Cross-attention to text: every block has a CrossAttn sub-layer
    (Self-Attn → CrossAttn → MLP) when text is enabled. Text embeddings are
    T5 (`google/t5-v1_1-base`, 768-dim, frozen) projected through a 2-layer
    MLP to the AR's latent_dim. CFG: with prob `cond_drop_prob` during
    training, replace the projected text with a learned `cfg_uncond`
    parameter; at inference, pass cond + uncond as a duplicated batch and
    blend logits.

What's still out of scope here (handle later, see STAGE2_STATUS.md):
  * KV cache for inference (we recompute per-scale; fine for short
    sequences).
  * AdaLN conditioner (we use plain LayerNorm).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------- #
#  Helpers
# ---------------------------------------------------------------------------- #


def _build_block_causal_mask(scale_lens: List[int], device) -> torch.Tensor:
    """Block-wise causal mask (True == may attend).

    Within a scale, full attention. Across scales, later scales attend to all
    earlier scales but earlier scales cannot see later ones.
    """
    L = sum(scale_lens)
    starts = []
    cum = 0
    for ls in scale_lens:
        starts.append(cum)
        cum += ls
    scale_id_at_pos = torch.zeros(L, dtype=torch.long, device=device)
    for si, ls in enumerate(scale_lens):
        scale_id_at_pos[starts[si] : starts[si] + ls] = si
    q_scale = scale_id_at_pos.unsqueeze(1)
    k_scale = scale_id_at_pos.unsqueeze(0)
    return k_scale <= q_scale


# ---------------------------------------------------------------------------- #
#  Text encoding
# ---------------------------------------------------------------------------- #


class _TextProjector(nn.Module):
    """Wraps a frozen T5 encoder + a 2-layer MLP that projects from T5's
    hidden dim (768 for t5-v1_1-base) to the AR's latent_dim.

    Forward returns (proj, mask) where proj is (B, T5_max_len, latent_dim) and
    mask is (B, T5_max_len) with True at valid positions.
    """

    def __init__(
        self,
        latent_dim: int,
        device,
        t5_model: str = "google/t5-v1_1-base",
        t5_max_len: int = 20,
    ):
        super().__init__()
        from model.encode_text import T5TextEncoder  # type: ignore
        self.t5 = T5TextEncoder(device=device, local_files_only=False,
                                from_pretrained=t5_model, model_max_length=t5_max_len)
        text_dim = self.t5.model.config.d_model
        self.text_max_len = t5_max_len
        self.text_dim = text_dim
        self.proj = nn.Sequential(
            nn.Linear(text_dim, latent_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(latent_dim, latent_dim),
        )

    @torch.no_grad()
    def encode_text(self, texts: List[str]):
        emb, mask = self.t5.get_text_embeddings(texts)
        return emb, mask  # (B, t5_max_len, text_dim), (B, t5_max_len)

    def forward(self, texts: List[str]):
        emb, mask = self.encode_text(texts)
        return self.proj(emb), mask  # (B, t5_max_len, latent_dim), (B, t5_max_len)


# ---------------------------------------------------------------------------- #
#  Transformer block with optional text cross-attention
# ---------------------------------------------------------------------------- #


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float,
                 use_crossattn: bool = False):
        super().__init__()
        self.use_crossattn = use_crossattn
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        if use_crossattn:
            self.norm_xa = nn.LayerNorm(dim)
            self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask, kv=None, kv_pad_mask=None):
        # Self-attention; nn.MultiheadAttention treats `attn_mask=True` as "do
        # not attend", which is the inverse of our "may attend" convention.
        x_norm = self.norm1(x)
        sa, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=~attn_mask, need_weights=False)
        x = x + sa
        if self.use_crossattn and kv is not None:
            x_norm = self.norm_xa(x)
            ca, _ = self.cross_attn(x_norm, kv, kv,
                                    key_padding_mask=kv_pad_mask, need_weights=False)
            x = x + ca
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------- #
#  AR model
# ---------------------------------------------------------------------------- #


class SkelVQAR(nn.Module):
    def __init__(
        self,
        vq_model,
        latent_dim: int = 384,
        num_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_lvl_embed: bool = True,
        text_cond: bool = False,
        t5_model: str = "google/t5-v1_1-base",
        t5_max_len: int = 20,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.vq_model = vq_model
        self.code_dim = vq_model.code_dim
        self.effective_levels = vq_model.effective_levels
        self.scales = list(vq_model.scales)
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.text_cond = text_cond

        # Input projection: continuous dequantized FSQ code -> AR hidden dim.
        self.word_embed = nn.Linear(self.code_dim, latent_dim)

        # Per-scale embedding.
        self.use_lvl_embed = use_lvl_embed
        if use_lvl_embed:
            self.lvl_embed = nn.Embedding(len(self.scales), latent_dim)

        # Start-of-sequence embedding for scale 0 input positions.
        self.sos = nn.Parameter(torch.zeros(1, 1, latent_dim))
        nn.init.trunc_normal_(self.sos, std=0.02)

        # Text conditioning components.
        if text_cond:
            assert device is not None, "text_cond=True requires `device` for T5 placement"
            self.text_proj = _TextProjector(latent_dim, device, t5_model, t5_max_len)
            self.cfg_uncond = nn.Parameter(torch.empty(1, t5_max_len, latent_dim))
            nn.init.trunc_normal_(self.cfg_uncond.data, std=0.02)
            self.text_max_len = t5_max_len
        else:
            self.text_proj = None
            self.cfg_uncond = None
            self.text_max_len = 0

        self.blocks = nn.ModuleList([
            _Block(latent_dim, num_heads, mlp_ratio, dropout, use_crossattn=text_cond)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(latent_dim)
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

    # ------------------------------------------------------------------ #
    #  Sequence assembly
    # ------------------------------------------------------------------ #

    @staticmethod
    def _per_scale_lens(L: int, scales: List[int]) -> List[int]:
        return [max(1, L // s) for s in scales]

    def _build_lvl_ids(self, scale_lens: List[int], device) -> torch.Tensor:
        return torch.cat([
            torch.full((ls,), si, dtype=torch.long, device=device)
            for si, ls in enumerate(scale_lens)
        ], dim=0)

    def _embed_input_sequence(
        self,
        x_BLC_per_scale: List[torch.Tensor],
        scale_lens: List[int],
        device,
        B: Optional[int] = None,
    ) -> torch.Tensor:
        # Need batch size from somewhere — prefer the first scale-input tensor
        # if any, otherwise the explicit B (used at inference scale 0).
        if B is None:
            assert x_BLC_per_scale, "either provide B or at least one x_BLC_per_scale tensor"
            B = x_BLC_per_scale[0].shape[0]
        first_l = scale_lens[0]
        sos_block = self.sos.expand(B, first_l, -1)

        embedded_chunks = [sos_block]
        for s_input in x_BLC_per_scale:
            embedded_chunks.append(self.word_embed(s_input.transpose(1, 2)))
        x_BLC = torch.cat(embedded_chunks, dim=1)

        if self.use_lvl_embed:
            lvl_ids = self._build_lvl_ids(scale_lens, device)
            x_BLC = x_BLC + self.lvl_embed(lvl_ids).unsqueeze(0)
        return x_BLC

    # ------------------------------------------------------------------ #
    #  Text conditioning
    # ------------------------------------------------------------------ #

    def _encode_text(
        self,
        texts: Optional[List[str]],
        cond_drop_prob: float,
        B: int,
        device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns (kv, kv_pad_mask). kv is (B, t5_max_len, latent_dim) or None."""
        if not self.text_cond:
            return None, None
        if texts is None:
            kv = self.cfg_uncond.expand(B, -1, -1)
            kv_pad_mask = torch.zeros(B, self.text_max_len, dtype=torch.bool, device=device)
            return kv, kv_pad_mask

        proj, mask = self.text_proj(texts)        # (B, t5_max_len, latent_dim), (B, t5_max_len)
        proj = proj.to(device)
        mask = mask.to(device)
        kv_pad_mask = ~mask.bool()                 # nn.MultiheadAttention: True = pad

        if cond_drop_prob > 0 and self.training:
            drop = (torch.rand(B, device=device) < cond_drop_prob)
            if drop.any():
                proj = torch.where(
                    drop.view(B, 1, 1),
                    self.cfg_uncond.expand(B, -1, -1),
                    proj,
                )
                # Dropped samples: full attention to the cfg_uncond embedding.
                kv_pad_mask = torch.where(
                    drop.view(B, 1),
                    torch.zeros_like(kv_pad_mask),
                    kv_pad_mask,
                )

        return proj, kv_pad_mask

    # ------------------------------------------------------------------ #
    #  Forward (training / loss)
    # ------------------------------------------------------------------ #

    def forward(
        self,
        gt_idx_list: List[torch.Tensor],
        x_BLC_per_scale: List[torch.Tensor],
        texts: Optional[List[str]] = None,
        cond_drop_prob: float = 0.0,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        device = gt_idx_list[0].device
        scale_lens = [g.shape[1] for g in gt_idx_list]
        L_total = sum(scale_lens)
        B = gt_idx_list[0].shape[0]

        # Input sequence.
        x_BLC = self._embed_input_sequence(x_BLC_per_scale, scale_lens, device)

        # Text encoding (optional).
        kv, kv_pad_mask = self._encode_text(texts, cond_drop_prob, B, device)

        # Block-causal mask.
        attn_mask = _build_block_causal_mask(scale_lens, device)

        # Run blocks.
        for block in self.blocks:
            x_BLC = block(x_BLC, attn_mask, kv=kv, kv_pad_mask=kv_pad_mask)
        x_BLC = self.out_norm(x_BLC)

        logits_BLDV = self.head(x_BLC).reshape(B, L_total, self.code_dim, self.effective_levels)
        gt_BLD = torch.cat(gt_idx_list, dim=1)
        loss = F.cross_entropy(
            logits_BLDV.permute(0, 3, 1, 2).contiguous(),
            gt_BLD,
        )

        info = {}
        with torch.no_grad():
            pred = logits_BLDV.argmax(dim=-1)
            info["acc_per_channel"] = (pred == gt_BLD).float().mean().item()
            offset = 0
            for si, ls in enumerate(scale_lens):
                info[f"acc_scale_{si}"] = (pred[:, offset:offset + ls] == gt_BLD[:, offset:offset + ls]).float().mean().item()
                offset += ls

        if return_logits:
            return loss, logits_BLDV, info
        return loss, info

    # ------------------------------------------------------------------ #
    #  Inference (scale-by-scale sampling with CFG)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        texts: List[str],
        L_target: int,
        cfg_scale: float = 4.0,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        return_indices: bool = False,
    ):
        """Generate motion conditioned on `texts`.

        Args:
            texts: list of B caption strings.
            L_target: length of the bottleneck-space cell axis (e.g., for
                T=64 windows, L_target = T_b * J_b = 16 * 7 = 112; for T=196,
                L_target = 49 * 7 = 343).
            cfg_scale: classifier-free guidance scale. Set to 0 for unconditional.
            temperature, top_k, top_p: sampling controls applied per channel.

        Returns: motion of shape (B, T, pose_dim) where T = (L_target/J_b) * 4.
                 Plus per-scale int idx list if return_indices=True.
        """
        assert self.text_cond, "generate() requires the AR to be built with text_cond=True"
        device = next(self.parameters()).device
        self.train(False)
        B = len(texts)
        D = self.code_dim
        scales = self.scales
        scale_lens = self._per_scale_lens(L_target, scales)
        S = len(scales)

        # Encode text once for cond + duplicated for uncond pass.
        if cfg_scale > 0:
            proj_cond, mask_cond = self.text_proj(texts)
            kv_cond = proj_cond.to(device)
            kv_pad_cond = ~mask_cond.bool().to(device)
            kv_uncond = self.cfg_uncond.expand(B, -1, -1)
            kv_pad_uncond = torch.zeros(B, self.text_max_len, dtype=torch.bool, device=device)
            kv = torch.cat([kv_cond, kv_uncond], dim=0)               # (2B, t5_max_len, C)
            kv_pad = torch.cat([kv_pad_cond, kv_pad_uncond], dim=0)
            doubled = True
        else:
            proj, mask = self.text_proj(texts)
            kv = proj.to(device)
            kv_pad = ~mask.bool().to(device)
            doubled = False

        # State accumulated across scales.
        cum = torch.zeros(B, D, L_target, device=device)
        all_idx: List[torch.Tensor] = []

        # x_BLC_per_scale[s] is the input for scale s+1; we build it scale by scale.
        # Sequence layout: [SOS x first_l] + [embedded(x_BLC_per_scale[0])] + ... + [embedded(...[S-2])]
        # At inference, we run the model up to the *current* scale and read out its logits.
        for si in range(S):
            current_lens = scale_lens[: si + 1]
            L_so_far = sum(current_lens)

            # Build conditioning input tensors for scales 1..si (cumulative cum
            # downsampled to the *next* scale's length).
            x_inputs: List[torch.Tensor] = []
            for prev_si in range(si):
                next_size = scale_lens[prev_si + 1]
                if next_size != L_target:
                    x_in = F.interpolate(cum, size=next_size, mode="area")
                else:
                    x_in = cum
                x_inputs.append(x_in)

            # Embed input sequence so far. At scale 0 there are no x_inputs,
            # so we pass B explicitly.
            x_BLC = self._embed_input_sequence(x_inputs, current_lens, device, B=B)
            B_eff = x_BLC.shape[0]
            if doubled:
                x_BLC = x_BLC.repeat(2, 1, 1)

            # Block-causal mask over current sequence length.
            attn_mask = _build_block_causal_mask(current_lens, device)

            # Run blocks.
            for block in self.blocks:
                x_BLC = block(x_BLC, attn_mask, kv=kv, kv_pad_mask=kv_pad)
            x_BLC = self.out_norm(x_BLC)

            logits = self.head(x_BLC).reshape(x_BLC.shape[0], L_so_far, D, self.effective_levels)

            # CFG blend on the current scale's positions only.
            offset = sum(scale_lens[:si])
            cur_logits = logits[:, offset:offset + scale_lens[si]]    # (B*[2], L_s, D, V)

            if doubled:
                cond_log, uncond_log = cur_logits[:B], cur_logits[B:]
                cur_logits = uncond_log + cfg_scale * (cond_log - uncond_log)

            cur_logits = cur_logits / max(temperature, 1e-8)

            # Per-channel sampling: flatten (L_s, D) into batch dim.
            B_, L_s, D_, V = cur_logits.shape
            flat = cur_logits.reshape(B_ * L_s * D_, V)
            sampled = _sample_top_k_p(flat, top_k=top_k, top_p=top_p)   # (B_*L_s*D,)
            sampled = sampled.view(B_, L_s, D_)

            all_idx.append(sampled)

            # Update cumulative residual: dequantize + upsample + accumulate.
            q_native = self.vq_model.quantizer.indices_to_codes(sampled).permute(0, 2, 1).contiguous()  # (B, D, L_s)
            if q_native.shape[-1] != L_target:
                q_up = F.interpolate(q_native, size=L_target, mode="linear", align_corners=False)
            else:
                q_up = q_native
            cum = cum + q_up

        # Final: decode to motion via the wrapper's decode (uses SALAD's decoder).
        motion = self.vq_model.decode(cum)
        if return_indices:
            return motion, all_idx
        return motion


# ---------------------------------------------------------------------------- #
#  Sampling utility
# ---------------------------------------------------------------------------- #


def _sample_top_k_p(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    """Categorical sampling with optional top-k / top-p filtering.

    logits: (..., V) — flattened batch.
    Returns: (...,) integer indices.
    """
    V = logits.shape[-1]
    if top_k and 0 < top_k < V:
        topk = logits.topk(top_k, dim=-1)
        kept = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, topk.indices, True)
        logits = logits.masked_fill(~kept, float("-inf"))
    if top_p and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cum = sorted_probs.cumsum(dim=-1)
        sorted_remove = cum > top_p
        sorted_remove[..., 0] = False
        remove = torch.zeros_like(sorted_remove).scatter_(-1, sorted_idx, sorted_remove)
        logits = logits.masked_fill(remove, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    flat = probs.reshape(-1, V)
    sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return sampled.view(*logits.shape[:-1])
