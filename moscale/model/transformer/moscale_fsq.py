import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encode_text import T5TextEncoder
from functools import partial
from model.transformer.tools import *
from typing import Optional, Tuple
import math, random


from model.transformer.transformer_helper import AdaLNBeforeHead, AdaLNSelfAttn, FastRMSNorm, MultiQueryTextPool


def sample_with_top_k_top_p_(logits_BlV: torch.Tensor,
                            top_k: int = 0,
                            top_p: float = 0.0,
                            rng=None,
                            num_samples: int = 1,
                            temperature: float = 1.0) -> torch.Tensor:
    """
    Args:
        logits_BlV: [B, L, V] unnormalized logits
        top_k:     keep at most top_k tokens (0 = disabled)
        top_p:     nucleus threshold in (0,1]; keeps smallest set whose prob mass >= top_p (0/1 = disabled)
        rng:       torch.Generator for reproducible sampling
        num_samples: number of samples per position (>=1)
        temperature: >0. Scale logits before filtering & sampling

    Returns:
        indices [B, L, num_samples]
    """
    assert logits_BlV.dim() == 3, "Expected [B, L, V] logits"
    B, L, V = logits_BlV.shape
    assert num_samples >= 1, "num_samples must be >= 1"
    # Temperature scaling
    t = max(float(temperature), 1e-8)
    logits = logits_BlV / t

    # Start from: keep everything that isn't -inf
    keep = torch.isfinite(logits)

    # --- Top-k (cap strictly to k indices; avoids tie explosion) ---
    if top_k and 0 < top_k < V:
        topk_idx = logits.topk(top_k, dim=-1).indices                       # [B,L,top_k]
        keep_k = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, topk_idx, True)
        keep = keep & keep_k

    # --- Top-p (nucleus) ---
    if top_p and 0.0 < top_p < 1.0:
        # Apply softmax only over currently-kept tokens
        masked_logits = logits.masked_fill(~keep, float('-inf'))            # [B,L,V]

        # Sort descending for standard nucleus filtering
        sorted_logits, sorted_idx = torch.sort(masked_logits, dim=-1, descending=True)  # [B,L,V]
        sorted_probs = torch.softmax(sorted_logits, dim=-1)                 # NaN-safe below
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumprob > top_p; keep at least one token
        sorted_remove = cumsum > top_p
        sorted_remove[..., 0] = False

        # Map removal mask back to original vocab order
        remove_mask = torch.zeros_like(sorted_remove, dtype=torch.bool).scatter(-1, sorted_idx, sorted_remove)

        # Conjoin with existing keep
        keep = keep & (~remove_mask)

    # --- Guarantee at least one token kept per (B,L) ---
    any_kept = keep.any(dim=-1, keepdim=True)                               # [B,L,1]
    argmax_idx = logits.argmax(dim=-1, keepdim=True)                        # [B,L,1]
    force_one = torch.zeros_like(keep, dtype=torch.bool).scatter(-1, argmax_idx, True)
    keep = torch.where(any_kept, keep, force_one)

    # Final masked logits -> probabilities
    masked_logits = logits.masked_fill(~keep, float('-inf'))
    probs = torch.softmax(masked_logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # Sample
    replacement = num_samples > 1
    idx = torch.multinomial(probs.view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng)
    return idx.view(B, L, num_samples)


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C
    


@torch.no_grad()
def precompute_rope1d_for_batch(
    dim: int,
    lengths_per_sample,          # (N, 4) list/ndarray/tensor of ints: per-level valid lengths
    max_length: int,             # canonical length L*, e.g. 49
    pad_to_multiplier: int = 1,  # e.g., 128 for kernel-friendly shapes
    base: float = 10000.0,
    device=None,
    scaling_factor: float = 1.0,
):
    """
    Build scale-normalized 1D RoPE for a batch of variable per-level lengths.
    Outputs:
      rope_batch   : (N, 2, 1, 1, 1, L_batch_pad, dim//2)
      level_offsets: (N, 4) cumulative starts within each sample's concatenated sequence
      total_lens   : (N,)
    """
    assert dim % 2 == 0, "RoPE requires even head_dim per head"
    half = dim // 2

    # --- Master 1D cos/sin line on canonical grid [0..max_length-1] (fp32) ---
    inv_idx = torch.arange(0, half, 2, dtype=torch.float32, device=device) / half
    inv_freq = 1.0 / (base ** inv_idx)  # [half/2]

    t_line = torch.arange(max_length, dtype=torch.float32, device=device) / scaling_factor  # [L*]
    freqs = torch.outer(t_line, inv_freq)             # [L*, half/2]
    freqs = torch.repeat_interleave(freqs, 2, dim=-1) # [L*, half]  (even/odd share freq)
    master = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=0)  # [2, L*, half]

    # --- Normalize inputs ---
    lengths = torch.as_tensor(lengths_per_sample, device=device, dtype=torch.long)  # [N, 4]
    if lengths.ndim != 2 or lengths.size(1) != 4:
        raise ValueError("lengths_per_sample must be shape (N, 4)")
    N = lengths.size(0)

    # --- Build per-sample concatenated RoPE and offsets ---
    per_seq = []
    offsets = torch.zeros((N, 4), device=device, dtype=torch.long)
    total_lens = torch.zeros((N,), device=device, dtype=torch.long)

    for i in range(N):
        segs = []
        start = 0
        for lvl in range(4):
            L = int(lengths[i, lvl].item())
            offsets[i, lvl] = start
            if L > 0:
                # scale-normalized mapping: map each level position to canonical length
                idx = torch.round(
                    torch.arange(L, dtype=torch.float32, device=device) * (max_length / float(L))
                ).clamp_(0, max_length - 1).to(torch.long)  # [L]
                seg = master[:, idx, :]   # [2, L, half]
                segs.append(seg)
                start += L
            # if L == 0, just skip (no tokens at this level)
        total_lens[i] = start
        cat = torch.cat(segs, dim=1) if len(segs) else master.new_zeros((2, 0, half))
        per_seq.append(cat)  # [2, Li, half]

    # --- Batch pad to max total length (optionally rounded to multiplier) ---
    L_batch = int(total_lens.max().item()) if N > 0 else 0
    L_pad = L_batch

    # Allocate output and pack each sample's concatenated sequence
    rope_batch = master.new_zeros((N, 2, 1, 1, 1, L_pad, half))  # [N,2,1,1,1,L_pad,half]
    for i, cat in enumerate(per_seq):
        L_i = cat.shape[1]
        if L_i > 0:
            rope_batch[i, :, 0, 0, 0, :L_i, :] = cat  # fill valid tail; rest stays zero-padded

    return rope_batch, offsets, total_lens


@torch.no_grad()
def precompute_rope2d_for_batch(
    dim: int,                            # head_dim
    scale_shapes,                        # iterable of (T_s, J_s) pairs, one per scale
    max_t: int,                          # canonical T (finest scale's T), e.g. 49
    max_j: int,                          # canonical J (finest scale's J), e.g. 7
    base: float = 10000.0,
    device=None,
    scaling_factor: float = 1.0,
):
    """Build 2D RoPE for a sequence formed by concatenating per-scale flat
    rasterizations of (T_s × J_s) grids.

    Inside each scale, position k in [0, T_s*J_s) decodes to
    ``(t_local, j) = divmod(k, J_s)``. We then scale-normalize the temporal axis
    so each scale's t maps to the canonical ``max_t`` range:
        ``t_canon = round(t_local * max_t / T_s)``.
    The skeletal axis j is *not* scale-normalized — J_s = J across all scales
    in the Option-A scheme, so ``j_canon = j``.

    The per-position rotation channels (``dim/2`` of them in standard RoPE) are
    split: the first ``dim/4`` rotate with t_canon, the second ``dim/4`` with
    j_canon. Output tensor shape matches ``precompute_rope1d_for_batch``'s
    ``(N=1, 2, 1, 1, 1, L_pad, dim//2)`` so the attention block consumes it
    unchanged.

    Args:
        dim: per-head dimension (must be divisible by 4).
        scale_shapes: list/tuple of (T_s, J_s) pairs.
        max_t, max_j: canonical grid extents (typically the finest scale's).
        base, scaling_factor: standard RoPE knobs.
        device: torch device for the returned tensors.

    Returns:
        rope_batch    : (1, 2, 1, 1, 1, L_total, dim//2)
        level_offsets : (1, S) cumulative starts within the concat sequence
        total_lens    : (1,)
    """
    assert dim % 4 == 0, "RoPE2D requires head_dim divisible by 4 (split half ⇒ t + j)"
    half = dim // 2
    qtr = half // 2          # channels per axis

    # --- t-axis master: cos/sin on canonical [0..max_t-1] ---
    inv_idx_t = torch.arange(0, qtr, 2, dtype=torch.float32, device=device) / qtr
    inv_freq_t = 1.0 / (base ** inv_idx_t)                        # (qtr/2,)
    t_line = torch.arange(max_t, dtype=torch.float32, device=device) / scaling_factor  # (max_t,)
    freqs_t = torch.outer(t_line, inv_freq_t)                     # (max_t, qtr/2)
    freqs_t = torch.repeat_interleave(freqs_t, 2, dim=-1)         # (max_t, qtr)
    master_t = torch.stack([torch.cos(freqs_t), torch.sin(freqs_t)], dim=0)  # (2, max_t, qtr)

    # --- j-axis master: cos/sin on [0..max_j-1] ---
    inv_idx_j = torch.arange(0, qtr, 2, dtype=torch.float32, device=device) / qtr
    inv_freq_j = 1.0 / (base ** inv_idx_j)
    j_line = torch.arange(max_j, dtype=torch.float32, device=device) / scaling_factor
    freqs_j = torch.outer(j_line, inv_freq_j)
    freqs_j = torch.repeat_interleave(freqs_j, 2, dim=-1)
    master_j = torch.stack([torch.cos(freqs_j), torch.sin(freqs_j)], dim=0)  # (2, max_j, qtr)

    # --- Build the concat sequence's per-position (t_canon, j_canon) coords ---
    S = len(scale_shapes)
    offsets = torch.zeros((1, S), device=device, dtype=torch.long)
    segs = []
    start = 0
    for si, (T_s, J_s) in enumerate(scale_shapes):
        offsets[0, si] = start
        L_s = T_s * J_s
        # Row-major rasterization: position k = t_local * J_s + j.
        k = torch.arange(L_s, device=device)
        t_local = k // J_s                                        # (L_s,)
        j_idx = k % J_s                                            # (L_s,)
        # Scale-normalize t to canonical max_t. (j_canon = j_idx since J_s == max_j.)
        t_canon = torch.round(
            t_local.to(torch.float32) * (max_t / float(T_s))
        ).clamp_(0, max_t - 1).to(torch.long)
        j_canon = j_idx.clamp_(0, max_j - 1).to(torch.long)
        t_seg = master_t[:, t_canon, :]                            # (2, L_s, qtr)
        j_seg = master_j[:, j_canon, :]                            # (2, L_s, qtr)
        seg = torch.cat([t_seg, j_seg], dim=-1)                    # (2, L_s, half)
        segs.append(seg)
        start += L_s
    cat = torch.cat(segs, dim=1) if segs else master_t.new_zeros((2, 0, half))
    total_lens = torch.tensor([start], device=device, dtype=torch.long)

    rope_batch = master_t.new_zeros((1, 2, 1, 1, 1, start, half))
    rope_batch[0, :, 0, 0, 0, :, :] = cat
    return rope_batch, offsets, total_lens


############# Mask needed helpers
def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask


class MoScaleFSQ(nn.Module):
    """MoScale adapted for the SkelVQ-FSQ tokenizer.

    Surgical fork of MoScale (sibling moscale.py): same T5 cross-attention,
    AdaLN, RoPE, scale-by-scale layout, and (mostly) the same forward/generate
    bodies. The differences:

      * Output head emits ``code_dim * effective_levels`` logits per position
        (e.g., 32*7 = 224 for our ckpt) instead of one codebook index. Loss
        becomes 32 parallel ``effective_levels``-way cross-entropies.

      * ``dequantize`` calls the wrapper's ``quantizer.indices_to_codes`` over
        per-channel int indices ``(..., D)`` rather than a codebook lookup
        ``(...)`` shape int.

      * BERT-style mask augmentation removed. Pure next-scale AR: input is
        the cumulative partial reconstruction (``x_BLC_wo_prefix``) only, no
        per-position GT-token stream concat. Loss is over all positions, not
        only masked ones. Matches Infinity (arxiv 2412.04431).

      * ``generate`` is single-pass per scale (no iterative remasking inner
        loop). Top-k/top-p sampling is per-channel after reshaping logits to
        ``(B, L_s, D, V)``.
    """

    def __init__(self, code_dim, latent_dim=256,
                 num_heads=4, dropout=0.1, text_dim=512, cond_drop_prob=0.1, mlp_ratio=4,
                 device=None, cfg=None, full_length=80, scales=[8, 4, 2, 1],
                 shared_aln=False, norm_eps=1e-6, attn_drop_rate=0.,
                 flash_if_available=True, fused_if_available=True, token_embed_init=None,
                 rand_uncond=False, effective_levels=7):
        super(MoScaleFSQ, self).__init__()
        self.effective_levels = effective_levels

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        self.text_cond_length = 20
        self.cfg = cfg
        self.device = device
        self.full_length = full_length
        self.scales = scales

        # 2D / RoPE2D toggle (ScaleMoGen-style skeletal-temporal token map).
        # When True, per-scale token maps are interpreted as (T_s × J) grids and
        # the residual cascade uses 2D interpolation; the AR transformer's
        # positional encoding switches to RoPE2D.  Pair with a 2D-trained FSQ
        # tokenizer (--quantizer_cascade 2d).
        self.use_rope2d = cfg.model.get('use_rope2d', False)
        if self.use_rope2d:
            # Option A: J preserved across scales. Derive T_b from full_length / J.
            self.full_J = cfg.model.get('rope2d_J', 7)
            assert full_length % self.full_J == 0, (
                f"full_length={full_length} must be divisible by J={self.full_J} in RoPE2D mode"
            )
            self.full_T = full_length // self.full_J
            self.patch_sizes_2d = [(max(1, self.full_T // s), self.full_J) for s in self.scales]
            self.patch_sizes = [t * j for (t, j) in self.patch_sizes_2d]
            print(f'[MoScaleFSQ] use_rope2d=True; full_(T,J)=({self.full_T},{self.full_J}); patch_sizes_2d={self.patch_sizes_2d}')
        else:
            self.patch_sizes = [int(full_length // scale) for scale in self.scales]
            self.patch_sizes_2d = None
        self.L = sum(self.patch_sizes)
        self.first_l = self.patch_sizes[0]
        self.cond_drop_prob = cond_drop_prob
        self.num_heads = num_heads

        self.infer_use_kvcache = cfg.model.get('infer_use_kvcache', False)
        print(f'[MoScale] infer_use_kvcache = {self.infer_use_kvcache}')

        init_std = math.sqrt(1 / self.latent_dim / 3)

        self.num_layers = cfg.model.get('num_layers', 4)
        self.head_latent_dim = cfg.model.get('head_latent_dim', self.latent_dim)
        # FSQ has no learnable codebook to look up from — always False.
        self.use_learned_token_map = False
        print(f'[MoScaleFSQ] num_layers = {self.num_layers}')
        print(f'[MoScaleFSQ] head_latent_dim = {self.head_latent_dim}')
        print(f'[MoScaleFSQ] effective_levels = {self.effective_levels}, code_dim = {self.code_dim}')

        g = torch.Generator(device=self.device)
        g.manual_seed(cfg.exp.seed)
        self.rng = g

        self.perturb_rate = self.cfg.training.perturb_rate if ('perturb_rate' in self.cfg.training) else None
        print(f'[MoScale] perturb_rate = {self.perturb_rate}')

        self.input_process = nn.Linear(self.code_dim, self.latent_dim)

        # unconditional embedding for CFG
        cfg_uncond = torch.empty(self.text_cond_length, self.text_dim)
        rng = torch.Generator(device='cpu')
        rng.manual_seed(0)
        torch.nn.init.trunc_normal_(cfg_uncond, std=1.2, generator=rng)
        cfg_uncond /= self.text_dim ** 0.5
        self.cfg_uncond = nn.Parameter(cfg_uncond)


        # map the conditional features
        self.text_norm = FastRMSNorm(self.text_dim, elementwise_affine=True, eps=norm_eps)
        self.text_proj_for_sos = MultiQueryTextPool(self.text_dim, latent_dim, self.patch_sizes[0])
        # process the conditional features
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.latent_dim, 6*self.latent_dim)) if shared_aln else nn.Identity()

        # backbone transformer blocks building
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.use_crossattn = cfg.model.get('use_crossattn', False)
        if self.use_crossattn:
            self.text_proj_for_ca = nn.Sequential(
                nn.Linear(self.text_dim, latent_dim),
                nn.GELU(approximate='tanh'),
                nn.Linear(latent_dim, latent_dim),
                )

        ######################### prepare for MoScale #########################
        self.create_PE()

        attn_l2_norm = cfg.model.get('attn_l2_norm', True)
        print(f'[MoScale] attn_l2_norm = {attn_l2_norm}')

        num_layers = self.num_layers
        masked_drop_path_rate = 0.1 * num_layers/24
        masked_dpr = [x.item() for x in torch.linspace(0, masked_drop_path_rate, num_layers)]
        self.masked_blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.latent_dim, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.head_latent_dim, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=dropout, attn_drop=attn_drop_rate, drop_path=masked_dpr[block_idx], last_drop_p=0 if block_idx == 0 else masked_dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
                use_rope=self.mask_use_rope,
                use_crossattn=self.use_crossattn, kv_dim=self.latent_dim
            )
            for block_idx in range(num_layers)
        ])

        self.pad_id = -1
        self.mask_id = -2

        code_init_var = math.sqrt(1 / self.code_dim / 3)
        self.masked_token_embedding = nn.Parameter(torch.empty(1, self.code_dim))
        nn.init.trunc_normal_(self.masked_token_embedding.data, mean=0, std=code_init_var)
        self.padded_token_embedding = nn.Parameter(torch.empty(1, self.code_dim))
        nn.init.trunc_normal_(self.padded_token_embedding.data, mean=0, std=code_init_var)

        # project after concat: accumulated features + token embeddings -> block dim
        self.token_dim_proj = nn.Linear(self.code_dim, self.latent_dim)
        self.latent_dim_proj = nn.Linear(self.latent_dim*2, self.head_latent_dim)

        # FSQ pure-AR path: BERT mask aug is removed, so the residual stream
        # arrives at blocks at `latent_dim` (single stream), not `2*latent_dim`.
        # Project to `head_latent_dim` here so block embed_dim, pos_start,
        # lvl_embed, and head ops all line up when head_latent_dim != latent_dim.
        self.input_to_block = (
            nn.Linear(self.latent_dim, self.head_latent_dim)
            if self.head_latent_dim != self.latent_dim
            else nn.Identity()
        )

        self.noise_schedule = cosine_schedule

        self.bert_replace_prob = 0.1
        self.bert_mask_prob = 0.8
        self.mask_augment_prob = [0.0, 0.1]

        self.sample_level_times = self.cfg.training.sample_level_times if ('sample_level_times' in self.cfg.training) else None
        print(f'[MoScale] sample_level_times = {self.sample_level_times}')
        assert len(self.patch_sizes) == len(self.sample_level_times)
        
        head_latent_dim = self.head_latent_dim

        # block-wise causal mask
        patch_sizes = self.patch_sizes
        L = sum(patch_sizes)
        d = torch.cat([torch.full((pn,), i, dtype=torch.long) for i, pn in enumerate(patch_sizes)]).view(1, L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)    # [1, 91] [[0*20,1,1,1,1,1,1,1,1,1,1,1,1,2...]]
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, L, L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())   #[1, 1, 91, 91] Block-wise causal mask


        # FSQ output head: predict per-channel categorical over effective_levels.
        # Total logits per position = code_dim * effective_levels (e.g. 32*7=224).
        self.fsq_V = self.code_dim * self.effective_levels
        self.vq_nb_code = self.fsq_V   # alias kept for any code paths that reference this name
        self.head_nm = AdaLNBeforeHead(head_latent_dim, self.latent_dim, norm_layer=norm_layer)
        self.head = nn.Linear(head_latent_dim, self.fsq_V)

        self.init_weights()

        self.text_emb = T5TextEncoder(
            device,
            local_files_only=False,
            from_pretrained=cfg.text_embedder.version,
            model_max_length=cfg.data.max_text_length
        )

        self.infer_rope_base = self.rope_base.repeat(64, 1, 1, 1, 1, 1, 1)


    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.latent_dim / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.masked_blocks)

        for block_idx, sab in enumerate(self.masked_blocks):
            # -------- scaling factor (Infinity-style) --------
            scale = math.sqrt(2 * depth)  # keep same policy as your original code (div_ by sqrt(2*depth))

            # -------- Self-Attn proj (supports .self_attn or .attn) --------
            sa = getattr(sab, "self_attn", None) or getattr(sab, "attn", None)
            if sa is not None and hasattr(sa, "proj") and hasattr(sa.proj, "weight"):
                sa.proj.weight.data.div_(scale)

            # -------- Cross-Attn proj (if the block has it) --------
            ca = getattr(sab, "cross_attn", None)
            if ca is not None and hasattr(ca, "proj") and hasattr(ca.proj, "weight"):
                ca.proj.weight.data.div_(scale)

            # -------- FFN proj (fc2) --------
            if hasattr(sab, "ffn") and hasattr(sab.ffn, "fc2") and hasattr(sab.ffn.fc2, "weight"):
                sab.ffn.fc2.weight.data.div_(scale)

            # optional FFN gating (if present)
            if hasattr(sab, "ffn") and getattr(sab.ffn, "fcg", None) is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)

            # -------- AdaLN parameters (6*C layout: [gamma1, gamma2, scale1, scale2, shift1, shift2]) --------
            # learned conditioner path
            if getattr(sab, "ada_lin", None) is not None:
                lin = sab.ada_lin[-1]  # final Linear that outputs 6*C
                if hasattr(lin, "weight"):
                    # first 2*C rows = gammas
                    lin.weight.data[: 2 * self.latent_dim].mul_(init_adaln_gamma)
                    # last 4*C rows = scale & shift
                    lin.weight.data[2 * self.latent_dim :].mul_(init_adaln)
                if hasattr(lin, "bias") and lin.bias is not None:
                    lin.bias.data.zero_()

            # shared conditioner path
            elif getattr(sab, "ada_gss", None) is not None:
                # sab.ada_gss: [1, 1, 6, C]
                sab.ada_gss.data[:, :, :2, :].mul_(init_adaln_gamma)  # gammas
                sab.ada_gss.data[:, :,  2:, :].mul_(init_adaln)       # scale/shift



    def create_PE(self):
        # Position Encoding: scale-normalized RoPE + lvl embedding + pos_start
        init_std = math.sqrt(1 / self.head_latent_dim / 3)
        num_heads = self.num_heads
        self.mask_use_rope = True
        self.lvl_embed = nn.Embedding(len(self.patch_sizes), self.head_latent_dim)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        head_dim = self.head_latent_dim // num_heads
        if self.use_rope2d:
            self.rope_base, self.rope_offsets, self.rope_total_lens = precompute_rope2d_for_batch(
                head_dim, self.patch_sizes_2d,
                max_t=self.full_T, max_j=self.full_J, device=self.device,
            )
        else:
            self.rope_base, self.rope_offsets, self.rope_total_lens = precompute_rope1d_for_batch(
                head_dim, torch.tensor([self.patch_sizes]),
                max_length=self.patch_sizes[-1], device=self.device,
            )
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.head_latent_dim))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

    def encode_text(self, raw_text):
        text_embedding, mask = self.text_emb.get_text_embeddings(raw_text)
        return text_embedding, mask

    def get_logits(self, h: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor, non_pad_mask, vq_model) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in VAR inference
        current_mask = non_pad_mask[si].unsqueeze(-1).permute(0, 2, 1)  # (B, 1, L_s)
        h_BChw = h_BChw * current_mask.float()  # (B, D, L_s)

        if self.use_rope2d:
            # f_hat is carried as (B, D, T_b, J_b). h_BChw arrives flat (B, D, T_s*J).
            B, D, L_s = h_BChw.shape
            T_s, J_s = self.patch_sizes_2d[si]
            assert L_s == T_s * J_s, f"L_s={L_s} != T_s*J_s={T_s*J_s} at scale {si}"
            h_2d = h_BChw.reshape(B, D, T_s, J_s)
            T_full, J_full = self.full_T, self.full_J
            if (T_s, J_s) != (T_full, J_full):
                h_up = F.interpolate(h_2d, size=(T_full, J_full), mode='bilinear', align_corners=False)
            else:
                h_up = h_2d
            h_up = vq_model.quantizer.quant_resi[si / (SN - 1)](h_up)
            f_hat.add_(h_up)
            if si != SN - 1:
                T_next, J_next = self.patch_sizes_2d[si + 1]
                nxt = F.interpolate(f_hat, size=(T_next, J_next), mode='area')
                return f_hat, nxt.reshape(B, D, T_next * J_next)
            else:
                return f_hat, f_hat.reshape(B, D, T_full * J_full)

        # 1D legacy path.
        feat_seq_len = self.patch_sizes[-1]
        if si != SN-1:
            h = vq_model.quantizer.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=feat_seq_len, mode='linear'))     # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=self.patch_sizes[si+1], mode='area')
        else:
            h = vq_model.quantizer.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat
    

    def preprocess_motion_for_training(self, motion, m_lens, vq_model, perturb_rate, train=False):
        vq_model.train(False)

        # Wrapper's encode returns:
        #   id_list[s]:      (B, L_s, code_dim) int64
        #   raw_features[s]: (B, code_dim, L) float at full bottleneck L
        # `m_lens` is in original-frame units; the wrapper handles the L=T_b*J_b
        # conversion internally and returns fixed-length tensors.
        id_list, raw_features, _ = vq_model.encode(
            motion[..., :self.cfg.data.dim_pose],
            m_lens.clone() if m_lens is not None else None,
            perturb_rate=perturb_rate,
            train=train,
        )

        # SALAD-aligned pattern: no per-position masking anywhere in the AR /
        # cascade pipeline. The SALAD encoder/decoder run over the full padded
        # length without m_lens-awareness, so the AR also processes every
        # position as a real one. Labels at trailing cells are whatever FSQ
        # codes the (frozen) tokenizer produces from encoder-on-zero-padded
        # input — deterministic given the encoder, trivial to fit, doesn't
        # corrupt the model's text-to-motion learning. The only masking the
        # pipeline does is post-decoder zeroing of pred_motion past m_length,
        # which `evaluate_once` (and SALAD's eval_t2m) do before the BiGRU eval.
        #
        # `non_pad_mask` is kept all-True here so downstream call sites that
        # consume it (forward's CE, generate's f_hat cascade) reduce to no-op
        # masking. This is the explicit "match SALAD's no-masking" choice.
        non_pad_mask = [
            torch.ones(ele.shape[0], ele.shape[1], dtype=torch.bool, device=ele.device)
            for ele in id_list
        ]

        labels = torch.cat(id_list, dim=1)

        downsampled_feature_list = []
        for i in range(len(self.patch_sizes) - 1):
            if self.use_rope2d:
                # raw_features[i]: (B, D, T_b, J_b). Downsample to (T_{i+1}, J) and flatten.
                T_next, J_next = self.patch_sizes_2d[i + 1]
                ds_2d = F.interpolate(raw_features[i], size=(T_next, J_next), mode='area')
                ds = ds_2d.reshape(ds_2d.shape[0], ds_2d.shape[1], T_next * J_next)
                downsampled_feature_list.append(ds)
            else:
                next_size = self.patch_sizes[i + 1]
                downsampled_feature_list.append(F.interpolate(raw_features[i], size=next_size, mode='area'))
        x_BLC_wo_prefix = torch.cat(downsampled_feature_list, dim=2)

        return labels, x_BLC_wo_prefix.permute(0, 2, 1), torch.cat(non_pad_mask, dim=1), raw_features
    
    def dequantize(self, code_idx, vq_model):
        if not self.use_learned_token_map:
            return vq_model.quantizer.dequantize(code_idx)
        else:
            mask = code_idx == -1.
            code_idx = code_idx.masked_fill(mask, 0.)
            x = F.embedding(code_idx, self.token_emb)

            x[mask] = 0.
            return x

    def forward(self, motion, y, m_lens, vq_model, train=False):
        labels, x_BLC_wo_prefix, non_pad_mask, raw_features = self.preprocess_motion_for_training(motion, m_lens, vq_model, self.perturb_rate, train)
        B = x_BLC_wo_prefix.shape[0]

        with torch.no_grad():
            cond_embs, cond_att_mask = self.encode_text(y)
            cond_padding_mask = (cond_att_mask==0)

        max_seqlen_k = self.text_cond_length
        indi_length = [0]
        indi_feature = []

        for i in range(cond_embs.shape[0]):
            current_length = (~cond_padding_mask[i]).sum().item()
            indi_length.append(current_length)
            indi_feature.append(cond_embs[i, :current_length, :])

        cu_seqlens_k = torch.tensor(indi_length, device=self.device).cumsum(0).to(torch.int32)

        lens = indi_length[1:]
        kv_compact = torch.cat(indi_feature, dim=0)

        total = 0
        for le in lens:
            if random.random() < self.cond_drop_prob:
                kv_compact[total:total+le] = self.cfg_uncond[:le]
            total += le

        must_on_graph = self.cfg_uncond[0, 0] * 0   # trick for gradient graph in ddp
        kv_compact = self.text_norm(kv_compact).contiguous()
        sos = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).float().contiguous()
        cond_BD = sos.max(dim=1).values.unsqueeze(1)   # B, 1, C

        if self.use_crossattn:
            kv_compact = self.text_proj_for_ca(kv_compact).contiguous()
            kv_compact[0, 0] += must_on_graph
            ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        else:
            ca_kv = None

        cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous()

        # detect mixed precision dtype
        temp = x_BLC_wo_prefix.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        device = x_BLC_wo_prefix.device

        # ---- FSQ AR path: no BERT-style mask augmentation. ----
        # `labels` here is (B, L_total, code_dim) for FSQ — sum of per-scale
        # (B, L_s, code_dim) tensors concatenated along the L axis.
        ntokens = labels.shape[1]
        # padded positions: a row of all -1 across the channel axis (set by the
        # wrapper for tokens beyond m_lens). Treat them like the original code
        # treated pad_id=-1: ignored in CE.
        padded_token = (labels == -1).all(dim=-1)            # (B, L)
        non_pad_mask = ~padded_token

        # x_BLC: cumulative partial-reconstruction stream + SOS for scale 0.
        x_BLC_wo_prefix = self.input_process(x_BLC_wo_prefix)
        x_BLC = torch.cat((sos.expand(B, self.first_l, -1), x_BLC_wo_prefix), dim=1)

        attn_bias = self.attn_bias_for_masking.repeat(B, 1, 1, 1)
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        # Pure AR: single-stream (no per-position token concat).
        x_BLC_token = x_BLC
        x_BLC_token = self.input_to_block(x_BLC_token)

        for block_i, b in enumerate(self.masked_blocks):
            if block_i == 0:
                x_BLC_token[:, :self.first_l] = x_BLC_token[:, :self.first_l] + self.pos_start.expand(B, self.first_l, -1)
                x_BLC_token = x_BLC_token + self.lvl_embed(self.lvl_1L[:, :self.L].expand(B, -1))
            rope_batch = self.rope_base.repeat(B, 1, 1, 1, 1, 1, 1)
            x_BLC_token = b(x=x_BLC_token, cond_BD=cond_BD_or_gss, attn_bias=attn_bias, ca_kv=ca_kv, rope_batch=rope_batch)

        predict = self.get_logits(x_BLC_token.float(), cond_BD)
        # predict: (B, L_total, code_dim * effective_levels)
        predict = predict.reshape(B, ntokens, self.code_dim, self.effective_levels)
        # F.cross_entropy expects (B, V, ...spatial...) with target (B, ...spatial...).
        # We treat (L_total, code_dim) as the spatial axes and class-dim = effective_levels.
        # gt = labels: (B, L_total, code_dim) with values in [0, effective_levels).
        # Mask out padded positions by setting their target to ignore_index (-100).
        gt = labels.clone()
        gt[~non_pad_mask] = -100   # ignore_index for padded positions
        ce_loss = F.cross_entropy(
            predict.permute(0, 3, 1, 2).contiguous(),  # (B, V, L_total, code_dim)
            gt,                                          # (B, L_total, code_dim)
            ignore_index=-100,
        )

        with torch.no_grad():
            pred_idx = predict.argmax(dim=-1)             # (B, L_total, code_dim)
            mask_valid = non_pad_mask.unsqueeze(-1)        # (B, L_total, 1)
            correct = (pred_idx == labels) & mask_valid
            n = mask_valid.float().sum() * self.code_dim
            acc = (correct.float().sum() / n.clamp_min(1)).item() if n > 0 else 0.0

            # Per-scale accuracy: useful diagnostic since scale 0 has no
            # cumulative-recon hint (only SOS) while scales 1..S-1 do.
            per_scale_acc = {}
            offset = 0
            for si, pl in enumerate(self.patch_sizes):
                lo, hi = offset, offset + pl
                pred_s = pred_idx[:, lo:hi]
                gt_s = labels[:, lo:hi]
                mask_s = non_pad_mask[:, lo:hi].unsqueeze(-1)
                n_s = mask_s.float().sum() * self.code_dim
                correct_s = (pred_s == gt_s) & mask_s
                per_scale_acc[f"acc_scale_{si}"] = (
                    (correct_s.float().sum() / n_s.clamp_min(1)).item() if n_s > 0 else 0.0
                )
                offset = hi

        return ce_loss, pred_idx, acc, per_scale_acc

    def extra_repr(self):
        return f'num_layers={self.num_layers}'

    @torch.no_grad()
    @eval_decorator
    def generate(self, conds, m_lens, cond_scale,
                 temperature=1, top_p_thres=0.9,
                 vq_model=None, sample_time=None,
                 gt_prefix_indices=None, gt_prefix_count=0):
        # DIAGNOSTIC: gt_prefix_indices is an optional list of (B, L_s, D) int64
        # GT indices per scale. If supplied with gt_prefix_count > 0, the first
        # `gt_prefix_count` scales use GT indices instead of AR samples (the
        # cascade then continues with AR sampling). This localizes per-scale
        # cascade drift without changing any training behaviour.

        if sample_time is not None:
            self.sample_level_times = sample_time

        # SALAD-aligned pattern: no per-position masking inside the AR.
        # We still build a non_pad_mask list of all-True tensors so the
        # f_hat cascade's existing call to get_next_autoregressive_input
        # (which multiplies by non_pad_mask[si]) becomes a no-op. m_lens is
        # carried through the API for backward compatibility but doesn't
        # affect per-position behavior inside generate; it's used only by
        # the caller (evaluate_once) to zero pred_motion past m_length
        # before passing to the BiGRU eval — same one masking point SALAD
        # itself does in eval_t2m.py.
        B_local = m_lens.shape[0] if hasattr(m_lens, "shape") else len(m_lens)
        device = m_lens.device if hasattr(m_lens, "device") else self.device
        # In 2D mode patch_sizes = [T_s*J for ...], so `full_length // scale`
        # diverges from the actual per-scale token count (e.g. 343//4=85 vs 84
        # for T_s=12, J=7). Always use self.patch_sizes[i] as the source of
        # truth for the per-scale token count — that's what the cascade and
        # AR layout were built against.
        non_pad_mask = [
            torch.ones(B_local, self.patch_sizes[i],
                       dtype=torch.bool, device=device)
            for i in range(len(self.scales))
        ]
        non_pad_mask_stack = torch.cat(non_pad_mask, dim=1).repeat(2, 1)  # [2*B, L]

        # first get the text conditions; FOR CFG (Classifier Freee Guidance)
        B = len(conds)
        cond_embs, cond_att_mask = self.encode_text(conds)
        cond_padding_mask = (cond_att_mask==0)

        # prepare the cond_embs
        max_seqlen_k = self.text_cond_length
        indi_length = [0]
        indi_feature = []

        for i in range(cond_embs.shape[0]):
            current_length = (~cond_padding_mask[i]).sum().item()
            indi_length.append(current_length)
            indi_feature.append(cond_embs[i, :current_length, :])

        cu_seqlens_k = torch.tensor(indi_length, device=self.device).cumsum(0).to(torch.int32)

        lens = indi_length[1:]
        kv_compact = torch.cat(indi_feature, dim=0)


        # prepare for the conds for CFG
        kv_compact_un = kv_compact.clone()
        total = 0
        for le in lens:
            kv_compact_un[total:total+le] = (self.cfg_uncond)[:le]
            total += le
        kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
        cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)

        # begin the processing
        kv_compact = self.text_norm(kv_compact).contiguous()
        sos = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).float().contiguous()
        cond_BD = sos.max(dim=1).values.unsqueeze(1)   # B, 1, C

        if self.use_crossattn:
            kv_compact = self.text_proj_for_ca(kv_compact).contiguous()
            ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        else:
            ca_kv = None


        cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous()  # gss: gamma, scale, shift; cond_BD_or_gss should be float32

        feat_seq_len = self.patch_sizes[-1]; num_stages_minus_1 = len(self.patch_sizes) - 1
        if self.use_rope2d:
            f_hat = cond_BD.new_zeros(B, self.code_dim, self.full_T, self.full_J)
        else:
            f_hat = cond_BD.new_zeros(B, self.code_dim, feat_seq_len)
        return_list = []

        ######################## Per-scale inference with block-causal self-attn ########################
        # Pure FSQ AR: single forward pass per scale (no iterative remasking).
        # The cumulative reconstruction stream `x_in` is built from the
        # *previous* scales' partial reconstructions; for scale 0 the input is
        # just the SOS broadcast over patch_sizes[0] positions.
        attn_bias = self.attn_bias_for_masking.repeat(2 * B, 1, 1, 1)

        # next_token_map: input embedding for the *next* scale's positions.
        # At scale 0 this is SOS; for subsequent scales it's the output of
        # get_next_autoregressive_input applied to the previous f_hat.
        next_token_map = sos.expand(2 * B, self.patch_sizes[0], -1)
        cur_L = 0
        x_in = None

        for i, pl in enumerate(self.patch_sizes):
            cur_L += pl
            if i == 0:
                x_in = next_token_map
            else:
                x_in = torch.cat([x_in, next_token_map], dim=1)
            x_out_token = x_in.clone()
            x_out_token = self.input_to_block(x_out_token)

            cur_attn_bias = attn_bias[:, :, :cur_L, :cur_L]

            # Add positional embeddings.
            x_out_token[:, :self.first_l] = x_out_token[:, :self.first_l] + self.pos_start.expand(2 * B, self.first_l, -1)
            x_out_token = x_out_token + self.lvl_embed(self.lvl_1L[:, :cur_L].expand(2 * B, -1))
            rope_batch = self.infer_rope_base[: x_out_token.shape[0], ..., :cur_L, :]

            for block_i, b in enumerate(self.masked_blocks):
                x_out_token = b(x=x_out_token, cond_BD=cond_BD_or_gss,
                                attn_bias=cur_attn_bias, ca_kv=ca_kv, rope_batch=rope_batch)

            # Logits for the full sequence so far -> read off the current scale's positions only.
            logits_BlV = self.get_logits(x_out_token.float(), cond_BD)   # (2B, cur_L, fsq_V)
            cur_logits = logits_BlV[:, -pl:, :]                            # (2B, pl, fsq_V)

            # Reshape to (2B, pl, code_dim, effective_levels) and apply CFG per channel.
            cur_logits = cur_logits.reshape(2 * B, pl, self.code_dim, self.effective_levels)
            t = cond_scale + (i - 1) * (-0.25)
            cur_logits = (1 + t) * cur_logits[:B] - t * cur_logits[B:]    # (B, pl, D, V)

            # Per-channel sampling: flatten (pl, code_dim) -> 1D for the sampler.
            cur_logits = cur_logits.reshape(B, pl * self.code_dim, self.effective_levels)
            sampled = sample_with_top_k_top_p_(
                cur_logits, rng=self.rng, top_k=0, top_p=top_p_thres,
                num_samples=1, temperature=temperature,
            )[:, :, 0]
            idx_BlD = sampled.reshape(B, pl, self.code_dim)               # (B, pl, code_dim)

            # DIAGNOSTIC override: replace AR sample with GT for the first
            # `gt_prefix_count` scales. Cascade continues from the GT-driven
            # f_hat for subsequent (still-sampled) scales.
            if gt_prefix_indices is not None and i < gt_prefix_count:
                idx_BlD = gt_prefix_indices[i].to(idx_BlD.device).to(idx_BlD.dtype)

            cur_pad = ~non_pad_mask[i]
            # NOTE: do NOT overwrite padded positions with -1 in return_list.
            # decode_from_indices treats every entry as a valid FSQ idx;
            # injecting -1 (and then clamping to 0 downstream) yields code
            # (0-half)/half ≈ -0.86 per channel which the conv decoder bleeds
            # into the boundary of the real motion (huge FID regression).
            # Caller is responsible for masking pred_motion past m_length
            # (evaluate_once already does this).
            return_list.append(idx_BlD)

            # Convert the sampled indices to continuous codes for the residual cascade.
            # Wrapper's quantizer.indices_to_codes handles (..., D) -> (..., D) floats.
            valid_idx = idx_BlD.clamp(min=0)                              # avoid negative idx in dequant
            h_BlD = vq_model.quantizer.indices_to_codes(valid_idx)         # (B, pl, D)
            # Zero-out padded positions.
            h_BlD = torch.where(cur_pad.unsqueeze(-1).expand_as(h_BlD), torch.zeros_like(h_BlD), h_BlD)
            h_BChw = h_BlD.transpose(1, 2).contiguous()                    # (B, D, pl)

            # Update f_hat (cumulative reconstruction at full L) and build next-scale input.
            f_hat, next_token_map = self.get_next_autoregressive_input(
                i, len(self.patch_sizes), f_hat, h_BChw, non_pad_mask, vq_model,
            )
            if i != num_stages_minus_1:
                next_token_map = next_token_map.transpose(1, 2)
                next_token_map = self.input_process(next_token_map)
                next_token_map = next_token_map.repeat(2, 1, 1)

        return return_list