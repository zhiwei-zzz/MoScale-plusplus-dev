"""SkelVQ-FSQ wrapper that exposes MoScale's HRVQVAE-shaped API.

Lets MoScale's transformer + trainer treat SkelVQ-FSQ (SALAD encoder/decoder
with Mentzer FSQ at L=8) as a drop-in tokenizer replacing HRVQVAE. The
fundamental shape difference vs. HRVQVAE:

  HRVQVAE:    one codebook index per (scale, position).        Vocab = nb_code (e.g. 512).
  SkelVQ-FSQ: 32 ordered FSQ levels per (scale, position).     Vocab = effective_levels = 7 (per-channel).

So `idx_list[s]` from this wrapper is `(B, L_s, code_dim)` int64 in
[0, effective_levels), not `(B, L_s)`. Downstream (MoScale's transformer head)
predicts one `effective_levels`-way categorical per (position, channel) — see
Infinity (arxiv 2412.04431) and the project plan for the recipe.

Usage (from inside the moscale/ subdir, with the SALAD repo as parent):
    from model.vq.skelvq_wrapper import SkelVQWrapper
    vq_model = SkelVQWrapper(ckpt_path="<SALAD>/checkpoints/t2m/skelvq_fsq/model/net_best_fid.tar")
    idx_list, q_cum_list, f_hat = vq_model.encode(motion, m_lens=None)
    out_motion = vq_model.decode_from_indices(idx_list)
"""
from __future__ import annotations

import os
import sys
import types
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_salad_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", ".."))


_SALAD_ROOT = _resolve_salad_root()


def _import_skelvq_with_isolated_utils():
    """Import `SkelVQ` and `MultiScaleFSQ` from the SALAD repo.

    Both SALAD and the parent `moscale/` directory contain a sibling `utils/`
    with overlapping module names but different file sets. Bare imports like
    `from utils.skeleton import *` (used inside SALAD's models/skeleton/pool.py)
    resolve against the **first** `utils` package on sys.path / in sys.modules.
    If moscale was loaded first — as is typical when running from `moscale/`
    — SALAD's `utils.skeleton` is invisible.

    Workaround: temporarily insert SALAD root at sys.path[0] and detach any
    cached moscale-side `utils*` / `models*` / `common*` modules from
    sys.modules for the duration of the SALAD import chain. After SkelVQ is
    imported, reattach moscale's modules so the rest of moscale keeps
    functioning unchanged. Module-level imports inside SkelVQ are bound at
    SALAD-import time and don't re-resolve later.
    """
    saved_modules = {}
    saved_path = list(sys.path)
    saved_cwd = os.getcwd()
    try:
        # Park any moscale-side modules whose names collide with SALAD's package layout.
        collision_prefixes = ("utils", "models", "common", "data", "options")
        for name in list(sys.modules.keys()):
            if any(name == p or name.startswith(p + ".") for p in collision_prefixes):
                saved_modules[name] = sys.modules.pop(name)
        # Replace sys.path with a minimal path that only sees SALAD + std lib.
        moscale_root = os.path.join(_SALAD_ROOT, "moscale")
        std_paths = [p for p in saved_path
                     if not p.startswith(moscale_root)
                     and os.path.abspath(p or os.getcwd()) != moscale_root
                     and p not in ("", ".")]
        sys.path[:] = [_SALAD_ROOT] + std_paths
        # Also chdir away from moscale/ if we're inside it, since "" on sys.path
        # would re-introduce moscale's utils/ via cwd resolution.
        if os.path.abspath(saved_cwd) == moscale_root:
            os.chdir(_SALAD_ROOT)

        from models.vae.skel_vq import SkelVQ            # noqa: WPS433
        from models.vae.bsq import MultiScaleFSQ         # noqa: WPS433
        # Snapshot the SALAD-side modules we just loaded (so a second
        # SkelVQWrapper construction reuses them rather than reloading).
        salad_side = {k: v for k, v in sys.modules.items()
                      if any(k == p or k.startswith(p + ".") for p in collision_prefixes)}
    finally:
        # Tear down the SALAD-side `utils` etc. so moscale's later imports
        # rebind to its own versions. Reinstate any moscale-side modules
        # we parked above.
        for name in list(sys.modules.keys()):
            if any(name == p or name.startswith(p + ".") for p in ("utils", "models", "common", "data", "options")):
                # Keep `models.vae.*` and `models.skeleton.*` in place — they're
                # the dependencies of the SkelVQ instance we just built.
                if name.startswith("models.vae") or name.startswith("models.skeleton") \
                   or name == "common.quaternion" or name == "common.skeleton":
                    continue
                # The SALAD modules we want to keep accessible to SkelVQ at
                # runtime are now bound inside SkelVQ's class closure, so
                # we can drop them from sys.modules as long as they aren't
                # being re-imported. To be safe, only purge the ones that
                # could conflict with moscale's expected imports.
                if name in saved_modules or name == "utils" or name.startswith("utils."):
                    sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path
        os.chdir(saved_cwd)
    return SkelVQ, MultiScaleFSQ


# Default opt that matches the trained SkelVQ-FSQ (skelvq_fsq run, 2026-05).
# Sourced from SALAD's options/skel_vq_option.py defaults + the values printed
# in checkpoints/t2m/skelvq_fsq/opt.txt.
_DEFAULT_OPT = dict(
    # encoder/decoder arch
    pose_dim=263,
    joints_num=22,
    contact_joints=[7, 10, 8, 11],   # t2m kinematic chain feet/toes
    dataset_name="t2m",
    latent_dim=32,
    code_dim=32,
    kernel_size=3,
    n_layers=2,
    n_extra_layers=1,
    norm="none",
    activation="gelu",
    dropout=0.1,
    # quantizer
    quantizer_type="fsq",
    fsq_levels=8,
    fsq_inv_temperature=20.0,
    fsq_entropy_weight=0.0,
    fsq_zeta=1.0,
    scales=[8, 4, 2, 1],
    start_drop=-1,
    quantize_dropout_prob=0.0,
    # MSQuantizer-only knobs (unused for FSQ but SkelVQ.__init__ reads them)
    nb_code=512,
    mu=0.99,
    share_quant_resi=4,
    quant_resi=0.0,
    inv_temperature=100.0,
    entropy_weight=0.1,
    zeta=1.0,
    # window — used only for the J_b probe (any value >= 2**n_layers works)
    window_size=64,
)


def _build_skelvq_opt(overrides: Optional[dict] = None) -> types.SimpleNamespace:
    o = dict(_DEFAULT_OPT)
    if overrides:
        o.update(overrides)
    return types.SimpleNamespace(**o)


class _MultiScaleFSQAlias(nn.Module):
    """Thin shim that forwards .indices_to_codes / .effective_levels / .scales /
    .code_dim / a `.dequantize` alias to the underlying MultiScaleFSQ.

    Why a wrapper class instead of returning the raw MultiScaleFSQ? Because
    MoScale's transformer code at moscale/model/transformer/moscale.py:461
    calls `vq_model.quantizer.dequantize(code_idx)` in some paths (it's the
    HRVQVAE MSQuantizer API). Adding the alias here keeps the AR side unchanged
    in spirit — only the per-channel reshape downstream of `dequantize` differs.
    """

    def __init__(self, fsq_module):
        super().__init__()
        self.fsq = fsq_module     # MultiScaleFSQ instance

    @property
    def scales(self):
        return self.fsq.scales

    @property
    def code_dim(self):
        return self.fsq.code_dim

    @property
    def effective_levels(self) -> int:
        return self.fsq.effective_levels

    @property
    def codebook_dim(self) -> int:
        # Match Infinity's API name; same value as code_dim for FSQ.
        return self.fsq.code_dim

    def indices_to_codes(self, idx: torch.Tensor) -> torch.Tensor:
        return self.fsq.indices_to_codes(idx)

    # MSQuantizer alias used in moscale.py
    def dequantize(self, idx: torch.Tensor) -> torch.Tensor:
        return self.fsq.indices_to_codes(idx)

    # MSQuantizer's quant_resi is a learned conv-residual module indexed by the
    # scale ratio in [0, 1]. FSQ has no such module — the residual cascade is
    # pure interpolate-and-subtract. Provide an identity shim so AR-inference
    # call sites that do `vq_model.quantizer.quant_resi[α](feat)` still work.
    class _IdentityIndexed:
        def __getitem__(self, _):
            return lambda x: x

    @property
    def quant_resi(self):
        return self._IdentityIndexed()


class SkelVQWrapper(nn.Module):
    """Wrap a trained SkelVQ-FSQ as a frozen tokenizer, exposing HRVQVAE's API.

    Args:
        ckpt_path: path to net_best_fid.tar (or any state_dict tar) of a SkelVQ-FSQ run.
        opt_overrides: dict to override fields of the default opt (e.g., for KIT).
        device: torch device ("cuda:0" by default).

    Attributes that match MoScale's vq_model API:
        .down_t      → unused but present (set to 0; we override the m_lens path
                       in MoScale's preprocess to use the wrapper's compute_bottleneck_lens).
        .encode      → (idx_list, q_cum_list, f_hat)
        .decode      → continuous-features → motion via SkelVQ's decoder
        .quantizer   → an alias module exposing indices_to_codes / dequantize /
                       quant_resi / scales / code_dim / effective_levels.

    Stage-2-specific extras:
        .J_b                       → joint count after STPool (7 for t2m).
        .effective_levels          → per-channel categorical size (7).
        .compute_bottleneck_lens(m_lens) → bottleneck-space lengths per sample,
                                           accounting for both time downsampling
                                           and the J_b joint flatten.
        .decode_from_indices(idx_list) → motion from per-scale FSQ indices.
    """

    def __init__(
        self,
        ckpt_path: str,
        opt_overrides: Optional[dict] = None,
        device: str = "cuda:0",
    ):
        super().__init__()
        SkelVQ, MultiScaleFSQ = _import_skelvq_with_isolated_utils()

        self.opt = _build_skelvq_opt(opt_overrides)
        self.opt.device = torch.device(device)
        self.opt.gpu_id = 0 if device.startswith("cuda") else -1
        self.opt.is_train = False

        skelvq = SkelVQ(self.opt)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt["vae"] if "vae" in ckpt else ckpt
        skelvq.load_state_dict(state)
        skelvq.to(device)
        skelvq.train(False)
        for p in skelvq.parameters():
            p.requires_grad = False
        self.skelvq = skelvq
        assert isinstance(skelvq.quantizer, MultiScaleFSQ), \
            f"SkelVQWrapper expects FSQ tokenizer; got {type(skelvq.quantizer).__name__}"

        # MoScale-shaped accessors
        self.quantizer = _MultiScaleFSQAlias(skelvq.quantizer)
        self.J_b = skelvq.J_b
        self.code_dim = skelvq.code_dim
        self.effective_levels = skelvq.quantizer.effective_levels
        self.scales = list(skelvq.quantizer.scales)
        self.n_layers = self.opt.n_layers       # 2 for SkelVQ default
        # Time-only downsample factor (matches HRVQVAE.down_t semantics, but
        # MoScale should call compute_bottleneck_lens for the full L = T_b * J_b
        # conversion since J_b is not a power-of-2 factor).
        self.down_t = self.n_layers
        self.ckpt_path = ckpt_path

    def compute_bottleneck_lens(self, m_lens: torch.Tensor) -> torch.Tensor:
        """Map original-frame motion lengths → bottleneck-space cell lengths.

        L = (m_lens // 2**n_layers) * J_b
        """
        return (m_lens // (2 ** self.n_layers)) * self.J_b

    @torch.no_grad()
    def _encode_to_grid(self, motion: torch.Tensor) -> torch.Tensor:
        """Run SkelVQ's encoder up to the pre-quantize 1D flatten.

        Returns z_1d of shape (B, D, L) where L = T_b * J_b.
        """
        x = motion.detach().float()
        h = self.skelvq.motion_enc(x)            # (B, T, J=22, D)
        h = self.skelvq.conv_enc(h)              # (B, T_b, J_b, D)
        B, T_b, J_b, D = h.shape
        z_1d = h.reshape(B, T_b * J_b, D).transpose(1, 2).contiguous()  # (B, D, L)
        return z_1d, T_b, J_b

    @torch.no_grad()
    def encode(
        self,
        motion: torch.Tensor,
        m_lens: Optional[torch.Tensor] = None,
        perturb_rate=None,
        codebook=None,
        train: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Drop-in for HRVQVAE.encode.

        Args:
            perturb_rate: optional [lo, hi] pair. When non-zero AND `train=True`,
                level indices in the residual cascade are perturbed (uniform
                Categorical resample to non-true levels) at a per-step rate
                drawn uniformly in [lo, hi]. The cascade re-quantizes with the
                perturbed codes so the propagated `q_cum` reflects corruption,
                while the returned `idx_list` stays the **clean** GT — that's
                the AR transformer's CE target. Mirrors MoScale's MSQuantizer
                perturb / Infinity's Bitwise Self-Correction.
            train: must be True for perturbation to fire.

        Returns:
            idx_list[s]:   (B, L_s, code_dim) int64,  ∈ [0, effective_levels)
                           — clean GT indices, suitable as CE targets.
            q_cum_list[s]: (B, code_dim, L)  float,   cumulative dequantized
                           after scale s. Reflects perturbation when active.
            f_hat:         (B, code_dim, L)  float,   == q_cum_list[-1]
        """
        del codebook  # unused; HRVQVAE-only knob.
        z_1d, _T_b, _J_b = self._encode_to_grid(motion)
        idx_list, _q_per_scale, q_cum_list = self.skelvq.quantizer.encode_indices(
            z_1d, perturb_rate=perturb_rate, train=train,
        )
        f_hat = q_cum_list[-1]
        return idx_list, q_cum_list, f_hat

    @torch.no_grad()
    def decode(self, z: torch.Tensor, m_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode bottleneck features back to motion.

        z: (B, D, L) flat or (B, T_b, J_b, D) structured.
        """
        if z.dim() == 3:
            B, D, L = z.shape
            assert L % self.J_b == 0, f"flat L={L} not divisible by J_b={self.J_b}"
            T_b = L // self.J_b
            z_grid = z.transpose(1, 2).reshape(B, T_b, self.J_b, D).contiguous()
        else:
            z_grid = z
        return self.skelvq.decode(z_grid)

    @torch.no_grad()
    def decode_from_indices(self, idx_list: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct motion from per-scale FSQ indices.

        idx_list[s]: (B, L_s, code_dim) int in [0, effective_levels).
        Returns motion of shape (B, T, pose_dim).
        """
        assert len(idx_list) == len(self.scales)
        # Determine full L from the finest-scale entry (scale=1).
        finest = idx_list[-1]
        B, L, D = finest.shape
        z_cum = torch.zeros(B, D, L, device=finest.device, dtype=torch.float32)
        for idx in idx_list:
            q_native = self.quantizer.indices_to_codes(idx).permute(0, 2, 1).contiguous()  # (B, D, L_s)
            if q_native.shape[-1] != L:
                q_up = F.interpolate(q_native, size=L, mode="linear", align_corners=False)
            else:
                q_up = q_native
            z_cum = z_cum + q_up
        return self.decode(z_cum)
