"""SkelVQ: same encoder/decoder as SALAD's VAE, but with MoScale-style residual VQ
at the bottleneck instead of a Gaussian (mu, log_var) + KL.

The encode path:
    motion_enc -> conv_enc -> joint-pool to 1D -> MSQuantizer -> broadcast back to joints
The decode path is unchanged from SALAD's VAE:
    conv_dec -> motion_dec
Loss in the trainer becomes recon + lambda_pos*pos + lambda_vel*vel + lambda_commit*commit_loss
(with KL dropped).

Phase-1 design: option (a) - joint-pool before VQ, broadcast back after. Reuses
MoScale's MSQuantizer unchanged.
"""
import torch
import torch.nn as nn

from models.vae.encdec import (
    MotionEncoder, MotionDecoder, STConvEncoder, STConvDecoder,
)
from models.vae.quantizer import MSQuantizer
from models.vae.bsq import MultiScaleBSQ, MultiScaleFSQ


class SkelVQ(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        # Same encoder/decoder as SALAD's VAE.
        self.motion_enc = MotionEncoder(opt)
        self.motion_dec = MotionDecoder(opt)
        self.conv_enc = STConvEncoder(opt)
        self.conv_dec = STConvDecoder(opt, self.conv_enc)

        # Bottleneck swap.
        assert opt.code_dim == opt.latent_dim, (
            f"prototype assumes code_dim ({opt.code_dim}) == latent_dim ({opt.latent_dim})"
        )
        self.code_dim = opt.code_dim
        self.scales = list(opt.scales)
        self.quantizer_type = getattr(opt, "quantizer_type", "msvq")
        if self.quantizer_type == "msvq":
            self.quantizer = MSQuantizer(
                nb_code=opt.nb_code,
                code_dim=opt.code_dim,
                mu=opt.mu,
                scales=self.scales,
                share_quant_resi=opt.share_quant_resi,
                quant_resi=opt.quant_resi,
            )
        elif self.quantizer_type == "bsq":
            self.quantizer = MultiScaleBSQ(
                code_dim=opt.code_dim,
                scales=self.scales,
                inv_temperature=getattr(opt, "inv_temperature", 100.0),
                zeta=getattr(opt, "zeta", 1.0),
                entropy_weight=getattr(opt, "entropy_weight", 0.1),
                use_decay_factor=False,
            )
        elif self.quantizer_type == "fsq":
            self.quantizer = MultiScaleFSQ(
                code_dim=opt.code_dim,
                scales=self.scales,
                levels=getattr(opt, "fsq_levels", 8),
                use_decay_factor=False,
                inv_temperature=getattr(opt, "fsq_inv_temperature", 20.0),
                entropy_weight=getattr(opt, "fsq_entropy_weight", 0.0),
                zeta=getattr(opt, "fsq_zeta", 1.0),
            )
        else:
            raise ValueError(f"Unknown quantizer_type: {self.quantizer_type}")

        # Probe the bottleneck's joint count once at construct time.
        self.J_b = self._probe_joint_count()

    def _probe_joint_count(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 16, self.opt.pose_dim)  # T must be >= 2**n_layers for STPool
            h = self.motion_enc(dummy)
            h = self.conv_enc(h)
        return h.shape[2]

    def freeze(self):
        self.train(False)
        for p in self.parameters():
            p.requires_grad = False

    def encode(self, x, m_lens=None):
        """
        x: (B, T, pose_dim).
        Returns z_q (B, T_b, J_b, D), loss_dict {commit, perplexity}.

        Option (b): per-cell VQ on the (T_b, J_b) grid - no joint pooling.
        We flatten (T_b, J_b) -> 1D sequence of length T_b * J_b, quantize each cell
        independently with a shared codebook, then reshape back.
        """
        x = x.detach().float()
        h = self.motion_enc(x)            # (B, T, J=22, D)
        h = self.conv_enc(h)              # (B, T_b, J_b, D)
        B, T_b, J_b, D = h.shape
        L = T_b * J_b                     # number of cells per clip

        # Reshape to (B, D, T_b*J_b) - joints major within each timestep, then advance time.
        # This mirrors VAR's raster scan and gives the residual VQ a single 1D axis.
        z_1d = h.reshape(B, T_b * J_b, D).transpose(1, 2).contiguous()  # (B, D, L)

        # MSQuantizer.forward requires m_lens. With fixed-length training clips no padding
        # is needed, so all L cells are valid.
        if m_lens is None:
            m_lens_q = torch.full((B,), L, dtype=torch.long, device=x.device)
        else:
            # m_lens at the input resolution -> bottleneck cell count.
            m_lens_q = (m_lens // (2 ** self.opt.n_layers)) * J_b
            m_lens_q = m_lens_q.clamp(min=1, max=L).long()

        z_q, q_loss, q_diag = self.quantizer(
            z_1d,
            temperature=0.5,
            m_lens=m_lens_q,
            start_drop=getattr(self.opt, "start_drop", -1),
            quantize_dropout_prob=getattr(self.opt, "quantize_dropout_prob", 0.0),
        )                                  # (B, D, L)

        # Reshape back to (B, T_b, J_b, D) so the decoder sees the structured grid.
        z_q_per_joint = z_q.transpose(1, 2).reshape(B, T_b, J_b, D).contiguous()

        if self.quantizer_type == "msvq":
            # q_loss is unweighted commit loss; q_diag is perplexity.
            return z_q_per_joint, {"loss_commit": q_loss, "perplexity": q_diag}
        elif self.quantizer_type == "bsq":
            # BSQ: q_loss is already the entropy regularizer (zeta * entropy * weight / inv_t).
            # q_diag is bit_balance (~0.5 = healthy).
            return z_q_per_joint, {"loss_entropy": q_loss, "bit_balance": q_diag}
        else:
            # FSQ: q_loss is zero (no aux loss). q_diag is mean abs value diagnostic.
            return z_q_per_joint, {"loss_quant": q_loss, "fsq_diag": q_diag}

    def decode(self, z):
        z = self.conv_dec(z)
        z = self.motion_dec(z)
        return z

    def forward(self, x, m_lens=None):
        """Mirror VAE.forward signature: returns (out, loss_dict)."""
        z, loss_dict = self.encode(x, m_lens=m_lens)
        out = self.decode(z)
        return out, loss_dict
