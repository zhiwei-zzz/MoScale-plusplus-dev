"""CLI options for unconditional next-scale AR training over SkelVQ-FSQ tokens.

Mirrors options/skel_vq_option.py's structure so users can step between the
two scripts without surprises. Text conditioning + CFG come later — see
STAGE2_STATUS.md.
"""
import argparse
import os
import torch
from os.path import join as pjoin
from utils import paramUtil


def arg_parse(is_train=False):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # basic setup
    p.add_argument("--name", type=str, default="skelvq_ar_default")
    p.add_argument("--seed", default=1234, type=int)
    p.add_argument("--gpu_id", type=int, default=0)

    # dataloader
    p.add_argument("--dataset_name", type=str, default="t2m", choices=["t2m", "kit"])
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--window_size", type=int, default=64,
                   help="Match the tokenizer's training window. AR sequence length "
                        "= sum_s (L/scale_s) where L = window/4 * J_b.")
    p.add_argument("--num_workers", type=int, default=4)

    # tokenizer
    p.add_argument("--tokenizer_ckpt", type=str,
                   default="./checkpoints/t2m/skelvq_fsq/model/net_best_fid.tar",
                   help="Frozen SkelVQ-FSQ checkpoint.")

    # AR transformer
    p.add_argument("--latent_dim", type=int, default=384, help="transformer hidden dim")
    p.add_argument("--head_latent_dim", type=int, default=-1,
                   help="output-head hidden dim. -1 (default) means follow --latent_dim.")
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use_lvl_embed", action="store_true", default=True)

    # text conditioning + classifier-free guidance
    p.add_argument("--text_cond", action="store_true", default=True,
                   help="Use T5 cross-attention. Default on. Pass --no_text_cond for unconditional.")
    p.add_argument("--no_text_cond", dest="text_cond", action="store_false")
    p.add_argument("--t5_model", type=str, default="google/t5-v1_1-base")
    p.add_argument("--t5_max_len", type=int, default=20)
    p.add_argument("--cond_drop_prob", type=float, default=0.1,
                   help="Probability of replacing text condition with cfg_uncond at training time.")

    # level perturbation — MoScale's perturb_rate / Infinity's BSC, FSQ-flavored.
    # When `hi > 0` AND model is in train mode, at each residual scale a
    # per-step rate is sampled uniformly in [lo, hi]; with that probability
    # each (cell, channel) GT level index is replaced with a uniform
    # Categorical resample over {0..effective_levels-2}+1@true (non-true
    # levels). The cascade re-quantizes with the perturbed codes (Infinity's
    # `noise_apply_requant=1` semantic) so the propagated f_hat reflects
    # corruption; CE targets stay the clean GT indices. Defaults match
    # MoScale's HRVQVAE setting.
    p.add_argument("--perturb_lo", type=float, default=0.0,
                   help="lower bound of the per-step perturbation rate")
    p.add_argument("--perturb_hi", type=float, default=0.6,
                   help="upper bound of the per-step perturbation rate. Set 0 to disable.")

    # (legacy LSC knobs, unused by MoScaleFSQ but kept for the minimal
    # SkelVQAR backbone)
    p.add_argument("--noise_apply_layers", type=int, default=-1)
    p.add_argument("--noise_apply_strength", type=float, default=0.3)
    p.add_argument("--noise_apply_requant", action="store_true", default=True)

    # optimization
    p.add_argument("--max_epoch", default=200, type=int)
    p.add_argument("--warm_up_iter", default=2000, type=int)
    p.add_argument("--lr", default=3e-4, type=float)
    p.add_argument("--weight_decay", default=0.05, type=float)
    p.add_argument("--grad_clip", default=2.0, type=float)
    p.add_argument("--milestones", default=[150_000, 250_000], nargs="+", type=int)
    p.add_argument("--gamma", default=0.05, type=float)

    # logging / checkpoints
    p.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    p.add_argument("--log_every", default=10, type=int)
    p.add_argument("--save_latest", default=500, type=int)
    p.add_argument("--eval_every_e", default=1, type=int,
                   help="Cheap CE+acc validation cadence (epochs).")
    p.add_argument("--is_continue", action="store_true")

    # periodic generation-quality eval (FID + R-precision + Diversity)
    p.add_argument("--fid_every_e", default=4, type=int,
                   help="Run a SALAD-evaluator generation-quality eval every N epochs. "
                        "Set -1 to disable. Default 4. ~30-60s per repeat on a 3090.")
    p.add_argument("--fid_repeat_times", default=1, type=int,
                   help="Repeat count for the in-training gen eval. Use 1 (default) for "
                        "cheap; 20 for the full paper protocol (or use eval_skelvq_ar.py "
                        "post-training for that).")
    p.add_argument("--fid_cond_scale", default=4.0, type=float,
                   help="Classifier-free guidance scale at gen-eval time.")
    p.add_argument("--fid_top_p", default=0.9, type=float)
    p.add_argument("--fid_temperature", default=1.0, type=float)
    p.add_argument("--glove_dir", default="./glove",
                   help="GloVe embeddings dir (downloaded by prepare/download_glove.sh).")

    # wandb (optional; mirrors TensorBoard scalars when enabled)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="moscale-plusplus")
    p.add_argument("--wandb_entity", type=str, default="zhiwei-z-org")
    p.add_argument("--wandb_team", type=str, default="zhiwei-z")
    p.add_argument("--wandb_run_name", type=str, default="")

    opt = p.parse_args()
    torch.cuda.set_device(opt.gpu_id)
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else f"cuda:{opt.gpu_id}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, "model")
    opt.meta_dir = pjoin(opt.save_root, "meta")
    opt.log_dir = pjoin("./log", opt.dataset_name, opt.name)
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == "t2m":
        opt.data_root = "./dataset/humanml3d/"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.text_dir = pjoin(opt.data_root, "texts")
        opt.joints_num = 22
        opt.pose_dim = 263
        opt.fps = 20
        opt.radius = 4
        opt.kinematic_chain = paramUtil.t2m_kinematic_chain
        opt.dataset_opt_path = "./checkpoints/t2m/Comp_v6_KLD005/opt.txt"
    elif opt.dataset_name == "kit":
        opt.data_root = "./dataset/kit-ml/"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.text_dir = pjoin(opt.data_root, "texts")
        opt.joints_num = 21
        opt.pose_dim = 251
        opt.fps = 12.5
        opt.radius = 240 * 8
        opt.kinematic_chain = paramUtil.kit_kinematic_chain
        opt.dataset_opt_path = "./checkpoints/kit/Comp_v6_KLD005/opt.txt"
    else:
        raise KeyError(f"Unknown dataset {opt.dataset_name}")

    opt.is_train = is_train
    if is_train:
        with open(pjoin(opt.save_root, "opt.txt"), "wt") as f:
            f.write("------------ Options -------------\n")
            for k, v in sorted(vars(opt).items()):
                f.write(f"{k}: {v}\n")
            f.write("-------------- End ----------------\n")

    print("------------ Options -------------")
    for k, v in sorted(vars(opt).items()):
        print(f"{k}: {v}")
    print("-------------- End ----------------")
    return opt
