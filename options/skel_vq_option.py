"""SkelVQ options - fork of options/vae_option.py.

Key differences vs vae_option.py:
  - --lambda_kl removed (no KL term)
  - --lambda_commit added (default 0.02 to mirror SALAD's lambda_kl)
  - --code_dim, --nb_code, --scales added for the residual VQ
  - --mu, --share_quant_resi, --quant_resi (MoScale MSQuantizer hyperparams)
"""
import argparse
import os
import torch
from os.path import join as pjoin
from utils import paramUtil


def arg_parse(is_train=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # basic setup
    parser.add_argument("--name", type=str, default="skel_vq_default", help="Name of this trial")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")

    # dataloader
    parser.add_argument("--dataset_name", type=str, default="t2m", choices=["t2m", "kit"])
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    # optimization (mirrors SALAD VAE)
    parser.add_argument("--max_epoch", default=50, type=int)
    parser.add_argument("--warm_up_iter", default=2000, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--milestones", default=[150_000, 250_000], nargs="+", type=int)
    parser.add_argument("--gamma", default=0.05, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)

    parser.add_argument("--recon_loss", type=str, default="l1_smooth")
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_pos", type=float, default=0.5)
    parser.add_argument("--lambda_vel", type=float, default=0.5)
    parser.add_argument("--lambda_commit", type=float, default=0.02, help="commitment loss weight (mirrors SALAD's lambda_kl=0.02)")

    # encoder/decoder arch (mirrors SALAD VAE)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_extra_layers", type=int, default=1)
    parser.add_argument("--norm", type=str, default="none", choices=["none", "batch", "layer"])
    parser.add_argument("--activation", type=str, default="gelu", choices=["relu", "silu", "gelu"])
    parser.add_argument("--dropout", type=float, default=0.1)

    # residual VQ (MoScale defaults)
    parser.add_argument("--quantizer_type", type=str, default="msvq",
                        choices=["msvq", "bsq", "fsq"],
                        help="msvq = MoScale residual VQ with codebook; bsq = Infinity-style "
                             "binary spherical quantization (bitwise / LFQ); "
                             "fsq = Mentzer-style finite scalar quantization with multiple "
                             "levels per channel.")
    parser.add_argument("--code_dim", type=int, default=32, help="codebook embedding dim (must equal latent_dim)")
    parser.add_argument("--nb_code", type=int, default=512, help="codebook size per scale (msvq only)")
    parser.add_argument("--scales", type=int, nargs="+", default=[8, 4, 2, 1], help="residual VQ scales")
    parser.add_argument("--mu", type=float, default=0.99, help="EMA decay for codebook (msvq only)")
    parser.add_argument("--share_quant_resi", type=int, default=4, help="msvq only")
    parser.add_argument("--quant_resi", type=float, default=0.0, help="msvq only")
    parser.add_argument("--start_drop", type=int, default=-1, help="msvq only")
    parser.add_argument("--quantize_dropout_prob", type=float, default=0.0, help="msvq only")
    # BSQ-specific knobs
    parser.add_argument("--inv_temperature", type=float, default=100.0, help="bsq: inverse temp in entropy term")
    parser.add_argument("--entropy_weight", type=float, default=0.1, help="bsq: weight on entropy regularizer")
    parser.add_argument("--zeta", type=float, default=1.0, help="bsq: scaling on entropy term")
    # FSQ-specific knobs
    parser.add_argument("--fsq_levels", type=int, default=8,
                        help="fsq: number of discrete levels per channel (8 = 3 bits/channel)")
    parser.add_argument("--fsq_inv_temperature", type=float, default=20.0,
                        help="fsq: inverse temperature for the soft-assignment entropy regularizer")
    parser.add_argument("--fsq_entropy_weight", type=float, default=0.0,
                        help="fsq: weight on entropy regularizer (default 0 = pure FSQ; "
                             "set ~0.1 to enable LFQ-style level-usage regularization)")
    parser.add_argument("--fsq_zeta", type=float, default=1.0,
                        help="fsq: scaling on the entropy term (mirrors BSQ)")

    # other
    parser.add_argument("--is_continue", action="store_true")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--save_latest", default=500, type=int)
    parser.add_argument("--eval_every_e", default=1, type=int)

    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu_id)
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    args = vars(opt)

    if opt.dataset_name == "t2m":
        opt.data_root = './dataset/humanml3d/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.pose_dim = 263
        opt.contact_joints = [7, 10, 8, 11]
        opt.fps = 20
        opt.radius = 4
        opt.kinematic_chain = paramUtil.t2m_kinematic_chain
        opt.dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    elif opt.dataset_name == "kit":
        opt.data_root = './dataset/kit-ml/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        opt.pose_dim = 251
        opt.contact_joints = [19, 20, 14, 15]
        opt.fps = 12.5
        opt.radius = 240 * 8
        opt.kinematic_chain = paramUtil.kit_kinematic_chain
        opt.dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'
    else:
        raise KeyError('Dataset Does not Exists')

    if opt.code_dim != opt.latent_dim:
        print(f"[skel_vq_option] WARN: code_dim={opt.code_dim} != latent_dim={opt.latent_dim}; forcing equality.")
        opt.code_dim = opt.latent_dim

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    opt.is_train = is_train
    if is_train:
        expr_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

    return opt
