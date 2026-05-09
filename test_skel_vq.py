"""Test SkelVQ - fork of test_vae.py."""
import numpy as np
import os
from os.path import join as pjoin
from torch.utils.data import DataLoader

from models.vae.skel_vq_trainer import SkelVQTrainer
from options.skel_vq_option import arg_parse
from utils.fixseed import fixseed

import torch
from utils.get_opt import get_opt
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.vae.skel_vq import SkelVQ


def load_skel_vq(opt, filename):
    model = SkelVQ(opt)
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', filename),
                      map_location='cpu')
    # SkelVQTrainer saves under 'vae' key (mirrors VAETrainer for consistency).
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    print(f'Loaded SkelVQ {filename}')
    return model


if __name__ == "__main__":
    opt = arg_parse(is_train=False)
    fixseed(opt.seed)

    dataset_opt_path = f"checkpoints/{opt.dataset_name}/Comp_v6_KLD005/opt.txt"
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=opt.device)

    skel = load_skel_vq(opt, "net_best_fid.tar").to(opt.device)

    trainer = SkelVQTrainer(opt, skel)
    trainer.test(eval_wrapper, eval_val_loader, 20,
                 save_dir=pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'eval'),
                 cal_mm=False)
