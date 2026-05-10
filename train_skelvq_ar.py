"""Unconditional next-scale AR training over SkelVQ-FSQ tokens.

Path A from STAGE2_STATUS.md: the smallest end-to-end thing that's actually a
training run. No text conditioning, no CFG, no inference decoder yet — just
verify the AR can fit p(motion) over HumanML3D given the frozen tokenizer.

Reuses SALAD's MotionDataset (no captions, fixed-window slicing, same Z
normalization the tokenizer was trained against), so there are no
distribution-shift concerns for the encoder side.

Run:
    python train_skelvq_ar.py --name skelvq_ar_smoke --batch_size 8 --max_epoch 1
    # or full:
    python train_skelvq_ar.py --name skelvq_ar_v0 --batch_size 64 --max_epoch 200 \
        --noise_apply_layers 3 --noise_apply_strength 0.3 \
        --use_wandb
"""
from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict, defaultdict
from os.path import join as pjoin

# Make moscale/ importable so we can grab SkelVQAR / LSC / wrapper.
# IMPORTANT: append, don't insert(0). Both repos have a `utils/` package; we
# need SALAD's to win (`utils.get_opt` etc.) while moscale's `model/` (singular,
# unique to moscale) imports cleanly via fall-through. SkelVQWrapper's internal
# isolation will swap in SALAD's `utils.skeleton` when loading SkelVQ.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "moscale"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.t2m_dataset import MotionDataset
from utils.get_opt import get_opt
from utils.utils import print_current_loss
from models.vae.wandb_helper import make_logger

from options.skelvq_ar_option import arg_parse

# moscale/ side
from model.vq.skelvq_wrapper import SkelVQWrapper             # type: ignore
from model.level_self_correction import LevelSelfCorrection   # type: ignore
from model.transformer.skelvq_ar import SkelVQAR              # type: ignore


def def_value():
    return 0.0


def build_dataset(opt):
    """Reuse SALAD's MotionDataset, normalized with the same evaluator stats
    the tokenizer was trained against."""
    wrapper_opt = get_opt(opt.dataset_opt_path, opt.device)
    mean = np.load(pjoin(wrapper_opt.meta_dir, "mean.npy"))
    std = np.load(pjoin(wrapper_opt.meta_dir, "std.npy"))
    train_split = pjoin(opt.data_root, "train.txt")
    val_split = pjoin(opt.data_root, "val.txt")
    train_ds = MotionDataset(opt, mean, std, train_split)
    val_ds = MotionDataset(opt, mean, std, val_split)
    return train_ds, val_ds


def run_validation(ar, vq_model, lsc, val_loader, device, max_batches=20):
    ar.train(False)
    losses = []
    accs = []
    with torch.no_grad():
        for i, motion in enumerate(val_loader):
            if i >= max_batches:
                break
            motion = motion.to(device, dtype=torch.float32)
            z_1d, _, _ = vq_model._encode_to_grid(motion)
            gt_idx, x_cond, _ = lsc.perturb_requant(z_1d)
            loss, info = ar.forward(gt_idx, x_cond)
            losses.append(loss.item())
            accs.append(info["acc_per_channel"])
    ar.train(True)
    return float(np.mean(losses)) if losses else 0.0, float(np.mean(accs)) if accs else 0.0


def main():
    opt = arg_parse(is_train=True)
    torch.manual_seed(opt.seed)

    # ----- data
    train_ds, val_ds = build_dataset(opt)
    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, drop_last=True, pin_memory=True)

    # ----- frozen tokenizer
    print(f"loading tokenizer ckpt: {opt.tokenizer_ckpt}")
    vq_model = SkelVQWrapper(opt.tokenizer_ckpt, device=str(opt.device))
    print(f"  J_b={vq_model.J_b}  effective_levels={vq_model.effective_levels}  scales={vq_model.scales}")

    # ----- LSC + AR
    lsc = LevelSelfCorrection(
        vq_model,
        noise_apply_layers=opt.noise_apply_layers,
        noise_apply_strength=opt.noise_apply_strength,
        noise_apply_requant=opt.noise_apply_requant,
    )
    ar = SkelVQAR(
        vq_model,
        latent_dim=opt.latent_dim,
        num_layers=opt.num_layers,
        num_heads=opt.num_heads,
        mlp_ratio=opt.mlp_ratio,
        dropout=opt.dropout,
        use_lvl_embed=opt.use_lvl_embed,
    ).to(opt.device)
    n_params = sum(p.numel() for p in ar.parameters() if p.requires_grad)
    print(f"AR params: {n_params/1e6:.2f}M")

    # ----- optim
    optim = torch.optim.AdamW(
        ar.parameters(), lr=opt.lr, betas=(0.9, 0.99),
        weight_decay=opt.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=opt.milestones, gamma=opt.gamma,
    )

    # ----- logging
    logger = make_logger(opt)
    if opt.is_continue:
        ckpt = torch.load(pjoin(opt.model_dir, "latest.tar"),
                          map_location=opt.device, weights_only=False)
        ar.load_state_dict(ckpt["ar"])
        optim.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])
        epoch = ckpt["epoch"]
        it = ckpt["total_iter"]
        print(f"resumed at epoch {epoch}, iter {it}")
    else:
        epoch = 0
        it = 0

    start_time = time.time()
    total_iters = opt.max_epoch * len(train_loader)
    print(f"Total epochs: {opt.max_epoch}  iters/epoch: {len(train_loader)}  total iters: {total_iters}")

    best_val = float("inf")
    logs = defaultdict(def_value, OrderedDict())

    # ----- train loop
    while epoch < opt.max_epoch:
        ar.train(True)
        for i, motion in enumerate(train_loader):
            it += 1
            # warm-up LR
            if it < opt.warm_up_iter:
                cur_lr = opt.lr * (it + 1) / (opt.warm_up_iter + 1)
                for g in optim.param_groups:
                    g["lr"] = cur_lr

            motion = motion.to(opt.device, dtype=torch.float32)
            with torch.no_grad():
                z_1d, _, _ = vq_model._encode_to_grid(motion)
                gt_idx, x_cond, _ = lsc.perturb_requant(z_1d)

            optim.zero_grad()
            loss, info = ar.forward(gt_idx, x_cond)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(ar.parameters(), opt.grad_clip)
            optim.step()

            if it >= opt.warm_up_iter:
                scheduler.step()

            logs["loss"] += loss.item()
            logs["lr"] += optim.param_groups[0]["lr"]
            logs["grad_norm"] += float(grad_norm)
            logs["acc_per_channel"] += info["acc_per_channel"]
            for k, v in info.items():
                if k.startswith("acc_scale_"):
                    logs[k] += v

            if it % opt.log_every == 0:
                mean_loss = OrderedDict()
                for k, v in logs.items():
                    avg = v / opt.log_every
                    logger.add_scalar(f"Train/{k}", avg, it)
                    mean_loss[k] = avg
                logs = defaultdict(def_value, OrderedDict())
                print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

            if it % opt.save_latest == 0:
                torch.save({"ar": ar.state_dict(),
                            "optim": optim.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "epoch": epoch, "total_iter": it},
                           pjoin(opt.model_dir, "latest.tar"))

        torch.save({"ar": ar.state_dict(),
                    "optim": optim.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1, "total_iter": it},
                   pjoin(opt.model_dir, "latest.tar"))

        epoch += 1
        if epoch % opt.eval_every_e == 0:
            val_loss, val_acc = run_validation(ar, vq_model, lsc, val_loader, opt.device)
            print(f"[val] ep {epoch}: loss {val_loss:.4f}  acc {val_acc:.4f}")
            logger.add_scalar("Val/loss", val_loss, it)
            logger.add_scalar("Val/acc_per_channel", val_acc, it)
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"ar": ar.state_dict(), "epoch": epoch, "val_loss": val_loss},
                           pjoin(opt.model_dir, "net_best_loss.tar"))
                print(f"  new best val loss {val_loss:.4f}")


if __name__ == "__main__":
    main()
