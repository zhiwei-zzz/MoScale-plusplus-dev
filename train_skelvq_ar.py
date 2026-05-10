"""Train MoScaleFSQ — text-to-motion AR over the SkelVQ-FSQ tokenizer.

Pure next-scale AR (no BERT-style mask augmentation; see moscale_fsq.py for the
diff vs upstream MoScale). Uses SALAD's data pipeline + the trained SkelVQ-FSQ
checkpoint from stage 1.

Run:
    # smoke test
    python train_skelvq_ar.py --name skelvq_ar_smoke --batch_size 4 \
        --max_epoch 1 --num_workers 0 --log_every 5

    # full text-to-motion training
    python train_skelvq_ar.py --name skelvq_ar_v1 --batch_size 32 \
        --max_epoch 200 --use_wandb
"""
from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict, defaultdict
from os.path import join as pjoin

# Make moscale/ importable.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "moscale"))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.t2m_dataset import MotionDataset
from data.t2m_caption_dataset import Text2MotionWindowDataset, collate_fn as caption_collate
from utils.get_opt import get_opt
from utils.utils import print_current_loss
from models.vae.wandb_helper import make_logger

from options.skelvq_ar_option import arg_parse

from model.vq.skelvq_wrapper import SkelVQWrapper             # type: ignore
from model.transformer.moscale_fsq import MoScaleFSQ          # type: ignore


def def_value():
    return 0.0


def build_dataset(opt):
    wrapper_opt = get_opt(opt.dataset_opt_path, opt.device)
    mean = np.load(pjoin(wrapper_opt.meta_dir, "mean.npy"))
    std = np.load(pjoin(wrapper_opt.meta_dir, "std.npy"))
    train_split = pjoin(opt.data_root, "train.txt")
    val_split = pjoin(opt.data_root, "val.txt")
    if opt.text_cond:
        train_ds = Text2MotionWindowDataset(opt, mean, std, train_split)
        val_ds = Text2MotionWindowDataset(opt, mean, std, val_split)
    else:
        train_ds = MotionDataset(opt, mean, std, train_split)
        val_ds = MotionDataset(opt, mean, std, val_split)
    return train_ds, val_ds


def _unpack_batch(batch, text_cond: bool):
    """Returns (texts_or_dummy_strs, motion_tensor)."""
    if text_cond:
        captions, motion = batch
        return list(captions), motion
    # Unconditional path: feed empty strings (model will replace with cfg_uncond
    # at the configured drop probability).
    return [""] * batch.shape[0], batch


def build_cfg(opt) -> OmegaConf:
    """Synthesize the OmegaConf config object MoScaleFSQ expects."""
    return OmegaConf.create({
        "model": dict(
            latent_dim=opt.latent_dim, head_latent_dim=opt.latent_dim,
            num_layers=opt.num_layers, n_heads=opt.num_heads,
            mlp_ratio=opt.mlp_ratio, dropout=opt.dropout, attn_drop_rate=0.0,
            use_crossattn=True, attn_l2_norm=False, infer_use_kvcache=False,
        ),
        "text_embedder": dict(dim_embed=768, version=opt.t5_model),
        "data": dict(dim_pose=opt.pose_dim, max_motion_length=opt.window_size,
                     max_text_length=opt.t5_max_len),
        "training": dict(
            perturb_rate=[0.0, 0.0],
            cond_drop_prob=opt.cond_drop_prob,
            sample_level_times=[1, 1, 1, 1],
        ),
        "vq": dict(code_dim=32, scales=[8, 4, 2, 1]),
        "exp": dict(seed=opt.seed),
    })


def run_validation(ar, vq_model, val_loader, device, text_cond, max_batches=20):
    ar.train(False)
    losses, accs = [], []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            texts, motion = _unpack_batch(batch, text_cond)
            motion = motion.to(device, dtype=torch.float32)
            m_lens = torch.full((motion.shape[0],), motion.shape[1], device=device, dtype=torch.long)
            loss, pred_idx, acc = ar.forward(motion, texts, m_lens.clone(), vq_model, train=False)
            losses.append(loss.item())
            accs.append(acc)
    ar.train(True)
    return float(np.mean(losses)) if losses else 0.0, float(np.mean(accs)) if accs else 0.0


def main():
    opt = arg_parse(is_train=True)
    torch.manual_seed(opt.seed)

    # ----- data
    train_ds, val_ds = build_dataset(opt)
    collate = caption_collate if opt.text_cond else None
    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, drop_last=True, pin_memory=True,
                              collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, drop_last=True, pin_memory=True,
                            collate_fn=collate)

    # ----- frozen tokenizer
    print(f"loading tokenizer ckpt: {opt.tokenizer_ckpt}")
    vq_model = SkelVQWrapper(opt.tokenizer_ckpt, device=str(opt.device))
    print(f"  J_b={vq_model.J_b}  effective_levels={vq_model.effective_levels}  scales={vq_model.scales}")

    # ----- AR
    cfg = build_cfg(opt)
    full_length = (opt.window_size // (2 ** vq_model.n_layers)) * vq_model.J_b
    ar = MoScaleFSQ(
        code_dim=vq_model.code_dim,
        latent_dim=opt.latent_dim, num_heads=opt.num_heads, dropout=opt.dropout,
        text_dim=cfg.text_embedder.dim_embed, cond_drop_prob=opt.cond_drop_prob,
        mlp_ratio=opt.mlp_ratio, device=opt.device, cfg=cfg,
        full_length=full_length, scales=list(vq_model.scales),
        effective_levels=vq_model.effective_levels,
    ).to(opt.device)
    n_params = sum(p.numel() for p in ar.parameters() if p.requires_grad)
    print(f"AR trainable params: {n_params/1e6:.2f}M")

    # ----- optim
    optim = torch.optim.AdamW(ar.parameters(), lr=opt.lr, betas=(0.9, 0.99),
                              weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=opt.milestones, gamma=opt.gamma)

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
        for i, batch in enumerate(train_loader):
            it += 1
            if it < opt.warm_up_iter:
                cur_lr = opt.lr * (it + 1) / (opt.warm_up_iter + 1)
                for g in optim.param_groups:
                    g["lr"] = cur_lr

            texts, motion = _unpack_batch(batch, opt.text_cond)
            motion = motion.to(opt.device, dtype=torch.float32)
            m_lens = torch.full((motion.shape[0],), motion.shape[1], device=opt.device, dtype=torch.long)

            optim.zero_grad()
            loss, _pred_idx, acc = ar.forward(motion, texts, m_lens.clone(), vq_model, train=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(ar.parameters(), opt.grad_clip)
            optim.step()

            if it >= opt.warm_up_iter:
                scheduler.step()

            logs["loss"] += loss.item()
            logs["lr"] += optim.param_groups[0]["lr"]
            logs["grad_norm"] += float(grad_norm)
            logs["acc"] += acc

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
                            "epoch": epoch, "total_iter": it,
                            "opt": vars(opt)},
                           pjoin(opt.model_dir, "latest.tar"))

        torch.save({"ar": ar.state_dict(),
                    "optim": optim.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1, "total_iter": it,
                    "opt": vars(opt)},
                   pjoin(opt.model_dir, "latest.tar"))

        epoch += 1
        if epoch % opt.eval_every_e == 0:
            val_loss, val_acc = run_validation(ar, vq_model, val_loader, opt.device, opt.text_cond)
            print(f"[val] ep {epoch}: loss {val_loss:.4f}  acc {val_acc:.4f}")
            logger.add_scalar("Val/loss", val_loss, it)
            logger.add_scalar("Val/acc", val_acc, it)
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"ar": ar.state_dict(), "epoch": epoch, "val_loss": val_loss,
                            "opt": vars(opt)},
                           pjoin(opt.model_dir, "net_best_loss.tar"))
                print(f"  new best val loss {val_loss:.4f}")


if __name__ == "__main__":
    main()
