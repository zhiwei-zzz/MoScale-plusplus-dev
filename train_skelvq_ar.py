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

from data.t2m_dataset import (
    MotionDataset, Text2MotionDataset, Text2MotionDatasetEval,
    collate_fn as eval_collate,
)
from data.t2m_caption_dataset import (
    Text2MotionWindowDataset,
    collate_fn as caption_collate,
)
from utils.get_opt import get_opt
from utils.utils import print_current_loss
from utils.skelvq_ar_eval import make_gen_func, evaluate_once, aggregate_repeats
from models.vae.wandb_helper import make_logger
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer

from options.skelvq_ar_option import arg_parse

from model.vq.skelvq_wrapper import SkelVQWrapper             # type: ignore
from model.transformer.moscale_fsq import MoScaleFSQ          # type: ignore


def def_value():
    return 0.0


def build_gen_eval(opt, vq_model_device):
    """Set up the SALAD generation-quality evaluator bundle. Heavy (loads
    SALAD's evaluator weights + GloVe + a separate Text2MotionDatasetEval),
    so we instantiate once and reuse every fid_every_e epochs.

    Returns: (eval_loader, eval_wrapper) or None if --fid_every_e <= 0.
    """
    if opt.fid_every_e <= 0:
        return None
    print(f"Setting up generation-quality evaluator (fid_every_e={opt.fid_every_e})")
    wrapper_opt = get_opt(opt.dataset_opt_path, vq_model_device)
    mean = np.load(pjoin(wrapper_opt.meta_dir, "mean.npy"))
    std = np.load(pjoin(wrapper_opt.meta_dir, "std.npy"))
    val_split = pjoin(opt.data_root, "val.txt")
    w_vec = WordVectorizer(opt.glove_dir, "our_vab")
    eval_ds = Text2MotionDatasetEval(wrapper_opt, mean, std, val_split, w_vec)
    # collate_fn sorts by sent_len desc — required by the SALAD/MoScale evaluator's
    # pack_padded_sequence which uses the default enforce_sorted=True. Mirrors how
    # both upstream projects build their eval DataLoaders (motion_loaders/dataset_motion_loader.py
    # and moscale/model/evaluator/hml/dataset_motion_loader.py).
    eval_loader = DataLoader(eval_ds, batch_size=opt.batch_size, shuffle=False,
                             num_workers=opt.num_workers, drop_last=False, pin_memory=True,
                             collate_fn=eval_collate)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    print(f"  eval set: {len(eval_ds)} samples")
    return eval_loader, eval_wrapper


def build_dataset(opt):
    wrapper_opt = get_opt(opt.dataset_opt_path, opt.device)
    mean = np.load(pjoin(wrapper_opt.meta_dir, "mean.npy"))
    std = np.load(pjoin(wrapper_opt.meta_dir, "std.npy"))
    train_split = pjoin(opt.data_root, "train.txt")
    val_split = pjoin(opt.data_root, "val.txt")
    if opt.text_cond:
        if getattr(opt, "variable_length", True):
            # SALAD's stage-2 (denoiser) training data class. Variable-length up
            # to opt.max_motion_length, zero-padded, with the "single/double"
            # unit_length coin flip for mild length randomization. Returns
            # (caption, padded_motion, m_length) — same shape our caption_collate
            # already handles.
            train_ds = Text2MotionDataset(opt, mean, std, train_split)
            val_ds = Text2MotionDataset(opt, mean, std, val_split)
        else:
            train_ds = Text2MotionWindowDataset(opt, mean, std, train_split)
            val_ds = Text2MotionWindowDataset(opt, mean, std, val_split)
    else:
        train_ds = MotionDataset(opt, mean, std, train_split)
        val_ds = MotionDataset(opt, mean, std, val_split)
    return train_ds, val_ds


def _unpack_batch(batch, text_cond: bool):
    """Normalize the batch into (texts, motion, m_lengths_or_None)."""
    if text_cond:
        if len(batch) == 3:
            captions, motion, m_lengths = batch
            return list(captions), motion, m_lengths
        captions, motion = batch
        return list(captions), motion, None
    # Unconditional path: feed empty strings (model will replace with cfg_uncond
    # at the configured drop probability).
    return [""] * batch.shape[0], batch, None


def build_cfg(opt) -> OmegaConf:
    """Synthesize the OmegaConf config object MoScaleFSQ expects."""
    head_latent_dim = opt.head_latent_dim if opt.head_latent_dim > 0 else opt.latent_dim
    return OmegaConf.create({
        "model": dict(
            latent_dim=opt.latent_dim, head_latent_dim=head_latent_dim,
            num_layers=opt.num_layers, n_heads=opt.num_heads,
            mlp_ratio=opt.mlp_ratio, dropout=opt.dropout, attn_drop_rate=0.0,
            use_crossattn=True, attn_l2_norm=False, infer_use_kvcache=False,
        ),
        "text_embedder": dict(dim_embed=768, version=opt.t5_model),
        "data": dict(dim_pose=opt.pose_dim, max_motion_length=opt.window_size,
                     max_text_length=opt.t5_max_len),
        "training": dict(
            # perturb_rate is plumbed through to SkelVQWrapper.encode and on
            # to MultiScaleFSQ.encode_indices, which applies a per-step
            # uniform Categorical level resample at training time only.
            perturb_rate=[opt.perturb_lo, opt.perturb_hi],
            cond_drop_prob=opt.cond_drop_prob,
            sample_level_times=[1, 1, 1, 1],
        ),
        "vq": dict(code_dim=32, scales=[8, 4, 2, 1]),
        "exp": dict(seed=opt.seed),
    })


def run_validation(ar, vq_model, val_loader, device, text_cond, max_batches=20):
    ar.train(False)
    losses, accs = [], []
    per_scale_accs = defaultdict(list)
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            texts, motion, m_lens_in = _unpack_batch(batch, text_cond)
            motion = motion.to(device, dtype=torch.float32)
            if m_lens_in is None:
                m_lens = torch.full((motion.shape[0],), motion.shape[1], device=device, dtype=torch.long)
            else:
                m_lens = m_lens_in.to(device, dtype=torch.long)
            loss, _, acc, ps_acc = ar.forward(motion, texts, m_lens.clone(), vq_model, train=False)
            losses.append(loss.item())
            accs.append(acc)
            for k, v in ps_acc.items():
                per_scale_accs[k].append(v)
    ar.train(True)
    val_loss = float(np.mean(losses)) if losses else 0.0
    val_acc = float(np.mean(accs)) if accs else 0.0
    val_ps_acc = {k: float(np.mean(v)) for k, v in per_scale_accs.items()}
    return val_loss, val_acc, val_ps_acc


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

    # ----- (optional) generation-quality evaluator bundle
    gen_eval_bundle = build_gen_eval(opt, opt.device)

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
    best_fid = float("inf")
    logs = defaultdict(def_value, OrderedDict())

    # ----- pre-train baseline eval (matches MoScale's pattern at
    # transformer_trainer.py:106-112: get a random-init / resumed baseline
    # before any optimizer step, plus a sanity check that the gen-eval
    # pipeline is wired correctly *before* committing 200+ epochs of compute).
    print(f"\n=== pre-train baseline eval at it={it} ===")
    val_loss, val_acc, val_ps_acc = run_validation(ar, vq_model, val_loader, opt.device, opt.text_cond)
    ps_str = "  ".join(f"{k}={v:.3f}" for k, v in val_ps_acc.items())
    print(f"[val] pre-train: loss {val_loss:.4f}  acc {val_acc:.4f}  {ps_str}")
    logger.add_scalar("Val/loss", val_loss, it)
    logger.add_scalar("Val/acc", val_acc, it)
    for k, v in val_ps_acc.items():
        logger.add_scalar(f"Val/{k}", v, it)

    if gen_eval_bundle is not None:
        eval_loader, eval_wrapper = gen_eval_bundle
        gen_func = make_gen_func(
            ar, vq_model, cond_scale=opt.fid_cond_scale,
            temperature=opt.fid_temperature, top_p=opt.fid_top_p,
            device=str(opt.device),
        )
        metric_runs = []
        for r in range(opt.fid_repeat_times):
            m = evaluate_once(ar, vq_model, eval_loader, eval_wrapper, gen_func, str(opt.device))
            metric_runs.append(m)
        agg = aggregate_repeats(metric_runs)
        ar.train(True)
        msg = (
            f"[gen-eval] pre-train  "
            f"FID {agg['fid']:.4f}±{agg.get('fid_conf95', 0):.4f}  "
            f"top1 {agg['top1']:.4f}  top2 {agg['top2']:.4f}  top3 {agg['top3']:.4f}  "
            f"Div {agg['diversity']:.3f} (real {agg['diversity_real']:.3f})  "
            f"Match {agg['matching']:.3f}"
        )
        print(msg)
        for k, v in agg.items():
            if not k.endswith("_conf95"):
                logger.add_scalar(f"GenEval/{k}", v, it)
        # Don't save net_best_fid.tar at pre-train (random init or just-resumed
        # ckpt; the parent ckpt is already on disk if resumed). Just record the
        # baseline FID so subsequent in-loop evals compare against it.
        best_fid = agg["fid"]

    # ----- train loop
    while epoch < opt.max_epoch:
        ar.train(True)
        for i, batch in enumerate(train_loader):
            it += 1
            if it < opt.warm_up_iter:
                cur_lr = opt.lr * (it + 1) / (opt.warm_up_iter + 1)
                for g in optim.param_groups:
                    g["lr"] = cur_lr

            texts, motion, m_lens_in = _unpack_batch(batch, opt.text_cond)
            motion = motion.to(opt.device, dtype=torch.float32)
            if m_lens_in is None:
                m_lens = torch.full((motion.shape[0],), motion.shape[1], device=opt.device, dtype=torch.long)
            else:
                m_lens = m_lens_in.to(opt.device, dtype=torch.long)

            optim.zero_grad()
            loss, _pred_idx, acc, ps_acc = ar.forward(motion, texts, m_lens.clone(), vq_model, train=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(ar.parameters(), opt.grad_clip)
            optim.step()

            if it >= opt.warm_up_iter:
                scheduler.step()

            logs["loss"] += loss.item()
            logs["lr"] += optim.param_groups[0]["lr"]
            logs["grad_norm"] += float(grad_norm)
            logs["acc"] += acc
            for k, v in ps_acc.items():
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
            val_loss, val_acc, val_ps_acc = run_validation(ar, vq_model, val_loader, opt.device, opt.text_cond)
            ps_str = "  ".join(f"{k}={v:.3f}" for k, v in val_ps_acc.items())
            print(f"[val] ep {epoch}: loss {val_loss:.4f}  acc {val_acc:.4f}  {ps_str}")
            logger.add_scalar("Val/loss", val_loss, it)
            logger.add_scalar("Val/acc", val_acc, it)
            for k, v in val_ps_acc.items():
                logger.add_scalar(f"Val/{k}", v, it)
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"ar": ar.state_dict(), "epoch": epoch, "val_loss": val_loss,
                            "opt": vars(opt)},
                           pjoin(opt.model_dir, "net_best_loss.tar"))
                print(f"  new best val loss {val_loss:.4f}")

        # ----- periodic generation-quality eval (FID + R-precision + Diversity)
        if gen_eval_bundle is not None and epoch % opt.fid_every_e == 0:
            eval_loader, eval_wrapper = gen_eval_bundle
            gen_func = make_gen_func(
                ar, vq_model,
                cond_scale=opt.fid_cond_scale,
                temperature=opt.fid_temperature,
                top_p=opt.fid_top_p,
                device=str(opt.device),
            )
            metric_runs = []
            for r in range(opt.fid_repeat_times):
                m = evaluate_once(ar, vq_model, eval_loader, eval_wrapper, gen_func, str(opt.device))
                metric_runs.append(m)
            agg = aggregate_repeats(metric_runs)
            ar.train(True)
            msg = (
                f"[gen-eval] ep {epoch}  "
                f"FID {agg['fid']:.4f}±{agg.get('fid_conf95', 0):.4f}  "
                f"top1 {agg['top1']:.4f}  top2 {agg['top2']:.4f}  top3 {agg['top3']:.4f}  "
                f"Div {agg['diversity']:.3f} (real {agg['diversity_real']:.3f})  "
                f"Match {agg['matching']:.3f}"
            )
            print(msg)
            for k, v in agg.items():
                if not k.endswith("_conf95"):
                    logger.add_scalar(f"GenEval/{k}", v, it)
            if agg["fid"] < best_fid:
                best_fid = agg["fid"]
                torch.save({"ar": ar.state_dict(), "epoch": epoch, "fid": best_fid,
                            "opt": vars(opt)},
                           pjoin(opt.model_dir, "net_best_fid.tar"))
                print(f"  new best FID {best_fid:.4f}")


if __name__ == "__main__":
    main()
