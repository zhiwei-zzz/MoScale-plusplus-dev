"""HumanML3D loaders for the SkelVQ-FSQ AR pipeline.

Two dataset variants:

* `Text2MotionWindowDataset` — original fixed-window loader. Picks a random
  `window_size`-frame slice per item and pairs it with a caption. Suitable
  for window-based training when the AR's full_length matches the window.

* `Text2MotionPaddedDataset` — variable-length, padded-to-max loader. Mirrors
  MoScale's `Text2MotionDataset`: each item is the full motion (Z-normalized
  + zero-padded to `max_motion_length`) along with its true length so the
  AR can mask padded positions in the loss. This is what you want when the
  AR is trained to generate at the original motion length (so eval doesn't
  need to truncate generated motion to match real motion).
"""
from __future__ import annotations

import random
import codecs as cs
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm


class Text2MotionWindowDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        self.window_size = opt.window_size
        self.fps = getattr(opt, "fps", 20)

        self.mean = mean
        self.std = std

        id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # entries: list of (motion_array, caption_str) — one entry per
        # (motion, sub-clip-with-caption) combination, materialized so the
        # __getitem__ is O(1).
        self.entries = []
        for name in tqdm(id_list, desc=f"loading {split_file}"):
            try:
                full_motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
            except Exception:
                continue
            if full_motion.shape[0] < self.window_size:
                continue

            captions_path = pjoin(opt.text_dir, name + ".txt")
            try:
                with cs.open(captions_path) as fh:
                    lines = list(fh.readlines())
            except Exception:
                continue

            for line in lines:
                parts = line.strip().split("#")
                if len(parts) < 4:
                    continue
                caption = parts[0]
                f_tag = float(parts[2]) if parts[2] not in ("", "nan") else 0.0
                to_tag = float(parts[3]) if parts[3] not in ("", "nan") else 0.0
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                if f_tag == 0.0 and to_tag == 0.0:
                    motion = full_motion
                else:
                    a = int(f_tag * self.fps)
                    b = int(to_tag * self.fps)
                    motion = full_motion[a:b]
                if motion.shape[0] < self.window_size:
                    continue
                self.entries.append((motion, caption))

        print(f"[Text2MotionWindowDataset] {split_file}: "
              f"{len(self.entries)} (motion, caption) entries")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        motion, caption = self.entries[idx]
        # Random window
        start = random.randint(0, motion.shape[0] - self.window_size)
        window = motion[start : start + self.window_size]
        # Z normalize
        window = (window - self.mean) / self.std
        return caption, window.astype(np.float32)


class Text2MotionPaddedDataset(data.Dataset):
    """Variable-length motion + caption, padded to `opt.max_motion_length`.

    Per item:
      * load the motion + a randomly chosen caption (handling sub-clip
        timestamps if present)
      * align the motion length down to a multiple of `unit_length` (so the
        bottleneck-space length is well-defined for the SkelVQ tokenizer)
      * Z-normalize, then zero-pad on the right to `max_motion_length`
      * return (caption, padded_motion, m_length)

    Used when training the AR at the original motion length (so eval pred
    matches real motion length 1:1, no truncation needed in evaluate_once).
    """

    MIN_LEN = 40

    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        self.fps = getattr(opt, "fps", 20)
        self.max_motion_length = opt.max_motion_length
        self.unit_length = getattr(opt, "unit_length", 4)
        self.mean = mean
        self.std = std

        id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        self.entries = []  # list of (motion_np, caption)
        for name in tqdm(id_list, desc=f"loading {split_file}"):
            try:
                full_motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
            except Exception:
                continue
            if full_motion.shape[0] < self.MIN_LEN:
                continue

            captions_path = pjoin(opt.text_dir, name + ".txt")
            try:
                with cs.open(captions_path) as fh:
                    lines = list(fh.readlines())
            except Exception:
                continue

            for line in lines:
                parts = line.strip().split("#")
                if len(parts) < 4:
                    continue
                caption = parts[0]
                f_tag = float(parts[2]) if parts[2] not in ("", "nan") else 0.0
                to_tag = float(parts[3]) if parts[3] not in ("", "nan") else 0.0
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                if f_tag == 0.0 and to_tag == 0.0:
                    motion = full_motion
                else:
                    a = int(f_tag * self.fps)
                    b = int(to_tag * self.fps)
                    motion = full_motion[a:b]

                if motion.shape[0] < self.MIN_LEN:
                    continue
                # Cap at max_motion_length
                if motion.shape[0] > self.max_motion_length:
                    # Random crop to the cap (keeps caption-motion alignment
                    # approximate for very long motions; for in-range ones
                    # this is a no-op).
                    start = random.randint(0, motion.shape[0] - self.max_motion_length)
                    motion = motion[start : start + self.max_motion_length]
                self.entries.append((motion, caption))

        print(f"[Text2MotionPaddedDataset] {split_file}: "
              f"{len(self.entries)} (motion, caption) entries")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        motion, caption = self.entries[idx]
        m_length = motion.shape[0]
        # Snap m_length down to a multiple of unit_length so the tokenizer's
        # time downsampling factor (2**n_layers = 4) divides cleanly.
        m_length = (m_length // self.unit_length) * self.unit_length
        m_length = max(m_length, self.unit_length)
        motion = motion[:m_length]

        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            pad = np.zeros((self.max_motion_length - m_length, motion.shape[1]),
                           dtype=motion.dtype)
            padded = np.concatenate([motion, pad], axis=0)
        else:
            padded = motion
        return caption, padded.astype(np.float32), int(m_length)


def collate_fn(batch):
    """Collator for both Text2MotionWindowDataset (2-tuple) and
    Text2MotionPaddedDataset (3-tuple). Returns:

      window mode:  (captions: list[str], motions: (B, T, D))
      padded mode:  (captions: list[str], motions: (B, T_max, D), m_lengths: (B,))
    """
    captions = [b[0] for b in batch]
    motions = torch.from_numpy(np.stack([b[1] for b in batch], axis=0))
    if len(batch[0]) == 3:
        m_lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)
        return captions, motions, m_lengths
    return captions, motions
