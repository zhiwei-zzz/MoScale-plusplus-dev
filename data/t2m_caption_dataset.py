"""HumanML3D loader: returns (caption, motion_window) pairs.

A window-based caption-aware variant of SALAD's MotionDataset. For each item:
  - pick a uniform-random window of `window_size` frames out of the motion
  - pick a uniform-random caption from the motion's caption file
  - return both, with the motion Z-normalized

The caption-window misalignment (caption describes the whole motion, the
window is a slice) is a known approximation but standard in motion AR work.
For exact alignment, use the (f_tag, to_tag) timestamps in the captions file
(some lines have non-zero start/end and describe a sub-clip) — handled here.

Returns from __getitem__:
  caption (str), motion (np.ndarray of shape (window_size, pose_dim))
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


def collate_fn(batch):
    """Collate (caption_str, motion_np) pairs. Captions stay as a list of
    strings (T5 tokenizer handles them); motions stack into a tensor."""
    captions = [b[0] for b in batch]
    motions = torch.from_numpy(np.stack([b[1] for b in batch], axis=0))
    return captions, motions
