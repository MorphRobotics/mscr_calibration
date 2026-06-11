#!/usr/bin/env python3
"""
dataset.py — PyTorch Dataset for MSCR rod segmentation.

Loads (image, mask) pairs from seg/dataset/, resizes to a fixed network
resolution, and applies light augmentation tuned for a thin elongated target:
random flips, small rotations, scale/shift, brightness/contrast jitter, and
mild blur.  Masks are kept binary through nearest-neighbour resizing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

HERE     = Path(__file__).parent
IMG_DIR  = HERE / "dataset" / "images"
MASK_DIR = HERE / "dataset" / "masks"

# Network input resolution (W, H) — 320×192 keeps ~16:9 and is /32 divisible
NET_W, NET_H = 320, 192

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_pairs(require_nonempty: bool = True):
    """Return list of (image_path, mask_path) that both exist."""
    pairs = []
    for ip in sorted(IMG_DIR.glob("frame_*.png")):
        mp = MASK_DIR / ip.name
        if not mp.exists():
            continue
        if require_nonempty:
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None or m.max() == 0:
                continue
        pairs.append((ip, mp))
    return pairs


def _augment(img: np.ndarray, mask: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray]:
    H, W = img.shape[:2]

    # Horizontal flip
    if np.random.rand() < 0.5:
        img  = img[:, ::-1]
        mask = mask[:, ::-1]
    # Vertical flip (rod has no inherent up/down)
    if np.random.rand() < 0.3:
        img  = img[::-1]
        mask = mask[::-1]

    # Affine: rotation + scale + translation
    if np.random.rand() < 0.8:
        ang   = np.random.uniform(-25, 25)
        scale = np.random.uniform(0.85, 1.15)
        tx    = np.random.uniform(-0.08, 0.08) * W
        ty    = np.random.uniform(-0.08, 0.08) * H
        M = cv2.getRotationMatrix2D((W/2, H/2), ang, scale)
        M[0, 2] += tx; M[1, 2] += ty
        img  = cv2.warpAffine(img,  M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Brightness / contrast jitter (rod must survive lighting changes)
    if np.random.rand() < 0.8:
        alpha = np.random.uniform(0.6, 1.4)   # contrast
        beta  = np.random.uniform(-40, 40)    # brightness
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Mild blur
    if np.random.rand() < 0.25:
        k = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


class RodSegDataset(Dataset):
    def __init__(self, pairs, train: bool = True):
        self.pairs = pairs
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img  = cv2.imread(str(ip))                       # BGR
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)

        img  = cv2.resize(img,  (NET_W, NET_H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (NET_W, NET_H), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        if self.train:
            img, mask = _augment(img, mask)

        # BGR→RGB, normalise
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        rgb = np.transpose(rgb, (2, 0, 1))               # CHW

        return (torch.from_numpy(rgb).float(),
                torch.from_numpy(mask[None]).float())     # (1,H,W)


def preprocess_bgr(bgr: np.ndarray) -> torch.Tensor:
    """Convert a full-res BGR frame to a normalised network input tensor."""
    img = cv2.resize(bgr, (NET_W, NET_H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    rgb = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(rgb[None]).float()
