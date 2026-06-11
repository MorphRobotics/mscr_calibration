#!/usr/bin/env python3
"""
infer.py — real-time inference wrapper for the trained rod U-Net.

RodSegmenter.segment(bgr) → full-resolution 0/255 binary mask.

The model runs at NET_W×NET_H and the output mask is resized back to the
input frame size.  Designed to be a drop-in replacement for the classical
dark-threshold segmentation step in MSCRTracker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

# Allow importing sibling modules whether run as package or loose files
try:
    from .unet import UNet
    from .dataset import preprocess_bgr, NET_W, NET_H
except ImportError:  # pragma: no cover
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from unet import UNet
    from dataset import preprocess_bgr, NET_W, NET_H


class RodSegmenter:
    """
    Loads rod_seg.pt and segments full-resolution BGR frames.

    Args:
        ckpt_path : path to the trained checkpoint
        device    : "cuda" | "cpu" | None (auto)
        thr       : sigmoid threshold for the binary mask
        half      : use FP16 on CUDA for speed
    """

    def __init__(self,
                 ckpt_path: str,
                 device: Optional[str] = None,
                 thr: float = 0.5,
                 half: bool = True):
        if not _TORCH:
            raise RuntimeError("PyTorch not installed — cannot run RodSegmenter.")

        self.thr = thr
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.half   = half and device == "cuda"

        ckpt = torch.load(ckpt_path, map_location=device)
        base_ch = ckpt.get("base_ch", 24)
        self.net_w = ckpt.get("net_w", NET_W)
        self.net_h = ckpt.get("net_h", NET_H)

        self.model = UNet(in_ch=3, base_ch=base_ch).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        if self.half:
            self.model.half()

        print(f"[RodSegmenter] loaded {ckpt_path}  device={device}  "
              f"half={self.half}  net={self.net_w}x{self.net_h}")

    def segment(self, bgr: np.ndarray) -> np.ndarray:
        """Return a full-resolution 0/255 uint8 rod mask for a BGR frame."""
        H, W = bgr.shape[:2]
        x = preprocess_bgr(bgr).to(self.device)
        if self.half:
            x = x.half()
        with torch.no_grad():
            prob = torch.sigmoid(self.model(x).float())[0, 0].cpu().numpy()
        small = (prob > self.thr).astype(np.uint8) * 255
        return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)

    def segment_prob(self, bgr: np.ndarray) -> np.ndarray:
        """Return the full-resolution probability map (float32 0..1)."""
        H, W = bgr.shape[:2]
        x = preprocess_bgr(bgr).to(self.device)
        if self.half:
            x = x.half()
        with torch.no_grad():
            prob = torch.sigmoid(self.model(x).float())[0, 0].cpu().numpy()
        return cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Test RodSegmenter on an image")
    ap.add_argument("--ckpt",  default=str(Path(__file__).parent / "rod_seg.pt"))
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    seg = RodSegmenter(args.ckpt)
    bgr = cv2.imread(args.image)
    mask = seg.segment(bgr)
    ovl = bgr.copy(); ovl[mask > 0] = (0, 255, 128)
    out = cv2.addWeighted(bgr, 0.6, ovl, 0.4, 0)
    cv2.imwrite("seg_test_overlay.png", out)
    print("Wrote seg_test_overlay.png")
