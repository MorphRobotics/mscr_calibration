#!/usr/bin/env python3
"""
autolabel.py — generate seed segmentation masks for collected frames.

Uses the same classical cues as the original tracker (dark threshold +
optional depth gate + largest elongated connected component + skeleton-based
thinness check) to produce an initial rod mask for every image.  These seeds
are NOT perfect — run label_tool.py afterwards to correct the bad ones.

Writes:
    seg/dataset/masks/frame_xxxxx.png      (0/255 binary mask)
    seg/dataset/overlays/frame_xxxxx.png   (colour overlay for quick review)

Usage:
    python seg/autolabel.py [--threshold 60] [--use-depth]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

HERE      = Path(__file__).parent
IMG_DIR   = HERE / "dataset" / "images"
DEPTH_DIR = HERE / "dataset" / "depth"
MASK_DIR  = HERE / "dataset" / "masks"
OVL_DIR   = HERE / "dataset" / "overlays"


def seed_mask(color: np.ndarray,
              depth: np.ndarray | None,
              dark_threshold: int,
              depth_max_m: float,
              depth_scale: float,
              min_area: int = 120,
              max_area_frac: float = 0.06) -> np.ndarray:
    """Return a 0/255 seed mask of the most rod-like dark elongated blob."""
    H, W = color.shape[:2]
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)

    if depth is not None:
        depth_m = depth.astype(np.float32) * depth_scale
        far = ((depth > 0) & (depth_m > depth_max_m)).astype(np.uint8) * 255
        far = cv2.erode(far, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        dark = cv2.bitwise_and(dark, cv2.bitwise_not(far))

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,  se, iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, se, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    maxpx = int(H * W * max_area_frac)
    best, best_score = None, -1.0
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if not (min_area <= a <= maxpx):
            continue
        pts = np.column_stack(np.where(labels == i))[:, ::-1].astype(np.float32)
        if len(pts) < 5:
            continue
        try:
            (_, _), (ma, Mi), _ = cv2.fitEllipse(pts)
            minor, major = sorted([ma, Mi])
            ecc = float(np.sqrt(1.0 - (minor / max(major, 1e-6)) ** 2))
        except Exception:
            ecc = 0.0
        score = ecc * 1000 + a * 0.001    # favour elongation, tie-break on size
        if score > best_score:
            best_score, best = score, i

    mask = np.zeros((H, W), np.uint8)
    if best is not None:
        mask[labels == best] = 255
    return mask


def main():
    ap = argparse.ArgumentParser(description="Auto-label rod masks")
    ap.add_argument("--threshold",   type=int,   default=60)
    ap.add_argument("--use-depth",   action="store_true",
                    help="use saved depth .npy to gate background")
    ap.add_argument("--depth-max",   type=float, default=0.80)
    ap.add_argument("--depth-scale", type=float, default=0.001)
    args = ap.parse_args()

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    OVL_DIR.mkdir(parents=True, exist_ok=True)

    imgs = sorted(IMG_DIR.glob("frame_*.png"))
    if not imgs:
        print(f"No images in {IMG_DIR}. Run collect.py first.")
        return

    print(f"Auto-labelling {len(imgs)} frames…")
    n_empty = 0
    for p in imgs:
        color = cv2.imread(str(p))
        depth = None
        if args.use_depth:
            dp = DEPTH_DIR / (p.stem + ".npy")
            if dp.exists():
                depth = np.load(str(dp))

        mask = seed_mask(color, depth, args.threshold,
                         args.depth_max, args.depth_scale)
        if mask.max() == 0:
            n_empty += 1

        cv2.imwrite(str(MASK_DIR / p.name), mask)

        ovl = color.copy()
        ovl[mask > 0] = (0, 255, 128)
        ovl = cv2.addWeighted(color, 0.6, ovl, 0.4, 0)
        cv2.imwrite(str(OVL_DIR / p.name), ovl)

    print(f"Done. Masks → {MASK_DIR}")
    print(f"{n_empty}/{len(imgs)} frames produced an EMPTY mask (need manual labelling).")
    print(f"Review overlays in {OVL_DIR}, then run:  python seg/label_tool.py")


if __name__ == "__main__":
    main()
