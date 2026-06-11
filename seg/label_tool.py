#!/usr/bin/env python3
"""
label_tool.py — manual brush correction of seed masks.

Cycle through frames, paint (add rod) or erase (remove false positives) the
mask with the mouse.  The goal is fast cleanup of autolabel.py output, not
pixel-perfect labels — a U-Net tolerates slightly noisy masks.

Controls:
    Left-drag    paint   (add to mask)
    Right-drag   erase   (remove from mask)
    [  /  ]      brush smaller / larger
    n  /  SPACE  next frame (saves current)
    p            previous frame (saves current)
    c            clear mask
    f            flood-fill the largest dark blob under a single left click region
    r            re-run nothing — just reload original seed (undo session edits)
    s            save now
    q / ESC      save and quit

Usage:
    python seg/label_tool.py [--start 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

HERE     = Path(__file__).parent
IMG_DIR  = HERE / "dataset" / "images"
MASK_DIR = HERE / "dataset" / "masks"

brush      = 6
painting   = 0     # 0 none, 1 paint, -1 erase
mask       = None
last_pt    = None


def on_mouse(event, x, y, flags, param):
    global painting, last_pt, mask
    if event == cv2.EVENT_LBUTTONDOWN:
        painting = 1
        last_pt = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        painting = -1
        last_pt = (x, y)
    elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        painting = 0
        last_pt = None
    elif event == cv2.EVENT_MOUSEMOVE and painting != 0 and mask is not None:
        val = 255 if painting == 1 else 0
        if last_pt is not None:
            cv2.line(mask, last_pt, (x, y), val, brush * 2)
        cv2.circle(mask, (x, y), brush, val, -1)
        last_pt = (x, y)


def render(color, mask):
    ovl = color.copy()
    ovl[mask > 0] = (0, 255, 128)
    vis = cv2.addWeighted(color, 0.6, ovl, 0.4, 0)
    cv2.putText(vis, f"brush={brush}  L=paint R=erase  n/p=next/prev  c=clear  q=quit",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return vis


def main():
    global mask, brush
    ap = argparse.ArgumentParser(description="Manual mask correction")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    imgs = sorted(IMG_DIR.glob("frame_*.png"))
    if not imgs:
        print(f"No images in {IMG_DIR}.")
        return

    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("label", on_mouse)

    i = max(0, min(args.start, len(imgs) - 1))

    def load(idx):
        global mask
        color = cv2.imread(str(imgs[idx]))
        mp = MASK_DIR / imgs[idx].name
        if mp.exists():
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8) * 255
        else:
            mask = np.zeros(color.shape[:2], np.uint8)
        return color

    def save(idx):
        if mask is not None:
            cv2.imwrite(str(MASK_DIR / imgs[idx].name), mask)

    color = load(i)
    print(f"Labelling {len(imgs)} frames starting at {i}. q to save+quit.")
    while True:
        cv2.imshow("label", render(color, mask))
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            save(i); break
        elif k in (ord("n"), ord(" ")):
            save(i); i = min(i + 1, len(imgs) - 1); color = load(i)
            print(f"  frame {i}/{len(imgs)-1}  {imgs[i].name}")
        elif k == ord("p"):
            save(i); i = max(i - 1, 0); color = load(i)
            print(f"  frame {i}/{len(imgs)-1}  {imgs[i].name}")
        elif k == ord("c"):
            mask[:] = 0
        elif k == ord("["):
            brush = max(1, brush - 1)
        elif k == ord("]"):
            brush = min(60, brush + 1)
        elif k == ord("s"):
            save(i); print(f"  saved {imgs[i].name}")
        elif k == ord("r"):
            color = load(i)

    cv2.destroyAllWindows()
    print("Saved. Done labelling.")


if __name__ == "__main__":
    main()
