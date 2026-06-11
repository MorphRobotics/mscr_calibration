#!/usr/bin/env python3
"""
collect.py — capture training frames from the RealSense D435i.

Saves colour PNGs (and optionally raw depth .npy) into seg/dataset/images/.
Move the rod through many poses, lighting angles, and backgrounds while this
runs to build a varied dataset.

Usage:
    python seg/collect.py --n 300 --every 3
        --n     total frames to save
        --every save every Nth frame (skip near-duplicates)

Controls (a preview window is shown):
    SPACE  force-save the current frame
    q/ESC  stop early
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

HERE     = Path(__file__).parent
IMG_DIR  = HERE / "dataset" / "images"
DEPTH_DIR = HERE / "dataset" / "depth"


def main():
    ap = argparse.ArgumentParser(description="Collect MSCR segmentation frames")
    ap.add_argument("--n",      type=int, default=300, help="frames to save")
    ap.add_argument("--every",  type=int, default=3,   help="save every Nth frame")
    ap.add_argument("--width",  type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps",    type=int, default=30)
    ap.add_argument("--save-depth", action="store_true",
                    help="also save raw depth .npy (for RGB-D training)")
    args = ap.parse_args()

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    if args.save_depth:
        DEPTH_DIR.mkdir(parents=True, exist_ok=True)

    # Continue numbering from any existing frames
    existing = sorted(IMG_DIR.glob("frame_*.png"))
    start_idx = (int(existing[-1].stem.split("_")[1]) + 1) if existing else 0

    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16,  args.fps)
    profile = pipeline.start(cfg)
    align   = rs.align(rs.stream.color)

    print(f"Collecting up to {args.n} frames → {IMG_DIR}")
    print("Move the rod through many poses/backgrounds. SPACE=save, q=quit.\n")

    saved = 0
    frame_i = 0
    cv2.namedWindow("collect", cv2.WINDOW_NORMAL)
    try:
        while saved < args.n:
            frames  = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            cf, df  = aligned.get_color_frame(), aligned.get_depth_frame()
            if not cf or not df:
                continue
            color = np.asanyarray(cf.get_data())
            depth = np.asanyarray(df.get_data())
            frame_i += 1

            key = cv2.waitKey(1) & 0xFF
            force = (key == ord(" "))
            if force or frame_i % args.every == 0:
                idx = start_idx + saved
                cv2.imwrite(str(IMG_DIR / f"frame_{idx:05d}.png"), color)
                if args.save_depth:
                    np.save(str(DEPTH_DIR / f"frame_{idx:05d}.npy"), depth)
                saved += 1

            vis = color.copy()
            cv2.putText(vis, f"saved {saved}/{args.n}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("collect", vis)
            if key in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    print(f"\nDone. {saved} frames saved to {IMG_DIR}")


if __name__ == "__main__":
    main()
