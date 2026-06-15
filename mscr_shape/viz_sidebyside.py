#!/usr/bin/env python3
"""
viz_sidebyside.py — render raw left-IR video next to the predicted 3D
reconstruction in ONE animated GIF, so the reconstruction can be judged
qualitatively against the actual rod.

Left panel : rectified left-IR frame with the predicted 3D centerline
             reprojected onto it (green) + tip (red).
Right panel: the predicted 3D centerline in the left-IR camera frame, with the
             tip trajectory accumulating over time.

Run:  python viz_sidebyside.py --session s09 [--stride N] [--max-frames N] [--out path]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from calib import resolve_calib
from cfg import load_config
from labeler import project
from infer import TemporalFilter, preprocess, predict, load_model


def render(session: str, stride: int, max_frames: Optional[int], out: Path,
           elev: float = 18.0, azim: float = -70.0) -> None:
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(__file__).parent / cfg["train"]["ckpt"]
    net, cfg = load_model(ckpt, device)
    calib = resolve_calib(cfg)

    data_root = Path(cfg["paths"]["data_root"])
    files = sorted((data_root / "raw" / session / "left").glob("*.png"))[::stride]
    if max_frames:
        files = files[:max_frames]
    if not files:
        raise RuntimeError(f"no frames in {data_root/'raw'/session/'left'}")

    tf = TemporalFilter(cfg["infer"]["ema_alpha"], cfg["infer"]["max_jump_mm"])
    imgs, curves, projs = [], [], []
    for f in files:
        rect = calib.rectify_left(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE))
        pts, _ = predict(net, preprocess(rect, cfg), device)
        smoothed, _ = tf.update(pts)
        imgs.append(rect)
        curves.append(smoothed.copy())
        projs.append(project(calib.P1, smoothed))     # (N,2) px in rectified left
    seq = np.stack(curves)
    tips = seq[:, -1, :]
    print(f"predicted {len(seq)} frames")

    allpts = seq.reshape(-1, 3)
    ctr = (allpts.min(0) + allpts.max(0)) / 2
    rng = (allpts.max(0) - allpts.min(0)).max() / 2 * 1.1

    fig = plt.figure(figsize=(12, 5))
    ax_im = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

    def draw(i):
        ax_im.clear(); ax_3d.clear()
        # left: raw IR + reprojected centerline
        vis = cv2.cvtColor(imgs[i], cv2.COLOR_GRAY2BGR)
        p = projs[i].astype(np.int32)
        cv2.polylines(vis, [p], False, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.circle(vis, tuple(p[-1]), 5, (0, 0, 255), -1)   # tip
        cv2.circle(vis, tuple(p[0]), 5, (0, 0, 0), -1)      # base
        ax_im.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax_im.set_title(f"{session}  raw left-IR + reprojected r(s)")
        ax_im.axis("off")
        # right: 3D reconstruction
        c = seq[i]
        ax_3d.plot(c[:, 0], c[:, 1], c[:, 2], "-", lw=3, color="tab:green")
        ax_3d.scatter(*c[0], color="black", s=40, label="base")
        ax_3d.scatter(*c[-1], color="red", s=40, label="tip")
        ax_3d.plot(tips[:i + 1, 0], tips[:i + 1, 1], tips[:i + 1, 2],
                   "-", lw=1, color="tab:red", alpha=0.6, label="tip path")
        for a, lo, hi in zip("xyz", ctr - rng, ctr + rng):
            getattr(ax_3d, f"set_{a}lim")(lo, hi)
        ax_3d.set_xlabel("X (mm)"); ax_3d.set_ylabel("Y (mm)"); ax_3d.set_zlabel("Z (mm)")
        ax_3d.set_title(f"predicted 3D reconstruction  frame {i+1}/{len(seq)}")
        ax_3d.view_init(elev=elev, azim=azim)
        ax_3d.legend(loc="upper right", fontsize=8)

    anim = FuncAnimation(fig, draw, frames=len(seq), interval=80)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    anim.save(str(out), writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"saved side-by-side GIF -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="raw video | 3D reconstruction GIF")
    ap.add_argument("--session", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(__file__).parent / f"sidebyside_{args.session}.gif"
    render(args.session, args.stride, args.max_frames, out)


if __name__ == "__main__":
    main()
