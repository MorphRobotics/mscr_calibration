#!/usr/bin/env python3
"""Grab one D435 color frame, save it, and try several checkerboard sizes so we
can see what's actually in view and which inner-corner count detects."""
import cv2, numpy as np, yaml
from pathlib import Path
from tip_sensor import D435Camera

HERE = Path(__file__).parent
cfg = yaml.safe_load(open(HERE / "config.yaml"))
cam = D435Camera(cfg["camera"]["width"], cfg["camera"]["height"],
                 cfg["camera"]["fps"], cfg["camera"]["laser_power"])
color, _ = cam.get_frames()
cam.close()
out = HERE / "results"; out.mkdir(exist_ok=True)
cv2.imwrite(str(out / "board_debug.png"), color)
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
print(f"frame {color.shape}, gray mean={gray.mean():.0f} (very low=too dark, very high=washed out)")

def try_size(gray, size):
    """Most thorough sector-based detector; fall back to the classic one."""
    if hasattr(cv2, "findChessboardCornersSB"):
        flg = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, size, flags=flg)
        if found:
            return True, corners
    f2 = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    return cv2.findChessboardCorners(gray, size, f2)

# brute scan plausible inner-corner grids (asymmetric, larger side first)
hit = None
detections = []
for w in range(6, 17):
    for h in range(5, w):          # h < w so we report (long, short)
        found, corners = try_size(gray, (w, h))
        if found:
            detections.append((w, h))
            if hit is None:
                hit = ((w, h), corners)
print("detected grids:", detections if detections else "none")
if hit:
    size, corners = hit
    vis = color.copy()
    cv2.drawChessboardCorners(vis, size, corners, True)
    cv2.imwrite(str(out / "board_detected.png"), vis)
    print(f"\nWORKS with inner-corner size {size}. Set config cols={size[0]}, rows={size[1]}.")
    print("Overlay saved -> results/board_detected.png")
else:
    print("\nNo size detected. Open results/board_debug.png and check:")
    print("  - whole board fully in frame, flat, not clipped at edges")
    print("  - even lighting, no glare/shadow across squares, in focus")
    print("  - enough white margin (quiet zone) around the board")
