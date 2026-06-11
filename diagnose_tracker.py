#!/usr/bin/env python3
"""
Capture one frame and save every intermediate mask as a PNG so we can
diagnose what the tracker is segmenting.

Usage:
    python3 diagnose_tracker.py --depth-min 0.1 --depth-max 0.8 --threshold 80
"""
import argparse
import numpy as np
import cv2
import pyrealsense2 as rs
from mscr_tracker import MSCRTracker, TrackerParams

ap = argparse.ArgumentParser()
ap.add_argument("--depth-min",  type=float, default=0.10)
ap.add_argument("--depth-max",  type=float, default=0.80)
ap.add_argument("--threshold",  type=int,   default=85)
ap.add_argument("--entry",      default="top")
ap.add_argument("--frames",     type=int,   default=60,
                help="warm-up frames before capturing the diagnostic frame")
args = ap.parse_args()

pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16,  30)
profile  = pipeline.start(cfg)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align    = rs.align(rs.stream.color)

print(f"Warming up ({args.frames} frames)…")
for _ in range(args.frames):
    pipeline.wait_for_frames()

frames   = align.process(pipeline.wait_for_frames())
pipeline.stop()

color = np.asanyarray(frames.get_color_frame().get_data())
depth = np.asanyarray(frames.get_depth_frame().get_data())

np.save("diag_depth.npy", depth)
cv2.imwrite("diag_color.png", color)
print("Saved diag_color.png and diag_depth.npy")

# ── Replay the pipeline steps visually ──────────────────────────────────────

# 1. depth gate
depth_m  = depth.astype(np.float32) * depth_scale
gate     = ((depth_m >= args.depth_min) & (depth_m <= args.depth_max) & (depth > 0)).astype(np.uint8) * 255
se_d     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
gate_dil = cv2.dilate(gate, se_d)

# 2. dark threshold
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
_, dark = cv2.threshold(gray, args.threshold, 255, cv2.THRESH_BINARY_INV)

# 3. combined mask
combined = cv2.bitwise_and(dark, gate_dil)

# 4. morph open/close
se3      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
morphed  = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  se3, iterations=2)
morphed  = cv2.morphologyEx(morphed,  cv2.MORPH_CLOSE, se3, iterations=3)

# 5. all connected components coloured
n, labels, stats, _ = cv2.connectedComponentsWithStats(morphed, connectivity=8)
cc_vis   = np.zeros((*morphed.shape, 3), dtype=np.uint8)
rng      = np.random.default_rng(0)
for i in range(1, n):
    col  = rng.integers(80, 255, 3).tolist()
    area = int(stats[i, cv2.CC_STAT_AREA])
    cc_vis[labels == i] = col
    cx = int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]  / 2)
    cy = int(stats[i, cv2.CC_STAT_TOP]  + stats[i, cv2.CC_STAT_HEIGHT] / 2)
    cv2.putText(cc_vis, str(area), (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

# depth range visualisation
depth_vis = np.clip((depth_m - args.depth_min) / max(args.depth_max - args.depth_min, 1e-6), 0, 1)
depth_vis = (depth_vis * 255).astype(np.uint8)
depth_col = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# ── Save ────────────────────────────────────────────────────────────────────
cv2.imwrite("diag_1_gate.png",     gate_dil)
cv2.imwrite("diag_2_dark.png",     dark)
cv2.imwrite("diag_3_combined.png", combined)
cv2.imwrite("diag_4_morphed.png",  morphed)
cv2.imwrite("diag_5_cc.png",       cc_vis)
cv2.imwrite("diag_6_depth_jet.png",depth_col)

print(f"\nFound {n-1} connected components after morphology:")
for i in range(1, n):
    print(f"  CC {i:3d}  area={stats[i,cv2.CC_STAT_AREA]:6d}px  "
          f"bbox=({stats[i,cv2.CC_STAT_LEFT]},{stats[i,cv2.CC_STAT_TOP]},"
          f"{stats[i,cv2.CC_STAT_WIDTH]}x{stats[i,cv2.CC_STAT_HEIGHT]})")

print("\nSaved: diag_1_gate.png  diag_2_dark.png  diag_3_combined.png")
print("       diag_4_morphed.png  diag_5_cc.png  diag_6_depth_jet.png")
print("\nLook at diag_6_depth_jet.png to find the rod's depth range,")
print("then re-run with tighter --depth-min / --depth-max.")
