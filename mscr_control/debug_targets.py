#!/usr/bin/env python3
"""Grab a D435 frame; test checkerboard AND all common ArUco dictionaries +
report depth validity, so we can pick the most reliable calibration target."""
import cv2, numpy as np, yaml
from pathlib import Path
from tip_sensor import D435Camera
from handeye_calibrate import find_chessboard

HERE = Path(__file__).parent
cfg = yaml.safe_load(open(HERE / "config.yaml"))
cam = D435Camera(cfg["camera"]["width"], cfg["camera"]["height"], cfg["camera"]["fps"], 200)
# grab a few, keep the last
for _ in range(8):
    color, depth = cam.get_frames()
cam.close()
out = HERE / "results"; out.mkdir(exist_ok=True)
cv2.imwrite(str(out / "targets_debug.png"), color)
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
dvalid = (depth > 0).mean()
print(f"frame {color.shape}  gray mean={gray.mean():.0f}  depth valid={dvalid*100:.0f}%")

# checkerboard
f, c = find_chessboard(gray, (cfg["handeye"]["target"]["cols"], cfg["handeye"]["target"]["rows"]))
print(f"checkerboard (10,9): {'DETECTED' if f else 'no'}")

# ArUco across dictionaries
dicts = ["DICT_4X4_50", "DICT_4X4_250", "DICT_5X5_50", "DICT_5X5_250",
         "DICT_6X6_50", "DICT_6X6_250", "DICT_7X7_50", "DICT_ARUCO_ORIGINAL"]
params = cv2.aruco.DetectorParameters_create()
best = None
for dn in dicts:
    d = cv2.aruco.Dictionary_get(getattr(cv2.aruco, dn))
    corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=params)
    n = 0 if ids is None else len(ids)
    if n:
        print(f"  ArUco {dn}: {n} marker(s), ids={ids.flatten().tolist()}")
        if best is None:
            best = (dn, corners, ids)
if best:
    dn, corners, ids = best
    vis = color.copy()
    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    cv2.imwrite(str(out / "aruco_detected.png"), vis)
    print(f"\nArUco WORKS with {dn}. Overlay -> results/aruco_detected.png")
else:
    print("  no ArUco markers detected in any dictionary")
