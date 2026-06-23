#!/usr/bin/env python3
"""
handeye_touch.py — eye-to-hand calibration WITHOUT mounting anything on the arm.

Point-correspondence method (no AX=XB, no rigid board on the flange):
  1. Lay the checkerboard fixed on the table, fully in the D435's view.
  2. The camera localizes every inner corner in 3D (camera frame) via solvePnP
     using the known board geometry — done ONCE, with a clear view.
  3. You drive the robot TCP (a pointer / the magnet tip — with the TCP offset
     set to that tip on the pendant) to physically TOUCH a handful of those
     corners. At each touch we record getActualTCPPose -> the corner in the BASE
     frame.
  4. We fit the rigid transform mapping camera-frame corners -> base-frame
     corners (Kabsch/Umeyama, proper rotation) = T_base_camera.

Needs >= 3 non-collinear corners; 5+ well spread across the board is better.
The corners are coplanar (the board is flat) — that still uniquely fixes the
6-DOF transform, but spreading the touches widely improves accuracy.

Output: results/T_base_camera.npz   (same format the controller loads)

Run:  python handeye_touch.py
SAFETY: you jog the robot by hand/pendant; this script only READS poses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from se3 import Rt_to_T, pose_to_T, transform_point
from transforms import save_T_base_camera
from handeye_calibrate import CheckerboardTarget, find_chessboard

HERE = Path(__file__).parent


def deproject(u, v, Z, K):
    """Pixel + metric depth -> 3D point in the camera frame (m)."""
    return np.array([(u - K[0, 2]) * Z / K[0, 0],
                     (v - K[1, 2]) * Z / K[1, 1], Z], float)


def corners_depth_3d(color, depth_u16, target, K, depth_scale, win=2):
    """Detect checkerboard corners on `color`, return their 3D from the ALIGNED
    depth (robust to grazing angle / square-size errors, unlike solvePnP).

    Returns (ok, px (Nc,2), cam3d (Nc,3) with NaN where depth was missing)."""
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    found, corners = find_chessboard(gray, target.size)
    if not found:
        return False, None, None
    px = corners.reshape(-1, 2)
    H, W = depth_u16.shape
    out = np.full((len(px), 3), np.nan)
    for i, (u, v) in enumerate(px):
        ui, vi = int(round(u)), int(round(v))
        if not (win <= ui < W - win and win <= vi < H - win):
            continue
        patch = depth_u16[vi - win:vi + win + 1, ui - win:ui + win + 1].astype(float)
        vals = patch[patch > 0] * depth_scale
        if vals.size:
            out[i] = deproject(u, v, float(np.median(vals)), K)
    return True, px, out


def pick_targets_px(px, valid):
    """Choose well-spread, unambiguous targets by IMAGE position: the 4 extreme
    corners (TL,TR,BL,BR) + the most-central, among corners with valid depth."""
    idx = np.where(valid)[0]
    P = px[idx]
    s, d = P[:, 0] + P[:, 1], P[:, 0] - P[:, 1]
    c = P - P.mean(0)
    picks = [idx[np.argmin(s)], idx[np.argmax(d)], idx[np.argmin(d)],
             idx[np.argmax(s)], idx[np.argmin(np.linalg.norm(c, axis=1))]]
    # dedupe preserving order
    seen, out = set(), []
    for k in picks:
        if k not in seen:
            seen.add(k); out.append(int(k))
    return out


def kabsch(A, B):
    """Rigid transform (proper rotation) mapping A->B: B ~= R@A + t. (N,3)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    cA, cB = A.mean(0), B.mean(0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cB - R @ cA
    return R, t


def pick_targets(size):
    """Indices to touch: 4 extreme corners + center (well spread)."""
    cols, rows = size
    return [0, cols - 1, (rows - 1) * cols, rows * cols - 1, (rows // 2) * cols + cols // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--headless", action="store_true",
                    help="no cv2 window; saves results/touch_targets.png instead")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    import rtde_receive
    rtde_r = rtde_receive.RTDEReceiveInterface(cfg["robot"]["ip"])
    tool_offset = np.asarray(cfg["robot"].get("tool_tip_offset_m", [0, 0, 0]), float)
    if np.any(tool_offset):
        print(f"applying tool tip offset {tool_offset} m (tool frame)")

    from tip_sensor import D435Camera
    # Emitter ON for calibration: gives reliable depth on the low-texture board
    # (corner 3D comes from depth, not solvePnP).
    cam = D435Camera(cfg["camera"]["width"], cfg["camera"]["height"],
                     cfg["camera"]["fps"], laser_power=200)
    target = CheckerboardTarget(cfg["handeye"]["target"])

    print("Place the checkerboard FIXED on the table, fully in view, CLEAR of the "
          "robot. Press ENTER to lock in the camera detection…")
    input()
    # corner 3D from ALIGNED DEPTH, median over several frames
    acc = None; cnt = None; px = None; color = None
    for _ in range(15):
        color, depth = cam.get_frames()
        ok, p_px, p_cam = corners_depth_3d(color, depth, target, cam.K, cam.depth_scale)
        if not ok:
            continue
        px = p_px
        good = np.isfinite(p_cam).all(axis=1)
        if acc is None:
            acc = np.nan_to_num(p_cam); cnt = good.astype(float)
        else:
            acc += np.nan_to_num(p_cam); cnt += good.astype(float)
    if px is None:
        print("Checkerboard not detected. Check cols/rows/lighting and retry.")
        cam.close(); return
    with np.errstate(invalid="ignore"):
        cam_pts = acc / cnt[:, None]
    valid = cnt > 0
    print(f"Detected {len(px)} corners; {int(valid.sum())} have valid depth.")
    if valid.sum() < 4:
        print("Too few corners with depth. Turn on better lighting / move board "
              "into the depth range (>~0.2 m), keep it flat. Aborting.")
        cam.close(); return

    idxs = pick_targets_px(px, valid)
    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)

    def describe(p, shape):
        h, w = shape[:2]
        horiz = "LEFT" if p[0] < w / 3 else "RIGHT" if p[0] > 2 * w / 3 else "middle"
        vert = "FAR" if p[1] < h / 3 else "NEAR" if p[1] > 2 * h / 3 else "mid"
        return f"{vert}-{horiz}"

    # Build ONE labelled image of all targets (numbered), save it, then CLOSE the
    # camera so its stream can't crash during the long manual freedrive phase.
    board = color.copy()
    for q in px:
        cv2.circle(board, tuple(q.astype(int)), 2, (0, 150, 0), -1)
    descs = []
    for n, idx in enumerate(idxs):
        u, v = px[idx].astype(int)
        descs.append(describe(px[idx], color.shape))
        cv2.circle(board, (u, v), 11, (0, 0, 255), 2)
        cv2.putText(board, str(n + 1), (u + 12, v + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    tgt_png = out_dir / "touch_targets.png"
    cv2.imwrite(str(tgt_png), board)
    cam.close()   # <-- camera not needed any more; avoids stream timeout/crash

    print(f"\nSaved target map -> {tgt_png}")
    print("Open it once (VS Code explorer, or: xdg-open results/touch_targets.png).")
    print("It shows 5 numbered red targets. Freedrive the TCP tip to each IN ORDER:\n")
    for n, (idx, d) in enumerate(zip(idxs, descs)):
        print(f"   #{n+1}: board {d} corner")
    print()

    A_cam, B_base = [], []
    for n, idx in enumerate(idxs):
        print(f"[{n+1}/{len(idxs)}] freedrive TCP tip onto target #{n+1} "
              f"({descs[n]}), then ENTER (s=skip)…")
        if input().strip().lower() == "s":
            continue
        pose = rtde_r.getActualTCPPose()        # base <- tool0 (pendant TCP)
        # attachment tip in base = T_base_tool0 * tool_offset  (orientation-aware)
        tip_base = transform_point(pose_to_T(pose), tool_offset)
        A_cam.append(cam_pts[idx])
        B_base.append(tip_base)
        print(f"   camera={np.round(cam_pts[idx],3)} m   tip_base={np.round(tip_base,3)} m")

    if len(A_cam) < 3:
        print(f"Only {len(A_cam)} points — need >=3 non-collinear. Aborting.")
        return
    A_cam = np.array(A_cam); B_base = np.array(B_base)
    R, t = kabsch(A_cam, B_base)
    T_base_camera = Rt_to_T(R, t)
    resid = np.linalg.norm((A_cam @ R.T + t) - B_base, axis=1)

    # --- congruence diagnosis: same physical points => equal inter-distances ---
    def pdist(P):
        return np.array([np.linalg.norm(P[i] - P[j])
                         for i in range(len(P)) for j in range(i + 1, len(P))])
    dc, db = pdist(A_cam), pdist(B_base)
    ratio = dc / np.maximum(db, 1e-6)
    print("\n--- diagnosis ---")
    print(f"camera pairwise dist (mm): {np.round(dc*1000,1)}")
    print(f"base   pairwise dist (mm): {np.round(db*1000,1)}")
    print(f"ratio camera/base        : {np.round(ratio,2)}  (want all ~1.00)")
    print(f"base touch-plane z spread: {(B_base[:,2].max()-B_base[:,2].min())*1000:.1f} mm")
    if np.median(ratio) > 1.15 or np.median(ratio) < 0.87:
        print("-> camera distances scaled vs base: likely DEPTH BIAS at grazing angle "
              "or wrong correspondence. Make the board flatter & more head-on.")
    elif ratio.std() > 0.15:
        print("-> distances inconsistent: a touch hit the wrong corner, or the board "
              "moved after detection, or the TCP isn't the touched tip.")
    else:
        print("-> point sets are congruent; residual is touch noise. Good to use.")

    out = args.out or str(HERE / cfg["handeye"]["transform_file"])
    save_T_base_camera(out, T_base_camera,
                       {"residual_m": float(resid.mean()), "n_points": len(A_cam),
                        "method": "touch_point",
                        "cam_points": A_cam, "base_points": B_base})
    np.set_printoptions(precision=4, suppress=True)
    print(f"\nT_base_camera =\n{T_base_camera}")
    print(f"fit residual: mean {resid.mean()*1000:.2f} mm  max {resid.max()*1000:.2f} mm "
          f"({len(A_cam)} points) -> saved {out}")
    if resid.mean() > 0.005:
        print("WARNING: residual > 5 mm. Touch more precisely / set the pendant TCP "
              "to the exact pointer tip / spread touches wider.")


if __name__ == "__main__":
    main()
