#!/usr/bin/env python3
"""
handeye_calibrate.py — eye-to-hand calibration for the fixed D435 + UR5e.

Setup (eye-to-hand): the camera is STATIONARY in the cell; a calibration target
(a plain CHECKERBOARD by default, or ChArUco) is rigidly attached to the UR5e
tool flange. We move the robot to N varied poses, and at each pose record:
    - T_base_tool0   (from RTDE)
    - T_camera_target (target pose in the camera, from cv2 solvePnP)
Then solve for the constant camera pose in the base frame, T_base_camera, using
the eye-to-hand trick: feed cv2.calibrateHandEye the *inverted* gripper poses
(base<-tool0 becomes tool0<-base) together with target<-camera, so the returned
X is camera<-base, i.e. T_base_camera.

Usage:
    python handeye_calibrate.py                 # interactive: jog robot, press
                                                # ENTER to capture each pose
    python handeye_calibrate.py --poses poses.npy   # auto: moveL through saved poses

Output: results/T_base_camera.npz  (T_base_camera 4x4 + residual + metadata)

SAFETY: in auto mode the robot moves itself. Keep the e-stop in hand. In
interactive mode you jog with the teach pendant and this script only reads poses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from se3 import pose_to_T, invert_T, Rt_to_T
from transforms import save_T_base_camera

HERE = Path(__file__).parent


# ── Calibration target: plain checkerboard (default) or ChArUco ─────────────
_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def find_chessboard(gray, size):
    """Robust checkerboard corner finder. Prefers the sector-based detector
    (findChessboardCornersSB, robust to perspective/ripple/lighting); falls back
    to the classic detector. Returns (found, corners[N,1,2] float32)."""
    if hasattr(cv2, "findChessboardCornersSB"):
        flg = (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE |
               cv2.CALIB_CB_ACCURACY)
        found, corners = cv2.findChessboardCornersSB(gray, size, flags=flg)
        if found:
            return True, corners          # SB corners are already subpixel
    f2 = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, size, f2)
    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX)
    return found, corners


class CheckerboardTarget:
    """Plain checkerboard. cols/rows = INNER corner counts (squares-1)."""

    def __init__(self, tcfg):
        self.size = (int(tcfg["cols"]), int(tcfg["rows"]))   # (w, h) inner corners
        s = float(tcfg["square_len_m"])
        objp = np.zeros((self.size[0] * self.size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.size[0], 0:self.size[1]].T.reshape(-1, 2) * s
        self.objp = objp

    def detect(self, gray, K, dist):
        found, corners = find_chessboard(gray, self.size)
        if not found:
            return False, None, None
        ok, rvec, tvec = cv2.solvePnP(self.objp, corners, K, dist)
        return bool(ok), rvec, tvec


class CharucoTarget:
    def __init__(self, tcfg):
        dict_id = getattr(cv2.aruco, tcfg["aruco_dict"])
        self.aruco_dict = cv2.aruco.Dictionary_get(dict_id)
        self.board = cv2.aruco.CharucoBoard_create(
            tcfg["squares_x"], tcfg["squares_y"],
            tcfg["square_len_m"], tcfg["marker_len_m"], self.aruco_dict)

    def detect(self, gray, K, dist):
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=params)
        if ids is None or len(ids) < 4:
            return False, None, None
        n, ch_c, ch_i = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
        if n is None or n < 6:
            return False, None, None
        rvec = np.zeros((3, 1)); tvec = np.zeros((3, 1))
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            ch_c, ch_i, self.board, K, dist, rvec, tvec)
        return bool(ok), rvec, tvec


def make_target(tcfg):
    t = tcfg.get("type", "checkerboard")
    if t == "checkerboard":
        return CheckerboardTarget(tcfg)
    if t == "charuco":
        return CharucoTarget(tcfg)
    raise ValueError(f"unknown target type: {t}")


def solve(R_g2b, t_g2b, R_t2c, t_t2c):
    """Eye-to-hand: invert gripper->base so X = camera->base = T_base_camera."""
    R_b2g, t_b2g = [], []
    for Rg, tg in zip(R_g2b, t_g2b):
        T = invert_T(Rt_to_T(Rg, tg))
        R_b2g.append(T[:3, :3]); t_b2g.append(T[:3, 3])
    R_x, t_x = cv2.calibrateHandEye(
        R_b2g, t_b2g, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI)
    return Rt_to_T(R_x, t_x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--poses", default=None,
                    help="optional .npy of UR poses (N,6) to auto-moveL through")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    import rtde_receive
    rtde_r = rtde_receive.RTDEReceiveInterface(cfg["robot"]["ip"])
    rtde_c = None
    auto_poses = None
    if args.poses:
        import rtde_control
        rtde_c = rtde_control.RTDEControlInterface(cfg["robot"]["ip"])
        auto_poses = np.load(args.poses)

    from tip_sensor import D435Camera
    cam = D435Camera(cfg["camera"]["width"], cfg["camera"]["height"],
                     cfg["camera"]["fps"], cfg["camera"]["laser_power"])
    target = make_target(cfg["handeye"]["target"])

    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    n_target = auto_poses.shape[0] if auto_poses is not None else cfg["handeye"]["n_poses"]
    print(f"Collecting {n_target} poses for eye-to-hand calibration.")
    print("Vary the board orientation a lot between poses for a good solve.\n")

    try:
        for i in range(n_target):
            if auto_poses is not None:
                rtde_c.moveL(auto_poses[i].tolist(),
                             cfg["robot"]["tcp_speed"], cfg["robot"]["tcp_accel"])
            else:
                input(f"[{i+1}/{n_target}] Jog the robot to a new pose, then ENTER…")

            color, _ = cam.get_frames()
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            ok, rvec, tvec = target.detect(gray, cam.K, cam.dist)
            if not ok:
                print("   board not detected — reposition and retry this index.")
                if auto_poses is None:
                    # let the user retry the same index
                    while not ok:
                        input("   ENTER to retry…")
                        color, _ = cam.get_frames()
                        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                        ok, rvec, tvec = target.detect(gray, cam.K, cam.dist)
                else:
                    continue

            pose = rtde_r.getActualTCPPose()           # base <- tool0
            Tgb = pose_to_T(pose)
            R_g2b.append(Tgb[:3, :3]); t_g2b.append(Tgb[:3, 3])
            Rt, _ = cv2.Rodrigues(rvec)
            R_t2c.append(Rt); t_t2c.append(tvec.reshape(3))
            print(f"   captured. board t_cam = {np.round(tvec.reshape(3), 3)} m")

        if len(R_g2b) < 3:
            print("Not enough valid poses (<3). Aborting.")
            return

        T_base_camera = solve(R_g2b, t_g2b, R_t2c, t_t2c)

        # residual: target position in base should be consistent across poses
        pts = []
        for Rg, tg, Rc, tc in zip(R_g2b, t_g2b, R_t2c, t_t2c):
            T_base_target = T_base_camera @ Rt_to_T(Rc, tc)
            pts.append(T_base_target[:3, 3])
        pts = np.array(pts)
        resid = float(np.linalg.norm(pts - pts.mean(0), axis=1).mean())

        out = args.out or str(HERE / cfg["handeye"]["transform_file"])
        save_T_base_camera(out, T_base_camera,
                           {"residual_m": resid, "n_poses": len(R_g2b)})
        np.set_printoptions(precision=4, suppress=True)
        print(f"\nT_base_camera =\n{T_base_camera}")
        print(f"target-consistency residual = {resid*1000:.2f} mm "
              f"({len(R_g2b)} poses)  -> saved {out}")
        if resid > 0.005:
            print("WARNING: residual > 5 mm — add more/varied poses; check board size.")
    finally:
        cam.close()
        if rtde_c is not None:
            rtde_c.stopScript()


if __name__ == "__main__":
    main()
