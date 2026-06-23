#!/usr/bin/env python3
"""
trace_circle.py — drive the UR5e magnet so the MSCR TIP traces a circle.

Closed loop (per waypoint):
    1. read the MSCR tip from the D435 (camera frame, m)
    2. transform it to the base frame via the hand-eye T_base_camera
    3. error = desired_circle_point - tip_base
    4. desired tip-delta = feedforward(circle offset) + kp * error
    5. inverse magnet model -> magnet/TCP position (base frame)
    6. low-pass, SAFETY-CLAMP to the workspace box, moveL the UR5e
    7. settle; iterate until |error| < tol or max_iter, then advance.

Frames: base = UR base; camera = D435 color optical; tip = MSCR tip.
The inverse model's tip-delta axes are the BASE x/y (matches mscr_inv_control),
so the default circle plane is base_xy.

Run modes:
    python trace_circle.py --dry-run        # no robot, no camera: print the plan
                                            # + simulated loop (geometry check)
    python trace_circle.py --no-move        # real camera + transforms, NO robot motion
    python trace_circle.py                  # FULL closed loop (asks to confirm first)

SAFETY: full mode moves a real robot. The script clamps every command to
robot.workspace_min/max, runs slowly, prints the plan and waits for an explicit
'yes', and always stopScript()s on exit. Keep the e-stop in hand.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import yaml

from se3 import pose_to_T
from inverse_model import InverseMagnetModel
from transforms import load_T_base_camera, tip_camera_to_base

HERE = Path(__file__).parent
_PLANE_AXES = {"base_xy": (0, 1), "base_xz": (0, 2), "base_yz": (1, 2)}


def circle_points(center, radius, plane, n):
    """n points (closed loop) on a circle of given radius in the chosen base plane."""
    a0, a1 = _PLANE_AXES[plane]
    pts = np.tile(np.asarray(center, float), (n, 1))
    for k in range(n):
        th = 2 * np.pi * k / n
        pts[k, a0] = center[a0] + radius * np.cos(th)
        pts[k, a1] = center[a1] + radius * np.sin(th)
    return pts


def clamp_to_box(p, lo, hi):
    p = np.asarray(p, float)
    return np.clip(p, np.asarray(lo, float), np.asarray(hi, float))


def in_box(p, lo, hi):
    p = np.asarray(p, float)
    return bool(np.all(p >= np.asarray(lo, float)) and np.all(p <= np.asarray(hi, float)))


class CircleTracer:
    def __init__(self, cfg, dry_run=False, no_move=False):
        self.cfg = cfg
        self.dry_run = dry_run
        self.no_move = no_move
        self.model = InverseMagnetModel(cfg["magnet_model"]["onnx"],
                                        cfg["magnet_model"]["norm"],
                                        cfg["magnet_model"])
        self.tool_rotvec = np.asarray(cfg["robot"]["tool_rotvec"], float)
        self.lo = cfg["robot"]["workspace_min"]
        self.hi = cfg["robot"]["workspace_max"]
        self.prev_pm = None

        # hardware (skipped in dry-run)
        self.rtde_c = self.rtde_r = self.cam = self.tip = None
        self.T_base_camera = None
        if not dry_run:
            self.T_base_camera = load_T_base_camera(
                HERE / cfg["handeye"]["transform_file"]
                if not Path(cfg["handeye"]["transform_file"]).is_absolute()
                else cfg["handeye"]["transform_file"])
            from tip_sensor import D435Camera, TipSensor
            self.cam = D435Camera(cfg["camera"]["width"], cfg["camera"]["height"],
                                  cfg["camera"]["fps"], cfg["camera"]["laser_power"])
            self.tip = TipSensor(self.cam)
            import rtde_receive
            self.rtde_r = rtde_receive.RTDEReceiveInterface(cfg["robot"]["ip"])
            if not no_move:
                import rtde_control
                self.rtde_c = rtde_control.RTDEControlInterface(cfg["robot"]["ip"])

    # ── tip sensing ──────────────────────────────────────────────────────────
    def read_tip_base(self):
        if self.dry_run:
            return self._sim_tip
        tip_cam, _ = self.tip.read_tip()
        if tip_cam is None:
            return None
        return tip_camera_to_base(self.T_base_camera, tip_cam)

    # ── magnet command ───────────────────────────────────────────────────────
    def command_magnet(self, tip_delta):
        pm = self.model.magnet_position(tip_delta)
        a = self.cfg["control"]["lpf_alpha"]
        pm = pm if self.prev_pm is None else a * pm + (1 - a) * self.prev_pm
        self.prev_pm = pm
        safe = clamp_to_box(pm, self.lo, self.hi)
        if not in_box(pm, self.lo, self.hi):
            print(f"   ! magnet target {np.round(pm,3)} outside workspace box -> clamped")
        pose = [*safe.tolist(), *self.tool_rotvec.tolist()]
        if self.dry_run:
            # crude sim: the tip follows a fraction of the commanded delta
            self._sim_tip = self._sim_center + 0.9 * np.asarray(tip_delta, float)
        elif not self.no_move:
            self.rtde_c.moveL(pose, self.cfg["robot"]["tcp_speed"],
                              self.cfg["robot"]["tcp_accel"])
        return safe, pose

    # ── main trace ───────────────────────────────────────────────────────────
    def run(self):
        cc = self.cfg["circle"]; ctrl = self.cfg["control"]
        # circle center
        if isinstance(cc["center"], str) and cc["center"] == "current_tip":
            if self.dry_run:
                center = np.array([-0.144, -0.436, 0.20])
                self._sim_center = center.copy(); self._sim_tip = center.copy()
            else:
                center = self.read_tip_base()
                if center is None:
                    raise RuntimeError("could not sense the MSCR tip to set the circle center")
        else:
            center = np.asarray(cc["center"], float)
            if self.dry_run:
                self._sim_center = center.copy(); self._sim_tip = center.copy()
        a0, a1 = _PLANE_AXES[cc["plane"]]

        n = int(cc["steps_per_rev"] * cc["revolutions"])
        targets = circle_points(center, cc["radius_m"], cc["plane"],
                                int(cc["steps_per_rev"]))
        targets = np.array([targets[k % len(targets)] for k in range(n)])

        print("\n=== TRACE PLAN ===")
        print(f"  circle center (tip, base) : {np.round(center,4)} m")
        print(f"  radius={cc['radius_m']*1000:.1f} mm  plane={cc['plane']}  "
              f"revs={cc['revolutions']}  waypoints={n}")
        print(f"  workspace box (magnet)    : {self.lo} .. {self.hi} m")
        print(f"  kp={ctrl['kp']}  tol={ctrl['pos_tol_m']*1000:.1f} mm  "
              f"settle={ctrl['settle_s']}s  speed={self.cfg['robot']['tcp_speed']} m/s")
        mode = "DRY-RUN (no hardware)" if self.dry_run else (
            "NO-MOVE (camera only)" if self.no_move else "FULL CLOSED LOOP (robot will move)")
        print(f"  mode: {mode}\n")
        if not self.dry_run and not self.no_move:
            if input("Type 'yes' to start moving the robot: ").strip().lower() != "yes":
                print("aborted."); return

        log_path = HERE / "results" / "trace_circle_log.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(log_path, "w", newline="")
        w = csv.writer(f)
        w.writerow(["step", "des_x", "des_y", "des_z", "tip_x", "tip_y", "tip_z",
                    "err_mm", "mag_x", "mag_y", "mag_z"])
        try:
            for k, des in enumerate(targets):
                offset = des - center                      # feedforward tip delta
                for _ in range(int(ctrl["max_iter_per_waypoint"])):
                    tip = self.read_tip_base()
                    if tip is None:
                        print(f"step {k}: tip not detected, retrying…")
                        time.sleep(0.2); continue
                    err = des - tip
                    err_n = float(np.linalg.norm(err))
                    tip_delta = offset + ctrl["kp"] * err
                    mag, _pose = self.command_magnet(tip_delta)
                    w.writerow([k, *np.round(des, 5), *np.round(tip, 5),
                                round(err_n * 1000, 2), *np.round(mag, 5)])
                    if err_n < ctrl["pos_tol_m"]:
                        break
                    if not self.dry_run:
                        time.sleep(ctrl["settle_s"])
                if k % max(1, n // 12) == 0:
                    print(f"  step {k:3d}/{n}  desired={np.round(des,4)}  "
                          f"tip={np.round(tip,4)}  err={err_n*1000:.1f} mm")
            print(f"\ndone. log -> {log_path}")
        finally:
            f.close()
            self.close()

    def close(self):
        if self.rtde_c is not None:
            try: self.rtde_c.stopScript()
            except Exception: pass
        if self.cam is not None:
            self.cam.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--dry-run", action="store_true",
                    help="no robot, no camera — print plan + simulated geometry loop")
    ap.add_argument("--no-move", action="store_true",
                    help="real camera + transforms, but do NOT move the robot")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tracer = CircleTracer(cfg, dry_run=args.dry_run, no_move=args.no_move)
    tracer.run()


if __name__ == "__main__":
    main()
