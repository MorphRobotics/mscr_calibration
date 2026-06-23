#!/usr/bin/env python3
"""
approach_tip.py — verify the UR5e can reach the MSCR tip, and (optionally) make a
slow guarded approach to a standoff above it.

Pipeline:
  1. sense the MSCR tip with the D435 (camera frame, m)  [mscr_tracker]
  2. tip_base = T_base_camera * tip_camera                [hand-eye]
  3. target = tip_base + standoff (default 5 cm up in base Z)
  4. command flange pose so the ATTACHMENT TIP reaches target, accounting for the
     tool_tip_offset (orientation-aware): flange = target - R(tool_rotvec)*offset
  5. report reachability (workspace box + IK) and distance from the current TCP
  6. only with --move does it actually moveL (slow), after a typed 'yes'

Modes:
    python approach_tip.py            # SENSE + report only, never moves
    python approach_tip.py --move     # also do the guarded approach to standoff
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from se3 import pose_to_T, T_to_pose
from scipy.spatial.transform import Rotation as Rot
from transforms import load_T_base_camera, tip_camera_to_base

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--standoff-m", type=float, default=0.05,
                    help="height above the tip to approach (base +Z), default 5 cm")
    ap.add_argument("--move", action="store_true", help="actually moveL to the standoff")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    # approach-specific safety box (falls back to the robot box if absent)
    ap_cfg = cfg.get("approach", {})
    lo = np.asarray(ap_cfg.get("workspace_min", cfg["robot"]["workspace_min"]), float)
    hi = np.asarray(ap_cfg.get("workspace_max", cfg["robot"]["workspace_max"]), float)
    max_move = float(ap_cfg.get("max_move_m", 0.6))
    tool_rotvec = np.asarray(cfg["robot"]["tool_rotvec"], float)
    offset = np.asarray(cfg["robot"].get("tool_tip_offset_m", [0, 0, 0]), float)

    T_base_camera = load_T_base_camera(
        HERE / cfg["handeye"]["transform_file"]
        if not Path(cfg["handeye"]["transform_file"]).is_absolute()
        else cfg["handeye"]["transform_file"])

    import rtde_receive
    rtde_r = rtde_receive.RTDEReceiveInterface(cfg["robot"]["ip"])
    cur = np.asarray(rtde_r.getActualTCPPose(), float)

    from tip_sensor import D435Camera, TipSensor
    cam = D435Camera(cfg["camera"]["width"], cfg["camera"]["height"],
                     cfg["camera"]["fps"], cfg["camera"]["laser_power"])
    tip = TipSensor(cam, cfg.get('tracker'))
    print("Sensing MSCR tip (need a clear view of the rod)…")
    tip_cam, _ = tip.read_tip(n_avg=5)
    cam.close()
    if tip_cam is None:
        print("MSCR tip not detected. Improve lighting / framing and retry.")
        return

    tip_base = tip_camera_to_base(T_base_camera, tip_cam)
    target = tip_base + np.array([0, 0, args.standoff_m])
    # flange pose so the attachment tip sits at `target`
    R_tool = Rot.from_rotvec(tool_rotvec).as_matrix()
    flange_pos = target - R_tool @ offset
    pose = [*flange_pos.tolist(), *tool_rotvec.tolist()]

    in_box = bool(np.all(flange_pos >= lo) and np.all(flange_pos <= hi))
    # Single RTDE control connection, reused for the safety check AND the move
    # (creating two at once trips "RTDE input registers already in use").
    rc = None
    reachable_ik = True
    try:
        import rtde_control
        rc = rtde_control.RTDEControlInterface(cfg["robot"]["ip"])
        reachable_ik = rc.isPoseWithinSafetyLimits(pose) if hasattr(
            rc, "isPoseWithinSafetyLimits") else True
    except Exception as e:
        reachable_ik = None
        print(f"(IK/limits check skipped: {str(e)[:80]})")

    np.set_printoptions(precision=4, suppress=True)
    print("\n=== APPROACH REPORT ===")
    print(f"  MSCR tip  (camera) : {np.round(tip_cam,4)} m")
    print(f"  MSCR tip  (base)   : {np.round(tip_base,4)} m")
    print(f"  standoff target    : {np.round(target,4)} m  (+{args.standoff_m*100:.0f} cm Z)")
    print(f"  flange pose target : {np.round(np.asarray(pose),4)}")
    print(f"  current TCP pose   : {np.round(cur,4)}")
    print(f"  move distance      : {np.linalg.norm(flange_pos - cur[:3])*1000:.0f} mm")
    print(f"  within workspace box: {in_box}   ({lo} .. {hi})")
    print(f"  within safety limits: {reachable_ik}")
    if not in_box:
        print("  -> target outside the workspace box. If the box is wrong for your "
              "cell, edit robot.workspace_min/max; do NOT move until it's right.")

    move_dist = np.linalg.norm(flange_pos - cur[:3])
    print(f"  move within max_move ({max_move} m): {move_dist <= max_move}")

    def cleanup():
        if rc is not None:
            try: rc.stopScript()
            except Exception: pass
            try: rc.disconnect()
            except Exception: pass

    if not args.move:
        print("\n(report only — pass --move to approach the standoff)")
        cleanup(); return
    if not in_box:
        print("\nRefusing to move: target outside the approach safety box "
              "(edit approach.workspace_min/max if the box is wrong).")
        cleanup(); return
    if reachable_ik is False:
        print("\nRefusing to move: robot reports pose outside safety limits.")
        cleanup(); return
    if move_dist > max_move:
        print(f"\nRefusing to move: {move_dist*1000:.0f} mm exceeds max_move "
              f"{max_move*1000:.0f} mm (raise approach.max_move_m if intended).")
        cleanup(); return
    if rc is None:
        print("\nNo robot control connection; cannot move."); return
    if input("\nType 'yes' to slowly approach the standoff above the tip: ").strip().lower() != "yes":
        print("aborted."); cleanup(); return
    try:
        rc.moveL(pose, cfg["robot"]["tcp_speed"], cfg["robot"]["tcp_accel"])
        print("reached standoff above the MSCR tip.")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
