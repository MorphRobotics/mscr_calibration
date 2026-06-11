#!/usr/bin/env python3
"""
MSCR RealSense IR Stereo Calibration Image Capture

Streams both IR cameras simultaneously and saves synchronised matched pairs to
  images/left/frame_<ts>.png
  images/right/frame_<ts>.png

IMPORTANT: the IR projector (dot emitter) is disabled automatically so the
           structured-light pattern does not corrupt checkerboard detection.

Controls:
  SPACE  - save pair (only when corners found in BOTH frames)
  D      - toggle depth map saving alongside IR pairs
  Q/ESC  - quit

Target: 40-60 matched pairs with varied board positions, tilts, and distances.
"""

import os, sys, time, json
import cv2
import numpy as np
import pyrealsense2 as rs
from config import (BOARD_COLS, BOARD_ROWS, CAMERA_WIDTH, CAMERA_HEIGHT,
                    CAMERA_FPS, LEFT_DIR, RIGHT_DIR, DEPTH_DIR, RESULTS_DIR)

# ──────────────────────────────────────────────────────────────────────────────
GUIDANCE_STAGES = [
    ( 0,  8, "Center frame — vary distance 30-100 cm"),
    ( 8, 16, "Tilt board LEFT and RIGHT while keeping all corners in view"),
    (16, 24, "Tilt board UP and DOWN"),
    (24, 32, "Aim corners of the IMAGE at the board centre"),
    (32, 40, "Close distance ~20-25 cm, mixed angles"),
    (40, 999,"Keep mixing position, tilt and distance — aim for 50+"),
]

# Display scale — show at half-res side by side without needing a huge monitor
DISPLAY_SCALE = 0.5


def get_guidance(n):
    for lo, hi, msg in GUIDANCE_STAGES:
        if lo <= n < hi:
            return msg
    return "Good coverage — keep going!"


def draw_panel(gray_img, found, corners, label):
    """Return a BGR panel with corner overlay and label."""
    bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    if found:
        cv2.drawChessboardCorners(bgr, (BOARD_COLS, BOARD_ROWS), corners, found)
        cv2.putText(bgr, f"{label}: FOUND", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 220, 20), 2)
    else:
        cv2.putText(bgr, f"{label}: not detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 220), 2)
    return bgr


def draw_status_bar(canvas, both_found, count, guidance):
    """Draw a status bar at the bottom of the combined display canvas."""
    h, w = canvas.shape[:2]
    bar_h = 70
    cv2.rectangle(canvas, (0, h - bar_h), (w, h), (0, 0, 0), -1)

    if both_found:
        status_text = f"BOTH CAMERAS READY  |  Press SPACE to save  ({count} pairs saved)"
        col = (20, 220, 20)
    else:
        status_text = f"Waiting for corners in both views...  ({count} pairs saved)"
        col = (40, 40, 220)

    cv2.putText(canvas, status_text, (10, h - bar_h + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
    cv2.putText(canvas, f"Hint: {guidance}", (10, h - bar_h + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 220, 60), 1)
    cv2.putText(canvas, "SPACE=save pair   D=toggle depth   Q=quit",
                (10, h - bar_h + 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
    return canvas


def save_firmware_intrinsics(profile):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def stream_intrinsics(stream_type, index=0):
        sp = profile.get_stream(stream_type, index).as_video_stream_profile()
        return sp.get_intrinsics()

    left_intr  = stream_intrinsics(rs.stream.infrared, 1)
    right_intr = stream_intrinsics(rs.stream.infrared, 2)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()

    # Extrinsics between left and right IR
    left_stream  = profile.get_stream(rs.stream.infrared, 1)
    right_stream = profile.get_stream(rs.stream.infrared, 2)
    extr = left_stream.get_extrinsics_to(right_stream)

    def intr_dict(intr):
        return dict(width=intr.width, height=intr.height,
                    fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy,
                    distortion_model=str(intr.model),
                    distortion_coeffs=list(intr.coeffs))

    data = dict(
        left_ir=intr_dict(left_intr),
        right_ir=intr_dict(right_intr),
        depth_scale_m_per_unit=depth_scale,
        firmware_extrinsics=dict(
            rotation=list(extr.rotation),      # 3×3 row-major
            translation=list(extr.translation) # metres
        )
    )
    path = os.path.join(RESULTS_DIR, "rs_firmware_intrinsics.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    baseline_mm = np.linalg.norm(extr.translation) * 1000
    print("\n── RealSense firmware intrinsics (saved to results/) ──────────────────")
    print(f"  LEFT  IR: fx={left_intr.fx:.2f}  fy={left_intr.fy:.2f}"
          f"  cx={left_intr.ppx:.2f}  cy={left_intr.ppy:.2f}")
    print(f"  RIGHT IR: fx={right_intr.fx:.2f}  fy={right_intr.fy:.2f}"
          f"  cx={right_intr.ppx:.2f}  cy={right_intr.ppy:.2f}")
    print(f"  Baseline (firmware): {baseline_mm:.2f} mm")
    print(f"  Depth scale: {depth_scale} m/unit")
    print("────────────────────────────────────────────────────────────────────────\n")
    return data


def main():
    for d in (LEFT_DIR, RIGHT_DIR, DEPTH_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1,
                      CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.y8, CAMERA_FPS)
    cfg.enable_stream(rs.stream.infrared, 2,
                      CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.y8, CAMERA_FPS)
    cfg.enable_stream(rs.stream.depth,
                      CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)

    try:
        profile = pipeline.start(cfg)
    except Exception as e:
        print(f"[ERROR] Could not start RealSense pipeline: {e}")
        print("  Check: camera connected, USB 3.0 port, udev rules installed.")
        sys.exit(1)

    # Disable the IR structured-light projector — essential for checkerboard detection
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.emitter_enabled, 0)
    print("IR emitter disabled (required for checkerboard calibration).")

    save_firmware_intrinsics(profile)

    save_depth   = False
    count        = 0
    detect_flags = cv2.CALIB_CB_FAST_CHECK
    criteria     = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("Capture window open. Follow the on-screen guidance.")
    print("Both IR images must show green overlays before SPACE will save.\n")

    dw = int(CAMERA_WIDTH  * DISPLAY_SCALE)
    dh = int(CAMERA_HEIGHT * DISPLAY_SCALE)

    try:
        while True:
            frames = pipeline.wait_for_frames()

            lf = frames.get_infrared_frame(1)
            rf = frames.get_infrared_frame(2)
            df = frames.get_depth_frame()

            if not lf or not rf:
                continue

            left_raw  = np.asanyarray(lf.get_data())
            right_raw = np.asanyarray(rf.get_data())
            depth_raw = np.asanyarray(df.get_data()) if df else None

            found_l, corners_l = cv2.findChessboardCorners(
                left_raw,  (BOARD_COLS, BOARD_ROWS), flags=detect_flags)
            found_r, corners_r = cv2.findChessboardCorners(
                right_raw, (BOARD_COLS, BOARD_ROWS), flags=detect_flags)

            both_found = bool(found_l and found_r)

            # Build side-by-side display (scaled down)
            panel_l = draw_panel(left_raw,  found_l, corners_l, "LEFT IR")
            panel_r = draw_panel(right_raw, found_r, corners_r, "RIGHT IR")
            panel_l = cv2.resize(panel_l, (dw, dh))
            panel_r = cv2.resize(panel_r, (dw, dh))

            # Add 70 px status bar at bottom
            canvas = np.zeros((dh + 70, dw * 2, 3), dtype=np.uint8)
            canvas[:dh, :dw]     = panel_l
            canvas[:dh, dw:dw*2] = panel_r
            canvas = draw_status_bar(canvas, both_found, count, get_guidance(count))

            cv2.imshow("MSCR IR Stereo Calibration Capture", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('d'):
                save_depth = not save_depth
                print(f"Depth saving: {'ON' if save_depth else 'OFF'}")
            elif key == ord(' '):
                if not both_found:
                    print("  Corners not found in both views — reposition and try again.")
                    continue

                # Subpixel refinement before saving
                cv2.cornerSubPix(left_raw,  corners_l, (11, 11), (-1, -1), criteria)
                cv2.cornerSubPix(right_raw, corners_r, (11, 11), (-1, -1), criteria)

                ts   = int(time.time() * 1000)
                stem = f"frame_{ts:015d}"

                cv2.imwrite(os.path.join(LEFT_DIR,  f"{stem}.png"), left_raw)
                cv2.imwrite(os.path.join(RIGHT_DIR, f"{stem}.png"), right_raw)

                if save_depth and depth_raw is not None:
                    np.save(os.path.join(DEPTH_DIR, f"{stem}.npy"), depth_raw)

                count += 1
                print(f"  [{count:3d}] saved pair {stem}"
                      + ("  (+depth)" if save_depth and depth_raw is not None else ""))

    finally:
        # Re-enable projector on exit (restore normal operation)
        try:
            depth_sensor.set_option(rs.option.emitter_enabled, 1)
        except Exception:
            pass
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\nCapture complete — {count} matched pair(s) saved.")
        print(f"  Left frames : {LEFT_DIR}/")
        print(f"  Right frames: {RIGHT_DIR}/")
        if count < 20:
            print("  WARNING: fewer than 20 pairs — aim for 40-60 for a robust stereo calibration.")
        print("Run:  python3 calibrate.py")


if __name__ == "__main__":
    main()
