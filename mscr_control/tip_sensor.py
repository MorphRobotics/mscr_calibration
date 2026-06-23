"""Live D435 camera + MSCR tip sensing.

`D435Camera` opens an aligned color+depth stream and exposes the color-camera
intrinsics (used by both tip sensing and hand-eye calibration).
`TipSensor` runs the existing real-time MSCRTracker on each frame and returns the
rod tip in the COLOR-CAMERA frame, in METRES.

The camera frame follows the RealSense convention: +Z forward (out of the lens),
+X right, +Y down.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# MSCRTracker lives at the repo root (mscr_calibration/mscr_tracker.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class D435Camera:
    def __init__(self, width=848, height=480, fps=30, laser_power=0):
        import pyrealsense2 as rs
        import time
        self.rs = rs
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        # Open with one self-heal: a prior unclean exit (e.g. force-quit) can leave
        # the device "busy"; a hardware_reset + retry clears it.
        last = None
        for attempt in range(2):
            self.pipeline = rs.pipeline()
            try:
                self.profile = self.pipeline.start(cfg)
                break
            except RuntimeError as e:
                last = e
                if "busy" in str(e).lower() and attempt == 0:
                    try:
                        devs = rs.context().query_devices()
                        if len(devs):
                            devs[0].hardware_reset()
                    except Exception:
                        pass
                    time.sleep(6)
                    continue
                raise
        else:
            raise last
        dev = self.profile.get_device()
        ds = dev.first_depth_sensor()
        self.depth_scale = ds.get_depth_scale()
        if ds.supports(rs.option.emitter_enabled):
            ds.set_option(rs.option.emitter_enabled, 0 if laser_power == 0 else 1)
            if laser_power > 0:
                ds.set_option(rs.option.laser_power, float(laser_power))
        self.align = rs.align(rs.stream.color)
        # prime + grab color intrinsics
        for _ in range(5):
            self.pipeline.wait_for_frames(timeout_ms=5000)
        frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms=5000))
        self.intr = frames.get_color_frame().get_profile().as_video_stream_profile().get_intrinsics()
        self.K = np.array([[self.intr.fx, 0, self.intr.ppx],
                           [0, self.intr.fy, self.intr.ppy],
                           [0, 0, 1.0]], dtype=np.float64)
        self.dist = np.array(self.intr.coeffs, dtype=np.float64)  # k1,k2,p1,p2,k3

    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms=5000))
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        return color, depth

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


class TipSensor:
    def __init__(self, camera: D435Camera, overrides: Optional[dict] = None):
        from mscr_tracker import MSCRTracker, TrackerParams
        self.cam = camera
        overrides = dict(overrides or {})
        # our own (non-TrackerParams) key: which image side the free tip is on
        self.tip_side = overrides.pop("tip_image_side", None)
        params = TrackerParams()
        for k, v in overrides.items():
            if hasattr(params, k):
                setattr(params, k, v)
            else:
                print(f"  (tracker override '{k}' ignored — no such param)")
        self.tracker = MSCRTracker(params)
        self.tracker.set_intrinsics(camera.intr)
        self.tracker.depth_scale = camera.depth_scale

    def _tip_from_result(self, res):
        """Tip (xyz mm) honoring tip_image_side: pick the centerline END point on
        the configured image side, overriding the tracker's base/tip choice."""
        if not self.tip_side:
            return np.asarray(res.tip_xyz_mm, dtype=float)
        cl_px = np.asarray(res.centerline_px).reshape(-1, 2)   # (N, col,row)
        cl_3d = np.asarray(res.centerline_3d).reshape(-1, 3)   # (N, x,y,z) mm
        if len(cl_px) < 2 or len(cl_3d) != len(cl_px):
            return np.asarray(res.tip_xyz_mm, dtype=float)
        col, row = cl_px[:, 0], cl_px[:, 1]
        end = {"top": int(np.argmin(row)), "bottom": int(np.argmax(row)),
               "left": int(np.argmin(col)), "right": int(np.argmax(col))}.get(
                   self.tip_side, -1)
        return cl_3d[end].astype(float)

    def read_tip(self, n_avg: int = 3, z_min: float = 0.1, z_max: float = 1.2
                 ) -> Tuple[Optional[np.ndarray], object]:
        """Return (tip_xyz_m in camera frame, last TrackResult).

        Averages the tip over a few frames; rejects implausible detections
        (depth outside [z_min, z_max] m). Returns (None, res) if none valid.
        """
        tips, last, rejected = [], None, 0
        for _ in range(max(1, n_avg)):
            color, depth = self.cam.get_frames()
            res = self.tracker.process_frame(color, depth)
            last = res
            t = self._tip_from_result(res) / 1000.0  # mm -> m (side-corrected)
            if res.valid and np.all(np.isfinite(t)) and any(t) and z_min <= t[2] <= z_max:
                tips.append(t)
            else:
                rejected += 1
        if not tips:
            if rejected:
                print(f"  tip rejected on all {rejected} frames "
                      f"(no valid detection in depth {z_min}-{z_max} m)")
            return None, last
        return np.mean(tips, axis=0), last
