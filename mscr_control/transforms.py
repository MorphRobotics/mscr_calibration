"""Frame composition for the camera / MSCR-tip / UR5e system.

Frames
------
- base   : UR5e robot base (RTDE poses are base<-tool0).
- tool0  : UR5e tool flange (the magnet is rigidly attached here).
- camera : D435 color-camera optical frame (RealSense convention).
- tip    : MSCR tip point (sensed by the camera).

Known at runtime
----------------
- T_base_tool0 : from RTDE getActualTCPPose() (robot kinematics).
- tip_camera   : from TipSensor (camera frame, metres).
- T_base_camera: from hand-eye calibration (handeye_calibrate.py), loaded here.

The camera is FIXED in the cell (eye-to-hand), so T_base_camera is constant.
The MSCR tip in the base frame is therefore:
        tip_base = T_base_camera @ tip_camera
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from se3 import transform_point


def load_T_base_camera(path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"hand-eye transform not found: {p}\n"
            f"Run:  python handeye_calibrate.py   (with the robot + ChArUco board)")
    data = np.load(p)
    T = data["T_base_camera"]
    assert T.shape == (4, 4), f"bad T_base_camera shape {T.shape}"
    return T


def tip_camera_to_base(T_base_camera: np.ndarray, tip_camera_m) -> np.ndarray:
    """MSCR tip (camera frame, m) -> base frame (m)."""
    return transform_point(T_base_camera, tip_camera_m)


def save_T_base_camera(path, T_base_camera: np.ndarray, meta: dict | None = None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, T_base_camera=np.asarray(T_base_camera, dtype=float),
             **(meta or {}))
    return p
