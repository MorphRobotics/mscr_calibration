"""Stereo calibration for the D435 left/right IR pair.

Loads per-camera intrinsics + distortion and stereo extrinsics from a YAML
file, then computes everything downstream code needs:

    * rectification rotations / maps for cv2.remap
    * rectified projection matrices P1, P2 (left-IR rectified camera frame)
    * the fundamental matrix F (for the *unrectified* images)
    * Q (disparity-to-depth reprojection matrix)

Frame convention: the LEFT-IR camera is the reference. After rectification
the epipolar lines are horizontal scanlines, so a left point at row v matches
a right point at the same row v. All 3D quantities are in mm in the
**left-IR rectified camera frame**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import yaml


def _K(cam: dict) -> np.ndarray:
    return np.array(
        [[cam["fx"], 0.0, cam["cx"]],
         [0.0, cam["fy"], cam["cy"]],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _dist(cam: dict) -> np.ndarray:
    """Reorder our (k1,k2,k3,p1,p2) convention into OpenCV (k1,k2,p1,p2,k3)."""
    k1, k2, k3, p1, p2 = cam["distortion"]
    return np.array([k1, k2, p1, p2, k3], dtype=np.float64)


@dataclass
class StereoCalib:
    image_size: Tuple[int, int]          # (width, height)
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray                        # right-relative-to-left rotation
    T: np.ndarray                        # right-relative-to-left translation (mm)

    # filled in by __post_init__
    R1: np.ndarray = field(default=None)
    R2: np.ndarray = field(default=None)
    P1: np.ndarray = field(default=None)
    P2: np.ndarray = field(default=None)
    Q: np.ndarray = field(default=None)
    map1x: np.ndarray = field(default=None)
    map1y: np.ndarray = field(default=None)
    map2x: np.ndarray = field(default=None)
    map2y: np.ndarray = field(default=None)
    F: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        w, h = self.image_size
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, (w, h), self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, self.P1, (w, h), cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, self.P2, (w, h), cv2.CV_32FC1)
        self.F = self._fundamental()

    def _fundamental(self) -> np.ndarray:
        """Fundamental matrix for the ORIGINAL (unrectified) image pair."""
        Tx = np.array([[0, -self.T[2], self.T[1]],
                       [self.T[2], 0, -self.T[0]],
                       [-self.T[1], self.T[0], 0]], dtype=np.float64)
        E = Tx @ self.R
        F = np.linalg.inv(self.K2).T @ E @ np.linalg.inv(self.K1)
        # F is defined only up to scale; F[2,2] is 0 for pure-horizontal
        # stereo, so normalize by the largest-magnitude entry instead.
        return F / np.max(np.abs(F))

    def rectify_left(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.map1x, self.map1y, cv2.INTER_LINEAR)

    def rectify_right(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.map2x, self.map2y, cv2.INTER_LINEAR)

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.T))


def load_calib(path: str | Path) -> StereoCalib:
    with open(path, "r") as f:
        c = yaml.safe_load(f)
    if not all(k in c for k in ("left", "right", "R", "T")):
        raise ValueError(
            f"{path} is not a stereo calibration (needs left/right/R/T). "
            "It looks monocular — generate a stereo calib with "
            "`python calib.py --from-device --save <file>` (reads the D435 "
            "factory IR-pair calibration)."
        )
    return StereoCalib(
        image_size=tuple(c["image_size"]),
        K1=_K(c["left"]), D1=_dist(c["left"]),
        K2=_K(c["right"]), D2=_dist(c["right"]),
        R=np.array(c["R"], dtype=np.float64),
        T=np.array(c["T"], dtype=np.float64).reshape(3),
    )


def resolve_calib(cfg: dict) -> StereoCalib:
    """Load the stereo calib named in config (path relative to this package)."""
    p = Path(cfg["paths"]["calib"])
    if not p.is_absolute():
        p = Path(__file__).parent / p
    if not p.exists():
        raise FileNotFoundError(
            f"stereo calib not found at {p}. Generate it from the connected "
            "D435 with:  python calib.py --from-device --save calib_stereo.yaml"
        )
    return load_calib(p)


def nominal_calib(width: int = 1280, height: int = 720) -> StereoCalib:
    """Nominal D435 IR-pair stereo calib (f~634 px, baseline 49.5 mm, no
    distortion). For self-tests only — real runs use the device/file calib."""
    K = np.array([[634.0, 0, width / 2], [0, 634.0, height / 2], [0, 0, 1]])
    D = np.zeros(5)
    return StereoCalib(image_size=(width, height), K1=K.copy(), D1=D.copy(),
                       K2=K.copy(), D2=D.copy(),
                       R=np.eye(3), T=np.array([-49.5, 0.0, 0.0]))


def _cam_dict(K: np.ndarray, D: np.ndarray) -> dict:
    """Serialize intrinsics + distortion in our (k1,k2,k3,p1,p2) YAML order."""
    k1, k2, p1, p2, k3 = [float(x) for x in D[:5]]
    return {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "distortion": [k1, k2, k3, p1, p2]}


def from_realsense(width: int = 1280, height: int = 720) -> StereoCalib:
    """Build a StereoCalib from the D435's factory left/right IR calibration.

    Reads per-stream intrinsics and the left->right extrinsics live from the
    device via pyrealsense2. The IR pair is factory-calibrated (baseline
    ~49.5 mm). LEFT IR = infrared stream index 1, RIGHT IR = index 2.
    """
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, 30)
    profile = pipeline.start(config)
    try:
        sp1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        sp2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        i1, i2 = sp1.get_intrinsics(), sp2.get_intrinsics()
        ext = sp1.get_extrinsics_to(sp2)  # transform a point from left -> right
    finally:
        pipeline.stop()

    def K(i):
        return np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]], dtype=np.float64)

    # librealsense intrinsic coeffs are [k1,k2,p1,p2,k3] (OpenCV order already)
    D1 = np.array(i1.coeffs, dtype=np.float64)
    D2 = np.array(i2.coeffs, dtype=np.float64)
    # rotation is column-major 3x3; translation in metres -> mm
    R = np.array(ext.rotation, dtype=np.float64).reshape(3, 3, order="F")
    T = np.array(ext.translation, dtype=np.float64) * 1000.0
    return StereoCalib(image_size=(width, height), K1=K(i1), D1=D1,
                       K2=K(i2), D2=D2, R=R, T=T)


def save_calib(calib: StereoCalib, path: str | Path) -> None:
    """Write a StereoCalib to our stereo YAML format."""
    out = {
        "image_size": [int(calib.image_size[0]), int(calib.image_size[1])],
        "left": _cam_dict(calib.K1, calib.D1),
        "right": _cam_dict(calib.K2, calib.D2),
        "R": calib.R.tolist(),
        "T": calib.T.reshape(3).tolist(),
    }
    with open(path, "w") as f:
        yaml.safe_dump(out, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Calibration smoke test / device export")
    ap.add_argument("--calib", default=str(Path(__file__).parent / "calib_stereo.yaml"))
    ap.add_argument("--from-device", action="store_true",
                    help="read the D435 factory IR-pair calibration")
    ap.add_argument("--save", default=None,
                    help="with --from-device, write the stereo calib to this YAML")
    args = ap.parse_args()

    if args.from_device:
        calib = from_realsense()
        if args.save:
            save_calib(calib, args.save)
            print(f"saved stereo calib -> {args.save}")
    else:
        calib = load_calib(args.calib)
    print(f"image_size  : {calib.image_size}")
    print(f"baseline    : {calib.baseline_mm:.2f} mm")
    print(f"P1:\n{calib.P1}")
    print(f"P2:\n{calib.P2}")
    print(f"focal (P1)  : {calib.P1[0, 0]:.2f} px")
    print(f"map shapes  : {calib.map1x.shape}, {calib.map2x.shape}")
    print(f"F:\n{calib.F}")
    # sanity: P2 should encode the -baseline*f translation in the x term
    print(f"P2[0,3]/P1[0,0] (≈ -baseline mm) : {calib.P2[0, 3] / calib.P1[0, 0]:.2f}")
    print("OK")
