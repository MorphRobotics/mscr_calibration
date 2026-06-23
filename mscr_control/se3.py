"""Minimal SE(3) helpers shared across the mscr_control scripts.

Conventions
-----------
- A pose/transform is a 4x4 homogeneous matrix T such that  p_A = T_A_B @ p_B
  (T_A_B maps a point expressed in frame B into frame A).
- UR / RTDE Cartesian poses are [x, y, z, rx, ry, rz] with the rotation as a
  rotation VECTOR (axis * angle), metres + radians. Use pose_to_T / T_to_pose.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def pose_to_T(pose) -> np.ndarray:
    """UR pose [x,y,z,rx,ry,rz] (rotvec) -> 4x4 homogeneous matrix."""
    pose = np.asarray(pose, dtype=float)
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(pose[3:6]).as_matrix()
    T[:3, 3] = pose[0:3]
    return T


def T_to_pose(T: np.ndarray) -> list:
    """4x4 homogeneous matrix -> UR pose [x,y,z,rx,ry,rz] (rotvec)."""
    T = np.asarray(T, dtype=float)
    rv = R.from_matrix(T[:3, :3]).as_rotvec()
    return [*T[:3, 3].tolist(), *rv.tolist()]


def Rt_to_T(Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(Rm, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    Ri = T[:3, :3].T
    ti = -Ri @ T[:3, 3]
    return Rt_to_T(Ri, ti)


def transform_point(T: np.ndarray, p_xyz) -> np.ndarray:
    """Apply 4x4 T to a 3-vector point."""
    p = np.asarray(p_xyz, dtype=float).reshape(3)
    return (np.asarray(T, dtype=float) @ np.array([p[0], p[1], p[2], 1.0]))[:3]


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 4x4 T to an (N,3) array of points."""
    pts = np.asarray(pts, dtype=float)
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (h @ np.asarray(T, dtype=float).T)[:, :3]
