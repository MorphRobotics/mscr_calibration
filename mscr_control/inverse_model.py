"""Inverse magnet model: desired MSCR tip-delta -> magnet (TCP) position.

Wraps the trained ONNX MLP + normalization from the magnetcontrol_ws
`magnet_control` package, and reproduces the exact magnet-workspace mapping used
by the proven mscr_inv_control.py node, so the closed-loop controller commands
the magnet the same way the open-loop node did.

Model: input  [N,3] = desired tip delta (m, base frame)
       output [N,3] = (B magnitude, azimuth az, elevation el)
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import scipy.io as sio


def _load_norm(path: str) -> dict:
    """Load normalization stats, handling either flat keys (mu_in, sig_in, ...)
    or a nested MATLAB struct 'invNorm' with those fields."""
    raw = sio.loadmat(path)
    keys = ("mu_in", "sig_in", "mu_out", "sig_out")
    if all(k in raw for k in keys):
        src = raw
        get = lambda k: raw[k]
    elif "invNorm" in raw:
        s = raw["invNorm"]
        get = lambda k: s[k][0, 0]
    else:
        raise KeyError(f"{path}: expected flat keys {keys} or an 'invNorm' struct; "
                       f"found {[k for k in raw if not k.startswith('__')]}")
    return {k: np.asarray(get(k)).flatten().astype(np.float32) for k in keys}


class InverseMagnetModel:
    def __init__(self, onnx_path: str, norm_path: str, mp: dict):
        self.sess = ort.InferenceSession(onnx_path,
                                         providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        norm = _load_norm(norm_path)
        self.mu_in = norm["mu_in"]
        self.sig_in = norm["sig_in"]
        self.mu_out = norm["mu_out"]
        self.sig_out = norm["sig_out"]

        self.cat_tip_center = np.asarray(mp["cat_tip_center"], dtype=np.float32)
        self.z_rel_mag = float(mp["z_rel_mag"])
        self.B_min = float(mp["B_min"])
        self.B_max = float(mp["B_max"])
        self.R_min = float(mp["R_min"])
        self.R_span = float(mp["R_span"])

    def predict_field(self, tip_delta_m):
        """desired tip delta (3,) m -> (B, az, el)."""
        x = np.asarray(tip_delta_m, dtype=np.float32).reshape(1, 3)
        x_norm = (x - self.mu_in) / self.sig_in
        y = np.array(self.sess.run(None, {self.in_name: x_norm})[0]).reshape(-1)
        y = y * self.sig_out + self.mu_out
        return float(y[0]), float(y[1]), float(y[2])

    def magnet_position(self, tip_delta_m) -> np.ndarray:
        """desired tip delta (3,) m -> magnet/TCP position (3,) m in base frame.

        Mirrors mscr_inv_control.py: clamp B, map to orbit radius R about the
        catheter-tip center, place the magnet at azimuth az, offset by z_rel_mag.
        """
        B, az, el = self.predict_field(tip_delta_m)
        Bc = float(np.clip(B, self.B_min, self.B_max))
        R = self.R_min + ((Bc - self.B_min) / (self.B_max - self.B_min + 1e-8)) * self.R_span
        pm = self.cat_tip_center + np.array(
            [R * np.cos(az), R * np.sin(az), self.z_rel_mag], dtype=np.float32)
        return pm.astype(float)
