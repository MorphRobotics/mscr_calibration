"""Phase 5 — inference, temporal smoothing, and ONNX export.

Runs a trained MoSSNet on a live left-IR stream or a saved session, draws the
predicted centerline reprojected onto the left-IR rectified image, applies a
temporal layer (EMA smoothing + tip-jump rejection), exports to ONNX, and
verifies the ONNX output matches PyTorch.

    python infer.py --session <name>          # run on saved frames
    python infer.py --live                    # run on the D435 left-IR stream
    python infer.py --export-onnx             # export + verify (no camera needed)

All 3D output is in the left-IR rectified camera frame (mm).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from calib import StereoCalib, load_calib, resolve_calib
from cfg import load_config
from labeler import project
from model import MoSSNet


# --------------------------------------------------------------------------- #
# Temporal layer
# --------------------------------------------------------------------------- #
class TemporalFilter:
    """EMA smoothing of the predicted curve + tip-jump rejection."""

    def __init__(self, alpha: float, max_jump_mm: float):
        self.alpha = alpha
        self.max_jump_mm = max_jump_mm
        self.smoothed: Optional[np.ndarray] = None

    def update(self, pts: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return (smoothed_curve, accepted). A rejected frame keeps the old
        estimate (no update)."""
        if self.smoothed is None:
            self.smoothed = pts.copy()
            return self.smoothed, True
        tip_jump = float(np.linalg.norm(pts[-1] - self.smoothed[-1]))
        if tip_jump > self.max_jump_mm:
            return self.smoothed, False
        self.smoothed = self.alpha * pts + (1 - self.alpha) * self.smoothed
        return self.smoothed, True


# --------------------------------------------------------------------------- #
# Model loading / prediction
# --------------------------------------------------------------------------- #
def load_model(ckpt_path: Path, device: str) -> tuple[MoSSNet, dict]:
    blob = torch.load(ckpt_path, map_location=device)
    cfg = blob["config"]
    net = MoSSNet(cfg["model"]["n_points"], pretrained=False).to(device)
    net.load_state_dict(blob["model"])
    net.eval()
    return net, cfg


def preprocess(img_gray: np.ndarray, cfg: dict) -> torch.Tensor:
    h, w = cfg["dataset"]["image_size"]
    img = cv2.resize(img_gray.astype(np.float32), (w, h)) / 255.0
    return torch.from_numpy(img[None, None]).float()


@torch.no_grad()
def predict(net: MoSSNet, img_t: torch.Tensor, device: str) -> tuple[np.ndarray, float]:
    pts, L = net(img_t.to(device))
    return pts[0].cpu().numpy(), float(L[0].item())


def draw_overlay(img_gray: np.ndarray, calib: StereoCalib, pts3d: np.ndarray,
                 accepted: bool, L_mm: float, fps: float) -> np.ndarray:
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    proj = project(calib.P1, pts3d)
    proj = proj.round().astype(int)
    for k in range(len(proj) - 1):
        cv2.line(vis, tuple(proj[k]), tuple(proj[k + 1]), (0, 220, 0), 2)
    cv2.circle(vis, tuple(proj[-1]), 5, (0, 0, 255), -1)  # tip
    banner = f"L={L_mm:.0f}mm  {fps:.1f} FPS  {'OK' if accepted else 'JUMP-REJECT'}"
    cv2.putText(vis, banner, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 220, 0) if accepted else (0, 0, 255), 2)
    return vis


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def run_session(net, cfg, calib, session: str, device: str, show: bool) -> None:
    data_root = Path(cfg["paths"]["data_root"])
    left_dir = data_root / "raw" / session / "left"
    files = sorted(left_dir.glob("*.png"))
    tf = TemporalFilter(cfg["infer"]["ema_alpha"], cfg["infer"]["max_jump_mm"])
    times = []
    for f in files:
        raw = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        rect = calib.rectify_left(raw)
        t0 = time.time()
        pts, L = predict(net, preprocess(rect, cfg), device)
        smoothed, accepted = tf.update(pts)
        times.append(time.time() - t0)
        fps = 1.0 / np.mean(times[-30:])
        if show:
            vis = draw_overlay(rect, calib, smoothed, accepted, L, fps)
            cv2.imshow("mscr_shape infer", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    if show:
        cv2.destroyAllWindows()
    if times:
        print(f"processed {len(times)} frames, mean {1000*np.mean(times):.1f} ms/frame "
              f"=> {1.0/np.mean(times):.1f} FPS (network only)")


def run_live(net, cfg, calib, device: str) -> None:
    from capture import RealSenseSource
    cap = cfg["capture"]
    src = RealSenseSource(cap["width"], cap["height"], cap["fps"], cap["emitter_enabled"])
    tf = TemporalFilter(cfg["infer"]["ema_alpha"], cfg["infer"]["max_jump_mm"])
    times = []
    try:
        for left, _right, _ts in src.frames():
            rect = calib.rectify_left(left)
            t0 = time.time()
            pts, L = predict(net, preprocess(rect, cfg), device)
            smoothed, accepted = tf.update(pts)
            times.append(time.time() - t0)
            fps = 1.0 / np.mean(times[-30:])
            vis = draw_overlay(rect, calib, smoothed, accepted, L, fps)
            cv2.imshow("mscr_shape infer (live)", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        src.close()
        cv2.destroyAllWindows()
    if times:
        print(f"full live loop: {1.0/np.mean(times):.1f} FPS")


# --------------------------------------------------------------------------- #
# ONNX export + verify
# --------------------------------------------------------------------------- #
def export_onnx(net: MoSSNet, cfg: dict, onnx_path: Path, tol: float, device: str) -> None:
    import onnxruntime as ort

    h, w = cfg["dataset"]["image_size"]
    dummy = torch.randn(1, 1, h, w, device=device)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    net.eval()
    torch.onnx.export(
        net, dummy, str(onnx_path),
        input_names=["image"], output_names=["points", "length"],
        dynamic_axes={"image": {0: "batch"}, "points": {0: "batch"}, "length": {0: "batch"}},
        opset_version=17,
    )
    with torch.no_grad():
        pt_pts, pt_L = net(dummy)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ox_pts, ox_L = sess.run(None, {"image": dummy.cpu().numpy()})
    d_pts = float(np.abs(pt_pts.cpu().numpy() - ox_pts).max())
    d_L = float(np.abs(pt_L.cpu().numpy() - ox_L).max())
    print(f"ONNX exported to {onnx_path}")
    print(f"max abs diff vs PyTorch: points={d_pts:.2e}  length={d_L:.2e} (tol={tol})")
    assert d_pts < tol and d_L < tol, "ONNX output diverges from PyTorch beyond tol"
    print("PASS: ONNX matches PyTorch within tolerance")


def main() -> None:
    ap = argparse.ArgumentParser(description="MoSSNet inference / export")
    ap.add_argument("--session", help="run on data/raw/<session>")
    ap.add_argument("--live", action="store_true", help="run on live D435 left-IR")
    ap.add_argument("--export-onnx", action="store_true")
    ap.add_argument("--no-show", action="store_true", help="disable preview window")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(__file__).parent / cfg["train"]["ckpt"]

    if args.export_onnx and not ckpt.exists():
        # allow export verification with a fresh (untrained) net
        net = MoSSNet(cfg["model"]["n_points"], pretrained=False).to(device).eval()
    else:
        net, ckpt_cfg = load_model(ckpt, device)
        cfg = {**ckpt_cfg, "infer": cfg["infer"], "paths": cfg["paths"]}

    if args.export_onnx:
        export_onnx(net, cfg, Path(__file__).parent / cfg["infer"]["onnx_path"],
                    cfg["infer"]["onnx_tol"], device)
    elif args.live:
        run_live(net, cfg, resolve_calib(cfg), device)
    elif args.session:
        run_session(net, cfg, resolve_calib(cfg), args.session, device, show=not args.no_show)
    else:
        ap.error("specify one of --session, --live, or --export-onnx")


if __name__ == "__main__":
    main()
