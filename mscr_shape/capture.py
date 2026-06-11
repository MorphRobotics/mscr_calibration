"""Phase 1 — synchronized left/right IR capture for the D435.

Streams the two monochrome IR imagers (global shutter, hardware-synced) at
1280x720 with the IR dot-pattern emitter DISABLED, since we image a bare rod
and the projected dots would only add noise. The RGB camera is never used.

Usage:
    python capture.py --session demo            # live capture from a D435
    python capture.py --session demo --fake     # synthetic, no hardware

Live preview shows both rectified-as-captured views side by side with a REC
indicator. Keys:  r = toggle record,  q = quit.

Output layout (continuous recording):
    data/raw/<session>/left/<idx>.png
    data/raw/<session>/right/<idx>.png
    data/raw/<session>/manifest.json

Frames whose left-image Laplacian variance is below `blur_var_threshold`
(config) are treated as motion-blurred and skipped before writing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from cfg import load_config


def laplacian_var(gray: np.ndarray) -> float:
    """Cheap sharpness proxy; low value => motion blur."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# --------------------------------------------------------------------------- #
# Frame sources
# --------------------------------------------------------------------------- #
class FrameSource:
    """Yields (left_gray, right_gray, host_timestamp_s) tuples."""

    serial: str = "unknown"
    settings: dict = {}

    def frames(self) -> Iterator[Tuple[np.ndarray, np.ndarray, float]]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class RealSenseSource(FrameSource):
    def __init__(self, width: int, height: int, fps: int, emitter_enabled: int):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        # Stream 1 = left IR, stream 2 = right IR (monochrome, y8).
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
        profile = self.pipeline.start(config)

        device = profile.get_device()
        self.serial = device.get_info(rs.camera_info.serial_number)
        depth_sensor = device.first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, float(emitter_enabled))
        self.settings = {
            "width": width, "height": height, "fps": fps,
            "emitter_enabled": emitter_enabled,
        }

    def frames(self) -> Iterator[Tuple[np.ndarray, np.ndarray, float]]:
        while True:
            fs = self.pipeline.wait_for_frames()
            left = fs.get_infrared_frame(1)
            right = fs.get_infrared_frame(2)
            if not left or not right:
                continue
            l = np.asanyarray(left.get_data())
            r = np.asanyarray(right.get_data())
            ts = left.get_timestamp() / 1000.0  # ms -> s
            yield l, r, ts

    def close(self) -> None:
        self.pipeline.stop()


class FakeSource(FrameSource):
    """Synthesizes a moving dark curve on a light background in both views,
    with a known horizontal disparity offset between left and right."""

    def __init__(self, width: int, height: int, fps: int, n_frames: int = 60):
        self.w, self.h, self.fps, self.n = width, height, fps, n_frames
        self.serial = "FAKE0000"
        self.settings = {"width": width, "height": height, "fps": fps,
                         "emitter_enabled": 0, "fake": True}

    def _render(self, phase: float, disparity: int) -> np.ndarray:
        img = np.full((self.h, self.w), 220, dtype=np.uint8)
        img += (np.random.randn(self.h, self.w) * 4).astype(np.int16).clip(-20, 20).astype(np.uint8)
        # a dark quadratic "rod" curving with phase, base near image bottom
        ys = np.linspace(self.h * 0.85, self.h * 0.2, 220)
        bend = 90.0 * np.sin(phase)
        t = np.linspace(0, 1, ys.size)
        xs = self.w * 0.5 + bend * t * t - disparity
        for x, y in zip(xs, ys):
            cv2.circle(img, (int(round(x)), int(round(y))), 3, 30, -1)
        return img

    def frames(self) -> Iterator[Tuple[np.ndarray, np.ndarray, float]]:
        t0 = time.time()
        for i in range(self.n):
            phase = 2 * np.pi * i / self.n
            left = self._render(phase, disparity=0)
            right = self._render(phase, disparity=24)  # rod nearer => +disparity
            yield left, right, t0 + i / self.fps
            time.sleep(1.0 / self.fps * 0.1)  # keep the test fast

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Capture loop
# --------------------------------------------------------------------------- #
def run_capture(
    source: FrameSource,
    session: str,
    data_root: Path,
    blur_thresh: float,
    preview_scale: float,
    headless: bool = False,
    auto_record: bool = False,
    max_frames: Optional[int] = None,
) -> dict:
    out_dir = data_root / "raw" / session
    left_dir, right_dir = out_dir / "left", out_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    recording = auto_record
    records: list[dict] = []
    idx = 0
    seen = 0

    show = not headless
    if show:
        try:
            cv2.namedWindow("mscr_shape capture", cv2.WINDOW_NORMAL)
        except cv2.error:
            show = False  # no display available

    for left, right, ts in source.frames():
        seen += 1
        fv = laplacian_var(left)
        sharp = fv >= blur_thresh

        if recording and sharp:
            name = f"{idx:06d}.png"
            cv2.imwrite(str(left_dir / name), left)
            cv2.imwrite(str(right_dir / name), right)
            records.append({"idx": idx, "file": name, "timestamp": ts,
                            "laplacian_var": fv})
            idx += 1

        if show:
            _draw_preview(left, right, recording, sharp, fv, preview_scale)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                recording = not recording
            elif key == ord("q"):
                break

        if max_frames is not None and seen >= max_frames:
            break

    if show:
        cv2.destroyAllWindows()
    source.close()

    manifest = {
        "session": session,
        "device_serial": source.serial,
        "settings": source.settings,
        "blur_var_threshold": blur_thresh,
        "n_saved": idx,
        "n_seen": seen,
        "frames": records,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _draw_preview(left, right, recording, sharp, fv, scale):
    lr = np.hstack([left, right])
    vis = cv2.cvtColor(lr, cv2.COLOR_GRAY2BGR)
    if scale != 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale)
    if recording:
        cv2.circle(vis, (20, 20), 10, (0, 0, 255), -1)
        cv2.putText(vis, "REC", (38, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
    txt = f"lapvar={fv:.0f} {'OK' if sharp else 'BLUR'}"
    cv2.putText(vis, txt, (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0) if sharp else (0, 165, 255), 2)
    cv2.imshow("mscr_shape capture", vis)


def main() -> None:
    cfg = load_config()
    cap = cfg["capture"]
    ap = argparse.ArgumentParser(description="D435 left/right IR capture")
    ap.add_argument("--session", required=True)
    ap.add_argument("--fake", action="store_true", help="synthesize frames, no hardware")
    ap.add_argument("--headless", action="store_true", help="no preview window")
    ap.add_argument("--auto-record", action="store_true",
                    help="start recording immediately (useful with --fake/--headless)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--data-root", default=cfg["paths"]["data_root"])
    args = ap.parse_args()

    if args.fake:
        source: FrameSource = FakeSource(cap["width"], cap["height"], cap["fps"])
    else:
        source = RealSenseSource(cap["width"], cap["height"], cap["fps"],
                                 cap["emitter_enabled"])

    manifest = run_capture(
        source, args.session, Path(args.data_root),
        blur_thresh=cap["blur_var_threshold"],
        preview_scale=cap["preview_scale"],
        headless=args.headless,
        auto_record=args.auto_record or args.fake,
        max_frames=args.max_frames,
    )
    print(f"session '{manifest['session']}': saved {manifest['n_saved']} / "
          f"{manifest['n_seen']} seen frames to {args.data_root}/raw/{manifest['session']}")


if __name__ == "__main__":
    main()
