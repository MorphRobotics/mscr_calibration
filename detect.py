#!/usr/bin/env python3
"""
MSCR Body and Tip Detection + 3D Localisation

Streams colour + depth from the RealSense, segments the black PDMS-NdFeB
catheter, extracts an ordered body curve (base → tip), fits a B-spline, and
projects the tip to 3D using the depth map and calibrated intrinsics.

Controls
--------
  B      — capture background frame (greatly improves segmentation)
  CLICK  — click the catheter BASE in the image to fix the traversal direction
  +/-    — raise / lower the segmentation threshold by 5
  S      — save current result (body curve + tip) to results/detection/
  Q/ESC  — quit
"""

import os, sys, json, time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import yaml
import pyrealsense2 as rs
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize as sk_skeletonize

# ─────────────────────────────────────────────────────────────────────────────
CALIBRATION_YAML = "results/stereo_calibration.yaml"
SAVE_DIR         = "results/detection"

# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Segmentation
    dark_thresh: int   = 60      # pixel intensity below this → MSCR (no background)
    bg_thresh:   int   = 30      # background-subtraction difference threshold
    close_k:     int   = 7       # morphological closing kernel size (fills gaps)
    open_k:      int   = 3       # morphological opening kernel size (removes noise)
    min_area:    int   = 400     # minimum contour area to be considered MSCR (at process_scale)

    # Skeleton / curve
    smooth_s:    float = 0.0     # spline smoothing (0 = interpolate exactly)
    n_curve_pts: int   = 150     # number of resampled curve points

    # Performance — skeletonize cost scales with image area, not catheter size.
    # 0.5 cuts skeletonize from ~22 ms to ~3 ms; results are scaled back to full-res coords.
    process_scale: float = 0.5   # fraction of full resolution used for segmentation

    # Camera
    width:  int = 1280
    height: int = 720
    fps:    int = 30


@dataclass
class DetectionResult:
    body_px:   np.ndarray          # (N, 2)  [u, v] pixel coords base → tip
    body_mm:   np.ndarray          # (N, 3)  [X, Y, Z] mm in camera frame
    tip_px:    tuple               # (u, v)
    tip_mm:    Optional[np.ndarray]  # [X, Y, Z] mm or None if depth invalid
    mask:      np.ndarray          # binary segmentation mask
    timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────

class MSCRDetector:

    def __init__(self, cfg: Config = Config()):
        self.cfg        = cfg
        self.background: Optional[np.ndarray] = None  # grayscale background
        self.base_hint:  Optional[tuple]       = None  # (u, v) base click

        # Load calibration
        self.fx = self.fy = self.cx = self.cy = None
        self.depth_scale = 0.001
        self._load_calibration()

        # Morphological kernels
        self._kc = np.ones((cfg.close_k, cfg.close_k), np.uint8)
        self._ko = np.ones((cfg.open_k,  cfg.open_k),  np.uint8)

    # ── Calibration ──────────────────────────────────────────────────────────

    def _load_calibration(self):
        if not os.path.exists(CALIBRATION_YAML):
            print(f"[WARN] {CALIBRATION_YAML} not found — using firmware intrinsics.")
            return
        with open(CALIBRATION_YAML) as f:
            cal = yaml.safe_load(f)
        K = np.array(cal["left_camera"]["camera_matrix"]["data"]).reshape(3, 3)
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.depth_scale  = cal.get("depth_scale_m_per_unit", 0.001)
        print(f"Calibration loaded: fx={self.fx:.2f} fy={self.fy:.2f} "
              f"depth_scale={self.depth_scale}")

    def set_intrinsics_from_profile(self, profile):
        """Fall back to firmware intrinsics if YAML not loaded."""
        if self.fx is not None:
            return
        intr = (profile.get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
        self.fx, self.fy = intr.fx, intr.fy
        self.cx, self.cy = intr.ppx, intr.ppy
        print(f"Firmware intrinsics: fx={self.fx:.2f} fy={self.fy:.2f}")

    # ── Segmentation ─────────────────────────────────────────────────────────

    def capture_background(self, gray: np.ndarray):
        self.background = gray.copy()
        print("Background captured.")

    def _segment(self, gray: np.ndarray) -> np.ndarray:
        if self.background is not None:
            diff = cv2.absdiff(self.background, gray)
            _, mask = cv2.threshold(diff, self.cfg.bg_thresh,
                                    255, cv2.THRESH_BINARY)
        else:
            _, mask = cv2.threshold(gray, self.cfg.dark_thresh,
                                    255, cv2.THRESH_BINARY_INV)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kc)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._ko)
        return mask

    def _largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component above min_area."""
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        best_area, best_label = 0, 0
        for i in range(1, n_labels):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a > best_area and a >= self.cfg.min_area:
                best_area, best_label = a, i
        if best_label == 0:
            return np.zeros_like(mask)
        return (labels == best_label).astype(np.uint8) * 255

    # ── Skeleton extraction ───────────────────────────────────────────────────

    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0)
        skel   = sk_skeletonize(binary)
        return skel.astype(np.uint8) * 255

    def _order_skeleton(self, skel: np.ndarray) -> Optional[np.ndarray]:
        """
        Return skeleton pixels ordered from base to tip as (N, 2) [row, col].
        Uses DFS from the endpoint nearest to base_hint (or the bottommost point).
        """
        pts  = np.column_stack(np.where(skel > 0))   # (N, 2) row, col
        if len(pts) < 4:
            return None

        pset = {(int(r), int(c)) for r, c in pts}

        def nbrs(p):
            r, c = p
            return [(r+dr, c+dc)
                    for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if (dr, dc) != (0, 0) and (r+dr, c+dc) in pset]

        # Endpoints: exactly one neighbour
        endpoints = [p for p in pset if len(nbrs(p)) == 1]
        if len(endpoints) < 2:
            # Degenerate — just return points sorted by row
            return pts[np.argsort(pts[:, 0])[::-1]]

        # Choose start = endpoint closest to base_hint (or max-row endpoint)
        if self.base_hint is not None:
            bv, bu = self.base_hint[1], self.base_hint[0]   # row, col
            start  = min(endpoints,
                         key=lambda p: (p[0] - bv) ** 2 + (p[1] - bu) ** 2)
        else:
            start = max(endpoints, key=lambda p: p[0])   # lowest in image

        # DFS traversal (chain = no real branching for a single catheter)
        visited, path = set(), []
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            path.append(node)
            for nb in sorted(nbrs(node),
                             key=lambda p: len([x for x in nbrs(p) if x not in visited])):
                if nb not in visited:
                    stack.append(nb)

        return np.array(path)   # (N, 2) row, col

    # ── Curve fitting ─────────────────────────────────────────────────────────

    def _fit_spline(self, ordered_pts: np.ndarray) -> np.ndarray:
        """
        Fit a B-spline to the ordered skeleton and resample to n_curve_pts.
        Returns (N, 2) array of [u, v] pixel coordinates.
        """
        if len(ordered_pts) < 4:
            return ordered_pts[:, ::-1]   # row,col → u,v

        rows = ordered_pts[:, 0].astype(float)
        cols = ordered_pts[:, 1].astype(float)

        # Deduplicate consecutive identical points
        keep = np.concatenate([[True], np.any(np.diff(np.column_stack([rows, cols]),
                                                       axis=0) != 0, axis=1)])
        rows, cols = rows[keep], cols[keep]
        if len(rows) < 4:
            return np.column_stack([cols, rows])

        try:
            tck, _ = splprep([cols, rows], s=self.cfg.smooth_s, k=3,
                             per=False, quiet=True)
            u_new   = np.linspace(0, 1, self.cfg.n_curve_pts)
            cu, cv  = splev(u_new, tck)
            return np.column_stack([cu, cv])   # (N, 2) [u, v]
        except Exception:
            return np.column_stack([cols, rows])

    # ── 3D projection ─────────────────────────────────────────────────────────

    def _px_to_3d(self, u: float, v: float, depth_img: np.ndarray
                  ) -> Optional[np.ndarray]:
        """
        Project pixel (u, v) to 3D using depth map.
        Returns [X, Y, Z] in mm in camera frame, or None if depth is invalid.
        Samples a 3x3 patch and takes the median to reduce noise.
        """
        ui, vi = int(round(u)), int(round(v))
        h, w   = depth_img.shape
        if not (2 <= vi < h - 2 and 2 <= ui < w - 2):
            return None
        patch = depth_img[vi - 2:vi + 3, ui - 2:ui + 3].astype(float)
        valid = patch[patch > 0]
        if len(valid) < 3:
            return None
        d_m = float(np.median(valid)) * self.depth_scale
        X   = (u - self.cx) * d_m / self.fx
        Y   = (v - self.cy) * d_m / self.fy
        Z   = d_m
        return np.array([X * 1000, Y * 1000, Z * 1000])   # mm

    def _curve_to_3d(self, curve_uv: np.ndarray,
                     depth_img: np.ndarray) -> np.ndarray:
        """Project each curve point to 3D; interpolate over invalid depths."""
        pts_3d = []
        for u, v in curve_uv:
            p3 = self._px_to_3d(u, v, depth_img)
            pts_3d.append(p3 if p3 is not None else None)

        # Forward-fill then back-fill missing points
        out = []
        last_valid = None
        for p in pts_3d:
            if p is not None:
                last_valid = p
                out.append(p.copy())
            else:
                out.append(last_valid.copy() if last_valid is not None
                           else np.array([np.nan, np.nan, np.nan]))
        # Back-fill leading Nones
        first_valid = next((p for p in out if not np.isnan(p[0])), None)
        if first_valid is not None:
            for i, p in enumerate(out):
                if np.isnan(p[0]):
                    out[i] = first_valid.copy()
                else:
                    break

        return np.array(out)   # (N, 3)

    # ── Main detection ────────────────────────────────────────────────────────

    def detect(self, color_img: np.ndarray,
               depth_img: np.ndarray) -> Optional[DetectionResult]:
        s  = self.cfg.process_scale
        fh, fw = color_img.shape[:2]
        ph, pw = int(fh * s), int(fw * s)

        gray_full = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        # Downsample for all heavy processing
        gray = cv2.resize(gray_full, (pw, ph), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Background must also be at process resolution
        bg_small = (cv2.resize(self.background, (pw, ph), interpolation=cv2.INTER_AREA)
                    if self.background is not None else None)
        # Temporarily swap so _segment works at small scale
        bg_orig, self.background = self.background, bg_small
        mask       = self._segment(gray)
        self.background = bg_orig

        mask_clean = self._largest_component(mask)
        if mask_clean.max() == 0:
            return None

        skel    = self._skeletonize(mask_clean)
        ordered = self._order_skeleton(skel)
        if ordered is None or len(ordered) < 4:
            return None

        # Scale skeleton coordinates back to full-resolution pixel space
        ordered_full = ordered.astype(float) / s   # row, col at full res

        curve_uv  = self._fit_spline(ordered_full)            # (N,2) [u,v] full res
        curve_3d  = self._curve_to_3d(curve_uv, depth_img)   # (N,3) mm

        tip_uv = tuple(curve_uv[-1].astype(int))
        tip_3d = self._px_to_3d(tip_uv[0], tip_uv[1], depth_img)

        # Return mask upscaled for display
        mask_display = cv2.resize(mask_clean, (fw, fh), interpolation=cv2.INTER_NEAREST)

        return DetectionResult(
            body_px=curve_uv,
            body_mm=curve_3d,
            tip_px=tip_uv,
            tip_mm=tip_3d,
            mask=mask_display,
        )

    # ── Visualisation ─────────────────────────────────────────────────────────

    def visualize(self, frame: np.ndarray,
                  result: Optional[DetectionResult]) -> np.ndarray:
        out = frame.copy()

        if result is None:
            cv2.putText(out, "No detection", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)
            return out

        # Body curve — colour gradient base (blue) → tip (red)
        pts = result.body_px.astype(np.int32)
        n   = len(pts)
        for i in range(n - 1):
            t   = i / max(n - 2, 1)
            col = (int(220 * (1 - t)), 50, int(220 * t))   # BGR: blue→red
            cv2.line(out, tuple(pts[i]), tuple(pts[i + 1]), col, 2, cv2.LINE_AA)

        # Tip marker
        tu, tv = result.tip_px
        cv2.circle(out, (tu, tv), 8,  (0, 0, 255), -1)
        cv2.circle(out, (tu, tv), 10, (255, 255, 255), 1)

        # Base marker
        bu, bv = tuple(pts[0])
        cv2.circle(out, (bu, bv), 6, (255, 180, 0), -1)

        # 3D tip readout
        if result.tip_mm is not None:
            x, y, z = result.tip_mm
            label = f"Tip: X={x:+.1f} Y={y:+.1f} Z={z:.1f} mm"
        else:
            label = "Tip: depth invalid"
        cv2.rectangle(out, (10, 8), (10 + len(label) * 10, 35), (0, 0, 0), -1)
        cv2.putText(out, label, (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

        # Controls hint
        hints = "B=background  CLICK=set base  +/-=threshold  S=save  Q=quit"
        cv2.putText(out, hints, (14, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return out


# ─────────────────────────────────────────────────────────────────────────────

def save_result(result: DetectionResult, idx: int):
    os.makedirs(SAVE_DIR, exist_ok=True)
    stem = os.path.join(SAVE_DIR, f"detection_{idx:05d}")
    np.save(f"{stem}_body_px.npy",  result.body_px)
    np.save(f"{stem}_body_mm.npy",  result.body_mm)
    tip_data = {"tip_px": list(map(int, result.tip_px)),
                "tip_mm": result.tip_mm.tolist() if result.tip_mm is not None else None,
                "timestamp": result.timestamp}
    with open(f"{stem}_tip.json", "w") as f:
        json.dump(tip_data, f, indent=2)
    print(f"Saved → {stem}_*.{{npy,json}}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg      = Config()
    detector = MSCRDetector(cfg)
    save_idx = 0

    # ── RealSense setup ──────────────────────────────────────────────────────
    pipeline = rs.pipeline()
    rs_cfg   = rs.config()
    rs_cfg.enable_stream(rs.stream.color, cfg.width, cfg.height,
                         rs.format.bgr8, cfg.fps)
    rs_cfg.enable_stream(rs.stream.depth, cfg.width, cfg.height,
                         rs.format.z16,  cfg.fps)
    try:
        profile = pipeline.start(rs_cfg)
    except Exception as e:
        print(f"[ERROR] RealSense: {e}")
        sys.exit(1)

    detector.set_intrinsics_from_profile(profile)

    # Update depth scale from device
    depth_sensor = profile.get_device().first_depth_sensor()
    detector.depth_scale = depth_sensor.get_depth_scale()

    align = rs.align(rs.stream.color)

    # ── Trackbar window for threshold ────────────────────────────────────────
    cv2.namedWindow("MSCR Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MSCR Detection", 1280, 760)

    bg_mode   = False   # whether background has been captured

    def on_thresh(val):
        if bg_mode:
            cfg.bg_thresh   = val
        else:
            cfg.dark_thresh = val

    cv2.createTrackbar("Threshold", "MSCR Detection",
                       cfg.dark_thresh, 120, on_thresh)

    # ── Mouse callback to set base hint ──────────────────────────────────────
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            detector.base_hint = (x, y)
            print(f"Base hint set to (u={x}, v={y})")

    cv2.setMouseCallback("MSCR Detection", on_mouse)

    print("\nMSCR detector running.")
    print("  Press B to capture the current frame as background (recommended).")
    print("  Click the base of the catheter to fix traversal direction.\n")

    result = None
    try:
        while True:
            frames   = pipeline.wait_for_frames()
            aligned  = align.process(frames)
            cf       = aligned.get_color_frame()
            df       = aligned.get_depth_frame()
            if not cf or not df:
                continue

            color = np.asanyarray(cf.get_data())
            depth = np.asanyarray(df.get_data())   # uint16, units = depth_scale metres

            t_detect  = time.perf_counter()
            result    = detector.detect(color, depth)
            det_ms    = (time.perf_counter() - t_detect) * 1000
            display   = detector.visualize(color, result)
            cv2.putText(display, f"{det_ms:.0f} ms/frame  ({1000/det_ms:.0f} fps)",
                        (display.shape[1] - 200, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)

            # Mask inset (top-right corner)
            if result is not None:
                mask_bgr = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
                h, w     = display.shape[:2]
                th, tw   = h // 5, w // 5
                inset    = cv2.resize(mask_bgr, (tw, th))
                display[10:10 + th, w - tw - 10:w - 10] = inset

            if result is not None and result.tip_mm is not None:
                x, y, z = result.tip_mm
                print(f"\rTip  X={x:+7.2f}  Y={y:+7.2f}  Z={z:7.2f} mm"
                      f"  |  body pts={len(result.body_px)}", end="", flush=True)

            cv2.imshow("MSCR Detection", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("b"):
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                detector.capture_background(gray)
                bg_mode = True
                cv2.setTrackbarPos("Threshold", "MSCR Detection", cfg.bg_thresh)
                print("\nBackground captured — trackbar now controls bg_thresh.")
            elif key == ord("+") or key == ord("="):
                if bg_mode:
                    cfg.bg_thresh   = min(cfg.bg_thresh   + 5, 120)
                    cv2.setTrackbarPos("Threshold", "MSCR Detection", cfg.bg_thresh)
                else:
                    cfg.dark_thresh = min(cfg.dark_thresh + 5, 200)
                    cv2.setTrackbarPos("Threshold", "MSCR Detection", cfg.dark_thresh)
            elif key == ord("-"):
                if bg_mode:
                    cfg.bg_thresh   = max(cfg.bg_thresh   - 5, 5)
                    cv2.setTrackbarPos("Threshold", "MSCR Detection", cfg.bg_thresh)
                else:
                    cfg.dark_thresh = max(cfg.dark_thresh - 5, 5)
                    cv2.setTrackbarPos("Threshold", "MSCR Detection", cfg.dark_thresh)
            elif key == ord("s") and result is not None:
                save_result(result, save_idx)
                save_idx += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\nDone.")


if __name__ == "__main__":
    main()
