#!/usr/bin/env python3
"""
mscr_tracker.py — Real-time 3D segmentation, centerline extraction, and tip
tracking for a black Magnetic Soft Continuum Robot (MSCR) using an Intel
RealSense D435i.

Pipeline per frame:
  1. Color/depth alignment + RealSense post-processing filters
  2. Dark-pixel mask ∩ depth gate  →  rod binary mask
  3. Morphological cleanup + largest elongated connected component
  4. Skeletonisation  →  ordered 1-px centerline
  5. rs2_deproject_pixel_to_point  →  ordered 3-D point cloud (m)
  6. Missing-depth interpolation + 3-D B-spline smoothing
  7. Arc length (mm) and tip (X, Y, Z in mm) extraction
  8. Overlay visualisation + on-screen HUD

Usage (live stream):
    python mscr_tracker.py [--debug] [--threshold 55] [--depth-max 0.6]

Usage (library):
    tracker = MSCRTracker(TrackerParams())
    tracker.set_intrinsics(intr)       # rs.intrinsics object
    tracker.depth_scale = 0.001
    result = tracker.process_frame(color_bgr, depth_uint16)

All 2-D coordinates follow OpenCV convention: (col, row) = (u, v).
All 3-D coordinates are in the RealSense camera frame (right-hand, Z forward).
"""

from __future__ import annotations

import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize as _skeletonize

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackerParams:
    # ── Stream ────────────────────────────────────────────────────────────────
    width:              int   = 848
    height:             int   = 480
    fps:                int   = 30
    laser_power:        int   = 0       # 0 = emitter off (passive depth for dark rod)

    # ── Depth gating ──────────────────────────────────────────────────────────
    depth_min_m:        float = 0.05
    depth_max_m:        float = 0.80
    # Pixels with zero depth (IR absorbed by the dark rod) are always KEPT.
    depth_dilation_px:  int   = 3       # erode exclusion mask to avoid clipping rod
                                        # edges (3, not 6: a hand near the rod
                                        # perturbs depth — less erosion keeps the
                                        # thin valid rod data at the boundary)

    # ── Segmentation ──────────────────────────────────────────────────────────
    dark_threshold:     int   = 85      # THRESH_BINARY_INV; the MSCR rod sits at
                                        # ~40-90 grey on the white fixture, so 85
                                        # captures it; 55 missed the rod entirely
    morph_open_iter:    int   = 1
    morph_close_iter:   int   = 2
    min_area_px:        int   = 150
    max_area_frac:      float = 0.05    # ignore blobs larger than 5 % of frame
    min_eccentricity:   float = 0.80    # fitted ellipse eccentricity; rods score ~0.90+

    # ── Hand / slender-component isolation ────────────────────────────────────
    # A morphological opening with a kernel wider than the rod removes the
    # thick hand/arm/fingers while the slender rod survives; subtracting that
    # "thick" set from the mask isolates the rod even when a hand touches it.
    slender_isolation:  bool  = True
    rod_max_width_px:   int   = 12      # max expected rod width (px); anything
                                        # thicker (fingers ≈ 15-30 px) is treated
                                        # as hand and removed before CC labelling
    hand_thick_frac:    float = 0.30    # if >this fraction of the foreground is
                                        # "thick", a hand is deemed present →
                                        # ROI guards relax & depth search widens
    hand_relax_factor:  float = 2.5     # multiplier on ROI growth/shift ceilings
                                        # while a hand is present in the frame
    depth_search_r_hand: int  = 14      # widened depth-proxy search radius when a
                                        # hand perturbs depth near the rod

    # ── Skeleton / spline ─────────────────────────────────────────────────────
    min_skel_pts:       int   = 15
    max_skel_pts:       int   = 1000
    n_resample:         int   = 200     # resampled points along the 3-D spline
    spline_smooth:      float = 2.0     # 3-D splprep smoothing factor multiplier
    # Minimum fraction of skeleton points that must have valid depth (> 0).
    # Keep low: spatial_filter fills most zeros with background values so the
    # raw zero-depth mask (pre-filter) is used for inclusion instead.
    min_valid_depth_frac: float = 0.02

    # ── Entry side (base heuristic when no external prior is given) ───────────
    entry:              str   = "top"     # "top" | "bottom" | "left" | "right"
                                          # rod base attaches at the TOP of the
                                          # fixture; the free tip is the lower end

    # ── Temporal ROI ──────────────────────────────────────────────────────────
    roi_pad_px:         int   = 60
    max_lost_frames:    int   = 8
    # ROI explosion guard: a candidate bbox is rejected as noise when its area
    # ratio vs the previous bbox exceeds roi_max_growth (or its reciprocal) OR
    # its centre shifts more than roi_max_shift_px in a single frame.
    roi_max_growth:     float = 3.0
    roi_max_shift_px:   float = 140.0
    # When tracking is lost, the search window grows by this many px per lost
    # frame (incremental widening) instead of instantly jumping to full frame.
    roi_expand_px:      int   = 30

    # ── Base/tip disambiguation ───────────────────────────────────────────────
    # The base is the FIXED mount (static); the tip is the moving free end.
    # We anchor on the base — a slow EMA that stays put — and classify the
    # endpoint nearest the anchor as the base.  This prevents tip/base flips
    # when the rod bends sharply (the moving tip is an unreliable anchor).
    base_ema_alpha:     float = 0.10    # slow EMA for the static base anchor
    base_lock_dist_px:  float = 120.0   # if the nearest endpoint to the anchor
                                        # is farther than this, the detection is
                                        # suspect → reject (likely a bad frame)

    # ── Temporal smoothing ────────────────────────────────────────────────────
    ema_alpha:          float = 0.25    # exponential moving average for tip/arc
    max_jump_mm:        float = 30.0   # reject updates where tip jumps > this (mm)

    # ── Depth neighbourhood search ────────────────────────────────────────────
    # The rod absorbs IR → centerline pixels return depth=0.  We search this
    # many pixels outward from each skeleton pixel to find the nearest non-zero
    # depth value (background just beside the rod ≈ rod depth for a thin rod).
    depth_search_r:     int   = 7

    # ── Semantic segmentation (learned U-Net) ─────────────────────────────────
    # When seg_model_path points to a trained checkpoint, the U-Net replaces
    # the classical dark-threshold step.  Leave None to use the classical path.
    seg_model_path:     Optional[str]  = None
    seg_threshold:      float = 0.5      # sigmoid threshold for the rod mask
    seg_use_depth_gate: bool  = False    # AND the NN mask with the depth gate
    seg_half:           bool  = True     # FP16 inference on CUDA

    # ── Debug ─────────────────────────────────────────────────────────────────
    debug:              bool  = False


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackResult:
    """
    All 2-D positions are (col, row) float; 3-D positions are (X, Y, Z) in mm.
    """
    valid:          bool

    # 2-D centerline: (N, 2) float32 array of (col, row) full-image coords
    centerline_px:  np.ndarray

    # 3-D centerline: (N, 3) float64 array of (X, Y, Z) in mm, camera frame
    centerline_3d:  np.ndarray

    # Arc length of the smoothed 3-D curve (mm)
    arc_length_mm:  float

    # Tip and base in 2-D (col, row)
    tip_px:         Tuple[float, float]
    base_px:        Tuple[float, float]

    # Tip in 3-D (mm)
    tip_xyz_mm:     Tuple[float, float, float]

    # Intermediate (50 %) point along the rod in 3-D (mm)
    mid_xyz_mm:     Tuple[float, float, float]

    debug_frame:    Optional[np.ndarray] = field(default=None, repr=False)

    @staticmethod
    def invalid() -> "TrackResult":
        empty2 = np.zeros((0, 2), dtype=np.float32)
        empty3 = np.zeros((0, 3), dtype=np.float64)
        return TrackResult(
            valid          = False,
            centerline_px  = empty2,
            centerline_3d  = empty3,
            arc_length_mm  = 0.0,
            tip_px         = (0.0, 0.0),
            base_px        = (0.0, 0.0),
            tip_xyz_mm     = (0.0, 0.0, 0.0),
            mid_xyz_mm     = (0.0, 0.0, 0.0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────

class MSCRTracker:
    """
    Real-time 3-D tracker for a black MSCR rod against a static background.

    Requires camera intrinsics to be set before calling process_frame.
    When using run(), intrinsics are extracted automatically from the stream.
    """

    def __init__(self, params: TrackerParams = TrackerParams()):
        self.p           = params
        self.depth_scale = 0.001          # metres per raw depth unit (D435 default)
        self._intr: Optional[rs.intrinsics] = None

        # Structuring elements (allocated lazily)
        self._se3      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._se_depth: Optional[np.ndarray] = None

        # RealSense post-processing filters (applied inside run())
        self._spatial  = rs.spatial_filter()
        self._temporal = rs.temporal_filter()
        self._spatial.set_option(rs.option.filter_magnitude,   2)
        self._spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self._spatial.set_option(rs.option.filter_smooth_delta, 20)
        self._temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self._temporal.set_option(rs.option.filter_smooth_delta, 20)

        # Temporal tracking state
        self._last_bbox:     Optional[Tuple[int,int,int,int]] = None
        self._lost_frames:   int   = 0
        self._smooth_tip_3d: Optional[np.ndarray] = None  # (3,) mm
        self._smooth_arc:    float = 0.0
        # EMA-smoothed gate image for stable right-panel visualisation
        self._gate_ema:      Optional[np.ndarray] = None   # float32, full frame
        # Smoothed tip in full-image (col, row) pixels — kept for reference/HUD
        self._smooth_tip_px: Optional[np.ndarray] = None   # (2,) float64
        # Slow-EMA anchor of the STATIC base (fixed mount) in full-image px.
        # Primary cue for base/tip disambiguation — see _assign_base_tip.
        self._anchor_base_px: Optional[np.ndarray] = None  # (2,) float64
        # Last accepted centerline (full-image px) for soft-fail fallback.
        self._last_good_curve_px: Optional[np.ndarray] = None
        # Set per-frame when a hand/arm is detected (large thick blob present).
        # Relaxes the ROI guards and widens the depth-proxy search this frame.
        self._hand_present:  bool = False

        # Optional learned segmenter (U-Net). None → classical dark threshold.
        self._segmenter = None
        if params.seg_model_path:
            try:
                import os, sys
                seg_dir = os.path.join(os.path.dirname(
                    os.path.abspath(__file__)), "seg")
                if seg_dir not in sys.path:
                    sys.path.insert(0, seg_dir)
                from infer import RodSegmenter
                self._segmenter = RodSegmenter(
                    params.seg_model_path,
                    thr=params.seg_threshold,
                    half=params.seg_half)
            except Exception as e:
                print(f"[WARN] Could not load segmentation model "
                      f"'{params.seg_model_path}': {e}\n"
                      f"        Falling back to classical dark-threshold.")
                self._segmenter = None

    # ── Intrinsics ────────────────────────────────────────────────────────────

    def set_intrinsics(self, intr: rs.intrinsics) -> None:
        """Provide camera intrinsics for 3-D deprojection."""
        self._intr = intr

    def _deproject(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """Convert a (u, v, depth_m) tuple to (X, Y, Z) metres using rs2 intrinsics."""
        if self._intr is None:
            raise RuntimeError("Camera intrinsics not set. Call set_intrinsics() first.")
        return np.array(
            rs.rs2_deproject_pixel_to_point(self._intr, [u, v], depth_m),
            dtype=np.float64)

    # ── Public: process one frame pair ───────────────────────────────────────

    def process_frame(self,
                      color:     np.ndarray,
                      depth:     np.ndarray,
                      depth_raw: Optional[np.ndarray] = None) -> TrackResult:
        """
        Process one aligned (color, depth) frame pair.

        Args:
            color     : (H, W, 3) uint8  BGR
            depth     : (H, W)    uint16 post-processed depth (spatial+temporal filtered)
            depth_raw : (H, W)    uint16 raw depth BEFORE post-processing filters.
                        If provided, pixels that were zero in the raw frame (the rod
                        absorbs IR → no return) are forced into the inclusion mask
                        regardless of what the spatial filter interpolated into them.
                        If omitted, falls back to using filtered depth only.

        Returns:
            TrackResult with arc_length_mm and tip_xyz_mm populated when valid.
        """
        H, W = color.shape[:2]

        # ── Step 1a: determine search region (temporal ROI) ───────────────────
        if (self._last_bbox is not None
                and self._lost_frames < self.p.max_lost_frames):
            x1, y1, x2, y2 = self._last_bbox
            # Incremental widening: each consecutive lost frame grows the search
            # window rather than instantly exploding to the full frame.  This
            # rejects momentary noise/dropout without losing the lock.
            pad  = self.p.roi_pad_px + self.p.roi_expand_px * self._lost_frames
            rx1  = max(0, x1 - pad);  ry1 = max(0, y1 - pad)
            rx2  = min(W, x2 + pad);  ry2 = min(H, y2 + pad)
            # Guard: bbox may have degenerate coords after large motion
            if rx2 <= rx1 or ry2 <= ry1:
                rx1, ry1, rx2, ry2 = 0, 0, W, H
                self._last_bbox    = None
        else:
            # Lost for too long (or never acquired) → full-frame search and drop
            # the stale window so a fresh lock can form.  Also clear the base
            # anchor so a repositioned fixture can re-bootstrap via the entry
            # heuristic instead of soft-failing forever against a stale anchor.
            rx1, ry1, rx2, ry2 = 0, 0, W, H
            self._last_bbox     = None
            self._anchor_base_px = None

        ox, oy       = rx1, ry1
        color_roi    = color[ry1:ry2, rx1:rx2]
        depth_roi    = depth[ry1:ry2, rx1:rx2]

        # ── Step 1b: depth gate mask ──────────────────────────────────────────
        # Build gate from the post-processed depth.  Then forcibly include any
        # pixel that had zero depth in the RAW frame (rod silhouette): the
        # spatial filter fills those holes with interpolated background values,
        # which would otherwise be gated out as background.
        gate = self._depth_gate(depth_roi)
        if depth_raw is not None:
            raw_roi    = depth_raw[ry1:ry2, rx1:rx2]
            rod_holes  = (raw_roi == 0).astype(np.uint8) * 255  # where IR was absorbed
            gate       = cv2.bitwise_or(gate, rod_holes)

        # ── Step 2: rod mask — learned U-Net OR classical dark threshold ──────
        if self._segmenter is not None:
            # Semantic segmentation: the network knows "rod" vs "dark stuff".
            fg = self._segmenter.segment(color_roi)        # 0/255, ROI-sized
            # Optionally intersect with the depth gate to drop far-background
            # false positives (network usually handles this, but cheap insurance)
            mask = cv2.bitwise_and(fg, gate) if self.p.seg_use_depth_gate else fg
        else:
            # Classical fallback: dark-pixel threshold ∩ depth gate
            gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
            _, dark = cv2.threshold(
                gray, self.p.dark_threshold, 255, cv2.THRESH_BINARY_INV)
            if int(dark.sum()) < self.p.min_area_px * 255 * 4:
                gray_clipped = np.clip(gray, 0, 120).astype(np.uint8)
                _, dark = cv2.threshold(
                    gray_clipped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            mask = cv2.bitwise_and(dark, gate)

        # ── Step 3: morphological cleanup + elongated connected component ─────
        mask     = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._se3,
                                    iterations=self.p.morph_open_iter)
        mask     = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._se3,
                                    iterations=self.p.morph_close_iter)
        # Pass the base anchor (ROI-local) so the rod CC can be picked by
        # walking from the fixed base, ignoring any attached hand component.
        anchor_roi = (None if self._anchor_base_px is None
                      else self._anchor_base_px - np.array([ox, oy], dtype=np.float64))
        rod_mask = self._largest_elongated_cc(mask, base_anchor_roi=anchor_roi)
        if rod_mask is None:
            return self._fail(color, depth, gate, None, None, None, (ox,oy))

        # ── Step 4: skeletonisation → ordered skeleton ────────────────────────
        skel_pts = self._skeletonise(rod_mask)
        if skel_pts is None:
            return self._fail(color, depth, gate, rod_mask, None, None, (ox,oy))

        ep_a, ep_b      = self._find_endpoints(skel_pts)
        base_rc, tip_rc = self._assign_base_tip(ep_a, ep_b, offset=(ox, oy))
        ordered         = self._order_path(skel_pts, base_rc, tip_rc)

        # ── Step 4b: 2-D centerline from skeleton (always available) ──────────
        # Build curve_px directly from skeleton pixels so the overlay is drawn
        # even when 3-D projection fails (no depth / intrinsics not ready).
        skel_curve_px = (ordered[:, ::-1].astype(np.float32)
                         + np.array([ox, oy], dtype=np.float32))  # (N,2) col,row
        tip_px_skel   = tuple(skel_curve_px[-1].tolist())
        base_px_skel  = tuple(skel_curve_px[0].tolist())
        cand_bbox     = self._bbox_of(skel_curve_px)

        # ── ROI explosion guard ───────────────────────────────────────────────
        # If the new bbox blew up / teleported (noise pulling in background),
        # reject this frame's detection without destroying the lock.
        if not self._roi_plausible(cand_bbox):
            return self._soft_fail(color, depth, gate, rod_mask,
                                   skel_curve_px, (ox, oy))

        # ── Base-anchor sanity ────────────────────────────────────────────────
        # With a base anchor established, the assigned base must sit near it.
        # If the nearest endpoint is implausibly far, the detection is bad
        # (e.g. a transient blob) — soft-fail rather than corrupt the anchor.
        base_uv = np.array(base_px_skel, dtype=np.float64)
        if self._anchor_base_px is not None:
            if float(np.linalg.norm(base_uv - self._anchor_base_px)) > \
                    self.p.base_lock_dist_px:
                return self._soft_fail(color, depth, gate, rod_mask,
                                       skel_curve_px, (ox, oy))

        # Accept: lock the ROI, update the static base anchor, reset lost count.
        self._last_bbox          = cand_bbox
        self._lost_frames        = 0
        self._last_good_curve_px = skel_curve_px
        self._update_base_anchor(base_uv)

        # Keep a tip EMA too (reference / HUD only; not used for disambiguation)
        tip_px_arr = np.array(tip_px_skel, dtype=np.float64)
        if self._smooth_tip_px is None:
            self._smooth_tip_px = tip_px_arr.copy()
        else:
            self._smooth_tip_px = (self.p.ema_alpha * tip_px_arr
                                   + (1 - self.p.ema_alpha) * self._smooth_tip_px)

        # ── Step 5: project skeleton pixels → 3-D points ─────────────────────
        pts3d_m, valid_mask_3d = self._project_skeleton_3d(
            ordered, depth_roi, ox, oy)

        if pts3d_m is None:
            # 3-D unavailable — still return a result with 2-D overlay visible
            result = TrackResult(
                valid         = False,
                centerline_px = skel_curve_px,
                centerline_3d = np.zeros((0, 3), dtype=np.float64),
                arc_length_mm = self._smooth_arc,
                tip_px        = tip_px_skel,
                base_px       = base_px_skel,
                tip_xyz_mm    = tuple(self._smooth_tip_3d.tolist())
                                if self._smooth_tip_3d is not None else (0., 0., 0.),
                mid_xyz_mm    = (0., 0., 0.),
            )
            if self.p.debug:
                result.debug_frame = self._make_debug(
                    color, depth, gate, rod_mask, skel_curve_px, result, (ox, oy))
            return result

        # ── Step 6: interpolate missing depths + 3-D B-spline ─────────────────
        pts3d_smooth, u_vals = self._smooth_3d_spline(pts3d_m, valid_mask_3d)

        if pts3d_smooth is None:
            result = TrackResult(
                valid         = False,
                centerline_px = skel_curve_px,
                centerline_3d = np.zeros((0, 3), dtype=np.float64),
                arc_length_mm = self._smooth_arc,
                tip_px        = tip_px_skel,
                base_px       = base_px_skel,
                tip_xyz_mm    = tuple(self._smooth_tip_3d.tolist())
                                if self._smooth_tip_3d is not None else (0., 0., 0.),
                mid_xyz_mm    = (0., 0., 0.),
            )
            if self.p.debug:
                result.debug_frame = self._make_debug(
                    color, depth, gate, rod_mask, skel_curve_px, result, (ox, oy))
            return result

        # ── Step 7a: arc length (mm) ──────────────────────────────────────────
        diffs      = np.diff(pts3d_smooth, axis=0)
        seg_lens_m = np.linalg.norm(diffs, axis=1)
        arc_mm     = float(np.sum(seg_lens_m)) * 1000.0

        # ── Step 7b: tip and mid in 3-D (mm) ─────────────────────────────────
        tip_3d_mm = pts3d_smooth[-1] * 1000.0
        mid_3d_mm = pts3d_smooth[len(pts3d_smooth) // 2] * 1000.0

        # ── Step 7c: prefer re-projected 2-D centerline when intrinsics ready ─
        # NOTE: the authoritative ROI (_last_bbox) was already set + validated
        # from the skeleton curve in Step 4b.  We do NOT overwrite it here:
        # reprojection can clamp degenerate-Z points to the principal point,
        # which would create outliers and re-explode the bbox.  Reproj is used
        # for the displayed centerline only.
        reproj = self._reproject_3d_to_2d(pts3d_smooth)
        if reproj is not None and self._roi_plausible(self._bbox_of(reproj)):
            curve_px = reproj
            tip_px   = tuple(curve_px[-1].tolist())
            base_px  = tuple(curve_px[0].tolist())
        else:
            curve_px = skel_curve_px
            tip_px   = tip_px_skel
            base_px  = base_px_skel
        self._last_good_curve_px = curve_px

        # ── Step 8: temporal smoothing + spike rejection ───────────────────────
        if self._smooth_tip_3d is not None:
            jump_mm = float(np.linalg.norm(tip_3d_mm - self._smooth_tip_3d))
            if jump_mm > self.p.max_jump_mm:
                result = TrackResult(
                    valid         = False,
                    centerline_px = curve_px,
                    centerline_3d = pts3d_smooth * 1000.0,
                    arc_length_mm = self._smooth_arc,
                    tip_px        = tip_px,
                    base_px       = base_px,
                    tip_xyz_mm    = tuple(self._smooth_tip_3d.tolist()),
                    mid_xyz_mm    = tuple(mid_3d_mm.tolist()),
                )
                if self.p.debug:
                    result.debug_frame = self._make_debug(
                        color, depth, gate, rod_mask, curve_px, result, (ox, oy))
                return result

        a = self.p.ema_alpha
        if self._smooth_tip_3d is None:
            self._smooth_tip_3d = tip_3d_mm.copy()
            self._smooth_arc    = arc_mm
        else:
            self._smooth_tip_3d = a * tip_3d_mm + (1 - a) * self._smooth_tip_3d
            self._smooth_arc    = a * arc_mm     + (1 - a) * self._smooth_arc

        result = TrackResult(
            valid         = True,
            centerline_px = curve_px,
            centerline_3d = pts3d_smooth * 1000.0,
            arc_length_mm = self._smooth_arc,
            tip_px        = tip_px,
            base_px       = base_px,
            tip_xyz_mm    = tuple(self._smooth_tip_3d.tolist()),
            mid_xyz_mm    = tuple(mid_3d_mm.tolist()),
        )

        if self.p.debug:
            result.debug_frame = self._make_debug(
                color, depth, gate, rod_mask, curve_px, result, (ox, oy))

        return result

    # ── Live streaming generator ──────────────────────────────────────────────

    def run(self, report_path: Optional[str] = None):
        """
        Generator that streams from the connected RealSense D435i and yields
        one TrackResult per frame.  Press Q or ESC to stop.

        Args:
            report_path: if given, a PDF performance report is written to this
                         path automatically when the stream ends.
        """
        pipeline = rs.pipeline()
        cfg      = rs.config()
        W, H, FPS = self.p.width, self.p.height, self.p.fps
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16,  FPS)

        profile      = pipeline.start(cfg)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        try:
            if self.p.laser_power == 0:
                depth_sensor.set_option(rs.option.emitter_enabled, 0)
            else:
                hi  = depth_sensor.get_option_range(rs.option.laser_power).max
                pwr = max(1, min(int(hi), self.p.laser_power))
                depth_sensor.set_option(rs.option.emitter_enabled, 1)
                depth_sensor.set_option(rs.option.laser_power, pwr)
        except Exception as e:
            print(f"[WARN] emitter config: {e}")

        align = rs.align(rs.stream.color)
        for _ in range(5):
            frames  = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
        color_profile = aligned.get_color_frame().get_profile()
        self.set_intrinsics(
            color_profile.as_video_stream_profile().get_intrinsics())
        print(f"Intrinsics: fx={self._intr.fx:.2f} fy={self._intr.fy:.2f} "
              f"cx={self._intr.ppx:.2f} cy={self._intr.ppy:.2f}")

        fps_buf  = deque(maxlen=30)
        win_name = "MSCR Tracker" if self.p.debug else None
        if win_name:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, W * 2, H)

        print(f"Streaming {W}×{H} @ {FPS} fps  |  depth_scale={self.depth_scale:.5f} m/unit")
        print("Press  Q  or  ESC  to stop.\n")

        # Metric buffers (populated only when report_path is set)
        _metrics: list = []
        _t_start        = time.perf_counter()
        _prev_tip       = np.zeros(3)

        try:
            while True:
                t0      = time.perf_counter()
                frames  = pipeline.wait_for_frames(timeout_ms=5000)
                aligned = align.process(frames)

                color_f = aligned.get_color_frame()
                depth_f = aligned.get_depth_frame()
                if not color_f or not depth_f:
                    continue

                color     = np.asanyarray(color_f.get_data())
                depth_raw = np.asanyarray(depth_f.get_data())
                depth_filt = self._spatial.process(depth_f)
                depth_filt = self._temporal.process(depth_filt)
                depth      = np.asanyarray(depth_filt.get_data())

                result  = self.process_frame(color, depth, depth_raw=depth_raw)
                proc_ms = (time.perf_counter() - t0) * 1000.0

                fps_buf.append(1000.0 / max(proc_ms, 1e-3))
                fps = float(np.mean(fps_buf))

                skel_visible = len(result.centerline_px) > 0
                if result.valid:
                    x, y, z = result.tip_xyz_mm
                    print(f"arc={result.arc_length_mm:7.2f} mm  |  "
                          f"tip=({x:+7.2f}, {y:+7.2f}, {z:+7.2f}) mm  |  "
                          f"{fps:.1f} fps", flush=True)
                elif skel_visible:
                    n = len(result.centerline_px)
                    print(f"[skeleton {n}px — no 3D depth]  {fps:.1f} fps",
                          flush=True)
                else:
                    print(f"[no detection]  {fps:.1f} fps", flush=True)

                # Accumulate metrics for the report
                if report_path:
                    tip    = np.array(result.tip_xyz_mm)
                    jump   = (float(np.linalg.norm(tip - _prev_tip))
                              if result.valid else 0.0)
                    if result.valid:
                        _prev_tip = tip.copy()
                    _metrics.append({
                        "frame_idx":        len(_metrics),
                        "timestamp_s":      time.perf_counter() - _t_start,
                        "proc_time_ms":     proc_ms,
                        "valid":            result.valid,
                        "skeleton_found":   len(result.centerline_px) > 0,
                        "n_skel_pts":       getattr(self, "_last_n_skel", 0),
                        "valid_depth_frac": getattr(self, "_last_valid_df", 0.0),
                        "arc_length_mm":    result.arc_length_mm,
                        "tip_x_mm":         tip[0],
                        "tip_y_mm":         tip[1],
                        "tip_z_mm":         tip[2],
                        "tip_jump_mm":      jump,
                    })

                yield result

                if win_name and result.debug_frame is not None:
                    cv2.imshow(win_name, result.debug_frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

        except KeyboardInterrupt:
            pass
        finally:
            pipeline.stop()
            if win_name:
                cv2.destroyAllWindows()

        # Generate PDF after stream ends
        if report_path and _metrics:
            print(f"\nGenerating performance report → {report_path} …")
            try:
                from mscr_performance_report import (
                    FrameMetric, generate_pdf)
                frame_metrics = [FrameMetric(**m) for m in _metrics]
                generate_pdf(frame_metrics, out_path=report_path,
                             params=self.p)
            except Exception as e:
                print(f"[WARN] Report generation failed: {e}")

    # ── Step helpers ──────────────────────────────────────────────────────────

    def _depth_gate(self, depth: np.ndarray) -> np.ndarray:
        """
        Return a uint8 mask of pixels that should NOT be excluded as background.

        Pixels with zero depth (rod absorbs IR → no return) are always INCLUDED
        so we never accidentally erase the robot silhouette.
        """
        if depth.size == 0:
            return np.zeros(depth.shape, dtype=np.uint8)

        depth_m = depth.astype(np.float32) * self.depth_scale
        far_bg  = ((depth > 0) & (
                   (depth_m > self.p.depth_max_m) |
                   (depth_m < self.p.depth_min_m))).astype(np.uint8) * 255

        # Erode the exclusion mask so rod edges adjacent to background survive.
        # Skip if the ROI is smaller than the structuring element.
        if self.p.depth_dilation_px > 0:
            if self._se_depth is None:
                r = self.p.depth_dilation_px * 2 + 1
                self._se_depth = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (r, r))
            se_h, se_w = self._se_depth.shape[:2]
            if depth.shape[0] >= se_h and depth.shape[1] >= se_w:
                far_bg = cv2.erode(far_bg, self._se_depth)

        return cv2.bitwise_not(far_bg)

    def _isolate_slender(self, mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Remove thick blobs (hand/arm/fingers) and keep slender structures.

        A morphological OPEN with a kernel wider than the rod erases everything
        thinner than ~rod_max_width_px (the rod) and keeps thicker blobs (the
        hand).  Subtracting that 'thick' set from the original mask therefore
        leaves the slender rod while discarding the hand body.

        Returns:
            thin_mask  : the mask with thick blobs removed
            thick_frac : fraction of the original foreground that was thick
                         (used to decide whether a hand is present)
        """
        if not self.p.slender_isolation:
            return mask, 0.0
        fg = int(np.count_nonzero(mask))
        if fg == 0:
            return mask, 0.0

        # Kernel diameter must sit BETWEEN the rod width and the finger width:
        # an OPEN with diameter D removes structures narrower than D.  We want
        # to remove the rod (≤ rod_max_width_px) and keep fingers (wider), so
        # D = rod_max_width_px + small margin.  (Too large a D also erases the
        # fingers and they leak back into the thin mask.)
        k  = self.p.rod_max_width_px + 3
        if k % 2 == 0:
            k += 1
        k  = max(3, k)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        thick = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)   # blobs wider than rod
        thin  = cv2.subtract(mask, thick)                    # slender structures
        # Re-close small gaps the subtraction may have opened along the rod
        thin  = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, self._se3, iterations=1)

        thick_frac = float(np.count_nonzero(thick)) / max(fg, 1)
        # If isolation removed essentially all foreground (rod thicker than the
        # kernel, or no rod), fall back to the original mask so we never blind
        # ourselves — the downstream eccentricity test still filters blobs.
        if int(np.count_nonzero(thin)) < self.p.min_area_px:
            return mask, thick_frac
        return thin, thick_frac

    def _largest_elongated_cc(self,
                              mask: np.ndarray,
                              base_anchor_roi: Optional[np.ndarray] = None
                              ) -> Optional[np.ndarray]:
        """
        Select the rod's connected component, robust to a hand in the frame.

        Steps:
          1. Slender isolation: strip thick blobs (hand/arm) before labelling.
          2. Among elongated candidates, when a base anchor is known, pick the
             component NEAREST the anchor — i.e. walk down the rod from its
             fixed base and ignore any large branching hand component.  Without
             an anchor, fall back to the most eccentric (and larger) candidate.

        Sets self._hand_present for this frame.
        """
        # ── Step 1: isolate slender structures (drop the hand body) ───────────
        thin, thick_frac  = self._isolate_slender(mask)
        self._hand_present = thick_frac > self.p.hand_thick_frac
        work = thin

        H, W   = work.shape
        maxpx  = int(H * W * self.p.max_area_frac)
        # With slender isolation the hand is already gone, so a generous area
        # ceiling is safe and avoids dropping a long rod on a big frame.
        if self.p.slender_isolation:
            maxpx = max(maxpx, int(H * W * 0.20))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            work, connectivity=8)

        candidates = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not (self.p.min_area_px <= area <= maxpx):
                continue
            pts = np.column_stack(np.where(labels == i))[:, ::-1].astype(np.float32)
            if len(pts) < 5:
                continue
            try:
                (_, _), (ma, Mi), _ = cv2.fitEllipse(pts)
                minor, major = sorted([ma, Mi])
                ecc = float(np.sqrt(1.0 - (minor / max(major, 1e-6)) ** 2))
            except Exception:
                ecc = 0.0
            # Distance from this CC to the known base anchor (ROI-local)
            if base_anchor_roi is not None:
                anchor_dist = float(np.min(
                    np.linalg.norm(pts - base_anchor_roi, axis=1)))
            else:
                anchor_dist = np.inf
            candidates.append((i, area, ecc, anchor_dist))

        if not candidates:
            return None

        ecc_ok = [c for c in candidates if c[2] >= self.p.min_eccentricity]
        pool   = ecc_ok if ecc_ok else candidates

        if base_anchor_roi is not None and np.isfinite(pool[0][3]):
            # Walk down the rod from its base: choose the component that the
            # base anchor belongs to / is nearest, ignoring hand branches.
            pool.sort(key=lambda c: c[3])               # nearest the base
        else:
            pool.sort(key=lambda c: (c[2], c[1]), reverse=True)  # most elongated
        return (labels == pool[0][0]).astype(np.uint8) * 255

    def _skeletonise(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Skeletonise the rod mask and prune short side-branches so the
        returned pixel set is as close to a single unbranched curve as
        possible.

        Returns (N, 2) array of (row, col) or None if the result is
        outside [min_skel_pts, max_skel_pts].
        """
        skel = _skeletonize(mask > 0)
        pts  = np.column_stack(np.where(skel))
        if len(pts) < self.p.min_skel_pts:
            return None

        # Build neighbour count map — branch junctions have ≥ 3 neighbours
        pset = {(int(r), int(c)) for r, c in pts}
        pts  = self._prune_skeleton_branches(pset)

        if len(pts) < self.p.min_skel_pts or len(pts) > self.p.max_skel_pts:
            return None
        return pts

    def _prune_skeleton_branches(self,
                                  pset: set,
                                  min_branch_len: int = 8
                                  ) -> np.ndarray:
        """
        Remove skeleton side-branches shorter than `min_branch_len` pixels.

        Algorithm:
          1. Find all junction pixels (≥ 3 neighbours).
          2. From every non-junction endpoint, walk the skeleton. If the
             walk reaches a junction before accumulating min_branch_len
             steps, remove those pixels.
          3. Repeat until stable (one pass is usually enough).
        """
        def nbrs(r, c):
            return [(r+dr, c+dc) for dr in (-1,0,1) for dc in (-1,0,1)
                    if (dr,dc) != (0,0) and (r+dr,c+dc) in pset]

        def n_nbr(r, c): return len(nbrs(r, c))

        for _ in range(3):   # iterate to handle cascading short branches
            junctions = {p for p in pset if n_nbr(*p) >= 3}
            endpoints  = [p for p in pset if n_nbr(*p) == 1]
            to_remove  = set()
            for ep in endpoints:
                arm   = [ep]
                visited = {ep}
                cur   = ep
                while True:
                    nbs = [n for n in nbrs(*cur) if n not in visited]
                    if not nbs:
                        break
                    nxt = nbs[0]
                    if nxt in junctions:
                        # Reached a junction — prune if arm is short
                        if len(arm) < min_branch_len:
                            to_remove.update(arm)
                        break
                    arm.append(nxt)
                    visited.add(nxt)
                    cur = nxt
                    if len(arm) >= min_branch_len:
                        break
            if not to_remove:
                break
            pset -= to_remove

        return np.array(sorted(pset), dtype=np.float32)   # (N,2) row,col

    def _find_endpoints(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the two true endpoints of the pruned skeleton — pixels with
        exactly one 8-connected neighbour.  Falls back to the most-distant
        pair if the skeleton still has junctions.
        """
        pset = {(int(r), int(c)) for r, c in pts}

        def n_nbrs(r, c):
            return sum(1 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                       if (dr, dc) != (0, 0) and (r+dr, c+dc) in pset)

        endpoints = [np.array(p) for p in pset if n_nbrs(*p) == 1]
        if len(endpoints) < 2:
            # Closed loop or single-endpoint stub — fall back to array extremes
            endpoints = [pts[0], pts[-1]]
        if len(endpoints) == 2:
            return endpoints[0], endpoints[1]

        # Pick the most-distant pair
        best_d, ep_a, ep_b = -1.0, endpoints[0], endpoints[1]
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                d = float(np.linalg.norm(endpoints[i] - endpoints[j]))
                if d > best_d:
                    best_d, ep_a, ep_b = d, endpoints[i], endpoints[j]
        return ep_a, ep_b

    def _order_path(self,
                    pts:   np.ndarray,
                    start: np.ndarray,
                    end:   np.ndarray) -> np.ndarray:
        """
        Greedy nearest-neighbour walk from `start` toward `end` through the
        skeleton pixel set.

        Unlike DFS, this never backtracks so it cannot detour through
        side-branches and always produces a smooth, monotone path.
        """
        pset      = {(int(r), int(c)) for r, c in pts}
        remaining = set(pset)
        path      = []
        cur       = (int(start[0]), int(start[1]))
        end_rc    = (int(end[0]),   int(end[1]))

        while remaining:
            path.append(cur)
            remaining.discard(cur)

            # 8-connected neighbours still in the unvisited set
            nbs = [(r, c) for r, c in [
                       (cur[0]+dr, cur[1]+dc)
                       for dr in (-1,0,1) for dc in (-1,0,1)
                       if (dr,dc) != (0,0)]
                   if (r, c) in remaining]

            if not nbs:
                break

            # If only one neighbour: take it (common case on a clean skeleton)
            if len(nbs) == 1:
                cur = nbs[0]
                continue

            # Multiple neighbours: prefer the one closest to `end`
            er, ec = end_rc
            nbs.sort(key=lambda p: (p[0]-er)**2 + (p[1]-ec)**2)
            cur = nbs[0]

        return np.array(path, dtype=np.float32) if len(path) >= 2 else pts

    def _assign_base_tip(self,
                         ep_a:    np.ndarray,
                         ep_b:    np.ndarray,
                         offset:  Tuple[int, int] = (0, 0)
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign (base, tip) from two skeleton endpoints.

        Strategy — anchor on the STATIC base, not the moving tip:
          1. If a base anchor exists (slow EMA of the fixed mount), the
             endpoint nearer to it is the base, the other is the tip.  Because
             the base barely moves, this is far more stable than tip-proximity
             and does not flip when the rod bends sharply.
          2. Otherwise bootstrap with the entry-border heuristic.

        Returns endpoints in their original ROI-local (row, col) form.
        """
        ox, oy = offset
        a_uv = np.array([ep_a[1] + ox, ep_a[0] + oy], dtype=float)
        b_uv = np.array([ep_b[1] + ox, ep_b[0] + oy], dtype=float)

        if self._anchor_base_px is not None:
            da = float(np.linalg.norm(a_uv - self._anchor_base_px))
            db = float(np.linalg.norm(b_uv - self._anchor_base_px))
            # Endpoint nearest the static base anchor → base
            if da <= db:
                return ep_a, ep_b
            return ep_b, ep_a

        # Bootstrap: entry-border heuristic (no anchor yet)
        entry = self.p.entry

        def border_score(ep):
            r, c = ep[0], ep[1]
            if entry == "bottom": return -r
            if entry == "top":    return  r
            if entry == "right":  return -c
            if entry == "left":   return  c
            return 0.0

        if border_score(ep_a) <= border_score(ep_b):
            return ep_a, ep_b
        return ep_b, ep_a

    def _project_skeleton_3d(self,
                              ordered:   np.ndarray,
                              depth_roi: np.ndarray,
                              ox: int,
                              oy: int,
                              search_r:  int = -1
                              ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        For each ordered skeleton pixel (row, col), find the nearest non-zero
        depth value within a search_r-pixel radius rather than sampling
        exactly at the centerline pixel.

        Rationale: the rod absorbs IR and returns depth=0 at its centreline.
        The background immediately beside a slender rod is at essentially the
        same depth as the rod itself, so a nearby non-zero pixel is a reliable
        depth proxy.  Sampling at the centerline pixel directly gives zero for
        almost every point, making 3-D reconstruction fail even when the
        skeleton is perfectly segmented.

        The search is done via a pre-built ring of offsets sorted by distance
        so the closest valid pixel is found first with minimal iterations.

        Returns:
            pts3d_m    : (N, 3) float64 metres  (zeros where no neighbour found)
            valid_mask : (N,) bool — True where a valid depth neighbour existed
        """
        if self._intr is None:
            return None, np.array([], dtype=bool)

        if search_r < 0:
            search_r = self.p.depth_search_r
            # A hand near the rod perturbs the surrounding depth; widen the
            # proxy search so we still find a valid neighbour for each pixel.
            if self._hand_present:
                search_r = max(search_r, self.p.depth_search_r_hand)

        # Pre-build offset ring: all (dr, dc) within search_r, sorted by dist
        offsets = sorted(
            [(dr, dc)
             for dr in range(-search_r, search_r + 1)
             for dc in range(-search_r, search_r + 1)
             if dr * dr + dc * dc <= search_r * search_r],
            key=lambda p: p[0] * p[0] + p[1] * p[1])

        N          = len(ordered)
        pts3d_m    = np.zeros((N, 3), dtype=np.float64)
        valid_mask = np.zeros(N, dtype=bool)
        dh, dw     = depth_roi.shape

        for i, (r, c) in enumerate(ordered):
            ri, ci = int(r), int(c)

            # Walk outward from the skeleton pixel until a non-zero depth found
            d_raw = 0
            best_r, best_c = ri, ci
            for dr, dc in offsets:
                nr, nc = ri + dr, ci + dc
                if 0 <= nr < dh and 0 <= nc < dw:
                    val = int(depth_roi[nr, nc])
                    if val > 0:
                        d_raw  = val
                        best_r = nr
                        best_c = nc
                        break

            if d_raw > 0:
                d_m           = d_raw * self.depth_scale
                u_full        = float(best_c + ox)
                v_full        = float(best_r + oy)
                pts3d_m[i]    = self._deproject(u_full, v_full, d_m)
                valid_mask[i] = True

        valid_frac = float(valid_mask.sum()) / max(N, 1)
        if valid_frac < self.p.min_valid_depth_frac:
            return None, valid_mask

        return pts3d_m, valid_mask

    def _smooth_3d_spline(self,
                           pts3d_m:    np.ndarray,
                           valid_mask: np.ndarray
                           ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        1. Linearly interpolate missing depth values along the skeleton.
        2. Fit a 3-D B-spline (splprep) to the full point sequence.
        3. Uniformly resample to n_resample points.

        Returns (smoothed_pts_m, u_vals) or (None, None) on failure.
        """
        N = len(pts3d_m)

        # ── Linear interpolation of missing-depth points ──────────────────────
        indices = np.arange(N)
        valid_i = indices[valid_mask]

        if len(valid_i) < 2:
            return None, None

        for dim in range(3):
            pts3d_m[:, dim] = np.interp(
                indices, valid_i, pts3d_m[valid_i, dim])

        # ── 3-D B-spline fit ──────────────────────────────────────────────────
        s = N * self.p.spline_smooth
        try:
            tck, _ = splprep(
                [pts3d_m[:, 0], pts3d_m[:, 1], pts3d_m[:, 2]],
                s=s, k=3, quiet=True)
        except Exception:
            return None, None

        # Dense evaluation to compute true arc-length parametrisation
        u_dense = np.linspace(0.0, 1.0, 4000)
        x_d, y_d, z_d = splev(u_dense, tck)
        dl       = np.sqrt(np.diff(x_d)**2 + np.diff(y_d)**2 + np.diff(z_d)**2)
        arc_cum  = np.concatenate([[0.0], np.cumsum(dl)])
        arc_total = float(arc_cum[-1])

        # Resample uniformly by arc length
        u_uni = np.interp(
            np.linspace(0.0, arc_total, self.p.n_resample),
            arc_cum, u_dense)

        xu, yu, zu = splev(u_uni, tck)
        smoothed   = np.column_stack([xu, yu, zu])   # (n_resample, 3) metres

        return smoothed, u_uni

    def _reproject_3d_to_2d(self,
                             pts3d_m: np.ndarray) -> Optional[np.ndarray]:
        """
        Project 3-D points (metres) back onto the image plane using the stored
        camera intrinsics.  Returns (N, 2) float32 array of (col, row).
        """
        if self._intr is None or len(pts3d_m) == 0:
            return None

        fx, fy = self._intr.fx, self._intr.fy
        cx, cy = self._intr.ppx, self._intr.ppy

        X, Y, Z = pts3d_m[:, 0], pts3d_m[:, 1], pts3d_m[:, 2]
        # Guard against Z ≈ 0 (degenerate points after interpolation)
        valid = np.abs(Z) > 1e-4
        u = np.where(valid, fx * X / Z + cx, cx)
        v = np.where(valid, fy * Y / Z + cy, cy)
        return np.column_stack([u, v]).astype(np.float32)

    # ── ROI helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _bbox_of(curve_px: np.ndarray) -> Tuple[int, int, int, int]:
        return (int(curve_px[:, 0].min()), int(curve_px[:, 1].min()),
                int(curve_px[:, 0].max()), int(curve_px[:, 1].max()))

    def _roi_plausible(self, cand: Tuple[int, int, int, int]) -> bool:
        """
        True if a candidate bbox is a physically plausible successor to the
        previous one — i.e. it did not explode in size or teleport.

        Rejects noise-driven blow-ups (mask suddenly engulfs the background)
        and impossible single-frame jumps.  Always True on the first detection.
        """
        if self._last_bbox is None:
            return True
        px1, py1, px2, py2 = self._last_bbox
        cx1, cy1, cx2, cy2 = cand
        # Compare bbox DIAGONAL span (≈ rod length), not area: a thin rod's
        # bbox area collapses to ~0 when it is vertical/horizontal, so an
        # area ratio would spuriously flag normal frames as explosions.
        prev_diag = max(8.0, float(np.hypot(px2 - px1, py2 - py1)))
        cand_diag = max(8.0, float(np.hypot(cx2 - cx1, cy2 - cy1)))

        # Relax the ceilings while a hand is present: manual manipulation
        # introduces larger, legitimate single-frame changes that the rigid
        # thresholds would otherwise reject.
        relax  = self.p.hand_relax_factor if self._hand_present else 1.0
        grow   = self.p.roi_max_growth   * relax
        shift  = self.p.roi_max_shift_px * relax

        ratio = cand_diag / prev_diag
        if ratio > grow or ratio < 1.0 / grow:
            return False
        prev_c = np.array([(px1 + px2) / 2, (py1 + py2) / 2])
        cand_c = np.array([(cx1 + cx2) / 2, (cy1 + cy2) / 2])
        if float(np.linalg.norm(cand_c - prev_c)) > shift:
            return False
        return True

    def _update_base_anchor(self, base_uv: np.ndarray) -> None:
        """Slow-EMA update of the static base anchor (full-image col,row)."""
        base_uv = np.asarray(base_uv, dtype=np.float64)
        if self._anchor_base_px is None:
            self._anchor_base_px = base_uv.copy()
        else:
            a = self.p.base_ema_alpha
            self._anchor_base_px = a * base_uv + (1 - a) * self._anchor_base_px

    def _soft_fail(self, color, depth, gate, rod_mask, curve_px, offset):
        """
        Reject the current frame's detection but DON'T blow up the ROI.

        Increments the lost counter (so the search window widens incrementally
        next frame) and returns the last accepted centerline/tip so downstream
        consumers see a held pose rather than a spike or a hard LOST.
        """
        self._lost_frames += 1
        fallback = (self._last_good_curve_px
                    if self._last_good_curve_px is not None else curve_px)
        if fallback is None or len(fallback) < 2:
            return self._fail(color, depth, gate, rod_mask, None, curve_px, offset)
        result = TrackResult(
            valid         = False,
            centerline_px = fallback,
            centerline_3d = np.zeros((0, 3), dtype=np.float64),
            arc_length_mm = self._smooth_arc,
            tip_px        = tuple(fallback[-1].tolist()),
            base_px       = tuple(fallback[0].tolist()),
            tip_xyz_mm    = tuple(self._smooth_tip_3d.tolist())
                            if self._smooth_tip_3d is not None else (0., 0., 0.),
            mid_xyz_mm    = (0., 0., 0.),
        )
        if self.p.debug:
            result.debug_frame = self._make_debug(
                color, depth, gate, rod_mask, fallback, result, offset)
        return result

    # ── Failure helper ────────────────────────────────────────────────────────

    def _fail(self, color, depth, gate, rod_mask, skel, curve_px, offset):
        self._lost_frames += 1
        result = TrackResult.invalid()
        if self.p.debug:
            result.debug_frame = self._make_debug(
                color, depth, gate, rod_mask, curve_px, result, offset)
        return result

    # ── Debug visualisation ───────────────────────────────────────────────────

    def _make_debug(self,
                    color:    np.ndarray,
                    depth:    np.ndarray,
                    gate:     np.ndarray,
                    rod_mask: Optional[np.ndarray],
                    curve_px: Optional[np.ndarray],
                    result:   TrackResult,
                    offset:   Tuple[int, int]) -> np.ndarray:
        """
        Compose a side-by-side debug frame:
          Left  — colour image with 3-D centerline overlay and HUD
          Right — depth-gate mask with rod silhouette overlay
        """
        ox, oy = offset
        H, W   = color.shape[:2]
        vis    = color.copy()

        # ── Draw rod centerline (blue polyline) ───────────────────────────────
        if curve_px is not None and len(curve_px) > 1:
            pts_draw = curve_px.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts_draw], False, (255, 120, 0), 2, cv2.LINE_AA)

            base_pt = tuple(curve_px[0].astype(int))
            mid_pt  = tuple(curve_px[len(curve_px) // 2].astype(int))
            tip_pt  = tuple(curve_px[-1].astype(int))

            # Base: green filled circle
            cv2.circle(vis, base_pt, 8, (0, 210, 0), -1)
            cv2.putText(vis, "base", (base_pt[0] + 10, base_pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 0), 1)

            # Mid: cyan marker
            cv2.circle(vis, mid_pt, 6, (0, 220, 220), -1)

            # Tip: red filled circle + outer ring
            cv2.circle(vis, tip_pt, 10, (0, 0, 255), 2)
            cv2.circle(vis, tip_pt, 5,  (0, 0, 255), -1)
            cv2.putText(vis, "tip", (tip_pt[0] + 12, tip_pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 60, 255), 1)

        # ── HUD text ─────────────────────────────────────────────────────────
        hud_lines = []
        has_curve  = curve_px is not None and len(curve_px) > 1
        if result.valid:
            x, y, z = result.tip_xyz_mm
            hud_lines += [
                (f"Arc length : {result.arc_length_mm:7.2f} mm", (0, 220, 80)),
                (f"Tip  X     : {x:+8.3f} mm",                   (80, 200, 255)),
                (f"Tip  Y     : {y:+8.3f} mm",                   (80, 200, 255)),
                (f"Tip  Z     : {z:+8.3f} mm",                   (80, 200, 255)),
            ]
        elif has_curve:
            # Skeleton found but 3-D not yet available
            hud_lines += [("2-D skeleton — no depth", (0, 200, 200))]
        else:
            hud_lines += [("SEARCHING...", (0, 180, 230))]

        for k, (text, col) in enumerate(hud_lines):
            cv2.putText(vis, text, (10, 30 + k * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

        # ── Right panel: EMA-smoothed depth-gate mask + rod overlay ──────────
        gate_full = np.zeros((H, W), dtype=np.float32)
        rh, rw    = gate.shape[:2]
        gate_full[oy:oy + rh, ox:ox + rw] = gate.astype(np.float32)

        # EMA smooth the gate image across frames to kill per-pixel flicker
        if self._gate_ema is None or self._gate_ema.shape != gate_full.shape:
            self._gate_ema = gate_full.copy()
        else:
            a = 0.35  # gate EMA alpha — lower = smoother but slower to update
            self._gate_ema = a * gate_full + (1.0 - a) * self._gate_ema

        gate_bgr = cv2.applyColorMap(
            self._gate_ema.astype(np.uint8), cv2.COLORMAP_BONE)

        if rod_mask is not None:
            rod_full = np.zeros((H, W), dtype=np.uint8)
            rod_full[oy:oy + rod_mask.shape[0],
                     ox:ox + rod_mask.shape[1]] = rod_mask
            gate_bgr[rod_full > 0] = (0, 255, 128)   # teal = rod silhouette

        # Status label on both panels
        if result.valid:
            status, s_color = "TRACKING",  (0, 220, 0)
        elif has_curve:
            status, s_color = "SKELETON",  (0, 200, 200)
        else:
            status, s_color = "LOST",      (0, 50, 230)
        cv2.rectangle(vis,      (0, 0), (W, 24), (0, 0, 0), -1)
        cv2.rectangle(gate_bgr, (0, 0), (W, 24), (0, 0, 0), -1)
        cv2.putText(vis,      status,       (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, s_color,     1, cv2.LINE_AA)
        cv2.putText(gate_bgr, "depth gate", (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190,190,190), 1, cv2.LINE_AA)

        return np.hstack([vis, gate_bgr])


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Real-time 3-D MSCR tracker — Intel RealSense D435i")
    ap.add_argument("--width",      type=int,   default=848,
                    choices=[848, 1280])
    ap.add_argument("--height",     type=int,   default=480,
                    choices=[480, 720])
    ap.add_argument("--fps",        type=int,   default=30)
    ap.add_argument("--depth-min",  type=float, default=0.05,
                    help="Minimum valid depth (m)")
    ap.add_argument("--depth-max",  type=float, default=0.80,
                    help="Maximum valid depth (m)")
    ap.add_argument("--threshold",  type=int,   default=85,
                    help="Dark-pixel intensity threshold (0-255). The grey MSCR "
                         "rod needs ~85; lower values miss it.")
    ap.add_argument("--entry",      default="top",
                    choices=["top", "bottom", "left", "right"],
                    help="Image border where the robot base attaches (top for "
                         "the standard fixture)")
    ap.add_argument("--power",      type=int,   default=0,
                    help="Laser/emitter power (0 = off)")
    ap.add_argument("--debug",      action="store_true",
                    help="Show live debug visualisation window")
    ap.add_argument("--report",     nargs="?", const="mscr_report2.pdf",
                    metavar="FILE",
                    help="Write a PDF performance report on exit "
                         "(default filename: mscr_report2.pdf)")
    ap.add_argument("--seg-model",  default=None, metavar="CKPT",
                    help="Path to trained U-Net checkpoint (seg/rod_seg.pt). "
                         "When set, semantic segmentation replaces the dark "
                         "threshold.")
    ap.add_argument("--seg-depth-gate", action="store_true",
                    help="Intersect the U-Net mask with the depth gate")
    args = ap.parse_args()

    params = TrackerParams(
        width          = args.width,
        height         = args.height,
        fps            = args.fps,
        depth_min_m    = args.depth_min,
        depth_max_m    = args.depth_max,
        dark_threshold = args.threshold,
        entry          = args.entry,
        laser_power    = args.power,
        debug          = args.debug,
        seg_model_path     = args.seg_model,
        seg_use_depth_gate = args.seg_depth_gate,
    )

    tracker = MSCRTracker(params)
    for result in tracker.run(report_path=args.report):
        pass
