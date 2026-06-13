"""Phase 2 — stereo ground-truth labeler.

Turns a rectified left/right IR pair of the rod into a 3D centerline r(s)
expressed in the **left-IR rectified camera frame** (mm). Pipeline per pair:

    1. rectify both views (calib maps)
    2. segment the rod (swappable; default = inverse-threshold + morphology)
    3. skeletonize, order pixels into a base->tip path
    4. smoothing-spline + dense resample + subpixel normal refine (left & right)
    5. epipolar correspondence on rectified scanlines (rows = epipolar lines),
       flagging right-curve samples that run parallel to scanlines as DEGENERATE
    6. triangulate valid correspondences with cv2.triangulatePoints(P1, P2)
    7. 3D smoothing spline, uniform-arclength resample -> r(s)
    8. QC: reproject into both views, accept iff mean reproj error < threshold

CLI:  python labeler.py --session <name>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import splev, splprep
from skimage.morphology import skeletonize

from calib import StereoCalib, load_calib, nominal_calib, resolve_calib
from cfg import load_config

# A segmenter maps a grayscale image + params -> binary mask (uint8 0/255).
Segmenter = Callable[[np.ndarray, dict], np.ndarray]


# --------------------------------------------------------------------------- #
# 2. Segmentation  (swappable — a learned segmenter can replace this)
# --------------------------------------------------------------------------- #
def _pick_rod_component(mask: np.ndarray, min_area: int,
                        expect_vertical: bool = True,
                        reject_border_axis: bool = True) -> np.ndarray:
    """Keep the most rod-like connected component.

    The rod is a long thin streak, so score each component by elongation
    (bbox longer-side / shorter-side) weighted by sqrt(area). This beats
    'largest blob', which is fooled by hands, shirts, or vignette corners.

    With a clamped base the scene gains rigid background edges (a table/desk lip
    or windowsill) that are MORE elongated than the rod, so plain elongation
    locks onto them. The rod's orientation is known from base_side: a bottom/top
    base means a VERTICAL rod (bbox taller than wide), so reject HORIZONTAL blobs
    (the table edge is w>>h), and vice-versa for a left/right base. Also reject
    components touching the two borders parallel to the rod axis (a vertical rod
    never reaches the left/right image edge, but background edges run off-frame).
    """
    H, W = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return np.zeros_like(mask)
    best, best_score = 0, -1.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # Orientation gate: drop blobs whose long axis disagrees with the rod's.
        if expect_vertical and h <= w:
            continue
        if not expect_vertical and w <= h:
            continue
        # Border gate: drop blobs running to the image edges parallel to the rod.
        if reject_border_axis:
            if expect_vertical and (x == 0 or x + w >= W):
                continue
            if not expect_vertical and (y == 0 or y + h >= H):
                continue
        elong = max(w, h) / max(1, min(w, h))
        score = elong * np.sqrt(area)
        if score > best_score:
            best_score, best = score, i
    if best == 0:
        return np.zeros_like(mask)
    return ((labels == best) * 255).astype(np.uint8)


def _pick_component_at_anchor(mask: np.ndarray, anchor: Tuple[float, float],
                              min_area: int = 20, max_dist: int = 50) -> np.ndarray:
    """Keep the connected component at the (fixed, clamped) rod-base anchor.

    Region-grow from a known base pixel: the rod is whatever foreground blob
    contains the anchor (or the nearest blob within max_dist if the anchor falls
    in a 1-px gap). Background edges, hand and window are SEPARATE components and
    are simply not connected to the base, so they vanish — robust to clutter that
    defeats elongation/orientation scoring. anchor = (x, y) in this view (px).
    """
    H, W = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return np.zeros_like(mask)
    ax, ay = int(round(anchor[0])), int(round(anchor[1]))
    lab = labels[ay, ax] if (0 <= ay < H and 0 <= ax < W) else 0
    if lab > 0 and stats[lab, cv2.CC_STAT_AREA] >= min_area:
        return ((labels == lab) * 255).astype(np.uint8)
    # anchor missed the mask — snap to the nearest foreground pixel within range
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros_like(mask)
    d2 = (xs - ax) ** 2 + (ys - ay) ** 2
    k = int(np.argmin(d2))
    if d2[k] > max_dist * max_dist:
        return np.zeros_like(mask)
    lab = labels[ys[k], xs[k]]
    if stats[lab, cv2.CC_STAT_AREA] < min_area:
        return np.zeros_like(mask)
    return ((labels == lab) * 255).astype(np.uint8)


def _select_component(mask: np.ndarray, params: dict) -> np.ndarray:
    """Pick the rod blob: anchor region-grow if a base anchor is set, else the
    orientation/elongation heuristic."""
    if params.get("_roi") is not None:
        mask = cv2.bitwise_and(mask, params["_roi"])
    anchor = params.get("_anchor")
    if anchor is not None:
        return _pick_component_at_anchor(mask, anchor,
                                         params.get("min_component_area", 50))
    expect_vertical = params.get("base_side", "bottom") in ("bottom", "top")
    return _pick_rod_component(mask, params.get("min_component_area", 50),
                               expect_vertical,
                               params.get("reject_border_axis", True))


def threshold_segmenter(gray: np.ndarray, params: dict) -> np.ndarray:
    """Inverse threshold (rod is dark) + morph open/close + most rod-like CC."""
    _, mask = cv2.threshold(gray, params["inv_threshold"], 255, cv2.THRESH_BINARY_INV)
    k = params["morph_kernel"]
    it = params["morph_iterations"]
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se, iterations=it)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se, iterations=it)
    return _select_component(mask, params)


def blackhat_segmenter(gray: np.ndarray, params: dict) -> np.ndarray:
    """Robust rod segmentation via black-hat morphology.

    Black-hat = closing(img) - img highlights thin DARK structures on a lighter
    background, independent of absolute brightness and slow illumination/vignette
    gradients. We then Otsu-threshold the response and keep the most elongated
    component. Far more robust than a global inverse threshold when rod contrast
    is low or the frame vignettes (as real D435 IR does).
    """
    ksz = params.get("blackhat_kernel", 21)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
    bh = cv2.GaussianBlur(bh, (3, 3), 0)
    _, mask = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = params["morph_kernel"]
    sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, sm, iterations=params["morph_iterations"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, sm, iterations=params["morph_iterations"])
    return _select_component(mask, params)


# Registry so the active segmenter is selectable from config (and a learned
# segmenter can be registered here later).
SEGMENTERS: dict[str, Segmenter] = {
    "threshold": threshold_segmenter,
    "blackhat": blackhat_segmenter,
}


def get_segmenter(params: dict) -> Segmenter:
    return SEGMENTERS[params.get("segmenter", "blackhat")]


# --------------------------------------------------------------------------- #
# 3. Skeleton ordering
# --------------------------------------------------------------------------- #
def _neighbors(skel: np.ndarray) -> np.ndarray:
    """Count of 8-neighbors that are also skeleton pixels."""
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(skel.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)


def order_skeleton(mask: np.ndarray, base_side: str,
                   anchor: Optional[Tuple[float, float]] = None) -> Optional[np.ndarray]:
    """Return ordered (x, y) skeleton path from base endpoint to tip.

    Endpoints have exactly one skeleton neighbor; if more than two exist we
    take the farthest-apart pair. The base endpoint is the one nearest `anchor`
    (the fixed clamp base, when provided), else chosen by `base_side`.
    """
    skel = skeletonize(mask > 0)
    pts = np.column_stack(np.nonzero(skel))  # (row, col)
    if pts.shape[0] < 5:
        return None

    nb = _neighbors(skel)
    ep = np.column_stack(np.nonzero((skel) & (nb == 1)))  # endpoints (row,col)
    if ep.shape[0] < 2:
        # closed/looped skeleton — fall back to the two farthest skeleton pts
        ep = pts
    # farthest-apart pair among endpoint candidates
    d = np.linalg.norm(ep[:, None, :] - ep[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    e0, e1 = ep[i], ep[j]

    # pick base: nearest endpoint to the clamp anchor if given, else by entry side
    if anchor is not None:
        a = np.array([anchor[1], anchor[0]])  # (row, col)
        base, tip = (e0, e1) if np.linalg.norm(e0 - a) <= np.linalg.norm(e1 - a) else (e1, e0)
    else:
        def side_score(p):  # p = (row, col)
            return {"bottom": p[0], "top": -p[0], "left": -p[1], "right": p[1]}[base_side]
        base, tip = (e0, e1) if side_score(e0) >= side_score(e1) else (e1, e0)

    # greedy nearest-neighbour walk from base to tip over skeleton pixels
    remaining = {tuple(p) for p in pts}
    path = [tuple(base)]
    remaining.discard(tuple(base))
    cur = tuple(base)
    while remaining:
        cr, cc = cur
        # search growing window for the closest remaining pixel
        best, bestd = None, None
        for p in remaining:
            dd = (p[0] - cr) ** 2 + (p[1] - cc) ** 2
            if bestd is None or dd < bestd:
                bestd, best = dd, p
        if bestd is None or bestd > 8:  # gap too large -> stop
            break
        path.append(best)
        remaining.discard(best)
        cur = best
    ordered = np.array(path, dtype=np.float64)
    # return as (x, y) = (col, row)
    return ordered[:, ::-1].copy()


# --------------------------------------------------------------------------- #
# 4. Spline fit, dense resample, subpixel normal refine
# --------------------------------------------------------------------------- #
def fit_resample_2d(path_xy: np.ndarray, smooth: float, n: int) -> np.ndarray:
    """Smoothing spline through ordered (x,y) px; dense uniform-param resample."""
    if path_xy.shape[0] < 4:
        return path_xy
    tck, _ = splprep([path_xy[:, 0], path_xy[:, 1]], s=smooth, k=3)
    u = np.linspace(0, 1, n)
    x, y = splev(u, tck)
    return np.column_stack([x, y])


def subpixel_refine(gray: np.ndarray, curve: np.ndarray, halfwidth: int) -> np.ndarray:
    """Move each sample to the intensity-weighted centroid across the rod normal.

    The rod is dark, so we weight by (255 - intensity). Tangents are estimated
    by finite differences; normals are perpendicular.
    """
    h, w = gray.shape
    tang = np.gradient(curve, axis=0)
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])
    offs = np.arange(-halfwidth, halfwidth + 1)
    out = curve.copy()
    for i, (p, nvec) in enumerate(zip(curve, normal)):
        samp = p[None, :] + offs[:, None] * nvec[None, :]
        xi = np.clip(samp[:, 0], 0, w - 1)
        yi = np.clip(samp[:, 1], 0, h - 1)
        vals = gray[np.round(yi).astype(int), np.round(xi).astype(int)].astype(np.float64)
        wgt = (255.0 - vals)
        if wgt.sum() < 1e-6:
            continue
        out[i, 0] = (xi * wgt).sum() / wgt.sum()
        out[i, 1] = (yi * wgt).sum() / wgt.sum()
    return out


# --------------------------------------------------------------------------- #
# 5. Epipolar correspondence on rectified scanlines
# --------------------------------------------------------------------------- #
def right_x_at_row(right_curve: np.ndarray, v: float) -> Tuple[Optional[float], float]:
    """Interpolate the right curve's x where it crosses row v.

    Returns (x, |dv/ds|) where dv/ds is the local row-change rate along the
    right curve (small => curve parallel to scanlines => DEGENERATE).
    """
    rv = right_curve[:, 1]
    best_x, best_slope = None, 0.0
    for k in range(len(right_curve) - 1):
        v0, v1 = rv[k], rv[k + 1]
        if (v0 - v) * (v1 - v) <= 0 and v0 != v1:
            t = (v - v0) / (v1 - v0)
            x = right_curve[k, 0] + t * (right_curve[k + 1, 0] - right_curve[k, 0])
            seg = right_curve[k + 1] - right_curve[k]
            slope = abs(seg[1]) / (np.linalg.norm(seg) + 1e-9)  # |dv/ds|
            if best_x is None or slope > best_slope:
                best_x, best_slope = x, slope
    return best_x, best_slope


def correspond(left_curve: np.ndarray, right_curve: np.ndarray,
               degenerate_dvds: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each left sample (row v), find matching right x at the same row.

    Returns left_pts (M,2), right_pts (M,2), valid (M,) bool. Matches that hit
    a near-horizontal piece of the right curve are flagged DEGENERATE (invalid).
    """
    L, R, valid = [], [], []
    for p in left_curve:
        rx, slope = right_x_at_row(right_curve, p[1])
        if rx is None:
            continue
        L.append(p)
        R.append([rx, p[1]])           # same row (rectified epipolar geometry)
        valid.append(slope >= degenerate_dvds)
    return np.array(L), np.array(R), np.array(valid, dtype=bool)


# --------------------------------------------------------------------------- #
# 6-7. Triangulate + 3D spline + uniform arclength resample
# --------------------------------------------------------------------------- #
def triangulate(P1, P2, left_pts, right_pts) -> np.ndarray:
    X = cv2.triangulatePoints(P1, P2, left_pts.T, right_pts.T)
    X /= X[3]
    return X[:3].T  # (M,3) mm


def robust_filter_3d(pts3d: np.ndarray, lp: dict) -> np.ndarray:
    """Drop outlier triangulated points before the 3D spline fit.

    Points arrive ordered base->tip (from the ordered left centerline). Two
    rejections: (1) depth gate — points outside [depth_min, depth_max] are
    near-degenerate stereo and unreliable; (2) step gate — a point whose jump
    from the running curve exceeds `max_seg_jump` x the median step is a bad
    correspondence (these inflate arc length even when depth looks fine).
    Returns the boolean inlier mask over the input order.
    """
    n = len(pts3d)
    keep = (pts3d[:, 2] >= lp["depth_min_mm"]) & (pts3d[:, 2] <= lp["depth_max_mm"])
    if keep.sum() < 4:
        return keep
    pts = pts3d[keep]
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    med = np.median(steps)
    if med <= 1e-9:
        return keep
    # a large step means the *following* point jumped; flag it as outlier
    good = np.ones(len(pts), dtype=bool)
    good[1:] = steps <= lp["max_seg_jump"] * med
    out = keep.copy()
    out[np.flatnonzero(keep)] = good
    return out


def fit_resample_3d(pts3d: np.ndarray, smooth: float, n: int
                    ) -> Tuple[np.ndarray, np.ndarray, float]:
    """3D smoothing spline; resample uniformly in arclength.

    Returns r_s (n,3), arclength samples (n,), total length L (mm).
    """
    tck, _ = splprep([pts3d[:, 0], pts3d[:, 1], pts3d[:, 2]], s=smooth, k=3)
    # dense sample to measure arclength
    ud = np.linspace(0, 1, 1000)
    dense = np.array(splev(ud, tck)).T
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    L = float(cum[-1])
    target = np.linspace(0, L, n)
    u_at = np.interp(target, cum, ud)
    r_s = np.array(splev(u_at, tck)).T
    return r_s, target, L


# --------------------------------------------------------------------------- #
# 8. QC reprojection
# --------------------------------------------------------------------------- #
def project(P: np.ndarray, pts3d: np.ndarray) -> np.ndarray:
    X = np.hstack([pts3d, np.ones((len(pts3d), 1))])
    x = (P @ X.T).T
    return x[:, :2] / x[:, 2:3]


def mean_reproj_error(proj_pts: np.ndarray, curve2d: np.ndarray) -> float:
    """Mean distance from each projected point to the nearest 2D-curve sample."""
    d = np.linalg.norm(proj_pts[:, None, :] - curve2d[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


# --------------------------------------------------------------------------- #
# Full per-pair label
# --------------------------------------------------------------------------- #
class LabelResult:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def compute_activity_roi(frames: list[np.ndarray], base_side: str,
                         dilate: int = 31) -> Optional[np.ndarray]:
    """Per-session ROI mask of the rod's swept region from temporal variation.

    With a clamped base the rod is the only DARK object that moves; the board
    edges, door frame, table lip and clamp are all static, so a temporal-std map
    over the session lights up the rod's swept fan and nothing static. We keep the
    activity blob anchored at the base side (rejects a moving hand/magnet, which
    enters from the far side) and dilate it to recover the near-pivot base, where
    the rod barely moves. The result gates segmentation so background edges that
    out-score the rod on elongation are excluded. Returns None if too few frames.
    """
    if len(frames) < 8:
        return None
    std = np.stack(frames).astype(np.float32).std(axis=0)
    s8 = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    H, W = m.shape
    # Anchor score: how far the blob reaches toward the base edge (the rod is
    # clamped there; a hand/magnet enters from the opposite side).
    def anchor(i):
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        return {"bottom": y + h, "top": H - y, "left": W - x, "right": x + w}[base_side]
    cand = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 50]
    if not cand:
        return None
    best = max(cand, key=anchor)
    roi = ((labels == best) * 255).astype(np.uint8)
    roi = cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    return roi


def label_pair(left_rect: np.ndarray, right_rect: np.ndarray,
               calib: StereoCalib, lp: dict,
               segmenter: Optional[Segmenter] = None,
               roi_l: Optional[np.ndarray] = None,
               roi_r: Optional[np.ndarray] = None,
               anchor_l: Optional[Tuple[float, float]] = None,
               anchor_r: Optional[Tuple[float, float]] = None) -> Optional[LabelResult]:
    if segmenter is None:
        segmenter = get_segmenter(lp)
    mask_l = segmenter(left_rect, {**lp, "_roi": roi_l, "_anchor": anchor_l})
    mask_r = segmenter(right_rect, {**lp, "_roi": roi_r, "_anchor": anchor_r})
    pl = order_skeleton(mask_l, lp["base_side"], anchor_l)
    pr = order_skeleton(mask_r, lp["base_side"], anchor_r)
    if pl is None or pr is None:
        return None

    cl = subpixel_refine(left_rect, fit_resample_2d(pl, lp["spline_smooth_2d"], lp["resample_2d"]),
                         lp["normal_halfwidth"])
    cr = subpixel_refine(right_rect, fit_resample_2d(pr, lp["spline_smooth_2d"], lp["resample_2d"]),
                         lp["normal_halfwidth"])

    L_pts, R_pts, valid = correspond(cl, cr, lp["degenerate_dvds"])
    if valid.sum() < 6:
        return None
    pts3d = triangulate(calib.P1, calib.P2, L_pts[valid], R_pts[valid])

    # robust filtering: drop too-close/too-far and bad-correspondence outliers
    inlier = robust_filter_3d(pts3d, lp)
    if inlier.sum() < max(6, int(lp["min_inlier_frac"] * len(pts3d))):
        return None
    pts3d = pts3d[inlier]

    r_s, s_samples, L_mm = fit_resample_3d(pts3d, lp["spline_smooth_3d"], lp["n_output"])

    # QC
    proj_l = project(calib.P1, r_s)
    proj_r = project(calib.P2, r_s)
    err_l = mean_reproj_error(proj_l, cl)
    err_r = mean_reproj_error(proj_r, cr)
    mean_depth = float(r_s[:, 2].mean())
    depth_ok = lp["depth_min_mm"] <= mean_depth <= lp["depth_max_mm"]
    # per-frame accept; session-level length-consistency is applied in process_session
    accept = depth_ok and (err_l < lp["reproj_thresh_px"]) and (err_r < lp["reproj_thresh_px"])

    return LabelResult(
        r_s=r_s, L_mm=L_mm, s_samples=s_samples, mean_depth_mm=mean_depth,
        reproj_err_left=err_l, reproj_err_right=err_r, accept=accept,
        curve_left=cl, curve_right=cr, proj_left=proj_l, proj_right=proj_r,
        n_valid=int(valid.sum()), n_corr=int(len(valid)),
    )


# --------------------------------------------------------------------------- #
# Overlay JPEG
# --------------------------------------------------------------------------- #
def save_overlay(path: Path, left_rect, right_rect, res: Optional[LabelResult]):
    vl = cv2.cvtColor(left_rect, cv2.COLOR_GRAY2BGR)
    vr = cv2.cvtColor(right_rect, cv2.COLOR_GRAY2BGR)
    if res is not None:
        for img, curve, proj in ((vl, res.curve_left, res.proj_left),
                                 (vr, res.curve_right, res.proj_right)):
            for p in curve:
                cv2.circle(img, (int(p[0]), int(p[1])), 1, (0, 200, 0), -1)
            for p in proj:
                cv2.circle(img, (int(p[0]), int(p[1])), 1, (0, 0, 255), -1)
        banner = (f"{'ACCEPT' if res.accept else 'REJECT'}  "
                  f"errL={res.reproj_err_left:.2f} errR={res.reproj_err_right:.2f}px "
                  f"L={res.L_mm:.1f}mm")
        color = (0, 180, 0) if res.accept else (0, 0, 255)
    else:
        banner, color = "REJECT (no centerline)", (0, 0, 255)
    vis = np.hstack([vl, vr])
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(vis, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])


# --------------------------------------------------------------------------- #
# Clamp-base anchor (one click per session; the base is fixed)
# --------------------------------------------------------------------------- #
def _anchor_path(data_root: Path, session: str) -> Path:
    return data_root / "raw" / session / "base_anchor.json"


def load_base_anchor(data_root: Path, session: str):
    """Return (anchor_left_xy, anchor_right_xy) in rectified px, or (None, None)."""
    p = _anchor_path(data_root, session)
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    return tuple(d["left"]), tuple(d["right"])


def set_base_anchor(session: str, cfg: dict, calib: StereoCalib, frame: int = 0) -> None:
    """Interactive: click the rod BASE in the rectified left then right view.

    Stored once per session to data/raw/<session>/base_anchor.json. The base is
    clamped (fixed), so one anchor serves every frame in the session.
    """
    data_root = Path(cfg["paths"]["data_root"])
    raw = data_root / "raw" / session
    files = sorted((raw / "left").glob("*.png"))
    if not files:
        raise RuntimeError(f"no frames in {raw/'left'}")
    f = files[min(frame, len(files) - 1)]
    left = calib.rectify_left(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE))
    right = calib.rectify_right(cv2.imread(str(raw / "right" / f.name), cv2.IMREAD_GRAYSCALE))

    picks = {}
    for name, img in (("left", left), ("right", right)):
        clicked = []
        disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        def on_mouse(ev, x, y, flags, _):
            if ev == cv2.EVENT_LBUTTONDOWN:
                clicked[:] = [(x, y)]
                v = disp.copy()
                cv2.drawMarker(v, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.imshow(win, v)

        win = f"click rod BASE ({name}) - ENTER=ok, r=redo, q=abort"
        cv2.imshow(win, disp)
        cv2.setMouseCallback(win, on_mouse)
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k in (13, 10) and clicked:      # ENTER
                break
            if k == ord("r"):
                clicked.clear(); cv2.imshow(win, disp)
            if k == ord("q"):
                cv2.destroyAllWindows(); raise SystemExit("aborted")
        cv2.destroyWindow(win)
        picks[name] = clicked[0]

    _anchor_path(data_root, session).write_text(
        json.dumps({"left": list(picks["left"]), "right": list(picks["right"])}))
    print(f"[{session}] base anchor saved: left={picks['left']} right={picks['right']} "
          f"-> {_anchor_path(data_root, session)}")


# --------------------------------------------------------------------------- #
# Session driver
# --------------------------------------------------------------------------- #
def process_session(session: str, cfg: dict, calib: StereoCalib) -> dict:
    data_root = Path(cfg["paths"]["data_root"])
    lp = cfg["labeler"]
    raw = data_root / "raw" / session
    left_dir = raw / "left"
    qc_dir = data_root / "qc" / session
    lab_dir = data_root / "labels" / session
    lab_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(left_dir.glob("*.png"))

    # Clamp-base anchor (set once per session via --set-base): region-grow the
    # rod from this fixed pixel, robust to background edges / hand / window.
    anchor_l, anchor_r = load_base_anchor(data_root, session)
    if anchor_l is not None:
        print(f"[{session}] using base anchor left={anchor_l} right={anchor_r}")
    else:
        print(f"[{session}] no base anchor (run: python labeler.py --session {session} "
              f"--set-base); using elongation heuristic")

    # Per-session activity ROI: temporal-std map isolates the rod's swept region
    # (only the rod moves; board/door/table/clamp are static), so background
    # edges that out-score the rod on elongation are gated out. Built from a
    # subsample of rectified frames in each view.
    roi_l = roi_r = None
    if lp.get("use_activity_roi", False):
        sample = files[:: max(1, len(files) // 80)]
        lefts, rights = [], []
        for f in sample:
            li = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            ri = cv2.imread(str(raw / "right" / f.name), cv2.IMREAD_GRAYSCALE)
            if li is None or ri is None:
                continue
            lefts.append(calib.rectify_left(li))
            rights.append(calib.rectify_right(ri))
        roi_l = compute_activity_roi(lefts, lp["base_side"])
        roi_r = compute_activity_roi(rights, lp["base_side"])
        if roi_l is None or roi_r is None:
            print(f"[{session}] WARNING: activity ROI empty in a view; "
                  f"falling back to ungated segmentation")
            roi_l = roi_r = None

    # Pass 1: label every frame (per-frame accept = depth + reproj gates).
    results: dict[str, Optional[LabelResult]] = {}
    for f in files:
        left = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(raw / "right" / f.name), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            continue
        results[f.stem] = label_pair(calib.rectify_left(left),
                                     calib.rectify_right(right), calib, lp,
                                     roi_l=roi_l, roi_r=roi_r,
                                     anchor_l=anchor_l, anchor_r=anchor_r)

    # Session length-consistency gate: the rod is rigid, so its triangulated
    # length should cluster. Reject per-frame-accepted frames whose L deviates
    # more than length_tol_frac from the median of the per-frame-accepted set.
    Ls = [r.L_mm for r in results.values() if r is not None and r.accept]
    med_L = float(np.median(Ls)) if Ls else 0.0
    tol = lp["length_tol_frac"]
    for r in results.values():
        if r is not None and r.accept and med_L > 0:
            if abs(r.L_mm - med_L) > tol * med_L:
                r.accept = False
                r.reject_reason = "length"

    # Pass 2: draw overlays (final accept) and save accepted labels.
    n_accept = 0
    for f in files:
        res = results.get(f.stem)
        left = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(raw / "right" / f.name), cv2.IMREAD_GRAYSCALE)
        lr, rr = calib.rectify_left(left), calib.rectify_right(right)
        save_overlay(qc_dir / f"{f.stem}.jpg", lr, rr, res)
        if res is not None and res.accept:
            np.savez(lab_dir / f"{f.stem}.npz",
                     r_s=res.r_s, L_mm=res.L_mm, s_samples=res.s_samples,
                     mean_depth_mm=res.mean_depth_mm,
                     reproj_err_left=res.reproj_err_left,
                     reproj_err_right=res.reproj_err_right,
                     left_image=str(f))
            n_accept += 1

    stats = {"session": session, "n_frames": len(files), "n_accept": n_accept,
             "accept_rate": n_accept / max(1, len(files)), "median_L_mm": med_L}
    print(f"[{session}] frames={stats['n_frames']} accepted={n_accept} "
          f"({100*stats['accept_rate']:.1f}%)  median L={med_L:.1f}mm  "
          f"labels -> {lab_dir}  qc -> {qc_dir}")
    return stats


# --------------------------------------------------------------------------- #
# Synthetic test: known 3D curve projected through P1/P2 + noise
# --------------------------------------------------------------------------- #
def _synthetic_test() -> None:
    cfg = load_config()
    calib = nominal_calib()  # file-free nominal D435 stereo calib for the self-test
    lp = cfg["labeler"]
    w, h = calib.image_size
    rng = np.random.default_rng(0)

    # A helix segment ~250 mm in front of the camera, ~60 mm tall.
    t = np.linspace(0, 1, 120)
    X = np.column_stack([
        20.0 * np.cos(2 * t) - 10.0,
        60.0 * t - 30.0,
        250.0 + 20.0 * np.sin(2 * t),
    ])
    proj_l = project(calib.P1, X)
    proj_r = project(calib.P2, X)

    def render(proj):
        img = np.full((h, w), 220, np.uint8)
        for k in range(len(proj) - 1):
            p0 = tuple(np.round(proj[k]).astype(int))
            p1 = tuple(np.round(proj[k + 1]).astype(int))
            cv2.line(img, p0, p1, 25, 2, cv2.LINE_AA)
        noise = (rng.standard_normal((h, w)) * 1).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    left_img, right_img = render(proj_l), render(proj_r)
    res = label_pair(left_img, right_img, calib, lp)
    assert res is not None, "labeler produced no result on synthetic pair"

    # Compare reconstruction against ground-truth curve: nearest-point mean dist.
    # Densify GT first so the nearest-point metric isn't limited by GT spacing.
    tg = np.linspace(0, 1, 4000)
    Xd = np.column_stack([20.0 * np.cos(2 * tg) - 10.0,
                          60.0 * tg - 30.0,
                          250.0 + 20.0 * np.sin(2 * tg)])
    d = np.linalg.norm(res.r_s[:, None, :] - Xd[None, :, :], axis=2).min(axis=1)
    mean_err = float(d.mean())
    print(f"synthetic reconstruction mean error = {mean_err:.3f} mm "
          f"(L={res.L_mm:.1f} mm, reproj L/R = "
          f"{res.reproj_err_left:.2f}/{res.reproj_err_right:.2f} px, accept={res.accept})")
    assert mean_err < 0.5, f"reconstruction error {mean_err:.3f} mm >= 0.5 mm"
    print("PASS: synthetic reconstruction error < 0.5 mm")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stereo 3D centerline labeler")
    ap.add_argument("--session", help="process data/raw/<session>")
    ap.add_argument("--test", action="store_true", help="run synthetic self-test")
    ap.add_argument("--set-base", action="store_true",
                    help="click the rod base once for --session (clamped base anchor)")
    ap.add_argument("--base-frame", type=int, default=0,
                    help="which frame index to show when setting the base anchor")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.test or not args.session:
        _synthetic_test()
    elif args.set_base:
        calib = resolve_calib(cfg)
        set_base_anchor(args.session, cfg, calib, args.base_frame)
    else:
        calib = resolve_calib(cfg)
        process_session(args.session, cfg, calib)


if __name__ == "__main__":
    main()
