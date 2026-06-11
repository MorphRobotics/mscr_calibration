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
def threshold_segmenter(gray: np.ndarray, params: dict) -> np.ndarray:
    """Inverse threshold (rod is dark) + morph open/close + largest CC."""
    _, mask = cv2.threshold(gray, params["inv_threshold"], 255, cv2.THRESH_BINARY_INV)
    k = params["morph_kernel"]
    it = params["morph_iterations"]
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se, iterations=it)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se, iterations=it)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return np.zeros_like(mask)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == largest) * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 3. Skeleton ordering
# --------------------------------------------------------------------------- #
def _neighbors(skel: np.ndarray) -> np.ndarray:
    """Count of 8-neighbors that are also skeleton pixels."""
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(skel.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)


def order_skeleton(mask: np.ndarray, base_side: str) -> Optional[np.ndarray]:
    """Return ordered (x, y) skeleton path from base endpoint to tip.

    Endpoints have exactly one skeleton neighbor; if more than two exist we
    take the farthest-apart pair. The base is chosen by `base_side`.
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

    # pick base by entry side
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


def label_pair(left_rect: np.ndarray, right_rect: np.ndarray,
               calib: StereoCalib, lp: dict,
               segmenter: Segmenter = threshold_segmenter) -> Optional[LabelResult]:
    mask_l = segmenter(left_rect, lp)
    mask_r = segmenter(right_rect, lp)
    pl = order_skeleton(mask_l, lp["base_side"])
    pr = order_skeleton(mask_r, lp["base_side"])
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

    r_s, s_samples, L_mm = fit_resample_3d(pts3d, lp["spline_smooth_3d"], lp["n_output"])

    # QC
    proj_l = project(calib.P1, r_s)
    proj_r = project(calib.P2, r_s)
    err_l = mean_reproj_error(proj_l, cl)
    err_r = mean_reproj_error(proj_r, cr)
    accept = (err_l < lp["reproj_thresh_px"]) and (err_r < lp["reproj_thresh_px"])

    return LabelResult(
        r_s=r_s, L_mm=L_mm, s_samples=s_samples,
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
    n_accept = 0
    for f in files:
        left = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(raw / "right" / f.name), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            continue
        lr = calib.rectify_left(left)
        rr = calib.rectify_right(right)
        res = label_pair(lr, rr, calib, lp)
        save_overlay(qc_dir / f"{f.stem}.jpg", lr, rr, res)
        if res is not None and res.accept:
            np.savez(lab_dir / f"{f.stem}.npz",
                     r_s=res.r_s, L_mm=res.L_mm, s_samples=res.s_samples,
                     reproj_err_left=res.reproj_err_left,
                     reproj_err_right=res.reproj_err_right,
                     left_image=str(f))
            n_accept += 1

    stats = {"session": session, "n_frames": len(files), "n_accept": n_accept,
             "accept_rate": n_accept / max(1, len(files))}
    print(f"[{session}] frames={stats['n_frames']} accepted={n_accept} "
          f"({100*stats['accept_rate']:.1f}%)  labels -> {lab_dir}  qc -> {qc_dir}")
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
        noise = (rng.standard_normal((h, w)) * 2).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    left_img, right_img = render(proj_l), render(proj_r)
    res = label_pair(left_img, right_img, calib, lp)
    assert res is not None, "labeler produced no result on synthetic pair"

    # Compare reconstruction against ground-truth curve: nearest-point mean dist
    d = np.linalg.norm(res.r_s[:, None, :] - X[None, :, :], axis=2).min(axis=1)
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
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.test or not args.session:
        _synthetic_test()
    else:
        calib = resolve_calib(cfg)
        process_session(args.session, cfg, calib)


if __name__ == "__main__":
    main()
