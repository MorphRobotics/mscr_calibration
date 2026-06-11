#!/usr/bin/env python3
"""
MSCR Stereo Camera Calibration Pipeline

Processes matched left/right IR pairs from capture.py and produces:
  results/stereo_calibration.yaml   — load in control code
  results/calibration_report.pdf   — full document with graphs
  results/figures/                  — individual PNG figures
"""

import os, sys, glob, json
from datetime import datetime

import cv2
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D        # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak, HRFlowable,
)

from config import (BOARD_COLS, BOARD_ROWS, SQUARE_SIZE_MM,
                    LEFT_DIR, RIGHT_DIR, RESULTS_DIR, MIN_CALIB_IMAGES,
                    CAMERA_WIDTH, CAMERA_HEIGHT)

FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
DPI = 150

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Load matched pairs and detect corners
# ─────────────────────────────────────────────────────────────────────────────

def detect_stereo_corners(left_dir, right_dir):
    """
    Find images present in both left/ and right/ with matching filenames.
    Detect checkerboard corners in each pair; keep only pairs where both succeed.

    Returns: obj_pts, img_pts_L, img_pts_R, valid_stems, image_size
    """
    pattern  = (BOARD_COLS, BOARD_ROWS)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    objp[:, :2] = (np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS]
                   .T.reshape(-1, 2) * SQUARE_SIZE_MM)

    left_stems  = {os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(left_dir,  "*.png"))}
    right_stems = {os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(right_dir, "*.png"))}
    common = sorted(left_stems & right_stems)

    if not common:
        return [], [], [], [], (CAMERA_WIDTH, CAMERA_HEIGHT)

    print(f"\nMatched pairs found: {len(common)}  "
          f"(left-only: {len(left_stems-right_stems)}  "
          f"right-only: {len(right_stems-left_stems)})")

    obj_pts, img_pts_L, img_pts_R, valid_stems = [], [], [], []
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    skip  = 0

    for stem in common:
        lp = os.path.join(left_dir,  f"{stem}.png")
        rp = os.path.join(right_dir, f"{stem}.png")
        gl = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        gr = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        if gl is None or gr is None:
            skip += 1; continue

        fl, cl = cv2.findChessboardCorners(gl, pattern, flags=flags)
        fr, cr = cv2.findChessboardCorners(gr, pattern, flags=flags)

        if not (fl and fr):
            skip += 1
            continue

        cv2.cornerSubPix(gl, cl, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(gr, cr, (11, 11), (-1, -1), criteria)

        obj_pts.append(objp)
        img_pts_L.append(cl)
        img_pts_R.append(cr)
        valid_stems.append(stem)

    print(f"  Corner detection:  accepted={len(valid_stems)}  skipped={skip}")

    sample = cv2.imread(os.path.join(left_dir, f"{valid_stems[0]}.png"),
                        cv2.IMREAD_GRAYSCALE) if valid_stems else None
    image_size = (sample.shape[1], sample.shape[0]) if sample is not None \
        else (CAMERA_WIDTH, CAMERA_HEIGHT)

    if len(valid_stems) < MIN_CALIB_IMAGES:
        print(f"  WARNING: only {len(valid_stems)} valid pairs — "
              f"recommend ≥ {MIN_CALIB_IMAGES}.")
    return obj_pts, img_pts_L, img_pts_R, valid_stems, image_size


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Iterative outlier rejection (stereo-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _per_pair_rms(obj_pts, img_pts_L, img_pts_R, image_size):
    """Calibrate each camera independently, return per-pair max(RMS_L, RMS_R)."""
    _, K_L, D_L, rvL, tvL = cv2.calibrateCamera(obj_pts, img_pts_L, image_size, None, None)
    _, K_R, D_R, rvR, tvR = cv2.calibrateCamera(obj_pts, img_pts_R, image_size, None, None)
    per = []
    for op, ipL, rvl, tvl, ipR, rvr, tvr in zip(
            obj_pts, img_pts_L, rvL, tvL, img_pts_R, rvR, tvR):
        pL, _ = cv2.projectPoints(op, rvl, tvl, K_L, D_L)
        pR, _ = cv2.projectPoints(op, rvr, tvr, K_R, D_R)
        eL = float(np.sqrt(np.mean((ipL.reshape(-1,2) - pL.reshape(-1,2))**2)))
        eR = float(np.sqrt(np.mean((ipR.reshape(-1,2) - pR.reshape(-1,2))**2)))
        per.append(max(eL, eR))
    return per


def iterative_refine(obj_pts, img_pts_L, img_pts_R, valid_stems, image_size,
                     max_iter=8, sigma=2.0):
    keep_obj, keep_L, keep_R, keep_stems = (list(obj_pts), list(img_pts_L),
                                            list(img_pts_R), list(valid_stems))
    removed = 0
    print("\nIterative outlier rejection …")

    for it in range(max_iter):
        per = _per_pair_rms(keep_obj, keep_L, keep_R, image_size)
        mean_e, std_e = np.mean(per), np.std(per)
        thresh = mean_e + sigma * std_e
        bad = {i for i, e in enumerate(per) if e > thresh}
        if not bad:
            print(f"  Round {it+1}: RMS={np.sqrt(np.mean(np.array(per)**2)):.4f} px"
                  f" — no outliers  ({len(keep_stems)} pairs kept)")
            break
        keep_obj   = [x for i,x in enumerate(keep_obj)   if i not in bad]
        keep_L     = [x for i,x in enumerate(keep_L)     if i not in bad]
        keep_R     = [x for i,x in enumerate(keep_R)     if i not in bad]
        keep_stems = [x for i,x in enumerate(keep_stems) if i not in bad]
        removed   += len(bad)
        print(f"  Round {it+1}: removed {len(bad)} pair(s)"
              f"  thresh={thresh:.3f} px  remaining={len(keep_stems)}")

    print(f"  Total removed: {removed}")
    return keep_obj, keep_L, keep_R, keep_stems


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Stereo calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_individual(obj_pts, img_pts_L, img_pts_R, image_size):
    """Monocular calibration for each camera — produces initial estimates."""
    print("\nMonocular calibration (initial estimates) …")
    rms_L, K_L, D_L, _, _ = cv2.calibrateCamera(obj_pts, img_pts_L, image_size, None, None)
    rms_R, K_R, D_R, _, _ = cv2.calibrateCamera(obj_pts, img_pts_R, image_size, None, None)
    print(f"  Left  RMS: {rms_L:.4f} px")
    print(f"  Right RMS: {rms_R:.4f} px")
    return K_L, D_L, K_R, D_R


def stereo_calibrate(obj_pts, img_pts_L, img_pts_R, K_L, D_L, K_R, D_R, image_size):
    """Joint stereo calibration refining intrinsics and estimating R, T."""
    print("\nStereo calibration …")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
    flags    = cv2.CALIB_USE_INTRINSIC_GUESS

    rms, K_L, D_L, K_R, D_R, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_pts_L, img_pts_R,
        K_L, D_L, K_R, D_R,
        image_size, criteria=criteria, flags=flags)

    D_L, D_R = D_L.flatten(), D_R.flatten()
    baseline  = float(np.linalg.norm(T))
    print(f"  Stereo RMS: {rms:.4f} px")
    print(f"  Baseline  : {baseline:.3f} mm")
    print(f"  T (mm)    : {T.flatten()}")
    return K_L, D_L, K_R, D_R, R, T.flatten(), E, F, rms, baseline


def stereo_rectify(K_L, D_L, K_R, D_R, R, T, image_size):
    R1, R2, P1, P2, Q, roi_L, roi_R = cv2.stereoRectify(
        K_L, D_L, K_R, D_R, image_size, R, T, alpha=0)
    return R1, R2, P1, P2, Q, roi_L, roi_R


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Error analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_errors(K_L, D_L, K_R, D_R, R, T, obj_pts, img_pts_L, img_pts_R, F):
    """
    Per-pair reprojection errors for each camera + symmetric epipolar error.
    Per-pair right-camera pose is derived from left pose via R, T.
    """
    per_rms_L, per_rms_R, all_err_L, all_err_R = [], [], [], []

    # Solve PnP for each pair using the left camera to get per-image poses
    for op, ipL, ipR in zip(obj_pts, img_pts_L, img_pts_R):
        ok, rvL, tvL = cv2.solvePnP(op, ipL, K_L, D_L)
        if not ok:
            per_rms_L.append(np.nan); per_rms_R.append(np.nan)
            continue

        # Left reprojection
        pL, _ = cv2.projectPoints(op, rvL, tvL, K_L, D_L)
        dL = ipL.reshape(-1,2) - pL.reshape(-1,2)
        all_err_L.append(dL)
        per_rms_L.append(float(np.sqrt(np.mean(dL**2))))

        # Right pose: R_right = R @ R_left,  T_right = R @ T_left + T
        R_L, _ = cv2.Rodrigues(rvL)
        R_R    = R @ R_L
        T_R    = (R @ tvL).flatten() + T
        rvR, _ = cv2.Rodrigues(R_R)

        pR, _ = cv2.projectPoints(op, rvR, T_R.reshape(3,1), K_R, D_R)
        dR = ipR.reshape(-1,2) - pR.reshape(-1,2)
        all_err_R.append(dR)
        per_rms_R.append(float(np.sqrt(np.mean(dR**2))))

    all_err_L = np.vstack(all_err_L)
    all_err_R = np.vstack(all_err_R)

    # Symmetric epipolar error using fundamental matrix F
    epi_errors = []
    for ipL, ipR in zip(img_pts_L, img_pts_R):
        pL = ipL.reshape(-1, 2)
        pR = ipR.reshape(-1, 2)
        n  = len(pL)
        hL = np.hstack([pL, np.ones((n,1))])   # (N,3)
        hR = np.hstack([pR, np.ones((n,1))])
        lR = (F @ hL.T).T                       # epilines in right
        lL = (F.T @ hR.T).T                     # epilines in left
        dR = np.abs(np.sum(hR*lR, axis=1)) / np.sqrt(lR[:,0]**2 + lR[:,1]**2)
        dL = np.abs(np.sum(hL*lL, axis=1)) / np.sqrt(lL[:,0]**2 + lL[:,1]**2)
        epi_errors.extend(((dL + dR) / 2).tolist())

    epi_errors = np.array(epi_errors)
    print(f"\n  Left  per-image RMS  mean={np.nanmean(per_rms_L):.4f}  "
          f"max={np.nanmax(per_rms_L):.4f}")
    print(f"  Right per-image RMS  mean={np.nanmean(per_rms_R):.4f}  "
          f"max={np.nanmax(per_rms_R):.4f}")
    print(f"  Epipolar error       mean={epi_errors.mean():.4f}  "
          f"max={epi_errors.max():.4f} px")

    return per_rms_L, per_rms_R, all_err_L, all_err_R, epi_errors


# ─────────────────────────────────────────────────────────────────────────────
# Plot style
# ─────────────────────────────────────────────────────────────────────────────

C_LEFT  = "#1a6eb5"
C_RIGHT = "#c0392b"
C_WARN  = "#e67e22"
C_HDR   = "#2c5f8a"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "grid.color":        "#aaaaaa",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "legend.fontsize":   8,
    "legend.framealpha": 0.85,
    "lines.linewidth":   1.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Figures
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, name, tight=True):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    kwargs = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(path, dpi=DPI, **kwargs)
    plt.close(fig)
    return path


def fig_reprojection(per_rms_L, per_rms_R, all_err_L, all_err_R, epi_errors):
    from matplotlib.patches import Ellipse
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                             gridspec_kw={"wspace": 0.42})
    fig.suptitle("Stereo Reprojection Error Analysis", fontsize=11, fontweight="bold", y=1.01)

    ax = axes[0]
    n   = len(per_rms_L)
    idx = np.argsort([(l + r) / 2 for l, r in zip(per_rms_L, per_rms_R)])
    x   = np.arange(n)
    bL  = ax.bar(x - 0.2, [per_rms_L[i] for i in idx], 0.38,
                 label="Left",  color=C_LEFT,  alpha=0.85, linewidth=0)
    bR  = ax.bar(x + 0.2, [per_rms_R[i] for i in idx], 0.38,
                 label="Right", color=C_RIGHT, alpha=0.85, linewidth=0)
    for bars, vals in [(bL, per_rms_L), (bR, per_rms_R)]:
        for bar, oi in zip(bars, idx):
            if vals[oi] > 1.0:
                bar.set_facecolor(C_WARN)
    ax.axhline(0.5, color="green", ls=":",  lw=1.2, label="0.5 px target")
    ax.axhline(1.0, color=C_WARN,  ls=":",  lw=1.2, label="1.0 px limit")
    ax.axhline(np.nanmean(per_rms_L), color=C_LEFT,  ls="--", lw=1, alpha=0.6)
    ax.axhline(np.nanmean(per_rms_R), color=C_RIGHT, ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Image pair (sorted by mean error)")
    ax.set_ylabel("RMS reprojection error (px)")
    ax.set_title("Per-Pair RMS")
    ax.set_xticks([])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.text(0.99, 0.97,
            f"n = {n}\nL: {np.nanmean(per_rms_L):.3f} px\nR: {np.nanmean(per_rms_R):.3f} px",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))

    for ax, errs, label, col in [
            (axes[1], all_err_L, "Left",  C_LEFT),
            (axes[2], all_err_R, "Right", C_RIGHT)]:
        ax.scatter(errs[:, 0], errs[:, 1], s=0.8, alpha=0.2, color=col, linewidths=0)
        cov    = np.cov(errs.T)
        evals, evecs = np.linalg.eigh(cov)
        angle  = np.degrees(np.arctan2(evecs[1, -1], evecs[0, -1]))
        for ns, a in [(1, 0.85), (2, 0.4)]:
            ax.add_patch(Ellipse((0, 0),
                width=2 * ns * np.sqrt(evals[-1]), height=2 * ns * np.sqrt(evals[0]),
                angle=angle, edgecolor="tomato", facecolor="none",
                linewidth=1.4, alpha=a, zorder=3))
        ax.axhline(0, color="#888888", lw=0.6, zorder=2)
        ax.axvline(0, color="#888888", lw=0.6, zorder=2)
        lim = max(float(np.abs(errs).max()) * 1.1, 1.0)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("dx (px)"); ax.set_ylabel("dy (px)")
        ax.set_title(f"{label} Camera Error Vectors")
        rms_val = float(np.sqrt(np.mean(errs ** 2)))
        ax.text(0.03, 0.97, f"RMS = {rms_val:.4f} px\nn = {len(errs)}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))

    fig.tight_layout()
    return _save(fig, "fig1_reprojection.png")


def fig_rectification(K_L, D_L, K_R, D_R, R1, R2, P1, P2, image_size, valid_stems):
    stem = valid_stems[len(valid_stems) // 2]
    gl = cv2.imread(os.path.join(LEFT_DIR,  f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    gr = cv2.imread(os.path.join(RIGHT_DIR, f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    if gl is None or gr is None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Image pair not found", ha="center",
                transform=ax.transAxes, fontsize=10)
        ax.axis("off")
        return _save(fig, "fig2_rectification.png")

    w, h = image_size
    mL1, mL2 = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, (w, h), cv2.CV_32FC1)
    mR1, mR2 = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, (w, h), cv2.CV_32FC1)
    rL = cv2.remap(gl, mL1, mL2, cv2.INTER_LINEAR)
    rR = cv2.remap(gr, mR1, mR2, cv2.INTER_LINEAR)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gl_d = clahe.apply(gl); gr_d = clahe.apply(gr)
    rL_d = clahe.apply(rL); rR_d = clahe.apply(rR)

    rL_c = cv2.cvtColor(rL_d, cv2.COLOR_GRAY2BGR)
    rR_c = cv2.cvtColor(rR_d, cv2.COLOR_GRAY2BGR)
    for y in range(0, h, 45):
        cv2.line(rL_c, (0, y), (w, y), (0, 180, 0), 1, cv2.LINE_AA)
        cv2.line(rR_c, (0, y), (w, y), (0, 180, 0), 1, cv2.LINE_AA)

    fig, axes = plt.subplots(2, 2, figsize=(14, 6.5),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.04})
    fig.suptitle("Stereo Rectification Verification", fontsize=11, fontweight="bold")
    panels = [
        (gl_d,  "Original - Left IR"),
        (gr_d,  "Original - Right IR"),
        (cv2.cvtColor(rL_c, cv2.COLOR_BGR2RGB), "Rectified - Left (epipolar lines)"),
        (cv2.cvtColor(rR_c, cv2.COLOR_BGR2RGB), "Rectified - Right (epipolar lines)"),
    ]
    for ax, (im, title) in zip(axes.flat, panels):
        cmap = "gray" if im.ndim == 2 else None
        ax.imshow(im, cmap=cmap, interpolation="bilinear")
        ax.set_title(title, fontsize=9, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.text(0.5, 0.005,
             "Corresponding features must lie on the same horizontal scanline in the "
             "rectified views. Vertical offset indicates residual calibration error.",
             ha="center", fontsize=8, color="#555555", style="italic")
    fig.subplots_adjust(hspace=0.08, wspace=0.04, top=0.93, bottom=0.04,
                        left=0.02, right=0.98)
    return _save(fig, "fig2_rectification.png")


def fig_distortion(K_L, D_L, K_R, D_R, image_size):
    w, h = image_size

    def dist_magnitude(K, D):
        # Keep original K as destination so we measure pure lens distortion;
        # getOptimalNewCameraMatrix(alpha=0) crops and introduces large edge
        # artifacts that dominate the colour scale.
        m1, m2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
        yy, xx = np.mgrid[0:h, 0:w]
        return np.sqrt((m1 - xx) ** 2 + (m2 - yy) ** 2), m1, m2

    mag_L, mL1, mL2 = dist_magnitude(K_L, D_L)
    mag_R, mR1, mR2 = dist_magnitude(K_R, D_R)
    vmax  = max(mag_L.max(), mag_R.max())
    step  = max(w // 28, 1)
    ys    = np.arange(step // 2, h, step)
    xs    = np.arange(step // 2, w, step)
    XX, YY = np.meshgrid(xs, ys)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"wspace": 0.38})
    fig.suptitle("Lens Distortion Magnitude", fontsize=11, fontweight="bold")

    for ax, mag, m1, m2, label in [
            (axes[0], mag_L, mL1, mL2, "Left IR"),
            (axes[1], mag_R, mR1, mR2, "Right IR")]:
        ax.grid(False)
        im = ax.imshow(mag, cmap="plasma", vmin=0, vmax=vmax, origin="upper", aspect="equal")
        dxq = m1[YY, XX] - XX
        dyq = m2[YY, XX] - YY
        ax.quiver(XX, YY, dxq, dyq, color="white", alpha=0.5,
                  scale=None, scale_units="xy", angles="xy",
                  width=0.003, headwidth=3, headlength=4)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Displacement (px)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
        ax.tick_params(labelsize=7)
        ax.text(0.02, 0.97, f"max = {mag.max():.2f} px\nmean = {mag.mean():.2f} px",
                transform=ax.transAxes, va="top", fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc="#00000066", ec="none"))
        ax.grid(False)

    return _save(fig, "fig3_distortion.png")


def fig_pose_coverage(K_L, D_L, obj_pts, img_pts_L, image_size):
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    c_idx = [0, BOARD_COLS - 1,
             BOARD_COLS * (BOARD_ROWS - 1), BOARD_COLS * BOARD_ROWS - 1]
    corners_board = obj_pts[0][c_idx]

    ctrs     = []
    outlines = []
    for op, ipL in zip(obj_pts, img_pts_L):
        ok, rvL, tvL = cv2.solvePnP(op, ipL, K_L, D_L)
        if not ok:
            continue
        R_L, _ = cv2.Rodrigues(rvL)
        t_L    = tvL.flatten()
        pts    = (R_L @ corners_board.T).T + t_L
        # Closed loop: TL → TR → BR → BL → TL
        outlines.append(pts[[0, 1, 3, 2, 0]])
        ctrs.append((R_L @ op.mean(axis=0)) + t_L)

    ctrs    = np.array(ctrs)
    z_vals  = ctrs[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max() + 1e-6
    cmap_fn = plt.cm.get_cmap("coolwarm")

    # Draw each board as 4 line segments — avoids Poly3DCollection crash on mpl 3.5
    for outline, ctr in zip(outlines, ctrs):
        t   = float((ctr[2] - z_min) / (z_max - z_min))
        col = cmap_fn(t)
        ax.plot3D(outline[:, 0], outline[:, 1], outline[:, 2],
                  color=col, alpha=0.55, linewidth=0.9)

    # Board centres coloured by depth
    sc = ax.scatter(ctrs[:, 0], ctrs[:, 1], ctrs[:, 2],
                    c=z_vals, cmap="coolwarm", s=22, alpha=0.9,
                    depthshade=False, zorder=5)

    ax.scatter(0, 0, 0, s=140, marker="^", color=C_RIGHT,
               zorder=10, depthshade=False, label="Camera origin")

    ax.set_xlabel("X (mm)", labelpad=14, fontsize=9, fontweight="bold")
    ax.set_ylabel("Y (mm)", labelpad=14, fontsize=9, fontweight="bold")
    ax.set_zlabel("Z (mm)", labelpad=14, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7, pad=2)
    ax.set_title("Board Pose Coverage — Left Camera Frame",
                 fontsize=10, fontweight="bold", pad=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.view_init(elev=18, azim=-50)

    # Manual margins — tight_layout clips 3D labels; bbox_inches="tight" must be off
    fig.subplots_adjust(left=0.0, right=0.82, top=0.93, bottom=0.05)
    cbar_ax = fig.add_axes([0.85, 0.22, 0.025, 0.55])
    ax.grid(False)
    cb = fig.colorbar(sc, cax=cbar_ax)
    cb.set_label("Z depth (mm)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    return _save(fig, "fig4_pose_coverage.png", tight=False)


def fig_depth_resolution(K_L, T, baseline):
    fx   = float(K_L[0, 0])
    B    = baseline
    deps = np.linspace(50, 600, 800)
    disp = fx * B / deps
    res  = deps ** 2 / (fx * B)

    fig, (ax_chart, ax_tbl) = plt.subplots(1, 2, figsize=(14, 5),
                                            gridspec_kw={"wspace": 0.32})
    fig.suptitle(
        f"Stereo Depth Resolution  (baseline = {B:.2f} mm, fx = {fx:.1f} px)",
        fontsize=11, fontweight="bold")

    ln1, = ax_chart.plot(deps, disp, color=C_LEFT, label="Disparity (px)")
    ax_chart.set_xlabel("Working distance (mm)")
    ax_chart.set_ylabel("Expected disparity (px)", color=C_LEFT)
    ax_chart.tick_params(axis="y", labelcolor=C_LEFT)
    ax_chart.spines["left"].set_color(C_LEFT)
    ax_chart.set_xlim(50, 600); ax_chart.set_ylim(bottom=0)

    ax2 = ax_chart.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(C_RIGHT)
    ax2.spines["top"].set_visible(False)
    ln2, = ax2.plot(deps, res, color=C_RIGHT, ls="--", label="Depth res. (mm/px)")
    ax2.set_ylabel("Depth res. (mm/px)", color=C_RIGHT, labelpad=8)
    ax2.tick_params(axis="y", labelcolor=C_RIGHT)
    ax2.set_ylim(bottom=0)

    ax_chart.set_title("Disparity and Depth Resolution vs Distance", fontsize=10)
    ax_chart.legend([ln1, ln2], [ln1.get_label(), ln2.get_label()],
                    loc="upper right", fontsize=8)
    ax_chart.grid(True, alpha=0.3, linestyle="--", color="#aaaaaa")

    ax_tbl.axis("off")
    ref     = [50, 100, 150, 200, 250, 300, 400, 500]
    headers = ["Depth\n(mm)", "Disp.\n(px)", "Res.\n(mm/px)", "px /\n1 mm"]
    rows    = []
    for d in ref:
        dp = fx * B / d
        rp = d ** 2 / (fx * B)
        rows.append([f"{d}", f"{dp:.1f}", f"{rp:.4f}", f"{1/rp:.1f}"])
    tbl = ax_tbl.table(cellText=rows, colLabels=headers,
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.3, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor(C_HDR)
            cell.set_text_props(color="white", fontweight="bold", fontsize=8)
        elif r % 2 == 0:
            cell.set_facecolor("#eef3f8")
        else:
            cell.set_facecolor("white")
    ax_tbl.set_title("Depth Reference Table", fontsize=10, fontweight="bold", pad=14)
    fig.tight_layout()
    return _save(fig, "fig5_depth_resolution.png")



# Phase 6 — Save YAML
# ─────────────────────────────────────────────────────────────────────────────

def save_yaml(K_L, D_L, K_R, D_R, R, T, E, F_mat, R1, R2, P1, P2, Q,
              rms, baseline, valid_stems, image_size):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    w, h  = image_size
    hfov  = float(np.degrees(2 * np.arctan(w / (2*K_L[0,0]))))
    vfov  = float(np.degrees(2 * np.arctan(h / (2*K_L[1,1]))))

    fw_path = os.path.join(RESULTS_DIR, "rs_firmware_intrinsics.json")
    depth_scale = None
    if os.path.exists(fw_path):
        with open(fw_path) as f:
            depth_scale = json.load(f).get("depth_scale_m_per_unit")

    def mat(arr):
        a = np.array(arr)
        return {"rows": int(a.shape[0]), "cols": int(a.shape[1] if a.ndim>1 else 1),
                "data": a.flatten().tolist()}

    data = {
        "calibration_date":     datetime.now().isoformat(timespec="seconds"),
        "type":                 "stereo_ir",
        "board": {
            "cols_squares": BOARD_COLS + 1, "rows_squares": BOARD_ROWS + 1,
            "inner_corners": [BOARD_COLS, BOARD_ROWS],
            "square_size_mm": float(SQUARE_SIZE_MM),
        },
        "image_size":           {"width": int(w), "height": int(h)},
        "left_camera": {
            "camera_matrix": mat(K_L),
            "dist_coeffs":   {"description": "k1 k2 p1 p2 k3", **mat(D_L)},
        },
        "right_camera": {
            "camera_matrix": mat(K_R),
            "dist_coeffs":   {"description": "k1 k2 p1 p2 k3", **mat(D_R)},
        },
        "stereo": {
            "R":             mat(R),
            "T_mm":          T.tolist(),
            "E":             mat(E),
            "F":             mat(F_mat),
            "baseline_mm":   round(float(baseline), 6),
        },
        "rectification": {
            "R1": mat(R1), "R2": mat(R2),
            "P1": mat(P1), "P2": mat(P2),
            "Q":  mat(Q),
        },
        "rms_stereo_px":        round(float(rms), 6),
        "num_pairs_used":       len(valid_stems),
        "left_fov_deg":         {"horizontal": round(hfov,3), "vertical": round(vfov,3)},
        "depth_scale_m_per_unit": depth_scale,
    }
    path = os.path.join(RESULTS_DIR, "stereo_calibration.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"\nCalibration saved → {path}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — PDF Report
# ─────────────────────────────────────────────────────────────────────────────

def _p(text, style):
    return Paragraph(text, style)



# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — PDF Report
# ─────────────────────────────────────────────────────────────────────────────

def _p(text, style):
    return Paragraph(text, style)


def _tbl_style(has_header=True):
    rules = [
        ("FONTSIZE",       (0, 0), (-1, -1), 8),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("LEFTPADDING",    (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 5),
    ]
    if has_header:
        rules += [
            ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#2c5f8a")),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#eef3f8"), colors.white]),
        ]
    return TableStyle(rules)


def build_report(K_L, D_L, K_R, D_R, R, T, baseline, rms,
                 per_rms_L, per_rms_R, epi_errors,
                 cal_data, figure_paths, rs_intrinsics):

    out_path = os.path.join(RESULTS_DIR, "calibration_report.pdf")
    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            leftMargin=2.2*cm, rightMargin=2.2*cm,
                            topMargin=2.4*cm, bottomMargin=2*cm)

    SS    = getSampleStyleSheet()
    TITLE = ParagraphStyle("Title2", parent=SS["Title"],   fontSize=20, spaceAfter=4, leading=24)
    H1    = ParagraphStyle("H1",     parent=SS["Heading1"],fontSize=12, spaceAfter=3,
                            textColor=colors.HexColor("#2c5f8a"))
    H2    = ParagraphStyle("H2",     parent=SS["Heading2"],fontSize=9, spaceAfter=2,
                            textColor=colors.HexColor("#444444"))
    BODY  = ParagraphStyle("Body",   parent=SS["Normal"],  fontSize=8.5, spaceAfter=3,
                            leading=13, textColor=colors.HexColor("#222222"))
    SMALL = ParagraphStyle("Small",  parent=SS["Normal"],  fontSize=7.5, spaceAfter=2,
                            leading=11, textColor=colors.HexColor("#555555"))
    CAP   = ParagraphStyle("Cap",    parent=SS["Normal"],  fontSize=7.5,
                            textColor=colors.HexColor("#666666"), alignment=TA_CENTER,
                            spaceAfter=6)
    MONO  = ParagraphStyle("Mono",   parent=SS["Code"],    fontSize=8, spaceAfter=2,
                            fontName="Courier")

    def embed_img(path, width=15.5*cm):
        if not path or not os.path.exists(path):
            return _p(f"[missing: {path}]", SMALL)
        from PIL import Image as PILImage
        pil  = PILImage.open(path)
        pw, ph = pil.size
        return RLImage(path, width=width, height=width * (ph / pw))

    def K_table(K):
        rows = [["", "col 0 (x)", "col 1", "col 2 (y)"],
                ["row 0", f"{K[0,0]:.6f}", f"{K[0,1]:.6f}", f"{K[0,2]:.6f}"],
                ["row 1", f"{K[1,0]:.6f}", f"{K[1,1]:.6f}", f"{K[1,2]:.6f}"],
                ["row 2", f"{K[2,0]:.6f}", f"{K[2,1]:.6f}", f"{K[2,2]:.6f}"]]
        t = Table(rows, colWidths=[2.2*cm, 4.2*cm, 4.2*cm, 4.2*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#dce8f5")),
            ("BACKGROUND",    (0, 0), (0, -1),  colors.HexColor("#dce8f5")),
            ("FONTNAME",      (0, 0), (-1, -1), "Courier"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
            ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return t

    def D_table(D):
        rows = [["k1", "k2", "p1", "p2", "k3"],
                [f"{D[0]:.8f}", f"{D[1]:.8f}", f"{D[2]:.8f}",
                 f"{D[3]:.8f}", f"{D[4]:.8f}"]]
        t = Table(rows, colWidths=[3.1*cm] * 5)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#dce8f5")),
            ("FONTNAME",      (0, 0), (-1, -1), "Courier"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return t

    # Derived scalars
    w_img, h_img  = cal_data["image_size"]["width"], cal_data["image_size"]["height"]
    hfov          = cal_data["left_fov_deg"]["horizontal"]
    vfov          = cal_data["left_fov_deg"]["vertical"]
    epi_mean      = float(epi_errors.mean())
    epi_max       = float(epi_errors.max())
    qual_rms      = ("EXCELLENT" if rms  < 0.5 else "GOOD"     if rms  < 1.0
                     else "ACCEPTABLE"  if rms  < 1.5 else "POOR")
    qual_epi      = ("PASS"      if epi_mean < 1.0 else "MARGINAL" if epi_mean < 2.0 else "FAIL")
    qual_col      = ("#1a7a1a" if rms < 0.5 else "#4a7a1a" if rms < 1.0
                     else "#a07a00" if rms < 1.5 else "#aa0000")
    epi_col       = ("#1a7a1a" if epi_mean < 1.0 else "#a07a00" if epi_mean < 2.0 else "#aa0000")
    rvec_ax, _    = cv2.Rodrigues(R)
    rot_deg       = float(np.degrees(np.linalg.norm(rvec_ax)))
    tx, ty, tz    = float(T[0]), float(T[1]), float(T[2])
    depth_res_200 = 200.0 ** 2 / (float(K_L[0, 0]) * baseline)

    story = []

    # Title page
    story += [
        Spacer(1, 0.8*cm),
        _p("MSCR Stereo Camera Calibration Report", TITLE),
        HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2c5f8a"), spaceAfter=6),
        _p(f"Date: {cal_data['calibration_date']}    "
           f"Sensor: RealSense D435i IR stereo    "
           f"Mount: fixed", SMALL),
        Spacer(1, 0.4*cm),
    ]

    sum_rows = [
        ["Parameter",                  "Value"],
        ["Resolution",                 f"{w_img} x {h_img} px"],
        ["Stereo RMS reprojection",    f"{rms:.4f} px  ({qual_rms})"],
        ["Mean epipolar error",        f"{epi_mean:.4f} px  ({qual_epi})"],
        ["Max epipolar error",         f"{epi_max:.4f} px"],
        ["Image pairs used",           str(cal_data["num_pairs_used"])],
        ["Left  fx / fy",              f"{K_L[0,0]:.3f} / {K_L[1,1]:.3f} px"],
        ["Right fx / fy",              f"{K_R[0,0]:.3f} / {K_R[1,1]:.3f} px"],
        ["Stereo baseline",            f"{baseline:.3f} mm"],
        ["Inter-camera rotation",      f"{rot_deg:.4f} deg"],
        ["Horizontal / Vertical FOV",  f"{hfov:.2f} / {vfov:.2f} deg"],
    ]
    st = Table(sum_rows, colWidths=[8*cm, 6.8*cm])
    st.setStyle(_tbl_style(has_header=True))
    story += [st, Spacer(1, 0.4*cm),
              _p(f'Quality: <font color="{qual_col}"><b>{qual_rms}</b></font> '
                 f'(RMS = {rms:.4f} px)    '
                 f'Epipolar: <font color="{epi_col}"><b>{qual_epi}</b></font> '
                 f'(mean = {epi_mean:.4f} px)', BODY),
              _p("Thresholds: RMS below 0.5 px (excellent), epipolar below 1.0 px (pass). "
                 "Results above 1.5 px / 2.0 px respectively indicate insufficient pose "
                 "diversity or board motion during capture.", SMALL),
              PageBreak()]

    # Section 1: Intrinsics
    story += [_p("1. Intrinsic Parameters", H1)]
    for sec, label, K, D in [("1.1", "Left IR Camera",  K_L, D_L),
                               ("1.2", "Right IR Camera", K_R, D_R)]:
        fx_c = K[0, 0]; fy_c = K[1, 1]; cx_c = K[0, 2]; cy_c = K[1, 2]
        hf   = float(np.degrees(2 * np.arctan(w_img / (2 * fx_c))))
        vf   = float(np.degrees(2 * np.arctan(h_img / (2 * fy_c))))
        story += [
            _p(f"{sec}  {label}", H2),
            _p(f"fx = {fx_c:.4f} px    fy = {fy_c:.4f} px    "
               f"cx = {cx_c:.4f} px    cy = {cy_c:.4f} px    "
               f"FOV = {hf:.2f} x {vf:.2f} deg", MONO),
            Spacer(1, 0.2*cm),
            K_table(K), Spacer(1, 0.25*cm),
            D_table(D), Spacer(1, 0.15*cm),
            _p("Brown-Conrady model. Radial: k1, k2, k3.  Tangential: p1, p2.", SMALL),
            Spacer(1, 0.35*cm),
        ]

    if rs_intrinsics and "left_ir" in rs_intrinsics:
        fw_L = rs_intrinsics["left_ir"]
        fw_R = rs_intrinsics["right_ir"]
        def d(cal, fw):
            try:    return f"{cal - float(fw):+.3f}"
            except: return "n/a"
        story += [_p("1.3  Factory Firmware vs This Calibration", H2),
                  _p("Factory intrinsics shown for reference only. The values above "
                     "should be used in the control pipeline.", SMALL),
                  Spacer(1, 0.1*cm)]
        cmp = [["", "Cal Left", "FW Left", "Delta", "Cal Right", "FW Right", "Delta"],
               ["fx", f"{K_L[0,0]:.3f}", f"{fw_L['fx']:.3f}", d(K_L[0,0], fw_L['fx']),
                       f"{K_R[0,0]:.3f}", f"{fw_R['fx']:.3f}", d(K_R[0,0], fw_R['fx'])],
               ["fy", f"{K_L[1,1]:.3f}", f"{fw_L['fy']:.3f}", d(K_L[1,1], fw_L['fy']),
                       f"{K_R[1,1]:.3f}", f"{fw_R['fy']:.3f}", d(K_R[1,1], fw_R['fy'])],
               ["cx", f"{K_L[0,2]:.3f}", f"{fw_L['cx']:.3f}", d(K_L[0,2], fw_L['cx']),
                       f"{K_R[0,2]:.3f}", f"{fw_R['cx']:.3f}", d(K_R[0,2], fw_R['cx'])],
               ["cy", f"{K_L[1,2]:.3f}", f"{fw_L['cy']:.3f}", d(K_L[1,2], fw_L['cy']),
                       f"{K_R[1,2]:.3f}", f"{fw_R['cy']:.3f}", d(K_R[1,2], fw_R['cy'])]]
        ct = Table(cmp, colWidths=[1.3*cm,2.4*cm,2.4*cm,1.5*cm,2.4*cm,2.4*cm,1.5*cm])
        ct.setStyle(_tbl_style(has_header=True))
        story += [ct, Spacer(1, 0.3*cm)]

    story.append(PageBreak())

    # Section 2: Stereo Extrinsics
    story += [_p("2. Stereo Extrinsics", H1)]
    Rf   = R.flatten()
    Rr   = [["R[row,col]", "col 0",         "col 1",         "col 2"],
             ["row 0",      f"{Rf[0]:.8f}",  f"{Rf[1]:.8f}",  f"{Rf[2]:.8f}"],
             ["row 1",      f"{Rf[3]:.8f}",  f"{Rf[4]:.8f}",  f"{Rf[5]:.8f}"],
             ["row 2",      f"{Rf[6]:.8f}",  f"{Rf[7]:.8f}",  f"{Rf[8]:.8f}"]]
    rt   = Table(Rr, colWidths=[2.6*cm, 4.2*cm, 4.2*cm, 4.2*cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#dce8f5")),
        ("BACKGROUND",    (0, 0), (0, -1),  colors.HexColor("#dce8f5")),
        ("FONTNAME",      (0, 0), (-1, -1), "Courier"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [
        _p("Rotation R (left to right IR) and translation T:", H2),
        rt, Spacer(1, 0.2*cm),
        _p(f"T = [{tx:.5f},  {ty:.5f},  {tz:.5f}] mm     "
           f"|T| = {baseline:.4f} mm     "
           f"Rotation = {rot_deg:.4f} deg", MONO),
        Spacer(1, 0.15*cm),
        _p("Rotation angle below 0.5 deg indicates parallel-mounted cameras (nominal "
           "for the D435i). Values above 1 deg may indicate a mounting issue.", SMALL),
        PageBreak(),
    ]

    # Sections 3-7: Figures
    sections = [
        ("3. Reprojection Error Analysis", "reprojection", 15.5*cm,
         "Per-pair RMS for left (blue) and right (red) after iterative outlier rejection. "
         "Orange bars exceed the 1.0 px limit. Right panels show per-corner error vectors "
         "with 1-sigma and 2-sigma ellipses. A symmetric, tight cluster near the origin "
         "indicates well-conditioned calibration.",
         "Figure 1 - Per-pair RMS (left) and per-corner error vectors for left and right cameras."),

        ("4. Stereo Rectification Verification", "rectification", 15.5*cm,
         "Original IR pair (top) and rectified pair with epipolar lines (bottom). "
         "After rectification, any corresponding scene point must lie on the same "
         "horizontal scanline in both views. Vertical offset is residual calibration error.",
         "Figure 2 - Original and rectified IR pair. Green lines are epipolar lines."),

        ("5. Lens Distortion Maps", "distortion", 15.5*cm,
         "Per-pixel undistortion displacement for left and right IR sensors. "
         "Arrows show the direction and magnitude of the correction at a sampled grid. "
         "Barrel distortion dominates at the sensor periphery as expected.",
         "Figure 3 - Distortion displacement magnitude (px) for left and right cameras."),

        ("6. Calibration Pose Coverage", "poses", 11*cm,
         "Estimated checkerboard positions in the left camera frame across all accepted "
         "image pairs. Broad spatial coverage across X, Y, and Z produces a well-conditioned "
         "calibration; clustering in one region can inflate error at other positions.",
         "Figure 4 - Board pose distribution in the left camera frame (mm)."),

        ("7. Depth Resolution for MSCR Control", "depth_resolution", 15.5*cm,
         f"Depth Z = fx * B / d, where B = {baseline:.2f} mm and fx = {K_L[0,0]:.1f} px. "
         f"At 200 mm working distance, depth resolution is {depth_res_200:.3f} mm per disparity "
         f"pixel; sub-pixel disparity estimation is required for sub-millimetre "
         f"endpoint tracking at this range.",
         "Figure 5 - Disparity and depth resolution vs working distance, with reference table."),
    ]

    for title, key, w_fig, body_text, caption_text in sections:
        story += [
            _p(title, H1),
            _p(body_text, BODY),
            Spacer(1, 0.25*cm),
            embed_img(figure_paths.get(key), width=w_fig),
            Spacer(1, 0.08*cm),
            _p(caption_text, CAP),
            PageBreak(),
        ]

    doc.build(story)
    print(f"Report saved -> {out_path}")
    return out_path


# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Check directory structure
    if not os.path.isdir(LEFT_DIR) or not os.path.isdir(RIGHT_DIR):
        print(f"[ERROR] Expected '{LEFT_DIR}/' and '{RIGHT_DIR}/' directories.")
        print("  Run capture.py first to collect stereo IR pairs.")
        sys.exit(1)

    # Load firmware intrinsics if available
    rs_intrinsics = None
    fw_path = os.path.join(RESULTS_DIR, "rs_firmware_intrinsics.json")
    if os.path.exists(fw_path):
        with open(fw_path) as f:
            rs_intrinsics = json.load(f)

    # Phase 1 — detect corners
    obj_pts, img_pts_L, img_pts_R, valid_stems, image_size = \
        detect_stereo_corners(LEFT_DIR, RIGHT_DIR)

    if len(valid_stems) < 6:
        print(f"[ERROR] Only {len(valid_stems)} valid pairs. Recapture.")
        sys.exit(1)

    # Phase 2 — outlier rejection
    obj_pts, img_pts_L, img_pts_R, valid_stems = \
        iterative_refine(obj_pts, img_pts_L, img_pts_R, valid_stems, image_size)

    if len(valid_stems) < 6:
        print("[ERROR] Too few pairs remaining after rejection. Recapture.")
        sys.exit(1)

    # Phase 3 — calibrate
    K_L, D_L, K_R, D_R = calibrate_individual(
        obj_pts, img_pts_L, img_pts_R, image_size)

    K_L, D_L, K_R, D_R, R, T, E, F_mat, rms, baseline = stereo_calibrate(
        obj_pts, img_pts_L, img_pts_R, K_L, D_L, K_R, D_R, image_size)

    R1, R2, P1, P2, Q, roi_L, roi_R = stereo_rectify(
        K_L, D_L, K_R, D_R, R, T, image_size)

    # Phase 4 — error analysis
    per_rms_L, per_rms_R, all_err_L, all_err_R, epi_errors = analyse_errors(
        K_L, D_L, K_R, D_R, R, T,
        obj_pts, img_pts_L, img_pts_R, F_mat)

    # Phase 5 — figures
    print("\nGenerating figures …")
    fps = {}
    fps["reprojection"]    = fig_reprojection(per_rms_L, per_rms_R,
                                               all_err_L, all_err_R, epi_errors)
    fps["rectification"]   = fig_rectification(K_L, D_L, K_R, D_R,
                                                R1, R2, P1, P2, image_size, valid_stems)
    fps["distortion"]      = fig_distortion(K_L, D_L, K_R, D_R, image_size)
    fps["poses"]           = fig_pose_coverage(K_L, D_L, obj_pts, img_pts_L, image_size)
    fps["depth_resolution"]= fig_depth_resolution(K_L, T, baseline)
    print("  All figures saved to results/figures/")

    # Phase 6 — YAML
    cal_data = save_yaml(K_L, D_L, K_R, D_R, R, T, E, F_mat,
                         R1, R2, P1, P2, Q, rms, baseline, valid_stems, image_size)

    # Phase 7 — PDF
    print("\nBuilding PDF report …")
    report_path = build_report(K_L, D_L, K_R, D_R, R, T, baseline, rms,
                               per_rms_L, per_rms_R, epi_errors,
                               cal_data, fps, rs_intrinsics)

    print(f"\n{'─'*60}")
    print(f"  DONE")
    print(f"  Calibration : {os.path.join(RESULTS_DIR, 'stereo_calibration.yaml')}")
    print(f"  Report      : {report_path}")
    print(f"  Stereo RMS  : {rms:.4f} px")
    print(f"  Epipolar    : {epi_errors.mean():.4f} px  (mean)")
    print(f"  Baseline    : {baseline:.3f} mm")
    print(f"  Pairs used  : {len(valid_stems)}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
