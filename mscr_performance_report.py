#!/usr/bin/env python3
"""
mscr_performance_report.py — Live performance benchmarking for MSCRTracker.

Runs the tracker for a configurable number of frames, collects per-frame
metrics, then writes a multi-page PDF report with plots and summary tables.

Usage:
    python mscr_performance_report.py --frames 300 --out report.pdf [tracker flags]
    python mscr_performance_report.py --frames 300 --debug --threshold 60

The camera stream is opened, N frames are collected, the stream is closed,
then the PDF is generated and saved.  No display window is opened unless
--debug is passed (in which case the tracker's normal debug view is shown
live during collection).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")                     # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

from mscr_tracker import MSCRTracker, TrackerParams


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame metric container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameMetric:
    frame_idx:      int
    timestamp_s:    float          # wall-clock seconds since stream start
    proc_time_ms:   float          # process_frame() latency (ms)
    valid:          bool           # 3-D tracking succeeded
    skeleton_found: bool           # rod was segmented (2-D or 3-D)
    n_skel_pts:     int            # skeleton pixel count (0 if not found)
    valid_depth_frac: float        # fraction of skeleton pts with real depth
    arc_length_mm:  float          # smoothed arc length (0 if invalid)
    tip_x_mm:       float
    tip_y_mm:       float
    tip_z_mm:       float
    tip_jump_mm:    float          # Euclidean distance from previous tip (mm)


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented sub-class that records metrics without changing tracker logic
# ─────────────────────────────────────────────────────────────────────────────

class InstrumentedTracker(MSCRTracker):
    """Wraps MSCRTracker and intercepts _skeletonise / process_frame to record
    additional per-frame diagnostics that are not exposed in TrackResult."""

    def __init__(self, params: TrackerParams):
        super().__init__(params)
        self._last_n_skel    = 0
        self._last_valid_df  = 0.0

    # Intercept skeletonise to capture skeleton size
    def _skeletonise(self, mask):
        result = super()._skeletonise(mask)
        self._last_n_skel = len(result) if result is not None else 0
        return result

    # Intercept 3-D projection to capture valid-depth fraction
    def _project_skeleton_3d(self, ordered, depth_roi, ox, oy):
        pts, vmask = super()._project_skeleton_3d(ordered, depth_roi, ox, oy)
        if vmask is not None and len(vmask):
            self._last_valid_df = float(vmask.sum()) / max(len(vmask), 1)
        else:
            self._last_valid_df = 0.0
        return pts, vmask


# ─────────────────────────────────────────────────────────────────────────────
# Collection loop
# ─────────────────────────────────────────────────────────────────────────────

def collect_metrics(params: TrackerParams, n_frames: int) -> List[FrameMetric]:
    """Stream from the RealSense and collect metrics for `n_frames` frames."""
    import pyrealsense2 as rs
    from collections import deque

    tracker = InstrumentedTracker(params)
    metrics: List[FrameMetric] = []

    pipeline = rs.pipeline()
    cfg      = rs.config()
    W, H, FPS = params.width, params.height, params.fps
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16,  FPS)

    profile      = pipeline.start(cfg)
    depth_sensor = profile.get_device().first_depth_sensor()
    tracker.depth_scale = depth_sensor.get_depth_scale()

    try:
        if params.laser_power == 0:
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
        else:
            hi  = depth_sensor.get_option_range(rs.option.laser_power).max
            pwr = max(1, min(int(hi), params.laser_power))
            depth_sensor.set_option(rs.option.emitter_enabled, 1)
            depth_sensor.set_option(rs.option.laser_power, pwr)
    except Exception as e:
        print(f"[WARN] emitter: {e}")

    align = rs.align(rs.stream.color)

    # Warm-up + intrinsics
    print("Warming up camera (5 frames)…")
    for _ in range(5):
        frames   = pipeline.wait_for_frames(timeout_ms=5000)
        aligned  = align.process(frames)
    color_profile = aligned.get_color_frame().get_profile()
    tracker.set_intrinsics(
        color_profile.as_video_stream_profile().get_intrinsics())

    print(f"Collecting {n_frames} frames…")
    t_stream_start = time.perf_counter()
    prev_tip = np.zeros(3)
    collected = 0

    if params.debug:
        import cv2
        cv2.namedWindow("MSCR Perf Collect", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MSCR Perf Collect", W * 2, H)

    try:
        while collected < n_frames:
            frames  = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)

            color_f = aligned.get_color_frame()
            depth_f = aligned.get_depth_frame()
            if not color_f or not depth_f:
                continue

            color     = np.asanyarray(color_f.get_data())
            depth_raw = np.asanyarray(depth_f.get_data())
            depth_filt = tracker._spatial.process(depth_f)
            depth_filt = tracker._temporal.process(depth_filt)
            depth      = np.asanyarray(depth_filt.get_data())

            ts   = time.perf_counter() - t_stream_start
            t0   = time.perf_counter()
            result = tracker.process_frame(color, depth, depth_raw=depth_raw)
            proc_ms = (time.perf_counter() - t0) * 1000.0

            tip = np.array(result.tip_xyz_mm)
            jump = float(np.linalg.norm(tip - prev_tip)) if result.valid else 0.0
            if result.valid:
                prev_tip = tip.copy()

            metrics.append(FrameMetric(
                frame_idx       = collected,
                timestamp_s     = ts,
                proc_time_ms    = proc_ms,
                valid           = result.valid,
                skeleton_found  = len(result.centerline_px) > 0,
                n_skel_pts      = tracker._last_n_skel,
                valid_depth_frac= tracker._last_valid_df,
                arc_length_mm   = result.arc_length_mm,
                tip_x_mm        = tip[0],
                tip_y_mm        = tip[1],
                tip_z_mm        = tip[2],
                tip_jump_mm     = jump,
            ))

            collected += 1
            if collected % 50 == 0:
                det_rate = sum(m.valid for m in metrics) / len(metrics) * 100
                print(f"  {collected}/{n_frames}  "
                      f"detect={det_rate:.1f}%  "
                      f"proc={proc_ms:.1f}ms", flush=True)

            if params.debug and result.debug_frame is not None:
                import cv2
                cv2.imshow("MSCR Perf Collect", result.debug_frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted — using frames collected so far.")
    finally:
        pipeline.stop()
        if params.debug:
            import cv2
            cv2.destroyAllWindows()

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# PDF generation
# ─────────────────────────────────────────────────────────────────────────────

# ── Colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#2196F3"
C_GREEN  = "#4CAF50"
C_ORANGE = "#FF9800"
C_RED    = "#F44336"
C_GRAY   = "#9E9E9E"
C_TEAL   = "#009688"

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
    "lines.linewidth":   1.2,
    "figure.dpi":        150,
})


def _page_header(fig: plt.Figure, title: str, subtitle: str = "") -> None:
    fig.text(0.5, 0.97, title,    ha="center", va="top",
             fontsize=14, fontweight="bold", color="#212121")
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="top",
                 fontsize=9, color="#757575")


def _summary_stats(arr: np.ndarray, label: str) -> dict:
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return {label: {"n": 0, "mean": np.nan, "std": np.nan,
                        "min": np.nan, "max": np.nan, "p95": np.nan}}
    return {label: {
        "n":    len(valid),
        "mean": float(np.mean(valid)),
        "std":  float(np.std(valid)),
        "min":  float(np.min(valid)),
        "max":  float(np.max(valid)),
        "p95":  float(np.percentile(valid, 95)),
    }}


def generate_pdf(metrics: List[FrameMetric], out_path: str,
                 params: TrackerParams) -> None:

    if not metrics:
        print("[ERROR] No metrics to report.")
        return

    # ── Pre-compute arrays ────────────────────────────────────────────────────
    N          = len(metrics)
    t          = np.array([m.timestamp_s   for m in metrics])
    proc_ms    = np.array([m.proc_time_ms  for m in metrics])
    valid      = np.array([m.valid         for m in metrics], dtype=bool)
    skel_found = np.array([m.skeleton_found for m in metrics], dtype=bool)
    n_skel     = np.array([m.n_skel_pts    for m in metrics])
    vdf        = np.array([m.valid_depth_frac for m in metrics])
    arc        = np.array([m.arc_length_mm for m in metrics])
    tx         = np.array([m.tip_x_mm      for m in metrics])
    ty         = np.array([m.tip_y_mm      for m in metrics])
    tz         = np.array([m.tip_z_mm      for m in metrics])
    jump       = np.array([m.tip_jump_mm   for m in metrics])

    detect_rate  = valid.sum()  / N * 100
    skel_rate    = skel_found.sum() / N * 100
    mean_fps     = 1000.0 / float(np.mean(proc_ms))
    duration_s   = float(t[-1] - t[0]) if N > 1 else 0.0

    arc_valid    = arc[valid]
    tx_v, ty_v, tz_v = tx[valid], ty[valid], tz[valid]
    jump_v       = jump[valid]

    # Rolling detection rate (window = 30 frames ≈ 1 s)
    W30 = 30
    rolling_det = np.convolve(valid.astype(float),
                              np.ones(W30)/W30, mode="same") * 100

    with PdfPages(out_path) as pdf:

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 1 — Executive summary
        # ══════════════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(11, 8.5))
        _page_header(fig, "MSCR Tracker — Performance Analytics",
                     subtitle=f"Session: {N} frames  |  {duration_s:.1f} s  |  "
                              f"Resolution {params.width}×{params.height} @ {params.fps} fps")

        gs = gridspec.GridSpec(3, 3, figure=fig,
                               top=0.88, bottom=0.08,
                               left=0.07, right=0.97,
                               hspace=0.55, wspace=0.38)

        # ── KPI tiles ─────────────────────────────────────────────────────────
        kpis = [
            ("3-D Detection\nRate",   f"{detect_rate:.1f} %",
             C_GREEN if detect_rate > 70 else C_ORANGE),
            ("Skeleton\nFound Rate",  f"{skel_rate:.1f} %",
             C_BLUE),
            ("Mean Proc.\nTime",      f"{np.mean(proc_ms):.1f} ms",
             C_TEAL),
            ("Throughput\n(est.)",    f"{mean_fps:.1f} fps",
             C_TEAL),
            ("Mean Arc\nLength",      f"{float(np.mean(arc_valid)):.1f} mm"
                                       if len(arc_valid) else "—", C_ORANGE),
            ("Tip Jitter\n(p95)",     f"{float(np.percentile(jump_v,95)):.2f} mm"
                                       if len(jump_v) else "—", C_RED),
        ]
        for k, (label, value, color) in enumerate(kpis):
            ax = fig.add_subplot(gs[0, k % 3] if k < 3 else gs[1, k % 3])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis("off")
            ax.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.8,
                         facecolor=color, alpha=0.12, transform=ax.transAxes))
            ax.text(0.5, 0.72, value,  ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color,
                    transform=ax.transAxes)
            ax.text(0.5, 0.28, label,  ha="center", va="center",
                    fontsize=8.5, color="#424242",
                    transform=ax.transAxes)

        # ── Rolling detection rate timeline ───────────────────────────────────
        ax_r = fig.add_subplot(gs[2, :])
        ax_r.fill_between(t, rolling_det, alpha=0.25, color=C_GREEN)
        ax_r.plot(t, rolling_det, color=C_GREEN, lw=1.2, label="Rolling detect %")
        ax_r.axhline(detect_rate, color=C_GRAY, ls="--", lw=0.9,
                     label=f"Session mean {detect_rate:.1f}%")
        ax_r.set_xlabel("Time (s)")
        ax_r.set_ylabel("Detection rate (%)")
        ax_r.set_title("Rolling 3-D Detection Rate (30-frame window)")
        ax_r.set_ylim(-5, 105)
        ax_r.legend(fontsize=8, loc="lower right")

        # Parameter table
        param_text = (
            f"dark_threshold={params.dark_threshold}  |  "
            f"depth_gate=[{params.depth_min_m:.2f}, {params.depth_max_m:.2f}] m  |  "
            f"min_ecc={params.min_eccentricity:.2f}  |  "
            f"ema_α={params.ema_alpha:.2f}  |  "
            f"min_skel={params.min_skel_pts}  |  "
            f"n_resample={params.n_resample}"
        )
        fig.text(0.5, 0.04, param_text, ha="center", fontsize=7.5, color="#616161")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 2 — Processing latency and skeleton quality
        # ══════════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        _page_header(fig, "Processing Latency & Skeleton Quality")
        fig.subplots_adjust(top=0.88, bottom=0.08, hspace=0.45, wspace=0.35)
        axes = axes.flatten()

        # Processing time over time
        ax = axes[0]
        ax.plot(t, proc_ms, color=C_BLUE, lw=0.8, alpha=0.7)
        ax.axhline(np.mean(proc_ms), color=C_RED, ls="--", lw=1,
                   label=f"Mean {np.mean(proc_ms):.1f} ms")
        ax.axhline(np.percentile(proc_ms, 95), color=C_ORANGE, ls=":",
                   lw=1, label=f"p95 {np.percentile(proc_ms,95):.1f} ms")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Latency (ms)")
        ax.set_title("process_frame() Latency per Frame")
        ax.legend(fontsize=8)

        # Latency histogram
        ax = axes[1]
        ax.hist(proc_ms, bins=40, color=C_BLUE, alpha=0.7, edgecolor="white",
                linewidth=0.4)
        ax.axvline(np.mean(proc_ms), color=C_RED,    ls="--", lw=1.2,
                   label=f"Mean {np.mean(proc_ms):.1f} ms")
        ax.axvline(np.percentile(proc_ms,95), color=C_ORANGE, ls=":", lw=1.2,
                   label=f"p95 {np.percentile(proc_ms,95):.1f} ms")
        ax.set_xlabel("Latency (ms)"); ax.set_ylabel("Frame count")
        ax.set_title("Latency Distribution")
        ax.legend(fontsize=8)

        # Skeleton pixel count
        ax = axes[2]
        ax.scatter(t[skel_found], n_skel[skel_found], s=2,
                   color=C_TEAL, alpha=0.5, label="Skeleton found")
        ax.scatter(t[~skel_found], n_skel[~skel_found], s=2,
                   color=C_RED, alpha=0.4, label="Not found")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Pixel count")
        ax.set_title("Skeleton Pixel Count per Frame")
        ax.legend(fontsize=8, markerscale=3)

        # Valid depth fraction
        ax = axes[3]
        ax.plot(t, vdf * 100, color=C_ORANGE, lw=0.8, alpha=0.7)
        ax.axhline(params.min_valid_depth_frac * 100, color=C_RED,
                   ls="--", lw=1.2,
                   label=f"Threshold {params.min_valid_depth_frac*100:.0f}%")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Valid depth (%)")
        ax.set_title("Fraction of Skeleton Pixels with Valid Depth")
        ax.set_ylim(-2, 102)
        ax.legend(fontsize=8)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 3 — 3-D tip position trajectories
        # ══════════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        _page_header(fig, "3-D Tip Position Trajectories")
        fig.subplots_adjust(top=0.88, bottom=0.08, hspace=0.45, wspace=0.35)
        axes = axes.flatten()

        tv = t[valid]

        for ax, coord, label, color in [
            (axes[0], tx_v, "Tip X (mm)", C_BLUE),
            (axes[1], ty_v, "Tip Y (mm)", C_GREEN),
            (axes[2], tz_v, "Tip Z (mm)", C_ORANGE),
        ]:
            ax.plot(tv, coord, color=color, lw=0.9, alpha=0.8)
            ax.set_xlabel("Time (s)"); ax.set_ylabel(label)
            ax.set_title(label + " vs Time")
            # Shade ±1σ band
            mu, sigma = float(np.mean(coord)), float(np.std(coord))
            ax.axhline(mu, color=C_GRAY, ls="--", lw=0.8,
                       label=f"μ={mu:.2f} mm")
            ax.fill_between(tv, mu - sigma, mu + sigma,
                            alpha=0.12, color=color,
                            label=f"±σ={sigma:.2f} mm")
            ax.legend(fontsize=8)

        # 3-D scatter: X vs Y coloured by Z
        ax = axes[3]
        if len(tx_v) > 0:
            sc = ax.scatter(tx_v, ty_v, c=tz_v, cmap="plasma",
                            s=4, alpha=0.6, linewidths=0)
            plt.colorbar(sc, ax=ax, label="Z (mm)", pad=0.02)
            ax.set_xlabel("Tip X (mm)"); ax.set_ylabel("Tip Y (mm)")
            ax.set_title("Tip XY Trajectory (colour = Z)")
            # Mark start and end
            ax.plot(tx_v[0], ty_v[0], "go", ms=7, label="Start")
            ax.plot(tx_v[-1], ty_v[-1], "rs", ms=7, label="End")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No valid 3-D detections",
                    ha="center", va="center", transform=ax.transAxes,
                    color=C_GRAY)
            ax.set_title("Tip XY Trajectory")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 4 — Arc length & tip jitter
        # ══════════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        _page_header(fig, "Arc Length & Tip Stability")
        fig.subplots_adjust(top=0.88, bottom=0.08, hspace=0.45, wspace=0.35)
        axes = axes.flatten()

        # Arc length over time
        ax = axes[0]
        ax.plot(tv, arc_valid, color=C_TEAL, lw=0.9, alpha=0.85)
        mu_arc = float(np.mean(arc_valid)) if len(arc_valid) else 0
        ax.axhline(mu_arc, color=C_GRAY, ls="--", lw=0.9,
                   label=f"Mean {mu_arc:.1f} mm")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Arc length (mm)")
        ax.set_title("3-D Arc Length over Time")
        ax.legend(fontsize=8)

        # Arc length histogram
        ax = axes[1]
        if len(arc_valid):
            ax.hist(arc_valid, bins=35, color=C_TEAL, alpha=0.75,
                    edgecolor="white", linewidth=0.4)
            ax.axvline(mu_arc, color=C_RED, ls="--", lw=1.2,
                       label=f"Mean {mu_arc:.1f} mm")
            ax.axvline(float(np.percentile(arc_valid, 5)), color=C_ORANGE,
                       ls=":", lw=1.2,
                       label=f"p5–p95 [{float(np.percentile(arc_valid,5)):.1f},"
                             f"{float(np.percentile(arc_valid,95)):.1f}]")
            ax.axvline(float(np.percentile(arc_valid, 95)), color=C_ORANGE,
                       ls=":", lw=1.2)
            ax.set_xlabel("Arc length (mm)"); ax.set_ylabel("Frame count")
            ax.set_title("Arc Length Distribution")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center",
                    transform=ax.transAxes, color=C_GRAY)

        # Frame-to-frame tip jump
        ax = axes[2]
        if len(jump_v):
            ax.plot(tv, jump_v, color=C_RED, lw=0.7, alpha=0.7)
            ax.axhline(float(np.mean(jump_v)), color=C_GRAY, ls="--", lw=0.9,
                       label=f"Mean {float(np.mean(jump_v)):.2f} mm")
            ax.axhline(params.max_jump_mm, color=C_ORANGE, ls=":", lw=1.1,
                       label=f"Spike threshold {params.max_jump_mm:.0f} mm")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Δ tip (mm)")
            ax.set_title("Frame-to-Frame Tip Displacement")
            ax.legend(fontsize=8)

        # Tip jitter histogram (log scale)
        ax = axes[3]
        if len(jump_v):
            bins = np.linspace(0, min(float(np.percentile(jump_v, 99)) * 1.1, 50), 40)
            ax.hist(jump_v, bins=bins, color=C_RED, alpha=0.7,
                    edgecolor="white", linewidth=0.4)
            ax.axvline(float(np.mean(jump_v)), color=C_GRAY, ls="--", lw=1.2,
                       label=f"Mean {float(np.mean(jump_v)):.2f} mm")
            ax.axvline(float(np.percentile(jump_v, 95)), color=C_ORANGE,
                       ls=":", lw=1.2,
                       label=f"p95 {float(np.percentile(jump_v,95)):.2f} mm")
            ax.set_xlabel("Δ tip (mm)"); ax.set_ylabel("Frame count")
            ax.set_title("Tip Displacement Distribution")
            ax.legend(fontsize=8)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 5 — Summary statistics table
        # ══════════════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(11, 8.5))
        _page_header(fig, "Summary Statistics Table")

        stats_rows = []

        def add(label, arr, unit=""):
            if len(arr) == 0:
                stats_rows.append([label, "—", "—", "—", "—", "—", "—", unit])
                return
            stats_rows.append([
                label,
                f"{float(np.mean(arr)):.3f}",
                f"{float(np.std(arr)):.3f}",
                f"{float(np.min(arr)):.3f}",
                f"{float(np.max(arr)):.3f}",
                f"{float(np.percentile(arr, 5)):.3f}",
                f"{float(np.percentile(arr, 95)):.3f}",
                unit,
            ])

        add("Process latency",      proc_ms,  "ms")
        add("Skeleton pixel count", n_skel[skel_found].astype(float), "px")
        add("Valid depth fraction", vdf[skel_found] * 100, "%")
        add("Arc length",           arc_valid, "mm")
        add("Tip X",                tx_v,      "mm")
        add("Tip Y",                ty_v,      "mm")
        add("Tip Z",                tz_v,      "mm")
        add("Tip jump Δ",           jump_v,    "mm")

        col_labels = ["Metric", "Mean", "Std", "Min", "Max", "p5", "p95", "Unit"]
        ax_t = fig.add_axes([0.04, 0.25, 0.92, 0.58])
        ax_t.axis("off")
        tbl = ax_t.table(
            cellText   = stats_rows,
            colLabels  = col_labels,
            loc        = "center",
            cellLoc    = "center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.7)

        # Style header
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#1565C0")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        # Alternate row shading
        for i in range(1, len(stats_rows) + 1):
            for j in range(len(col_labels)):
                tbl[i, j].set_facecolor("#EEF2FF" if i % 2 == 0 else "white")

        # Session-level counts
        summary_text = (
            f"Total frames: {N}   |   "
            f"3-D valid: {valid.sum()} ({detect_rate:.1f}%)   |   "
            f"Skeleton found: {skel_found.sum()} ({skel_rate:.1f}%)   |   "
            f"Duration: {duration_s:.1f} s   |   "
            f"Mean throughput: {mean_fps:.1f} fps"
        )
        fig.text(0.5, 0.20, summary_text, ha="center", fontsize=9,
                 color="#212121", fontweight="bold")

        fig.text(0.5, 0.10,
                 "Note: 'Tip jump' is frame-to-frame Euclidean distance "
                 "of the smoothed tip in 3-D (mm). "
                 "Only computed on consecutive valid frames.",
                 ha="center", fontsize=8, color="#616161")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── PDF metadata ──────────────────────────────────────────────────────
        d = pdf.infodict()
        d["Title"]   = "MSCR Tracker Performance Analytics"
        d["Author"]  = "mscr_performance_report.py"
        d["Subject"] = f"{N} frames, {duration_s:.1f}s session"

    print(f"\nReport saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Collect MSCR tracker metrics and write a PDF report")
    ap.add_argument("--frames",     type=int,   default=300,
                    help="Number of frames to collect (default 300 ≈ 10 s)")
    ap.add_argument("--out",        default="mscr_performance_report.pdf",
                    help="Output PDF path")
    ap.add_argument("--width",      type=int,   default=848)
    ap.add_argument("--height",     type=int,   default=480)
    ap.add_argument("--fps",        type=int,   default=30)
    ap.add_argument("--depth-min",  type=float, default=0.05)
    ap.add_argument("--depth-max",  type=float, default=0.80)
    ap.add_argument("--threshold",  type=int,   default=55)
    ap.add_argument("--entry",      default="bottom",
                    choices=["top", "bottom", "left", "right"])
    ap.add_argument("--power",      type=int,   default=0)
    ap.add_argument("--debug",      action="store_true",
                    help="Show live debug window during collection")
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
    )

    metrics = collect_metrics(params, n_frames=args.frames)
    if metrics:
        generate_pdf(metrics, out_path=args.out, params=params)
    else:
        print("No frames collected — is the camera connected?")
