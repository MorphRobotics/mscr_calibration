#!/usr/bin/env python3
"""
generate_handoff.py — produce mscr_handoff.pdf, a self-contained
context document for continuing this project in a new session.

Run:   python generate_handoff.py
Output: mscr_handoff.pdf  (same directory)
"""

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).parent / "mscr_handoff.pdf"

# ── shared style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

C_DARK  = "#212121"
C_MID   = "#424242"
C_LIGHT = "#757575"
C_BLUE  = "#1565C0"
C_TEAL  = "#00695C"
C_RED   = "#B71C1C"
C_AMB   = "#E65100"


def page(fig, title, subtitle=""):
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.965, title,    ha="center", va="top",
             fontsize=15, fontweight="bold", color=C_DARK)
    if subtitle:
        fig.text(0.5, 0.935, subtitle, ha="center", va="top",
                 fontsize=9,  color=C_LIGHT)


def ruled_text(ax, lines, x=0.0, y=1.0, lh=0.072, fs=8.8,
               indent_color=C_BLUE):
    """
    Render a list of (indent_level, text) tuples as a structured list.
    indent_level 0 = section heading, 1 = body, 2 = sub-item.
    """
    ax.axis("off")
    cursor = y
    for level, text in lines:
        if level == 0:
            ax.text(x, cursor, text, transform=ax.transAxes,
                    fontsize=fs + 0.5, fontweight="bold", color=C_DARK,
                    va="top")
        elif level == 1:
            ax.text(x + 0.02, cursor, "•  " + text,
                    transform=ax.transAxes,
                    fontsize=fs, color=C_MID, va="top")
        else:
            ax.text(x + 0.06, cursor, "–  " + text,
                    transform=ax.transAxes,
                    fontsize=fs - 0.5, color=C_LIGHT, va="top",
                    style="italic")
        cursor -= lh
    return cursor


def two_col_table(ax, rows, col_w=(0.38, 0.62), header_bg=C_BLUE,
                  fs=8.5, row_h=0.068):
    ax.axis("off")
    y = 1.0
    for i, (k, v) in enumerate(rows):
        bg = header_bg if i == 0 else ("#EEF2FF" if i % 2 == 0 else "white")
        fc = "white" if i == 0 else C_MID
        fw = "bold"  if i == 0 else "normal"
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y - row_h), 1.0, row_h,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=bg, transform=ax.transAxes, clip_on=False))
        ax.text(0.01,          y - row_h * 0.35, k, transform=ax.transAxes,
                fontsize=fs, fontweight=fw, color=fc, va="center")
        ax.text(col_w[0] + 0.01, y - row_h * 0.35, v, transform=ax.transAxes,
                fontsize=fs, color=fc, va="center")
        y -= row_h
    ax.set_xlim(0, 1); ax.set_ylim(y - 0.02, 1.01)


def wrap(text, width=88):
    return "\n".join(textwrap.wrap(text, width))


# ═════════════════════════════════════════════════════════════════════════════

with PdfPages(str(OUT)) as pdf:

    # ── PAGE 1: Title & Project Overview ─────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    page(fig, "MSCR Tracker — Session Handoff Document",
         subtitle="Intel RealSense D435i  ·  Real-time 3-D Rod Segmentation & Tip Tracking")

    # Accent bar
    fig.add_axes([0.07, 0.885, 0.86, 0.006]).set_axis_off()
    fig.axes[-1].add_patch(mpatches.FancyBboxPatch(
        (0,0), 1, 1, boxstyle="square,pad=0",
        facecolor=C_BLUE, transform=fig.axes[-1].transAxes))

    ax = fig.add_axes([0.07, 0.08, 0.86, 0.79])
    ax.axis("off")

    overview = [
        (0, "Project Goal"),
        (1, "Segment, localize, and track a black Magnetic Soft Continuum Robot (MSCR)"),
        (1, "in real-time using an Intel RealSense D435i depth camera."),
        (1, "Output: 3-D arc length (mm) and tip (X, Y, Z) in the camera frame."),
        (0, ""),
        (0, "Repository Layout   (/home/dozie/mscr_calibration/)"),
        (1, "mscr_tracker.py              — main tracker (1 083 lines, all logic here)"),
        (1, "mscr_performance_report.py   — standalone PDF analytics generator"),
        (1, "generate_handoff.py          — this handoff document generator"),
        (1, "config.py                    — camera/board constants"),
        (1, "calibrate.py                 — intrinsic calibration pipeline"),
        (1, "capture.py                   — RealSense frame capture utility"),
        (1, "results/calibration.yaml     — calibrated K matrix + distortion coeffs"),
        (1, "results/calibration_report.pdf  — full calibration report"),
        (0, ""),
        (0, "Hardware"),
        (1, "Camera  : Intel RealSense D435i  (USB 3, /dev/video*)"),
        (1, "Default stream  : 848 × 480 @ 30 fps (colour BGR8 + depth Z16, aligned)"),
        (1, "Emitter : off by default (--power 0) — passive IR to avoid blooming"),
        (1, "Rod     : black, slender, highly flexible cylindrical MSCR"),
        (0, ""),
        (0, "Calibration (already done — do NOT re-run unless camera is moved)"),
        (1, "RMS reprojection error : 0.734 px  (excellent; < 1.0 px is acceptable)"),
        (1, "Intrinsics (1280×720): fx=892.08, fy=893.93, cx=646.65, cy=367.62"),
        (1, "Distortion (k1..k3, p1, p2): [0.108, -0.463, 0.007, 0.006, 0.427]"),
        (1, "Depth scale : 0.001 m/unit  (1 mm per raw depth unit)"),
        (2, "Intrinsics are fetched live from rs2 stream — calibration.yaml is reference only"),
        (0, ""),
        (0, "Quick-start Commands"),
        (1, "python mscr_tracker.py --debug               # live view, no report"),
        (1, "python mscr_tracker.py --debug --report      # live view + PDF on exit"),
        (1, "python mscr_tracker.py --debug --report my.pdf --threshold 60"),
        (1, "python mscr_performance_report.py --frames 300 --out perf.pdf"),
    ]
    ruled_text(ax, overview, lh=0.055, fs=8.8)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 2: Pipeline Architecture ────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    page(fig, "Pipeline Architecture",
         subtitle="process_frame(color_bgr, depth_filtered, depth_raw) → TrackResult")

    ax = fig.add_axes([0.05, 0.06, 0.90, 0.84])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    steps = [
        ("1",  "Raw depth snapshot",
         "depth_raw = asarray(depth_frame)  before any post-processing.\n"
         "Zero-depth pixels (rod absorbs IR → no return) are captured here.\n"
         "Used to build rod_holes mask in Step 2.",
         C_TEAL),
        ("2",  "Depth filtering + gate mask",
         "rs.spatial_filter + rs.temporal_filter applied to depth_frame.\n"
         "depth_gate(): pixels confirmed outside [depth_min, depth_max] → excluded.\n"
         "rod_holes (from raw, Step 1) force-OR'd into gate → rod never gated out.",
         C_BLUE),
        ("3",  "Dark-pixel segmentation",
         "BGR→Gray; THRESH_BINARY_INV at dark_threshold (default 55).\n"
         "Otsu fallback on clipped histogram when too few dark pixels found.\n"
         "mask = dark AND gate.",
         C_BLUE),
        ("4",  "Morphological cleanup + CC selection",
         "MORPH_OPEN (1 iter) → MORPH_CLOSE (2 iter) on 3×3 ellipse SE.\n"
         "_largest_elongated_cc(): fitEllipse on each CC; prefer highest eccentricity\n"
         "(threshold 0.80) and area in [min_area_px, 5% of frame].",
         C_BLUE),
        ("5",  "Skeletonisation + branch pruning",
         "skimage.morphology.skeletonize → 1-px-wide centerline.\n"
         "_prune_skeleton_branches(): remove arms < 8 px reaching a junction (3 passes).\n"
         "Result: clean unbranched curve.",
         C_AMB),
        ("6",  "Endpoint detection + ordered path",
         "_find_endpoints(): degree-1 pixels; fallback to most-distant pair.\n"
         "_assign_base_tip(): uses smooth_tip_px prior (EMA) to prevent base/tip swap;\n"
         "falls back to entry-border heuristic (default: bottom=base).\n"
         "_order_path(): greedy nearest-neighbour walk base→tip (no DFS backtracking).",
         C_AMB),
        ("7",  "3-D projection + spline",
         "rs2_deproject_pixel_to_point() for each skeleton pixel using rs2 intrinsics.\n"
         "np.interp fills zero-depth gaps per axis.\n"
         "splprep (k=3) → arc-length-uniform resample to n_resample=200 pts.",
         C_RED),
        ("8",  "Arc length + tip extraction",
         "arc_mm = sum(||P_{i+1}-P_i||) × 1000.  tip_xyz_mm = last point × 1000.\n"
         "EMA (alpha=0.25) smooths tip and arc across frames.\n"
         "Spike rejection: |tip_jump| > max_jump_mm=30 → hold previous estimate.",
         C_RED),
    ]

    y = 9.6
    for num, title, desc, color in steps:
        # Box
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.1, y - 0.82), 9.8, 0.88,
            boxstyle="round,pad=0.05", linewidth=0.8,
            edgecolor=color, facecolor=color + "18"))
        # Step number badge
        ax.add_patch(plt.Circle((0.55, y - 0.38), 0.22,
                                color=color, zorder=3))
        ax.text(0.55, y - 0.38, num, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=4)
        ax.text(1.1, y - 0.20, title,
                fontsize=8.5, fontweight="bold", color=color, va="top")
        ax.text(1.1, y - 0.44, desc,
                fontsize=7.5, color=C_MID, va="top", linespacing=1.35)
        y -= 1.10

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 3: Key Classes & API ─────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    page(fig, "Key Classes & Public API")

    axL = fig.add_axes([0.05, 0.06, 0.43, 0.84])
    axR = fig.add_axes([0.52, 0.06, 0.43, 0.84])

    left = [
        (0, "TrackerParams  (dataclass)"),
        (1, "width / height / fps"),
        (2, "848×480 @ 30 — stream resolution"),
        (1, "laser_power = 0"),
        (2, "emitter off; set >0 for active IR"),
        (1, "depth_min_m / depth_max_m"),
        (2, "0.05 – 0.80 m depth gate"),
        (1, "depth_dilation_px = 6"),
        (2, "erode exclusion mask near rod edges"),
        (1, "dark_threshold = 55"),
        (2, "inverse threshold (lower = darker only)"),
        (1, "min_eccentricity = 0.80"),
        (2, "CC eccentricity gate; rods ≈ 0.90+"),
        (1, "min_skel_pts / max_skel_pts"),
        (2, "15 – 1000 valid skeleton pixels"),
        (1, "min_valid_depth_frac = 0.02"),
        (2, "≥2% of skel pts must have depth"),
        (1, "n_resample = 200"),
        (2, "final spline sample count"),
        (1, "spline_smooth = 2.0"),
        (2, "splprep s = N × smooth_factor"),
        (1, "entry = 'bottom'"),
        (2, "'top'|'bottom'|'left'|'right'"),
        (1, "roi_pad_px = 60"),
        (2, "padding added to temporal ROI"),
        (1, "max_lost_frames = 8"),
        (2, "before ROI resets to full frame"),
        (1, "ema_alpha = 0.25"),
        (2, "tip/arc smoothing rate"),
        (1, "max_jump_mm = 30.0"),
        (2, "spike rejection threshold (mm)"),
        (1, "debug = False"),
        (2, "enable side-by-side debug window"),
        (0, ""),
        (0, "MSCRTracker  (class)"),
        (1, "__init__(params)"),
        (1, "set_intrinsics(rs.intrinsics)"),
        (2, "must call before process_frame"),
        (1, "process_frame(color, depth,"),
        (2, "  depth_raw=None) → TrackResult"),
        (1, "run(report_path=None)  → generator"),
        (2, "yields TrackResult per frame"),
        (2, "writes PDF on exit if report_path set"),
    ]
    ruled_text(axL, left, lh=0.052, fs=8.3)

    right = [
        (0, "TrackResult  (dataclass)"),
        (1, "valid : bool"),
        (2, "True = 3-D arc+tip computed"),
        (2, "False + centerline_px non-empty"),
        (2, "  → skeleton visible, no 3-D depth"),
        (2, "False + centerline_px empty → lost"),
        (1, "centerline_px : (N,2) float32"),
        (2, "(col, row) full-image coords"),
        (1, "centerline_3d : (N,3) float64"),
        (2, "(X, Y, Z) in mm, camera frame"),
        (1, "arc_length_mm : float"),
        (2, "EMA-smoothed 3-D arc length"),
        (1, "tip_px : (col, row)"),
        (2, "2-D tip position"),
        (1, "base_px : (col, row)"),
        (2, "2-D base position"),
        (1, "tip_xyz_mm : (X, Y, Z)"),
        (2, "EMA-smoothed 3-D tip in mm"),
        (1, "mid_xyz_mm : (X, Y, Z)"),
        (2, "50% arc-length point in mm"),
        (1, "debug_frame : ndarray | None"),
        (2, "side-by-side BGR image for imshow"),
        (0, ""),
        (0, "Terminal Status Lines"),
        (1, "arc=... mm | tip=(X,Y,Z) mm"),
        (2, "valid=True, full 3-D tracking"),
        (1, "[skeleton Npx — no 3D depth]"),
        (2, "rod segmented; depth unavailable"),
        (1, "[no detection]"),
        (2, "rod not found in frame"),
        (0, ""),
        (0, "Coordinate Conventions"),
        (1, "2-D : (col, row) = OpenCV (u, v)"),
        (1, "3-D : RealSense camera frame"),
        (2, "X = right,  Y = down,  Z = forward"),
        (1, "Units : metres internally, mm output"),
    ]
    ruled_text(axR, right, lh=0.052, fs=8.3)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 4: Bugs Fixed & Design Decisions ─────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    page(fig, "Bugs Fixed & Non-Obvious Design Decisions",
         subtitle="Critical context — do not revert these without understanding the rationale")

    ax = fig.add_axes([0.05, 0.06, 0.90, 0.84])

    bugs = [
        (0, "BUG (fixed) — spatial_filter erases the rod"),
        (1, "rs.spatial_filter fills zero-depth holes (where rod absorbs IR) with"),
        (1, "interpolated background depth values. After filtering those pixels carry"),
        (1, "background-range depths → depth gate excludes them → rod vanishes from mask."),
        (1, "FIX: depth_raw captured BEFORE filters. rod_holes=(raw==0) is OR'd into"),
        (1, "the gate unconditionally so the rod silhouette always survives filtering."),
        (0, ""),
        (0, "BUG (fixed) — cv2.erode on empty ROI array crashes"),
        (1, "When temporal bbox drifts to frame edge the ROI slice becomes zero-size."),
        (1, "FIX: _depth_gate() guards depth.size==0 early return; also checks ROI"),
        (1, "dimensions against structuring element size before calling erode."),
        (1, "process_frame() resets bbox to full frame when rx2<=rx1 or ry2<=ry1."),
        (0, ""),
        (0, "BUG (fixed) — base/tip endpoint swap when the rod bends"),
        (1, "Anchoring on the MOVING tip flips base/tip when the rod bends sharply."),
        (1, "FIX: anchor on the STATIC base instead. _anchor_base_px is a slow EMA"),
        (1, "(base_ema_alpha=0.10) of the fixed mount; _assign_base_tip() calls the"),
        (1, "endpoint nearest that anchor the base. The base barely moves, so it"),
        (1, "cannot flip. Entry-border heuristic is bootstrap-only; anchor clears on"),
        (1, "a full-frame reset so a repositioned fixture re-acquires cleanly."),
        (0, ""),
        (0, "BUG (fixed) — DFS ordering detours through skeleton side-branches"),
        (1, "skimage.skeletonize() produces short side-spurs at blob irregularities."),
        (1, "DFS could backtrack and traverse a spur, producing a non-monotone path."),
        (1, "FIX: _prune_skeleton_branches() removes arms <8px reaching a junction"),
        (1, "(3 passes). _order_path() replaced with greedy nearest-neighbour walk"),
        (1, "that cannot backtrack and always advances toward the target endpoint."),
        (0, ""),
        (0, "DESIGN — min_valid_depth_frac = 0.02 (intentionally very low)"),
        (1, "The black rod absorbs nearly all structured IR light. Even with emitter on,"),
        (1, "most skeleton pixels return depth=0. Linear interpolation from the tiny"),
        (1, "fraction with real returns gives usable 3-D shape; the spline smooths it."),
        (1, "Raising this threshold above ~0.05 will cause constant 3-D detection failure."),
        (0, ""),
        (0, "DESIGN — EMA gate display (alpha=0.35) for right debug panel"),
        (1, "Raw depth gate re-rendered per frame produces per-pixel flicker even with"),
        (1, "temporal depth filtering. Gate image is blended across frames (35/65 mix)"),
        (1, "before applyColorMap. This is visualisation-only; the actual gate mask"),
        (1, "used for segmentation is computed fresh each frame (no smoothing there)."),
    ]
    ruled_text(ax, bugs, lh=0.051, fs=8.6)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 4b: Robustness — flips, ROI, hands ───────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    page(fig, "Robustness: Tip-flip, ROI Explosion & Hand Intrusion",
         subtitle="Latest hardening pass — manual manipulation with hands in the frame")

    ax = fig.add_axes([0.05, 0.06, 0.90, 0.84])
    robust = [
        (0, "Tip / base flip  →  static base anchor"),
        (1, "_anchor_base_px : slow EMA (base_ema_alpha=0.10) of the fixed mount."),
        (1, "_assign_base_tip() classifies the endpoint nearest the anchor as base."),
        (1, "base_lock_dist_px=120 : detection rejected if nearest endpoint is farther."),
        (0, ""),
        (0, "ROI explosion  →  plausibility guard + incremental widening"),
        (1, "_roi_plausible() compares bbox DIAGONAL span (not area — a thin rod's"),
        (1, "area collapses to ~0 when vertical/horizontal, giving false explosions)."),
        (1, "Rejects bbox if span ratio > roi_max_growth (3.0) or centre shift >"),
        (1, "roi_max_shift_px (140). On reject → _soft_fail() HOLDS the last good"),
        (1, "curve (no spike, no hard LOST). When lost, the ROI grows by"),
        (1, "roi_expand_px (30) per frame instead of snapping to full frame."),
        (0, ""),
        (0, "Hand / arm intrusion  →  slender isolation + base walk"),
        (1, "_isolate_slender(): morphological OPEN with a kernel of diameter"),
        (1, "rod_max_width_px+3 (~15 px) — between rod width (≤12) and finger width"),
        (1, "(≥18). OPEN keeps blobs WIDER than the kernel (the hand); subtracting"),
        (1, "that from the mask leaves only the slender rod. CRITICAL: the kernel"),
        (1, "must be SMALLER than a finger or fingers leak back into the thin mask."),
        (1, "_largest_elongated_cc(base_anchor_roi): among elongated candidates,"),
        (1, "picks the component NEAREST the base anchor — walks down the rod from"),
        (1, "its fixed base, ignoring any attached hand component."),
        (1, "hand_thick_frac=0.30 sets self._hand_present when >30% of fg is thick."),
        (0, ""),
        (0, "Hand-present adaptive behaviour (self._hand_present)"),
        (1, "ROI guards relax ×hand_relax_factor (2.5) — manual motion is larger."),
        (1, "Depth-proxy search widens to depth_search_r_hand (14) — a hand perturbs"),
        (1, "depth near the rod, so a bigger neighbourhood is needed for valid depth."),
        (1, "depth_dilation_px lowered 6→3 so the gate erosion doesn't eat thin rod"),
        (1, "data at boundaries when a hand is adjacent."),
        (0, ""),
        (0, "Expected behaviour under occlusion"),
        (1, "When a finger crosses the rod, the visible rod is segmented up to the"),
        (1, "occlusion; the 3-D spike guard HOLDS the last tip (valid=False, curve"),
        (1, "still drawn) rather than jumping. Full LOST only if the base region"),
        (1, "itself is fully covered — recovers within max_lost_frames (8)."),
    ]
    ruled_text(ax, robust, lh=0.052, fs=8.4)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 5: Tuning Guide ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    page(fig, "Tuning Guide — What to Change When")

    axL = fig.add_axes([0.05, 0.06, 0.42, 0.84])
    axR = fig.add_axes([0.53, 0.06, 0.42, 0.84])

    tuning_rows = [
        ("Symptom", "Parameter / fix"),
        ("Rod not detected at all",
         "Lower dark_threshold (try 65–80);\ncheck depth_min_m not too high"),
        ("Hand grabs the rod /\ntracking lost on touch",
         "slender_isolation on; tune rod_max_width_px\n(< finger width); base anchor walks rod"),
        ("Fingers leak into mask",
         "rod_max_width_px too large — lower it so\nkernel stays below finger width"),
        ("Centerline jumps/flips",
         "EMA already active; check entry= setting;\ninspect _smooth_tip_px reset"),
        ("Skeleton too short",
         "Lower min_skel_pts; increase morph_close_iter"),
        ("[skeleton — no 3D depth]\nconstant",
         "Lower min_valid_depth_frac (<0.02);\ntry --power 150 (enable emitter)"),
        ("3-D arc length noisy",
         "Lower ema_alpha (more smoothing);\nraise n_resample; raise spline_smooth"),
        ("Tip jumps large",
         "Lower max_jump_mm; lower ema_alpha"),
        ("Slow / low FPS",
         "Switch to 848×480 (default);\nraise min_skel_pts to skip small blobs"),
        ("Rod exits frame\nand re-enters",
         "max_lost_frames=8 resets ROI;\nroi_pad_px=60 gives search margin"),
        ("Rod near frame edge\ngets clipped",
         "Raise roi_pad_px;\ncheck depth_dilation_px not too large"),
    ]
    two_col_table(axL, tuning_rows, row_h=0.073, fs=8.0)

    right_tun = [
        (0, "Recommended Workflow for New Scene"),
        (1, "1. Run with --debug and observe right panel"),
        (2, "Is the rod region black (zero depth) or coloured?"),
        (1, "2. If rod not segmenting: lower dark_threshold"),
        (2, "Try 60, 65, 70 in steps"),
        (1, "3. If false positives: raise min_eccentricity"),
        (2, "Inspect CC eccentricity with print statements"),
        (1, "4. If 3-D always fails: lower min_valid_depth_frac"),
        (2, "0.01 is safe lower bound given spline interpolation"),
        (1, "5. If tip unstable: tune ema_alpha"),
        (2, "0.15 = heavy smoothing, 0.40 = fast response"),
        (0, ""),
        (0, "Environment Variables / Paths"),
        (1, "Working dir   : /home/dozie/mscr_calibration/"),
        (1, "Python env    : system Python3, no venv detected"),
        (1, "Key deps      : pyrealsense2, opencv-python,"),
        (2, "scipy, scikit-image, numpy, matplotlib"),
        (1, "Camera rules  : 99-realsense-libusb.rules"),
        (2, "(copy to /etc/udev/rules.d/ if device not seen)"),
        (0, ""),
        (0, "Known Limitations / Open Tasks"),
        (1, "No extrinsic calibration (camera→robot base)"),
        (2, "tip_xyz_mm is in camera frame, not robot frame"),
        (1, "No handling of rod occlusion by hand/fixture"),
        (1, "Entry border heuristic fails for horizontal rod"),
        (2, "Use external base-point prior if available"),
        (1, "Emitter-off mode gives sparse depth on rod"),
        (2, "Emitter-on may help but increases IR noise"),
        (1, "Performance report requires matplotlib + camera"),
        (2, "Cannot be generated from saved video yet"),
    ]
    ruled_text(axR, right_tun, lh=0.056, fs=8.4)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 6: Next Steps ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    page(fig, "Next Steps & Integration Notes")

    ax = fig.add_axes([0.05, 0.06, 0.90, 0.84])

    nextsteps = [
        (0, "Immediate: verify 3-D accuracy"),
        (1, "Hold the rod at a known distance (e.g. 300 mm from camera lens)."),
        (1, "Check tip_xyz_mm.Z ≈ 300 ± 5 mm. If not, verify depth_scale from calibration.yaml."),
        (1, "Measure known arc length manually and compare to arc_length_mm output."),
        (0, ""),
        (0, "Extrinsic Calibration (camera→robot-base transform)"),
        (1, "Needed to express tip_xyz_mm in the robot workspace frame."),
        (1, "Approach: fix a calibration target at a known robot position, observe it"),
        (1, "in the camera frame, solve for R, t using cv2.solvePnP or hand-eye calibration."),
        (1, "Store result in results/extrinsic.yaml alongside calibration.yaml."),
        (0, ""),
        (0, "Control Integration"),
        (1, "TrackResult.tip_xyz_mm provides (X, Y, Z) in mm at 30 fps."),
        (1, "Consume the run() generator in a control loop:"),
        (2, "for result in tracker.run():"),
        (2, "    if result.valid:"),
        (2, "        send_to_controller(result.tip_xyz_mm, result.arc_length_mm)"),
        (1, "EMA smoothing (ema_alpha=0.25) already applied — additional filtering optional."),
        (1, "result.centerline_3d is (200, 3) float64 — full shape curve for shape control."),
        (0, ""),
        (0, "Performance Benchmarking"),
        (1, "python mscr_tracker.py --debug --report my_session.pdf"),
        (1, "Report auto-generated on Q/ESC exit. Contains:"),
        (2, "Detection rate, latency, arc length distribution, tip jitter (5 pages)"),
        (1, "Standalone: python mscr_performance_report.py --frames 300 --out perf.pdf"),
        (0, ""),
        (0, "Regenerating This Document"),
        (1, "python generate_handoff.py"),
        (1, "Output: mscr_handoff.pdf in /home/dozie/mscr_calibration/"),
        (1, "Edit generate_handoff.py to add new findings before running."),
    ]
    ruled_text(ax, nextsteps, lh=0.055, fs=8.8)

    # Footer
    fig.text(0.5, 0.025,
             "Generated by generate_handoff.py  ·  /home/dozie/mscr_calibration/",
             ha="center", fontsize=7.5, color=C_LIGHT)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    d = pdf.infodict()
    d["Title"]   = "MSCR Tracker Handoff Document"
    d["Author"]  = "generate_handoff.py"
    d["Subject"] = "Context document for continuing the MSCR tracking project"

print(f"Handoff PDF written → {OUT}")
