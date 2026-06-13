#!/usr/bin/env python3
"""
generate_pipeline_doc.py — produce mscr_shape_pipeline.pdf, an end-to-end
explainer of the mscr_shape system: data gathering -> labeling -> training ->
inference, and how each stage maps onto the MoSSNet architecture.

Run:   python generate_pipeline_doc.py
Output: mscr_shape_pipeline.pdf  (same directory)
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).parent
OUT = HERE / "mscr_shape_pipeline.pdf"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_DARK, C_MID, C_LIGHT = "#212121", "#424242", "#757575"
C_BLUE, C_TEAL, C_RED, C_AMB, C_GRN = "#1565C0", "#00695C", "#B71C1C", "#E65100", "#2E7D32"


def header(fig, title, subtitle=""):
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.95, title, ha="left", va="top",
             fontsize=16, fontweight="bold", color=C_DARK)
    if subtitle:
        fig.text(0.06, 0.915, subtitle, ha="left", va="top",
                 fontsize=9.5, color=C_LIGHT)
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.905, 0.905], color="#E0E0E0", lw=1))


def footer(fig, page):
    fig.text(0.94, 0.04, f"mscr_shape · {page}", ha="right", va="bottom",
             fontsize=7.5, color=C_LIGHT)


def textblock(ax, lines, x=0.0, y=1.0, lh=0.045, fs=9.0):
    """lines: list of (level, text). level 0=heading, 1=body, 2=sub-bullet, 3=note."""
    ax.axis("off")
    cur = y
    for level, text in lines:
        if level == 0:
            cur -= 0.012
            ax.text(x, cur, text, transform=ax.transAxes, fontsize=fs + 2.0,
                    fontweight="bold", color=C_BLUE, va="top")
            cur -= lh * 1.15
        elif level == 1:
            ax.text(x, cur, text, transform=ax.transAxes, fontsize=fs,
                    color=C_DARK, va="top")
            cur -= lh
        elif level == 2:
            ax.text(x + 0.03, cur, "•  " + text, transform=ax.transAxes,
                    fontsize=fs - 0.5, color=C_MID, va="top")
            cur -= lh * 0.92
        else:
            ax.text(x + 0.01, cur, text, transform=ax.transAxes, fontsize=fs - 1.0,
                    color=C_LIGHT, va="top", style="italic")
            cur -= lh * 0.92
    return cur


def box(ax, xy, wh, label, fc, ec=None, fs=8.5, tc="white"):
    x, y = xy; w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                                fc=fc, ec=ec or fc, lw=1.2, transform=ax.transAxes))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            transform=ax.transAxes, fontsize=fs, color=tc, fontweight="bold")


def arrow(ax, p0, p1, color=C_MID):
    ax.add_patch(FancyArrowPatch(p0, p1, transform=ax.transAxes, arrowstyle="-|>",
                                 mutation_scale=12, lw=1.4, color=color))


# ─────────────────────────────────────────────────────────────────────────────
def page_overview(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    header(fig, "Monocular 3D shape sensing for an MSCR",
           "D435 stereo-IR ground truth · MoSSNet-style ResNet18 inference · units = mm in the left-IR rectified frame")

    ax = fig.add_axes([0.06, 0.40, 0.88, 0.46]); ax.axis("off")
    textblock(ax, [
        (0, "The problem"),
        (1, "A magnetic soft continuum robot (MSCR) is a ~1 mm black rod that bends elastically"),
        (1, "under an external magnet. We want its full 3D centerline r(s) from a SINGLE camera"),
        (1, "view, fast enough for closed-loop use. A monocular image is ambiguous in depth, so a"),
        (1, "network must learn the rod's shape prior from data."),
        (0, "The approach — two cameras to teach one"),
        (1, "The Intel D435 has a calibrated STEREO INFRARED pair. We use BOTH IR views only to"),
        (1, "build accurate 3D ground-truth labels (triangulation). The trained network then infers"),
        (1, "3D shape from the LEFT IR image ALONE — at inference the right view is never used."),
        (3, "RGB camera and the IR dot projector are disabled throughout; the rod is the only thing imaged."),
        (0, "Five phases (this document follows them end to end)"),
    ], y=1.0)

    ax2 = fig.add_axes([0.06, 0.10, 0.88, 0.26]); ax2.axis("off")
    ys = 0.80
    names = [("0 · Calibrate", "stereo IR intrinsics\n+ rectification", C_LIGHT),
             ("1 · Capture", "record rod video\n(left+right IR)", C_TEAL),
             ("2 · Label", "stereo triangulate\n-> 3D r(s) + QC", C_BLUE),
             ("3 · Dataset", "pair image<->r(s),\nconfig-split", C_MID),
             ("4 · Train", "MoSSNet: image\n-> r(s) + length", C_AMB),
             ("5 · Infer", "live left-IR ->\n3D + ONNX", C_RED)]
    w = 0.135; gap = (1.0 - 6 * w) / 5
    for i, (t, s, c) in enumerate(names):
        x = i * (w + gap)
        box(ax2, (x, ys), (w, 0.16), t, c, fs=8.2)
        ax2.text(x + w / 2, ys - 0.03, s, ha="center", va="top",
                 transform=ax2.transAxes, fontsize=6.6, color=C_MID)
        if i < 5:
            arrow(ax2, (x + w + 0.005, ys + 0.08), (x + w + gap - 0.005, ys + 0.08))
    ax2.text(0.0, 0.45, "Phases 0–1 are hardware setup; 2–3 manufacture training data; 4–5 are\n"
             "the learned model. The data quality in 2–3 dominates the final accuracy.",
             transform=ax2.transAxes, fontsize=8.2, color=C_MID, va="top")

    footer(fig, "overview")
    pdf.savefig(fig); plt.close(fig)


def page_data(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    header(fig, "1 · Data gathering",
           "capture.py — synchronized left/right IR frames of the rod under hand-held magnet actuation")

    ax = fig.add_axes([0.06, 0.46, 0.88, 0.42])
    textblock(ax, [
        (0, "What is recorded"),
        (1, "capture.py streams the D435 IR pair (y8, 1280×720 @30fps) with the emitter OFF, and"),
        (1, "writes raw/<session>/{left,right}/<frame>.png + a manifest. The operator sweeps a"),
        (1, "magnet to bend the rod through many shapes while it records."),
        (0, "Hard-won capture constraints (the rod must be ALL of these at once)"),
        (2, "Distance 15–30 cm. Closer than ~13 cm → degenerate stereo (garbage depth);"),
        (2, "farther than ~40 cm → too little disparity. VARY distance within the band for Z range."),
        (2, "Mostly VERTICAL in frame. A horizontal rod runs parallel to the epipolar scanlines,"),
        (2, "so every sample is degenerate and triangulation fails."),
        (2, "Base CLAMPED & unoccluded. A hand-held base occludes the rod differently each frame,"),
        (2, "so the measured length drifts between sessions — inconsistent label noise."),
        (0, "USB gotcha"),
        (1, "At USB 2.0 the D435 cannot do 720p IR @30 (\"Couldn't resolve requests\"). Must be on a"),
        (1, "USB-3 port with a USB-3 cable; many USB-C cables are charge/USB-2 only."),
        (0, "Sessions used for the current model (clamped base)"),
    ], y=1.0)

    # sessions table
    axt = fig.add_axes([0.06, 0.22, 0.88, 0.20]); axt.axis("off")
    rows = [["session", "frames", "accepted", "median length", "note"],
            ["s09", "900", "90.4 %", "45.9 mm", "clamped, varied bend"],
            ["s10", "1030", "100 %", "44.6 mm", "clamped"],
            ["s11", "917", "100 %", "45.0 mm", "clamped"],
            ["s03", "—", "(parked)", "55 mm", "OLD hand-held; length inconsistent → excluded"]]
    tbl = axt.table(cellText=rows, cellLoc="left", loc="upper left",
                    colWidths=[0.13, 0.12, 0.13, 0.18, 0.44])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.2); tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#E0E0E0")
        if r == 0:
            cell.set_facecolor(C_BLUE); cell.set_text_props(color="white", fontweight="bold")
        elif rows[r][0] == "s03":
            cell.set_text_props(color=C_LIGHT)
    axt.text(0.0, -0.12, "Clamping made the rod's measured length agree across sessions (~45 mm)\n"
             "— the key that lets sessions be COMBINED without hurting the model.",
             transform=axt.transAxes, fontsize=8.0, color=C_MID, va="top")

    footer(fig, "data gathering")
    pdf.savefig(fig); plt.close(fig)


def page_labeling(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    header(fig, "2 · Labeling — stereo ground truth",
           "labeler.py — turn each left/right IR pair into a metric 3D centerline r(s), with QC")

    ax = fig.add_axes([0.06, 0.52, 0.88, 0.36])
    textblock(ax, [
        (0, "Per-pair pipeline"),
        (2, "Rectify both views (calibration maps) so epipolar lines are image rows."),
        (2, "Segment the rod (black-hat morphology → thin dark features, vignette-invariant)."),
        (2, "Pick the rod component, skeletonize, order pixels base→tip."),
        (2, "Smoothing-spline + subpixel refine each 2D curve; match left↔right on scanlines."),
        (2, "Triangulate correspondences (P1,P2) → 3D points; robust depth/jump filtering."),
        (2, "3D smoothing spline, uniform-arclength resample → r(s) of N=64 points."),
        (2, "QC: reproject into both views; accept iff mean reproj error < 1.5 px in BOTH."),
        (0, "The clutter problem and its fix (base-anchor region-grow)"),
        (1, "Clamping put rigid background edges in view (table lip, board edge, door molding) that"),
        (1, "out-scored the short rod on elongation — the labeler traced the wrong line. Because the"),
        (1, "clamp base is FIXED, the operator clicks it once per session (labeler.py --set-base);"),
        (1, "the labeler then keeps the connected component CONTAINING that anchor. Clutter, hand and"),
        (1, "window are separate components, so they are ignored — robust where heuristics failed."),
    ], y=1.0)

    # embed a QC overlay if present
    axi = fig.add_axes([0.06, 0.10, 0.88, 0.36]); axi.axis("off")
    qc = None
    for cand in [HERE / "data/qc/s11/000000.jpg", HERE / "data/qc/s10/000067.jpg",
                 HERE / "data/qc/s09/000050.jpg"]:
        if cand.exists():
            qc = cand; break
    if qc is not None:
        axi.imshow(mpimg.imread(str(qc)))
        axi.set_title("QC overlay (left | right): green = detected 2D centerline, "
                      "red = reprojected 3D r(s), banner = accept + length",
                      fontsize=7.8, color=C_MID)
    else:
        axi.text(0.5, 0.5, "(run the labeler to generate QC overlays)", ha="center",
                 color=C_LIGHT, transform=axi.transAxes)

    footer(fig, "labeling")
    pdf.savefig(fig); plt.close(fig)


def page_mossnet(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    header(fig, "4a · The MoSSNet architecture",
           "model.py — one ResNet18 encoder, two decoder heads: centerline points + scalar length")

    # architecture diagram
    ax = fig.add_axes([0.06, 0.56, 0.88, 0.30]); ax.axis("off")
    box(ax, (0.00, 0.45), (0.16, 0.30), "Left-IR\nimage\n1×288×512", C_TEAL, fs=7.6)
    arrow(ax, (0.16, 0.60), (0.22, 0.60))
    box(ax, (0.22, 0.42), (0.24, 0.36), "ResNet18 encoder\n(conv1 → 1-channel)\nlayers up to layer4", C_BLUE, fs=7.8)
    arrow(ax, (0.46, 0.60), (0.52, 0.60))
    box(ax, (0.52, 0.45), (0.16, 0.30), "feature map\n512×9×16", C_MID, fs=7.6)
    # two heads
    arrow(ax, (0.68, 0.66), (0.74, 0.74))
    arrow(ax, (0.68, 0.54), (0.74, 0.46))
    box(ax, (0.74, 0.66), (0.24, 0.22), "point head\nAdaptiveAvgPool(3×4)\n→ N×3 r(s) [mm]", C_AMB, fs=7.2)
    box(ax, (0.74, 0.36), (0.24, 0.20), "length head\nglobal pool\n→ L [mm]", C_RED, fs=7.4)

    ax2 = fig.add_axes([0.06, 0.10, 0.88, 0.42])
    textblock(ax2, [
        (0, "How it maps to MoSSNet"),
        (1, "MoSSNet (monocular soft-robot shape sensing) = a shared CNN encoder that regresses the"),
        (1, "whole centerline as an ordered point set, plus an auxiliary length output. We use a"),
        (1, "ResNet18 backbone with conv1 re-seeded to single-channel (mean of the RGB filters) so it"),
        (1, "ingests one IR channel while keeping ImageNet features."),
        (0, "The spatial-head fix (why bending is now captured)"),
        (1, "A stock ResNet ends in GLOBAL AVERAGE POOLING, which discards WHERE things are — but a"),
        (1, "rod's in-plane bend IS a position. With global pooling the net learned only depth and"),
        (1, "predicted a near-rigid rod. We drop the final avg-pool and feed the point head an"),
        (1, "AdaptiveAvgPool(3×4) grid that PRESERVES coarse position. (3×4 divides the 9×16 map"),
        (1, "evenly so ONNX export of the adaptive pool is valid.) The length head keeps a global"),
        (1, "feature since total length is position-independent."),
        (0, "Output contract"),
        (2, "points: (B, 64, 3) — ordered base→tip centerline in mm, left-IR rectified frame."),
        (2, "length: (B, 1) — total arclength in mm (auxiliary regularizer)."),
    ], y=1.0)

    footer(fig, "architecture")
    pdf.savefig(fig); plt.close(fig)


def page_training(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    header(fig, "4b · Training",
           "train.py — supervise the network with the triangulated r(s) labels")

    ax = fig.add_axes([0.06, 0.50, 0.88, 0.38])
    textblock(ax, [
        (0, "Objective"),
        (1, "Loss = per-axis-normalized MSE on the N×3 points  +  λ · MSE on length."),
        (1, "Each coordinate residual is divided by that axis' TRAIN-set std (train.normalize_targets)"),
        (1, "so the small in-plane X/Y bending is not swamped by the large depth offset — otherwise"),
        (1, "the net nails Z in one epoch and parks at the mean in-plane shape."),
        (0, "Protocol"),
        (2, "Config split: whole tip-position grid CELLS go to train/val/test (not random frames),"),
        (2, "so near-identical poses cannot leak across splits. Photometric augmentation only"),
        (2, "(brightness/contrast/blur) — never geometric, because the labels are metric 3D."),
        (2, "AdamW + cosine LR decay, early stop on val mean error; best-val checkpoint saved."),
        (0, "Result on the clamped data (s09+s10+s11, split 1850 / 501 / 427)"),
        (1, "TEST mean along-body error = 3.04 mm   ·   tip error = 5.31 mm   (best to date)."),
    ], y=1.0)

    # bending correlation bar chart
    axb = fig.add_axes([0.10, 0.12, 0.80, 0.30])
    axes_lbl = ["X (lateral bend)", "Y (along-axis)", "Z (depth)"]
    before = [-0.22, -0.36, 0.86]
    after = [0.99, 0.22, 0.99]
    xpos = np.arange(3); bw = 0.36
    axb.bar(xpos - bw / 2, before, bw, label="before (global pool, s03)", color="#BDBDBD")
    axb.bar(xpos + bw / 2, after, bw, label="after (spatial head, clamped)", color=C_AMB)
    axb.axhline(0, color=C_DARK, lw=0.8)
    axb.axhline(0.7, color=C_GRN, lw=1.0, ls="--")
    axb.text(2.5, 0.72, "success = 0.7", color=C_GRN, fontsize=7.5, va="bottom", ha="right")
    axb.set_xticks(xpos); axb.set_xticklabels(axes_lbl, fontsize=8.5)
    axb.set_ylabel("corr(pred tip, GT tip)"); axb.set_ylim(-0.6, 1.1)
    axb.legend(fontsize=7.8, loc="lower left", frameon=False)
    axb.set_title("Per-axis tip-tracking correlation — bending is now captured (X, Z); Y still weak",
                  fontsize=8.6, color=C_MID)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)

    footer(fig, "training")
    pdf.savefig(fig); plt.close(fig)


def page_inference(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    header(fig, "5 · Inference & deployment",
           "infer.py — left-IR image → 3D r(s) in real time; ONNX export; 3D visualization")

    ax = fig.add_axes([0.06, 0.55, 0.88, 0.33])
    textblock(ax, [
        (0, "Runtime path"),
        (2, "Take ONLY the left-IR rectified image, resize to 288×512, normalize to [0,1]."),
        (2, "Forward pass → (r(s) points in mm, length). The right view is NOT used at inference."),
        (2, "Temporal EMA smoothing on the tip + a max-jump reject gate suppress per-frame jitter."),
        (2, "Overlay the reprojected r(s) on the live image and report FPS."),
        (0, "Portability"),
        (1, "infer.py --export-onnx writes checkpoints/mscr_shape.onnx and verifies torch vs"),
        (1, "onnxruntime agree to ~1e-3. The (3×4) adaptive pool was chosen specifically so this"),
        (1, "export is valid (a 4×4 grid fails to export on the 9×16 feature map)."),
        (0, "Visualization"),
        (1, "infer.py --session <s> --viz3d renders the predicted 3D centerline (green), base"),
        (1, "(black) and tip (red) with an accumulating tip trail, and dumps r_s_seq [T,64,3]."),
    ], y=1.0)

    # render a 3D frame + tip trail from a viz npz if available
    axn = None
    for cand in [HERE / "viz3d_s11_clamped.npz", HERE / "viz3d_s10_full.npz",
                 HERE / "viz3d_s09_full.npz", HERE / "viz3d_s03_full.npz"]:
        if cand.exists():
            axn = cand; break
    ax3 = fig.add_axes([0.10, 0.10, 0.80, 0.38], projection="3d")
    if axn is not None:
        d = np.load(axn)
        seq, tips = d["r_s_seq"], d["tips"]
        mid = seq[len(seq) // 2]
        ax3.plot(seq[len(seq)//2][:, 0], mid[:, 1], mid[:, 2], color=C_GRN, lw=2.5,
                 label="predicted r(s)")
        ax3.scatter(*mid[0], color="black", s=30)
        ax3.plot(tips[:, 0], tips[:, 1], tips[:, 2], color=C_RED, lw=0.8, alpha=0.7,
                 label="tip trajectory")
        ax3.set_xlabel("X (mm)", fontsize=7); ax3.set_ylabel("Y (mm)", fontsize=7)
        ax3.set_zlabel("Z (mm)", fontsize=7)
        ax3.legend(fontsize=7.5, loc="upper left")
        ax3.set_title(f"Predicted 3D centerline + swept tip path  ({axn.name})",
                      fontsize=8.4, color=C_MID)
        ax3.tick_params(labelsize=6)
    else:
        ax3.text2D(0.5, 0.5, "(run infer.py --viz3d to populate)", ha="center",
                   transform=ax3.transAxes, color=C_LIGHT)

    footer(fig, "inference")
    pdf.savefig(fig); plt.close(fig)


def page_tieback(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    header(fig, "How it ties back to MoSSNet",
           "one learned encoder, stereo-taught labels, position-aware decoding")

    ax = fig.add_axes([0.06, 0.30, 0.88, 0.58])
    textblock(ax, [
        (0, "The throughline"),
        (1, "MoSSNet's premise is that a single image of a slender body carries enough cues (apparent"),
        (1, "curvature, foreshortening, thickness) to recover its 3D shape IF the network has learned"),
        (1, "the body's deformation prior. Every phase here serves that premise:"),
        (2, "Stereo IR (phases 0–2) MANUFACTURES the 3D supervision a monocular net cannot get itself."),
        (2, "Clamping + base-anchor labeling make that supervision CONSISTENT, so the prior is clean."),
        (2, "The ResNet18 encoder is the shared MoSSNet trunk; the point head is its centerline"),
        (2, "regressor; the length head is the auxiliary geometric constraint."),
        (2, "Keeping spatial features (not global pooling) is what lets the prior express BENDING,"),
        (2, "not just depth — the difference between a rigid-looking and an elastic reconstruction."),
        (0, "Where it stands"),
        (1, "X (lateral) and Z (depth) tip tracking are near-perfect (corr +0.99) with matching span."),
        (1, "Remaining work: the Y (camera-vertical / along-axis) tip motion is still under-tracked"),
        (1, "(corr +0.22) — likely an observability limit; needs data that bends in that plane and/or"),
        (1, "a tighter high-res ROI crop around the rod."),
        (0, "Repro: the whole pipeline in one column"),
        (3, "calib.py --from-device → capture.py --session sNN → labeler.py --session sNN --set-base"),
        (3, "→ labeler.py --session sNN → dataset.py → train.py → infer.py --session sNN [--viz3d]"),
    ], y=1.0)

    footer(fig, "synthesis")
    pdf.savefig(fig); plt.close(fig)


def main():
    with PdfPages(OUT) as pdf:
        page_overview(pdf)
        page_data(pdf)
        page_labeling(pdf)
        page_mossnet(pdf)
        page_training(pdf)
        page_inference(pdf)
        page_tieback(pdf)
        meta = pdf.infodict()
        meta["Title"] = "mscr_shape — pipeline & MoSSNet explainer"
        meta["Author"] = "mscr_shape"
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
