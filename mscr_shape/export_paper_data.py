#!/usr/bin/env python3
"""
export_paper_data.py — dump all meaningful mscr_shape results to an Excel
workbook + matching PNG plots + a README explaining how to re-plot in MATLAB.

Run:   python export_paper_data.py
Output (in paper_export/):
    mscr_shape_results.xlsx     one sheet per dataset (+ a README sheet)
    *.png                       reference plots Claude generated
    README.md                   per-sheet column docs + MATLAB plotting recipes
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cfg import load_config
from calib import resolve_calib
from model import MoSSNet
from dataset import LabelRecord, MSCRShapeDataset

HERE = Path(__file__).parent
OUT = HERE / "paper_export"
OUT.mkdir(exist_ok=True)
SESSIONS = ["s09", "s10", "s11"]
CKPT = HERE / "checkpoints" / "mscr_shape.pt"

plt.rcParams.update({"font.size": 10, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})


# ── gather predictions over every labeled frame ─────────────────────────────
def collect():
    cfg = load_config()
    calib = resolve_calib(cfg)
    ck = torch.load(CKPT, map_location="cpu")
    net = MoSSNet(cfg["model"]["n_points"], pretrained=False)
    net.load_state_dict(ck["model"]); net.eval()
    axis_std = ck.get("axis_std")
    n = cfg["model"]["n_points"]

    rows, profiles, curves = [], [], {}
    example = None
    for s in SESSIONS:
        files = sorted((HERE / "data/labels" / s).glob("*.npz"))
        recs = [LabelRecord(p) for p in files]
        ds = MSCRShapeDataset(recs, calib, cfg, augment=False)
        tips_pred, tips_gt = [], []
        with torch.no_grad():
            for i, p in enumerate(files):
                img, r_s, L = ds[i]
                pts, Lp = net(img[None])
                pts = pts[0].numpy(); r_s = r_s.numpy()
                err = np.linalg.norm(pts - r_s, axis=1)        # (N,) per-point mm
                rows.append(dict(
                    session=s, frame=int(p.stem),
                    gt_tip_x=r_s[-1, 0], gt_tip_y=r_s[-1, 1], gt_tip_z=r_s[-1, 2],
                    pred_tip_x=pts[-1, 0], pred_tip_y=pts[-1, 1], pred_tip_z=pts[-1, 2],
                    tip_err_mm=float(err[-1]), mean_err_mm=float(err.mean()),
                    L_gt_mm=float(L.item()), L_pred_mm=float(Lp.item()),
                    mean_depth_mm=float(r_s[:, 2].mean())))
                profiles.append(err)
                tips_pred.append(pts[-1]); tips_gt.append(r_s[-1])
                if example is None and s == "s11":
                    example = (np.linspace(0, 1, n), r_s.copy(), pts.copy())
        curves[s] = (np.array(tips_pred), np.array(tips_gt))
    df = pd.DataFrame(rows)
    err_profile = np.stack(profiles)  # (n_frames, N)
    return cfg, df, err_profile, curves, example, axis_std


# ── per-axis correlation summary (after = this model; before = documented) ───
def axis_summary(df):
    out = []
    before = {"X": -0.22, "Y": -0.36, "Z": 0.86}   # session-2 global-pool baseline (s03)
    for ax, gt, pr in [("X", "gt_tip_x", "pred_tip_x"),
                       ("Y", "gt_tip_y", "pred_tip_y"),
                       ("Z", "gt_tip_z", "pred_tip_z")]:
        g, p = df[gt].values, df[pr].values
        out.append(dict(
            axis=ax,
            corr_after=float(np.corrcoef(p, g)[0, 1]),
            corr_before_baseline=before[ax],
            gt_span_mm=float(g.ptp()), pred_span_mm=float(p.ptp()),
            rmse_mm=float(np.sqrt(np.mean((p - g) ** 2)))))
    return pd.DataFrame(out)


def session_stats(df):
    g = df.groupby("session")
    return pd.DataFrame(dict(
        n_frames=g.size(),
        median_L_mm=g["L_gt_mm"].median(),
        std_L_mm=g["L_gt_mm"].std(),
        mean_err_mm=g["mean_err_mm"].mean(),
        tip_err_mm=g["tip_err_mm"].mean(),
        median_depth_mm=g["mean_depth_mm"].median())).reset_index()


def parse_training_log(path):
    if not path.exists():
        return None
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("epoch"):
            t = line.replace("*", "").split()
            try:
                rows.append(dict(
                    epoch=int(t[1]), lr=float(t[2].split("=")[1]),
                    val_loss=float(t[3].split("=")[1]),
                    mean_err_mm=float(t[4].split("=")[1].replace("mm", "")),
                    tip_err_mm=float(t[5].split("=")[1].replace("mm", ""))))
            except (IndexError, ValueError):
                pass
    return pd.DataFrame(rows) if rows else None


# ── plots ───────────────────────────────────────────────────────────────────
def plot_corr(df):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, name, gt, pr in zip(axs, "XYZ",
                                ["gt_tip_x", "gt_tip_y", "gt_tip_z"],
                                ["pred_tip_x", "pred_tip_y", "pred_tip_z"]):
        g, p = df[gt].values, df[pr].values
        ax.scatter(g, p, s=6, alpha=0.3, color="#1565C0")
        lo, hi = min(g.min(), p.min()), max(g.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#B71C1C", lw=1)
        ax.set_xlabel(f"GT tip {name} (mm)"); ax.set_ylabel(f"pred tip {name} (mm)")
        ax.set_title(f"{name}: r = {np.corrcoef(p, g)[0,1]:+.2f}")
    fig.suptitle("Predicted vs ground-truth tip position, per axis")
    fig.tight_layout(); fig.savefig(OUT / "tip_pred_vs_gt.png"); plt.close(fig)


def plot_error_profile(s_norm, prof):
    m, sd = prof.mean(0), prof.std(0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(s_norm, m, color="#00695C", lw=2, label="mean")
    ax.fill_between(s_norm, m - sd, m + sd, color="#00695C", alpha=0.2, label="±1 std")
    ax.set_xlabel("normalized arclength s (0 = base, 1 = tip)")
    ax.set_ylabel("position error (mm)")
    ax.set_title("Along-body reconstruction error vs arclength")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(OUT / "error_along_body.png"); plt.close(fig)


def plot_length_hist(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, c in zip(SESSIONS, ["#1565C0", "#E65100", "#2E7D32"]):
        ax.hist(df[df.session == s]["L_gt_mm"], bins=40, alpha=0.55, label=s, color=c)
    ax.set_xlabel("triangulated rod length L (mm)"); ax.set_ylabel("frames")
    ax.set_title("Per-frame rod length by session (clamped → consistent ~45 mm)")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(OUT / "length_consistency.png"); plt.close(fig)


def plot_training(tdf):
    if tdf is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ep = tdf.epoch.to_numpy()
    ax.plot(ep, tdf.mean_err_mm.to_numpy(), color="#1565C0", label="val mean error")
    ax.plot(ep, tdf.tip_err_mm.to_numpy(), color="#B71C1C", label="val tip error")
    ax.set_xlabel("epoch"); ax.set_ylabel("error (mm)")
    ax.set_title("Validation convergence")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(OUT / "training_curve.png"); plt.close(fig)


def plot_tip_traj(curves):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for s, c in zip(SESSIONS, ["#1565C0", "#E65100", "#2E7D32"]):
        gt = curves[s][1]
        axs[0].plot(gt[:, 0], gt[:, 1], lw=0.8, color=c, label=s)
        axs[1].plot(gt[:, 0], gt[:, 2], lw=0.8, color=c, label=s)
    axs[0].set_xlabel("X (mm)"); axs[0].set_ylabel("Y (mm)"); axs[0].set_title("tip path — XY")
    axs[1].set_xlabel("X (mm)"); axs[1].set_ylabel("Z (mm)"); axs[1].set_title("tip path — XZ")
    for a in axs:
        a.legend(frameon=False); a.set_aspect("equal", "box")
    fig.suptitle("Ground-truth tip trajectories (label data) per session")
    fig.tight_layout(); fig.savefig(OUT / "tip_trajectory.png"); plt.close(fig)


# ── README ──────────────────────────────────────────────────────────────────
README = """# mscr_shape — paper data export

`mscr_shape_results.xlsx` holds every meaningful quantity behind the figures.
Each sheet is a flat table (first row = column headers) so it loads cleanly with
MATLAB `readtable` / `readmatrix`. The PNGs here are reference plots Claude made;
reproduce them in MATLAB for the paper. Units: 3D = mm in the left-IR rectified
camera frame; image = px. Model checkpoint: checkpoints/mscr_shape.pt.

## Sheets
- **tip_pred_vs_gt** — one row per labeled frame. Columns: session, frame,
  gt_tip_{x,y,z}, pred_tip_{x,y,z}, tip_err_mm, mean_err_mm (mean along-body),
  L_gt_mm, L_pred_mm, mean_depth_mm. The core per-frame table.
- **axis_summary** — per tip axis (X,Y,Z): corr_after (this model),
  corr_before_baseline (session-2 global-pool model on s03), gt_span_mm,
  pred_span_mm, rmse_mm.
- **error_along_body** — mean & std position error vs normalized arclength s
  (0 = base, 1 = tip), N=64 samples. For the along-body error figure.
- **session_stats** — per session: n_frames, median/std length, mean/tip error,
  median depth.
- **training_curve** — per epoch: lr, val_loss, val mean_err_mm, val tip_err_mm
  (empty if training_log.txt was absent).
- **tip_trajectory_<session>** — ground-truth tip path over the session:
  frame_order, x, y, z. For trajectory plots.
- **example_centerline** — one representative frame (s11): s (0..1) and the full
  64-point gt_{x,y,z} vs pred_{x,y,z} centerline.

## MATLAB recipes
```matlab
T = readtable('mscr_shape_results.xlsx','Sheet','tip_pred_vs_gt');

% 1) Tip correlation, X axis
figure; scatter(T.gt_tip_x, T.pred_tip_x, 8, 'filled'); hold on
lims=[min(T.gt_tip_x) max(T.gt_tip_x)]; plot(lims,lims,'r--');
xlabel('GT tip X (mm)'); ylabel('pred tip X (mm)'); axis equal
r = corr(T.gt_tip_x, T.pred_tip_x); title(sprintf('X: r=%.2f', r));

% 2) Along-body error
E = readtable('mscr_shape_results.xlsx','Sheet','error_along_body');
figure; plot(E.s, E.mean_err_mm,'LineWidth',2); hold on
fill([E.s; flipud(E.s)], [E.mean_err_mm-E.std_err_mm; flipud(E.mean_err_mm+E.std_err_mm)], ...
     [0 .4 .35],'FaceAlpha',.2,'EdgeColor','none');
xlabel('normalized arclength s'); ylabel('error (mm)');

% 3) Length consistency
figure; hold on
for s = {'s09','s10','s11'}
  m = strcmp(T.session, s{1});  histogram(T.L_gt_mm(m), 40);
end
xlabel('rod length L (mm)'); ylabel('frames'); legend('s09','s10','s11');

% 4) Training convergence
C = readtable('mscr_shape_results.xlsx','Sheet','training_curve');
figure; plot(C.epoch, C.mean_err_mm, C.epoch, C.tip_err_mm,'LineWidth',1.5);
xlabel('epoch'); ylabel('error (mm)'); legend('mean','tip');

% 5) Tip trajectory (XZ)
P = readtable('mscr_shape_results.xlsx','Sheet','tip_trajectory_s11');
figure; plot(P.x, P.z,'LineWidth',1); xlabel('X (mm)'); ylabel('Z (mm)'); axis equal

% 6) Example centerline (3D)
G = readtable('mscr_shape_results.xlsx','Sheet','example_centerline');
figure; plot3(G.gt_x,G.gt_y,G.gt_z,'g','LineWidth',2); hold on
plot3(G.pred_x,G.pred_y,G.pred_z,'b--','LineWidth',2);
legend('GT','pred'); xlabel X; ylabel Y; zlabel Z; grid on
```
"""


def main():
    cfg, df, prof, curves, example, axis_std = collect()
    n = cfg["model"]["n_points"]
    s_norm = np.linspace(0, 1, n)

    asum = axis_summary(df)
    sstats = session_stats(df)
    eprof = pd.DataFrame(dict(s=s_norm, mean_err_mm=prof.mean(0), std_err_mm=prof.std(0)))
    tdf = parse_training_log(HERE / "training_log.txt")

    s_ex, gt_ex, pr_ex = example
    ex = pd.DataFrame(dict(s=s_ex,
                           gt_x=gt_ex[:, 0], gt_y=gt_ex[:, 1], gt_z=gt_ex[:, 2],
                           pred_x=pr_ex[:, 0], pred_y=pr_ex[:, 1], pred_z=pr_ex[:, 2]))

    xlsx = OUT / "mscr_shape_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        pd.DataFrame({"README": README.splitlines()}).to_excel(xl, "README", index=False)
        df.to_excel(xl, "tip_pred_vs_gt", index=False)
        asum.to_excel(xl, "axis_summary", index=False)
        eprof.to_excel(xl, "error_along_body", index=False)
        sstats.to_excel(xl, "session_stats", index=False)
        if tdf is not None:
            tdf.to_excel(xl, "training_curve", index=False)
        for s in SESSIONS:
            gt = curves[s][1]
            pd.DataFrame(dict(frame_order=np.arange(len(gt)),
                              x=gt[:, 0], y=gt[:, 1], z=gt[:, 2])).to_excel(
                xl, f"tip_trajectory_{s}", index=False)
        ex.to_excel(xl, "example_centerline", index=False)

    (OUT / "README.md").write_text(README)
    plot_corr(df); plot_error_profile(s_norm, prof); plot_length_hist(df)
    plot_training(tdf); plot_tip_traj(curves)

    print(f"wrote {xlsx}")
    print(f"sheets: README, tip_pred_vs_gt ({len(df)} rows), axis_summary, "
          f"error_along_body, session_stats, "
          f"{'training_curve, ' if tdf is not None else ''}"
          f"tip_trajectory_*, example_centerline")
    print("PNGs:", ", ".join(p.name for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
