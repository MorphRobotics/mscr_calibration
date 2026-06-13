# mscr_shape — paper data export

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
