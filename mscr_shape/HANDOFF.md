# mscr_shape — Handoff / Resume Notes

Working dir: `/home/dozie/mscr_calibration/mscr_shape`
Last updated: 2026-06-11 (session 2)
GitHub: MorphRobotics/mscr_calibration (PRIVATE), branch `master`. Commit & push as you go.

> ⚡ **READ THIS FIRST — current frontier (session 2).** All 5 phases work end-to-end on
> REAL D435 data. A real model is trained (best: **5.19 mm mean / 7.85 mm tip**, session
> s03 only). BUT a 3D-viz diagnostic revealed the model **only tracks depth (Z), not the
> in-plane bending (X,Y)** — it underfits and predicts a near-mean shape. **The #1 open task
> is fixing that underfit.** Leading hypothesis + planned experiment are in the
> "Session-2 model diagnosis" section below. Jump there.
>
> 🟢 **SESSION 3 UPDATE — underfit FIXED + clamped-base data working.** Bending is
> now tracked (X corr +0.99, Z +0.99; TEST 3.04mm/5.31mm tip on clamped sessions
> s09-s11). See "SESSION 3 LOG" at the bottom. Current checkpoint matches the
> (3,4) model and is trained on s09+s10+s11. ONE weak axis remains: **Y tip motion
> under-tracked (corr +0.22)** — the new #1 open task.

---

Monocular 3D shape sensing for a magnetic soft continuum robot (MSCR, ~1 mm
black rod) using a D435 **stereo IR pair** for ground truth and a MoSSNet-style
ResNet18 network for inference. 3D units = mm in the **left-IR rectified camera
frame**; image units = px. RGB camera and IR emitter are never used.

## Build status by phase

| Phase | File(s) | Test | Status |
|------|---------|------|--------|
| 0 | `calib.py`, `cfg.py`, `config.yaml` | `python calib.py` | ✅ done |
| 1 | `capture.py` | `python capture.py --session demo --fake --headless` (60/60 saved) | ✅ done |
| 2 | `labeler.py` | `python labeler.py --test` → **0.486 mm** < 0.5 mm | ✅ done |
| 3 | `dataset.py` | `python dataset.py` (config-split, shapes) | ✅ done |
| 4 | `model.py`, `train.py` | `python train.py --overfit-test` → **0.674 mm** < 1.0 mm | ✅ done |
| 5 | `infer.py` | `python infer.py --export-onnx` → diff ~1e-6 | ✅ done (ONNX path) |

### Phase 4 — the one open item
The overfit test (`train.py --overfit-test`) fabricates 20 in-memory samples
(distinct sinusoidal input patterns → distinct smooth 3D arcs) to verify the
train loop, NOT to learn anything real. It must reach mean error < 1.0 mm.

History of attempts:
- v1: random point-cloud targets + random-noise images → stuck ~48 mm.
  Two root causes found & fixed:
  1. **Length term exploded the loss** — switched to realistic ~100 mm arcs.
  2. **Random-noise images are indistinguishable after global avg-pool** —
     switched to per-frame sinusoidal patterns (separable features).
  3. **Dropout(0.2) in the point head blocked exact memorization** — disabled
     dropout inside the overfit test only.
- With those fixes the error fell monotonically: 800 epochs → 4.2 mm,
  so epochs were raised to **2000** (current run). Trend at last check:
  iter600 = 7.4 mm and still dropping fast; expected to pass < 1 mm.

**If the run did NOT cross 1 mm:** it is purely an optimization-budget issue,
not a bug. Bump `range(2000)` in `overfit_test()` (train.py) to 3000–4000, or
add a cosine LR decay. The loss curve is clean and monotonic.

The overfit changes (epoch count, disabled dropout) live ONLY in
`overfit_test()`. Real training (`train()`) is untouched and keeps dropout.

## CRITICAL prerequisite before any real run

The user pasted a **monocular** calibration into `calib.yaml` (one camera_matrix,
no stereo R/T; fx≈892 / 71° FOV → looks like the RGB lens). It is **unusable**
for triangulation. Decision taken: pull the **factory stereo IR calibration**
from the D435. With the camera plugged in, run ONCE:

```bash
python calib.py --from-device --save calib_stereo.yaml
```

`config.yaml: paths.calib` already points at `calib_stereo.yaml`. Until that file
exists, capture works but labeler/dataset/train/infer stop with a clear
"generate it from the D435" error (by design). `calib.py:from_realsense()` reads
per-IR-stream intrinsics + left→right extrinsics (translation m→mm, rotation
column-major). Self-tests use `calib.nominal_calib()` (file-free) so they never
depend on hardware.

## Real run order (once calib_stereo.yaml exists)

```bash
python calib.py --from-device --save calib_stereo.yaml   # 1x, D435 attached
python capture.py --session s01           # real rod + hand-held magnet (r=rec, q=quit)
python labeler.py --session s01           # inspect data/qc/s01/*.jpg overlays
python dataset.py                         # confirm train/val/test split sizes
python train.py                           # real training -> checkpoints/mscr_shape.pt
python infer.py --session s01             # or --live ; reprojected overlay + FPS
python infer.py --export-onnx             # checkpoints/mscr_shape.onnx
```

NOTE: there is currently NO real captured data — only a synthetic `--fake`
session. A *useful* trained model is impossible until real D435 footage is
captured and labeled. `train.py` (no flag) needs `data/labels/<session>/`.

## Architecture notes / gotchas

- `calib.py`: stereoRectify → P1/P2 + maps; F normalized by max|entry| (F[2,2]=0
  for pure-horizontal stereo). distortion YAML order is **k1,k2,k3,p1,p2**
  (reordered to OpenCV k1,k2,p1,p2,k3 in `_dist`); librealsense coeffs are
  already OpenCV order.
- `labeler.py`: segmentation is a swappable `Segmenter` callable
  (`threshold_segmenter` default) so a learned segmenter can drop in. Epipolar
  match uses rectified rows as scanlines; right-curve samples ~parallel to
  scanlines flagged DEGENERATE. QC accepts iff mean reproj < `reproj_thresh_px`
  in BOTH views; writes overlay JPEG for every frame + NPZ for accepted ones.
- `dataset.py`: split is **by configuration** (tip-position grid clusters),
  never by random frame. Augmentation is photometric ONLY (no geometric — the
  labels are metric 3D).
- `model.py`: ResNet18, conv1 reseeded to 1-channel from mean of RGB weights;
  point head (N×3) + length head. `shape_errors()` → mean & tip error (mm).
- All params in `config.yaml`; load via `cfg.load_config()`.

================================================================================
# SESSION 2 LOG (the real-data work) — read top-to-bottom
================================================================================

Everything above was the initial build. Session 2 took it onto real hardware.
All the config values below are already committed; the pipeline WORKS. The only
unsolved thing is the model underfit (last section).

## What got fixed to make real capture work (in order encountered)

1. **D435 firmware was dead-old (5.11.1.100, ~2019).** IR streams returned zero
   frames ("Frame didn't arrive within 5000") while depth worked. Flashed to
   **5.16.0.1** via `rs-fw-update -f /home/dozie/librealsense/build/common/fw/D4XX_FW_Image-5.16.0.1.bin`.
   IR now streams. If IR ever dies again, suspect firmware first.
2. **calib**: `calib.yaml` the user pasted is MONOCULAR (unusable). Real stereo
   calib is `calib_stereo.yaml`, generated via `python calib.py --from-device`.
   It exists on disk (gitignored). baseline 49.94 mm, f 639.5 px — sane.
3. **Capture/scene tuning** (all in `config.yaml`, already committed):
   - `blur_var_threshold: 1.0` — real IR of a thin rod on a plain background is
     low-texture; the synthetic-tuned 40 dropped 100% of frames.
   - `inv_threshold: 40` is now only used by the legacy `threshold` segmenter.
   - The "white" background reads as mid-grey (~87) in IR, rod ~31–70.
4. **Segmentation rewrite (big one).** Global inverse-threshold failed: rod
   brightness varies with distance/lighting and D435 **vignetting makes corner
   pixels as dark as the rod**, so it locked onto corner specks (false positives
   that still passed reproj QC). Fix in `labeler.py`:
   - `blackhat_segmenter` (now DEFAULT, `config: segmenter: blackhat`): black-hat
     morphology extracts thin dark features independent of absolute brightness /
     vignette. `blackhat_kernel: 21` (must exceed rod px width).
   - `_pick_rod_component`: choose the MOST ELONGATED component (rod is a streak),
     not the largest blob. `SEGMENTERS` registry + `get_segmenter(lp)`.
   - This took one session from 4.5% false-positive labels → 56% true labels.
5. **Robust triangulation + QC gates** (in `labeler.py`, `config.yaml`): reproj
   error alone is too weak — near-degenerate stereo (rod too close / foreshortened)
   gives low reproj error but garbage 3D (negative Z, thousands of mm). Added:
   - `robust_filter_3d`: depth gate `depth_min_mm:120 / depth_max_mm:500` + per-point
     step-jump rejection (`max_seg_jump:6.0`, `min_inlier_frac:0.5`).
   - per-frame mean-depth accept gate; **session-level length-consistency gate**
     (`length_tol_frac:0.35`, reject frames whose L deviates from session median —
     rod is rigid). `process_session` is now 2-pass (label all → compute median L →
     re-gate → save). NPZ now also stores `mean_depth_mm`.

## CAPTURE CONSTRAINTS (hard-won — see memory `mscr-shape-capture-distance`)
The rod must be, SIMULTANEOUSLY:
- **15–30 cm from camera** (D435 stereo floor ~10 cm). Too close (s01 ~4 cm,
  s02 ~7 cm, s04/s08 ~11–13 cm) → degenerate. Too far (s06 ~70 cm) → no disparity.
- **mostly VERTICAL** in frame (crosses many rows). HORIZONTAL = parallel to
  epipolar scanlines = every sample DEGENERATE, zero triangulation (this killed s05).
- ideally **base clamped & unoccluded** (not hand-held). Hand-held base occludes
  the base differently per frame → the *detected* rod length varies session to
  session (s03=55 mm vs s04/s08=44 mm), which is inconsistent label noise.

## SESSIONS captured & their verdicts
- **s03 — THE GOOD ONE.** ~14 cm, vertical, 465 accepted labels (56%), length
  55±5 mm, depth 130–158 mm, reproj ~0.35 px. This is what the model is trained on.
- s01, s02: too close, deleted.
- s04 (157 labels @ ~13 cm), s08 (99 @ ~13 cm): valid but ~same depth as s03 and
  *inconsistent length* (44 mm) → COMBINING THEM HURT (see below).
- s05: rod horizontal → 0 usable. s06: too far → 0 usable.
- s07: only 12 labels but the ONLY depth variety (157–225 mm) — too few to help.
- Labels for s04/s07/s08 are currently PARKED in `/tmp/mscr_labels_park/` (moved out
  of `data/labels/` so dataset.py sees s03 only). `data/labels/` should contain
  only `s03/` right now.

## MODEL STATUS
- Best model = **s03 only**: TEST **5.19 mm mean / 7.85 mm tip**. Checkpoint
  `checkpoints/mscr_shape.pt` currently holds THIS model.
- **Combining sessions made it WORSE** (13.4 mm) — inconsistent labels + the
  config-split test set then includes OOD far poses. Quality > quantity here.
- Added to `train()`: cosine LR decay + early stopping (`early_stop_patience:15`).
  Confirmed the ~6 mm plateau is a DATA/MODEL ceiling, not optimization.

## ⚠️ THE #1 OPEN PROBLEM — model only learns depth, not bending
Diagnostic (via the new `infer.py --viz3d`, which renders the predicted 3D
centerline + tip path as a GIF; `viz3d_s03_full.gif` exists):
Per-frame predicted vs GT TIP, on s03:

| axis | pred span | GT span | corr(pred,gt) |
|------|-----------|---------|---------------|
| X (in-plane bend) | 1.1 mm | 52.7 mm | **-0.22** |
| Y (in-plane bend) | 0.2 mm | 24.0 mm | -0.36 |
| Z (depth)         | 19.4 mm | 36.9 mm | **0.86** |

The model tracks DEPTH well but predicts a **near-constant in-plane shape** — it
underfits the bending (bad corr even on TRAIN frames). NOT a viz bug; the model
genuinely outputs a near-static rod. NOT a resolution problem (rod is 13 px raw /
~5 px at the 288×512 input — clearly visible; tip sweeps ~90 px in the input).
The base is well-anchored in s03 (std ~4 mm), so it's not a position-ambiguity issue.

**LEADING HYPOTHESIS (next experiment to run): unnormalized regression targets.**
Training loss crashed 2186→72 in ONE epoch = model instantly nails the large Z
offset (~135 mm); afterwards X/Y values (~0) produce tiny gradients that get
swamped, so it parks at the mean shape. FIX TO TRY:
- Standardize targets per axis (z-score using TRAIN-set mean/std) so X/Y/Z
  contribute comparably. Cleanest impl: weight the point loss per axis by
  1/var_axis in `loss_fn` (mean cancels in the difference, so just divide each
  axis residual by its train std). Keep inference in mm.
- Then RE-MEASURE the X/Y correlation table above (script pattern is in the chat;
  load `checkpoints/mscr_shape.pt`, predict over `data/labels/s03/*.npz`, corrcoef
  pred tip vs GT tip per axis). Success = X corr goes from ~0 toward >0.7.
- If target-normalization alone doesn't fix it, try: remove point-head Dropout,
  train longer, and/or a tight ROI crop around the rod fed at higher res.
- DURABLE fix regardless: capture more CONSISTENT data — clamped vertical base,
  several sessions across 15–30 cm. Then combining sessions will help not hurt.

## viz3d usage
`python infer.py --session s03 --viz3d --out viz3d_s03.gif [--stride N] [--max-frames N]`
Renders fixed-view (no spin) GIF of predicted 3D centerline (green), base (black),
tip (red) + accumulating tip path; also dumps `<out>.npz` (r_s_seq [T,64,3], tips
[T,3]). View GIF in VS Code explorer or `xdg-open`.

## Tests still green (run after any change)
- `python labeler.py --test` → ~0.44 mm < 0.5 mm
- `python train.py --overfit-test` → < 1.0 mm
- `python infer.py --export-onnx` → diff ~1e-6
- `python model.py`, `python dataset.py`, `python capture.py --session demo --fake --headless`

## Loose ends / housekeeping
- s04/s07/s08 labels parked in `/tmp/mscr_labels_park/` — restore into
  `data/labels/` only when doing multi-session experiments (and expect it to hurt
  until labels are made consistent).
- `viz3d_*.gif/.npz` are working-output files in the project dir (not gitignored;
  large — consider not committing the 15 MB ones).
- gitignore excludes `data/`, `checkpoints/`, `calib_stereo.yaml`.

================================================================================
# SESSION 3 LOG — fixed the in-plane underfit (the "elastic shape not apparent")
================================================================================

Goal: make the elastic bending / shape change apparent in the 3D reconstruction
(session-2's #1 open problem: model tracked depth Z but predicted a near-rigid
in-plane shape, X/Y tip corr ~ -0.2 / -0.4).

## Two changes made (both committed in working tree; NOT yet git-committed)
1. **Per-axis target normalization** (`train.py`): `loss_fn` now takes `axis_w`
   = 1/std-per-axis computed from the TRAIN targets (`compute_axis_weight`), so
   the small in-plane (X,Y) residuals aren't swamped by the large depth offset.
   Toggle: `config.yaml train.normalize_targets: true`. Checkpoint also stores
   `axis_std`. Train-set std came out X=6.4 Y=13.9 Z=6.0 mm.
   - Result ALONE: helped error a little (4.66/6.45) but did NOT fix bending
     (X corr -0.09, Y -0.38, pred span ~0). This matched the handoff's
     "if normalization alone doesn't fix it" contingency.
2. **Spatial head — THE REAL FIX** (`model.py`): the ResNet18 encoder ended in
   global average pooling, which destroys spatial location — but in-plane
   bending *is* location. Now the encoder stops before avgpool (`children[:-2]`)
   and the point head reads an `AdaptiveAvgPool2d(grid)`-flattened feature that
   preserves a coarse position grid. Length head still uses a global-pool feat.
   - `self.grid = (3, 4)` chosen because the 9x16 feature map (for 288x512 input)
     must divide evenly or ONNX export of adaptive_avg_pool2d FAILS. (Tried (4,4)
     first — trained great but `infer.py --export-onnx` errored: "output size not
     a factor of input size", since 9 % 4 != 0. (3,4) divides 9 and 16.)

## Measured improvement (s03 tip corr, via the corr script — see below)
With spatial head **(4,4)** + normalization, TEST 4.04mm / 5.06mm tip:
| axis | before (session 2) | after |
|------|--------------------|-------|
| X corr | -0.22 | **+0.86** |
| Y corr | -0.36 | **+0.49** |
| Z corr |  0.86 | +0.88 |
Pred X tip span went 1.1mm -> 14.6mm (GT 52.9mm). Bending is now clearly
tracked — the elastic shape change is apparent. (Span still under-shoots GT =
classic regression-to-the-mean; the durable cure is more consistent multi-pose
data, as already noted.)

## ⚠️ STATE AT PAUSE — DO THIS FIRST NEXT SESSION
- The `(3,4)`-grid retrain was INTERRUPTED. So `checkpoints/mscr_shape.pt` on
  disk is the **(4,4)** model and will NOT load into the current `model.py`.
  **Run `python train.py` once** to regenerate a matching checkpoint (~3-4 min
  on the GPU, 50 epochs). Expect TEST ~4-5mm mean / ~5-6mm tip.
- Then re-confirm bending with the corr script and regenerate the viz GIF:
  `python infer.py --session s03 --viz3d --out viz3d_s03_fixed.gif --stride 4`
- Then `python infer.py --export-onnx` (should now pass with the (3,4) grid).
- `python train.py --overfit-test` still uses `loss_fn` with no axis_w (unchanged
  path) but DOES exercise the new model — confirm it still crosses <1mm.

## corr script (re-measure X/Y/Z tip correlation on s03)
Load `checkpoints/mscr_shape.pt`, build `MSCRShapeDataset` over
`data/labels/s03/*.npz` (augment=False), predict each frame, then per axis print
`np.corrcoef(pred_tip[:,a], gt_tip[:,a])`. Success = X corr > 0.7. (Full script
was run from the package dir this session; trivial to reconstruct.)

## Files changed this session (uncommitted)
`model.py` (spatial head), `train.py` (axis_w normalization), `config.yaml`
(`normalize_targets`). Nothing else touched.

================================================================================
# SESSION 3 LOG (cont.) — clamped-base data + region-grow labeler
================================================================================

## Capture: base is now CLAMPED (sessions s09, s10, s11)
USB had dropped to 2.0 ("Couldn't resolve requests" at 1280x720@30); fixed by
reconnecting on USB 3 (cable/port — many USB-C cables are USB2-only). Then
captured s09/s10/s11 (~900-1030 frames each) with the rod clamped to a fixed
base, mostly vertical, varied bend + distance.

## Labeler problem found: background clutter beat the rod
With the clamp scene, the global "most-elongated component" locked onto STRAIGHT
BACKGROUND EDGES (table/desk lip = horizontal; grey-board edge + door molding =
vertical) instead of the short curved rod. Symptoms: bogus median L=256mm,
per-session L wildly inconsistent (7.8 / 93.5 / 61.3 mm). Diagnosed by viewing
data/qc/<s>/*.jpg overlays — the traced curve sat on the table edge / board edge.

Fixes added to labeler.py (all committed in working tree, NOT git-committed):
1. **Orientation + border gate** in `_pick_rod_component` (keyed to base_side):
   a bottom/top base => VERTICAL rod, reject horizontal blobs (table lip) and
   blobs touching the left/right borders. Helped s10 but not s09/s11 (they had a
   competing *vertical* board edge).
2. **Activity ROI** (`compute_activity_roi`, temporal-std) — ABANDONED / default
   OFF (`use_activity_roi: false`): the moving hand+magnet and a bright window
   out-moved the rod, so the ROI landed on them. Code left in place, disabled.
3. **THE FIX — base-anchor region-grow** (user picked this). The clamp base is a
   FIXED pixel, so:
   - `python labeler.py --session <s> --set-base` opens the rectified LEFT then
     RIGHT view; click the rod base in each (ENTER ok / r redo / q abort). Saved
     to `data/raw/<s>/base_anchor.json` {left:[x,y], right:[x,y]}. One per session.
   - `_pick_component_at_anchor` selects the connected component CONTAINING the
     anchor (nearest fg pixel within 50px if it lands in a gap). Background edges,
     hand, window are separate components => ignored.
   - `order_skeleton(..., anchor)` picks the base endpoint nearest the anchor.
   - `_select_component` dispatches: anchor region-grow if anchor set, else the
     orientation/elongation heuristic. ROI gate still applied first (now off).
   - A headless `--base-left x y --base-right x y` was offered but NOT added (user
     used the GUI). Add it if a future session is headless.

## RESULT — labels are now clean AND length-consistent across sessions
| session | accept | median L |
|---------|--------|----------|
| s09 | 90.4% | 45.9 mm |
| s10 | 100%  | 44.6 mm |
| s11 | 100%  | 45.0 mm |
The rigid rod finally reconstructs to the SAME ~45mm everywhere (clamped base =
no occlusion-length variation). reproj errors ~0.12-0.4 px. So combining sessions
now HELPS. (s03's 55mm came from a hand-held/occluded base — INCONSISTENT with
the clamped truth, so s03 was PARKED to /tmp/mscr_labels_park/s03. data/labels/
holds s09/s10/s11 only.)

## MODEL retrained on s09+s10+s11 (split 1850/501/427)
TEST **3.04 mm mean / 5.31 mm tip** (best yet). Bending correlation (tip, n=926):
| axis | pred span | GT span | corr |
|------|-----------|---------|------|
| X (lateral bend) | 60.2 | 67.8 | +0.99 |
| Y (vertical/along-axis) | 3.5 | 58.5 | +0.22 |
| Z (depth) | 110.2 | 126.7 | +0.99 |
X and Z near-perfect with MATCHING span (no more mean-collapse) — the elastic
bend is apparent. Viz: `viz3d_s11_clamped.gif`.

## ⚠️ NEW #1 OPEN TASK — Y tip axis under-tracked (corr +0.22, span 3.5/58.5mm)
X (lateral) and Z (depth) are solved; Y is not. Y tip motion runs ~along the
rod's near-vertical image axis (foreshortening) — the hardest DOF to see
monocularly. Ideas to try next: (a) check whether Y bending is even visually
present/separable in the images (it may be partly degenerate); (b) per-axis loss
weight is already 1/std (Y std 12.0 ~ X std 9.6, so it's NOT a loss-weighting
issue — the signal/observability is the suspect); (c) capture a session that
deliberately bends in the camera-vertical plane; (d) tighter ROI crop / higher
input res around the rod. Re-measure with the corr script over s09/s10/s11.

## Files changed this session (uncommitted): model.py, train.py, config.yaml,
labeler.py. New per-session files: data/raw/<s>/base_anchor.json.
