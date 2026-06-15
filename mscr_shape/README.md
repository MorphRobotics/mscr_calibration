# mscr_shape

Data acquisition and monocular 3D shape reconstruction for a **magnetic soft
continuum robot** (MSCR — a black slender rod ~1 mm diameter), using an Intel
RealSense **D435** stereo IR pair and a **MoSSNet-style** shape-sensing network.

The two monochrome IR imagers (global-shutter, hardware-synced) provide the
stereo pair we use to build ground truth; the **left-IR rectified image** is the
network input. The RGB camera and the IR dot-pattern emitter are never used
(the emitter is disabled in all capture code). All 3D quantities are in **mm in
the left-IR rectified camera frame**; image quantities are in **px**.

Actuation is a **hand-held magnet** — capture is continuous (no robot-pose
logging), and frames where a hand/magnet occludes the rod are rejected at QC,
not at capture.

## Pipeline / run order

| Phase | Module | What it does |
|------|--------|--------------|
| 0 | `calib.py`, `cfg.py` | Load `calib.yaml` -> rectification maps, `P1/P2`, `F`; load `config.yaml`. |
| 1 | `capture.py`  | Stream synced left/right IR @ 1280×720, emitter off; save PNG pairs + manifest; drop motion-blurred frames. |
| 2 | `labeler.py`  | Rectify -> segment -> skeletonize -> 2D spline + subpixel -> epipolar correspondence -> triangulate -> 3D spline -> QC. Outputs per-frame NPZ labels + QC overlays. |
| 3 | `dataset.py`  | Pair each accepted label with its rectified left image; PyTorch Dataset; **split by configuration** (tip-position grid), photometric-only augmentation. |
| 4 | `model.py`, `train.py` | ResNet18 encoder + centerline head + length head; train with config-split val; report test mean/tip error (mm). |
| 5 | `infer.py`    | Run on saved/live frames, EMA smoothing + tip-jump rejection, reprojected overlay; ONNX export + verify; report FPS. |

```bash
# 0. configure: edit calib.yaml (real values) and config.yaml (parameters)
python calib.py                              # sanity-check calibration

# 1. capture
python capture.py --session s01              # live (r=record, q=quit)
python capture.py --session demo --fake --headless   # no hardware (test)

# 2. label
python labeler.py --test                     # synthetic self-test (< 0.5 mm)
python labeler.py --session s01              # label a whole session

# 3. dataset
python dataset.py                            # build + inspect splits

# 4. train
python train.py --overfit-test              # verify the training loop
python train.py                              # full training -> checkpoints/

# 5. infer / export
python infer.py --session s01                # reprojected overlay on saved frames
python infer.py --live                       # live D435 left-IR loop
python infer.py --export-onnx                # export + verify ONNX
```

## Per-phase tests

- **Phase 1** — `python capture.py --session demo --fake --headless`: synthesizes
  a moving dark curve in both views with a known disparity; runs the full save path.
- **Phase 2** — `python labeler.py --test`: a known 3D helix projected through
  `P1/P2` with noise; asserts reconstruction mean error < 0.5 mm.
- **Phase 3** — `python dataset.py`: builds config splits and checks tensor shapes.
- **Phase 4** — `python train.py --overfit-test`: overfits 20 frames to near-zero error.
- **Phase 5** — `python infer.py --export-onnx`: asserts ONNX ≈ PyTorch within tol.

## Files

- `config.yaml` — every tunable parameter (single source of truth).
- `calib.yaml` — stereo calibration (intrinsics, distortion `k1,k2,k3,p1,p2`,
  extrinsics `R`,`T` in mm, image size). **Placeholder values shipped — replace
  with real D435 calibration.**
- `data/raw/<session>/` — captured PNG pairs + `manifest.json`.
- `data/qc/<session>/` — per-frame ACCEPT/REJECT overlay JPEGs.
- `data/labels/<session>/` — per-accepted-frame NPZ (`r_s`, `L_mm`, …).
- `checkpoints/` — trained model + exported ONNX.

## Conventions

- Python 3.10+, current library versions (NumPy/OpenCV/SciPy/scikit-image/PyTorch/
  torchvision/onnx/onnxruntime/pyrealsense2). Not pinned to the original MoSSNet
  Python 3.7 / CUDA 11.1 stack — reimplemented, not vendored.
- Units: **mm** in 3D, **px** in image space. Frames documented in docstrings as
  "left-IR rectified camera frame".
- Every module has a `__main__` smoke test; segmentation is a swappable function
  so a learned segmenter can replace the threshold default.
