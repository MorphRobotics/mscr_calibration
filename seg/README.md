# MSCR Semantic Segmentation (tiny U-Net)

Learned rod segmentation to replace the fragile dark-threshold method. The
U-Net learns what the *rod* looks like, so it ignores hair, shadows, dark
clothing, and backlit edges that defeat intensity thresholding.

- **Model:** 4-level U-Net, ~1.1 M params, RGB in → 1-channel mask
- **Resolution:** 320×192 (network) → upsampled to full frame
- **Speed:** ~55 fps on CUDA (FP16) — real-time
- **Deps:** `torch` (CUDA build already present), `opencv-python`, `numpy`

## Workflow

### 1. Collect frames
```bash
python seg/collect.py --n 300 --every 3
# move the rod through many poses, angles, lighting, backgrounds
# SPACE = force-save, q = stop. Add --save-depth for RGB-D later.
```
Frames → `seg/dataset/images/`.

### 2. Auto-label seed masks
```bash
python seg/autolabel.py --threshold 60           # add --use-depth if you saved depth
```
Generates initial masks in `seg/dataset/masks/` and review overlays in
`seg/dataset/overlays/`. It reports how many frames came out empty (those
need manual labelling).

### 3. Correct the masks
```bash
python seg/label_tool.py
# Left-drag = paint rod, Right-drag = erase, [ ] = brush size
# n/SPACE = next, p = prev, c = clear, q = save+quit
```
Aim for ~150+ clean frames. You don't need pixel perfection — the U-Net
tolerates slightly noisy masks. Skim through, fix the obviously-wrong ones,
delete hopeless frames (remove both the image and its mask).

### 4. Train
```bash
python seg/train.py --epochs 60 --batch 16
```
Saves the best-validation checkpoint to `seg/rod_seg.pt`. Watch `val_dice`;
0.7+ is usable, 0.85+ is good for a thin target.

### 5. Run the tracker with the learned segmenter
```bash
python mscr_tracker.py --debug --seg-model seg/rod_seg.pt
# optionally AND with the depth gate to kill far-background false positives:
python mscr_tracker.py --debug --seg-model seg/rod_seg.pt --seg-depth-gate
```
Everything downstream (skeleton, 3-D projection, arc length, tip) is unchanged
— only the segmentation step is swapped. Drop the `--seg-model` flag to fall
back to the classical dark-threshold path.

## Files
| File | Role |
|---|---|
| `collect.py` | Capture color (+depth) frames from the D435i |
| `autolabel.py` | Classical seed masks for every frame |
| `label_tool.py` | Mouse brush to correct masks |
| `unet.py` | The U-Net model |
| `dataset.py` | Dataset + augmentation + preprocessing |
| `train.py` | Training loop (BCE + Dice), saves `rod_seg.pt` |
| `infer.py` | `RodSegmenter` real-time inference wrapper |

## Tips
- **Variety beats volume.** 150 varied frames (different backgrounds, rod
  bends, lighting) beat 500 near-duplicates.
- Include hard negatives: frames with your hand, hair, and dark clothing in
  view but the rod in different positions, so the net learns to exclude them.
- If inference is too slow on a weaker GPU, lower `base_ch` in `train.py`
  (e.g. 16) or the resolution in `dataset.py` (`NET_W/NET_H`).
- `seg_threshold` (default 0.5) in `TrackerParams` trades recall vs precision;
  lower it if the net misses thin sections, raise it if it bleeds.
