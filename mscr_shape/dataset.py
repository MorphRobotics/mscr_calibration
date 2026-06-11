"""Phase 3 — dataset assembly.

Pairs each accepted 3D-centerline label (from labeler.py) with its LEFT-IR
**rectified** image and serves them as a PyTorch Dataset:

    sample = (image[1,H,W] float in [0,1],  r_s[N,3] mm,  L[1] mm)

The image is the network input: single-channel left-IR rectified, resized to
config `dataset.image_size`. The target r_s lives in the left-IR rectified
camera frame (mm), resampled to a fixed N points.

Augmentations are **photometric only** (brightness/contrast jitter, Gaussian
blur). No geometric augmentation: the labels are metric 3D, so rotating /
flipping the 2D image would invalidate them.

Splitting is **by configuration, not by frame**. Frames are clustered by tip
position on a coarse grid (`config_grid_mm`) and whole clusters are assigned to
train/val/test, so near-identical poses can't leak across splits.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from calib import StereoCalib, load_calib, resolve_calib
from cfg import load_config


def resample_arclength(r_s: np.ndarray, n: int) -> np.ndarray:
    """Resample an (M,3) polyline to n points uniform in arclength."""
    seg = np.linalg.norm(np.diff(r_s, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] < 1e-9:
        return np.repeat(r_s[:1], n, axis=0)
    target = np.linspace(0, cum[-1], n)
    out = np.empty((n, 3), dtype=np.float64)
    for d in range(3):
        out[:, d] = np.interp(target, cum, r_s[:, d])
    return out


class LabelRecord:
    """One accepted label + the path to its raw left image."""

    def __init__(self, npz_path: Path):
        d = np.load(npz_path, allow_pickle=True)
        self.r_s = d["r_s"].astype(np.float32)          # (N,3) mm
        self.L_mm = float(d["L_mm"])
        self.left_image = str(d["left_image"])
        self.npz_path = npz_path

    @property
    def tip(self) -> np.ndarray:
        return self.r_s[-1]


def discover_records(data_root: Path, sessions: Optional[List[str]] = None) -> List[LabelRecord]:
    lab_root = data_root / "labels"
    sess_dirs = ([lab_root / s for s in sessions] if sessions
                 else sorted(p for p in lab_root.iterdir() if p.is_dir()))
    recs: List[LabelRecord] = []
    for sd in sess_dirs:
        for npz in sorted(sd.glob("*.npz")):
            recs.append(LabelRecord(npz))
    return recs


def config_split(records: List[LabelRecord], grid_mm: float,
                 fracs: Tuple[float, float, float], seed: int
                 ) -> Tuple[List[int], List[int], List[int]]:
    """Assign whole tip-position grid cells to train/val/test."""
    cells: dict[tuple, list[int]] = {}
    for i, r in enumerate(records):
        key = tuple(np.round(r.tip / grid_mm).astype(int))
        cells.setdefault(key, []).append(i)

    keys = list(cells.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    splits = {"train": keys[:n_tr], "val": keys[n_tr:n_tr + n_va], "test": keys[n_tr + n_va:]}
    out = {k: [i for key in v for i in cells[key]] for k, v in splits.items()}
    return out["train"], out["val"], out["test"]


class MSCRShapeDataset(Dataset):
    """left-IR rectified image -> (image, r_s target, length)."""

    def __init__(self, records: List[LabelRecord], calib: StereoCalib, cfg: dict,
                 indices: Optional[List[int]] = None, augment: bool = False):
        self.records = records
        self.indices = indices if indices is not None else list(range(len(records)))
        self.calib = calib
        self.n_points = cfg["dataset"]["n_points"]
        self.out_h, self.out_w = cfg["dataset"]["image_size"]
        self.aug_cfg = cfg["dataset"]["aug"]
        self.augment = augment
        self._rng = np.random.default_rng(cfg["dataset"]["seed"])

    def __len__(self) -> int:
        return len(self.indices)

    def _load_image(self, rec: LabelRecord) -> np.ndarray:
        img = cv2.imread(rec.left_image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(rec.left_image)
        img = self.calib.rectify_left(img)
        return img

    def _photometric(self, img: np.ndarray) -> np.ndarray:
        a = self.aug_cfg
        f = img.astype(np.float32)
        b = 1.0 + self._rng.uniform(-a["contrast"], a["contrast"])      # contrast
        c = self._rng.uniform(-a["brightness"], a["brightness"]) * 255  # brightness
        f = b * (f - 127.5) + 127.5 + c
        if self._rng.random() < a["blur_prob"]:
            sigma = self._rng.uniform(0.1, a["blur_sigma_max"])
            f = cv2.GaussianBlur(f, (0, 0), sigma)
        return np.clip(f, 0, 255)

    def __getitem__(self, i: int):
        rec = self.records[self.indices[i]]
        img = self._load_image(rec)
        if self.augment:
            img = self._photometric(img)
        img = cv2.resize(img.astype(np.float32), (self.out_w, self.out_h))
        img_t = torch.from_numpy(img[None] / 255.0).float()       # (1,H,W)

        r_s = resample_arclength(rec.r_s, self.n_points).astype(np.float32)
        return img_t, torch.from_numpy(r_s), torch.tensor([rec.L_mm], dtype=torch.float32)


def build_datasets(cfg: dict, calib: StereoCalib, sessions: Optional[List[str]] = None
                   ) -> Tuple[MSCRShapeDataset, MSCRShapeDataset, MSCRShapeDataset]:
    data_root = Path(cfg["paths"]["data_root"])
    recs = discover_records(data_root, sessions)
    if not recs:
        raise RuntimeError(f"no labels found under {data_root/'labels'}")
    d = cfg["dataset"]
    tr, va, te = config_split(recs, d["config_grid_mm"], d["split"], d["seed"])
    return (MSCRShapeDataset(recs, calib, cfg, tr, augment=True),
            MSCRShapeDataset(recs, calib, cfg, va, augment=False),
            MSCRShapeDataset(recs, calib, cfg, te, augment=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset assembly smoke test")
    ap.add_argument("--config", default=None)
    ap.add_argument("--sessions", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    calib = resolve_calib(cfg)
    tr, va, te = build_datasets(cfg, calib, args.sessions)
    print(f"records: train={len(tr)} val={len(va)} test={len(te)}")
    img, r_s, L = tr[0]
    print(f"image {tuple(img.shape)} dtype={img.dtype} range=[{img.min():.2f},{img.max():.2f}]")
    print(f"r_s {tuple(r_s.shape)}  L={L.item():.1f} mm  tip={r_s[-1].numpy()}")
    assert img.shape[0] == 1 and r_s.shape == (cfg["dataset"]["n_points"], 3)
    print("OK")


if __name__ == "__main__":
    main()
