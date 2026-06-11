#!/usr/bin/env python3
"""
train.py — train the thin-rod U-Net.

Loss = BCE + Dice (Dice handles the heavy class imbalance of a thin target).
Saves the best-val checkpoint to seg/rod_seg.pt along with the net resolution.

Usage:
    python seg/train.py --epochs 60 --batch 16
    python seg/train.py --epochs 80 --rgbd     # (reserved; RGB only for now)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import RodSegDataset, list_pairs, NET_W, NET_H
from unet import UNet, count_params

HERE = Path(__file__).parent
CKPT = HERE / "rod_seg.pt"


def dice_loss(logits, target, eps=1.0):
    prob = torch.sigmoid(logits)
    num  = 2 * (prob * target).sum(dim=(1, 2, 3)) + eps
    den  = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


@torch.no_grad()
def dice_score(logits, target, thr=0.5, eps=1.0):
    pred = (torch.sigmoid(logits) > thr).float()
    num  = 2 * (pred * target).sum(dim=(1, 2, 3)) + eps
    den  = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (num / den).mean().item()


def main():
    ap = argparse.ArgumentParser(description="Train MSCR rod U-Net")
    ap.add_argument("--epochs", type=int,   default=60)
    ap.add_argument("--batch",  type=int,   default=16)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--base-ch",type=int,   default=24)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int,  default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    pairs = list_pairs(require_nonempty=True)
    if len(pairs) < 10:
        print(f"Only {len(pairs)} labelled non-empty frames found. "
              f"Collect + label more before training (aim for 150+).")
        if len(pairs) == 0:
            return
    rng = np.random.default_rng(0)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_frac))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    tr_ds = RodSegDataset(train_pairs, train=True)
    va_ds = RodSegDataset(val_pairs,   train=False)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    model = UNet(in_ch=3, base_ch=args.base_ch).to(device)
    print(f"Model params: {count_params(model)/1e6:.2f} M")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce   = nn.BCEWithLogitsLoss()

    best_val = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for img, msk in tr_ld:
            img, msk = img.to(device), msk.to(device)
            opt.zero_grad()
            logits = model(img)
            loss = bce(logits, msk) + dice_loss(logits, msk)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * img.size(0)
        tr_loss /= len(tr_ds)
        sched.step()

        model.eval()
        val_d = 0.0
        with torch.no_grad():
            for img, msk in va_ld:
                img, msk = img.to(device), msk.to(device)
                val_d += dice_score(model(img), msk) * img.size(0)
        val_d /= len(va_ds)

        flag = ""
        if val_d > best_val:
            best_val = val_d
            torch.save({
                "model":   model.state_dict(),
                "base_ch": args.base_ch,
                "net_w":   NET_W,
                "net_h":   NET_H,
            }, CKPT)
            flag = "  ✓ saved"
        print(f"epoch {ep:3d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  val_dice={val_d:.4f}{flag}",
              flush=True)

    print(f"\nBest val Dice: {best_val:.4f}")
    print(f"Checkpoint → {CKPT}")


if __name__ == "__main__":
    main()
