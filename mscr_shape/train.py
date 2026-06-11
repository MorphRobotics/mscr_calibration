"""Phase 4 — training loop for MoSSNet.

Trains on the config-split train set, validates on the config-split val set,
checkpoints on best val mean shape error, and reports final test metrics
(mean along-body error, tip error; both mm).

    python train.py                 # full training on data/labels/*
    python train.py --overfit-test  # overfit 20 synthetic frames -> near-zero loss

Loss = MSE(points) + lambda * MSE(length).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from calib import resolve_calib
from cfg import load_config
from dataset import build_datasets
from model import MoSSNet, shape_errors


def loss_fn(pred_pts, pred_L, gt_pts, gt_L, lam):
    mse = nn.functional.mse_loss
    return mse(pred_pts, gt_pts) + lam * mse(pred_L, gt_L)


@torch.no_grad()
def evaluate(net, loader, device, lam) -> Tuple[float, float, float]:
    net.eval()
    tot_loss, tot_me, tot_te, n = 0.0, 0.0, 0.0, 0
    for img, r_s, L in loader:
        img, r_s, L = img.to(device), r_s.to(device), L.to(device)
        pts, Lp = net(img)
        tot_loss += loss_fn(pts, Lp, r_s, L, lam).item() * len(img)
        me, te = shape_errors(pts, r_s)
        tot_me += me.item() * len(img)
        tot_te += te.item() * len(img)
        n += len(img)
    return tot_loss / n, tot_me / n, tot_te / n


def train(cfg: dict) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calib = resolve_calib(cfg)
    tr_ds, va_ds, te_ds = build_datasets(cfg, calib)
    tc = cfg["train"]
    tr = DataLoader(tr_ds, batch_size=tc["batch_size"], shuffle=True,
                    num_workers=tc["num_workers"])
    va = DataLoader(va_ds, batch_size=tc["batch_size"], num_workers=tc["num_workers"])
    te = DataLoader(te_ds, batch_size=tc["batch_size"], num_workers=tc["num_workers"])

    net = MoSSNet(cfg["model"]["n_points"], cfg["model"]["pretrained"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    # Cosine LR decay so the optimizer settles instead of bouncing at a fixed LR.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
    lam = tc["lambda_length"]
    patience = tc.get("early_stop_patience", 0)  # 0 disables early stopping

    ckpt = Path(__file__).parent / tc["ckpt"]
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    since_best = 0
    for ep in range(tc["epochs"]):
        net.train()
        for img, r_s, L in tr:
            img, r_s, L = img.to(device), r_s.to(device), L.to(device)
            opt.zero_grad()
            pts, Lp = net(img)
            loss = loss_fn(pts, Lp, r_s, L, lam)
            loss.backward()
            opt.step()
        sched.step()
        vl, vme, vte = evaluate(net, va, device, lam)
        flag = ""
        if vme < best:
            best = vme
            since_best = 0
            torch.save({"model": net.state_dict(), "config": cfg}, ckpt)
            flag = " *"
        else:
            since_best += 1
        print(f"epoch {ep:3d}  lr={sched.get_last_lr()[0]:.2e}  val_loss={vl:.3f}  "
              f"mean_err={vme:.2f}mm  tip_err={vte:.2f}mm{flag}")
        if patience and since_best >= patience:
            print(f"early stop: no val improvement for {patience} epochs (best {best:.2f}mm)")
            break

    # final test metrics with the best checkpoint
    net.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    _, tme, tte = evaluate(net, te, device, lam)
    print(f"\nTEST  mean along-body error = {tme:.2f} mm   tip error = {tte:.2f} mm")
    print(f"best checkpoint: {ckpt}")


def overfit_test(cfg: dict) -> None:
    """Overfit 20 fixed frames; assert the loss collapses to near zero."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    h, w = cfg["dataset"]["image_size"]
    n = cfg["model"]["n_points"]
    N = 20
    # Distinct low-frequency patterns per frame (random uniform noise survives
    # global pooling as near-identical statistics, so the encoder couldn't tell
    # frames apart — a sinusoidal pattern with per-frame freq/phase is separable).
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    imgs = torch.empty(N, 1, h, w)
    gi = torch.Generator().manual_seed(7)
    for k in range(N):
        fx, fy = 1 + 6 * torch.rand(1, generator=gi), 1 + 6 * torch.rand(1, generator=gi)
        ph = 6.28 * torch.rand(1, generator=gi)
        imgs[k, 0] = 0.5 + 0.5 * torch.sin(6.28 * (fx * xx + fy * yy) + ph)
    # distinct, smooth ~100 mm arcs (one per frame) at ~250 mm depth, so the
    # net must read the image to tell frames apart but targets stay realistic.
    t = torch.linspace(0, 1, n)
    g = torch.Generator().manual_seed(1)
    pts = torch.empty(N, n, 3)
    for k in range(N):
        bend = (torch.rand(3, generator=g) - 0.5) * 80
        base = torch.tensor([0.0, 0.0, 250.0]) + (torch.rand(3, generator=g) - 0.5) * 40
        pts[k] = base + torch.stack([60 * t, 80 * t, 0 * t], 1) + bend * (t ** 2)[:, None]
    Ls = torch.linalg.norm(pts[:, 1:] - pts[:, :-1], dim=2).sum(1, keepdim=True)
    ds = TensorDataset(imgs, pts, Ls)
    loader = DataLoader(ds, batch_size=10, shuffle=True)

    net = MoSSNet(n, pretrained=False).to(device)
    # Disable dropout so the net can memorize exactly (overfit sanity check only).
    for m in net.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    # Cosine LR decay so the optimizer settles into the minimum instead of
    # bouncing around it at a fixed high LR.
    n_epochs = 3000
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    lam = cfg["train"]["lambda_length"]
    last = None
    for ep in range(n_epochs):
        net.train()
        for img, r_s, L in loader:
            img, r_s, L = img.to(device), r_s.to(device), L.to(device)
            opt.zero_grad()
            p, Lp = net(img)
            loss = loss_fn(p, Lp, r_s, L, lam)
            loss.backward()
            opt.step()
        sched.step()
        if ep % 200 == 0 or ep == n_epochs - 1:
            l, me, te = evaluate(net, loader, device, lam)
            print(f"  iter {ep:3d}  loss={l:.4f}  mean_err={me:.3f}mm tip_err={te:.3f}mm", flush=True)
            last = me
    assert last < 1.0, f"overfit mean error {last:.3f} mm not near zero"
    print(f"PASS: overfit 20 frames to mean error {last:.3f} mm (< 1.0 mm)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train MoSSNet")
    ap.add_argument("--overfit-test", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.overfit_test:
        overfit_test(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
