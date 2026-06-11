#!/usr/bin/env python3
"""
unet.py — small U-Net for binary segmentation of a thin MSCR rod.

Design notes:
  * Encoder/decoder with skip connections to preserve thin (1–3 px wide)
    structures that a plain CNN would smear away.
  * base_ch=24 keeps the model small (~1–2 M params) so it runs well above
    30 fps on a CUDA GPU at the working resolution (320×192).
  * Output is a single-channel logit map; apply sigmoid for the mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    Compact 4-level U-Net.

    Args:
        in_ch   : input channels (3 = RGB, 4 = RGB+depth)
        base_ch : channels at the first level (doubles each level down)
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 24):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch*2, base_ch*4, base_ch*8

        self.inc   = DoubleConv(in_ch, c1)
        self.down1 = DoubleConv(c1, c2)
        self.down2 = DoubleConv(c2, c3)
        self.down3 = DoubleConv(c3, c4)
        self.pool  = nn.MaxPool2d(2)

        self.up3   = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.conv3 = DoubleConv(c4, c3)
        self.up2   = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.conv2 = DoubleConv(c3, c2)
        self.up1   = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.conv1 = DoubleConv(c2, c1)

        self.outc  = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)                 # c1, H
        x2 = self.down1(self.pool(x1))   # c2, H/2
        x3 = self.down2(self.pool(x2))   # c3, H/4
        x4 = self.down3(self.pool(x3))   # c4, H/8

        u3 = self.up3(x4)
        u3 = self.conv3(torch.cat([u3, x3], dim=1))
        u2 = self.up2(u3)
        u2 = self.conv2(torch.cat([u2, x2], dim=1))
        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, x1], dim=1))

        return self.outc(u1)             # (B, 1, H, W) logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = UNet(in_ch=3, base_ch=24)
    x = torch.randn(1, 3, 192, 320)
    y = m(x)
    print(f"UNet params: {count_params(m)/1e6:.2f} M")
    print(f"input  {tuple(x.shape)}  →  output {tuple(y.shape)}")
