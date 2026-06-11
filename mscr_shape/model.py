"""Phase 4 — MoSSNet-style monocular shape-sensing network.

A ResNet18 encoder (single-channel input) feeds two parallel decoder heads:
    (a) centerline head -> N x 3 points in the left-IR rectified camera frame (mm)
    (b) length head     -> scalar total length L (mm)

Input : image tensor (B, 1, H, W), values in [0, 1].
Output: (points (B, N, 3) mm, length (B, 1) mm).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class MoSSNet(nn.Module):
    def __init__(self, n_points: int = 64, pretrained: bool = True):
        super().__init__()
        self.n_points = n_points

        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)

        # Adapt the first conv to single-channel input, seeding it from the
        # pretrained RGB filters (mean across the colour axis).
        old = backbone.conv1
        new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = new

        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # -> (B,512,1,1)
        feat = 512

        self.point_head = nn.Sequential(
            nn.Linear(feat, 512), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, n_points * 3),
        )
        self.length_head = nn.Sequential(
            nn.Linear(feat, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        f = self.encoder(x).flatten(1)              # (B, 512)
        pts = self.point_head(f).view(-1, self.n_points, 3)
        length = self.length_head(f)                # (B, 1)
        return pts, length


def shape_errors(pred_pts: torch.Tensor, gt_pts: torch.Tensor):
    """Mean per-point (along-body) error and tip error, both in mm."""
    d = torch.linalg.norm(pred_pts - gt_pts, dim=2)   # (B, N)
    return d.mean(), d[:, -1].mean()


if __name__ == "__main__":
    from cfg import load_config
    cfg = load_config()
    h, w = cfg["dataset"]["image_size"]
    net = MoSSNet(cfg["model"]["n_points"], pretrained=False)
    x = torch.randn(2, 1, h, w)
    pts, L = net(x)
    print(f"input {tuple(x.shape)} -> points {tuple(pts.shape)}, length {tuple(L.shape)}")
    me, te = shape_errors(pts, torch.randn_like(pts))
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params={n_params/1e6:.2f}M  mean_err={me.item():.2f} tip_err={te.item():.2f}")
    assert pts.shape == (2, cfg["model"]["n_points"], 3) and L.shape == (2, 1)
    print("OK")
