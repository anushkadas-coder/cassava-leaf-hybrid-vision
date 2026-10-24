"""Compact CNN trained from scratch (baseline 1). Default config: ~100k parameters, CPU-friendly."""
from __future__ import annotations

import torch
from torch import nn


def _block(cin: int, cout: int, n_convs: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(n_convs):
        layers += [nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SmallCNN(nn.Module):
    """Conv blocks (3x3 conv-BN-ReLU, 2 convs in the last two blocks) -> global average pool -> dropout -> linear."""

    def __init__(self, in_channels: int, n_classes: int, widths: list[int] | None = None, dropout: float = 0.3):
        super().__init__()
        widths = widths or [16, 32, 48, 64]
        blocks, cin = [], in_channels
        for i, w in enumerate(widths):
            blocks.append(_block(cin, w, 2 if i >= len(widths) - 2 else 1))
            cin = w
        self.body = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed_dim = widths[-1]
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(widths[-1], n_classes))

    @property
    def gradcam_layer(self) -> nn.Module:
        """Last conv block before pooling (its output is the feature map used by Grad-CAM)."""
        return self.body[-1][-2]  # final ReLU, before the last max-pool

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.body(x)).flatten(1)

    def forward(self, x: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.embed(x))
