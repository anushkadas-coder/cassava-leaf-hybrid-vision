"""Feature-level fusion: CNN embedding concatenated with an MLP embedding of frequency descriptors.

    logits = Linear( concat( backbone.embed(x), MLP(freq) ) )

Controls (to separate "frequency information" from "more parameters/inputs"):
  control='shuffle': frequency vectors are permuted across the batch (informationless but identically distributed)
  control='zeros'  : frequency branch receives zeros
A real gain must beat these controls, not only the plain CNN.
"""
from __future__ import annotations

import torch
from torch import nn


class FusionNet(nn.Module):
    def __init__(self, backbone: nn.Module, freq_dim: int, n_classes: int, freq_hidden: int = 64, freq_dropout: float = 0.3, control: str = "none", dropout: float = 0.3):
        super().__init__()
        if control not in {"none", "shuffle", "zeros"}:
            raise ValueError("fusion.control must be none | shuffle | zeros")
        if freq_dim <= 0:
            raise ValueError("FusionNet needs frequency features (freq_dim > 0)")
        self.backbone, self.control = backbone, control
        self.freq_mlp = nn.Sequential(nn.Linear(freq_dim, freq_hidden), nn.ReLU(inplace=True), nn.Dropout(freq_dropout))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.embed_dim + freq_hidden, n_classes))

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.backbone.gradcam_layer

    def set_backbone_frozen(self, frozen: bool) -> None:
        if hasattr(self.backbone, "set_backbone_frozen"):
            self.backbone.set_backbone_frozen(frozen)

    def forward(self, x: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
        if f is None or f.numel() == 0:
            raise ValueError("FusionNet.forward requires the frequency feature tensor f")
        if self.control == "zeros":
            f = torch.zeros_like(f)
        elif self.control == "shuffle":
            g = torch.Generator(device="cpu").manual_seed(int(f.shape[0]) * 7919 + 13)
            f = f[torch.randperm(f.shape[0], generator=g).to(f.device)]
        z = torch.cat([self.backbone.embed(x), self.freq_mlp(f)], dim=1)
        return self.head(z)
