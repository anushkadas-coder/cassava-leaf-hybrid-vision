"""Transfer-learning baseline (baseline 2): torchvision ResNet-18 or EfficientNet-B0.

Pretrained weights
------------------
``pretrained: true`` makes torchvision download ImageNet weights on first use (needs internet). Offline, pass
``weights_path`` to a local ``state_dict`` file or set ``pretrained: false``. The torchvision code is BSD-3; the
ImageNet-trained weights carry the terms of the ImageNet dataset (research use) - check them before any commercial use.

Non-RGB inputs: the first convolution is rebuilt for C input channels. With pretrained weights the new kernel is
the mean of the pretrained RGB kernels repeated over channels and rescaled by 3/C. This is a heuristic: HSV/Lab inputs
are not what the weights were trained on, which is a confound to report when comparing representations.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tvm


class TransferNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        arch: str = "resnet18",
        pretrained: bool = True,
        weights_path: str | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if arch == "resnet18":
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained and not weights_path else None)
            if weights_path:
                m.load_state_dict(torch.load(weights_path, map_location="cpu"))
            if in_channels != 3:
                m.conv1 = self._adapt_conv(m.conv1, in_channels, pretrained)
            self.body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
            self.embed_dim = m.fc.in_features
            self._gradcam = m.layer4[-1]
        elif arch == "efficientnet_b0":
            m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained and not weights_path else None)
            if weights_path:
                m.load_state_dict(torch.load(weights_path, map_location="cpu"))
            if in_channels != 3:
                first = m.features[0][0]
                m.features[0][0] = self._adapt_conv(first, in_channels, pretrained)
            self.body = m.features
            self.embed_dim = m.classifier[1].in_features
            self._gradcam = m.features[-1]
        else:
            raise ValueError(f"Unknown arch '{arch}' (use resnet18 or efficientnet_b0)")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.embed_dim, n_classes))
        self._frozen = False

    @staticmethod
    def _adapt_conv(conv: nn.Conv2d, in_channels: int, pretrained: bool) -> nn.Conv2d:
        new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
        if pretrained:
            with torch.no_grad():
                w = conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) * (3.0 / in_channels)
                new.weight.copy_(w)
        return new

    @property
    def gradcam_layer(self) -> nn.Module:
        return self._gradcam

    def set_backbone_frozen(self, frozen: bool) -> None:
        self._frozen = frozen
        for p in self.body.parameters():
            p.requires_grad = not frozen

    def train(self, mode: bool = True):  # keep BN statistics fixed while the backbone is frozen
        super().train(mode)
        if self._frozen:
            self.body.eval()
        return self

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.body(x)).flatten(1)

    def forward(self, x: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.embed(x))
