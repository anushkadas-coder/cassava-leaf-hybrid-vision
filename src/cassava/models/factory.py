"""Build a model from a config dictionary."""
from __future__ import annotations

import torch
from torch import nn

from cassava.features.frequency import frequency_feature_dim
from cassava.models.cnn import SmallCNN
from cassava.models.fusion import FusionNet
from cassava.models.transfer import TransferNet
from cassava.preprocessing.colorspace import representation_channels


def build_model(cfg: dict, n_classes: int) -> nn.Module:
    """Create the torch model for ``cfg['model']['kind']`` in {cnn, transfer, fusion}."""
    kind = cfg["model"]["kind"]
    in_ch = representation_channels(cfg["data"]["representation"])
    mc = cfg["model"]
    if kind == "cnn":
        return SmallCNN(in_ch, n_classes, mc["cnn"]["widths"], mc["cnn"]["dropout"])
    if kind == "transfer":
        t = mc["transfer"]
        return TransferNet(in_ch, n_classes, t["arch"], t["pretrained"], t["weights_path"], t["dropout"])
    if kind == "fusion":
        fu = mc["fusion"]
        if fu["backbone"] == "cnn":
            backbone: nn.Module = SmallCNN(in_ch, n_classes, mc["cnn"]["widths"], mc["cnn"]["dropout"])
        elif fu["backbone"] in {"resnet18", "efficientnet_b0"}:
            t = mc["transfer"]
            backbone = TransferNet(in_ch, n_classes, fu["backbone"], t["pretrained"], t["weights_path"], t["dropout"])
        else:
            raise ValueError("fusion.backbone must be cnn | resnet18 | efficientnet_b0")
        return FusionNet(backbone, frequency_feature_dim(cfg["frequency"]), n_classes, fu["freq_hidden"], fu["freq_dropout"], fu["control"])
    raise ValueError(f"Unknown model.kind '{kind}' for torch models (expected cnn | transfer | fusion)")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
