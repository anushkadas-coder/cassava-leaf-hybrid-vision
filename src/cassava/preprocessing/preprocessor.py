"""The single preprocessing path shared by training, validation, evaluation and inference.

uint8 RGB (H, W, 3)  ->  representation (float32)  ->  standardisation  ->  torch tensor (C, H, W)
and optionally -> frequency descriptor vector -> standardisation.

Normalisation statistics are fitted on TRAINING images only and stored in checkpoints.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from cassava.features.frequency import frequency_feature_dim, frequency_feature_vector
from cassava.preprocessing.colorspace import convert, representation_channels

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Preprocessor:
    def __init__(self, representation: str = "rgb", normalize: str = "train_stats", freq_cfg: dict | None = None):
        self.representation = representation
        self.n_channels = representation_channels(representation)
        if normalize not in {"train_stats", "imagenet"}:
            raise ValueError(f"Unknown normalize mode '{normalize}'")
        if normalize == "imagenet" and representation != "rgb":
            raise ValueError("ImageNet normalisation is RGB-specific; use normalize='train_stats' for non-RGB inputs.")
        self.normalize = normalize
        self.freq_cfg = freq_cfg
        self.mean = IMAGENET_MEAN.copy() if normalize == "imagenet" else np.zeros(self.n_channels, np.float32)
        self.std = IMAGENET_STD.copy() if normalize == "imagenet" else np.ones(self.n_channels, np.float32)
        self.freq_dim = frequency_feature_dim(freq_cfg) if freq_cfg else 0
        self.freq_mean = np.zeros(self.freq_dim, np.float32)
        self.freq_std = np.ones(self.freq_dim, np.float32)
        self.fitted = normalize == "imagenet" and not freq_cfg

    def fit(self, images: Iterable[np.ndarray]) -> "Preprocessor":
        """Fit channel (and frequency) standardisation from an iterable of TRAIN uint8 RGB images."""
        s = np.zeros(self.n_channels, np.float64)
        s2 = np.zeros(self.n_channels, np.float64)
        count = 0
        feats = []
        for img in images:
            x = convert(img, self.representation).reshape(-1, self.n_channels)
            s += x.sum(0)
            s2 += (x**2).sum(0)
            count += x.shape[0]
            if self.freq_cfg:
                feats.append(frequency_feature_vector(img, self.freq_cfg))
        if count == 0:
            raise ValueError("Cannot fit Preprocessor on zero images")
        if self.normalize == "train_stats":
            self.mean = (s / count).astype(np.float32)
            self.std = np.sqrt(np.maximum(s2 / count - (s / count) ** 2, 1e-12)).astype(np.float32)
        if self.freq_cfg:
            f = np.stack(feats)
            self.freq_mean = f.mean(0).astype(np.float32)
            self.freq_std = np.maximum(f.std(0), 1e-6).astype(np.float32)
        self.fitted = True
        return self

    def image_array(self, rgb: np.ndarray) -> np.ndarray:
        x = convert(rgb, self.representation)
        return (x - self.mean) / self.std

    def __call__(self, rgb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (image tensor C,H,W, frequency tensor [freq_dim] (empty if no frequency branch))."""
        if not self.fitted:
            raise RuntimeError("Preprocessor.fit() must be called (on training data) before use.")
        x = torch.from_numpy(np.ascontiguousarray(self.image_array(rgb).transpose(2, 0, 1)))
        if self.freq_cfg:
            f = (frequency_feature_vector(rgb, self.freq_cfg) - self.freq_mean) / self.freq_std
            return x, torch.from_numpy(f.astype(np.float32))
        return x, torch.zeros(0)

    def state_dict(self) -> dict:
        return {
            "representation": self.representation,
            "normalize": self.normalize,
            "freq_cfg": self.freq_cfg,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "freq_mean": self.freq_mean.tolist(),
            "freq_std": self.freq_std.tolist(),
        }

    @classmethod
    def from_state_dict(cls, st: dict) -> "Preprocessor":
        p = cls(st["representation"], st["normalize"], st["freq_cfg"])
        p.mean = np.asarray(st["mean"], np.float32)
        p.std = np.asarray(st["std"], np.float32)
        p.freq_mean = np.asarray(st["freq_mean"], np.float32)
        p.freq_std = np.asarray(st["freq_std"], np.float32)
        p.fitted = True
        return p
