"""Handcrafted colour descriptors for the classical baseline: per-channel histograms and moments."""
from __future__ import annotations

import numpy as np

from cassava.preprocessing.colorspace import convert, representation_channels

_RANGES = {
    "rgb": [(0, 1)] * 3,
    "hsv": [(0, 1)] * 3,
    "hsv_circ": [(-1, 1), (-1, 1), (0, 1), (0, 1)],
    "lab": [(0, 1), (-1, 1), (-1, 1)],
    "rgb_lab": [(0, 1)] * 3 + [(0, 1), (-1, 1), (-1, 1)],
}


def color_feature_vector(rgb: np.ndarray, representation: str = "lab", bins: int = 16) -> np.ndarray:
    """Normalised per-channel histograms (fixed ranges) + mean/std in the chosen representation."""
    img = convert(rgb, representation)
    feats = []
    for c, rng in enumerate(_RANGES[representation]):
        ch = img[..., c].ravel()
        hist, _ = np.histogram(ch, bins=bins, range=rng)
        feats.append(hist / max(1, ch.size))
        feats.append(np.array([ch.mean(), ch.std()]))
    return np.concatenate(feats).astype(np.float32)


def color_feature_dim(representation: str, bins: int) -> int:
    return representation_channels(representation) * (bins + 2)
