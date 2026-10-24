"""Deterministic-given-RNG augmentation on uint8 RGB images (applied before colour conversion).

Photometric augmentation is OFF by default: it directly competes with the illumination-robustness
hypothesis for colour spaces, so run the colour-space study with and without it (see docs/experiments.md).
"""
from __future__ import annotations

import cv2
import numpy as np


class Augmenter:
    """Random flips, rotation+zoom (single affine warp) and optional photometric jitter."""

    def __init__(self, cfg: dict):
        self.hflip = bool(cfg.get("hflip", True))
        self.vflip = bool(cfg.get("vflip", True))
        self.rotate = float(cfg.get("rotate", 0.0))
        lo, hi = cfg.get("scale", [1.0, 1.0])
        if not (0 < lo <= hi <= 1.0):
            raise ValueError("augment.scale must satisfy 0 < lo <= hi <= 1 (fraction of the image kept)")
        self.scale = (float(lo), float(hi))
        self.photometric = bool(cfg.get("photometric", False))
        self.strength = float(cfg.get("photometric_strength", 0.2))

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.hflip and rng.random() < 0.5:
            img = img[:, ::-1]
        if self.vflip and rng.random() < 0.5:
            img = img[::-1]
        img = np.ascontiguousarray(img)
        if self.rotate > 0 or self.scale != (1.0, 1.0):
            h, w = img.shape[:2]
            angle = rng.uniform(-self.rotate, self.rotate)
            keep = rng.uniform(*self.scale)
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0 / keep)
            img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        if self.photometric:
            img = photometric_jitter(img, rng, self.strength)
        return img


def photometric_jitter(img: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Random brightness, contrast and per-channel gain (simulated white-balance shift)."""
    x = img.astype(np.float32)
    brightness = rng.uniform(1 - strength, 1 + strength)
    contrast = rng.uniform(1 - strength, 1 + strength)
    gains = rng.uniform(1 - strength / 2, 1 + strength / 2, size=3).astype(np.float32)
    mean = x.mean()
    x = (x - mean) * contrast + mean
    x = x * brightness * gains
    return np.clip(x, 0, 255).astype(np.uint8)
