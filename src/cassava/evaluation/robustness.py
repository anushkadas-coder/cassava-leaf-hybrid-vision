"""Synthetic image perturbations for robustness testing (uint8 RGB -> uint8 RGB).

These simulate lighting/colour/quality shifts on a CLEAN test set. They are controlled stress tests,
not a substitute for real-world field validation.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

ImageFn = Callable[[np.ndarray], np.ndarray]


def brightness(factor: float) -> ImageFn:
    return lambda im: np.clip(im.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def gamma(g: float) -> ImageFn:
    lut = (255.0 * (np.arange(256) / 255.0) ** g).clip(0, 255).astype(np.uint8)
    return lambda im: lut[im]


def white_balance(gains: tuple[float, float, float]) -> ImageFn:
    gv = np.asarray(gains, dtype=np.float32)
    return lambda im: np.clip(im.astype(np.float32) * gv, 0, 255).astype(np.uint8)


def gaussian_blur(sigma: float) -> ImageFn:
    return lambda im: cv2.GaussianBlur(im, (0, 0), sigma)


def gaussian_noise(std: float, seed: int = 0) -> ImageFn:
    def fn(im: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng([seed, int(im.sum()) % (2**31)])  # deterministic per image
        return np.clip(im.astype(np.float32) + rng.normal(0, std, im.shape), 0, 255).astype(np.uint8)

    return fn


def jpeg(quality: int) -> ImageFn:
    def fn(im: np.ndarray) -> np.ndarray:
        ok, enc = cv2.imencode(".jpg", im[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].copy() if ok else im

    return fn


def background_blank(mode: str = "background") -> ImageFn:
    """Shortcut test: blank the non-leaf ('background') or leaf ('leaf') pixels using the HSV leaf mask."""
    from cassava.detection.leaf_mask import leaf_mask_hsv

    def fn(im: np.ndarray) -> np.ndarray:
        m = leaf_mask_hsv(im)
        out = im.copy()
        out[~m if mode == "background" else m] = 127
        return out

    return fn


def default_suite() -> list[tuple[str, str, ImageFn]]:
    """(family, level, fn) triples used by the experiment runner."""
    s: list[tuple[str, str, ImageFn]] = []
    for f in (0.6, 0.8, 1.2, 1.4):
        s.append(("brightness", f"x{f}", brightness(f)))
    for g in (0.6, 1.6):
        s.append(("gamma", f"g{g}", gamma(g)))
    s.append(("white_balance", "warm", white_balance((1.15, 1.0, 0.85))))
    s.append(("white_balance", "cool", white_balance((0.85, 1.0, 1.15))))
    for sg in (1.0, 2.0):
        s.append(("blur", f"sigma{sg}", gaussian_blur(sg)))
    for sd in (8.0, 20.0):
        s.append(("noise", f"std{sd}", gaussian_noise(sd)))
    s.append(("jpeg", "q20", jpeg(20)))
    return s
