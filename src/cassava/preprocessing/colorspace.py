"""RGB -> HSV / CIELAB conversions with documented ranges.

Input: uint8 RGB image (H, W, 3). Output: float32 (H, W, C) with every channel scaled to
roughly [0, 1] (Lab a/b to roughly [-1, 1]) *before* dataset-level standardisation.

Conventions (OpenCV, float32 input in [0, 1]):
  HSV  : H in [0, 360) degrees, S, V in [0, 1]. We divide H by 360.
  Lab  : D65 white point, sRGB input. L in [0, 100], a/b roughly [-127, 127]. We divide by 100 / 128.

Hue is circular (0 and 1 are the same colour) and unstable when saturation is low, so besides the
plain ``hsv`` representation we offer ``hsv_circ`` = (sin 2*pi*h, cos 2*pi*h, s, v).
"""
from __future__ import annotations

import cv2
import numpy as np

REPRESENTATIONS: dict[str, int] = {"rgb": 3, "hsv": 3, "hsv_circ": 4, "lab": 3, "rgb_lab": 6}


def representation_channels(name: str) -> int:
    if name not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation '{name}'. Choose from {sorted(REPRESENTATIONS)}.")
    return REPRESENTATIONS[name]


def _check_rgb_uint8(img: np.ndarray) -> None:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB image, got shape {img.shape}")
    if img.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {img.dtype}")


def rgb_to_float(img: np.ndarray) -> np.ndarray:
    _check_rgb_uint8(img)
    return img.astype(np.float32) / 255.0


def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 HSV with H/360, S, V all in [0, 1]."""
    hsv = cv2.cvtColor(rgb_to_float(img), cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] % 360.0) / 360.0
    return hsv


def rgb_to_hsv_circ(img: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 (sin h, cos h, s, v); removes the hue wrap-around discontinuity."""
    hsv = rgb_to_hsv(img)
    ang = 2.0 * np.pi * hsv[..., 0]
    return np.stack([np.sin(ang), np.cos(ang), hsv[..., 1], hsv[..., 2]], axis=-1).astype(np.float32)


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 Lab scaled to L/100 in [0, 1] and a/128, b/128 in roughly [-1, 1]."""
    lab = cv2.cvtColor(rgb_to_float(img), cv2.COLOR_RGB2Lab)
    lab[..., 0] /= 100.0
    lab[..., 1:] /= 128.0
    return lab


def convert(img: np.ndarray, representation: str) -> np.ndarray:
    """Convert a uint8 RGB image to the requested representation (float32, H x W x C)."""
    representation_channels(representation)
    if representation == "rgb":
        return rgb_to_float(img)
    if representation == "hsv":
        return rgb_to_hsv(img)
    if representation == "hsv_circ":
        return rgb_to_hsv_circ(img)
    if representation == "lab":
        return rgb_to_lab(img)
    if representation == "rgb_lab":
        return np.concatenate([rgb_to_float(img), rgb_to_lab(img)], axis=-1)
    raise AssertionError(representation)  # pragma: no cover
