"""Annotation-free leaf localisation by HSV thresholding (the baseline alternative to a trained detector).

Rationale: the dataset has no bounding boxes. Before investing in a trained detector (e.g. Viola-Jones) it is
cheap to ask whether *removing or cropping the background* changes classification at all.

Caveats: hue 45-150 deg keeps yellow-green..green leaf tissue and rejects brown soil (~15-40 deg). Brown necrotic or
very pale tissue at the leaf EDGE is therefore rejected too (enclosed lesions are kept by hole filling), and
green weeds still pass. This is a heuristic, not a segmenter - inspect masks on real images before trusting it. ``localize`` reports when the mask is implausible and falls back to the full image.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


def leaf_mask_hsv(rgb: np.ndarray, hue_deg: tuple[float, float] = (45.0, 150.0), min_sat: float = 0.18, min_val: float = 0.12) -> np.ndarray:
    """Boolean mask (H, W) of plausible leaf pixels: yellow-green..green hues, not grey/dark; cleaned morphologically."""
    hsv = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h >= hue_deg[0]) & (h <= hue_deg[1]) & (s >= min_sat) & (v >= min_val)).astype(np.uint8)
    k = max(3, (min(rgb.shape[:2]) // 32) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Keep enclosed non-green regions (e.g. brown necrotic lesions fully surrounded by leaf tissue).
    return binary_fill_holes(mask.astype(bool))


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) exclusive-end box around True pixels, or None if empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def localize(rgb: np.ndarray, mode: str, min_area: float = 0.05, max_area: float = 0.97) -> np.ndarray:
    """Apply 'none' | 'hsv_crop' | 'hsv_blank'. Returns an image of the same size and dtype.

    hsv_crop : crop to the mask's bounding box, resize back to the original size.
    hsv_blank: replace non-leaf pixels by the mean grey (127) -> removes background colour/texture cues.
    Fallback : if mask area fraction is outside [min_area, max_area] the image is returned unchanged.
    """
    if mode == "none":
        return rgb
    if mode not in {"hsv_crop", "hsv_blank"}:
        raise ValueError(f"Unknown localization mode '{mode}'")
    mask = leaf_mask_hsv(rgb)
    frac = float(mask.mean())
    if not (min_area <= frac <= max_area):
        return rgb  # fallback: mask implausible
    if mode == "hsv_blank":
        out = rgb.copy()
        out[~mask] = 127
        return out
    x0, y0, x1, y1 = mask_to_bbox(mask)  # type: ignore[misc]
    h, w = rgb.shape[:2]
    if (x1 - x0) < 8 or (y1 - y0) < 8:
        return rgb
    return cv2.resize(rgb[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_LINEAR)
