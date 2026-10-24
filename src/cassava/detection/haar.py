"""Educational building blocks of Viola-Jones: integral image and Haar-like rectangle features.

Integral image II(y, x) = sum of img[:y, :x]; any rectangle sum then costs 4 lookups:
    sum(r0:r1, c0:c1) = II[r1, c1] - II[r0, c1] - II[r1, c0] + II[r0, c0]
A Haar-like feature is a signed combination of rectangle sums, e.g. (left half) - (right half) responds to a vertical edge.
AdaBoost picks, from the ~10^5+ possible features in a window, weak classifiers (feature + threshold + polarity) with
minimum weighted error; a cascade chains boosted stages so that most negative windows are rejected cheaply.
See docs/viola_jones.md for the feasibility assessment on cassava images.
"""
from __future__ import annotations

import numpy as np


def integral_image(img: np.ndarray) -> np.ndarray:
    """Zero-padded integral image of shape (H+1, W+1), float64."""
    if img.ndim != 2:
        raise ValueError("integral_image expects a 2-D grayscale array")
    ii = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = img.astype(np.float64).cumsum(0).cumsum(1)
    return ii


def rect_sum(ii: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> float:
    """Sum of img[r0:r1, c0:c1] from the integral image."""
    return float(ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0])


def haar_two_rect_horizontal(ii: np.ndarray, r: int, c: int, h: int, w: int) -> float:
    """Left half minus right half over an h x (2w) window at (r, c): responds to vertical edges."""
    return rect_sum(ii, r, c, r + h, c + w) - rect_sum(ii, r, c + w, r + h, c + 2 * w)


def haar_two_rect_vertical(ii: np.ndarray, r: int, c: int, h: int, w: int) -> float:
    """Top half minus bottom half over a (2h) x w window at (r, c): responds to horizontal edges."""
    return rect_sum(ii, r, c, r + h, c + w) - rect_sum(ii, r + h, c, r + 2 * h, c + w)


def haar_three_rect(ii: np.ndarray, r: int, c: int, h: int, w: int) -> float:
    """Outer two thirds minus 2x centre over an h x (3w) window: responds to thin vertical lines/veins."""
    return rect_sum(ii, r, c, r + h, c + w) - rect_sum(ii, r, c + w, r + h, c + 2 * w) + rect_sum(ii, r, c + 2 * w, r + h, c + 3 * w)


def count_haar_features_two_rect(win_h: int, win_w: int) -> int:
    """Number of horizontal two-rectangle features in a window (illustrates the search-space size AdaBoost selects from)."""
    total = 0
    for w in range(1, win_w // 2 + 1):
        for h in range(1, win_h + 1):
            total += (win_h - h + 1) * (win_w - 2 * w + 1)
    return total
