"""Frequency-domain feature extraction (FFT band energies / spectral statistics, DWT subband statistics).

Purpose and caveats
-------------------
* The 2-D DFT of a (Hann-windowed) grayscale image describes how much image energy lies at each spatial
  frequency/orientation. Radial band energies summarise coarse (low-frequency: illumination, large shapes) versus
  fine (high-frequency: edges, speckle, texture) content. A global spectrum discards *where* things happen, so it is
  a texture descriptor, NOT a lesion detector.
* The DWT (default Daubechies-2, 3 levels) keeps spatial localisation: per level there are horizontal, vertical and
  diagonal detail subbands. We summarise each by log-energy, standard deviation and mean absolute value.
* Whether either helps cassava classification is an empirical question tested in docs/experiments.md.
"""
from __future__ import annotations

import cv2
import numpy as np
import pywt

EPS = 1e-8


def to_channel(rgb: np.ndarray, channel: str = "gray") -> np.ndarray:
    """Single float32 channel in [0, 1] from uint8 RGB: 'gray' (luma) or 'L' (CIELAB lightness / 100)."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) uint8 RGB, got {rgb.shape} {rgb.dtype}")
    if channel == "gray":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if channel == "L":
        return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)[..., 0] / 100.0
    raise ValueError(f"Unknown frequency channel '{channel}' (use 'gray' or 'L').")


def fft_log_magnitude(gray: np.ndarray, window: bool = True) -> np.ndarray:
    """Centered log(1 + |F|) spectrum of a Hann-windowed image (window reduces edge leakage)."""
    g = gray - gray.mean()
    if window:
        g = g * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)
    spec = np.fft.fftshift(np.fft.fft2(g))
    return np.log1p(np.abs(spec)).astype(np.float32)


def _polar_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w]
    fy, fx = (yy - h // 2) / (h / 2), (xx - w // 2) / (w / 2)
    radius = np.sqrt(fx**2 + fy**2) / np.sqrt(2.0)  # in [0, ~1]
    angle = np.mod(np.arctan2(fy, fx), np.pi)  # orientation modulo 180 degrees
    return radius, angle


def fft_features(gray: np.ndarray, n_radial: int = 8, n_angular: int = 4) -> np.ndarray:
    """Relative radial band energies, angular energies, plus spectral centroid, entropy and 1/f slope.

    Dimensions: n_radial + n_angular + 3 (+1 total log energy).
    """
    g = gray - gray.mean()
    g = g * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)
    power = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = power.shape
    radius, angle = _polar_grid(h, w)
    power[h // 2, w // 2] = 0.0  # drop DC
    total = float(power.sum()) + EPS
    r_edges = np.linspace(0, radius.max() + 1e-6, n_radial + 1)
    r_idx = np.clip(np.digitize(radius, r_edges) - 1, 0, n_radial - 1)
    radial = np.bincount(r_idx.ravel(), weights=power.ravel(), minlength=n_radial) / total
    a_idx = np.clip((angle / np.pi * n_angular).astype(int), 0, n_angular - 1)
    angular = np.bincount(a_idx.ravel(), weights=power.ravel(), minlength=n_angular) / total
    p = power / total
    centroid = float((radius * p).sum())
    entropy = float(-(p * np.log(p + EPS)).sum() / np.log(p.size))
    # radial power spectrum slope in log-log (natural images roughly follow 1/f^2 power)
    centers = 0.5 * (r_edges[1:] + r_edges[:-1])
    ring_mean = np.bincount(r_idx.ravel(), weights=power.ravel(), minlength=n_radial) / (
        np.bincount(r_idx.ravel(), minlength=n_radial) + EPS
    )
    slope = float(np.polyfit(np.log(centers + EPS), np.log(ring_mean + EPS), 1)[0])
    return np.concatenate([np.log(radial + EPS), np.log(angular + EPS), [centroid, entropy, slope, np.log(total)]]).astype(np.float32)


def dwt_features(gray: np.ndarray, wavelet: str = "db2", level: int = 3) -> np.ndarray:
    """Per-subband log-energy, std and mean|coef| for each detail level, plus log energy of the approximation.

    Dimensions: level * 3 (H, V, D) * 3 statistics + 1.
    """
    max_level = pywt.dwt_max_level(min(gray.shape), pywt.Wavelet(wavelet).dec_len)
    if level > max_level:
        raise ValueError(f"wavelet_level={level} too large for image {gray.shape} (max {max_level})")
    coeffs = pywt.wavedec2(gray, wavelet=wavelet, level=level)
    feats: list[float] = [float(np.log(np.mean(coeffs[0] ** 2) + EPS))]
    for detail in coeffs[1:]:
        for sub in detail:
            feats += [float(np.log(np.mean(sub**2) + EPS)), float(sub.std()), float(np.abs(sub).mean())]
    return np.asarray(feats, dtype=np.float32)


def frequency_feature_dim(cfg: dict) -> int:
    d = 0
    if cfg.get("use_fft", True):
        d += cfg["n_radial_bands"] + cfg["n_angular_bands"] + 4
    if cfg.get("use_dwt", True):
        d += cfg["wavelet_level"] * 9 + 1
    if d == 0:
        raise ValueError("frequency config enables neither FFT nor DWT features")
    return d


def frequency_feature_vector(rgb: np.ndarray, cfg: dict) -> np.ndarray:
    """Concatenated FFT and/or DWT descriptors of a uint8 RGB image. Always finite (eps-guarded logs)."""
    gray = to_channel(rgb, cfg.get("channel", "gray"))
    parts = []
    if cfg.get("use_fft", True):
        parts.append(fft_features(gray, cfg["n_radial_bands"], cfg["n_angular_bands"]))
    if cfg.get("use_dwt", True):
        parts.append(dwt_features(gray, cfg["wavelet"], cfg["wavelet_level"]))
    if not parts:
        raise ValueError("frequency config enables neither FFT nor DWT features")
    return np.concatenate(parts)
