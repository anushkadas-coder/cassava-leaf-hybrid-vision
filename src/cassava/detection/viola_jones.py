"""Optional leaf-localisation interface built on OpenCV's Haar cascade runtime.

STATUS: experimental interface. The Kaggle cassava dataset ships NO bounding boxes, so no cassava cascade can be
trained from it without first annotating leaves. This module therefore does NOT contain a trained detector; it
wraps a cascade XML that YOU train (e.g. opencv_traincascade) and provides the evaluation + fallback logic.
Do not use the stock frontal-face cascades: they detect faces, not leaves.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


def evaluate_detections(preds: list[list[Box]], gts: list[Box], iou_thr: float = 0.5) -> dict:
    """Detection metrics for one-ground-truth-box-per-image: a prediction is a TP if IoU >= thr with that box.

    Reports precision, recall (fraction of images with at least one TP), miss rate and mean IoU of the best box.
    Kept separate from disease-classification metrics on purpose.
    """
    if len(preds) != len(gts):
        raise ValueError("preds and gts must have equal length")
    tp = fp = hit_images = 0
    best_ious = []
    for p, g in zip(preds, gts):
        ious = [iou(b, g) for b in p]
        good = sum(i >= iou_thr for i in ious)
        tp += min(good, 1)
        fp += len(p) - min(good, 1)
        hit_images += int(good > 0)
        best_ious.append(max(ious) if ious else 0.0)
    n = len(gts)
    return {"precision": tp / max(1, tp + fp), "recall": hit_images / max(1, n), "miss_rate": 1 - hit_images / max(1, n), "mean_best_iou": float(np.mean(best_ious)) if best_ious else 0.0, "n_images": n}


class CascadeLeafDetector:
    """Wrap a user-trained OpenCV cascade; ``crop`` returns the best box crop or the full image if detection fails."""

    def __init__(self, cascade_path: str | Path, scale_factor: float = 1.1, min_neighbors: int = 4, min_size: int = 24):
        p = Path(cascade_path) if cascade_path else None
        if p is None or not p.exists():
            raise FileNotFoundError(
                "No cascade file. Viola-Jones needs a cascade trained on annotated cassava leaves "
                "(see docs/viola_jones.md); pass data.cascade_path=PATH to a trained XML."
            )
        try:
            self.cascade = cv2.CascadeClassifier(str(p))
            empty = self.cascade.empty()
        except (cv2.error, SystemError) as e:  # OpenCV signals malformed XML differently across versions
            raise ValueError(f"OpenCV could not load a cascade from '{p}': {e}") from e
        if empty:
            raise ValueError(f"OpenCV could not load a cascade from '{p}'")
        self.kw = {"scaleFactor": scale_factor, "minNeighbors": min_neighbors, "minSize": (min_size, min_size)}

    def detect(self, rgb: np.ndarray) -> list[Box]:
        gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        return [tuple(int(v) for v in b) for b in self.cascade.detectMultiScale(gray, **self.kw)]

    def crop(self, rgb: np.ndarray, min_area_frac: float = 0.1) -> tuple[np.ndarray, bool]:
        """Return (image, detected). Largest box is used; boxes under ``min_area_frac`` of the image count as a miss
        (fallback = unchanged image). The crop is resized back to the input size."""
        boxes = self.detect(rgb)
        h, w = rgb.shape[:2]
        if not boxes:
            return rgb, False
        x, y, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
        if bw * bh < min_area_frac * h * w:
            return rgb, False
        return cv2.resize(rgb[y : y + bh, x : x + bw], (w, h), interpolation=cv2.INTER_LINEAR), True
