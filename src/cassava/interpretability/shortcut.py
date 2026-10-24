"""Quantitative checks to accompany Grad-CAM: does the model rely on the background?

1. leaf_mass_fraction: share of Grad-CAM mass that falls inside the HSV leaf mask (weak proxy; see leaf_mask caveats).
2. Conditioned evaluation: macro-F1 on the original test images vs. background blanked vs. leaf blanked.
   * background blanked: a large drop suggests the model needs context outside the heuristic mask (or the mask removed
     diseased tissue); little drop suggests it relies on the leaf region.
   * leaf blanked: accuracy far above chance means the model can classify from background alone = shortcut warning
     (e.g. photos of one disease were taken at one farm/soil/lighting).
"""
from __future__ import annotations

import numpy as np

from cassava.detection.leaf_mask import leaf_mask_hsv


def leaf_mass_fraction(cam: np.ndarray, rgb: np.ndarray) -> tuple[float, float]:
    """Return (fraction of CAM mass inside the leaf mask, mask area fraction). Ratio >> 1 means focus on the leaf."""
    mask = leaf_mask_hsv(rgb)
    total = float(cam.sum())
    return (float(cam[mask].sum() / total) if total > 1e-12 else 0.0), float(mask.mean())


def conditioned_evaluation(predict_fn, idx: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict:
    """predict_fn(idx, perturb_fn_or_None) -> probs. Returns accuracy/macro-F1 for the three conditions."""
    from cassava.evaluation.metrics import classification_report
    from cassava.evaluation.robustness import background_blank

    out = {}
    for cond, fn in (("original", None), ("background_blanked", background_blank("background")), ("leaf_blanked", background_blank("leaf"))):
        rep = classification_report(labels[idx], predict_fn(idx, fn).argmax(1), class_names)
        out[cond] = {"accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"]}
    out["chance_accuracy_majority"] = float(np.bincount(labels[idx]).max() / len(idx))
    return out
