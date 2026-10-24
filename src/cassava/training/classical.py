"""Classical (non-neural) baseline: handcrafted frequency and/or colour features -> sklearn classifier.

Used to answer: how far do interpretable frequency/colour descriptors get without a CNN, and do they carry
information a CNN lacks (see the fusion experiment)? Normalisation is fitted on the training split only.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cassava.features.color import color_feature_vector
from cassava.features.frequency import frequency_feature_vector


def feature_function(cfg: dict):
    """Return fn(uint8 RGB) -> 1-D feature vector according to model.classical.features."""
    parts = cfg["model"]["classical"]["features"]
    bad = set(parts) - {"freq", "color"}
    if bad or not parts:
        raise ValueError(f"model.classical.features must be a non-empty subset of ['freq', 'color'], got {parts}")
    rep, bins = cfg["data"]["representation"], cfg["model"]["classical"]["color_bins"]

    def fn(rgb: np.ndarray) -> np.ndarray:
        out = []
        if "freq" in parts:
            out.append(frequency_feature_vector(rgb, cfg["frequency"]))
        if "color" in parts:
            out.append(color_feature_vector(rgb, rep if rep != "rgb_lab" else "lab", bins))
        return np.concatenate(out)

    return fn


def build_classifier(name: str, seed: int):
    if name == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    elif name == "svm":
        clf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
    elif name == "hgb":
        clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, max_iter=200)
    else:
        raise ValueError("model.classical.classifier must be logreg | svm | hgb")
    return make_pipeline(StandardScaler(), clf)
