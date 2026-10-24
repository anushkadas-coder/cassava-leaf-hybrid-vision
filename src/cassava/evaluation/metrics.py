"""Classification metrics computed from the confusion matrix, plus percentile-bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if len(y_true) and (y_true.min() < 0 or y_pred.min() < 0 or y_true.max() >= n_classes or y_pred.max() >= n_classes):
        raise ValueError("labels out of range for n_classes")
    return np.bincount(y_true * n_classes + y_pred, minlength=n_classes**2).reshape(n_classes, n_classes)


def per_class_from_cm(cm: np.ndarray) -> dict[str, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(1).astype(np.float64)
    predicted = cm.sum(0).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def macro_f1_from_cm(cm: np.ndarray) -> float:
    """Unweighted mean F1 over all K classes; a class with no support and no predictions scores 0
    (identical to sklearn f1_score(average='macro', labels=range(K), zero_division=0))."""
    return float(per_class_from_cm(cm)["f1"].mean())


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    k = len(class_names)
    cm = confusion_matrix(y_true, y_pred, k)
    pc = per_class_from_cm(cm)
    n = int(cm.sum())
    return {
        "n": n,
        "accuracy": float(np.trace(cm) / n) if n else 0.0,
        "macro_f1": float(pc["f1"].mean()),
        "macro_precision": float(pc["precision"].mean()),
        "macro_recall": float(pc["recall"].mean()),
        "per_class": {
            class_names[i]: {m: float(pc[m][i]) for m in ("precision", "recall", "f1", "support")} for i in range(k)
        },
        "confusion_matrix": cm.tolist(),
    }


def bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> dict[str, list[float]]:
    """Percentile bootstrap CIs (resampling test images) for accuracy, macro-F1 and per-class recall."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    if n == 0 or n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    acc, mf1, rec = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        cm = confusion_matrix(y_true[idx], y_pred[idx], n_classes)
        pc = per_class_from_cm(cm)
        acc.append(np.trace(cm) / n)
        mf1.append(pc["f1"].mean())
        rec.append(pc["recall"])
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    rec = np.stack(rec)
    return {
        "accuracy": np.percentile(acc, q).tolist(),
        "macro_f1": np.percentile(mf1, q).tolist(),
        "recall_per_class": [np.percentile(rec[:, i], q).tolist() for i in range(n_classes)],
    }
