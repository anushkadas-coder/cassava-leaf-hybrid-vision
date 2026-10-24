import numpy as np
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix as sk_cm, f1_score, precision_score, recall_score

from cassava.evaluation.error_analysis import paired_comparison
from cassava.evaluation.metrics import bootstrap_ci, classification_report, confusion_matrix
from cassava.training.trainer import class_weights
import pandas as pd


def _data(n=400, k=5, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(k, n, p=[0.05, 0.1, 0.1, 0.55, 0.2])
    p = np.where(rng.random(n) < 0.7, y, rng.integers(0, k, n))
    return y, p, [f"c{i}" for i in range(k)]


def test_matches_sklearn():
    y, p, names = _data()
    rep = classification_report(y, p, names)
    labels = list(range(5))
    assert np.array_equal(rep["confusion_matrix"], sk_cm(y, p, labels=labels))
    assert rep["accuracy"] == pytest.approx(accuracy_score(y, p))
    assert rep["macro_f1"] == pytest.approx(f1_score(y, p, average="macro", labels=labels, zero_division=0))
    assert [rep["per_class"][n]["precision"] for n in names] == pytest.approx(precision_score(y, p, average=None, labels=labels, zero_division=0))
    assert [rep["per_class"][n]["recall"] for n in names] == pytest.approx(recall_score(y, p, average=None, labels=labels, zero_division=0))


def test_absent_class_scores_zero_no_nan():
    rep = classification_report(np.array([0, 0, 1]), np.array([0, 0, 1]), ["a", "b", "c"])
    assert rep["per_class"]["c"]["f1"] == 0.0 and np.isfinite(rep["macro_f1"])


def test_confusion_matrix_validation():
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 5]), np.array([0, 1]), 3)
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 1]), np.array([0]), 3)


def test_bootstrap_ci_brackets_point_estimate_and_is_reproducible():
    y, p, names = _data()
    rep = classification_report(y, p, names)
    ci = bootstrap_ci(y, p, 5, 300, seed=1)
    assert ci["macro_f1"][0] <= rep["macro_f1"] <= ci["macro_f1"][1] and ci["accuracy"][0] <= rep["accuracy"] <= ci["accuracy"][1]
    assert ci == bootstrap_ci(y, p, 5, 300, seed=1)
    assert len(ci["recall_per_class"]) == 5


def test_class_weights_inverse_frequency():
    w = class_weights(np.array([0] * 90 + [1] * 10), 2)
    assert w[1] / w[0] == pytest.approx(9.0) and w.mean() == pytest.approx(1.0)


def test_paired_comparison_counts():
    a = pd.DataFrame({"image_id": list("abcd"), "true": [0, 0, 1, 1], "pred": [0, 1, 1, 0], "confidence": 0.9})
    b = pd.DataFrame({"image_id": list("abcd"), "true": [0, 0, 1, 1], "pred": [0, 0, 0, 0], "confidence": 0.9})
    r = paired_comparison(a, b, ["x", "y"])
    assert r["fixed_by_b"] == 1 and r["broken_by_b"] == 1 and 0 <= r["mcnemar_exact_p"] <= 1
    with pytest.raises(ValueError):
        paired_comparison(a, b.iloc[:3], ["x", "y"])
