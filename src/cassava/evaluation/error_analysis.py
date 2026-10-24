"""Error analysis from saved ``test_predictions.csv`` files (single run, or a paired comparison of two runs)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def top_confusions(pred_df: pd.DataFrame, class_names: list[str], n: int = 8) -> pd.DataFrame:
    wrong = pred_df[pred_df["true"] != pred_df["pred"]]
    c = wrong.groupby(["true", "pred"]).size().sort_values(ascending=False).head(n).reset_index(name="count")
    c["true"] = c["true"].map(lambda i: class_names[i])
    c["pred"] = c["pred"].map(lambda i: class_names[i])
    return c


def confident_errors(pred_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Highest-confidence misclassifications (candidates for label noise or genuine shortcuts)."""
    return pred_df[pred_df["true"] != pred_df["pred"]].sort_values("confidence", ascending=False).head(n)


def paired_comparison(a: pd.DataFrame, b: pd.DataFrame, class_names: list[str]) -> dict:
    """Compare run A (reference) with run B on the same test images.

    Returns per-class recall change, counts of images fixed (A wrong -> B right) / broken (A right -> B wrong),
    and an exact two-sided McNemar test p-value on the discordant pairs.
    """
    from scipy.stats import binomtest

    m = a.merge(b, on="image_id", suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError("The two prediction files must cover the same test images")
    ok_a, ok_b = (m["true_a"] == m["pred_a"]).values, (m["true_b"] == m["pred_b"]).values
    fixed, broken = int((~ok_a & ok_b).sum()), int((ok_a & ~ok_b).sum())
    p = float(binomtest(fixed, fixed + broken, 0.5).pvalue) if fixed + broken > 0 else 1.0
    rec = {}
    for i, c in enumerate(class_names):
        sel = m["true_a"].values == i
        rec[c] = {"recall_a": float(ok_a[sel].mean()) if sel.any() else float("nan"), "recall_b": float(ok_b[sel].mean()) if sel.any() else float("nan")}
        rec[c]["delta"] = rec[c]["recall_b"] - rec[c]["recall_a"]
    return {"n": int(len(m)), "fixed_by_b": fixed, "broken_by_b": broken, "mcnemar_exact_p": p, "per_class_recall": rec,
            "accuracy_a": float(ok_a.mean()), "accuracy_b": float(ok_b.mean())}
