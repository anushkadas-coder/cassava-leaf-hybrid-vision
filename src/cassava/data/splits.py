"""Leak-free, stratified train/val/test splits.

Images belonging to one duplicate group are always placed in the same split
(StratifiedGroupKFold), so near-duplicate photos cannot straddle train and test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_splits(
    labels: np.ndarray, groups: np.ndarray, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> np.ndarray:
    """Return array of 'train'/'val'/'test' strings, grouped and (approximately) stratified."""
    if not (0 < val_frac < 0.5 and 0 < test_frac < 0.5 and val_frac + test_frac < 0.8):
        raise ValueError("val_frac and test_frac must be in (0, 0.5) and sum to < 0.8")
    n = len(labels)
    split = np.full(n, "train", dtype=object)

    def carve(idx: np.ndarray, frac: float, s: int) -> np.ndarray:
        k = max(2, int(round(1.0 / frac)))
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=s)
        _, held = next(sgkf.split(idx, labels[idx], groups[idx]))
        return idx[held]

    all_idx = np.arange(n)
    test_idx = carve(all_idx, test_frac, seed)
    split[test_idx] = "test"
    rest = all_idx[split != "test"]
    val_idx = carve(rest, val_frac / (1.0 - test_frac), seed + 1)
    split[val_idx] = "val"
    return split


def check_no_leakage(df: pd.DataFrame) -> dict:
    """Verify no duplicate group and no md5 spans multiple splits. Raises AssertionError on leakage."""
    grp_splits = df.groupby("group")["split"].nunique()
    md5_splits = df[df["md5"] != ""].groupby("md5")["split"].nunique()
    leaked_groups = int((grp_splits > 1).sum())
    leaked_md5 = int((md5_splits > 1).sum())
    if leaked_groups or leaked_md5:
        raise AssertionError(f"Leakage: {leaked_groups} duplicate groups and {leaked_md5} md5 hashes span splits.")
    return {"leaked_groups": 0, "leaked_md5": 0}
