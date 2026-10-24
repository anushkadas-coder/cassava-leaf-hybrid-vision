"""Programmatic version of scripts/prepare_data.py (audit -> dedup groups -> leak-free splits -> image cache)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cassava.data.audit import build_audit_report, duplicate_groups, inspect_images
from cassava.data.cache import build_cache
from cassava.data.records import load_records
from cassava.data.splits import check_no_leakage, make_splits
from cassava.utils import get_logger

log = get_logger()


def prepare_dataset(
    data_root: str | Path,
    cache_dir: str | Path,
    image_size: int = 128,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    max_hamming: int = 4,
    drop_conflicting_groups: bool = False,
) -> dict:
    """Run the full preparation; writes images_<S>.npy, splits.csv, class_names.json, audit_report.json. Returns the report."""
    df, class_names = load_records(data_root)
    log.info("Found %d records, %d classes: %s", len(df), len(class_names), class_names)
    df = inspect_images(df)
    bad = int((~df["valid"]).sum())
    if bad:
        log.warning("%d invalid/missing images will be excluded (examples: %s)", bad, df.loc[~df["valid"], "image_id"].head(5).tolist())
    df = df[df["valid"]].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid images found.")
    df["group"] = duplicate_groups(df["dhash"].values, max_hamming)
    report = build_audit_report(df.assign(valid=True), class_names)
    report["n_invalid_or_missing"] = bad
    if drop_conflicting_groups:
        conf = df.groupby("group")["label"].nunique()
        df = df[~df["group"].isin(conf[conf > 1].index)].reset_index(drop=True)
        log.info("Dropped groups with conflicting labels; %d images remain", len(df))
    df["split"] = make_splits(df["label"].values, df["group"].values, val_frac, test_frac, seed)
    report["leakage_check"] = check_no_leakage(df)
    report["split_counts"] = {
        s: {class_names[c]: int(n) for c, n in zip(*np.unique(df.loc[df["split"] == s, "label"], return_counts=True))} for s in ("train", "val", "test")
    }
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    build_cache(df["path"].tolist(), image_size, cache_dir / f"images_{image_size}.npy")
    df.drop(columns=["path"]).to_csv(cache_dir / "splits.csv", index=False)  # no machine-specific paths
    (cache_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))
    (cache_dir / "audit_report.json").write_text(json.dumps(report, indent=2, default=int))
    return report
