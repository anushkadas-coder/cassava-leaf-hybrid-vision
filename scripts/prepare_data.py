#!/usr/bin/env python
"""Audit a dataset, detect near-duplicates, build leak-free stratified splits and the resized image cache.

Example:  python scripts/prepare_data.py --data-root data/cassava --cache-dir data/cache --image-size 128
Outputs (in --cache-dir): images_<S>.npy, splits.csv, class_names.json, audit_report.json
Keep --seed fixed across ALL experiments so every run sees identical train/val/test partitions.
"""
import argparse
import json

from cassava.data.prepare import prepare_dataset
from cassava.data.records import DatasetError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/cassava")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42, help="split seed")
    ap.add_argument("--max-hamming", type=int, default=4, help="dHash Hamming distance treated as near-duplicate")
    ap.add_argument("--drop-conflicting-groups", action="store_true", help="drop duplicate groups whose members have different labels")
    a = ap.parse_args()
    try:
        report = prepare_dataset(a.data_root, a.cache_dir, a.image_size, a.val_frac, a.test_frac, a.seed, a.max_hamming, a.drop_conflicting_groups)
    except (DatasetError, ValueError) as e:
        raise SystemExit(f"ERROR: {e}")
    print(json.dumps(report, indent=2, default=int))


if __name__ == "__main__":
    main()
