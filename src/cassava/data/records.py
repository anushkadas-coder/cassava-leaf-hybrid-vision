"""Discover image records and labels from a local dataset directory.

Supported layouts (nothing is downloaded; see data/README.md):

1. Kaggle "Cassava Leaf Disease Classification" layout::

       root/train.csv                       columns: image_id,label (integer)
       root/label_num_to_disease_map.json   {"0": "Cassava Bacterial Blight (CBB)", ...}
       root/train_images/<image_id>

2. Class-folder layout::

       root/<class_name>/<image files>
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class DatasetError(RuntimeError):
    """Raised for missing or malformed dataset structure."""


def load_records(root: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Return (records, class_names). records has columns: image_id, path, label (int), class_name.

    Class names are taken from the dataset itself rather than hardcoded.
    """
    root = Path(root)
    if not root.exists():
        raise DatasetError(
            f"Dataset path '{root}' does not exist. See data/README.md for how to obtain the dataset "
            "and set data.root in the config (or pass --set data.root=PATH)."
        )
    csv_path = root / "train.csv"
    if csv_path.exists():
        return _from_csv(root, csv_path)
    subdirs = sorted(p for p in root.iterdir() if p.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in p.iterdir()))
    if subdirs:
        return _from_folders(subdirs)
    raise DatasetError(f"No train.csv and no class folders with images found under '{root}'.")


def _from_csv(root: Path, csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path)
    missing_cols = {"image_id", "label"} - set(df.columns)
    if missing_cols:
        raise DatasetError(f"{csv_path} is missing columns: {sorted(missing_cols)}")
    if df["label"].isna().any():
        raise DatasetError(f"{csv_path} has {int(df['label'].isna().sum())} rows with missing labels.")
    try:
        df["label"] = df["label"].astype(int)
    except (ValueError, TypeError) as e:
        raise DatasetError(f"{csv_path} has non-integer labels: {e}") from e
    map_path = root / "label_num_to_disease_map.json"
    if map_path.exists():
        with open(map_path) as f:
            mapping = {int(k): v for k, v in json.load(f).items()}
    else:
        mapping = {int(k): f"class_{int(k)}" for k in sorted(df["label"].unique())}
    if set(df["label"].unique()) - set(mapping):
        raise DatasetError(f"Labels {sorted(set(df['label'].unique()) - set(mapping))} not present in label map.")
    n_classes = max(mapping) + 1
    if sorted(mapping) != list(range(n_classes)):
        raise DatasetError(f"Label map keys must be contiguous 0..K-1, got {sorted(mapping)}")
    class_names = [mapping[i] for i in range(n_classes)]
    img_dir = root / "train_images" if (root / "train_images").exists() else root
    df["path"] = df["image_id"].map(lambda i: str(img_dir / str(i)))
    df["class_name"] = df["label"].map(lambda i: class_names[i])
    return df[["image_id", "path", "label", "class_name"]].reset_index(drop=True), class_names


def _from_folders(subdirs: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    class_names = [d.name for d in subdirs]
    rows = []
    for label, d in enumerate(subdirs):
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                rows.append({"image_id": f"{d.name}/{f.name}", "path": str(f), "label": label, "class_name": d.name})
    return pd.DataFrame(rows), class_names
