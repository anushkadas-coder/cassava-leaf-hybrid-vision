"""Resize-once image cache stored as one uint8 .npy array (memory-mappable)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def resize_square(img: Image.Image, size: int) -> np.ndarray:
    """Resize to size x size (aspect ratio not preserved; documented in docs/limitations.md)."""
    return np.asarray(img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def build_cache(paths: list[str], size: int, out_file: str | Path) -> Path:
    """Write array (N, size, size, 3) uint8 incrementally to disk."""
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(out_file, mode="w+", dtype=np.uint8, shape=(len(paths), size, size, 3))
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            arr[i] = resize_square(im, size)
    arr.flush()
    del arr
    return out_file


def load_cache(out_file: str | Path) -> np.ndarray:
    """Open cache read-only as a memmap."""
    out_file = Path(out_file)
    if not out_file.exists():
        raise FileNotFoundError(f"Image cache '{out_file}' not found. Run scripts/prepare_data.py first.")
    return np.load(out_file, mmap_mode="r")


def prepared_paths(cache_dir: str | Path, size: int, splits_file: str | None = None) -> tuple[Path, Path]:
    cache_dir = Path(cache_dir)
    return cache_dir / f"images_{size}.npy", Path(splits_file) if splits_file else cache_dir / "splits.csv"


def load_prepared(cache_dir: str | Path, size: int, splits_file: str | None = None) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Load (images, splits dataframe, class_names). The dataframe row i corresponds to images[i]."""
    import json

    img_file, split_file = prepared_paths(cache_dir, size, splits_file)
    if not split_file.exists():
        raise FileNotFoundError(f"Splits file '{split_file}' not found. Run scripts/prepare_data.py first.")
    df = pd.read_csv(split_file, keep_default_na=False)
    with open(Path(cache_dir) / "class_names.json") as f:
        class_names = json.load(f)
    images = load_cache(img_file)
    if len(images) != len(df):
        raise ValueError(f"Cache has {len(images)} images but splits file has {len(df)} rows; re-run prepare_data.py.")
    return images, df, class_names
