"""Dataset audit: invalid/missing files, image sizes, exact and near-duplicate detection.

Near-duplicates are found with a 64-bit difference hash (dHash) plus Hamming distance.
dHash is robust to resizing, mild brightness change and recompression, which is the typical
way duplicate photos arise in field datasets. It is NOT robust to crops or rotations, so
"no duplicates found" does not prove there is no leakage.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from cassava.utils import get_logger

log = get_logger()


def dhash64(img: Image.Image) -> int:
    """64-bit difference hash: compare horizontally adjacent pixels of a 9x8 grayscale thumbnail."""
    small = np.asarray(img.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def inspect_images(df: pd.DataFrame) -> pd.DataFrame:
    """Open every image once; add columns: valid, width, height, md5, dhash (uint64 as python int)."""
    out = df.copy()
    valid, widths, heights, md5s, hashes = [], [], [], [], []
    for p in out["path"]:
        try:  # compute everything first so a failure cannot leave the column lists misaligned
            with open(p, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            with Image.open(p) as im:
                im.load()
                w, h, dh = im.width, im.height, dhash64(im)
        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError):
            valid.append(False)
            widths.append(0)
            heights.append(0)
            md5s.append("")
            hashes.append(0)
            continue
        valid.append(True)
        widths.append(w)
        heights.append(h)
        md5s.append(md5)
        hashes.append(dh)
    out["valid"] = valid
    out["width"] = widths
    out["height"] = heights
    out["md5"] = md5s
    out["dhash"] = np.array(hashes, dtype=np.uint64)
    return out


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return lut[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1)


def duplicate_groups(hashes: np.ndarray, max_hamming: int = 4, chunk: int = 512) -> np.ndarray:
    """Cluster items whose dHash Hamming distance <= max_hamming (single-linkage via union-find).

    Returns an int array ``group`` of the same length; items in the same cluster share an id.
    Complexity O(n^2) in chunks, fine for ~20k images (tens of seconds).
    """
    n = len(hashes)
    hashes = np.asarray(hashes, dtype=np.uint64)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for s in range(0, n, chunk):
        block = hashes[s : s + chunk]
        dist = _popcount(block[:, None] ^ hashes[None, :])
        ii, jj = np.nonzero(dist <= max_hamming)
        for a, b in zip(ii + s, jj):
            if a < b:
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parent[rb] = ra
    roots = np.array([find(i) for i in range(n)])
    _, group = np.unique(roots, return_inverse=True)
    return group


def build_audit_report(df: pd.DataFrame, class_names: list[str]) -> dict:
    """Summarise an inspected dataframe that already has a ``group`` column."""
    valid = df[df["valid"]]
    counts = Counter(valid["label"].tolist())
    sizes = Counter(zip(valid["width"].tolist(), valid["height"].tolist()))
    grp = valid.groupby("group")
    multi = grp.filter(lambda g: len(g) > 1)
    conflicting = grp.filter(lambda g: g["label"].nunique() > 1)
    return {
        "n_records": int(len(df)),
        "n_invalid_or_missing": int((~df["valid"]).sum()),
        "invalid_examples": df.loc[~df["valid"], "image_id"].head(10).tolist(),
        "class_counts": {class_names[k]: int(v) for k, v in sorted(counts.items())},
        "class_fractions": {class_names[k]: round(v / max(1, len(valid)), 4) for k, v in sorted(counts.items())},
        "top_image_sizes": [{"size": list(k), "count": v} for k, v in sizes.most_common(5)],
        "n_exact_md5_duplicates": int(valid["md5"].duplicated().sum()),
        "n_images_in_duplicate_groups": int(len(multi)),
        "n_duplicate_groups": int(multi["group"].nunique()) if len(multi) else 0,
        "n_groups_with_conflicting_labels": int(conflicting["group"].nunique()) if len(conflicting) else 0,
        "note": "Near-duplicates detected with 64-bit dHash; crops/rotations of the same photo are not detected.",
    }
