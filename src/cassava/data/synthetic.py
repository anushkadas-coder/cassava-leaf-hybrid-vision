"""Synthetic cassava-LIKE dataset for tests and pipeline smoke checks ONLY.

These images are procedurally generated blobs, not leaves. Any accuracy measured on them says nothing about
real cassava performance; they exist so the full pipeline can run without downloading the real dataset.
Imbalanced class priors and injected duplicates mimic the structure of the real dataset (to exercise the audit).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CLASS_NAMES = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mottle (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy",
]
PRIORS = np.array([0.08, 0.12, 0.12, 0.45, 0.23])


def _render(label: int, rng: np.random.Generator, size: int = 96) -> np.ndarray:
    img = np.zeros((size, size, 3), np.float32)
    img[:] = np.array([110, 85, 60]) + rng.normal(0, 12, (size, size, 3))  # soil-like background
    mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(mask, (size // 2, size // 2), (int(size * rng.uniform(0.32, 0.42)), int(size * rng.uniform(0.2, 0.3))), rng.uniform(0, 180), 0, 360, 1, -1)
    leaf = np.array([50, 130, 45], np.float32) + rng.normal(0, 8, (size, size, 3))
    m = mask.astype(bool)
    if label == 0:  # dark brown spots
        for _ in range(rng.integers(6, 12)):
            cv2.circle(leaf, tuple(int(v) for v in rng.integers(size // 4, 3 * size // 4, 2)), int(rng.integers(2, 5)), (70, 45, 20), -1)
    elif label == 1:  # yellow streaks
        for _ in range(rng.integers(3, 6)):
            p = rng.integers(size // 4, 3 * size // 4, 4)
            cv2.line(leaf, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (200, 190, 60), 3)
    elif label == 2:  # fine high-frequency mottling
        noise = rng.normal(0, 40, (size, size, 1)).astype(np.float32)
        leaf += noise * np.array([1.0, 0.9, 0.2], np.float32)
    elif label == 3:  # large pale-yellow mosaic patches
        for _ in range(rng.integers(3, 6)):
            cv2.circle(leaf, tuple(int(v) for v in rng.integers(size // 4, 3 * size // 4, 2)), int(rng.integers(8, 16)), (170, 190, 60), -1)
    img[m] = leaf[m]
    img *= rng.uniform(0.75, 1.25)  # lighting variation
    return np.clip(img, 0, 255).astype(np.uint8)


def make_synthetic_dataset(root: str | Path, n: int = 300, seed: int = 0, n_duplicates: int = 12, size: int = 96) -> Path:
    """Write a Kaggle-layout dataset (train.csv, label map, train_images/) with injected exact+near duplicates."""
    from PIL import Image

    root = Path(root)
    (root / "train_images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    labels = rng.choice(len(CLASS_NAMES), size=n, p=PRIORS)
    rows = []
    for i, y in enumerate(labels):
        name = f"img_{i:05d}.jpg"
        Image.fromarray(_render(int(y), rng, size)).save(root / "train_images" / name, quality=95)
        rows.append({"image_id": name, "label": int(y)})
    for j in range(n_duplicates):  # exact copy and a slightly recompressed/resized near-duplicate
        src = rows[int(rng.integers(0, n))]
        name = f"dup_{j:03d}.jpg"
        im = Image.open(root / "train_images" / src["image_id"])
        if j % 2 == 0:
            im.save(root / "train_images" / name, quality=95)
        else:
            im.resize((size - 4, size - 4)).save(root / "train_images" / name, quality=80)
        rows.append({"image_id": name, "label": src["label"]})
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)
    with open(root / "label_num_to_disease_map.json", "w") as f:
        json.dump({str(i): c for i, c in enumerate(CLASS_NAMES)}, f, indent=2)
    return root
