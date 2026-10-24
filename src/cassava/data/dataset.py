"""Torch dataset over the cached uint8 image array.

Order of operations per sample (all in uint8 RGB until the Preprocessor):
    cached image -> [perturbation (robustness tests)] -> [localisation] -> [augmentation (train only)] -> Preprocessor
Augmentation randomness is derived from (seed, epoch, index) so results do not depend on worker count.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from cassava.detection.leaf_mask import localize
from cassava.preprocessing.augment import Augmenter
from cassava.preprocessing.preprocessor import Preprocessor

ImageFn = Callable[[np.ndarray], np.ndarray]


class CassavaDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        indices: np.ndarray,
        labels: np.ndarray,
        preprocessor: Preprocessor,
        augmenter: Augmenter | None = None,
        localization: str = "none",
        localizer: ImageFn | None = None,
        perturb: ImageFn | None = None,
        seed: int = 0,
    ):
        if len(indices) != len(labels):
            raise ValueError("indices and labels must have equal length")
        self.images, self.indices, self.labels = images, np.asarray(indices), np.asarray(labels)
        self.pre, self.aug, self.localization, self.localizer, self.perturb = preprocessor, augmenter, localization, localizer, perturb
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.indices)

    def raw_image(self, i: int) -> np.ndarray:
        """uint8 RGB image after perturbation + localisation, before augmentation."""
        img = np.array(self.images[self.indices[i]])
        if self.perturb is not None:
            img = self.perturb(img)
        if self.localizer is not None:
            img = self.localizer(img)
        elif self.localization != "none":
            img = localize(img, self.localization)
        return img

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        img = self.raw_image(i)
        if self.aug is not None:
            rng = np.random.default_rng([self.seed, self.epoch, int(self.indices[i])])
            img = self.aug(img, rng)
        x, f = self.pre(img)
        return x, f, int(self.labels[i])
