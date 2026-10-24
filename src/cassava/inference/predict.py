"""Single-image and batch inference from a saved checkpoint (same Preprocessor as training)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from cassava.data.cache import resize_square
from cassava.detection.leaf_mask import localize
from cassava.training.checkpoint import load_checkpoint


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, self.pre, self.cfg, self.class_names, _ = load_checkpoint(checkpoint, self.device)
        self.size = int(self.cfg["data"]["image_size"])
        self.localization = self.cfg["data"]["localization"]
        if self.localization == "cascade":
            raise NotImplementedError("Checkpoints trained with localization='cascade' need the detector at inference; not supported yet.")

    def load_image(self, path_or_array: str | Path | np.ndarray) -> np.ndarray:
        """Return a model-ready uint8 RGB array (size x size), validating the input."""
        if isinstance(path_or_array, np.ndarray):
            arr = path_or_array
            if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
                raise ValueError(f"Array input must be uint8 (H, W, 3); got {arr.shape} {arr.dtype}")
            img = resize_square(Image.fromarray(arr), self.size)
        else:
            p = Path(path_or_array)
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")
            try:
                with Image.open(p) as im:
                    img = resize_square(im, self.size)
            except (UnidentifiedImageError, OSError) as e:
                raise ValueError(f"Could not read '{p}' as an image: {e}") from e
        return localize(img, self.localization)

    @torch.no_grad()
    def predict_proba(self, inputs: list) -> np.ndarray:
        if not inputs:
            raise ValueError("No inputs given")
        xs, fs = zip(*(self.pre(self.load_image(i)) for i in inputs))
        logits = self.model(torch.stack(xs).to(self.device), torch.stack(fs).to(self.device))
        return torch.softmax(logits, 1).cpu().numpy()

    def predict(self, inputs: list) -> list[dict]:
        """List of {label, class_index, confidence, probabilities{class: p}} per input."""
        probs = self.predict_proba(inputs)
        return [
            {"label": self.class_names[int(p.argmax())], "class_index": int(p.argmax()), "confidence": float(p.max()),
             "probabilities": {c: float(v) for c, v in zip(self.class_names, p)}}
            for p in probs
        ]
