"""Save and restore a self-contained checkpoint (weights + config + preprocessing state + class names)."""
from __future__ import annotations

from pathlib import Path

import torch

from cassava.models.factory import build_model
from cassava.preprocessing.preprocessor import Preprocessor


def save_checkpoint(path: str | Path, model, cfg: dict, preprocessor: Preprocessor, class_names: list[str], extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": cfg, "preprocessor": preprocessor.state_dict(), "class_names": class_names, "extra": extra or {}},
        path,
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Return (model in eval mode, Preprocessor, config, class_names, extra)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:  # corrupt or incompatible file
        raise RuntimeError(f"Could not read checkpoint '{path}': {e}") from e
    for key in ("state_dict", "config", "preprocessor", "class_names"):
        if key not in ck:
            raise RuntimeError(f"Checkpoint '{path}' is missing '{key}'")
    cfg = ck["config"]
    cfg = {**cfg, "model": {**cfg["model"], "transfer": {**cfg["model"]["transfer"], "pretrained": False, "weights_path": None}}}
    model = build_model(cfg, len(ck["class_names"]))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, Preprocessor.from_state_dict(ck["preprocessor"]), cfg, ck["class_names"], ck.get("extra", {})
