"""Shared utilities: seeding, logging, hardware info, device selection."""
from __future__ import annotations

import logging
import os
import platform
import random
import sys
from typing import Any

import numpy as np


def get_logger(name: str = "cassava", level: int = logging.INFO) -> logging.Logger:
    """Return a module logger with a single stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch. Full bitwise determinism is not guaranteed on all backends."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(pref: str = "auto"):
    """Resolve 'auto' | 'cpu' | 'cuda' to a torch.device (falls back to CPU if CUDA is unavailable)."""
    import torch

    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        get_logger().warning("CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(pref)


def hardware_info() -> dict[str, Any]:
    """Collect hardware/software details for the experiment log."""
    import torch

    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
