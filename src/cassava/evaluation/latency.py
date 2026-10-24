"""Inference latency and size measurement (end-to-end including preprocessing, and model-only)."""
from __future__ import annotations

import time

import numpy as np
import torch

from cassava.models.factory import count_parameters, model_size_mb


@torch.no_grad()
def measure_latency(model: torch.nn.Module, preprocessor, sample_rgb: np.ndarray, runs: int = 30, warmup: int = 3) -> dict:
    """Batch-size-1 CPU latency in ms: 'preprocess', 'model', 'total'. Reported on CPU for comparability."""
    model = model.to("cpu").eval()
    pre_t, mod_t = [], []
    for i in range(runs + warmup):
        t0 = time.perf_counter()
        x, f = preprocessor(sample_rgb)
        t1 = time.perf_counter()
        model(x[None], f[None])
        t2 = time.perf_counter()
        if i >= warmup:
            pre_t.append(t1 - t0)
            mod_t.append(t2 - t1)
    pre, mod = float(np.median(pre_t) * 1e3), float(np.median(mod_t) * 1e3)
    return {
        "preprocess_ms": pre,
        "model_ms": mod,
        "total_ms": pre + mod,
        "n_params": count_parameters(model),
        "size_mb": model_size_mb(model),
        "device": "cpu",
    }
