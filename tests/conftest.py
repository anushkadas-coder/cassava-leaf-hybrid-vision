"""Shared fixtures: a tiny SYNTHETIC dataset prepared once per test session (no real data needed)."""
import numpy as np
import pytest

from cassava.config import load_config
from cassava.data.prepare import prepare_dataset
from cassava.data.synthetic import make_synthetic_dataset

SIZE = 48


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory):
    return make_synthetic_dataset(tmp_path_factory.mktemp("synth") / "raw", n=160, seed=1, n_duplicates=8, size=64)


@pytest.fixture(scope="session")
def prepared(synth_root, tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    report = prepare_dataset(synth_root, cache, image_size=SIZE, seed=3)
    return {"cache": cache, "report": report, "root": synth_root}


@pytest.fixture()
def tiny_cfg(prepared, tmp_path):
    """Minimal fast config pointing at the prepared synthetic cache."""
    cfg = load_config("configs/smoke.yaml")
    cfg["experiment"].update({"output_dir": str(tmp_path / "out"), "seed": 0})
    cfg["data"].update({"cache_dir": str(prepared["cache"]), "image_size": SIZE, "stats_samples": 50})
    cfg["train"].update({"epochs": 2, "batch_size": 16})
    cfg["eval"].update({"bootstrap": 20, "robustness": False, "latency_runs": 2})
    cfg["model"]["cnn"]["widths"] = [8, 8, 16, 16]
    return cfg


@pytest.fixture()
def rgb_image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)
