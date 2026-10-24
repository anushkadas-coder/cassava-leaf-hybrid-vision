"""YAML configuration with defaults, inheritance, dotted overrides and sweep expansion.

A config file may contain:
  extends: other.yaml          # optional, resolved relative to this file
  <any key from DEFAULTS>      # overrides
  seeds: [0, 1, 2]             # used by run_experiments.py
  sweep:
    variants:
      - name: rgb
        set: {data.representation: rgb}
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "experiment": {"name": "experiment", "seed": 0, "output_dir": "outputs", "log_file": "experiment_log.jsonl"},
    "data": {
        "root": "data/cassava",
        "cache_dir": "data/cache",
        "splits_file": None,
        "image_size": 128,
        "representation": "rgb",  # rgb | hsv | hsv_circ | lab | rgb_lab
        "normalize": "train_stats",  # train_stats | imagenet (rgb only)
        "localization": "none",  # none | hsv_crop | hsv_blank | cascade
        "cascade_path": None,
        "num_workers": 0,
        "subset": None,  # cap on number of training samples (debugging)
        "stats_samples": 2000,
        "augment": {
            "hflip": True,
            "vflip": True,
            "rotate": 30.0,
            "scale": [0.8, 1.0],
            "photometric": False,
            "photometric_strength": 0.2,
        },
    },
    "model": {
        "kind": "cnn",  # cnn | transfer | fusion | classical
        "cnn": {"widths": [16, 32, 48, 64], "dropout": 0.3},
        "transfer": {
            "arch": "resnet18",  # resnet18 | efficientnet_b0
            "pretrained": True,
            "weights_path": None,
            "dropout": 0.2,
            "freeze_epochs": 2,
            "backbone_lr_mult": 0.1,
        },
        "fusion": {"backbone": "cnn", "control": "none", "freq_hidden": 64, "freq_dropout": 0.3},
        "classical": {"features": ["freq"], "classifier": "logreg", "color_bins": 16},
    },
    "frequency": {
        "channel": "gray",  # gray | L
        "use_fft": True,
        "use_dwt": True,
        "n_radial_bands": 8,
        "n_angular_bands": 4,
        "wavelet": "db2",
        "wavelet_level": 3,
    },
    "train": {
        "epochs": 30,
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "scheduler": "cosine",  # cosine | plateau | none
        "label_smoothing": 0.0,
        "balance": "weighted_loss",  # weighted_loss | sampler | none
        "early_stopping_patience": 8,
        "monitor": "macro_f1",  # macro_f1 | accuracy | loss
        "grad_clip": 5.0,
        "device": "auto",
        "num_threads": None,
    },
    "eval": {
        "bootstrap": 1000,
        "robustness": False,
        "robustness_subset": 1000,
        "latency_runs": 30,
    },
    "seeds": [0],
    "sweep": {"variants": []},
}


def deep_update(base: dict, new: dict) -> dict:
    """Recursively merge ``new`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    """Set cfg['a']['b'] = value from 'a.b'. Raises KeyError on unknown top-level or leaf keys."""
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f"Unknown config section '{p}' in '{dotted}'")
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"Unknown config key '{dotted}'")
    node[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config (resolving ``extends``), merged onto DEFAULTS, then apply 'a.b=value' overrides."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    parent = raw.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
    else:
        base = copy.deepcopy(DEFAULTS)
    cfg = deep_update(base, raw)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        key, val = item.split("=", 1)
        set_dotted(cfg, key, yaml.safe_load(val))
    return cfg


def apply_variant(cfg: dict, variant: dict) -> dict:
    """Return a copy of cfg with a sweep variant's ``set`` overrides applied and name recorded."""
    out = copy.deepcopy(cfg)
    for k, v in (variant.get("set") or {}).items():
        set_dotted(out, k, v)
    out["experiment"]["name"] = variant["name"]
    return out


def expand_sweep(cfg: dict) -> list[dict]:
    """Expand a sweep config into one config per (variant, seed). No variants -> single named run per seed."""
    variants = cfg.get("sweep", {}).get("variants") or [{"name": cfg["experiment"]["name"], "set": {}}]
    runs = []
    for v in variants:
        for seed in cfg.get("seeds", [cfg["experiment"]["seed"]]):
            c = apply_variant(cfg, v)
            c["experiment"]["seed"] = int(seed)
            runs.append(c)
    return runs
