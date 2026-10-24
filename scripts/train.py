#!/usr/bin/env python
"""Train and evaluate ONE configuration.

Examples
  python scripts/train.py --config configs/baseline.yaml --variant cnn_rgb --seed 0
  python scripts/train.py --config configs/baseline.yaml --variant cnn_rgb --set train.epochs=5 data.subset=2000
"""
import argparse

from cassava.config import apply_variant, load_config
from cassava.experiments import run_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", help="name of a sweep variant inside the config (default: first variant, or the base config)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides, e.g. train.epochs=5")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    variants = cfg["sweep"]["variants"]
    if variants:
        match = [v for v in variants if v["name"] == a.variant] if a.variant else variants[:1]
        if not match:
            raise SystemExit(f"Variant '{a.variant}' not found. Available: {[v['name'] for v in variants]}")
        cfg = apply_variant(cfg, match[0])
    if a.seed is not None:
        cfg["experiment"]["seed"] = a.seed
    run_experiment(cfg)


if __name__ == "__main__":
    main()
