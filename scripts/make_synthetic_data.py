#!/usr/bin/env python
"""Create a SYNTHETIC cassava-like dataset for smoke tests (not real data; results are meaningless scientifically)."""
import argparse

from cassava.data.synthetic import make_synthetic_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print("Wrote synthetic dataset to", make_synthetic_dataset(a.out, a.n, a.seed))
