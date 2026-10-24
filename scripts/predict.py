#!/usr/bin/env python
"""Predict disease class for one or more images.  python scripts/predict.py --checkpoint outputs/<run>/seed0/best.pt img1.jpg img2.jpg"""
import argparse
import json

from cassava.inference.predict import Predictor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save", help="optional path to write predictions as JSON")
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    results = Predictor(a.checkpoint, a.device).predict(a.images)
    for path, r in zip(a.images, results):
        print(f"{path}: {r['label']} ({r['confidence']:.1%})")
    if a.save:
        json.dump([{"image": p, **r} for p, r in zip(a.images, results)], open(a.save, "w"), indent=2)


if __name__ == "__main__":
    main()
