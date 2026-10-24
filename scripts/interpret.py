#!/usr/bin/env python
"""Grad-CAM visualisations + shortcut diagnostics for a checkpoint.

  python scripts/interpret.py --checkpoint outputs/cnn_rgb/seed0/best.pt --split val --per-class 4
  python scripts/interpret.py --checkpoint A/best.pt --compare-checkpoint B/best.pt   # explanation consistency

Writes overlays to <run>/gradcam/<class>/{correct,wrong}_*.png and <run>/interpretation.json.
Grad-CAM is an interpretability tool, not an accuracy method; heatmaps do not prove correct reasoning.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from cassava.data.cache import load_prepared
from cassava.data.dataset import CassavaDataset
from cassava.interpretability.gradcam import GradCAM, cam_similarity, overlay
from cassava.interpretability.shortcut import conditioned_evaluation, leaf_mass_fraction
from cassava.training.checkpoint import load_checkpoint
from cassava.training.trainer import make_eval_loader, predict_loader


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--compare-checkpoint")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--per-class", type=int, default=4, help="max correct and max wrong examples saved per class")
    ap.add_argument("--cache-dir")
    ap.add_argument("--skip-shortcut", action="store_true")
    a = ap.parse_args()

    model, pre, cfg, names, _ = load_checkpoint(a.checkpoint, "cpu")
    d = cfg["data"]
    images, df, _ = load_prepared(a.cache_dir or d["cache_dir"], d["image_size"], d["splits_file"])
    idx = np.where(df["split"].values == a.split)[0]
    labels = df["label"].values
    ds = CassavaDataset(images, idx, labels[idx], pre, None, d["localization"])
    probs, y = predict_loader(model, make_eval_loader(ds, cfg), torch.device("cpu"))
    pred = probs.argmax(1)
    out_dir = Path(a.checkpoint).parent / "gradcam"
    cam_engine = GradCAM(model)
    other = None
    if a.compare_checkpoint:
        m2, pre2, cfg2, _, _ = load_checkpoint(a.compare_checkpoint, "cpu")
        ds2 = CassavaDataset(images, idx, labels[idx], pre2, None, cfg2["data"]["localization"])
        other = (GradCAM(m2), ds2)

    summary: dict = {"split": a.split, "per_class": {}}
    rng = np.random.default_rng(0)
    for c, cname in enumerate(names):
        stats = {"leaf_mass": [], "mask_area": [], "similarity": []}
        for kind, sel in (("correct", (y == c) & (pred == c)), ("wrong", (y == c) & (pred != c))):
            members = np.where(sel)[0]
            chosen = rng.permutation(members)[: a.per_class]
            for j in chosen:
                x, f, _ = ds[int(j)]
                cam, cls, p = cam_engine(x[None], f[None], None)
                raw = ds.raw_image(int(j))
                lm, area = leaf_mass_fraction(cam, raw)
                stats["leaf_mass"].append(lm)
                stats["mask_area"].append(area)
                if other is not None:
                    x2, f2, _ = other[1][int(j)]
                    stats["similarity"].append(cam_similarity(cam, other[0](x2[None], f2[None], cls)[0]))
                panel = np.concatenate([raw, overlay(raw, cam)], axis=1)
                panel = cv2.resize(panel, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                cv2.putText(panel, f"T:{cname[:14]} P:{names[cls][:14]} {p[cls]:.0%}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                folder = out_dir / cname.replace(" ", "_").replace("(", "").replace(")", "")
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(folder / f"{kind}_{df['image_id'].values[idx[j]].replace('/', '_')}.png"), panel[..., ::-1])
        if stats["leaf_mass"]:
            summary["per_class"][cname] = {"mean_cam_mass_in_leaf_mask": float(np.mean(stats["leaf_mass"])), "mean_leaf_mask_area": float(np.mean(stats["mask_area"])),
                                           "n_visualised": len(stats["leaf_mass"]), **({"mean_cam_similarity_to_other_model": float(np.mean(stats["similarity"]))} if stats["similarity"] else {})}
    if not a.skip_shortcut:
        def predict_fn(sub, fn):
            s = CassavaDataset(images, sub, labels[sub], pre, None, d["localization"], None, fn)
            return predict_loader(model, make_eval_loader(s, cfg), torch.device("cpu"))[0]
        summary["conditioned_evaluation"] = conditioned_evaluation(predict_fn, np.random.default_rng(0).permutation(idx)[: cfg["eval"]["robustness_subset"]], labels, names)
    (Path(a.checkpoint).parent / "interpretation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Overlays saved under {out_dir}")


if __name__ == "__main__":
    main()
