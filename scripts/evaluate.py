#!/usr/bin/env python
"""Evaluate a saved checkpoint, or compare two runs' test predictions.

  python scripts/evaluate.py --checkpoint outputs/cnn_rgb/seed0/best.pt --split val
  python scripts/evaluate.py --checkpoint outputs/cnn_rgb/seed0/best.pt --split test --robustness
  python scripts/evaluate.py --compare outputs/cnn_rgb/seed0 outputs/fusion/seed0 --out outputs/compare_cnn_vs_fusion.md

NOTE: evaluating on --split test is meant for FINAL reporting; do not use it to choose models or hyper-parameters.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cassava.data.cache import load_prepared
from cassava.data.dataset import CassavaDataset
from cassava.evaluation.error_analysis import confident_errors, paired_comparison, top_confusions
from cassava.evaluation.metrics import bootstrap_ci, classification_report
from cassava.evaluation.report import save_confusion_matrix, to_markdown
from cassava.evaluation.robustness import default_suite
from cassava.training.checkpoint import load_checkpoint
from cassava.training.trainer import make_eval_loader, predict_loader
from cassava.utils import get_logger

log = get_logger()


def evaluate_checkpoint(a) -> None:
    model, pre, cfg, class_names, _ = load_checkpoint(a.checkpoint, "cpu")
    d = cfg["data"]
    if d["localization"] == "cascade":
        raise SystemExit("Checkpoints using the cascade localiser are not supported by evaluate.py")
    images, df, _ = load_prepared(a.cache_dir or d["cache_dir"], d["image_size"], d["splits_file"])
    idx = np.where(df["split"].values == a.split)[0]
    if len(idx) == 0:
        raise SystemExit(f"Split '{a.split}' is empty")
    if a.split == "test":
        log.warning("Evaluating on the TEST split: use only for final reporting, never for model selection.")
    labels = df["label"].values

    def probs_for(sub, fn):
        ds = CassavaDataset(images, sub, labels[sub], pre, None, d["localization"], None, fn)
        return predict_loader(model, make_eval_loader(ds, cfg), torch.device("cpu"))[0]

    p = probs_for(idx, None)
    rep = classification_report(labels[idx], p.argmax(1), class_names)
    out = {"split": a.split, "report": rep, "ci": bootstrap_ci(labels[idx], p.argmax(1), len(class_names), cfg["eval"]["bootstrap"])}
    if a.robustness:
        sub = np.random.default_rng(0).permutation(idx)[: cfg["eval"]["robustness_subset"]]
        out["robustness"] = {f"{f}/{l}": classification_report(labels[sub], probs_for(sub, fn).argmax(1), class_names)["macro_f1"] for f, l, fn in default_suite()}
    out_dir = Path(a.checkpoint).parent
    pd.DataFrame({"image_id": df["image_id"].values[idx], "true": labels[idx], "pred": p.argmax(1), "confidence": p.max(1)}).to_csv(out_dir / f"{a.split}_predictions_reeval.csv", index=False)
    save_confusion_matrix(np.array(rep["confusion_matrix"]), class_names, out_dir / f"confusion_{a.split}_reeval.png")
    (out_dir / f"eval_{a.split}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("split",)} | {"accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"], "ci": out["ci"].get("macro_f1")}, indent=2))


def compare_runs(a) -> None:
    ra, rb = Path(a.compare[0]), Path(a.compare[1])
    class_names = json.loads((ra.parents[1] / "experiment_log.jsonl").read_text().splitlines()[0])["class_names"] if (ra.parents[1] / "experiment_log.jsonl").exists() else None
    pa, pb = pd.read_csv(ra / "test_predictions.csv"), pd.read_csv(rb / "test_predictions.csv")
    if class_names is None:
        class_names = [f"class_{i}" for i in range(int(max(pa["true"].max(), pa["pred"].max())) + 1)]
    cmp = paired_comparison(pa, pb, class_names)
    lines = [f"# Paired comparison\n\nA = `{ra}`\nB = `{rb}`\n",
             f"- accuracy A={cmp['accuracy_a']:.4f}, B={cmp['accuracy_b']:.4f}",
             f"- images fixed by B: {cmp['fixed_by_b']}, broken by B: {cmp['broken_by_b']}, exact McNemar p = {cmp['mcnemar_exact_p']:.4f} (n={cmp['n']})\n",
             "## Per-class recall\n", to_markdown(pd.DataFrame(cmp["per_class_recall"]).T.reset_index().rename(columns={"index": "class"})),
             "\n## Top confusions in A\n", to_markdown(top_confusions(pa, class_names)), "\n## Top confusions in B\n", to_markdown(top_confusions(pb, class_names)),
             "\n## Highest-confidence errors in B (inspect for label noise / shortcuts)\n", to_markdown(confident_errors(pb)[["image_id", "true", "pred", "confidence"]])]
    Path(a.out).write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--robustness", action="store_true")
    ap.add_argument("--cache-dir")
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A_DIR", "RUN_B_DIR"))
    ap.add_argument("--out", default="outputs/comparison.md")
    a = ap.parse_args()
    if a.compare:
        compare_runs(a)
    elif a.checkpoint:
        evaluate_checkpoint(a)
    else:
        ap.error("give --checkpoint or --compare")


if __name__ == "__main__":
    main()
