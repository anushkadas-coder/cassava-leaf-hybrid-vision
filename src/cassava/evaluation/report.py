"""Experiment log (JSONL), confusion-matrix figures and cross-run summaries."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def append_record(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=_default) + "\n")


def read_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], png_path: str | Path, csv_path: str | Path | None = None) -> None:
    cm = np.asarray(cm)
    if csv_path is not None:
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(csv_path)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.4 * len(class_names) + 2, 1.2 * len(class_names) + 1.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), [c[:18] for c in class_names], rotation=40, ha="right")
    ax.set_yticks(range(len(class_names)), [c[:18] for c in class_names])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def summarize_records(records: list[dict], baseline: str | None = None) -> pd.DataFrame:
    """One row per experiment name: mean +- std over seeds. ``delta_vs_baseline`` compares test macro-F1 means and
    ``beyond_seed_std`` flags deltas larger than the larger of the two seed standard deviations (a weak, necessary
    - not sufficient - sanity check; use paired tests / more seeds before claiming an effect)."""
    rows = []
    for r in records:
        rob = r.get("robustness") or {}
        rows.append({
            "name": r["name"], "seed": r["seed"], "kind": r["kind"], "representation": r["representation"],
            "test_macro_f1": r["test"]["macro_f1"], "test_accuracy": r["test"]["accuracy"],
            "val_macro_f1": r["val"]["macro_f1"], "params": r["latency"]["n_params"] if r.get("latency") else np.nan,
            "total_ms": r["latency"]["total_ms"] if r.get("latency") else np.nan,
            "train_time_s": r.get("train_time_s", np.nan),
            "rob_mean_macro_f1": rob.get("summary", {}).get("mean_macro_f1", np.nan),
            "rob_worst_macro_f1": rob.get("summary", {}).get("worst_macro_f1", np.nan),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    g = df.groupby("name", sort=False)
    out = g.agg(kind=("kind", "first"), representation=("representation", "first"), n_seeds=("seed", "nunique"),
                test_macro_f1=("test_macro_f1", "mean"), test_macro_f1_std=("test_macro_f1", "std"),
                test_accuracy=("test_accuracy", "mean"), test_accuracy_std=("test_accuracy", "std"),
                val_macro_f1=("val_macro_f1", "mean"), params=("params", "first"), total_ms=("total_ms", "mean"),
                train_time_s=("train_time_s", "mean"), rob_mean_macro_f1=("rob_mean_macro_f1", "mean"),
                rob_worst_macro_f1=("rob_worst_macro_f1", "mean")).reset_index()
    out[["test_macro_f1_std", "test_accuracy_std"]] = out[["test_macro_f1_std", "test_accuracy_std"]].fillna(0.0)
    if baseline and baseline in set(out["name"]):
        b = out[out["name"] == baseline].iloc[0]
        out["delta_vs_baseline"] = out["test_macro_f1"] - b["test_macro_f1"]
        out["beyond_seed_std"] = (out["delta_vs_baseline"].abs() > np.maximum(out["test_macro_f1_std"], b["test_macro_f1_std"])).astype(object)
        out.loc[out["name"] == baseline, "beyond_seed_std"] = np.nan
        out.loc[out["n_seeds"] < 2, "beyond_seed_std"] = np.nan  # seed std is undefined with a single seed
    return out


def to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "(no results)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(format(v, floatfmt) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("" if pd.isna(v) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
