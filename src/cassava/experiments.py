"""End-to-end experiment runner: one (config, seed) -> trained model, metrics, artefacts, log record."""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from cassava.data.cache import load_prepared
from cassava.data.dataset import CassavaDataset
from cassava.evaluation.latency import measure_latency
from cassava.evaluation.metrics import bootstrap_ci, classification_report
from cassava.evaluation.report import append_record, save_confusion_matrix
from cassava.evaluation.robustness import default_suite
from cassava.models.factory import build_model, count_parameters, model_size_mb
from cassava.preprocessing.augment import Augmenter
from cassava.preprocessing.preprocessor import Preprocessor
from cassava.training.checkpoint import save_checkpoint
from cassava.training.classical import build_classifier, feature_function
from cassava.training.trainer import fit, make_eval_loader, predict_loader
from cassava.utils import get_device, get_logger, hardware_info, seed_everything

log = get_logger()


def _stratified_cap(labels: np.ndarray, idx: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Keep at most ``cap`` samples from idx, preserving class proportions (at least 1 per present class)."""
    if cap is None or cap >= len(idx):
        return idx
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels[idx]):
        members = idx[labels[idx] == c]
        n = max(1, int(round(cap * len(members) / len(idx))))
        keep.append(rng.choice(members, size=min(n, len(members)), replace=False))
    return np.sort(np.concatenate(keep))


def _prediction_table(df: pd.DataFrame, idx: np.ndarray, probs: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({"image_id": df["image_id"].values[idx], "cache_index": idx, "true": df["label"].values[idx], "pred": probs.argmax(1), "confidence": probs.max(1)})
    for i, c in enumerate(class_names):
        out[f"p_{i}"] = probs[:, i]
    return out


def run_experiment(cfg: dict) -> dict:
    """Run one experiment. The test split is evaluated exactly once, after the best validation checkpoint is restored."""
    name, seed = cfg["experiment"]["name"], int(cfg["experiment"]["seed"])
    run_dir = Path(cfg["experiment"]["output_dir"]) / name / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    if cfg["train"]["num_threads"]:
        torch.set_num_threads(int(cfg["train"]["num_threads"]))
    device = get_device(cfg["train"]["device"])
    kind = cfg["model"]["kind"]
    d = cfg["data"]
    images, df, class_names = load_prepared(d["cache_dir"], d["image_size"], d["splits_file"])
    k, labels = len(class_names), df["label"].values
    idx = {s: np.where(df["split"].values == s)[0] for s in ("train", "val", "test")}
    for s, v in idx.items():
        if len(v) == 0:
            raise ValueError(f"Split '{s}' is empty; check the splits file.")
    idx["train"] = _stratified_cap(labels, idx["train"], d["subset"], seed)
    log.info("[%s | seed %d] kind=%s repr=%s loc=%s train/val/test=%d/%d/%d device=%s", name, seed, kind, d["representation"],
             d["localization"], len(idx["train"]), len(idx["val"]), len(idx["test"]), device)

    localizer = None
    if d["localization"] == "cascade":
        from cassava.detection.viola_jones import CascadeLeafDetector

        det = CascadeLeafDetector(d["cascade_path"])
        localizer = lambda im: det.crop(im)[0]  # noqa: E731
    freq_cfg = cfg["frequency"] if kind == "fusion" else None
    pre = Preprocessor(d["representation"], d["normalize"], freq_cfg)
    mk = lambda split, aug=None, perturb=None, sub=None: CassavaDataset(  # noqa: E731
        images, idx[split] if sub is None else sub, labels[idx[split] if sub is None else sub], pre, aug, d["localization"], localizer, perturb, seed)
    train_ds = mk("train", Augmenter(d["augment"]) if kind != "classical" else None)
    rng = np.random.default_rng(seed)
    fit_ids = rng.choice(len(train_ds), size=min(d["stats_samples"], len(train_ds)), replace=False)
    if kind != "classical":
        pre.fit(train_ds.raw_image(int(i)) for i in fit_ids)  # statistics from TRAIN only
    else:
        pre.fitted = True

    record: dict = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "name": name, "seed": seed, "kind": kind,
        "representation": d["representation"], "localization": d["localization"], "config": cfg, "hardware": hardware_info(),
        "split_sizes": {s: int(len(v)) for s, v in idx.items()}, "class_names": class_names,
        "splits_file": d["splits_file"] or str(Path(d["cache_dir"]) / "splits.csv"),
    }

    if kind == "classical":
        res = _run_classical(cfg, mk, idx, labels, df, class_names, run_dir, seed, record)
    else:
        res = _run_torch(cfg, mk, train_ds, pre, idx, labels, df, class_names, run_dir, seed, device, record)
    record.update(res)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({k_: record[k_] for k_ in ("val", "test", "test_ci", "latency", "robustness", "train_time_s") if k_ in record}, f, indent=2, default=float)
    append_record(Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["log_file"], record)
    log.info("[%s | seed %d] VAL macro-F1=%.4f acc=%.4f | TEST macro-F1=%.4f acc=%.4f", name, seed, record["val"]["macro_f1"],
             record["val"]["accuracy"], record["test"]["macro_f1"], record["test"]["accuracy"])
    return record


def _robustness(cfg, predict_fn, idx_test, labels, class_names, mk) -> dict:
    """Evaluate under synthetic perturbations on a fixed test subset (identical across runs for comparability)."""
    sub = np.random.default_rng(0).permutation(idx_test)[: cfg["eval"]["robustness_subset"]]
    k = len(class_names)
    out: dict = {"subset_size": int(len(sub)), "results": {}}
    clean = classification_report(labels[sub], predict_fn(sub, None).argmax(1), class_names)
    out["clean"] = {"macro_f1": clean["macro_f1"], "accuracy": clean["accuracy"]}
    for fam, lvl, fn in default_suite():
        rep = classification_report(labels[sub], predict_fn(sub, fn).argmax(1), class_names)
        out["results"][f"{fam}/{lvl}"] = {"macro_f1": rep["macro_f1"], "accuracy": rep["accuracy"]}
    f1s = [v["macro_f1"] for v in out["results"].values()]
    out["summary"] = {"mean_macro_f1": float(np.mean(f1s)), "worst_macro_f1": float(np.min(f1s)),
                      "mean_drop_vs_clean": float(out["clean"]["macro_f1"] - np.mean(f1s))}
    return out


def _run_torch(cfg, mk, train_ds, pre, idx, labels, df, class_names, run_dir, seed, device, record) -> dict:
    k = len(class_names)
    model = build_model(cfg, k)
    n_params, size_mb = count_parameters(model), model_size_mb(model)
    val_ds = mk("val")
    out = fit(model, train_ds, val_ds, labels[idx["train"]], class_names, cfg, device, run_dir, seed)
    model.load_state_dict(out["best_state"])  # restore best-on-validation weights
    save_checkpoint(run_dir / "best.pt", model, cfg, pre, class_names, {"best_epoch": out["best_epoch"], "seed": seed})

    pv, yv = predict_loader(model, make_eval_loader(val_ds, cfg), device)
    val = classification_report(yv, pv.argmax(1), class_names)
    test_ds = mk("test")  # test split used only now, after model selection
    pt, yt = predict_loader(model, make_eval_loader(test_ds, cfg), device)
    test = classification_report(yt, pt.argmax(1), class_names)
    ci = bootstrap_ci(yt, pt.argmax(1), k, cfg["eval"]["bootstrap"], seed=seed)
    _prediction_table(df, idx["test"], pt, class_names).to_csv(run_dir / "test_predictions.csv", index=False)
    save_confusion_matrix(np.array(test["confusion_matrix"]), class_names, run_dir / "confusion_matrix.png", run_dir / "confusion_matrix.csv")

    latency = measure_latency(model, pre, test_ds.raw_image(0), cfg["eval"]["latency_runs"])
    robustness = None
    if cfg["eval"]["robustness"]:
        def predict_fn(sub, fn):
            ds = mk("test", None, fn, sub)
            return predict_loader(model, make_eval_loader(ds, cfg), device)[0]
        robustness = _robustness(cfg, predict_fn, idx["test"], labels, class_names, mk)
    return {"val": val, "test": test, "test_ci": ci, "latency": latency, "robustness": robustness, "best_epoch": int(out["best_epoch"]),
            "epochs_run": int(len(out["history"])), "train_time_s": out["train_time_s"], "n_params": n_params, "size_mb": size_mb,
            "checkpoint": str(run_dir / "best.pt"), "output_dir": str(run_dir)}


def _run_classical(cfg, mk, idx, labels, df, class_names, run_dir, seed, record) -> dict:
    k = len(class_names)
    fe = feature_function(cfg)

    def extract(ds):
        return np.stack([fe(ds.raw_image(i)) for i in range(len(ds))])

    t0 = time.perf_counter()
    train_ds, val_ds, test_ds = mk("train"), mk("val"), mk("test")
    xtr = extract(train_ds)
    clf = build_classifier(cfg["model"]["classical"]["classifier"], seed)
    clf.fit(xtr, labels[idx["train"]])
    train_time = time.perf_counter() - t0
    val_probs, test_probs = clf.predict_proba(extract(val_ds)), clf.predict_proba(extract(test_ds))
    val = classification_report(labels[idx["val"]], val_probs.argmax(1), class_names)
    test = classification_report(labels[idx["test"]], test_probs.argmax(1), class_names)
    ci = bootstrap_ci(labels[idx["test"]], test_probs.argmax(1), k, cfg["eval"]["bootstrap"], seed=seed)
    _prediction_table(df, idx["test"], test_probs, class_names).to_csv(run_dir / "test_predictions.csv", index=False)
    save_confusion_matrix(np.array(test["confusion_matrix"]), class_names, run_dir / "confusion_matrix.png", run_dir / "confusion_matrix.csv")
    joblib.dump({"pipeline": clf, "config": cfg, "class_names": class_names}, run_dir / "model.joblib")

    sample = test_ds.raw_image(0)
    ts = []
    for _ in range(cfg["eval"]["latency_runs"]):
        t = time.perf_counter()
        clf.predict_proba(fe(sample)[None])
        ts.append(time.perf_counter() - t)
    n_params = int(getattr(clf[-1], "coef_", np.zeros(0)).size)
    latency = {"preprocess_ms": 0.0, "model_ms": 0.0, "total_ms": float(np.median(ts) * 1e3), "n_params": n_params, "size_mb": float(Path(run_dir / "model.joblib").stat().st_size / 1e6), "device": "cpu"}
    robustness = None
    if cfg["eval"]["robustness"]:
        def predict_fn(sub, fn):
            ds = mk("test", None, fn, sub)
            return clf.predict_proba(extract(ds))
        robustness = _robustness(cfg, predict_fn, idx["test"], labels, class_names, mk)
    return {"val": val, "test": test, "test_ci": ci, "latency": latency, "robustness": robustness, "best_epoch": None, "epochs_run": None,
            "train_time_s": train_time, "n_params": n_params, "size_mb": latency["size_mb"], "checkpoint": str(run_dir / "model.joblib"), "output_dir": str(run_dir)}
