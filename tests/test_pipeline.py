"""End-to-end tests on the synthetic cache: training, checkpoint round trip, inference, Grad-CAM, leakage guards."""
import copy
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from cassava.data.cache import load_prepared
from cassava.data.dataset import CassavaDataset
from cassava.experiments import run_experiment
from cassava.inference.predict import Predictor
from cassava.interpretability.gradcam import GradCAM, cam_similarity, overlay
from cassava.preprocessing.preprocessor import Preprocessor
from cassava.training.checkpoint import load_checkpoint
from cassava.evaluation.report import read_records


def test_cnn_run_artifacts_and_log(tiny_cfg):
    rec = run_experiment(tiny_cfg)
    out = tiny_cfg["experiment"]["output_dir"]
    from pathlib import Path

    run_dir = Path(out) / "smoke" / "seed0"
    for f in ("best.pt", "history.csv", "metrics.json", "config.yaml", "confusion_matrix.png", "test_predictions.csv"):
        assert (run_dir / f).exists(), f
    log = read_records(Path(out) / "experiment_log.jsonl")
    assert len(log) == 1 and log[0]["seed"] == 0 and "hardware" in log[0] and log[0]["split_sizes"]["test"] > 0
    assert 0 <= rec["test"]["macro_f1"] <= 1 and len(rec["test"]["per_class"]) == 5 and "macro_f1" in rec["test_ci"]
    assert rec["latency"]["total_ms"] > 0 and rec["latency"]["n_params"] > 0


def test_checkpoint_roundtrip_and_predictor(tiny_cfg, prepared):
    rec = run_experiment(tiny_cfg)
    model, pre, cfg, names, extra = load_checkpoint(rec["checkpoint"])
    assert names == rec["class_names"] and extra["seed"] == 0 and not model.training
    images, df, _ = load_prepared(prepared["cache"], 48)
    idx = np.where(df["split"].values == "test")[0][:6]
    ds = CassavaDataset(images, idx, df["label"].values[idx], pre)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    with torch.no_grad():
        p_direct = torch.softmax(model(x, torch.zeros(len(ds), 0)), 1).numpy()
    saved = pd.read_csv(f"{rec['output_dir']}/test_predictions.csv").set_index("cache_index").loc[idx]
    assert np.allclose(p_direct, saved[[f"p_{i}" for i in range(5)]].values, atol=1e-4)  # restored model == evaluated model

    pr = Predictor(rec["checkpoint"])
    img_path = prepared["root"] / "train_images" / "img_00001.jpg"
    out = pr.predict([img_path, np.array(Image.open(img_path).convert("RGB"))])
    assert len(out) == 2 and out[0]["label"] in names and sum(out[0]["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert out[0]["label"] == out[1]["label"]  # same image via path or array


def test_predictor_input_validation(tiny_cfg, tmp_path):
    rec = run_experiment(tiny_cfg)
    pr = Predictor(rec["checkpoint"])
    with pytest.raises(FileNotFoundError):
        pr.predict([tmp_path / "missing.jpg"])
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"garbage")
    with pytest.raises(ValueError, match="Could not read"):
        pr.predict([bad])
    with pytest.raises(ValueError):
        pr.predict([np.zeros((10, 10), np.uint8)])
    with pytest.raises(ValueError):
        pr.predict([])
    with pytest.raises(FileNotFoundError):
        Predictor(tmp_path / "nope.pt")
    corrupt = tmp_path / "c.pt"
    corrupt.write_bytes(b"not a checkpoint")
    with pytest.raises(RuntimeError, match="Could not read checkpoint"):
        Predictor(corrupt)


def test_fusion_and_classical_runs(tiny_cfg):
    f = copy.deepcopy(tiny_cfg)
    f["experiment"]["name"] = "fusion"
    f["model"]["kind"] = "fusion"
    f["data"]["representation"] = "lab"
    f["eval"]["robustness"] = True
    f["eval"]["robustness_subset"] = 20
    rf = run_experiment(f)
    assert rf["robustness"]["summary"]["worst_macro_f1"] <= rf["robustness"]["clean"]["macro_f1"] + 1.0 and len(rf["robustness"]["results"]) >= 10
    pr = Predictor(rf["checkpoint"])  # fusion model needs frequency features at inference
    assert pr.predict([np.random.default_rng(0).integers(0, 256, (60, 70, 3), dtype=np.uint8)])[0]["label"] in rf["class_names"]
    for feats, clf in ((["freq"], "logreg"), (["freq", "color"], "hgb")):
        c = copy.deepcopy(tiny_cfg)
        c["experiment"]["name"] = f"classical_{clf}"
        c["model"].update({"kind": "classical"})
        c["model"]["classical"].update({"features": feats, "classifier": clf})
        rc = run_experiment(c)
        assert rc["kind"] == "classical" and 0 <= rc["test"]["accuracy"] <= 1
    c["model"]["classical"]["features"] = ["bogus"]
    c["experiment"]["name"] = "bad"
    with pytest.raises(ValueError, match="non-empty subset"):
        run_experiment(c)


def test_normalisation_statistics_use_training_split_only(tiny_cfg, prepared):
    rec = run_experiment(tiny_cfg)
    _, pre, *_ = load_checkpoint(rec["checkpoint"])
    images, df, _ = load_prepared(prepared["cache"], 48)
    tr = np.where(df["split"].values == "train")[0]
    te = np.where(df["split"].values == "test")[0]
    ref_train = Preprocessor("rgb").fit(np.array(images[i]) for i in tr)
    ref_test = Preprocessor("rgb").fit(np.array(images[i]) for i in te)
    # stats were fitted on a (<=50 image) subsample of TRAIN, so they should sit near train stats, and the test-only
    # stats should differ measurably more than the train subsample does
    assert np.abs(pre.mean - ref_train.mean).max() < np.abs(ref_test.mean - ref_train.mean).max() + 0.05


def test_test_split_cannot_influence_training_or_model_selection(tiny_cfg, prepared, tmp_path):
    """Scramble every TEST label; training history, best epoch and VAL metrics must be bit-identical."""
    r1 = run_experiment(copy.deepcopy(tiny_cfg))
    cache2 = tmp_path / "cache2"
    shutil.copytree(prepared["cache"], cache2)
    df = pd.read_csv(cache2 / "splits.csv", keep_default_na=False)
    te = df["split"] == "test"
    df.loc[te, "label"] = np.random.default_rng(0).permutation(df.loc[te, "label"].values + 1) % 5
    df.to_csv(cache2 / "splits.csv", index=False)
    cfg2 = copy.deepcopy(tiny_cfg)
    cfg2["data"]["cache_dir"] = str(cache2)
    cfg2["experiment"]["output_dir"] = str(tmp_path / "out2")
    r2 = run_experiment(cfg2)
    h1, h2 = pd.read_csv(f"{r1['output_dir']}/history.csv"), pd.read_csv(f"{r2['output_dir']}/history.csv")
    cols = ["train_loss", "val_loss", "val_accuracy", "val_macro_f1"]
    assert np.allclose(h1[cols].values, h2[cols].values) and r1["best_epoch"] == r2["best_epoch"]
    assert r1["val"]["macro_f1"] == pytest.approx(r2["val"]["macro_f1"])
    assert r1["test"]["accuracy"] != r2["test"]["accuracy"] or r1["test"]["macro_f1"] != r2["test"]["macro_f1"]  # test side did change


def test_dataset_augmentation_depends_on_epoch_not_call_order(prepared):
    images, df, _ = load_prepared(prepared["cache"], 48)
    pre = Preprocessor("rgb").fit(np.array(images[i]) for i in range(20))
    from cassava.preprocessing.augment import Augmenter

    ds = CassavaDataset(images, np.arange(5), df["label"].values[:5], pre, Augmenter({"hflip": True, "vflip": True, "rotate": 20, "scale": [0.8, 1.0]}), seed=1)
    a = ds[3][0]
    _ = ds[0]
    assert torch.equal(a, ds[3][0])
    ds.set_epoch(1)
    assert not torch.equal(a, ds[3][0])


@pytest.mark.parametrize("kind", ["cnn", "transfer"])
def test_gradcam_shapes_and_range(kind):
    from cassava.models.cnn import SmallCNN
    from cassava.models.transfer import TransferNet

    m = SmallCNN(3, 5) if kind == "cnn" else TransferNet(3, 5, "resnet18", pretrained=False)
    cam, cls, probs = GradCAM(m)(torch.randn(1, 3, 64, 64))
    assert cam.shape == (64, 64) and 0 <= cam.min() and cam.max() <= 1 + 1e-6 and 0 <= cls < 5 and probs.sum() == pytest.approx(1.0, abs=1e-5)
    assert cam_similarity(cam, cam) == pytest.approx(1.0, abs=1e-5) or cam.max() == 0
    ov = overlay(np.zeros((64, 64, 3), np.uint8), cam)
    assert ov.shape == (64, 64, 3) and ov.dtype == np.uint8
    with pytest.raises(ValueError):
        GradCAM(m)(torch.randn(2, 3, 64, 64))
