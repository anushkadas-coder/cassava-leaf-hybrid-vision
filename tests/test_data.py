import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cassava.data.audit import dhash64, duplicate_groups, inspect_images
from cassava.data.cache import load_prepared
from cassava.data.records import DatasetError, load_records
from cassava.data.splits import check_no_leakage, make_splits
from cassava.data.synthetic import CLASS_NAMES


def test_records_label_mapping(synth_root):
    df, names = load_records(synth_root)
    assert names == CLASS_NAMES and set(df["label"]) <= set(range(len(names)))
    assert (df["class_name"] == df["label"].map(lambda i: names[i])).all()


def test_missing_dataset_path_message(tmp_path):
    with pytest.raises(DatasetError, match="data/README.md"):
        load_records(tmp_path / "nope")


def test_malformed_labels(tmp_path):
    (tmp_path / "train_images").mkdir()
    pd.DataFrame({"image_id": ["a.jpg"], "label": ["x"]}).to_csv(tmp_path / "train.csv", index=False)
    with pytest.raises(DatasetError, match="non-integer"):
        load_records(tmp_path)
    pd.DataFrame({"image_id": ["a.jpg"], "label": [None]}).to_csv(tmp_path / "train.csv", index=False)
    with pytest.raises(DatasetError, match="missing labels"):
        load_records(tmp_path)
    pd.DataFrame({"idx": ["a.jpg"]}).to_csv(tmp_path / "train.csv", index=False)
    with pytest.raises(DatasetError, match="missing columns"):
        load_records(tmp_path)
    pd.DataFrame({"image_id": ["a.jpg"], "label": [7]}).to_csv(tmp_path / "train.csv", index=False)
    (tmp_path / "label_num_to_disease_map.json").write_text(json.dumps({"0": "a", "1": "b"}))
    with pytest.raises(DatasetError, match="not present in label map"):
        load_records(tmp_path)


def test_class_folder_layout(tmp_path):
    for c in ("healthy", "cmd"):
        (tmp_path / c).mkdir()
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / c / "a.png")
    df, names = load_records(tmp_path)
    assert names == ["cmd", "healthy"] and len(df) == 2


def test_audit_flags_invalid_and_missing_images(tmp_path):
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / "ok.png")
    (tmp_path / "bad.png").write_bytes(b"not an image")
    df = pd.DataFrame({"image_id": ["ok.png", "bad.png", "gone.png"], "path": [str(tmp_path / n) for n in ("ok.png", "bad.png", "gone.png")], "label": [0, 0, 0]})
    out = inspect_images(df)
    assert out["valid"].tolist() == [True, False, False]


def test_duplicate_detection_groups_resized_copy(tmp_path):
    rng = np.random.default_rng(0)
    base = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    a = Image.fromarray(base)
    b = a.resize((60, 60))
    c = Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8))
    groups = duplicate_groups(np.array([dhash64(a), dhash64(b), dhash64(c)], dtype=np.uint64), max_hamming=6)
    assert groups[0] == groups[1] != groups[2]


def test_prepare_report_and_no_leakage(prepared):
    r = prepared["report"]
    assert r["leakage_check"] == {"leaked_groups": 0, "leaked_md5": 0}
    assert r["n_duplicate_groups"] >= 1  # injected duplicates were found
    images, df, names = load_prepared(prepared["cache"], 48)
    assert images.shape[1:] == (48, 48, 3) and len(images) == len(df)
    check_no_leakage(df)
    assert set(df["split"]) == {"train", "val", "test"}


def test_splits_stratified_and_grouped():
    rng = np.random.default_rng(0)
    labels = rng.choice(3, 600, p=[0.1, 0.3, 0.6])
    groups = np.arange(600)
    groups[100:120] = 7  # a duplicate cluster must stay together
    split = make_splits(labels, groups, 0.15, 0.15, seed=1)
    assert len(set(split[100:120])) == 1
    for s in ("train", "val", "test"):
        frac = np.bincount(labels[split == s], minlength=3) / (split == s).sum()
        assert np.allclose(frac, [0.1, 0.3, 0.6], atol=0.08)
    assert (split == "test").mean() == pytest.approx(0.15, abs=0.04)


def test_split_is_reproducible_and_validates_fractions():
    labels, groups = np.random.default_rng(0).integers(0, 3, 200), np.arange(200)
    assert (make_splits(labels, groups, seed=5) == make_splits(labels, groups, seed=5)).all()
    with pytest.raises(ValueError):
        make_splits(labels, groups, val_frac=0.6, test_frac=0.3)


def test_leakage_detector_raises():
    df = pd.DataFrame({"group": [1, 1], "md5": ["a", "b"], "split": ["train", "test"]})
    with pytest.raises(AssertionError, match="Leakage"):
        check_no_leakage(df)


def test_load_prepared_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        load_prepared(tmp_path, 32)
