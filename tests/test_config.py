import pytest

from cassava.config import DEFAULTS, expand_sweep, load_config, set_dotted


def test_load_defaults_and_extends():
    cfg = load_config("configs/color_space.yaml")
    assert cfg["data"]["image_size"] == 128 and cfg["eval"]["robustness"] is True
    assert cfg["train"]["epochs"] == 30  # inherited from baseline.yaml
    assert cfg["model"]["kind"] == DEFAULTS["model"]["kind"]


def test_override_and_unknown_key():
    cfg = load_config("configs/baseline.yaml", ["train.epochs=3", "data.augment.scale=[0.9,1.0]"])
    assert cfg["train"]["epochs"] == 3 and cfg["data"]["augment"]["scale"] == [0.9, 1.0]
    with pytest.raises(KeyError):
        set_dotted(cfg, "train.nonexistent", 1)
    with pytest.raises(ValueError):
        load_config("configs/baseline.yaml", ["no_equals_sign"])
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")


def test_sweep_expansion_counts():
    runs = expand_sweep(load_config("configs/color_space.yaml"))
    assert len(runs) == 7 * 3
    assert {r["experiment"]["name"] for r in runs} >= {"cnn_rgb", "cnn_lab", "cnn_hsv_circ"}
    assert runs[0]["experiment"]["seed"] == 0 and runs[1]["experiment"]["seed"] == 1


@pytest.mark.parametrize("name", ["baseline", "color_space", "frequency", "hybrid", "smoke"])
def test_all_shipped_configs_load_and_have_valid_variant_keys(name):
    for run in expand_sweep(load_config(f"configs/{name}.yaml")):
        assert run["model"]["kind"] in {"cnn", "transfer", "fusion", "classical"}
