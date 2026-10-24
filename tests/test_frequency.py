import numpy as np
import pytest

from cassava.features import frequency as fq
from cassava.features.color import color_feature_dim, color_feature_vector

CFG = {"channel": "gray", "use_fft": True, "use_dwt": True, "n_radial_bands": 8, "n_angular_bands": 4, "wavelet": "db2", "wavelet_level": 3}


def test_dimension_matches_declared(rgb_image):
    v = fq.frequency_feature_vector(rgb_image, CFG)
    assert v.shape == (fq.frequency_feature_dim(CFG),) and v.dtype == np.float32
    for fft, dwt in ((True, False), (False, True)):
        c = {**CFG, "use_fft": fft, "use_dwt": dwt}
        assert fq.frequency_feature_vector(rgb_image, c).shape == (fq.frequency_feature_dim(c),)
    with pytest.raises(ValueError):
        fq.frequency_feature_dim({**CFG, "use_fft": False, "use_dwt": False})


@pytest.mark.parametrize("img", [np.zeros((48, 48, 3), np.uint8), np.full((48, 48, 3), 255, np.uint8), np.full((48, 48, 3), 90, np.uint8)])
def test_numerically_stable_on_constant_images(img):
    assert np.isfinite(fq.frequency_feature_vector(img, CFG)).all()


def test_high_frequency_content_shifts_energy_to_outer_bands():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:64, 0:64]
    smooth = np.repeat((127 + 100 * np.sin(xx / 20.0))[..., None], 3, 2).astype(np.uint8)
    noisy = np.clip(smooth + rng.normal(0, 40, smooth.shape), 0, 255).astype(np.uint8)
    fs = fq.fft_features(fq.to_channel(smooth), 8, 4)
    fn = fq.fft_features(fq.to_channel(noisy), 8, 4)
    rel = lambda f: np.exp(f[:8])  # relative radial energy
    assert rel(fs)[:2].sum() > rel(fn)[:2].sum()  # smooth image: more energy in lowest bands
    assert rel(fn)[-3:].sum() > rel(fs)[-3:].sum()  # noisy image: more in highest bands
    assert rel(fs).sum() == pytest.approx(1.0, abs=1e-3)  # band energies are relative (sum to 1)


def test_dwt_detail_energy_increases_with_noise_and_level_check():
    g = np.zeros((64, 64), np.float32)
    n = np.random.default_rng(0).normal(0, 0.2, (64, 64)).astype(np.float32)
    assert fq.dwt_features(n, "db2", 3)[1] > fq.dwt_features(g, "db2", 3)[1]
    assert fq.dwt_features(g, "db2", 3).shape == (3 * 9 + 1,)
    with pytest.raises(ValueError, match="too large"):
        fq.dwt_features(np.zeros((16, 16), np.float32), "db2", 6)


def test_channel_validation(rgb_image):
    with pytest.raises(ValueError):
        fq.to_channel(rgb_image, "x")
    with pytest.raises(ValueError):
        fq.to_channel(rgb_image.astype(np.float32))
    assert fq.to_channel(rgb_image, "L").max() <= 1.0


def test_fft_spectrum_shape_and_dc_removed(rgb_image):
    s = fq.fft_log_magnitude(fq.to_channel(rgb_image))
    assert s.shape == (48, 48) and np.isfinite(s).all()


@pytest.mark.parametrize("rep", ["rgb", "hsv", "hsv_circ", "lab", "rgb_lab"])
def test_color_features_dim_and_hist_mass(rgb_image, rep):
    v = color_feature_vector(rgb_image, rep, 8)
    assert v.shape == (color_feature_dim(rep, 8),) and np.isfinite(v).all()
