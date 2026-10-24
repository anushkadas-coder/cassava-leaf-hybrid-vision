import numpy as np
import pytest

from cassava.preprocessing import colorspace as cs
from cassava.preprocessing.preprocessor import Preprocessor


def solid(r, g, b, n=8):
    return np.tile(np.array([r, g, b], np.uint8), (n, n, 1))


@pytest.mark.parametrize("rep,c", [("rgb", 3), ("hsv", 3), ("hsv_circ", 4), ("lab", 3), ("rgb_lab", 6)])
def test_shapes_dtypes(rgb_image, rep, c):
    out = cs.convert(rgb_image, rep)
    assert out.shape == (48, 48, c) and out.dtype == np.float32 and np.isfinite(out).all()
    assert cs.representation_channels(rep) == c


def test_known_hsv_values():
    red, green = cs.rgb_to_hsv(solid(255, 0, 0)), cs.rgb_to_hsv(solid(0, 255, 0))
    assert red[0, 0] == pytest.approx([0.0, 1.0, 1.0], abs=1e-4)
    assert green[0, 0] == pytest.approx([1 / 3, 1.0, 1.0], abs=1e-4)


def test_hue_circular_encoding_is_continuous_at_wrap():
    near_wrap_a, near_wrap_b = solid(255, 2, 0), solid(255, 0, 2)  # hue ~0.5 deg vs ~359.5 deg
    h = cs.rgb_to_hsv(near_wrap_a)[0, 0, 0], cs.rgb_to_hsv(near_wrap_b)[0, 0, 0]
    assert abs(h[0] - h[1]) > 0.9  # plain hue jumps across the wrap
    ca, cb = cs.rgb_to_hsv_circ(near_wrap_a)[0, 0, :2], cs.rgb_to_hsv_circ(near_wrap_b)[0, 0, :2]
    assert np.linalg.norm(ca - cb) < 0.1  # (sin, cos) encoding stays close


def test_lab_white_black_and_ranges():
    white, black = cs.rgb_to_lab(solid(255, 255, 255))[0, 0], cs.rgb_to_lab(solid(0, 0, 0))[0, 0]
    assert white[0] == pytest.approx(1.0, abs=1e-2) and abs(white[1]) < 0.02 and abs(white[2]) < 0.02
    assert black[0] == pytest.approx(0.0, abs=1e-3)
    rnd = cs.rgb_to_lab(np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8))
    assert rnd[..., 0].min() >= 0 and rnd[..., 0].max() <= 1.001 and np.abs(rnd[..., 1:]).max() < 1.1


def test_input_validation():
    with pytest.raises(ValueError):
        cs.convert(np.zeros((8, 8), np.uint8), "hsv")
    with pytest.raises(ValueError):
        cs.convert(np.zeros((8, 8, 3), np.float32), "lab")
    with pytest.raises(ValueError, match="Unknown representation"):
        cs.convert(np.zeros((8, 8, 3), np.uint8), "xyz")


def test_imagenet_normalisation_rejected_for_non_rgb():
    with pytest.raises(ValueError, match="RGB-specific"):
        Preprocessor("lab", "imagenet")
    Preprocessor("rgb", "imagenet")


def test_preprocessor_requires_fit_and_standardises(rgb_image):
    pre = Preprocessor("lab")
    with pytest.raises(RuntimeError):
        pre(rgb_image)
    imgs = [np.random.default_rng(i).integers(0, 256, (32, 32, 3), dtype=np.uint8) for i in range(6)]
    pre.fit(imgs)
    x = np.concatenate([pre.image_array(i).reshape(-1, 3) for i in imgs])
    assert np.allclose(x.mean(0), 0, atol=1e-3) and np.allclose(x.std(0), 1, atol=1e-3)
    t, f = pre(imgs[0])
    assert t.shape == (3, 32, 32) and f.numel() == 0


def test_preprocessor_state_roundtrip(rgb_image):
    cfg = {"channel": "gray", "use_fft": True, "use_dwt": True, "n_radial_bands": 4, "n_angular_bands": 2, "wavelet": "db2", "wavelet_level": 2}
    a = Preprocessor("rgb_lab", "train_stats", cfg).fit([rgb_image, rgb_image[::-1].copy()])
    b = Preprocessor.from_state_dict(a.state_dict())
    xa, fa = a(rgb_image)
    xb, fb = b(rgb_image)
    assert np.allclose(xa, xb) and np.allclose(fa, fb)
