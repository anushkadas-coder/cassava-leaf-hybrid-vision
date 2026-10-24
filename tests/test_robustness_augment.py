import numpy as np

from cassava.evaluation import robustness as rb
from cassava.preprocessing.augment import Augmenter
import pytest

AUG = {"hflip": True, "vflip": True, "rotate": 30, "scale": [0.8, 1.0], "photometric": True, "photometric_strength": 0.2}


def test_perturbations_preserve_shape_dtype_and_identity(rgb_image):
    for fam, lvl, fn in rb.default_suite():
        out = fn(rgb_image)
        assert out.shape == rgb_image.shape and out.dtype == np.uint8, (fam, lvl)
    assert np.array_equal(rb.brightness(1.0)(rgb_image), rgb_image)
    assert np.array_equal(rb.gamma(1.0)(rgb_image), rgb_image)
    assert rb.brightness(0.5)(rgb_image).mean() < rgb_image.mean() < rb.brightness(1.4)(rgb_image).mean() + 1


def test_noise_is_deterministic_per_image(rgb_image):
    assert np.array_equal(rb.gaussian_noise(10)(rgb_image), rb.gaussian_noise(10)(rgb_image))


def test_blur_reduces_high_frequency(rgb_image):
    assert rb.gaussian_blur(2.0)(rgb_image).astype(float).std() < rgb_image.astype(float).std()


def test_augmenter_deterministic_given_rng_and_shape(rgb_image):
    a = Augmenter(AUG)
    o1, o2 = a(rgb_image, np.random.default_rng(5)), a(rgb_image, np.random.default_rng(5))
    assert np.array_equal(o1, o2) and o1.shape == rgb_image.shape and o1.dtype == np.uint8
    assert not np.array_equal(a(rgb_image, np.random.default_rng(6)), o1)


def test_photometric_off_by_default_keeps_colours(rgb_image):
    a = Augmenter({**AUG, "photometric": False, "rotate": 0, "scale": [1.0, 1.0], "hflip": False, "vflip": False})
    assert np.array_equal(a(rgb_image, np.random.default_rng(0)), rgb_image)


def test_invalid_scale_rejected():
    with pytest.raises(ValueError):
        Augmenter({**AUG, "scale": [0.0, 1.0]})
