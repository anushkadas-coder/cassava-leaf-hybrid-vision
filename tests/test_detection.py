import numpy as np
import pytest

from cassava.detection.haar import (count_haar_features_two_rect, haar_three_rect, haar_two_rect_horizontal, haar_two_rect_vertical,
                                    integral_image, rect_sum)
from cassava.detection.leaf_mask import leaf_mask_hsv, localize, mask_to_bbox
from cassava.detection.viola_jones import CascadeLeafDetector, evaluate_detections, iou


def test_integral_image_matches_bruteforce():
    img = np.random.default_rng(0).integers(0, 255, (20, 30))
    ii = integral_image(img)
    assert ii.shape == (21, 31)
    for r0, c0, r1, c1 in [(0, 0, 20, 30), (3, 4, 9, 11), (5, 5, 6, 6)]:
        assert rect_sum(ii, r0, c0, r1, c1) == img[r0:r1, c0:c1].sum()


def test_haar_edge_responses():
    img = np.zeros((8, 8))
    img[:, :4] = 1.0  # bright left, dark right: vertical edge
    ii = integral_image(img)
    assert haar_two_rect_horizontal(ii, 0, 0, 8, 4) == 32.0
    assert haar_two_rect_vertical(ii, 0, 0, 4, 8) == 0.0  # no horizontal edge
    line = np.zeros((4, 9))
    line[:, 3:6] = 1.0  # bright centre strip
    assert haar_three_rect(integral_image(line), 0, 0, 4, 3) == -12.0
    with pytest.raises(ValueError):
        integral_image(np.zeros((3, 3, 3)))


def test_feature_count_grows_fast():
    assert count_haar_features_two_rect(24, 24) > 40_000


def test_iou_and_detection_metrics():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0 and iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    m = evaluate_detections([[(0, 0, 10, 10)], [], [(50, 50, 5, 5)]], [(0, 0, 10, 10)] * 3)
    assert m["recall"] == pytest.approx(1 / 3) and m["miss_rate"] == pytest.approx(2 / 3) and m["precision"] == pytest.approx(1 / 2)
    with pytest.raises(ValueError):
        evaluate_detections([[]], [(0, 0, 1, 1)] * 2)


def test_cascade_detector_requires_trained_cascade(tmp_path):
    with pytest.raises(FileNotFoundError, match="trained on annotated"):
        CascadeLeafDetector(None)
    bad = tmp_path / "c.xml"
    bad.write_text("<nope/>")
    with pytest.raises(ValueError):
        CascadeLeafDetector(bad)


def _leaf_scene():
    img = np.full((64, 64, 3), (110, 85, 60), np.uint8)  # soil
    img[16:48, 20:44] = (50, 140, 45)  # green leaf block
    return img


def test_leaf_mask_and_bbox():
    m = leaf_mask_hsv(_leaf_scene())
    assert 0.1 < m.mean() < 0.4
    x0, y0, x1, y1 = mask_to_bbox(m)
    assert 14 <= y0 <= 20 and 44 <= y1 <= 50 and 18 <= x0 <= 24 and 40 <= x1 <= 46
    assert mask_to_bbox(np.zeros((4, 4), bool)) is None


def test_localize_modes_and_fallback():
    img = _leaf_scene()
    assert localize(img, "none") is img
    blank = localize(img, "hsv_blank")
    assert blank.shape == img.shape and (blank[0, 0] == 127).all() and (blank[32, 32] == img[32, 32]).all()
    crop = localize(img, "hsv_crop")
    assert crop.shape == img.shape and crop.mean(0)[..., 1].mean() > img.mean(0)[..., 1].mean()  # greener after cropping
    soil = np.full((64, 64, 3), (110, 85, 60), np.uint8)
    assert np.array_equal(localize(soil, "hsv_crop"), soil)  # no leaf found -> fallback to full image
    with pytest.raises(ValueError):
        localize(img, "bogus")
