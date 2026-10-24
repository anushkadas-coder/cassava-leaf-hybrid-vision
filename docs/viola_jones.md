# Viola-Jones feasibility for cassava leaf localisation

## How the algorithm works (and where this repo implements it)
1. **Integral image** `II(y,x)=sum(img[:y,:x])`: any rectangle sum in 4 lookups (`detection/haar.py`, tested against brute force).
2. **Haar-like features**: signed sums of adjacent rectangles (edge, line, diagonal patterns). A 24x24 window has >100k of them (`count_haar_features_two_rect` shows the scale for just one type).
3. **AdaBoost** selects a few hundred discriminative features, each a threshold "stump" (feature, threshold, polarity), weighting examples the previous stumps got wrong.
4. **Cascade**: stages of increasing complexity; a window must pass every stage, so most background windows are rejected after 1-2 cheap stages.
OpenCV's `CascadeClassifier` is the runtime; `opencv_traincascade` / `opencv_createsamples` train one (OpenCV 3.x/4.x legacy tools; check your version).

## Is it suitable here? Assessment (reasoned, not measured on real data)
- **Rigid, consistent appearance** is what Haar cascades exploit (frontal faces). Cassava leaves are palmate with 5-9 lobes, variable pose, rotation, occlusion and overlap; Haar features are not rotation invariant. Expect many misses/false positives or a very large positive set.
- **The images are already mostly leaf close-ups**; a separate detector may add little. This is checkable: audit mask area fractions (`leaf_mask_hsv`) and Grad-CAM background reliance (`scripts/interpret.py`).
- **No annotations exist.** Training needs hundreds-thousands of boxed positives and several thousand negative crops. Boxing leaves by hand is the dominant cost.
- **Disease cues are inside the leaf** (mottling, streaks, lesions); a box does not add discriminative information, it only removes background. The HSV mask experiments (`hsv_crop`, `hsv_blank`) answer "does removing background help?" with zero annotation cost.
- Alternatives if localisation proves useful: colour/GrabCut segmentation, a small pretrained segmenter, or a lightweight CNN detector (all need some annotation or pretrained weights).

## What the repo provides
- `CascadeLeafDetector(cascade_path)`: wrapper with explicit errors, largest-box selection, minimum-area rule and **fallback to the full image on a miss**.
- `evaluate_detections` / `iou`: detection metrics (precision, recall/miss rate, mean best IoU) kept separate from classification metrics.
- Use: `data.localization=cascade data.cascade_path=leaf_cascade.xml` (commented variant in `configs/hybrid.yaml`).
- **No trained cascade is shipped.** Nothing here claims Viola-Jones works on cassava.

## Decision rule
Pursue annotation + cascade training only if the HSV-mask experiments show a gain beyond seed variation AND mask failure cases are the limiting factor. Otherwise report the negative/neutral result.
