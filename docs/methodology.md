# Methodology

## Research question
Can colour-space representations and frequency-domain features provide information complementary to a CNN for cassava leaf disease classification, and under what conditions does combining them improve performance, robustness, interpretability or efficiency? Secondary: is localisation (annotation-free HSV mask, or Viola-Jones) useful before classification?

## Data handling (leakage control)
1. Audit: unreadable files, label sanity, size distribution, class balance.
2. **Near-duplicate grouping** with a 64-bit dHash (Hamming <= 4, single linkage). Duplicates are common in field datasets; the same photo in train and test inflates scores. dHash is robust to resize/recompression/brightness shifts, not to crops/rotations, so residual leakage remains possible.
3. **StratifiedGroupKFold** splits: duplicate groups never straddle splits; class proportions approximately preserved. One fixed split seed for all experiments.
4. Normalisation statistics (channel mean/std, frequency feature mean/std) are fitted on a training subsample only and stored in the checkpoint.
5. Model selection (early stopping, best epoch) uses validation macro-F1 only. The test split is evaluated once per run after the best-validation weights are restored. A test (`test_test_split_cannot_influence_training_or_model_selection`) scrambles test labels and verifies training history and validation metrics are unchanged.

## Representations (`preprocessing/colorspace.py`)
- **HSV**: hue (angle), saturation, value. Hue is circular and ill-defined at low saturation -> `hsv_circ` uses (sin, cos, S, V).
- **CIELAB**: L (lightness) separated from a, b (opponent chromatic axes), approximately perceptually uniform. Illumination mostly changes L and scales brightness, so a/b may be more stable (hypothesis).
- `rgb_lab` concatenates 6 channels. Conversions follow OpenCV float32 conventions; channels are scaled to ~[0,1] then standardised with TRAIN statistics. ImageNet normalisation is allowed only for RGB.
- Confounds to remember: a CNN's first layer can already learn linear colour mixes; only the non-linearity of HSV/Lab is new. With pretrained weights, non-RGB inputs are off-distribution (first conv is adapted heuristically).

## Frequency features (`features/frequency.py`)
- **FFT** of the Hann-windowed grey image (windowing limits edge leakage). Features: relative energy in radial bands (coarse -> fine texture), angular bands (orientation structure), spectral centroid, normalised entropy, log-log radial slope (1/f^2-like statistics of natural images), total log energy. A global spectrum has no spatial localisation: it is a texture descriptor, not a lesion detector.
- **DWT** (db2, 3 levels): per detail subband log-energy, std, mean |coef|, plus approximation energy. Wavelets keep scale and location information in subbands.
- Hypothesis (to be tested): lesion texture (fine mottling in CGM, streaks in CBSD) is partly captured by band statistics. Prior expectation: modest or no gain over a CNN that learns its own filters.

## Fusion (`models/fusion.py`)
`logits = Linear(concat(CNN embedding, MLP(frequency vector)))`. Simplest defensible design: late feature-level fusion. **Controls** `shuffle` (frequency vectors permuted across the batch) and `zeros` isolate information from extra parameters; a fusion gain must exceed these controls.

## Localisation (`detection/`)
`hsv_crop` / `hsv_blank` (annotation-free), with automatic fallback when the mask is implausible; Viola-Jones interface (see `viola_jones.md`).

## Interpretability (`interpretability/`)
Grad-CAM: `ReLU(sum_k alpha_k A^k)` with `alpha_k` = spatially averaged gradient of the class score. Reported with quantitative companions: fraction of CAM mass inside the leaf mask, accuracy with background blanked, accuracy with the leaf blanked (shortcut warning), CAM correlation between models. Grad-CAM is not an accuracy method and plausible heatmaps do not establish correct reasoning.

## Evaluation
Accuracy, macro-F1 (primary: classes are imbalanced), per-class precision/recall/F1, confusion matrices, bootstrap 95% CIs (resampling test images), end-to-end CPU latency (preprocessing + model, batch 1), parameters, size. Robustness: brightness, gamma, white balance, blur, noise, JPEG on a fixed test subset (identical across runs). These are synthetic stress tests, not field validation.

## Threats to validity
Label noise (known in this dataset), residual undetected near-duplicates, single dataset/acquisition style, resizing to square distorts aspect ratio, few seeds, hyper-parameters not tuned per variant (risk of favouring the baseline), pretrained-weight mismatch for non-RGB inputs, heuristic leaf mask.
