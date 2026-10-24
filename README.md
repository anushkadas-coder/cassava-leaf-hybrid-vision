# cassava-leaf-hybrid-vision

**Does classical computer vision still add value to a CNN for cassava leaf disease classification?**

A reproducible research framework that tests, under controlled ablations, whether **color-space representations** (HSV, CIELAB), **frequency-domain features** (FFT, wavelets) and **background localization** improve a CNN baseline in macro-F1, illumination robustness, interpretability or compute cost. Components are hypotheses to test, not assumed improvements.

> **Status (read this first).** The code, tests and a synthetic-data smoke run were executed in a CPU-only environment. **No experiment on the real cassava dataset has been run**, so there are **no real accuracy results** here. Numbers marked "SYNTHETIC" in `docs/experiments.md` only prove the pipeline runs. 

## Research questions
1. Do HSV / CIELAB (or RGB+Lab) inputs improve macro-F1 or illumination robustness for a fixed CNN?
2. Do FFT/DWT descriptors carry information complementary to CNN features (beyond shuffled/zero controls)?
3. Does annotation-free background removal help? Is Viola-Jones worth pursuing? (`docs/viola_jones.md`)
4. Where do models look (Grad-CAM), and do they rely on background shortcuts?

## Features
- Leak-free data pipeline: file validation, near-duplicate grouping (dHash), stratified group splits
- Baselines: compact CNN (~105k params), ResNet-18 / EfficientNet-B0 transfer learning
- Color spaces: RGB, HSV, circular-hue HSV, CIELAB, RGB+Lab
- Frequency features: FFT and DWT descriptors, classical classifiers, CNN+frequency fusion with shuffle/zero controls
- Localization: HSV leaf mask (crop/blank) and a Viola-Jones interface with detection metrics
- Interpretability: Grad-CAM plus shortcut tests (background/leaf blanking)
- Evaluation: macro-F1, per-class metrics, bootstrap CIs, robustness suite, CPU latency
- Resumable multi-seed sweeps with a JSONL experiment log

## Install
```bash
git clone https://github.com/anushkadas-coder/cassava-leaf-hybrid-vision.git
cd cassava-leaf-hybrid-vision
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # Python >= 3.10; GPU optional (CPU works)
pytest -q                          # 83 tests on synthetic fixtures, ~10 s on CPU
```
Versions used for testing: `requirements-tested.txt`.

## Dataset
The dataset is not bundled. See `data/README.md` (Kaggle Cassava Leaf Disease Classification), then:
```bash
python scripts/prepare_data.py --data-root data/cassava --cache-dir data/cache --image-size 128
```
This audits files, groups near-duplicates, builds leak-free stratified splits and caches resized images. Check `data/cache/audit_report.json` afterwards.

## Train / evaluate / infer
```bash
python scripts/train.py --config configs/baseline.yaml --variant cnn_rgb --seed 0
python scripts/train.py --config configs/color_space.yaml --variant cnn_lab --set train.epochs=15
python scripts/run_experiments.py --config configs/frequency.yaml     # all variants x seeds, resumable
python scripts/run_experiments.py --summarize --baseline cnn_rgb
python scripts/evaluate.py --checkpoint outputs/cnn_rgb/seed0/best.pt --split test --robustness
python scripts/interpret.py --checkpoint outputs/cnn_rgb/seed0/best.pt --split val --per-class 4
python scripts/predict.py --checkpoint outputs/cnn_rgb/seed0/best.pt leaf1.jpg leaf2.jpg
```
Smoke test without real data: `python scripts/make_synthetic_data.py`, then `prepare_data.py --data-root data/synthetic --cache-dir data/synthetic_cache --image-size 64`, then `run_experiments.py --config configs/smoke.yaml --output-dir outputs/smoke`.

## Layout
```
configs/   baseline | color_space | frequency | hybrid | smoke
src/cassava/{data,preprocessing,features,models,training,evaluation,detection,interpretability,inference}
scripts/   prepare_data train run_experiments evaluate interpret predict make_synthetic_data
tests/     83 tests
docs/      methodology | experiments | limitations | viola_jones
```

## Results
*Not yet measured on real data.* Fill this table from `outputs/summary.md` after running the experiments. Hypotheses and decision rules for keeping or rejecting each component are in [`docs/experiments.md`](docs/experiments.md).

| Variant | Test macro-F1 (mean ± std) | Test accuracy | Robustness (mean macro-F1) | CPU ms/img | Params |
|---|---|---|---|---|---|
| cnn_rgb | | | | | |
| cnn_lab | | | | | |
| fusion_cnn_freq | | | | | |

## Verified vs unverified
| Verified (ran in the build environment) | Not verified |
|---|---|
| 83 unit/integration tests pass (data, color spaces, features, models, metrics vs sklearn, checkpoint round trip, inference, Grad-CAM, test-split isolation) | Any performance on real cassava images |
| All 7 smoke variants train/evaluate end to end | Clean-environment `pip install -e ".[dev]"` |
| Grad-CAM + shortcut diagnostics, paired comparison, predict CLI | Pretrained-weight download and fine-tuning |
| | GPU execution; Kaggle runtime |
| | Trained Viola-Jones cascade (none exists) |

## Limitations
See [`docs/limitations.md`](docs/limitations.md). In short: dataset label noise, undetected crops/rotations of duplicate photos, square resizing, a heuristic leaf mask, and synthetic-only robustness tests. Nothing here supports field-level diagnostic claims.

## Credits and licensing
Dataset: Mwebaze et al., Makerere AI Lab / Kaggle Cassava Leaf Disease Classification (check its terms). Methods: Viola & Jones (2001); Grad-CAM (Selvaraju et al., 2017); ResNet (He et al., 2016); EfficientNet (Tan & Le, 2019). A literature search for novelty was not performed; no novelty is claimed. Code: MIT (see `LICENSE`). Torchvision pretrained weights are ImageNet-derived; check their terms before commercial use.
