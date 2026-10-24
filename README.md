# Cassava leaf disease: classical computer vision meets deep learning

A reproducible research framework that tests, under controlled ablations, whether **colour-space representations**, **frequency-domain features** and **background localisation** add anything to a CNN for cassava leaf disease classification - and measures robustness and interpretability along the way. Components are hypotheses to test, not assumed improvements.

> **Status (read this first).** The code, tests and a synthetic-data smoke run were executed in a CPU-only sandbox. **No experiment on the real cassava dataset has been run**, so there are **no real accuracy results** here. Numbers under "SYNTHETIC" in `docs/experiments.md` only prove the pipeline runs.

## Research questions
1. Do HSV / CIELAB (or RGB+Lab) inputs improve macro-F1 or illumination robustness for a fixed CNN?
2. Do FFT/DWT descriptors carry information complementary to CNN features (beyond shuffled/zero controls)?
3. Does annotation-free background removal help? Is Viola-Jones worth pursuing? (`docs/viola_jones.md`)
4. Where do models look (Grad-CAM) and do they rely on background shortcuts?

## Install
```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # Python >= 3.10; GPU optional (CPU works). Tested versions: requirements-tested.txt
pytest -q                         # 83 tests, synthetic fixtures only, ~10 s on CPU
```

## Dataset
See `data/README.md` (Kaggle Cassava Leaf Disease Classification; not bundled). Then:
```
python scripts/prepare_data.py --data-root data/cassava --cache-dir data/cache --image-size 128
```
This audits files, groups near-duplicates, builds leak-free stratified splits and caches resized images.

## Train / evaluate / infer
```
python scripts/train.py --config configs/baseline.yaml --variant cnn_rgb --seed 0
python scripts/train.py --config configs/color_space.yaml --variant cnn_lab --set train.epochs=15
python scripts/run_experiments.py --config configs/frequency.yaml           # all variants x seeds, resumable
python scripts/run_experiments.py --summarize --baseline cnn_rgb
python scripts/evaluate.py --checkpoint outputs/cnn_rgb/seed0/best.pt --split test --robustness
python scripts/interpret.py --checkpoint outputs/cnn_rgb/seed0/best.pt --split val --per-class 4
python scripts/predict.py --checkpoint outputs/cnn_rgb/seed0/best.pt leaf1.jpg leaf2.jpg
```
Smoke test without real data: `python scripts/make_synthetic_data.py` -> `prepare_data.py --data-root data/synthetic --cache-dir data/synthetic_cache --image-size 64` -> `run_experiments.py --config configs/smoke.yaml --output-dir outputs/smoke`.

## Layout
```
configs/   baseline | color_space | frequency | hybrid | smoke   (sweeps: variants x seeds; `extends`, `--set a.b=c`)
src/cassava/{data,preprocessing,features,models,training,evaluation,detection,interpretability,inference}
scripts/   prepare_data train run_experiments evaluate interpret predict make_synthetic_data
tests/     83 tests    docs/  methodology experiments limitations viola_jones
```
Models: compact CNN (~105k params), ResNet-18 / EfficientNet-B0 transfer (configurable freeze-then-finetune), CNN+frequency fusion (with shuffle/zeros controls), classical features + sklearn.

## Experiment table and results
See `docs/experiments.md` (hypotheses, decision rules, commands, measured-vs-synthetic results).

## Verified vs unverified
| Verified (ran in sandbox) | Not verified |
|---|---|
| 83 unit/integration tests pass (data, colour spaces, features, models, metrics vs sklearn, checkpoint round trip, inference, Grad-CAM, test-split isolation) | Any performance on real cassava images |
| All 7 smoke variants train/evaluate end to end (CNN, Lab, HSV-circ+photometric aug, fusion, scratch ResNet-18, classical, HSV-blank) | Pretrained-weight download and fine-tuning |
| Grad-CAM + shortcut diagnostics, paired comparison, predict CLI | GPU execution; Kaggle runtime |
| | Trained Viola-Jones cascade (none exists) |

## Credits and licensing
Dataset: Mwebaze et al., Makerere AI Lab / Kaggle Cassava Leaf Disease Classification (check its terms). Methods: Viola & Jones (2001); Grad-CAM (Selvaraju et al., 2017); ResNet (He et al., 2016); EfficientNet (Tan & Le, 2019). A literature search for novelty was NOT performed here; no novelty is claimed. Code: MIT (see LICENSE). torchvision pretrained weights are ImageNet-derived; check their terms before commercial use.
