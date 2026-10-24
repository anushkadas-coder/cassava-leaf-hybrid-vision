# Experiments

All variants share data splits, augmentation, optimiser, epoch budget and early-stopping rule (set per config file). Run order = scientific value / cost.

| ID | Config / variant(s) | Question | Hypothesis (prior) |
|---|---|---|---|
| E0 | `prepare_data.py` | Data quality, duplicates, label conflicts | Duplicates/noise present (known from the community) |
| E1 | `baseline.yaml` `cnn_rgb`, `resnet18_rgb` | Reference performance | Pretrained > scratch |
| E2 | `color_space.yaml` rgb / hsv / hsv_circ / lab / rgb_lab | Does representation matter? | Small clean-accuracy change; better illumination robustness for Lab/HSV (weak-moderate prior) |
| E2b | `cnn_*_photoaug` | Does photometric augmentation close the robustness gap? | Yes, partly |
| E3 | `frequency.yaml` freq_only_*, color_only_*, dwt_only | How far do interpretable features go? | Well below CNN; better than majority baseline |
| E4 | `fusion_cnn_freq` vs `cnn_rgb` vs `fusion_control_shuffle/zeros` | Complementary information? | Little/no gain beyond controls (honest prior) |
| E5 | robustness block (`eval.robustness: true`) | Illumination/blur/noise/JPEG sensitivity | See E2 |
| E6 | `scripts/interpret.py` | Background reliance, explanation consistency | Some background reliance in field photos |
| E7 | `hybrid.yaml` hsvcrop / hsvblank | Does removing background help? | Neutral or slight gain; may hurt if lesions are masked |
| E8 | `fusion_lab`, `fusion_lab_hsvcrop` | Combined model | Run only if E2/E4/E7 each show gains |
| E9 | Viola-Jones | Needs annotations (`docs/viola_jones.md`) | Likely poor for palmate leaves |

## Decision rules (fix BEFORE looking at results)
Keep a component only if, over >= 3 seeds, its mean test macro-F1 (or robustness mean/worst-case macro-F1) beats the reference by more than the larger seed std AND the paired McNemar test on pooled predictions is not consistent with chance (p < 0.05, treat as indicative) AND, for fusion, it also beats the `shuffle` and `zeros` controls AND the latency/parameter cost is acceptable. Otherwise report it as not supported. Select hyper-parameters/models on validation only.

## Commands
```
python scripts/prepare_data.py --data-root data/cassava --cache-dir data/cache --image-size 128
python scripts/run_experiments.py --config configs/baseline.yaml
python scripts/run_experiments.py --config configs/color_space.yaml
python scripts/run_experiments.py --config configs/frequency.yaml
python scripts/run_experiments.py --config configs/hybrid.yaml
python scripts/run_experiments.py --summarize --baseline cnn_rgb          # outputs/summary.{csv,md}
python scripts/interpret.py --checkpoint outputs/cnn_rgb/seed0/best.pt --split val
python scripts/evaluate.py --compare outputs/cnn_rgb/seed0 outputs/fusion_cnn_freq/seed0
```
Note: variants named `cnn_rgb` in several configs share an output folder/log key; run configs into separate `experiment.output_dir` (e.g. `--set experiment.output_dir=outputs/color`) if you want isolated logs. Reduce cost on a weak CPU: `--set data.image_size=96 data.subset=8000 train.epochs=15`, or run on a Kaggle GPU notebook with identical code.

## Experiment log
`outputs/experiment_log.jsonl` (one JSON per run): config, seed, split sizes, model/representation, hardware, validation metrics, test metrics (written after model selection), bootstrap CIs, per-class metrics, confusion matrix, latency, parameters, robustness results, checkpoint path.

## Measured results

### Real cassava data: NONE YET
Run the commands above on your machine/Kaggle and paste `outputs/summary.md` here.

### SYNTHETIC smoke run (pipeline verification only - NOT cassava, single seed, 4 epochs, 64 px)
Produced by `python scripts/run_experiments.py --config configs/smoke.yaml` on procedurally generated images in the build sandbox (1 CPU core).
Purpose: show every code path executes end to end. The synthetic classes were constructed to differ in colour/texture, so e.g. the near-perfect `freq_logreg` score reflects how the generator was built, not any property of cassava. Do not interpret differences between rows.

| variant | kind | repr | test macro-F1 | test acc | params | CPU ms/img |
|---|---|---|---|---|---|---|
| cnn_rgb | cnn | rgb | 0.702 | 0.593 | 26,573 | 0.87 |
| cnn_lab | cnn | lab | 0.626 | 0.559 | 26,573 | 1.42 |
| cnn_hsvcirc_photo | cnn | hsv_circ | 0.485 | 0.610 | 26,645 | 0.98 |
| fusion | fusion | rgb | 0.893 | 0.864 | 29,938 | 2.49 |
| resnet18_scratch | transfer | rgb | 0.454 | 0.542 | 11,179,077 | 8.40 |
| freq_logreg | classical | rgb | 1.000 | 1.000 | 220 | 1.96 |
| cnn_hsvblank | cnn | rgb | 0.751 | 0.695 | 26,573 | 0.92 |
