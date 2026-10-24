# Limitations

- **No real-data results exist in this repository.** The sandbox used to build it had no access to the Kaggle dataset, a GPU, or the internet for pretrained weights. All numbers in `docs/experiments.md` marked SYNTHETIC come from procedurally generated images and carry no scientific meaning about cassava.
- ImageNet-pretrained ResNet-18 (`configs/baseline.yaml`) was only exercised with `pretrained: false` (random init, shape/plumbing tests). Downloading weights and fine-tuning is untested here.
- Viola-Jones: no trained cascade; only the building blocks, wrapper and metrics are tested (on synthetic data).
- Leaf mask is an HSV heuristic; fails on brown/pale tissue at leaf edges, green weeds, unusual lighting.
- dHash duplicates miss crops/rotations; label noise is reported (conflicting-label groups) but not corrected.
- Images are squashed to square (aspect ratio not preserved) at 128 px by default; small lesions may be lost.
- Synthetic robustness tests do not replace field validation; nothing here supports deployment or diagnostic claims.
- Multi-seed statistics: 3 seeds give weak variance estimates; use a paired test (`scripts/evaluate.py --compare`, McNemar on the same test images) and more seeds for borderline differences.
- Latency was measured on whichever CPU runs the code; compare only within one machine.
- Tested only on Linux/CPU with the versions in `requirements-tested.txt`.
