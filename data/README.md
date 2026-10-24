# Dataset setup

No dataset is bundled or downloaded by this repository. Obtain it from its legitimate source and check its terms of use.

## Cassava Leaf Disease Classification (Kaggle, 2020)
1. Accept the competition rules at <https://www.kaggle.com/competitions/cassava-leaf-disease-classification> (login required).
2. Download via the Kaggle website or `kaggle competitions download -c cassava-leaf-disease-classification`.
3. Extract so the layout is:
```
data/cassava/
├── train.csv                        # image_id,label
├── label_num_to_disease_map.json    # {"0": "Cassava Bacterial Blight (CBB)", ...}
└── train_images/*.jpg
```
(Only the labelled training set is used; the competition test set has no public labels. We create our own splits.)
Kaggle competition data is released for the competition's permitted use; read the rules before redistributing anything derived from it.

## Alternative layout
Class-folder layout (`data/cassava/<class_name>/*.jpg`) is also supported. Class names are read from the data, never hardcoded.

## Prepare (audit, dedup, split, cache) - run once
```
python scripts/prepare_data.py --data-root data/cassava --cache-dir data/cache --image-size 128
```
Writes `images_128.npy`, `splits.csv` (with duplicate `group` ids), `class_names.json`, `audit_report.json`.
**Keep the split seed fixed for every experiment.** Read `audit_report.json`: class balance, invalid files, duplicate groups,
and groups with conflicting labels (a direct measure of label noise).

## Synthetic data (pipeline testing only)
`python scripts/make_synthetic_data.py --out data/synthetic` creates procedurally generated blobs, NOT leaves. Never report results from it as cassava results.
