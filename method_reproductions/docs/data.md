# Data Pipeline

The data pipeline has three stages:

1. Download or point to raw data.
2. Build a normalized manifest.
3. Build deterministic few-shot splits.

All methods must consume the same generated manifests and split JSON files.

## Download

Install dependencies first:

```bash
pip install -e .
```

Then run:

```bash
python3 scripts/download_data.py --datasets eurosat flowers102 stanford_cars
```

The downloader uses these KaggleHub handles:

- `apollo2506/eurosat-dataset`
- `nunenuh/pytorch-challange-flower-dataset`
- `eduardo4jesus/stanford-cars-dataset`

KaggleHub downloads into its cache. The script records that cache path in:

```text
data/raw/<dataset>/SOURCE_PATH.txt
```

If the dataset is already on disk, set:

```bash
export FSL_CLIP_EUROSAT_ROOT=/path/to/eurosat
export FSL_CLIP_FLOWERS102_ROOT=/path/to/flowers102
export FSL_CLIP_STANFORD_CARS_ROOT=/path/to/stanford_cars
```

## Expected Raw Layouts

EuroSAT:

```text
EuroSAT/
  AnnualCrop/*.jpg
  Forest/*.jpg
  HerbaceousVegetation/*.jpg
  Highway/*.jpg
  Industrial/*.jpg
  Pasture/*.jpg
  PermanentCrop/*.jpg
  Residential/*.jpg
  River/*.jpg
  SeaLake/*.jpg
EuroSATallBands/
  ...
```

Use only `EuroSAT/` RGB JPG images for the shared OpenCLIP experiments.

Flowers102:

```text
dataset/
  train/<class_id>/*.jpg
  valid/<class_id>/*.jpg
  test/<class_id>/*.jpg
cat_to_name.json
```

The adapter maps numeric class folders through `cat_to_name.json`.

Stanford Cars:

```text
cars_train/**/*.jpg
cars_test/**/*.jpg
devkit/cars_meta.mat
devkit/cars_train_annos.mat
cars_test_annos_withlabels.mat
```

The adapter also supports common CSV annotation fallbacks. Always run `inspect_data.py` after the first download because Kaggle mirrors can vary in wrapper directory names.

## Manifest

Build manifests:

```bash
python3 scripts/build_manifests.py --datasets eurosat flowers102 stanford_cars
```

Each line in `data/manifests/<dataset>.jsonl` is an `ImageRecord`:

```json
{
  "dataset": "eurosat",
  "sample_id": "eurosat/EuroSAT/Forest/Forest_1",
  "image_path": "EuroSAT/Forest/Forest_1.jpg",
  "label_id": 1,
  "class_name": "forest",
  "source_split": "all",
  "metadata": {
    "folder_class": "Forest"
  }
}
```

Generated manifests are ignored by git because they are derived artifacts and can change if a Kaggle mirror changes its wrapper directories.

## Split Policy

Build splits:

```bash
python3 scripts/build_splits.py --datasets eurosat flowers102 stanford_cars --shots 1 2 4 8 16 --seeds 1 2 3
```

Current protocol: `few_shot_all_classes`.

The split logic is:

- If source train/val/test exists, use it.
- If source train/test exists, carve validation from train with a deterministic stratified split.
- If only class folders exist, create deterministic stratified train/val/test.
- Then sample `k` training examples per class from the train pool for each seed.

The split generator fails if a class has fewer than `k` train-pool examples unless `--allow-fewer` is passed. Do not pass `--allow-fewer` for main results unless the team explicitly decides to do so.

## No-Leakage Rules

- The test split is used only once for final reporting.
- Hyperparameters are selected with training and validation only.
- If a method uses unlabeled domain images, document which split those images came from.
- PromptKD-style unlabeled distillation using test images is a separate transductive setting and must not be mixed with strict few-shot results.
