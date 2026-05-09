# Data Directory

This directory is the default local data root. You can override it with:

```bash
export FSL_CLIP_DATA=/absolute/path/to/data
```

Large data files are ignored by git. The repo tracks only this documentation and `.gitkeep` placeholders.

## Layout

```text
data/
  raw/
    eurosat/
      SOURCE_PATH.txt
    flowers102/
      SOURCE_PATH.txt
    stanford_cars/
      SOURCE_PATH.txt
  manifests/
    eurosat.jsonl
    flowers102.jsonl
    stanford_cars.jsonl
  splits/
    <dataset>/<protocol>/shots_<k>/seed_<seed>.json
  cache/
    features/
```

`scripts/download_data.py` uses `kagglehub.dataset_download(...)`. KaggleHub stores the actual data in its cache, so the script writes `SOURCE_PATH.txt` files pointing to the cache path. If you already have a dataset locally, set one of:

```bash
export FSL_CLIP_EUROSAT_ROOT=/path/to/eurosat
export FSL_CLIP_FLOWERS102_ROOT=/path/to/flowers102
export FSL_CLIP_STANFORD_CARS_ROOT=/path/to/stanford_cars
```

Then run:

```bash
python3 scripts/inspect_data.py --datasets eurosat flowers102 stanford_cars
```

Do not commit generated manifests, split JSON files, caches, or raw data.
