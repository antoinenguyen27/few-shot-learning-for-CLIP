# Scripts

These scripts are thin command-line wrappers around `common/`.

- `download_data.py`: downloads selected Kaggle datasets with KaggleHub and writes `SOURCE_PATH.txt` markers.
- `inspect_data.py`: parses raw datasets without writing manifests and prints counts by split/class.
- `build_manifests.py`: writes normalized JSONL manifests under `data/manifests/`.
- `build_splits.py`: writes deterministic few-shot split JSON files under `data/splits/`.
- `summarize_results.py`: summarizes persisted JSONL run results across seeds.

Use them from the repo root:

```bash
python3 scripts/download_data.py --datasets eurosat flowers102 stanford_cars
python3 scripts/inspect_data.py --datasets eurosat flowers102 stanford_cars
python3 scripts/build_manifests.py --datasets eurosat flowers102 stanford_cars
python3 scripts/build_splits.py --datasets eurosat flowers102 stanford_cars --shots 1 2 4 8 16 --seeds 1 2 3
python3 scripts/summarize_results.py results/vit_b32_256_few_shot_all_classes.jsonl
```
