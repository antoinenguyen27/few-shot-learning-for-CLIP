# Zero-Shot CLIP Workspace

This folder contains the frozen OpenCLIP zero-shot baseline. It is the control
row for the few-shot comparison: no train images are fitted, but results are
logged for the same dataset/shot/seed split files so summaries line up with the
trained methods.

## Run One Split

From the repo root:

```bash
python3 ZeroShotCLIP/zero_shot_clip/runner.py --dataset eurosat --device cpu
```

Use `--device cuda` when a CUDA GPU is available. Add `--no-log` for a smoke
test that prints metrics without appending to `results/`.

## What It Uses

- `common.models.openclip.build_openclip_bundle` for the shared OpenCLIP model
  and preprocessing.
- `common.datasets.templates.get_templates` for dataset-specific class prompts.
- `common.models.openclip.build_zero_shot_classifier` for the averaged text
  classifier.
- `common.evaluation.results.RunResult` for standardized JSONL logging.

## Notes

- Result rows log `shots=0` and `seed=0` because the method is frozen and does
  not fit train images.
- By default, the runner reads the `shots_1/seed_1` split file for validation
  and test IDs. Use `--split-shots` and `--split-seed` to choose a different
  existing split source.
- Validation metrics are reported for comparison only; the baseline has no
  learned settings to select.
- Custom prompts can be supplied by repeating `--template`, for example:

```bash
python3 ZeroShotCLIP/zero_shot_clip/runner.py --dataset flowers102 --template "a photo of a {} flower."
```
