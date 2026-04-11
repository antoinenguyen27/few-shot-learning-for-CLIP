# Results

Use JSONL result records. JSONL means one JSON object per line.

Result files live in:

```text
results/
```

This directory is ignored by git.

## Log One Run

Use this pattern after your method finishes evaluating:

```python
from common.evaluation.results import RunResult, append_result, result_jsonl_path

append_result(
    RunResult(
        method="LP++",
        dataset="eurosat",
        protocol="few_shot_all_classes",
        model_name=bundle.model_name,
        pretrained=bundle.pretrained,
        shots=1,
        seed=1,
        metrics={
            "val/top1_accuracy": 0.0,
            "val/macro_accuracy": 0.0,
            "test/top1_accuracy": 0.0,
            "test/macro_accuracy": 0.0,
        },
        split_path=str(datasets.split_file),
        notes="",
    ),
    result_jsonl_path("vit_b32_256_few_shot_all_classes"),
)
```

Replace the zeros with your real numbers.
Replace `LP++` with your method name.

Use `notes` for important differences, for example:

- teacher model used by PromptKD
- DPC annotation file
- checkpoint path
- transductive unlabeled data source
- changed batch size or training schedule

## Summarize Results

Run:

```bash
python3 scripts/summarize_results.py results/vit_b32_256_few_shot_all_classes.jsonl --metric test/top1_accuracy
```

This prints mean and standard deviation across seeds.

## Required Metric Names

For the first protocol, use:

- `val/top1_accuracy`
- `val/macro_accuracy`
- `test/top1_accuracy`
- `test/macro_accuracy`

For later base-to-new experiments, also use:

- `test/base_accuracy`
- `test/new_accuracy`
- `test/harmonic_mean`
