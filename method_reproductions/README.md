# few-shot-learning-for-CLIP

This repo is for a controlled few-shot CLIP comparison. The team should implement methods in their assigned method folders, while the shared data loading, OpenCLIP setup, split generation, metrics, and result logging live in `common/`.

If you are new to the repo, start here:

```text
START_HERE.md
```

## What You Should Edit

- PromptSRC owner: edit `Promptsrc/`
- LP++ owner: edit `LP++/`
- DPC owner: edit `DPC/`
- PromptKD owner: edit `promptkd/`

Try not to edit `common/` unless you are intentionally changing the shared pipeline.

## Shared Experiment Defaults

- Model: OpenCLIP `ViT-B-32-256`
- Pretrained weights: `datacomp_s34b_b86k`
- Image preprocessing: OpenCLIP train/eval transforms
- Datasets: EuroSAT, Oxford Flowers 102, Stanford Cars
- Shots: `1, 2, 4, 8, 16`
- Seeds: `1, 2, 3`
- First protocol: `few_shot_all_classes`
- Main metric: test top-1 accuracy
- Secondary metric: macro accuracy

## Quick Commands

Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Download and prepare data:

```bash
python3 scripts/download_data.py --datasets eurosat flowers102 stanford_cars
python3 scripts/inspect_data.py --datasets eurosat flowers102 stanford_cars
python3 scripts/build_manifests.py --datasets eurosat flowers102 stanford_cars
python3 scripts/build_splits.py --datasets eurosat flowers102 stanford_cars --shots 1 2 4 8 16 --seeds 1 2 3
```

Run local checks:

```bash
python3 -m compileall common scripts Promptsrc 'LP++' DPC promptkd tests
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Summarize results after methods start logging:

```bash
python3 scripts/summarize_results.py results/vit_b32_256_few_shot_all_classes.jsonl --metric test/top1_accuracy
```

## Rules For Valid Results

- Use the shared split JSON files.
- Train on train data only.
- Use validation data for choices like learning rate or epoch selection.
- Use test data only for final reporting.
- Do not create private few-shot samples inside a method folder.
- Do not change preprocessing for one method unless the team agrees.
- Log teacher models, extra annotations, unlabeled data, or checkpoints in the result record.

More detail:

- `START_HERE.md`: beginner walkthrough.
- `docs/data.md`: dataset and split details.
- `docs/method-contract.md`: how method code should use `common/`.
- `docs/results.md`: how to log and summarize results.
- `docs/git-for-team.md`: beginner Git workflow.
