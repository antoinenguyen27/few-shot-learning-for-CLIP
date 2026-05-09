# PromptSRC-NC

Neighborhood-Consistent PromptSRC for unlabeled geometry-aware few-shot CLIP adaptation.

This repository is now a research workspace for a new method, not only a method reproduction sandbox. The active project asks:

> Can a completed few-shot PromptSRC solution be improved by a short second-stage adaptation that uses unlabeled target images only through frozen-CLIP nearest-neighbor relations?

The proposed method, **PromptSRC-NC: Neighborhood-Consistent PromptSRC**, starts from official PromptSRC, constructs frozen-CLIP nearest-neighbor pairs over unlabeled target images, and continues prompt optimization with a symmetric prediction-consistency loss over those pairs. It does not assign pseudo-labels, train a teacher/student system, recover a full manifold, or discover new classes.

## Research Claim

The intended contribution is modest and testable:

> A short, teacher-free unlabeled adaptation stage can improve PromptSRC when it uses meaningful frozen-CLIP neighbor structure. The shuffled-neighbor control tests whether any gain comes from real local geometry rather than extra training time or arbitrary smoothing.

If results are mixed, the scientifically valid conclusion is dataset-conditional: frozen CLIP neighborhoods may help when they align with class semantics and hurt when fine-grained classes are visually close but label-distinct.

## Active Method

Primary method name:

```text
PromptSRC-NC: Neighborhood-Consistent PromptSRC
```

Proposal title:

```text
Neighborhood-Consistent PromptSRC: Using Unlabeled Target Geometry for Few-Shot CLIP Prompt Learning
```

Core flow:

1. Train official PromptSRC on the few-shot labeled split.
2. Embed unlabeled training images with the unprompted frozen CLIP image encoder.
3. Build fixed mutual nearest-neighbor pairs.
4. Initialize from the completed PromptSRC checkpoint.
5. Continue prompt optimization with PromptSRC losses plus neighborhood prediction consistency.
6. Compare against a shuffled-neighbor control with the same images, same loss, and same training time.

## Repository Layout

- `promptSRC-NC/`: active research proposal and technical specification for the new method.
- `method_reproductions/`: archived reproduction/comparison scaffolding for PromptSRC, LP++, DPC, and PromptKD.
- `common/`: shared data, OpenCLIP, split, metric, feature-cache, and result helpers from the earlier comparison scaffold.
- `scripts/`: thin command-line wrappers around `common/`.
- `data/`: local generated manifests, splits, caches, and raw-data pointers. Do not commit generated data.
- `results/`: local result logs. Do not commit result dumps unless explicitly requested.
- `tests/`: tests for shared infrastructure and archived method code.

## Source Of Truth

Use sources in this order:

1. The official papers and official repositories for PromptSRC, LP++, DPC, PromptKD, CLIP, and related baselines.
2. `promptSRC-NC/neighborhood_promptsrc_research_proposal.md` for the research framing.
3. `promptSRC-NC/neighborhood_promptsrc_technical_spec.md` for the active implementation plan.
4. `common/` and `scripts/` for reusable local infrastructure.
5. `method_reproductions/` only as historical context.

The archived reproductions may be incomplete or inaccurate. Do not treat them as faithful implementations of the papers unless they have been checked against the official paper and official repo.

## Primary Experiment

Datasets:

- Flowers102, using PromptSRC dataset arg `oxford_flowers`
- EuroSAT, using PromptSRC dataset arg `eurosat`
- Stanford Cars, using PromptSRC dataset arg `stanford_cars`

Primary setting:

- Few-shot all-class learning.
- Start with 16 shots.
- Use seeds `1, 2, 3`.
- Use the remaining training-split images as the unlabeled pool.
- Do not use test images in the unlabeled pool.

Required variants:

1. `PromptSRC`: official baseline.
2. `PromptSRC-NC`: real frozen-CLIP neighbor pairs.
3. `PromptSRC-NC shuffled`: shuffled-neighbor control.

Primary metric:

```text
test top-1 accuracy, reported as mean +/- standard deviation over seeds
```

Required diagnostics:

- number of unlabeled images and neighbor pairs;
- mean real-pair and shuffled-pair cosine similarity;
- edge disagreement on real neighbor pairs;
- mean JS divergence on real neighbor pairs;
- mean entropy and confidence on the unlabeled pool.

## Scientific Guardrails

- Do not invent new train/validation/test splits inside a method folder.
- Do not tune hyperparameters on test results.
- Do not mix strict few-shot and transductive results.
- Do not use test images as unlabeled data in the primary setting.
- Do not describe PromptSRC-NC as pseudo-labeling.
- Do not add pseudo-labeling, entropy minimization, EMA teachers, PromptKD-style distillation, graph Laplacian variants, relation preservation, confidence thresholds, or graph-refresh schedules to the main method unless the research question is explicitly changed.
- Do not commit datasets, model weights, checkpoints, feature caches, neighbor caches, or large result dumps.
- Do not use notebooks for implementation.

## Implementation Notes

The active technical spec is written around the official PromptSRC repository and its Dassl/OpenAI-CLIP-style mechanics. Preserve official PromptSRC behavior first: backbone, transforms, prompt depths, PromptSRC self-regularization losses, GPA checkpointing, and dataset split semantics.

The earlier `common/` OpenCLIP scaffold is useful for shared data inspection, split/result utilities, and archived method comparison work. It should not override official PromptSRC behavior for PromptSRC-NC when the two differ.

For the main PromptSRC-NC implementation, keep the method narrow:

- mutual top-1 neighbors with top-5 fallback only if too few pairs are obtained;
- global defaults for `LAMBDA_NC`, Stage 2 epochs, and neighbor `k`;
- Stage 2 initialized from PromptSRC GPA weights;
- Stage 2 checkpoint as final prompt weights, without adding a new GPA knob unless unavoidable;
- all method deviations recorded in metadata and result notes.

## Setup

For the shared local Python utilities:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Prepare local data for the shared scaffold:

```bash
python3 scripts/download_data.py --datasets eurosat flowers102 stanford_cars
python3 scripts/inspect_data.py --datasets eurosat flowers102 stanford_cars
python3 scripts/build_manifests.py --datasets eurosat flowers102 stanford_cars
python3 scripts/build_splits.py --datasets eurosat flowers102 stanford_cars --shots 1 2 4 8 16 --seeds 1 2 3
```

For PromptSRC-NC runs that use the official PromptSRC codebase, follow the environment and dataset layout in:

```text
promptSRC-NC/neighborhood_promptsrc_technical_spec.md
```

## Reading Order

Start here:

1. `promptSRC-NC/neighborhood_promptsrc_research_proposal.md`
2. `promptSRC-NC/neighborhood_promptsrc_technical_spec.md`
3. Official PromptSRC paper and repository
4. `method_reproductions/README.md`
5. `method_reproductions/docs/method-contract.md`
6. `method_reproductions/docs/experiment-protocol.md`

Use the archived reproduction docs to understand the earlier scaffold and its constraints. Use official papers and repos to decide what the methods actually do.
