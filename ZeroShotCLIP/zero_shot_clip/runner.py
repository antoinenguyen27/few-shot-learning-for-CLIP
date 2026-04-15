#!/usr/bin/env python
"""Run the frozen zero-shot CLIP baseline on one shared few-shot split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.datasets.sources import DATASET_SOURCES
from common.datasets.torch_dataset import build_data_loaders, build_split_datasets
from common.evaluation.results import RunResult, append_result, result_jsonl_path
from common.models.openclip import build_openclip_bundle

from ZeroShotCLIP.zero_shot_clip.config import ZeroShotCLIPConfig
from ZeroShotCLIP.zero_shot_clip.method import ZeroShotCLIPMethod

ZERO_SHOT_RESULT_SHOTS = 0
ZERO_SHOT_RESULT_SEED = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_SOURCES), required=True)
    parser.add_argument(
        "--split-shots",
        "--shots",
        dest="split_shots",
        type=int,
        default=1,
        help=(
            "Existing few-shot split shot count used only to choose val/test IDs. "
            "Zero-shot result rows always log shots=0."
        ),
    )
    parser.add_argument(
        "--split-seed",
        "--seed",
        dest="split_seed",
        type=int,
        default=1,
        help=(
            "Existing few-shot split seed used only to choose val/test IDs. "
            "Zero-shot result rows always log seed=0."
        ),
    )
    parser.add_argument("--protocol", default="few_shot_all_classes")
    parser.add_argument("--data-root", default=None, help="Defaults to ./data or FSL_CLIP_DATA.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--precision", choices=("fp32", "amp"), default="fp32")
    parser.add_argument("--template", action="append", dest="templates", help="Custom prompt template; repeat to average.")
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--no-log", action="store_true", help="Run evaluation without appending a result JSONL row.")
    parser.add_argument("--run-name", default="vit_b32_256_few_shot_all_classes")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = ZeroShotCLIPConfig(
        precision=args.precision,
        max_eval_batches=args.max_eval_batches,
        show_progress=not args.no_progress,
    )

    bundle = build_openclip_bundle(device=args.device, precision="fp32")
    split_datasets = build_split_datasets(
        dataset=args.dataset,
        protocol=args.protocol,
        shots=args.split_shots,
        seed=args.split_seed,
        train_transform=bundle.preprocess_eval,
        eval_transform=bundle.preprocess_eval,
        data_root=args.data_root,
    )
    loaders = build_data_loaders(
        split_datasets,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        train_shuffle=False,
    )

    method = ZeroShotCLIPMethod(config=config, templates=args.templates)
    artifact = method.fit(
        loaders["train"],
        loaders["val"],
        split_datasets.classnames,
        bundle,
    )
    val_metrics = method.evaluate(
        artifact,
        loaders["val"],
        split_datasets.classnames,
        bundle,
        metric_prefix="val",
    )
    test_metrics = method.evaluate(
        artifact,
        loaders["test"],
        split_datasets.classnames,
        bundle,
        metric_prefix="test",
    )
    metrics = {**val_metrics, **test_metrics}

    split_source = {
        "shots": args.split_shots,
        "seed": args.split_seed,
        "split_path": str(split_datasets.split_file),
    }
    payload = {
        "logged_seed": ZERO_SHOT_RESULT_SEED,
        "logged_shots": ZERO_SHOT_RESULT_SHOTS,
        "metrics": metrics,
        "split_source": split_source,
        "templates": artifact.templates,
    }
    if not args.no_log:
        output = append_result(
            RunResult(
                method=method.method_name,
                dataset=args.dataset,
                protocol=args.protocol,
                model_name=bundle.model_name,
                pretrained=bundle.pretrained,
                shots=ZERO_SHOT_RESULT_SHOTS,
                seed=ZERO_SHOT_RESULT_SEED,
                metrics=metrics,
                split_path=str(split_datasets.split_file),
                notes="Frozen OpenCLIP zero-shot classifier; no train images are fitted.",
                extra={
                    "config": config.to_dict(),
                    "artifact_metadata": artifact.metadata,
                    "split_source": split_source,
                },
            ),
            result_jsonl_path(args.run_name),
        )
        payload["result_path"] = str(output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
