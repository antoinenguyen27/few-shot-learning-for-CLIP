#!/usr/bin/env python
"""Run DPC on one shared few-shot split."""

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

from DPC.dpc.config import DPCConfig
from DPC.dpc.method import DPCMethod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_SOURCES), required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol", default="few_shot_all_classes")
    parser.add_argument("--data-root", default=None, help="Defaults to ./data or FSL_CLIP_DATA.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--backbone-epochs", type=int, default=20)
    parser.add_argument("--dpc-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--n-ctx-text", type=int, default=4)
    parser.add_argument("--ctx-init", default="a photo of a")
    parser.add_argument("--stack-weight", type=float, default=0.2)
    parser.add_argument("--hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--retain-weight", type=float, default=1.0)
    parser.add_argument("--prompt-proximity-weight", type=float, default=0.1)
    parser.add_argument("--hard-negative-topk", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32", "amp"), default="fp32")
    parser.add_argument("--annotation-path", default=None)
    parser.add_argument("--backbone-checkpoint", default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--no-log", action="store_true", help="Run training/evaluation without appending a result JSONL row.")
    parser.add_argument("--run-name", default="vit_b32_256_few_shot_all_classes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DPCConfig(
        backbone_epochs=args.backbone_epochs,
        dpc_epochs=args.dpc_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_ctx_text=args.n_ctx_text,
        ctx_init=args.ctx_init,
        stack_weight=args.stack_weight,
        hard_negative_weight=args.hard_negative_weight,
        retain_weight=args.retain_weight,
        prompt_proximity_weight=args.prompt_proximity_weight,
        hard_negative_topk=args.hard_negative_topk,
        precision=args.precision,
        annotation_path=args.annotation_path,
        backbone_checkpoint=args.backbone_checkpoint,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        show_progress=not args.no_progress,
        seed=args.seed,
    )

    bundle = build_openclip_bundle(device=args.device, precision="fp32")
    split_datasets = build_split_datasets(
        dataset=args.dataset,
        protocol=args.protocol,
        shots=args.shots,
        seed=args.seed,
        train_transform=bundle.preprocess_train,
        eval_transform=bundle.preprocess_eval,
        data_root=args.data_root,
    )
    train_loaders = build_data_loaders(
        split_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_shuffle=True,
    )
    eval_loaders = build_data_loaders(
        split_datasets,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        train_shuffle=False,
    )

    method = DPCMethod(config=config)
    artifact = method.fit(
        train_loaders["train"],
        eval_loaders["val"],
        split_datasets.classnames,
        bundle,
    )
    val_metrics = method.evaluate(
        artifact,
        eval_loaders["val"],
        split_datasets.classnames,
        bundle,
        metric_prefix="val",
    )
    test_metrics = method.evaluate(
        artifact,
        eval_loaders["test"],
        split_datasets.classnames,
        bundle,
        metric_prefix="test",
    )
    metrics = {**val_metrics, **test_metrics}

    payload = {"metrics": metrics}
    if not args.no_log:
        output = append_result(
            RunResult(
                method=method.method_name,
                dataset=args.dataset,
                protocol=args.protocol,
                model_name=bundle.model_name,
                pretrained=bundle.pretrained,
                shots=args.shots,
                seed=args.seed,
                metrics=metrics,
                split_path=str(split_datasets.split_file),
                notes="DPC OpenCLIP port; PromptSRC-style text backbone plus parallel hard-negative prompt.",
                extra={
                    "config": config.to_dict(),
                    "artifact_metadata": artifact.metadata,
                    "source_repo": "https://github.com/JREion/DPC",
                },
            ),
            result_jsonl_path(args.run_name),
        )
        payload["result_path"] = str(output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
