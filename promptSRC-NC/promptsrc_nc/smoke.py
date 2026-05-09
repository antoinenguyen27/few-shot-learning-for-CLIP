"""Tiny end-to-end smoke test for the PromptSRC-NC pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .config import PromptSRCNCConfig, neighbor_dir, stage1_dir, stage2_dir
from .eval import evaluate_checkpoint
from .neighbors import build_neighbor_artifacts
from .structured_logging import append_jsonl, write_json
from .train import train_stage1, train_stage2


def run_smoke_test(
    config: PromptSRCNCConfig,
    data_root: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    smoke_config = PromptSRCNCConfig(
        **{
            **config.to_dict(),
            "shots": config.shots,
            "stage1_epochs": 1,
            "stage2_epochs": 1,
            "batch_size": min(config.batch_size, 4),
            "pair_batch_size": min(config.pair_batch_size, 4),
            "eval_batch_size": min(config.eval_batch_size, 32),
            "num_workers": 0,
            "max_train_batches": 1,
            "max_eval_batches": 1,
            "max_unlabeled_images": config.max_unlabeled_images or 256,
            "log_interval": 1,
            "save_every": 1,
        }
    )
    neighbor_path = build_neighbor_artifacts(smoke_config, data_root, run_root)
    stage1_ckpt = train_stage1(smoke_config, data_root, run_root)
    real_config = PromptSRCNCConfig(**{**smoke_config.to_dict(), "pair_mode": "real"})
    shuffled_config = PromptSRCNCConfig(**{**smoke_config.to_dict(), "pair_mode": "shuffled"})
    real_ckpt = train_stage2(real_config, data_root, run_root, init_checkpoint=stage1_ckpt, pair_artifact_dir=neighbor_path)
    shuffled_ckpt = train_stage2(shuffled_config, data_root, run_root, init_checkpoint=stage1_ckpt, pair_artifact_dir=neighbor_path)
    eval_record = evaluate_checkpoint(smoke_config, data_root, run_root, stage1_ckpt, split="val")
    record = {
        "event": "smoke_test_complete",
        "run_id": smoke_config.run_id,
        "dataset": smoke_config.dataset,
        "shots": smoke_config.shots,
        "seed": smoke_config.seed,
        "backbone": smoke_config.backbone,
        "neighbor_dir": str(neighbor_path),
        "stage1_checkpoint": str(stage1_ckpt),
        "stage2_real_checkpoint": str(real_ckpt),
        "stage2_shuffled_checkpoint": str(shuffled_ckpt),
        "eval_val_top1": eval_record["top1_accuracy"],
        "paths_checked": {
            "neighbors": str(neighbor_dir(run_root, smoke_config.run_id, smoke_config.dataset, smoke_config.shots, smoke_config.seed)),
            "stage1": str(stage1_dir(run_root, smoke_config.run_id, smoke_config.dataset, smoke_config.shots, smoke_config.seed, smoke_config.backbone)),
            "stage2_real": str(stage2_dir(run_root, smoke_config.run_id, smoke_config.dataset, smoke_config.shots, smoke_config.seed, smoke_config.backbone, "real")),
            "stage2_shuffled": str(stage2_dir(run_root, smoke_config.run_id, smoke_config.dataset, smoke_config.shots, smoke_config.seed, smoke_config.backbone, "shuffled")),
        },
    }
    out_dir = Path(run_root) / smoke_config.run_id / "smoke"
    write_json(out_dir / "smoke.json", record)
    append_jsonl(Path(run_root) / smoke_config.run_id / "logs" / "smoke.jsonl", record)
    return record


def config_from_args(args: Any) -> PromptSRCNCConfig:
    return PromptSRCNCConfig(
        dataset=args.dataset,
        shots=args.shots,
        seed=args.seed,
        backbone=args.backbone,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        pair_batch_size=args.pair_batch_size,
        num_workers=args.num_workers,
        precision=args.precision,
        max_unlabeled_images=args.max_unlabeled_images,
        run_id=args.run_id,
    )


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run a tiny PromptSRC-NC smoke path.")
    parser.add_argument("--data-root", default="/vol/data")
    parser.add_argument("--run-root", default="/vol/runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", default="eurosat")
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backbone", default="ViT-B-16")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--pair-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", choices=["fp32", "fp16", "amp"], default="amp")
    parser.add_argument("--max-unlabeled-images", type=int, default=256)
    args = parser.parse_args(argv)
    print(json.dumps(run_smoke_test(config_from_args(args), args.data_root, args.run_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

