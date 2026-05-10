"""Evaluation entrypoints for PromptSRC and PromptSRC-NC checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .config import PromptSRCNCConfig, stage1_dir, stage2_dir
from .data import build_data_loaders, load_split_records
from .model import PromptSRCModel, build_openclip_bundle
from .provenance import checkpoint_provenance, validate_checkpoint_for_config
from .structured_logging import append_jsonl, write_json
from .train import device_name, evaluate_loader


def checkpoint_for_ref(
    run_root: str | Path,
    run_id: str,
    dataset: str,
    shots: int,
    seed: int,
    backbone: str,
    checkpoint_ref: str,
) -> Path:
    if checkpoint_ref in {"stage1", "promptsrc", "PromptSRC"}:
        return stage1_dir(run_root, run_id, dataset, shots, seed, backbone) / "checkpoints" / "gpa.pt"
    if checkpoint_ref in {"stage2-real", "real", "promptsrc-nc-real"}:
        return stage2_dir(run_root, run_id, dataset, shots, seed, backbone, "real") / "checkpoints" / "final.pt"
    if checkpoint_ref in {"stage2-shuffled", "shuffled", "promptsrc-nc-shuffled"}:
        return stage2_dir(run_root, run_id, dataset, shots, seed, backbone, "shuffled") / "checkpoints" / "final.pt"
    return Path(checkpoint_ref)


def load_model_for_checkpoint(config: PromptSRCNCConfig, data_root: str | Path, checkpoint_path: str | Path):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_records, val_records, test_records, unlabeled_records, _split, classnames = load_split_records(
        data_root,
        config.dataset,
        config.shots,
        config.seed,
        config.protocol,
    )
    validate_checkpoint_for_config(checkpoint, config, artifact=f"checkpoint {checkpoint_path}", split=_split)
    bundle = build_openclip_bundle(config.backbone, config.pretrained, device=device_name(), precision=config.precision)
    loaders = build_data_loaders(
        train_records,
        val_records,
        test_records,
        unlabeled_records,
        bundle.train_preprocess,
        bundle.eval_preprocess,
        config.batch_size,
        config.eval_batch_size,
        config.num_workers,
    )
    model = PromptSRCModel(bundle, classnames, config)
    state = checkpoint.get("prompt_state") or checkpoint.get("gpa_state") or checkpoint["model_state"]
    model.load_prompt_state_dict(state)
    return model, loaders, checkpoint


def evaluate_checkpoint(
    config: PromptSRCNCConfig,
    data_root: str | Path,
    run_root: str | Path,
    checkpoint_path: str | Path,
    split: str = "test",
) -> dict[str, Any]:
    model, loaders, checkpoint = load_model_for_checkpoint(config, data_root, checkpoint_path)
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    metrics = evaluate_loader(model, loaders[split], model.device_name, config, max_batches=config.max_eval_batches)
    method = checkpoint.get("method", "PromptSRC")
    pair_mode = checkpoint.get("config", {}).get("pair_mode")
    if method == "PromptSRC":
        pair_mode = "none"
    record = {
        "event": "eval_summary",
        "run_id": config.run_id,
        "method": method,
        "pair_mode": pair_mode,
        "dataset": config.dataset,
        "shots": config.shots,
        "seed": config.seed,
        "backbone": config.backbone,
        "split": split,
        "top1_accuracy": metrics["top1_accuracy"],
        "macro_accuracy": metrics["macro_accuracy"],
        "num_examples": metrics["num_examples"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_provenance": checkpoint_provenance(checkpoint),
    }
    output_dir = Path(run_root) / config.run_id / "eval" / config.dataset / f"shot{config.shots}" / f"seed{config.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{method.lower().replace('-', '_')}_{pair_mode}_{split}".replace(" ", "_")
    write_json(output_dir / f"{suffix}.json", record)
    append_jsonl(Path(run_root) / config.run_id / "logs" / "eval_summary.jsonl", record)
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
        num_workers=args.num_workers,
        precision=args.precision,
        max_eval_batches=args.max_eval_batches,
        run_id=args.run_id,
    )


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate a PromptSRC-NC checkpoint.")
    parser.add_argument("--data-root", default="/vol/data")
    parser.add_argument("--run-root", default="/vol/runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backbone", default="ViT-B-16")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoint-ref", default="stage1")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", choices=["fp32", "fp16", "amp"], default="amp")
    parser.add_argument("--max-eval-batches", type=int, default=None)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    checkpoint = args.checkpoint or checkpoint_for_ref(
        args.run_root,
        args.run_id,
        config.dataset,
        config.shots,
        config.seed,
        config.backbone,
        args.checkpoint_ref,
    )
    print(json.dumps(evaluate_checkpoint(config, args.data_root, args.run_root, checkpoint, args.split), indent=2))


if __name__ == "__main__":
    main()
