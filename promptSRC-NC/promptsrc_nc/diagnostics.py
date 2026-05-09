"""Unlabeled neighbor diagnostics for PromptSRC-NC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .config import PromptSRCNCConfig, neighbor_dir
from .eval import checkpoint_for_ref, load_model_for_checkpoint
from .losses import entropy_from_logits, js_divergence_from_logits
from .pair_dataset import _read_jsonl
from .structured_logging import append_jsonl, read_json, write_json
from .train import _device_type, _use_amp


def run_diagnostics(
    config: PromptSRCNCConfig,
    data_root: str | Path,
    run_root: str | Path,
    checkpoint_path: str | Path,
    pair_dir: str | Path | None = None,
) -> dict[str, Any]:
    import torch
    from PIL import Image

    pair_dir = Path(pair_dir or neighbor_dir(run_root, config.run_id, config.dataset, config.shots, config.seed))
    model, _loaders, checkpoint = load_model_for_checkpoint(config, data_root, checkpoint_path)
    model.eval()
    items = _read_jsonl(pair_dir / "unlabeled_items.jsonl")
    pair_payload = torch.load(pair_dir / "real_pairs.pt", map_location="cpu")
    pairs = pair_payload["pairs"].long()
    metadata_path = pair_dir / "metadata.json"
    neighbor_metadata = read_json(metadata_path) if metadata_path.exists() else {}
    logits_chunks = []
    labels = []
    device = model.device_name
    transform = model.bundle.eval_preprocess
    with torch.no_grad():
        for start in range(0, len(items), config.eval_batch_size):
            batch_items = items[start : start + config.eval_batch_size]
            images = torch.stack([transform(Image.open(item["impath"]).convert("RGB")) for item in batch_items]).to(device)
            with torch.amp.autocast(_device_type(device), enabled=_use_amp(config, device)):
                logits = model.forward_logits(images)
            logits_chunks.append(logits.detach().cpu())
            labels.extend(int(item.get("label", -1)) for item in batch_items)
    logits_all = torch.cat(logits_chunks, dim=0)
    logits_i = logits_all[pairs[:, 0]]
    logits_j = logits_all[pairs[:, 1]]
    preds_i = logits_i.argmax(dim=-1)
    preds_j = logits_j.argmax(dim=-1)
    entropy = entropy_from_logits(logits_all)
    probs = torch.softmax(logits_all, dim=-1)
    confidence = probs.max(dim=-1).values
    label_disagree = None
    if labels and min(labels) >= 0:
        label_tensor = torch.tensor(labels)
        label_disagree = float((label_tensor[pairs[:, 0]] != label_tensor[pairs[:, 1]]).float().mean().item())
    method = checkpoint.get("method", "PromptSRC")
    pair_mode = checkpoint.get("config", {}).get("pair_mode")
    if method == "PromptSRC":
        pair_mode = "none"
    record = {
        "event": "neighbor_diagnostics",
        "run_id": config.run_id,
        "method": method,
        "pair_mode": pair_mode,
        "dataset": config.dataset,
        "shots": config.shots,
        "seed": config.seed,
        "backbone": config.backbone,
        "checkpoint": str(checkpoint_path),
        "num_unlabeled": len(items),
        "num_real_pairs": int(pairs.shape[0]),
        "neighbor_k": pair_payload.get("neighbor_k"),
        "edge_disagreement": float((preds_i != preds_j).float().mean().item()),
        "mean_js": float(js_divergence_from_logits(logits_i, logits_j, reduction="mean").detach().cpu()),
        "mean_entropy": float(entropy.mean().item()),
        "mean_confidence": float(confidence.mean().item()),
        "label_edge_disagreement_diagnostic_only": label_disagree,
        "mean_real_cosine": neighbor_metadata.get("mean_real_cosine"),
        "mean_shuffled_cosine": neighbor_metadata.get("mean_shuffled_cosine"),
    }
    output_dir = Path(run_root) / config.run_id / "diagnostics" / config.dataset / f"shot{config.shots}" / f"seed{config.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{method.lower().replace('-', '_')}_{pair_mode}".replace(" ", "_")
    write_json(output_dir / f"{suffix}.json", record)
    append_jsonl(Path(run_root) / config.run_id / "logs" / "diagnostics.jsonl", record)
    return record


def config_from_args(args: Any) -> PromptSRCNCConfig:
    return PromptSRCNCConfig(
        dataset=args.dataset,
        shots=args.shots,
        seed=args.seed,
        backbone=args.backbone,
        pretrained=args.pretrained,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        precision=args.precision,
        run_id=args.run_id,
    )


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compute PromptSRC-NC neighbor diagnostics.")
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
    parser.add_argument("--pair-dir", default="")
    parser.add_argument("--eval-batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", choices=["fp32", "fp16", "amp"], default="amp")
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
    print(json.dumps(run_diagnostics(config, args.data_root, args.run_root, checkpoint, args.pair_dir or None), indent=2))


if __name__ == "__main__":
    main()

