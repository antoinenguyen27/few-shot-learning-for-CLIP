"""Stage 1 PromptSRC and Stage 2 PromptSRC-NC training."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from .config import (
    PromptSRCNCConfig,
    gaussian_epoch_weights,
    neighbor_dir,
    stage1_dir,
    stage2_dir,
)
from .data import build_data_loaders, load_split_records
from .losses import js_divergence_from_logits, lambda_nc_for_progress, promptsrc_loss
from .model import PromptSRCModel, build_openclip_bundle, trainable_parameters
from .pair_dataset import build_pair_loader
from .structured_logging import JsonlLogger, append_jsonl, runtime_record, write_json


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_name() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _device_type(device: str) -> str:
    return "cuda" if str(device).startswith("cuda") else "cpu"


def _use_amp(config: PromptSRCNCConfig, device: str) -> bool:
    import torch

    return config.precision == "amp" and _device_type(device) == "cuda" and torch.cuda.is_available()


def _move_batch(batch: dict[str, Any], device: str):
    images = batch.get("image", batch.get("img")).to(device)
    labels = batch["label"]
    if not hasattr(labels, "to"):
        import torch

        labels = torch.as_tensor(labels)
    return images, labels.to(device).long()


def _build_scheduler(optimizer: Any, lr: float, epochs: int, warmup_epochs: int, warmup_cons_lr: float):
    import math
    import torch

    def lr_lambda(epoch_index: int) -> float:
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return max(warmup_cons_lr / lr, 1e-8)
        denom = max(epochs - warmup_epochs, 1)
        progress = (epoch_index - warmup_epochs + 1) / denom
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _state_weight(state: dict[str, Any], weight: float) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, dict):
            out[key] = _state_weight(value, weight)
        elif torch.is_tensor(value):
            out[key] = value.detach().cpu().clone() * weight
        else:
            out[key] = value
    return out


def _state_add(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    for key in left:
        if isinstance(left[key], dict):
            out[key] = _state_add(left[key], right[key])
        elif torch.is_tensor(left[key]):
            out[key] = left[key] + right[key]
        else:
            out[key] = left[key]
    return out


def _grad_norm(parameters: list[Any]) -> float:
    import math
    import torch

    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        norm = parameter.grad.detach().data.norm(2).item()
        total += norm * norm
    return float(math.sqrt(total))


def evaluate_loader(model: Any, loader: Any, device: str, config: PromptSRCNCConfig, max_batches: int | None = None) -> dict[str, float]:
    import torch

    model.eval()
    total = 0
    correct = 0
    per_class_total: dict[int, int] = {}
    per_class_correct: dict[int, int] = {}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images, labels = _move_batch(batch, device)
            with torch.amp.autocast(_device_type(device), enabled=_use_amp(config, device)):
                logits = model.forward_logits(images)
            preds = logits.argmax(dim=-1)
            total += int(labels.numel())
            correct += int((preds == labels).sum().item())
            for target, pred in zip(labels.detach().cpu().tolist(), preds.detach().cpu().tolist(), strict=True):
                per_class_total[target] = per_class_total.get(target, 0) + 1
                per_class_correct[target] = per_class_correct.get(target, 0) + int(target == pred)
            if max_batches is not None and batch_index >= max_batches:
                break
    if total == 0:
        raise ValueError("Cannot evaluate on an empty loader")
    class_acc = [
        per_class_correct.get(label, 0) / count
        for label, count in sorted(per_class_total.items())
        if count > 0
    ]
    return {
        "top1_accuracy": correct / total,
        "macro_accuracy": sum(class_acc) / max(len(class_acc), 1),
        "num_examples": total,
    }


def _save_checkpoint(
    path: str | Path,
    *,
    method: str,
    stage: str,
    epoch: int,
    config: PromptSRCNCConfig,
    model: Any,
    optimizer: Any | None,
    prompt_state: dict[str, Any] | None = None,
    gpa_state: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    import torch

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    prompt_state = prompt_state or model.prompt_state_dict()
    payload = {
        "method": method,
        "stage": stage,
        "epoch": epoch,
        "backbone": config.backbone,
        "pretrained": config.pretrained,
        "dataset": config.dataset,
        "shots": config.shots,
        "seed": config.seed,
        "protocol": config.protocol,
        "model_state": prompt_state,
        "prompt_state": prompt_state,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "gpa_state": gpa_state,
        "config": config.to_dict(),
        "metrics": metrics or {},
        "metadata": {
            "checkpoint_format": "prompt_only_openclip_weights_resolved_from_backbone_and_pretrained",
            **(metadata or {}),
        },
    }
    torch.save(payload, output)
    return output


def build_model_and_loaders(config: PromptSRCNCConfig, data_root: str | Path):
    train_records, val_records, test_records, unlabeled_records, split, classnames = load_split_records(
        data_root,
        config.dataset,
        config.shots,
        config.seed,
        config.protocol,
    )
    bundle = build_openclip_bundle(config.backbone, config.pretrained, device=device_name(), precision=config.precision)
    loaders = build_data_loaders(
        train_records,
        val_records,
        test_records,
        unlabeled_records,
        bundle.train_preprocess,
        bundle.eval_preprocess,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
    )
    model = PromptSRCModel(bundle, classnames, config)
    return model, loaders, split, classnames, bundle


def train_stage1(config: PromptSRCNCConfig, data_root: str | Path, run_root: str | Path) -> Path:
    import torch

    set_seed(config.seed)
    output_dir = stage1_dir(run_root, config.run_id, config.dataset, config.shots, config.seed, config.backbone)
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config.to_dict())
    logger = JsonlLogger(
        log_dir / "train.jsonl",
        base={
            "run_id": config.run_id,
            "stage": "stage1",
            "method": "PromptSRC",
            "dataset": config.dataset,
            "shots": config.shots,
            "seed": config.seed,
            "backbone": config.backbone,
        },
    )
    runtime_logger = JsonlLogger(log_dir / "runtime.jsonl", base={"run_id": config.run_id, "function": "train_stage1"})
    model, loaders, split, _classnames, _bundle = build_model_and_loaders(config, data_root)
    device = model.device_name
    params = trainable_parameters(model)
    optimizer = torch.optim.SGD(params, lr=config.stage1_lr, weight_decay=config.weight_decay)
    scheduler = _build_scheduler(optimizer, config.stage1_lr, config.stage1_epochs, config.warmup_epochs, config.warmup_cons_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=_use_amp(config, device))
    gpa_weights = gaussian_epoch_weights(config.stage1_epochs, config.gpa_mean, config.gpa_std)
    gpa_state: dict[str, Any] | None = None
    best_val = -1.0
    best_state = model.prompt_state_dict()
    global_step = 0
    start = time.perf_counter()
    for epoch in range(1, config.stage1_epochs + 1):
        model.train()
        epoch_examples = 0
        epoch_loss = 0.0
        num_batches = len(loaders["train"])
        for batch_index, batch in enumerate(loaders["train"], start=1):
            step_start = time.perf_counter()
            images, labels = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(_device_type(device), enabled=_use_amp(config, device)):
                outputs = model(images)
                loss, parts = promptsrc_loss(outputs, labels, config)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = _grad_norm(params)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = _grad_norm(params)
                optimizer.step()
            global_step += 1
            batch_size = int(labels.shape[0])
            epoch_examples += batch_size
            epoch_loss += float(loss.detach().cpu()) * batch_size
            seconds_per_step = time.perf_counter() - step_start
            if global_step % config.log_interval == 0 or batch_index == num_batches:
                logger.log(
                    "train_step",
                    epoch=epoch,
                    step=batch_index,
                    global_step=global_step,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    loss_total=float(loss.detach().cpu()),
                    grad_norm_prompt=grad_norm,
                    batch_size=config.batch_size,
                    seconds_per_step=seconds_per_step,
                    **parts,
                )
                runtime_logger.log("runtime_step", epoch=epoch, step=batch_index, global_step=global_step, **runtime_record())
            if config.max_train_batches is not None and batch_index >= config.max_train_batches:
                break
        scheduler.step()
        val_metrics = evaluate_loader(
            model,
            loaders["val"],
            device,
            config,
            max_batches=config.max_eval_batches,
        )
        avg_loss = epoch_loss / max(epoch_examples, 1)
        current_state = model.prompt_state_dict()
        weighted = _state_weight(current_state, gpa_weights[epoch - 1])
        gpa_state = weighted if gpa_state is None else _state_add(gpa_state, weighted)
        if val_metrics["top1_accuracy"] >= best_val:
            best_val = float(val_metrics["top1_accuracy"])
            best_state = current_state
        logger.log(
            "epoch_end",
            epoch=epoch,
            train_loss=avg_loss,
            val_top1=val_metrics["top1_accuracy"],
            val_macro=val_metrics["macro_accuracy"],
            val_num_examples=val_metrics["num_examples"],
            elapsed_seconds=time.perf_counter() - start,
        )
        if config.save_every > 0 and epoch % config.save_every == 0:
            _save_checkpoint(
                ckpt_dir / f"epoch_{epoch:03d}.pt",
                method="PromptSRC",
                stage="stage1",
                epoch=epoch,
                config=config,
                model=model,
                optimizer=optimizer,
                prompt_state=current_state,
                gpa_state=gpa_state,
                metrics={"val": val_metrics, "train_loss": avg_loss},
                metadata={"split": split.to_dict()},
            )
        if config.max_train_batches is not None:
            break
    final_state = model.prompt_state_dict()
    if gpa_state is not None:
        model.load_prompt_state_dict(gpa_state)
    gpa_metrics = evaluate_loader(model, loaders["val"], device, config, max_batches=config.max_eval_batches)
    final_path = _save_checkpoint(
        ckpt_dir / "final.pt",
        method="PromptSRC",
        stage="stage1",
        epoch=epoch,
        config=config,
        model=model,
        optimizer=optimizer,
        prompt_state=final_state,
        gpa_state=gpa_state,
        metrics={"best_val_top1": best_val, "val": val_metrics},
        metadata={"split": split.to_dict(), "gpa_loaded_for_final_inference": False},
    )
    gpa_path = _save_checkpoint(
        ckpt_dir / "gpa.pt",
        method="PromptSRC",
        stage="stage1",
        epoch=epoch,
        config=config,
        model=model,
        optimizer=optimizer,
        prompt_state=gpa_state or best_state,
        gpa_state=gpa_state,
        metrics={"best_val_top1": best_val, "val": gpa_metrics},
        metadata={"split": split.to_dict(), "gpa_loaded_for_final_inference": True, "non_gpa_final_checkpoint": str(final_path)},
    )
    write_json(output_dir / "metrics_val.json", {"best_val_top1": best_val, "gpa_val": gpa_metrics})
    append_jsonl(
        Path(run_root) / config.run_id / "logs" / "train_stage1.jsonl",
        {
            "event": "stage1_complete",
            "run_id": config.run_id,
            "dataset": config.dataset,
            "shots": config.shots,
            "seed": config.seed,
            "backbone": config.backbone,
            "checkpoint": str(gpa_path),
            "val_top1": gpa_metrics["top1_accuracy"],
        },
    )
    return gpa_path


def _load_stage1_checkpoint(path: str | Path, config: PromptSRCNCConfig) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("method") != "PromptSRC" or checkpoint.get("stage") != "stage1":
        raise ValueError(f"Stage 2 must initialize from a PromptSRC stage1 checkpoint, got {path}")
    for key, expected in (("dataset", config.dataset), ("shots", config.shots), ("seed", config.seed), ("backbone", config.backbone)):
        if checkpoint.get(key) != expected:
            raise ValueError(f"Checkpoint {path} has {key}={checkpoint.get(key)!r}; expected {expected!r}")
    return checkpoint


def train_stage2(
    config: PromptSRCNCConfig,
    data_root: str | Path,
    run_root: str | Path,
    init_checkpoint: str | Path | None = None,
    pair_artifact_dir: str | Path | None = None,
) -> Path:
    import torch

    set_seed(config.seed)
    output_dir = stage2_dir(run_root, config.run_id, config.dataset, config.shots, config.seed, config.backbone, config.pair_mode)
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config.to_dict())
    init_checkpoint = init_checkpoint or stage1_dir(
        run_root,
        config.run_id,
        config.dataset,
        config.shots,
        config.seed,
        config.backbone,
    ) / "checkpoints" / "gpa.pt"
    pair_artifact_dir = Path(pair_artifact_dir or neighbor_dir(run_root, config.run_id, config.dataset, config.shots, config.seed))
    checkpoint = _load_stage1_checkpoint(init_checkpoint, config)

    logger = JsonlLogger(
        log_dir / "train.jsonl",
        base={
            "run_id": config.run_id,
            "stage": "stage2",
            "method": "PromptSRC-NC",
            "pair_mode": config.pair_mode,
            "dataset": config.dataset,
            "shots": config.shots,
            "seed": config.seed,
            "backbone": config.backbone,
        },
    )
    runtime_logger = JsonlLogger(log_dir / "runtime.jsonl", base={"run_id": config.run_id, "function": "train_stage2"})
    model, loaders, split, _classnames, bundle = build_model_and_loaders(config, data_root)
    model.load_prompt_state_dict(checkpoint["prompt_state"])
    device = model.device_name
    pair_loader = build_pair_loader(
        pair_artifact_dir,
        config.pair_mode,
        bundle.train_preprocess,
        config.pair_batch_size,
        config.num_workers,
    )
    pair_iter = iter(pair_loader)
    params = trainable_parameters(model)
    optimizer = torch.optim.SGD(params, lr=config.stage2_lr, weight_decay=config.weight_decay)
    scheduler = _build_scheduler(optimizer, config.stage2_lr, config.stage2_epochs, 0, config.warmup_cons_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=_use_amp(config, device))
    global_step = 0
    start = time.perf_counter()
    for epoch in range(1, config.stage2_epochs + 1):
        model.train()
        num_batches = len(loaders["train"])
        epoch_loss = 0.0
        epoch_examples = 0
        for batch_index, batch in enumerate(loaders["train"], start=1):
            step_start = time.perf_counter()
            images, labels = _move_batch(batch, device)
            try:
                pair_batch = next(pair_iter)
            except StopIteration:
                pair_iter = iter(pair_loader)
                pair_batch = next(pair_iter)
            img_i = pair_batch["img_i"].to(device)
            img_j = pair_batch["img_j"].to(device)
            epoch_progress = (epoch - 1) + (batch_index - 1) / max(num_batches, 1)
            lambda_nc = lambda_nc_for_progress(epoch_progress, config.lambda_nc_max, config.lambda_nc_warmup_epochs)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(_device_type(device), enabled=_use_amp(config, device)):
                outputs = model(images)
                loss_promptsrc, parts = promptsrc_loss(outputs, labels, config)
                logits_i = model.forward_logits(img_i)
                logits_j = model.forward_logits(img_j)
                loss_nc = js_divergence_from_logits(logits_i, logits_j)
                loss = loss_promptsrc + lambda_nc * loss_nc
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = _grad_norm(params)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = _grad_norm(params)
                optimizer.step()
            global_step += 1
            batch_size = int(labels.shape[0])
            epoch_examples += batch_size
            epoch_loss += float(loss.detach().cpu()) * batch_size
            seconds_per_step = time.perf_counter() - step_start
            if global_step % config.log_interval == 0 or batch_index == num_batches:
                logger.log(
                    "train_step",
                    epoch=epoch,
                    step=batch_index,
                    global_step=global_step,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    loss_total=float(loss.detach().cpu()),
                    loss_promptsrc=float(loss_promptsrc.detach().cpu()),
                    loss_nc=float(loss_nc.detach().cpu()),
                    lambda_nc=lambda_nc,
                    lambda_nc_max=config.lambda_nc_max,
                    lambda_nc_warmup_epochs=config.lambda_nc_warmup_epochs,
                    grad_norm_prompt=grad_norm,
                    batch_size=config.batch_size,
                    pair_batch_size=config.pair_batch_size,
                    seconds_per_step=seconds_per_step,
                    **parts,
                )
                runtime_logger.log("runtime_step", epoch=epoch, step=batch_index, global_step=global_step, **runtime_record())
            if config.max_train_batches is not None and batch_index >= config.max_train_batches:
                break
        scheduler.step()
        val_metrics = evaluate_loader(model, loaders["val"], device, config, max_batches=config.max_eval_batches)
        avg_loss = epoch_loss / max(epoch_examples, 1)
        logger.log(
            "epoch_end",
            epoch=epoch,
            train_loss=avg_loss,
            val_top1=val_metrics["top1_accuracy"],
            val_macro=val_metrics["macro_accuracy"],
            val_num_examples=val_metrics["num_examples"],
            elapsed_seconds=time.perf_counter() - start,
        )
        if config.save_every > 0 and epoch % config.save_every == 0:
            _save_checkpoint(
                ckpt_dir / f"epoch_{epoch:03d}.pt",
                method="PromptSRC-NC",
                stage="stage2",
                epoch=epoch,
                config=config,
                model=model,
                optimizer=optimizer,
                metrics={"val": val_metrics, "train_loss": avg_loss},
                metadata={
                    "split": split.to_dict(),
                    "init_checkpoint": str(init_checkpoint),
                    "neighbor_dir": str(pair_artifact_dir),
                    "pair_mode": config.pair_mode,
                    "use_stage2_gpa": False,
                },
            )
        if config.max_train_batches is not None:
            break
    final_metrics = evaluate_loader(model, loaders["val"], device, config, max_batches=config.max_eval_batches)
    final_path = _save_checkpoint(
        ckpt_dir / "final.pt",
        method="PromptSRC-NC",
        stage="stage2",
        epoch=epoch,
        config=config,
        model=model,
        optimizer=optimizer,
        metrics={"val": final_metrics},
        metadata={
            "split": split.to_dict(),
            "init_checkpoint": str(init_checkpoint),
            "neighbor_dir": str(pair_artifact_dir),
            "pair_mode": config.pair_mode,
            "use_stage2_gpa": False,
        },
    )
    write_json(output_dir / "metrics_val.json", final_metrics)
    append_jsonl(
        Path(run_root) / config.run_id / "logs" / "train_stage2.jsonl",
        {
            "event": "stage2_complete",
            "run_id": config.run_id,
            "dataset": config.dataset,
            "shots": config.shots,
            "seed": config.seed,
            "backbone": config.backbone,
            "pair_mode": config.pair_mode,
            "checkpoint": str(final_path),
            "val_top1": final_metrics["top1_accuracy"],
        },
    )
    return final_path


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
        stage1_epochs=args.epochs if args.stage == "stage1" else args.stage1_epochs,
        stage2_epochs=args.epochs if args.stage == "stage2" else args.stage2_epochs,
        stage1_lr=args.lr if args.stage == "stage1" else args.stage1_lr,
        stage2_lr=args.lr if args.stage == "stage2" else args.stage2_lr,
        precision=args.precision,
        pair_mode=args.pair_mode,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        log_interval=args.log_interval,
        save_every=args.save_every,
        run_id=args.run_id,
    )


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train PromptSRC or PromptSRC-NC.")
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--data-root", default="/vol/data")
    parser.add_argument("--run-root", default="/vol/runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backbone", default="ViT-B-16")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=100)
    parser.add_argument("--pair-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--stage1-epochs", type=int, default=50)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--stage1-lr", type=float, default=0.0025)
    parser.add_argument("--stage2-lr", type=float, default=0.00025)
    parser.add_argument("--precision", choices=["fp32", "fp16", "amp"], default="amp")
    parser.add_argument("--pair-mode", choices=["real", "shuffled"], default="real")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--neighbor-dir", default="")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args(argv)
    if args.epochs is None:
        args.epochs = args.stage1_epochs if args.stage == "stage1" else args.stage2_epochs
    if args.lr is None:
        args.lr = args.stage1_lr if args.stage == "stage1" else args.stage2_lr
    config = config_from_args(args)
    if args.stage == "stage1":
        print(train_stage1(config, args.data_root, args.run_root))
    else:
        print(
            train_stage2(
                config,
                args.data_root,
                args.run_root,
                init_checkpoint=args.init_checkpoint or None,
                pair_artifact_dir=args.neighbor_dir or None,
            )
        )


if __name__ == "__main__":
    main()
