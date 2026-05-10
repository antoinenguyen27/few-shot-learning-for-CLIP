"""GPU step-time and cost profiling for PromptSRC-NC."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from .config import MODAL_GPU_PRICE_PER_SECOND, PromptSRCNCConfig, neighbor_dir, stage1_dir
from .data import load_split_records
from .losses import js_divergence_from_logits, promptsrc_loss
from .pair_dataset import build_pair_loader
from .neighbors import validate_neighbor_artifacts
from .provenance import validate_checkpoint_for_config
from .structured_logging import append_jsonl, emit_status, runtime_record
from .train import (
    _device_type,
    _load_stage1_checkpoint,
    _move_batch,
    _use_amp,
    build_prompt_optimizer,
    build_model_and_loaders,
    device_name,
    ensure_finite_gradients,
    set_seed,
)


def _gpu_name(default: str) -> str:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).upper()
            if "L4" in name:
                return "L4"
            if "T4" in name:
                return "T4"
    except Exception:
        pass
    return default


def _price_for_profile(requested_gpu_label: str, actual_gpu: str) -> float:
    return MODAL_GPU_PRICE_PER_SECOND.get(actual_gpu, MODAL_GPU_PRICE_PER_SECOND.get(requested_gpu_label, 0.0))


def _sync_if_cuda(device: str) -> None:
    import torch

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def profile_gpu_cost(
    config: PromptSRCNCConfig,
    data_root: str | Path,
    run_root: str | Path,
    gpu_label: str = "T4",
    stage: str = "stage1",
    warmup_steps: int = 10,
    timed_steps: int = 100,
    init_checkpoint: str | Path | None = None,
    pair_artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    import torch

    if stage not in {"stage1", "stage2"}:
        raise ValueError("stage must be stage1 or stage2")
    set_seed(config.seed)
    emit_status(
        "gpu_profile_start",
        run_id=config.run_id,
        stage=stage,
        dataset=config.dataset,
        shots=config.shots,
        seed=config.seed,
        backbone=config.backbone,
        requested_gpu_label=gpu_label,
        warmup_steps=warmup_steps,
        timed_steps=timed_steps,
        batch_size=config.batch_size,
        pair_batch_size=config.pair_batch_size if stage == "stage2" else None,
    )
    model_load_start = time.perf_counter()
    model, loaders, split, _classnames, bundle = build_model_and_loaders(config, data_root)
    model_load_seconds = time.perf_counter() - model_load_start
    device = model.device_name
    emit_status(
        "gpu_profile_model_ready",
        run_id=config.run_id,
        stage=stage,
        dataset=config.dataset,
        shots=config.shots,
        seed=config.seed,
        backbone=config.backbone,
        device=device,
        openclip_model_name=bundle.model_name,
        model_load_seconds=model_load_seconds,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    lr = config.stage1_lr if stage == "stage1" else config.stage2_lr
    optimizer = build_prompt_optimizer(params, config, lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=_use_amp(config, device))
    pair_iter = None
    if stage == "stage2":
        init_checkpoint = init_checkpoint or stage1_dir(
            run_root,
            config.run_id,
            config.dataset,
            config.shots,
            config.seed,
            config.backbone,
        ) / "checkpoints" / "gpa.pt"
        checkpoint = _load_stage1_checkpoint(init_checkpoint, config)
        validate_checkpoint_for_config(
            checkpoint,
            config,
            artifact=f"Stage 1 checkpoint {init_checkpoint}",
            expected_method="PromptSRC",
            expected_stage="stage1",
            expected_role="stage1_gpa",
            split=split,
        )
        model.load_prompt_state_dict(checkpoint["prompt_state"])
        pair_dir = Path(pair_artifact_dir or neighbor_dir(run_root, config.run_id, config.dataset, config.shots, config.seed))
        _train_records, _val_records, _test_records, active_unlabeled_records, _split_check, _classnames_check = load_split_records(
            data_root,
            config.dataset,
            config.shots,
            config.seed,
            config.protocol,
        )
        validate_neighbor_artifacts(
            pair_dir,
            config,
            split,
            unlabeled_records=active_unlabeled_records[: config.max_unlabeled_images] if config.max_unlabeled_images else active_unlabeled_records,
        )
        pair_loader = build_pair_loader(pair_dir, config.pair_mode, bundle.train_preprocess, config.pair_batch_size, config.num_workers)
        pair_iter = iter(pair_loader)

    total_needed = warmup_steps + timed_steps
    train_iter = iter(loaders["train"])
    timed_elapsed = 0.0
    timed_examples = 0
    completed_timed = 0
    model.train()
    for step in range(1, total_needed + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loaders["train"])
            batch = next(train_iter)
        images, labels = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        _sync_if_cuda(device)
        step_start = time.perf_counter()
        with torch.amp.autocast(_device_type(device), enabled=_use_amp(config, device)):
            outputs = model(images)
            loss, _parts = promptsrc_loss(outputs, labels, config)
            if stage == "stage2":
                assert pair_iter is not None
                try:
                    pair_batch = next(pair_iter)
                except StopIteration:
                    pair_loader = build_pair_loader(
                        Path(pair_artifact_dir or neighbor_dir(run_root, config.run_id, config.dataset, config.shots, config.seed)),
                        config.pair_mode,
                        bundle.train_preprocess,
                        config.pair_batch_size,
                        config.num_workers,
                    )
                    pair_iter = iter(pair_loader)
                    pair_batch = next(pair_iter)
                img_i = pair_batch["img_i"].to(device)
                img_j = pair_batch["img_j"].to(device)
                loss = loss + config.lambda_nc_max * js_divergence_from_logits(model.forward_logits(img_i), model.forward_logits(img_j))
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            ensure_finite_gradients(params)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            ensure_finite_gradients(params)
            optimizer.step()
        _sync_if_cuda(device)
        if step > warmup_steps:
            elapsed = time.perf_counter() - step_start
            timed_elapsed += elapsed
            timed_examples += int(labels.shape[0])
            completed_timed += 1
        elif step == warmup_steps:
            emit_status(
                "gpu_profile_warmup_complete",
                run_id=config.run_id,
                stage=stage,
                dataset=config.dataset,
                shots=config.shots,
                seed=config.seed,
                warmup_steps=warmup_steps,
            )
    actual_gpu = _gpu_name(gpu_label)
    price = _price_for_profile(gpu_label, actual_gpu)
    seconds_per_step = timed_elapsed / max(completed_timed, 1)
    record = {
        "event": "gpu_profile",
        "run_id": config.run_id,
        "gpu": actual_gpu,
        "requested_gpu_label": gpu_label,
        "actual_gpu_name": actual_gpu,
        "modal_price_per_second": price,
        "dataset": config.dataset,
        "shots": config.shots,
        "seed": config.seed,
        "backbone": config.backbone,
        "stage": stage,
        "batch_size": config.batch_size,
        "pair_batch_size": config.pair_batch_size if stage == "stage2" else None,
        "warmup_steps": warmup_steps,
        "timed_steps": completed_timed,
        "model_load_seconds": model_load_seconds,
        "seconds_total": timed_elapsed,
        "seconds_per_step": seconds_per_step,
        "images_per_second": timed_examples / max(timed_elapsed, 1e-8),
        "cost_per_1000_steps_usd": seconds_per_step * 1000 * price,
        **runtime_record(),
    }
    append_jsonl(Path(run_root) / config.run_id / "logs" / "cost_profile.jsonl", record)
    emit_status("gpu_profile_complete", **{key: value for key, value in record.items() if key != "event"})
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
        pair_mode=args.pair_mode,
        max_unlabeled_images=args.max_unlabeled_images,
        run_id=args.run_id,
    )


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Profile PromptSRC-NC GPU cost per step.")
    parser.add_argument("--data-root", default="/vol/data")
    parser.add_argument("--run-root", default="/vol/runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", default="stanford_cars")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backbone", default="ViT-B-16")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--stage", choices=["stage1", "stage2"], default="stage1")
    parser.add_argument("--gpu-label", default="T4")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--timed-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=100)
    parser.add_argument("--pair-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", choices=["fp32", "fp16", "amp"], default="fp32")
    parser.add_argument("--pair-mode", choices=["real", "shuffled"], default="real")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--neighbor-dir", default="")
    parser.add_argument("--max-unlabeled-images", type=int, default=None)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            profile_gpu_cost(
                config_from_args(args),
                args.data_root,
                args.run_root,
                gpu_label=args.gpu_label,
                stage=args.stage,
                warmup_steps=args.warmup_steps,
                timed_steps=args.timed_steps,
                init_checkpoint=args.init_checkpoint or None,
                pair_artifact_dir=args.neighbor_dir or None,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
