"""Modal app for PromptSRC-NC.

Run from the repository root, for example:

    uv run modal run promptSRC-NC/modal_app.py::prepare_data
"""

from __future__ import annotations

from pathlib import Path

import modal


APP_NAME = "promptsrc-nc"
DATA_ROOT = "/vol/data"
RUN_ROOT = "/vol/runs"
WEIGHTS_ROOT = "/vol/weights"

app = modal.App(APP_NAME)

data_vol = modal.Volume.from_name("promptsrc-nc-data", create_if_missing=True)
weights_vol = modal.Volume.from_name("promptsrc-nc-weights", create_if_missing=True)
runs_vol = modal.Volume.from_name("promptsrc-nc-runs", create_if_missing=True)
KAGGLE_SECRETS = [modal.Secret.from_name("kaggle")]
HF_SECRET = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
HF_SECRETS = [HF_SECRET]

VOLUME_MOUNTS = {
    DATA_ROOT: data_vol,
    WEIGHTS_ROOT: weights_vol,
    RUN_ROOT: runs_vol,
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .uv_sync("promptSRC-NC")
    .workdir("/root")
    .env(
        {
            "HF_HOME": f"{WEIGHTS_ROOT}/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TORCH_HOME": f"{WEIGHTS_ROOT}/torch",
            "OPENCLIP_CACHE_DIR": f"{WEIGHTS_ROOT}/openclip",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir("promptSRC-NC/promptsrc_nc", remote_path="/root/promptsrc_nc")
)


def _config(
    *,
    run_id: str,
    dataset: str,
    shots: int,
    seed: int,
    backbone: str,
    pretrained: str = "openai",
    gpu: str = "T4",
    pair_mode: str = "real",
    batch_size: int = 4,
    pair_batch_size: int = 8,
    precision: str = "fp32",
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    max_unlabeled_images: int | None = None,
):
    from promptsrc_nc.config import PromptSRCNCConfig

    return PromptSRCNCConfig(
        run_id=run_id,
        dataset=dataset,
        shots=shots,
        seed=seed,
        backbone=backbone,
        pretrained=pretrained,
        pair_mode=pair_mode,
        batch_size=batch_size,
        pair_batch_size=pair_batch_size,
        precision=precision,
        max_train_batches=max_train_batches,
        max_eval_batches=max_eval_batches,
        max_unlabeled_images=max_unlabeled_images,
    )


def _gpu_function(gpu: str, *, t4, l4):
    gpu_name = gpu.upper()
    if gpu_name == "T4":
        return t4
    if gpu_name == "L4":
        return l4
    raise ValueError(f"Unsupported --gpu {gpu!r}; supported values are T4 and L4")


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _weights_cache_summary() -> dict[str, object]:
    hf_root = Path(WEIGHTS_ROOT) / "hf"
    torch_root = Path(WEIGHTS_ROOT) / "torch"
    openclip_root = Path(WEIGHTS_ROOT) / "openclip"
    hf_models = sorted(path.name for path in (hf_root / "hub").glob("models--*")) if (hf_root / "hub").exists() else []
    return {
        "hf_home": str(hf_root),
        "hf_cache_size_mb": round(_path_size_bytes(hf_root) / (1024**2), 3),
        "torch_cache_size_mb": round(_path_size_bytes(torch_root) / (1024**2), 3),
        "openclip_cache_size_mb": round(_path_size_bytes(openclip_root) / (1024**2), 3),
        "hf_models": hf_models,
    }


def _modal_status(event: str, **payload: object) -> None:
    from promptsrc_nc.structured_logging import emit_status

    emit_status(event, **payload)


@app.function(
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=KAGGLE_SECRETS,
    timeout=60 * 60 * 6,
)
def prepare_data(datasets: str = "eurosat,oxford_flowers,stanford_cars", shots: str = "1,16", seeds: str = "1,2,3"):
    from promptsrc_nc.data import parse_dataset_list, parse_int_list, prepare_datasets

    _modal_status("modal_action_start", action="prepare_data", datasets=datasets, shots=shots, seeds=seeds)
    result = prepare_datasets(
        DATA_ROOT,
        parse_dataset_list(datasets),
        shots=parse_int_list(shots),
        seeds=parse_int_list(seeds),
        download=True,
        log_path=Path(RUN_ROOT) / "data_preparation.jsonl",
    )
    data_vol.commit()
    runs_vol.commit()
    _modal_status("modal_action_complete", action="prepare_data", datasets=datasets, shots=shots, seeds=seeds, result=result)
    return result


@app.function(image=image, volumes=VOLUME_MOUNTS, secrets=HF_SECRETS, timeout=60 * 60)
def prepare_weights(backbone: str = "ViT-B-16", pretrained: str = "openai"):
    from promptsrc_nc.model import build_openclip_bundle
    from promptsrc_nc.structured_logging import emit_status

    weights_vol.reload()
    before = _weights_cache_summary()
    emit_status("prepare_weights_start", backbone=backbone, pretrained=pretrained, cache=before)
    bundle = build_openclip_bundle(backbone, pretrained, device="cpu", precision="fp32")
    after = _weights_cache_summary()
    result = {
        "event": "prepare_weights_complete",
        "backbone": backbone,
        "pretrained": pretrained,
        "openclip_model_name": bundle.model_name,
        "cache": after,
    }
    emit_status(**result)
    weights_vol.commit()
    return result


@app.function(image=image, volumes=VOLUME_MOUNTS, secrets=HF_SECRETS, gpu="T4", timeout=60 * 60 * 2)
def smoke_test(
    run_id: str,
    dataset: str = "eurosat",
    shots: int = 1,
    seed: int = 1,
    backbone: str = "ViT-B-16",
    pretrained: str = "openai",
):
    return _run_smoke_test(run_id, dataset, shots, seed, backbone, pretrained)


@app.function(image=image, volumes=VOLUME_MOUNTS, secrets=HF_SECRETS, gpu="L4", timeout=60 * 60 * 2)
def smoke_test_l4(
    run_id: str,
    dataset: str = "eurosat",
    shots: int = 1,
    seed: int = 1,
    backbone: str = "ViT-B-16",
    pretrained: str = "openai",
):
    return _run_smoke_test(run_id, dataset, shots, seed, backbone, pretrained)


def _run_smoke_test(
    run_id: str,
    dataset: str,
    shots: int,
    seed: int,
    backbone: str,
    pretrained: str,
):
    from promptsrc_nc.smoke import run_smoke_test

    _modal_status("modal_action_start", action="smoke_test", run_id=run_id, dataset=dataset, shots=shots, seed=seed, backbone=backbone)
    data_vol.reload()
    weights_vol.reload()
    runs_vol.reload()
    config = _config(
        run_id=run_id,
        dataset=dataset,
        shots=shots,
        seed=seed,
        backbone=backbone,
        pretrained=pretrained,
        pair_batch_size=4,
        max_train_batches=1,
        max_eval_batches=1,
        max_unlabeled_images=256,
    )
    result = run_smoke_test(config, DATA_ROOT, RUN_ROOT)
    weights_vol.commit()
    runs_vol.commit()
    _modal_status("modal_action_complete", action="smoke_test", run_id=run_id, dataset=dataset, shots=shots, seed=seed, result=result)
    return result


@app.function(image=image, volumes=VOLUME_MOUNTS, secrets=HF_SECRETS, gpu="T4", timeout=60 * 60 * 2)
def profile_gpu_cost(
    run_id: str,
    dataset: str = "stanford_cars",
    shots: int = 16,
    seed: int = 1,
    backbone: str = "ViT-B-16",
    pretrained: str = "openai",
    stage: str = "stage1",
    gpu_label: str = "T4",
    warmup_steps: int = 10,
    timed_steps: int = 100,
    pair_mode: str = "real",
    pair_batch_size: int = 8,
    max_unlabeled_images: int | None = None,
):
    return _run_profile_gpu_cost(
        run_id,
        dataset,
        shots,
        seed,
        backbone,
        pretrained,
        stage,
        gpu_label,
        warmup_steps,
        timed_steps,
        pair_mode,
        pair_batch_size,
        max_unlabeled_images,
    )


@app.function(image=image, volumes=VOLUME_MOUNTS, secrets=HF_SECRETS, gpu="L4", timeout=60 * 60 * 2)
def profile_gpu_cost_l4(
    run_id: str,
    dataset: str = "stanford_cars",
    shots: int = 16,
    seed: int = 1,
    backbone: str = "ViT-B-16",
    pretrained: str = "openai",
    stage: str = "stage1",
    gpu_label: str = "L4",
    warmup_steps: int = 10,
    timed_steps: int = 100,
    pair_mode: str = "real",
    pair_batch_size: int = 8,
    max_unlabeled_images: int | None = None,
):
    return _run_profile_gpu_cost(
        run_id,
        dataset,
        shots,
        seed,
        backbone,
        pretrained,
        stage,
        gpu_label,
        warmup_steps,
        timed_steps,
        pair_mode,
        pair_batch_size,
        max_unlabeled_images,
    )


def _run_profile_gpu_cost(
    run_id: str,
    dataset: str,
    shots: int,
    seed: int,
    backbone: str,
    pretrained: str,
    stage: str,
    gpu_label: str,
    warmup_steps: int,
    timed_steps: int,
    pair_mode: str,
    pair_batch_size: int,
    max_unlabeled_images: int | None,
):
    from promptsrc_nc.cost_profile import profile_gpu_cost as run_profile

    _modal_status("modal_action_start", action="profile_gpu_cost", run_id=run_id, dataset=dataset, shots=shots, seed=seed, stage=stage, gpu=gpu_label)
    data_vol.reload()
    weights_vol.reload()
    runs_vol.reload()
    config = _config(
        run_id=run_id,
        dataset=dataset,
        shots=shots,
        seed=seed,
        backbone=backbone,
        pretrained=pretrained,
        pair_mode=pair_mode,
        pair_batch_size=pair_batch_size,
        max_unlabeled_images=max_unlabeled_images,
    )
    result = run_profile(
        config,
        DATA_ROOT,
        RUN_ROOT,
        gpu_label=gpu_label,
        stage=stage,
        warmup_steps=warmup_steps,
        timed_steps=timed_steps,
    )
    weights_vol.commit()
    runs_vol.commit()
    _modal_status("modal_action_complete", action="profile_gpu_cost", run_id=run_id, dataset=dataset, shots=shots, seed=seed, stage=stage, result=result)
    return result


@app.function(image=image, volumes=VOLUME_MOUNTS, secrets=HF_SECRETS, gpu="T4", timeout=60 * 60 * 6)
def build_neighbors(
    run_id: str,
    dataset: str,
    shots: int = 16,
    seed: int = 1,
    backbone: str = "ViT-B-16",
    pretrained: str = "openai",
    max_unlabeled_images: int | None = None,
):
    return _run_build_neighbors(run_id, dataset, shots, seed, backbone, pretrained, max_unlabeled_images)


@app.function(image=image, volumes=VOLUME_MOUNTS, secrets=HF_SECRETS, gpu="L4", timeout=60 * 60 * 6)
def build_neighbors_l4(
    run_id: str,
    dataset: str,
    shots: int = 16,
    seed: int = 1,
    backbone: str = "ViT-B-16",
    pretrained: str = "openai",
    max_unlabeled_images: int | None = None,
):
    return _run_build_neighbors(run_id, dataset, shots, seed, backbone, pretrained, max_unlabeled_images)


def _run_build_neighbors(
    run_id: str,
    dataset: str,
    shots: int,
    seed: int,
    backbone: str,
    pretrained: str,
    max_unlabeled_images: int | None,
):
    from promptsrc_nc.neighbors import build_neighbor_artifacts

    _modal_status("modal_action_start", action="build_neighbors", run_id=run_id, dataset=dataset, shots=shots, seed=seed, backbone=backbone)
    data_vol.reload()
    weights_vol.reload()
    runs_vol.reload()
    config = _config(
        run_id=run_id,
        dataset=dataset,
        shots=shots,
        seed=seed,
        backbone=backbone,
        pretrained=pretrained,
        max_unlabeled_images=max_unlabeled_images,
    )
    result = str(build_neighbor_artifacts(config, DATA_ROOT, RUN_ROOT, log_path=Path(RUN_ROOT) / run_id / "logs" / "neighbors.jsonl"))
    weights_vol.commit()
    runs_vol.commit()
    _modal_status("modal_action_complete", action="build_neighbors", run_id=run_id, dataset=dataset, shots=shots, seed=seed, neighbor_dir=result)
    return result


@app.function(image=image, volumes=VOLUME_MOUNTS, secrets=HF_SECRETS, gpu="T4", timeout=60 * 60 * 8)
def train_stage1(
    run_id: str,
    dataset: str,
    shots: int = 16,
    seed: int = 1,
    backbone: str = "ViT-B-16",
    pretrained: str = "openai",
):
    return _run_train_stage1(run_id, dataset, shots, seed, backbone, pretrained)


@app.function(image=image, volumes=VOLUME_MOUNTS, secrets=HF_SECRETS, gpu="L4", timeout=60 * 60 * 8)
def train_stage1_l4(
    run_id: str,
    dataset: str,
    shots: int = 16,
    seed: int = 1,
    backbone: str = "ViT-B-16",
    pretrained: str = "openai",
):
    return _run_train_stage1(run_id, dataset, shots, seed, backbone, pretrained)


def _run_train_stage1(
    run_id: str,
    dataset: str,
    shots: int,
    seed: int,
    backbone: str,
    pretrained: str,
):
    from promptsrc_nc.train import train_stage1 as run_train_stage1

    _modal_status("modal_action_start", action="train_stage1", run_id=run_id, dataset=dataset, shots=shots, seed=seed, backbone=backbone)
    data_vol.reload()
    weights_vol.reload()
    runs_vol.reload()
    config = _config(run_id=run_id, dataset=dataset, shots=shots, seed=seed, backbone=backbone, pretrained=pretrained)
    result = str(run_train_stage1(config, DATA_ROOT, RUN_ROOT))
    weights_vol.commit()
    runs_vol.commit()
    _modal_status("modal_action_complete", action="train_stage1", run_id=run_id, dataset=dataset, shots=shots, seed=seed, checkpoint=result)
    return result


@app.function(image=image, volumes=VOLUME_MOUNTS, secrets=HF_SECRETS, gpu="T4", timeout=60 * 60 * 4)
def train_stage2(
    run_id: str,
    dataset: str,
    shots: int = 16,
    seed: int = 1,
    backbone: str = "ViT-B-16",
    pretrained: str = "openai",
    pair_mode: str = "real",
    pair_batch_size: int = 8,
    max_unlabeled_images: int | None = None,
):
    return _run_train_stage2(run_id, dataset, shots, seed, backbone, pretrained, pair_mode, pair_batch_size, max_unlabeled_images)


@app.function(image=image, volumes=VOLUME_MOUNTS, secrets=HF_SECRETS, gpu="L4", timeout=60 * 60 * 4)
def train_stage2_l4(
    run_id: str,
    dataset: str,
    shots: int = 16,
    seed: int = 1,
    backbone: str = "ViT-B-16",
    pretrained: str = "openai",
    pair_mode: str = "real",
    pair_batch_size: int = 8,
    max_unlabeled_images: int | None = None,
):
    return _run_train_stage2(run_id, dataset, shots, seed, backbone, pretrained, pair_mode, pair_batch_size, max_unlabeled_images)


def _run_train_stage2(
    run_id: str,
    dataset: str,
    shots: int,
    seed: int,
    backbone: str,
    pretrained: str,
    pair_mode: str,
    pair_batch_size: int,
    max_unlabeled_images: int | None,
):
    from promptsrc_nc.train import train_stage2 as run_train_stage2

    _modal_status("modal_action_start", action="train_stage2", run_id=run_id, dataset=dataset, shots=shots, seed=seed, backbone=backbone, pair_mode=pair_mode)
    data_vol.reload()
    weights_vol.reload()
    runs_vol.reload()
    config = _config(
        run_id=run_id,
        dataset=dataset,
        shots=shots,
        seed=seed,
        backbone=backbone,
        pretrained=pretrained,
        pair_mode=pair_mode,
        pair_batch_size=pair_batch_size,
        max_unlabeled_images=max_unlabeled_images,
    )
    result = str(run_train_stage2(config, DATA_ROOT, RUN_ROOT))
    weights_vol.commit()
    runs_vol.commit()
    _modal_status("modal_action_complete", action="train_stage2", run_id=run_id, dataset=dataset, shots=shots, seed=seed, pair_mode=pair_mode, checkpoint=result)
    return result


@app.function(image=image, volumes=VOLUME_MOUNTS, secrets=HF_SECRETS, gpu="T4", timeout=60 * 60 * 2)
def evaluate(
    run_id: str,
    dataset: str,
    shots: int = 16,
    seed: int = 1,
    backbone: str = "ViT-B-16",
    pretrained: str = "openai",
    checkpoint_ref: str = "stage1",
    split: str = "test",
):
    return _run_evaluate(run_id, dataset, shots, seed, backbone, pretrained, checkpoint_ref, split)


@app.function(image=image, volumes=VOLUME_MOUNTS, secrets=HF_SECRETS, gpu="L4", timeout=60 * 60 * 2)
def evaluate_l4(
    run_id: str,
    dataset: str,
    shots: int = 16,
    seed: int = 1,
    backbone: str = "ViT-B-16",
    pretrained: str = "openai",
    checkpoint_ref: str = "stage1",
    split: str = "test",
):
    return _run_evaluate(run_id, dataset, shots, seed, backbone, pretrained, checkpoint_ref, split)


def _run_evaluate(
    run_id: str,
    dataset: str,
    shots: int,
    seed: int,
    backbone: str,
    pretrained: str,
    checkpoint_ref: str,
    split: str,
):
    from promptsrc_nc.eval import checkpoint_for_ref, evaluate_checkpoint

    _modal_status("modal_action_start", action="evaluate", run_id=run_id, dataset=dataset, shots=shots, seed=seed, checkpoint_ref=checkpoint_ref, split=split)
    data_vol.reload()
    weights_vol.reload()
    runs_vol.reload()
    config = _config(run_id=run_id, dataset=dataset, shots=shots, seed=seed, backbone=backbone, pretrained=pretrained)
    checkpoint = checkpoint_for_ref(RUN_ROOT, run_id, config.dataset, shots, seed, config.backbone, checkpoint_ref)
    result = evaluate_checkpoint(config, DATA_ROOT, RUN_ROOT, checkpoint, split=split)
    weights_vol.commit()
    runs_vol.commit()
    _modal_status("modal_action_complete", action="evaluate", run_id=run_id, dataset=dataset, shots=shots, seed=seed, checkpoint_ref=checkpoint_ref, split=split, result=result)
    return result


@app.function(image=image, volumes=VOLUME_MOUNTS, secrets=HF_SECRETS, gpu="T4", timeout=60 * 60 * 2)
def diagnostics(
    run_id: str,
    dataset: str,
    shots: int = 16,
    seed: int = 1,
    backbone: str = "ViT-B-16",
    pretrained: str = "openai",
    checkpoint_ref: str = "stage1",
    max_unlabeled_images: int | None = None,
):
    return _run_diagnostics(run_id, dataset, shots, seed, backbone, pretrained, checkpoint_ref, max_unlabeled_images)


@app.function(image=image, volumes=VOLUME_MOUNTS, secrets=HF_SECRETS, gpu="L4", timeout=60 * 60 * 2)
def diagnostics_l4(
    run_id: str,
    dataset: str,
    shots: int = 16,
    seed: int = 1,
    backbone: str = "ViT-B-16",
    pretrained: str = "openai",
    checkpoint_ref: str = "stage1",
    max_unlabeled_images: int | None = None,
):
    return _run_diagnostics(run_id, dataset, shots, seed, backbone, pretrained, checkpoint_ref, max_unlabeled_images)


def _run_diagnostics(
    run_id: str,
    dataset: str,
    shots: int,
    seed: int,
    backbone: str,
    pretrained: str,
    checkpoint_ref: str,
    max_unlabeled_images: int | None,
):
    from promptsrc_nc.diagnostics import run_diagnostics
    from promptsrc_nc.eval import checkpoint_for_ref

    _modal_status("modal_action_start", action="diagnostics", run_id=run_id, dataset=dataset, shots=shots, seed=seed, checkpoint_ref=checkpoint_ref)
    data_vol.reload()
    weights_vol.reload()
    runs_vol.reload()
    config = _config(
        run_id=run_id,
        dataset=dataset,
        shots=shots,
        seed=seed,
        backbone=backbone,
        pretrained=pretrained,
        max_unlabeled_images=max_unlabeled_images,
    )
    checkpoint = checkpoint_for_ref(RUN_ROOT, run_id, config.dataset, shots, seed, config.backbone, checkpoint_ref)
    result = run_diagnostics(config, DATA_ROOT, RUN_ROOT, checkpoint)
    weights_vol.commit()
    runs_vol.commit()
    _modal_status("modal_action_complete", action="diagnostics", run_id=run_id, dataset=dataset, shots=shots, seed=seed, checkpoint_ref=checkpoint_ref, result=result)
    return result


@app.function(image=image, volumes=VOLUME_MOUNTS, timeout=60 * 60)
def aggregate_results(run_id: str):
    from promptsrc_nc.aggregate import aggregate_run

    _modal_status("modal_action_start", action="aggregate_results", run_id=run_id)
    runs_vol.reload()
    result = aggregate_run(RUN_ROOT, run_id)
    runs_vol.commit()
    _modal_status("modal_action_complete", action="aggregate_results", run_id=run_id, result=result)
    return result


@app.local_entrypoint()
def run(
    action: str = "smoke_test",
    run_id: str = "dev-smoke",
    datasets: str = "",
    dataset: str = "eurosat",
    shots: int = 1,
    prepare_shots: str = "",
    prepare_seeds: str = "",
    seed: int = 1,
    backbone: str = "ViT-B-16",
    pretrained: str = "openai",
    gpu: str = "T4",
    stage: str = "stage1",
    pair_mode: str = "real",
    pair_batch_size: int = 8,
    checkpoint_ref: str = "stage1",
    split: str = "test",
    max_unlabeled_images: int | None = None,
):
    """Small convenience dispatcher.

    Use direct function refs for scripted runs. This entrypoint routes T4/L4
    choices to statically registered Modal functions:

        uv run modal run promptSRC-NC/modal_app.py --action profile_gpu_cost --gpu L4
    """

    if action == "prepare_data":
        print(prepare_data.remote(datasets or dataset, prepare_shots or str(shots), prepare_seeds or str(seed)))
    elif action == "prepare_weights":
        print(prepare_weights.remote(backbone, pretrained))
    elif action == "smoke_test":
        print(_gpu_function(gpu, t4=smoke_test, l4=smoke_test_l4).remote(run_id, dataset, shots, seed, backbone, pretrained))
    elif action == "profile_gpu_cost":
        print(
            _gpu_function(gpu, t4=profile_gpu_cost, l4=profile_gpu_cost_l4).remote(
                run_id,
                dataset,
                shots,
                seed,
                backbone,
                pretrained,
                stage,
                gpu,
                10,
                100,
                pair_mode,
                pair_batch_size,
                max_unlabeled_images,
            )
        )
    elif action == "build_neighbors":
        print(_gpu_function(gpu, t4=build_neighbors, l4=build_neighbors_l4).remote(run_id, dataset, shots, seed, backbone, pretrained, max_unlabeled_images))
    elif action == "train_stage1":
        print(_gpu_function(gpu, t4=train_stage1, l4=train_stage1_l4).remote(run_id, dataset, shots, seed, backbone, pretrained))
    elif action == "train_stage2_real":
        print(_gpu_function(gpu, t4=train_stage2, l4=train_stage2_l4).remote(run_id, dataset, shots, seed, backbone, pretrained, "real", pair_batch_size, max_unlabeled_images))
    elif action == "train_stage2_shuffled":
        print(_gpu_function(gpu, t4=train_stage2, l4=train_stage2_l4).remote(run_id, dataset, shots, seed, backbone, pretrained, "shuffled", pair_batch_size, max_unlabeled_images))
    elif action == "train_stage2":
        print(_gpu_function(gpu, t4=train_stage2, l4=train_stage2_l4).remote(run_id, dataset, shots, seed, backbone, pretrained, pair_mode, pair_batch_size, max_unlabeled_images))
    elif action == "evaluate":
        print(_gpu_function(gpu, t4=evaluate, l4=evaluate_l4).remote(run_id, dataset, shots, seed, backbone, pretrained, checkpoint_ref, split))
    elif action == "diagnostics":
        print(_gpu_function(gpu, t4=diagnostics, l4=diagnostics_l4).remote(run_id, dataset, shots, seed, backbone, pretrained, checkpoint_ref, max_unlabeled_images))
    elif action == "aggregate_results":
        print(aggregate_results.remote(run_id))
    else:
        raise ValueError(f"Unknown action: {action}")
