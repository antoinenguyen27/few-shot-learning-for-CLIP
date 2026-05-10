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

VOLUME_MOUNTS = {
    DATA_ROOT: data_vol,
    WEIGHTS_ROOT: weights_vol,
    RUN_ROOT: runs_vol,
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .uv_sync("promptSRC-NC")
    .add_local_dir("promptSRC-NC/promptsrc_nc", remote_path="/root/promptsrc_nc")
    .workdir("/root")
    .env(
        {
            "HF_HOME": f"{WEIGHTS_ROOT}/hf",
            "TORCH_HOME": f"{WEIGHTS_ROOT}/torch",
            "OPENCLIP_CACHE_DIR": f"{WEIGHTS_ROOT}/openclip",
            "PYTHONUNBUFFERED": "1",
        }
    )
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
    precision: str = "amp",
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


@app.function(
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("kaggle")],
    timeout=60 * 60 * 6,
)
def prepare_data(datasets: str = "eurosat,oxford_flowers,stanford_cars", shots: str = "1,16", seeds: str = "1,2,3"):
    from promptsrc_nc.data import parse_dataset_list, parse_int_list, prepare_datasets

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
    return result


@app.function(image=image, volumes=VOLUME_MOUNTS, gpu="T4", timeout=60 * 60 * 2)
def smoke_test(
    run_id: str,
    dataset: str = "eurosat",
    shots: int = 1,
    seed: int = 1,
    backbone: str = "ViT-B-16",
    pretrained: str = "openai",
):
    from promptsrc_nc.smoke import run_smoke_test

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
    return result


@app.function(image=image, volumes=VOLUME_MOUNTS, gpu="T4", timeout=60 * 60 * 2)
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
    from promptsrc_nc.cost_profile import profile_gpu_cost as run_profile

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
    return result


@app.function(image=image, volumes=VOLUME_MOUNTS, gpu="T4", timeout=60 * 60 * 6)
def build_neighbors(
    run_id: str,
    dataset: str,
    shots: int = 16,
    seed: int = 1,
    backbone: str = "ViT-B-16",
    pretrained: str = "openai",
    max_unlabeled_images: int | None = None,
):
    from promptsrc_nc.neighbors import build_neighbor_artifacts

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
    return result


@app.function(image=image, volumes=VOLUME_MOUNTS, gpu="T4", timeout=60 * 60 * 8)
def train_stage1(
    run_id: str,
    dataset: str,
    shots: int = 16,
    seed: int = 1,
    backbone: str = "ViT-B-16",
    pretrained: str = "openai",
):
    from promptsrc_nc.train import train_stage1 as run_train_stage1

    data_vol.reload()
    weights_vol.reload()
    runs_vol.reload()
    config = _config(run_id=run_id, dataset=dataset, shots=shots, seed=seed, backbone=backbone, pretrained=pretrained)
    result = str(run_train_stage1(config, DATA_ROOT, RUN_ROOT))
    weights_vol.commit()
    runs_vol.commit()
    return result


@app.function(image=image, volumes=VOLUME_MOUNTS, gpu="T4", timeout=60 * 60 * 4)
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
    from promptsrc_nc.train import train_stage2 as run_train_stage2

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
    return result


@app.function(image=image, volumes=VOLUME_MOUNTS, gpu="T4", timeout=60 * 60 * 2)
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
    from promptsrc_nc.eval import checkpoint_for_ref, evaluate_checkpoint

    data_vol.reload()
    weights_vol.reload()
    runs_vol.reload()
    config = _config(run_id=run_id, dataset=dataset, shots=shots, seed=seed, backbone=backbone, pretrained=pretrained)
    checkpoint = checkpoint_for_ref(RUN_ROOT, run_id, config.dataset, shots, seed, config.backbone, checkpoint_ref)
    result = evaluate_checkpoint(config, DATA_ROOT, RUN_ROOT, checkpoint, split=split)
    weights_vol.commit()
    runs_vol.commit()
    return result


@app.function(image=image, volumes=VOLUME_MOUNTS, gpu="T4", timeout=60 * 60 * 2)
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
    from promptsrc_nc.diagnostics import run_diagnostics
    from promptsrc_nc.eval import checkpoint_for_ref

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
    return result


@app.function(image=image, volumes=VOLUME_MOUNTS, timeout=60 * 60)
def aggregate_results(run_id: str):
    from promptsrc_nc.aggregate import aggregate_run

    runs_vol.reload()
    result = aggregate_run(RUN_ROOT, run_id)
    runs_vol.commit()
    return result


@app.local_entrypoint()
def run(
    action: str = "smoke_test",
    run_id: str = "dev-smoke",
    dataset: str = "eurosat",
    shots: int = 1,
    seed: int = 1,
    backbone: str = "ViT-B-16",
    gpu: str = "T4",
    stage: str = "stage1",
    pair_mode: str = "real",
    pair_batch_size: int = 8,
):
    """Small convenience dispatcher.

    Use direct function refs for scripted runs. For L4 profiling, this entrypoint
    calls the profiled function with a dynamic GPU option:

        uv run modal run promptSRC-NC/modal_app.py --action profile_gpu_cost --gpu L4
    """

    if action == "prepare_data":
        print(prepare_data.remote())
    elif action == "smoke_test":
        print(smoke_test.with_options(gpu=gpu).remote(run_id, dataset, shots, seed, backbone))
    elif action == "profile_gpu_cost":
        print(
            profile_gpu_cost.with_options(gpu=gpu).remote(
                run_id,
                dataset,
                shots,
                seed,
                backbone,
                "openai",
                stage,
                gpu,
                10,
                100,
                pair_mode,
                pair_batch_size,
            )
        )
    elif action == "build_neighbors":
        print(build_neighbors.with_options(gpu=gpu).remote(run_id, dataset, shots, seed, backbone))
    elif action == "train_stage1":
        print(train_stage1.with_options(gpu=gpu).remote(run_id, dataset, shots, seed, backbone))
    elif action == "train_stage2_real":
        print(train_stage2.with_options(gpu=gpu).remote(run_id, dataset, shots, seed, backbone, "openai", "real"))
    elif action == "train_stage2_shuffled":
        print(train_stage2.with_options(gpu=gpu).remote(run_id, dataset, shots, seed, backbone, "openai", "shuffled"))
    elif action == "evaluate":
        print(evaluate.with_options(gpu=gpu).remote(run_id, dataset, shots, seed, backbone))
    elif action == "aggregate_results":
        print(aggregate_results.remote(run_id))
    else:
        raise ValueError(f"Unknown action: {action}")
