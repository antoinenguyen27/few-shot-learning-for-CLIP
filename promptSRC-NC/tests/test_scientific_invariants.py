from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import savemat

from promptsrc_nc import eval as eval_module
from promptsrc_nc.aggregate import aggregate_run
from promptsrc_nc.config import PromptSRCNCConfig
from promptsrc_nc.data import (
    ManifestRecord,
    SplitSpec,
    build_flowers_manifest,
    build_stanford_cars_manifest,
    ensure_dataset_downloaded,
    load_split_records,
    prepare_datasets,
    write_manifest,
    write_split,
)
from promptsrc_nc.losses import js_divergence_from_logits, lambda_nc_for_progress
from promptsrc_nc.model import openclip_model_name_for_weights
from promptsrc_nc.neighbors import validate_neighbor_artifacts
from promptsrc_nc.provenance import ordered_ids_hash, split_hash


def _split() -> SplitSpec:
    return SplitSpec(
        dataset="eurosat",
        protocol="few_shot_all_classes",
        shots=16,
        seed=1,
        train_ids=("train-a",),
        val_ids=("val-a",),
        test_ids=("test-a",),
        unlabeled_ids=("unlab-a", "unlab-b"),
        metadata={
            "unlabeled_policy": "full_training_split_minus_fewshot_labeled_train",
            "uses_test_images_for_unlabeled": False,
        },
    )


def _config(**overrides) -> PromptSRCNCConfig:
    payload = PromptSRCNCConfig(dataset="eurosat", shots=16, seed=1, backbone="ViT-B-16", run_id="test").to_dict()
    payload.update(overrides)
    return PromptSRCNCConfig(**payload)


def _neighbor_dir(tmp_path: Path, pairs: torch.Tensor, *, include_features: bool = True) -> Path:
    split = _split()
    pair_dir = tmp_path / "neighbors"
    pair_dir.mkdir()
    metadata = {
        "dataset": "eurosat",
        "protocol": "few_shot_all_classes",
        "shots": 16,
        "seed": 1,
        "clip_backbone": "ViT-B-16",
        "openclip_model_name": openclip_model_name_for_weights("ViT-B-16", "openai"),
        "pretrained": "openai",
        "unlabeled_policy": "full_training_split_minus_fewshot_labeled_train",
        "uses_test_images_for_unlabeled": False,
        "feature_source": "frozen_unprompted_openclip_before_promptsrc",
        "num_unlabeled": 2,
        "max_unlabeled_images": None,
        "unlabeled_ids_hash": ordered_ids_hash(split.unlabeled_ids),
        "unlabeled_paths_hash": ordered_ids_hash([f"/current/{uid}.jpg" for uid in split.unlabeled_ids]),
        "split_hash": split_hash(split),
        "neighbor_k_requested": 5,
        "neighbor_k_used": 5,
        "fallback_used": False,
        "num_real_pairs": int(pairs.shape[0]),
        "num_shuffled_pairs": int(pairs.shape[0]),
    }
    (pair_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (pair_dir / "unlabeled_items.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "index": index,
                    "uid": uid,
                    "impath": f"/current/{uid}.jpg",
                    "label": index,
                    "classname": uid,
                }
            )
            for index, uid in enumerate(split.unlabeled_ids)
        )
        + "\n",
        encoding="utf-8",
    )
    if include_features:
        torch.save(
            {
                "features": torch.eye(2),
                "uids": list(split.unlabeled_ids),
                "impaths": [f"/current/{uid}.jpg" for uid in split.unlabeled_ids],
                "backbone": "ViT-B-16",
                "openclip_model_name": openclip_model_name_for_weights("ViT-B-16", "openai"),
                "pretrained": "openai",
                "feature_source": "frozen_unprompted_openclip_before_promptsrc",
            },
            pair_dir / "features.pt",
        )
    torch.save({"pairs": pairs, "neighbor_k": 5, "mutual": True}, pair_dir / "real_pairs.pt")
    torch.save({"pairs": pairs.clone(), "source": "degree_preserving_edge_swap"}, pair_dir / "shuffled_pairs.pt")
    return pair_dir


def test_loaded_split_must_match_train_minus_fewshot_invariant(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    processed = data_root / "processed" / "eurosat"
    records = [
        ManifestRecord("train-a", "eurosat", "/tmp/train-a.jpg", 0, "a", "train"),
        ManifestRecord("train-b", "eurosat", "/tmp/train-b.jpg", 1, "b", "train"),
        ManifestRecord("val-a", "eurosat", "/tmp/val-a.jpg", 0, "a", "val"),
        ManifestRecord("test-a", "eurosat", "/tmp/test-a.jpg", 0, "a", "test"),
    ]
    write_manifest(records, processed / "manifest.jsonl")
    write_split(
        SplitSpec(
            dataset="eurosat",
            protocol="few_shot_all_classes",
            shots=16,
            seed=1,
            train_ids=("train-a",),
            val_ids=(),
            test_ids=("test-a",),
            unlabeled_ids=("train-b", "val-a"),
            metadata={"uses_test_images_for_unlabeled": False},
        ),
        processed / "splits" / "few_shot_all_classes" / "shots_16" / "seed_1.json",
    )

    with pytest.raises(ValueError, match="unlabeled pool"):
        load_split_records(data_root, "eurosat", 16, 1)


def test_flowers_flat_kaggle_test_layout_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "oxford_flowers"
    for split_name in ("train", "valid"):
        class_dir = root / "dataset" / split_name / "1"
        class_dir.mkdir(parents=True)
        (class_dir / "image_00001.jpg").write_bytes(b"not-an-image")
    flat_test = root / "dataset" / "test"
    flat_test.mkdir(parents=True)
    (flat_test / "image_00002.jpg").write_bytes(b"not-an-image")
    (root / "cat_to_name.json").write_text(json.dumps({"1": "pink primrose"}), encoding="utf-8")

    with pytest.raises(ValueError, match="unlabeled Kaggle test"):
        build_flowers_manifest(root)


def test_flowers_official_labels_manifest_uses_promptsrc_compatible_splits(tmp_path: Path) -> None:
    root = tmp_path / "oxford_flowers"
    jpg = root / "jpg"
    jpg.mkdir(parents=True)
    labels = []
    for index in range(1, 81):
        (jpg / f"image_{index:05d}.jpg").write_bytes(b"not-an-image")
        labels.append(1 if index <= 40 else 2)
    savemat(root / "imagelabels.mat", {"labels": [labels]})
    savemat(root / "setid.mat", {"trnid": [[1, 41]], "valid": [[2, 42]], "tstid": [[3, 43]]})
    (root / "cat_to_name.json").write_text(
        json.dumps({"1": "pink primrose", "2": "hard-leaved pocket orchid"}),
        encoding="utf-8",
    )

    records = build_flowers_manifest(root)

    per_class_train_counts = {
        label: sum(record.source_split == "train" for record in records if record.label_id == label)
        for label in (0, 1)
    }
    split_counts = {split: sum(record.source_split == split for record in records) for split in ("train", "val", "test")}
    assert split_counts == {"train": 40, "val": 16, "test": 24}
    assert per_class_train_counts == {0: 20, 1: 20}
    assert {record.metadata["split_source"] for record in records} == {"promptsrc_style_50_20_30_from_labels"}


def test_existing_invalid_flowers_source_is_replaced_when_download_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "extracted" / "oxford_flowers"
    invalid_test = target / "dataset" / "test"
    invalid_test.mkdir(parents=True)
    (invalid_test / "image_00001.jpg").write_bytes(b"not-an-image")

    def fake_prepare(destination: Path) -> None:
        (destination / "jpg").mkdir(parents=True)
        (destination / "jpg" / "image_00001.jpg").write_bytes(b"not-an-image")
        savemat(destination / "imagelabels.mat", {"labels": [[1]]})
        savemat(destination / "setid.mat", {"trnid": [[1]], "valid": [[]], "tstid": [[]]})

    monkeypatch.setattr("promptsrc_nc.data._prepare_official_flowers102_source", fake_prepare)

    result = ensure_dataset_downloaded(tmp_path, "oxford_flowers", download=True)

    assert result == target
    assert (target / "jpg" / "image_00001.jpg").exists()
    assert not (target / "dataset").exists()


def _write_car_annos(path: Path, rows: list[tuple[str, int]]) -> None:
    dtype = [
        ("bbox_x1", "O"),
        ("bbox_y1", "O"),
        ("bbox_x2", "O"),
        ("bbox_y2", "O"),
        ("class", "O"),
        ("fname", "O"),
    ]
    annotations = np.empty((1, len(rows)), dtype=dtype)
    for index, (filename, label) in enumerate(rows):
        annotations[0, index]["bbox_x1"] = np.array([[0]])
        annotations[0, index]["bbox_y1"] = np.array([[0]])
        annotations[0, index]["bbox_x2"] = np.array([[1]])
        annotations[0, index]["bbox_y2"] = np.array([[1]])
        annotations[0, index]["class"] = np.array([[label]])
        annotations[0, index]["fname"] = np.array([filename], dtype=object)
    savemat(path, {"annotations": annotations})


def test_stanford_cars_kaggle_nested_layout_uses_official_labeled_test_annos(tmp_path: Path) -> None:
    root = tmp_path / "stanford_cars"
    train_dir = root / "cars_train" / "cars_train"
    test_dir = root / "cars_test" / "cars_test"
    devkit = root / "car_devkit" / "devkit"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    devkit.mkdir(parents=True)
    train_rows = []
    for index in range(1, 11):
        filename = f"{index:05d}.jpg"
        (train_dir / filename).write_bytes(b"not-an-image")
        train_rows.append((filename, 1 if index <= 5 else 2))
    test_rows = []
    for index in range(1, 3):
        filename = f"{index:05d}.jpg"
        (test_dir / filename).write_bytes(b"not-an-image")
        test_rows.append((filename, 2 if index == 1 else 1))
    _write_car_annos(devkit / "cars_train_annos.mat", train_rows)
    _write_car_annos(root / "cars_test_annos_withlabels.mat", test_rows)
    savemat(devkit / "cars_meta.mat", {"class_names": np.array([["make a 2001", "make b 2002"]], dtype=object)})

    records = build_stanford_cars_manifest(root)

    assert sum(record.source_split == "train" for record in records) == 8
    assert sum(record.source_split == "val" for record in records) == 2
    assert sum(record.source_split == "test" for record in records) == 2
    assert {record.metadata["source_layout"] for record in records} == {"stanford_cars_mat"}
    assert all(Path(record.image_path).exists() for record in records)
    assert {Path(record.image_path).parent.name for record in records} == {"cars_train", "cars_test"}


def test_prepare_data_rejects_manifest_with_missing_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing_records = [
        ManifestRecord("train-a", "eurosat", str(tmp_path / "missing-train.jpg"), 0, "a", "train"),
        ManifestRecord("val-a", "eurosat", str(tmp_path / "missing-val.jpg"), 0, "a", "val"),
        ManifestRecord("test-a", "eurosat", str(tmp_path / "missing-test.jpg"), 0, "a", "test"),
    ]
    monkeypatch.setattr("promptsrc_nc.data.ensure_dataset_downloaded", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr("promptsrc_nc.data.build_manifest_for_dataset", lambda *args, **kwargs: missing_records)

    with pytest.raises(FileNotFoundError, match="Manifest for eurosat references missing image files"):
        prepare_datasets(tmp_path, ["eurosat"], shots=[1], seeds=[1], download=False)


def test_stanford_cars_labeled_test_anno_download_uses_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_download(url: str, destination: Path, expected_md5: str | None = None) -> None:
        calls.append(url)
        if len(calls) == 1:
            raise RuntimeError("primary unavailable")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"mat")

    monkeypatch.setattr("promptsrc_nc.data._download_file", fake_download)

    from promptsrc_nc.data import _ensure_stanford_cars_labeled_test_annos

    _ensure_stanford_cars_labeled_test_annos(tmp_path, download=True)

    assert len(calls) == 2
    assert (tmp_path / "cars_test_annos_withlabels.mat").read_bytes() == b"mat"


@pytest.mark.parametrize("pairs", [torch.tensor([[0, 0]]), torch.tensor([[0, 1], [1, 0]])])
def test_neighbor_validation_rejects_self_loops_and_duplicate_undirected_edges(tmp_path: Path, pairs: torch.Tensor) -> None:
    pair_dir = _neighbor_dir(tmp_path, pairs.long())

    with pytest.raises(ValueError, match="pairs"):
        validate_neighbor_artifacts(pair_dir, _config(), _split())


def test_neighbor_validation_requires_feature_artifact(tmp_path: Path) -> None:
    pair_dir = _neighbor_dir(tmp_path, torch.tensor([[0, 1]], dtype=torch.long), include_features=False)

    with pytest.raises(FileNotFoundError, match="features.pt"):
        validate_neighbor_artifacts(pair_dir, _config(), _split())


def test_neighbor_validation_rejects_paths_that_do_not_match_manifest(tmp_path: Path) -> None:
    pair_dir = _neighbor_dir(tmp_path, torch.tensor([[0, 1]], dtype=torch.long))
    current_records = [
        ManifestRecord("unlab-a", "eurosat", "/current/unlab-a.jpg", 0, "unlab-a", "train"),
        ManifestRecord("unlab-b", "eurosat", "/different/unlab-b.jpg", 1, "unlab-b", "train"),
    ]

    with pytest.raises(ValueError, match="image_path"):
        validate_neighbor_artifacts(pair_dir, _config(), _split(), unlabeled_records=current_records)


def test_eval_rejects_non_gpa_promptsrc_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_path = tmp_path / "final.pt"
    split = _split()
    torch.save(
        {
            "method": "PromptSRC",
            "stage": "stage1",
            "checkpoint_role": "stage1_final_non_gpa",
            "dataset": "eurosat",
            "shots": 16,
            "seed": 1,
            "protocol": "few_shot_all_classes",
            "backbone": "ViT-B-16",
            "pretrained": "openai",
            "prompt_state": {},
            "metadata": {"split_hash": split_hash(split)},
        },
        checkpoint_path,
    )
    monkeypatch.setattr(eval_module, "load_split_records", lambda *args, **kwargs: ([], [], [], [], split, ["annual crop land"]))

    with pytest.raises(ValueError, match="stage1_gpa"):
        eval_module.load_model_for_checkpoint(_config(), tmp_path, checkpoint_path)


def test_aggregation_does_not_merge_val_and_test_from_different_checkpoints(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_id = "run-a"
    log_path = run_root / run_id / "logs" / "eval_summary.jsonl"
    log_path.parent.mkdir(parents=True)
    base = {
        "event": "eval_summary",
        "run_id": run_id,
        "method": "PromptSRC",
        "pair_mode": "none",
        "dataset": "eurosat",
        "shots": 16,
        "seed": 1,
        "backbone": "ViT-B-16",
        "protocol": "few_shot_all_classes",
    }
    rows = [
        {
            **base,
            "split": "val",
            "top1_accuracy": 0.4,
            "macro_accuracy": 0.3,
            "checkpoint": "/tmp/gpa.pt",
            "checkpoint_role": "stage1_gpa",
            "split_hash": "split-a",
        },
        {
            **base,
            "split": "test",
            "top1_accuracy": 0.5,
            "macro_accuracy": 0.45,
            "checkpoint": "/tmp/final.pt",
            "checkpoint_role": "stage1_final_non_gpa",
            "split_hash": "split-a",
        },
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    aggregate_run(run_root, run_id)

    run_rows = [
        json.loads(line)
        for line in (run_root / run_id / "results" / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(run_rows) == 2


def test_js_is_symmetric_and_lambda_warmup_starts_at_zero() -> None:
    logits_a = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    logits_b = torch.tensor([[0.0, 2.0], [1.0, 0.0]])

    assert js_divergence_from_logits(logits_a, logits_b) == pytest.approx(js_divergence_from_logits(logits_b, logits_a))
    assert lambda_nc_for_progress(0.0, 1.0, 1.0) == pytest.approx(0.0)
    assert lambda_nc_for_progress(0.5, 1.0, 1.0) == pytest.approx(0.5)
    assert lambda_nc_for_progress(1.0, 1.0, 1.0) == pytest.approx(1.0)
