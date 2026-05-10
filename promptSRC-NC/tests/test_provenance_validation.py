from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from promptsrc_nc.config import PromptSRCNCConfig
from promptsrc_nc.data import SplitSpec, validate_split_integrity
from promptsrc_nc import eval as eval_module
from promptsrc_nc import neighbors as neighbors_module
from promptsrc_nc.train import _load_stage1_checkpoint
from promptsrc_nc.provenance import ordered_ids_hash, split_hash, validate_checkpoint_for_config


def _config() -> PromptSRCNCConfig:
    return PromptSRCNCConfig(
        dataset="eurosat",
        shots=16,
        seed=1,
        backbone="ViT-B-16",
        pretrained="openai",
        run_id="test-run",
    )


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


def test_eval_rejects_checkpoint_for_wrong_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_path = tmp_path / "wrong.pt"
    torch.save(
        {
            "method": "PromptSRC",
            "stage": "stage1",
            "checkpoint_role": "stage1_gpa",
            "dataset": "oxford_flowers",
            "shots": 16,
            "seed": 1,
            "protocol": "few_shot_all_classes",
            "backbone": "ViT-B-16",
            "pretrained": "openai",
            "prompt_state": {"prompt_learner": {"ctx": torch.zeros(1)}},
        },
        checkpoint_path,
    )
    monkeypatch.setattr(
        eval_module,
        "load_split_records",
        lambda *args, **kwargs: ([], [], [], [], _split(), ["annual crop land"]),
    )
    monkeypatch.setattr(
        eval_module,
        "build_openclip_bundle",
        lambda *args, **kwargs: SimpleNamespace(train_preprocess=object(), eval_preprocess=object()),
    )
    monkeypatch.setattr(eval_module, "build_data_loaders", lambda *args, **kwargs: {})

    class DummyModel:
        def load_prompt_state_dict(self, state):
            pass

    monkeypatch.setattr(eval_module, "PromptSRCModel", lambda *args, **kwargs: DummyModel())

    with pytest.raises(ValueError, match="dataset"):
        eval_module.load_model_for_checkpoint(_config(), tmp_path, checkpoint_path)


def test_stage2_requires_stage1_gpa_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "final.pt"
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
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="stage1_gpa"):
        _load_stage1_checkpoint(checkpoint_path, _config())


def test_checkpoint_validation_accepts_legacy_split_object_without_hash() -> None:
    split = _split()
    checkpoint = {
        "method": "PromptSRC",
        "stage": "stage1",
        "checkpoint_role": "stage1_gpa",
        "dataset": "eurosat",
        "shots": 16,
        "seed": 1,
        "protocol": "few_shot_all_classes",
        "backbone": "ViT-B-16",
        "pretrained": "openai",
        "metadata": {"split": split.to_dict()},
    }

    validate_checkpoint_for_config(
        checkpoint,
        _config(),
        expected_method="PromptSRC",
        expected_stage="stage1",
        expected_role="stage1_gpa",
        split=split,
    )


def test_split_integrity_rejects_test_ids_in_unlabeled_pool() -> None:
    split = SplitSpec(
        dataset="eurosat",
        protocol="few_shot_all_classes",
        shots=16,
        seed=1,
        train_ids=("train-a",),
        val_ids=("val-a",),
        test_ids=("test-a",),
        unlabeled_ids=("test-a",),
        metadata={
            "unlabeled_policy": "full_training_split_minus_fewshot_labeled_train",
            "uses_test_images_for_unlabeled": False,
        },
    )

    with pytest.raises(ValueError, match="test-unlabeled overlap"):
        validate_split_integrity(split)


def test_neighbor_validation_rejects_backbone_mismatch(tmp_path: Path) -> None:
    pair_dir = tmp_path / "neighbors"
    pair_dir.mkdir()
    (pair_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": "eurosat",
                "protocol": "few_shot_all_classes",
                "shots": 16,
                "seed": 1,
                "clip_backbone": "ViT-B-32",
                "pretrained": "openai",
                "unlabeled_policy": "full_training_split_minus_fewshot_labeled_train",
                "uses_test_images_for_unlabeled": False,
                "num_unlabeled": 2,
                "unlabeled_ids_hash": "bad",
                "split_hash": "bad",
            }
        ),
        encoding="utf-8",
    )
    (pair_dir / "unlabeled_items.jsonl").write_text("", encoding="utf-8")
    torch.save({"pairs": torch.empty(0, 2, dtype=torch.long)}, pair_dir / "real_pairs.pt")
    torch.save({"pairs": torch.empty(0, 2, dtype=torch.long)}, pair_dir / "shuffled_pairs.pt")

    validate = getattr(neighbors_module, "validate_neighbor_artifacts", None)
    assert validate is not None
    with pytest.raises(ValueError, match="backbone"):
        validate(pair_dir, _config(), _split())


def test_neighbor_validation_allows_configured_unlabeled_cap(tmp_path: Path) -> None:
    pair_dir = tmp_path / "neighbors"
    pair_dir.mkdir()
    split = _split()
    capped_config = PromptSRCNCConfig(
        **{
            **_config().to_dict(),
            "max_unlabeled_images": 1,
        }
    )
    (pair_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": "eurosat",
                "protocol": "few_shot_all_classes",
                "shots": 16,
                "seed": 1,
                "clip_backbone": "ViT-B-16",
                "pretrained": "openai",
                "unlabeled_policy": "full_training_split_minus_fewshot_labeled_train",
                "uses_test_images_for_unlabeled": False,
                "num_unlabeled": 1,
                "max_unlabeled_images": 1,
                "unlabeled_ids_hash": ordered_ids_hash(split.unlabeled_ids[:1]),
                "split_hash": split_hash(split),
                "num_real_pairs": 0,
                "num_shuffled_pairs": 0,
            }
        ),
        encoding="utf-8",
    )
    (pair_dir / "unlabeled_items.jsonl").write_text(
        json.dumps({"index": 0, "uid": "unlab-a", "impath": "/tmp/a.jpg", "label": 0, "classname": "a"}) + "\n",
        encoding="utf-8",
    )
    torch.save({"pairs": torch.empty(0, 2, dtype=torch.long)}, pair_dir / "real_pairs.pt")
    torch.save({"pairs": torch.empty(0, 2, dtype=torch.long)}, pair_dir / "shuffled_pairs.pt")

    validate = getattr(neighbors_module, "validate_neighbor_artifacts", None)
    assert validate is not None
    validate(pair_dir, capped_config, split)


def test_neighbor_validation_rejects_stale_unlabeled_items(tmp_path: Path) -> None:
    pair_dir = tmp_path / "neighbors"
    pair_dir.mkdir()
    split = _split()
    (pair_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": "eurosat",
                "protocol": "few_shot_all_classes",
                "shots": 16,
                "seed": 1,
                "clip_backbone": "ViT-B-16",
                "pretrained": "openai",
                "unlabeled_policy": "full_training_split_minus_fewshot_labeled_train",
                "uses_test_images_for_unlabeled": False,
                "num_unlabeled": 2,
                "unlabeled_ids_hash": ordered_ids_hash(split.unlabeled_ids),
                "split_hash": split_hash(split),
                "num_real_pairs": 1,
                "num_shuffled_pairs": 1,
            }
        ),
        encoding="utf-8",
    )
    (pair_dir / "unlabeled_items.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"index": 0, "uid": "unlab-a", "impath": "/tmp/a.jpg", "label": 0, "classname": "a"}),
                json.dumps({"index": 1, "uid": "stale-id", "impath": "/tmp/b.jpg", "label": 1, "classname": "b"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    torch.save({"pairs": torch.tensor([[0, 1]], dtype=torch.long)}, pair_dir / "real_pairs.pt")
    torch.save({"pairs": torch.tensor([[0, 1]], dtype=torch.long)}, pair_dir / "shuffled_pairs.pt")

    with pytest.raises(ValueError, match="unlabeled_ids_hash"):
        neighbors_module.validate_neighbor_artifacts(pair_dir, _config(), split)


def test_openclip_compatibility_rejects_non_batch_first_transformer() -> None:
    from promptsrc_nc.model import validate_openclip_compatibility

    block = object()
    model = SimpleNamespace(
        token_embedding=object(),
        transformer=SimpleNamespace(batch_first=False, resblocks=[block]),
        positional_embedding=torch.zeros(77, 512),
        ln_final=object(),
        visual=SimpleNamespace(
            conv1=SimpleNamespace(out_channels=768, weight=torch.zeros(1)),
            class_embedding=torch.zeros(768),
            positional_embedding=torch.zeros(197, 768),
            ln_pre=object(),
            transformer=SimpleNamespace(batch_first=True, resblocks=[block]),
            ln_post=object(),
        ),
        encode_image=lambda *args, **kwargs: None,
        encode_text=lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="batch_first"):
        validate_openclip_compatibility(model)


def test_openclip_random_weight_alias_maps_to_no_pretrained_tag() -> None:
    from promptsrc_nc.model import _openclip_pretrained_arg

    assert _openclip_pretrained_arg("none") is None
    assert _openclip_pretrained_arg("random") is None
    assert _openclip_pretrained_arg("openai") == "openai"
