from __future__ import annotations

from pathlib import Path

from promptsrc_nc.config import PromptSRCNCConfig
from promptsrc_nc import smoke as smoke_module


def test_smoke_stage2_uses_active_nc_loss(monkeypatch, tmp_path: Path) -> None:
    captured_stage2_configs: list[PromptSRCNCConfig] = []

    def fake_build_neighbor_artifacts(config, data_root, run_root):
        return tmp_path / "neighbors"

    def fake_train_stage1(config, data_root, run_root):
        return tmp_path / "stage1.pt"

    def fake_train_stage2(config, data_root, run_root, init_checkpoint, pair_artifact_dir):
        captured_stage2_configs.append(config)
        return tmp_path / f"stage2-{config.pair_mode}.pt"

    def fake_evaluate_checkpoint(config, data_root, run_root, checkpoint, split):
        return {"top1_accuracy": 0.0}

    monkeypatch.setattr(smoke_module, "build_neighbor_artifacts", fake_build_neighbor_artifacts)
    monkeypatch.setattr(smoke_module, "train_stage1", fake_train_stage1)
    monkeypatch.setattr(smoke_module, "train_stage2", fake_train_stage2)
    monkeypatch.setattr(smoke_module, "evaluate_checkpoint", fake_evaluate_checkpoint)

    smoke_module.run_smoke_test(PromptSRCNCConfig(run_id="smoke-test"), tmp_path / "data", tmp_path / "runs")

    assert [config.pair_mode for config in captured_stage2_configs] == ["real", "shuffled"]
    assert all(config.lambda_nc_warmup_epochs == 0.0 for config in captured_stage2_configs)
