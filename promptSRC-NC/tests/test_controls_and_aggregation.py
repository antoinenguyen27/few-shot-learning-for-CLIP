from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from promptsrc_nc.aggregate import aggregate_run
from promptsrc_nc.neighbors import degree_preserving_shuffle


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def test_degree_preserving_shuffle_reports_audit_metadata() -> None:
    pairs = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 0],
        ],
        dtype=torch.long,
    )

    result = degree_preserving_shuffle(pairs, num_nodes=6, seed=3, return_audit=True)

    assert isinstance(result, tuple)
    shuffled, audit = result
    assert shuffled.shape == pairs.shape
    assert audit["degree_preserved"] is True
    assert audit["attempted_swaps"] == 10 * len(pairs)
    assert audit["accepted_swaps"] >= 0
    assert 0.0 <= audit["edge_overlap_fraction"] <= 1.0


def test_aggregate_preserves_validation_and_test_metrics(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_id = "run-a"
    log_path = run_root / run_id / "logs" / "eval_summary.jsonl"
    base = {
        "event": "eval_summary",
        "run_id": run_id,
        "method": "PromptSRC",
        "pair_mode": "none",
        "dataset": "eurosat",
        "shots": 16,
        "seed": 1,
        "backbone": "ViT-B-16",
        "checkpoint": "/tmp/gpa.pt",
    }
    _append_jsonl(log_path, {**base, "split": "val", "top1_accuracy": 0.4, "macro_accuracy": 0.3, "num_examples": 10})
    _append_jsonl(log_path, {**base, "split": "test", "top1_accuracy": 0.5, "macro_accuracy": 0.45, "num_examples": 20})

    aggregate_run(run_root, run_id)

    rows = [
        json.loads(line)
        for line in (run_root / run_id / "results" / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["val_top1"] == pytest.approx(0.4)
    assert rows[0]["val_macro"] == pytest.approx(0.3)
    assert rows[0]["test_top1"] == pytest.approx(0.5)
    assert rows[0]["test_macro"] == pytest.approx(0.45)
