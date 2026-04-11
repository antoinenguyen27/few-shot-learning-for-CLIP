from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from common.evaluation.results import (
    RunResult,
    append_result,
    read_results,
    result_jsonl_path,
    summarize_results,
)


class ResultsTests(unittest.TestCase):
    def test_result_jsonl_round_trip_and_summary(self) -> None:
        first = RunResult(
            method="LP++",
            dataset="eurosat",
            protocol="few_shot_all_classes",
            model_name="ViT-B-32-256",
            pretrained="datacomp_s34b_b86k",
            shots=1,
            seed=1,
            metrics={"test/top1_accuracy": 0.5, "val/top1_accuracy": 0.4},
            split_path="data/splits/eurosat/few_shot_all_classes/shots_1/seed_1.json",
        )
        second = RunResult(
            method="LP++",
            dataset="eurosat",
            protocol="few_shot_all_classes",
            model_name="ViT-B-32-256",
            pretrained="datacomp_s34b_b86k",
            shots=1,
            seed=2,
            metrics={"test/top1_accuracy": 0.7, "val/top1_accuracy": 0.6},
            split_path="data/splits/eurosat/few_shot_all_classes/shots_1/seed_2.json",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.jsonl"
            append_result(first, path)
            append_result(second, path)
            loaded = read_results(path)
        self.assertEqual(len(loaded), 2)
        summary = summarize_results(loaded)
        self.assertEqual(summary[0]["mean"], 0.6)
        self.assertEqual(summary[0]["num_seeds"], 2)

    def test_result_jsonl_path(self) -> None:
        self.assertEqual(result_jsonl_path("a/b").as_posix(), "results/a_b.jsonl")


if __name__ == "__main__":
    unittest.main()
