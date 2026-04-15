from __future__ import annotations

import unittest
from types import SimpleNamespace

from ZeroShotCLIP.zero_shot_clip.config import ZeroShotCLIPConfig
from ZeroShotCLIP.zero_shot_clip.method import ZeroShotCLIPMethod
from ZeroShotCLIP.zero_shot_clip.runner import (
    ZERO_SHOT_RESULT_SEED,
    ZERO_SHOT_RESULT_SHOTS,
    parse_args,
)


class ZeroShotCLIPTests(unittest.TestCase):
    def test_config_to_dict(self) -> None:
        config = ZeroShotCLIPConfig(precision="amp", max_eval_batches=2, show_progress=False)
        payload = config.to_dict()
        self.assertEqual(payload["precision"], "amp")
        self.assertEqual(payload["max_eval_batches"], 2)
        self.assertFalse(payload["show_progress"])

    def test_config_rejects_invalid_precision(self) -> None:
        with self.assertRaises(ValueError):
            ZeroShotCLIPConfig(precision="bf16")

    def test_config_rejects_non_positive_eval_limit(self) -> None:
        with self.assertRaises(ValueError):
            ZeroShotCLIPConfig(max_eval_batches=0)

    def test_infers_dataset_name_from_loader_records(self) -> None:
        loader = SimpleNamespace(dataset=SimpleNamespace(records=[SimpleNamespace(dataset="eurosat")]))
        method = ZeroShotCLIPMethod(config=ZeroShotCLIPConfig(show_progress=False))
        self.assertEqual(method._infer_dataset_name(loader), "eurosat")

    def test_custom_templates_do_not_need_loader_metadata(self) -> None:
        method = ZeroShotCLIPMethod(templates=["a photo of {}."])
        self.assertEqual(method._resolve_templates(None), ["a photo of {}."])

    def test_runner_defaults_keep_logged_zero_shot_fields_separate(self) -> None:
        args = parse_args(["--dataset", "eurosat"])
        self.assertEqual(args.split_shots, 1)
        self.assertEqual(args.split_seed, 1)
        self.assertEqual(ZERO_SHOT_RESULT_SHOTS, 0)
        self.assertEqual(ZERO_SHOT_RESULT_SEED, 0)

    def test_runner_legacy_shot_seed_flags_select_split_only(self) -> None:
        args = parse_args(["--dataset", "eurosat", "--shots", "4", "--seed", "3"])
        self.assertEqual(args.split_shots, 4)
        self.assertEqual(args.split_seed, 3)


if __name__ == "__main__":
    unittest.main()
