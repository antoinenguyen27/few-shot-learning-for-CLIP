from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from common.datasets.splits import FewShotSplitError, make_few_shot_split, read_split, write_split
from common.datasets.types import ImageRecord


def make_records() -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for label, class_name in enumerate(("alpha", "beta", "gamma")):
        for split, count in (("train", 5), ("val", 2), ("test", 3)):
            for idx in range(count):
                records.append(
                    ImageRecord(
                        dataset="toy",
                        sample_id=f"toy/{class_name}/{split}/{idx}",
                        image_path=f"{class_name}/{split}/{idx}.jpg",
                        label_id=label,
                        class_name=class_name,
                        source_split=split,
                    )
                )
    return records


class SplitTests(unittest.TestCase):
    def test_few_shot_split_counts(self) -> None:
        split = make_few_shot_split(make_records(), dataset="toy", shots=2, seed=1)
        self.assertEqual(len(split.train_ids), 6)
        self.assertEqual(len(split.val_ids), 6)
        self.assertEqual(len(split.test_ids), 9)

    def test_few_shot_split_is_deterministic(self) -> None:
        first = make_few_shot_split(make_records(), dataset="toy", shots=2, seed=7)
        second = make_few_shot_split(make_records(), dataset="toy", shots=2, seed=7)
        third = make_few_shot_split(make_records(), dataset="toy", shots=2, seed=8)
        self.assertEqual(first.train_ids, second.train_ids)
        self.assertNotEqual(first.train_ids, third.train_ids)

    def test_too_many_shots_fails(self) -> None:
        with self.assertRaises(FewShotSplitError):
            make_few_shot_split(make_records(), dataset="toy", shots=6, seed=1)

    def test_all_split_uses_requested_val_and_test_ratios(self) -> None:
        records = [
            ImageRecord(
                dataset="toy",
                sample_id=f"toy/{label}/{idx}",
                image_path=f"{label}/{idx}.jpg",
                label_id=label,
                class_name=str(label),
                source_split="all",
            )
            for label in range(2)
            for idx in range(10)
        ]
        split = make_few_shot_split(
            records,
            dataset="toy",
            shots=1,
            seed=1,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        self.assertEqual(len(split.val_ids), 4)
        self.assertEqual(len(split.test_ids), 4)
        self.assertEqual(split.metadata["base_split"], "stratified_from_all")

    def test_split_json_round_trip(self) -> None:
        split = make_few_shot_split(make_records(), dataset="toy", shots=1, seed=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "split.json"
            write_split(split, path)
            loaded = read_split(path)
        self.assertEqual(split, loaded)


if __name__ == "__main__":
    unittest.main()
