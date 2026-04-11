from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from common.datasets.manifest import class_names_from_records, read_manifest, summarize_records, write_manifest
from common.datasets.types import ImageRecord


class ManifestTests(unittest.TestCase):
    def test_jsonl_round_trip(self) -> None:
        records = [
            ImageRecord(
                dataset="toy",
                sample_id="toy/a",
                image_path="a.jpg",
                label_id=0,
                class_name="alpha",
                source_split="train",
            ),
            ImageRecord(
                dataset="toy",
                sample_id="toy/b",
                image_path="b.jpg",
                label_id=1,
                class_name="beta",
                source_split="test",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            write_manifest(records, path)
            loaded = read_manifest(path)
        self.assertEqual(records, loaded)

    def test_summarize_records(self) -> None:
        records = [
            ImageRecord("toy", "toy/a", "a.jpg", 0, "alpha", "train"),
            ImageRecord("toy", "toy/b", "b.jpg", 0, "alpha", "test"),
            ImageRecord("toy", "toy/c", "c.jpg", 1, "beta", "train"),
        ]
        summary = summarize_records(records)
        self.assertEqual(summary["num_records"], 3)
        self.assertEqual(summary["num_classes"], 2)
        self.assertEqual(summary["source_split_counts"], {"train": 2, "test": 1})

    def test_class_names_from_records(self) -> None:
        records = [
            ImageRecord("toy", "toy/a", "a.jpg", 0, "alpha", "train"),
            ImageRecord("toy", "toy/b", "b.jpg", 1, "beta", "test"),
        ]
        self.assertEqual(class_names_from_records(records), ["alpha", "beta"])


if __name__ == "__main__":
    unittest.main()
