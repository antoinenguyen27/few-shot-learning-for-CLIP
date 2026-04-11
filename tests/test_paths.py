from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from common.datasets.paths import manifest_path, raw_marker_path, resolve_raw_root, split_path


class PathTests(unittest.TestCase):
    def test_marker_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            marker = raw_marker_path("eurosat", data_root)
            marker.parent.mkdir(parents=True)
            marker.write_text("/tmp/example-eurosat\n", encoding="utf-8")
            self.assertEqual(resolve_raw_root("eurosat", data_root), Path("/tmp/example-eurosat").resolve())

    def test_manifest_and_split_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(manifest_path("flowers102", tmpdir).name, "flowers102.jsonl")
            path = split_path("flowers102", "few_shot_all_classes", 16, 1, tmpdir)
            self.assertEqual(
                path.parts[-4:],
                ("flowers102", "few_shot_all_classes", "shots_16", "seed_1.json"),
            )


if __name__ == "__main__":
    unittest.main()
