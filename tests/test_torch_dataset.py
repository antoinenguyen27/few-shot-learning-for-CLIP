from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - depends on local environment setup
    Image = None

from common.datasets.torch_dataset import ManifestImageDataset
from common.datasets.types import ImageRecord


class TorchDatasetTests(unittest.TestCase):
    @unittest.skipIf(Image is None, "Pillow is not installed in this environment")
    def test_output_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "a.jpg"
            Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_path)
            records = [ImageRecord("toy", "toy/a", "a.jpg", 2, "alpha", "train")]

            dict_item = ManifestImageDataset(records, root)[0]
            tuple_item = ManifestImageDataset(records, root, output_format="tuple")[0]
            dassl_item = ManifestImageDataset(records, root, output_format="dassl")[0]

        self.assertEqual(dict_item["label"], 2)
        self.assertEqual(tuple_item[1], 2)
        self.assertEqual(dassl_item["label"], 2)
        self.assertEqual(dassl_item["index"], 0)


if __name__ == "__main__":
    unittest.main()
