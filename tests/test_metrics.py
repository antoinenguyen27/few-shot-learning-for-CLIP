from __future__ import annotations

import unittest

from common.evaluation.metrics import (
    accuracy,
    base_new_harmonic_mean,
    grouped_accuracy,
    macro_accuracy,
    mean,
    per_class_accuracy,
)


class MetricsTests(unittest.TestCase):
    def test_accuracy(self) -> None:
        self.assertEqual(accuracy([1, 0, 1], [1, 1, 1]), 2 / 3)

    def test_per_class_accuracy(self) -> None:
        self.assertEqual(per_class_accuracy([0, 1, 1, 0], [0, 1, 0, 0]), {0: 2 / 3, 1: 1.0})

    def test_macro_accuracy(self) -> None:
        self.assertEqual(macro_accuracy([0, 1, 1, 0], [0, 1, 0, 0]), (2 / 3 + 1.0) / 2)

    def test_grouped_accuracy(self) -> None:
        self.assertEqual(
            grouped_accuracy([0, 1, 1, 0], [0, 1, 0, 0], {"base": [0], "new": [1]}),
            {"base": 2 / 3, "new": 1.0},
        )

    def test_base_new_harmonic_mean(self) -> None:
        self.assertEqual(base_new_harmonic_mean(0.5, 1.0), 2 * 0.5 * 1.0 / 1.5)

    def test_mean(self) -> None:
        self.assertEqual(mean([1.0, 2.0, 3.0]), 2.0)


if __name__ == "__main__":
    unittest.main()
