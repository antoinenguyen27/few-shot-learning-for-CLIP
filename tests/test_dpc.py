from __future__ import annotations

import unittest

from DPC.dpc.config import DPCConfig


class DPCTests(unittest.TestCase):
    def test_config_to_dict(self) -> None:
        config = DPCConfig(backbone_epochs=1, dpc_epochs=2, seed=7)
        payload = config.to_dict()
        self.assertEqual(payload["backbone_epochs"], 1)
        self.assertEqual(payload["dpc_epochs"], 2)
        self.assertEqual(payload["seed"], 7)
        self.assertTrue(payload["show_progress"])

    def test_config_rejects_invalid_stack_weight(self) -> None:
        with self.assertRaises(ValueError):
            DPCConfig(stack_weight=1.5)

    def test_config_rejects_zero_dpc_epochs(self) -> None:
        with self.assertRaises(ValueError):
            DPCConfig(dpc_epochs=0)


if __name__ == "__main__":
    unittest.main()
