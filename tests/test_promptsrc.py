from __future__ import annotations

import unittest

from Promptsrc.promptsrc.config import PromptSRCConfig, gaussian_epoch_weights


class PromptSRCTests(unittest.TestCase):
    def test_gaussian_epoch_weights_are_normalized(self) -> None:
        weights = gaussian_epoch_weights(epochs=5, mean=3, std=1)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertEqual(max(range(len(weights)), key=lambda index: weights[index]), 2)

    def test_config_rejects_unsupported_vision_prompts(self) -> None:
        with self.assertRaises(ValueError):
            PromptSRCConfig(trainable_vision_prompts=True)

    def test_config_to_dict(self) -> None:
        config = PromptSRCConfig(epochs=2, seed=7)
        payload = config.to_dict()
        self.assertEqual(payload["epochs"], 2)
        self.assertEqual(payload["seed"], 7)
        self.assertTrue(payload["show_progress"])


if __name__ == "__main__":
    unittest.main()
