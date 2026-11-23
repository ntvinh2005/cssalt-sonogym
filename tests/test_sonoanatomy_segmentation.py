from __future__ import annotations

import unittest

import torch
import numpy as np

from two_stage_navigation.sonoanatomy_segmentation import (
    SmallUNet,
    interface_loss,
    normalized_ultrasound,
)
from workflows.two_stage_navigation.train_s0010_sonoanatomy_segmentation import (
    FRAME_MINIMUM_PIXELS,
    FRAME_THRESHOLDS,
    calibrate_pair,
    pair_frame_metrics,
    should_stop_training,
)


class SonoanatomySegmentationTests(unittest.TestCase):
    def test_image_only_model_preserves_original_shape(self) -> None:
        model = SmallUNet(base_channels=8)
        image = torch.rand(2, 1, 200, 150)
        output = model(image)
        self.assertEqual(tuple(output.shape), (2, 4, 200, 150))

    def test_loss_ignores_ambiguous_pixels_and_is_finite(self) -> None:
        logits = torch.randn(1, 4, 8, 7, requires_grad=True)
        target = torch.zeros(1, 8, 7, dtype=torch.long)
        target[:, 2, 3] = 2
        target[:, 4, 5] = 255
        loss = interface_loss(logits, target, torch.tensor([0.05, 1.0, 4.0, 1.0]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_normalization_is_per_image_and_bounded(self) -> None:
        image = torch.stack((torch.arange(16), torch.arange(16) + 100)).reshape(2, 4, 4).float()
        normalized = normalized_ultrasound(image)
        self.assertGreaterEqual(float(normalized.min()), 0.0)
        self.assertLessEqual(float(normalized.max()), 1.0)
        self.assertTrue(torch.allclose(normalized[0], normalized[1]))

    def test_pair_metrics_require_both_tp_and_sap(self) -> None:
        probabilities = np.zeros((2, 4, 4, 4), dtype=np.float32)
        targets = np.zeros((2, 4, 4), dtype=np.uint8)
        targets[0, 1, 1] = 1
        targets[0, 1, 2] = 2
        targets[1, 2, 1] = 1
        probabilities[0, 1, 1, 1] = 0.9
        probabilities[0, 2, 1, 2] = 0.9
        probabilities[1, 1, 2, 1] = 0.9
        thresholds = {
            "TP": {"threshold": 0.5, "minimum_pixels": 1},
            "SAP": {"threshold": 0.5, "minimum_pixels": 1},
        }
        result = pair_frame_metrics(probabilities, targets, thresholds)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["tn"], 1)

    def test_pair_calibration_enforces_pair_gates_without_development_data(self) -> None:
        probabilities = np.zeros((12, 4, 4, 4), dtype=np.float32)
        targets = np.zeros((12, 4, 4), dtype=np.uint8)
        for index in range(6):
            targets[index, 1, 1] = 1
            targets[index, 1, 2] = 2
            probabilities[index, 1, 1, 1] = 0.95
            probabilities[index, 2, 1, 2] = 0.95
        probabilities[6:, 1, 2, 1] = 0.55
        probabilities[6:, 2, 2, 2] = 0.55

        result = calibrate_pair(probabilities, targets)

        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["metrics"]["tp"], 6)
        self.assertEqual(result["metrics"]["fp"], 0)
        self.assertEqual(result["metrics"]["recall"], 1.0)
        self.assertGreaterEqual(result["thresholds"]["TP"]["threshold"], 0.6)
        self.assertGreaterEqual(result["thresholds"]["SAP"]["threshold"], 0.6)

    def test_pair_calibration_reports_when_no_candidate_meets_gates(self) -> None:
        probabilities = np.zeros((2, 4, 2, 2), dtype=np.float32)
        targets = np.zeros((2, 2, 2), dtype=np.uint8)
        targets[0, 0, 0] = 1
        targets[0, 0, 1] = 2
        probabilities[:, 1, :, :] = max(FRAME_THRESHOLDS)
        probabilities[:, 2, :, :] = max(FRAME_THRESHOLDS)
        self.assertGreaterEqual(max(FRAME_MINIMUM_PIXELS), 4)

        result = calibrate_pair(probabilities, targets)

        self.assertFalse(result["gate_passed"])

    def test_early_stopping_waits_for_foreground_learning_window(self) -> None:
        self.assertFalse(should_stop_training(completed_epochs=11, stale_epochs=10))
        self.assertTrue(should_stop_training(completed_epochs=20, stale_epochs=10))
        self.assertFalse(should_stop_training(completed_epochs=20, stale_epochs=9))


if __name__ == "__main__":
    unittest.main()
