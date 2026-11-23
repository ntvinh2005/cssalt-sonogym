from __future__ import annotations

import unittest

import numpy as np

from evaluate_s0010_clinician_scan_perception import (
    binary_metrics,
    predicted_presence,
)


class ClinicianScanPerceptionTests(unittest.TestCase):
    def test_binary_metrics_report_explicit_counts(self) -> None:
        result = binary_metrics(
            np.asarray([True, True, False, False]),
            np.asarray([True, False, True, False]),
        )
        self.assertEqual(
            {name: result[name] for name in ("tp", "fp", "fn", "tn")},
            {"tp": 1, "fp": 1, "fn": 1, "tn": 1},
        )
        self.assertEqual(result["accepted_count"], 2)
        self.assertEqual(result["total_count"], 4)

    def test_presence_uses_frozen_threshold_and_minimum_pixel_count(self) -> None:
        probabilities = np.zeros((4, 5, 5), dtype=np.float32)
        probabilities[1, 1, 1:3] = 0.8
        probabilities[2, 2, 2] = 0.9
        thresholds = {
            name: {"threshold": 0.7, "minimum_pixels": 2}
            for name in ("TP", "SAP", "OTHER_BONE")
        }
        result = predicted_presence(probabilities, thresholds)
        self.assertTrue(result["tp_predicted"])
        self.assertFalse(result["sap_predicted"])
        self.assertEqual(result["tp_predicted_pixels"], 2)
        self.assertEqual(result["sap_predicted_pixels"], 1)


if __name__ == "__main__":
    unittest.main()
