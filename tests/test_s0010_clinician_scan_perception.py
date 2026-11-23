from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from evaluate_s0010_clinician_scan_perception import (
    binary_metrics,
    predicted_presence,
    read_dataset_manifest,
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

    def test_presence_does_not_count_a_nonwinning_class(self) -> None:
        probabilities = np.zeros((4, 3, 3), dtype=np.float32)
        probabilities[1, 1, 1] = 0.8
        probabilities[2, 1, 1] = 0.9
        thresholds = {
            name: {"threshold": 0.7, "minimum_pixels": 1}
            for name in ("TP", "SAP", "OTHER_BONE")
        }

        result = predicted_presence(probabilities, thresholds)

        self.assertFalse(result["tp_predicted"])
        self.assertTrue(result["sap_predicted"])

    def test_dataset_manifest_truth_is_scoped_to_capture_root(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            capture.mkdir()
            manifest = root / "manifest.csv"
            manifest.write_text(
                "source_root,session_id,frame_sequence,has_tp_interface,has_sap_interface\n"
                f"{capture},session,0,True,False\n"
                f"{root / 'other'},other,0,True,True\n",
                encoding="utf-8",
            )

            result = read_dataset_manifest(manifest, capture)

        self.assertEqual(set(result), {("session", 0)})
        self.assertEqual(result[("session", 0)]["both_supported"], "False")


if __name__ == "__main__":
    unittest.main()
