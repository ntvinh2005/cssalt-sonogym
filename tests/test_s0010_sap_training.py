"""Focused tests for SAP calibration gates and binary label handling."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
STAGE1_PACKAGES = PROJECT.parent / "cssalt-sonogym-runtime/envs/stage1-python"
if STAGE1_PACKAGES.exists():
    sys.path.insert(0, str(STAGE1_PACKAGES))
sys.path.insert(0, str(PROJECT / "workflows/two_stage_navigation"))
import train_s0010_sap_classifier as training  # noqa: E402


class SapTrainingTests(unittest.TestCase):
    def test_perfect_calibration_selects_a_passing_threshold(self) -> None:
        labels = np.asarray([1, 1, 0, 0])
        scores = np.asarray([0.9, 0.8, 0.2, 0.1])
        threshold, rows = training.select_threshold(labels, scores)
        self.assertIsNotNone(threshold)
        selected = training.metrics(labels, scores, float(threshold))
        self.assertTrue(selected["gate_pass"])
        self.assertTrue(any(row["gate_pass"] for row in rows))

    def test_reject_all_cannot_pass_by_precision_alone(self) -> None:
        labels = np.asarray([1, 1, 0, 0])
        scores = np.zeros(4)
        threshold, _ = training.select_threshold(labels, scores)
        self.assertIsNone(threshold)

    def test_only_two_binary_labels_are_supported(self) -> None:
        self.assertEqual(
            training.LABEL_ID,
            {
                training.NEGATIVE: 0,
                training.POSITIVE: 1,
            },
        )
        self.assertNotIn("BOUNDARY_OR_IDENTITY_AMBIGUOUS", training.LABEL_ID)

    def test_full_image_score_uses_all_actor_visible_proposals(self) -> None:
        class Estimator:
            classes_ = np.asarray([0, 1])

            def predict_proba(self, values: np.ndarray) -> np.ndarray:
                scores = values[:, 0]
                return np.column_stack((1.0 - scores, scores))

        sample_rows = [
            {
                "frame_sha256": "positive",
                "conditioned_level": "L4",
                "split": "calibration",
                "provisional_label": training.POSITIVE,
            },
            {
                "frame_sha256": "negative",
                "conditioned_level": "L4",
                "split": "calibration",
                "provisional_label": training.NEGATIVE,
            },
        ]
        proposal_rows = [
            {
                "frame_sha256": "positive",
                "conditioned_level": "L4",
                "split": "calibration",
                "proposal_feature_index": str(index),
            }
            for index in (0, 1)
        ] + [
            {
                "frame_sha256": "negative",
                "conditioned_level": "L4",
                "split": "calibration",
                "proposal_feature_index": "2",
            }
        ]
        features = np.zeros((3, 774))
        features[:, 0] = [0.2, 0.9, 0.1]
        labels, scores, hashes = training.full_image_scores(
            estimator=Estimator(),
            level="L4",
            split="calibration",
            sample_rows=sample_rows,
            proposal_rows=proposal_rows,
            proposal_features=features,
        )
        self.assertEqual(hashes, ["negative", "positive"])
        np.testing.assert_array_equal(labels, [0, 1])
        np.testing.assert_allclose(scores, [0.1, 0.9])


if __name__ == "__main__":
    unittest.main()
