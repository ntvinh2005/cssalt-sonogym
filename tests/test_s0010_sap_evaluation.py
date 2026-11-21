"""Tests for held-out SAP readiness and ambiguity accounting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflows/two_stage_navigation"))
import evaluate_s0010_sap_classifier as evaluation  # noqa: E402


class Estimator:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        scores = values[:, 0]
        return np.column_stack((1.0 - scores, scores))


class SapEvaluationTests(unittest.TestCase):
    def test_insufficient_holdout_stays_inconclusive_despite_descriptive_metrics(self) -> None:
        metrics = {"gate_pass": True}
        self.assertEqual(
            evaluation.heldout_status(False, metrics),
            "SAP_TRAINING_INCONCLUSIVE",
        )
        self.assertEqual(
            evaluation.heldout_status(True, metrics),
            "SAP_TRAINED_HELDOUT_PASS",
        )

    def test_external_frozen_correspondence_can_resolve_only_that_deficit(self) -> None:
        readiness = {
            "readiness": {
                "L5": {
                    "deficits": ["matched/mismatched correspondence gate failed"]
                }
            }
        }
        external = {"levels": {"L5": {"calibration_gate_pass": True}}}
        self.assertEqual(
            evaluation.level_is_ready("L5", readiness, external), (True, [])
        )
        readiness["readiness"]["L5"]["deficits"].append(
            "evaluation:OTHER_OR_BACKGROUND=9<10"
        )
        ready, deficits = evaluation.level_is_ready("L5", readiness, external)
        self.assertFalse(ready)
        self.assertEqual(deficits, ["evaluation:OTHER_OR_BACKGROUND=9<10"])

    def test_ambiguous_frames_are_abstentions_not_negatives(self) -> None:
        sample_rows = [
            {
                "frame_sha256": "clear",
                "conditioned_level": "L5",
                "split": "evaluation",
            }
        ]
        proposal_rows = [
            {
                "frame_sha256": "clear",
                "session_id": "session",
                "conditioned_level": "L5",
                "split": "evaluation",
                "proposal_feature_index": "0",
            },
            {
                "frame_sha256": "ambiguous",
                "session_id": "session",
                "conditioned_level": "L5",
                "split": "evaluation",
                "proposal_feature_index": "1",
            },
        ]
        features = np.zeros((2, 774), dtype=np.float64)
        features[:, 0] = [0.1, 0.9]
        result = evaluation.ambiguity_summary(
            estimator=Estimator(),
            threshold=0.55,
            level="L5",
            split="evaluation",
            sample_rows=sample_rows,
            proposal_rows=proposal_rows,
            proposal_features=features,
        )
        self.assertEqual(result["clear_labeled_images"], 1)
        self.assertEqual(result["ambiguous_images"], 1)
        self.assertEqual(result["ambiguous_activations"], 1)
        self.assertEqual(result["abstention_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
