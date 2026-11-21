"""Tests for frozen SAP correspondence evaluation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflows/two_stage_navigation"))
import evaluate_s0010_sap_correspondence as evaluation  # noqa: E402


class Estimator:
    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        scores = values[:, 0]
        return np.column_stack((1.0 - scores, scores))


def row(session: str, frame: int, comparison: str, score: float) -> dict[str, str]:
    values = {name: "0" for name in evaluation.validation.FEATURES}
    values.update(
        session_id=session,
        frame_sequence=str(frame),
        comparison=comparison,
        projection_source=f"{session}:{frame}",
    )
    values[evaluation.validation.FEATURES[0]] = str(score)
    return values


class SapCorrespondenceEvaluationTests(unittest.TestCase):
    def test_frozen_threshold_passes_without_fitting(self) -> None:
        rows = []
        for frame in range(10):
            rows.append(row("session", frame, "MATCHED", 0.9))
            rows.append(row("session", frame, "OTHER_LEVEL", 0.1))
            rows.append(row("session", frame, "DISTINCT_POSE", 0.1))
        summary, scores = evaluation.evaluate_level(rows, Estimator(), 0.5)
        self.assertTrue(summary["calibration_gate_pass"])
        self.assertEqual(summary["calibration_top_one_wins"], 10)
        self.assertEqual(len(scores), 30)

    def test_distinct_pose_false_alignment_fails(self) -> None:
        rows = []
        for frame in range(10):
            rows.append(row("session", frame, "MATCHED", 0.9))
            rows.append(row("session", frame, "OTHER_LEVEL", 0.1))
            rows.append(row("session", frame, "DISTINCT_POSE", 0.8))
        summary, _ = evaluation.evaluate_level(rows, Estimator(), 0.5)
        self.assertFalse(summary["calibration_gate_pass"])


if __name__ == "__main__":
    unittest.main()
