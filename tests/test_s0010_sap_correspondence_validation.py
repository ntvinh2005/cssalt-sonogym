"""Tests for session-separated SAP correspondence validation."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PATH = (
    Path(__file__).resolve().parents[1]
    / "workflows/two_stage_navigation/validate_s0010_sap_correspondence.py"
)
SPEC = importlib.util.spec_from_file_location("sap_correspondence_validation", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SapCorrespondenceValidationTests(unittest.TestCase):
    def test_bounded_estimators_share_the_binary_probability_contract(self) -> None:
        values = np.asarray(
            [
                [0.9] * len(MODULE.FEATURES),
                [0.8] * len(MODULE.FEATURES),
                [0.2] * len(MODULE.FEATURES),
                [0.1] * len(MODULE.FEATURES),
            ]
        )
        labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
        for name in ("logistic", "extra_trees"):
            estimator = MODULE.make_estimator(name)
            estimator.fit(values, labels)
            scores = estimator.predict_proba(values)
            self.assertEqual(scores.shape, (4, 2))
            self.assertTrue(np.all(np.isfinite(scores)))

    def test_training_threshold_requires_both_error_gates(self) -> None:
        labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
        scores = np.asarray([0.9, 0.8, 0.2, 0.1])
        threshold, rows = MODULE.select_threshold(labels, scores)
        self.assertIsNotNone(threshold)
        selected = MODULE.rates(labels, scores, float(threshold))
        self.assertGreaterEqual(selected["matched_acceptance_rate"], 0.5)
        self.assertLessEqual(selected["mismatched_acceptance_rate"], 0.1)
        self.assertTrue(rows)

    def test_top_one_is_grouped_by_source_frame(self) -> None:
        rows = [
            {"session_id": "a", "frame_sequence": "0", "comparison": "MATCHED"},
            {"session_id": "a", "frame_sequence": "0", "comparison": "DISTINCT_POSE"},
            {"session_id": "b", "frame_sequence": "0", "comparison": "MATCHED"},
            {"session_id": "b", "frame_sequence": "0", "comparison": "OTHER_LEVEL"},
        ]
        wins, total, rate = MODULE.top_one_rate(
            rows, np.asarray([0.8, 0.2, 0.3, 0.7])
        )
        self.assertEqual((wins, total, rate), (1, 2, 0.5))


if __name__ == "__main__":
    unittest.main()
