#!/usr/bin/env python3
"""Focused tests for the session-separated L4 image-only verifier."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "workflows/two_stage_navigation/run_s0010_l4_boundary_verifier.py"
SPEC = importlib.util.spec_from_file_location("l4_boundary_verifier", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFIER
SPEC.loader.exec_module(VERIFIER)


def row(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "projection_status": "VERIFIED",
        "l4_core_points_in_patch": "0",
        "l5_core_points_in_patch": "0",
        "l4_uncertain_points_in_patch": "0",
        "l5_uncertain_points_in_patch": "0",
        "adequate_projected_support": "False",
    }
    value.update(updates)
    return value


class BoundaryVerifierTests(unittest.TestCase):
    def test_ambiguous_is_not_negative(self) -> None:
        self.assertEqual(
            VERIFIER.row_state(row(l4_uncertain_points_in_patch="1")),
            VERIFIER.AMBIGUOUS,
        )
        self.assertEqual(
            VERIFIER.row_state(
                row(l4_core_points_in_patch="10", adequate_projected_support="False")
            ),
            VERIFIER.AMBIGUOUS,
        )

    def test_complete_positive_and_clear_negative_are_distinct(self) -> None:
        self.assertEqual(
            VERIFIER.row_state(
                row(l4_core_points_in_patch="10", adequate_projected_support="True")
            ),
            VERIFIER.POSITIVE,
        )
        self.assertEqual(VERIFIER.row_state(row()), VERIFIER.NEGATIVE)

    def test_competing_l5_does_not_become_l4_positive(self) -> None:
        self.assertEqual(
            VERIFIER.row_state(
                row(
                    l4_core_points_in_patch="10",
                    l5_core_points_in_patch="5",
                    adequate_projected_support="True",
                )
            ),
            VERIFIER.AMBIGUOUS,
        )
        self.assertEqual(
            VERIFIER.row_state(row(l5_core_points_in_patch="5")),
            VERIFIER.NEGATIVE,
        )

    def test_split_isolation_rejects_session_and_hash_leakage(self) -> None:
        with self.assertRaisesRegex(ValueError, "session crosses"):
            VERIFIER.validate_split_isolation(
                [
                    {"session_id": "s", "split": "train", "frame_sha256": "a"},
                    {"session_id": "s", "split": "calibration", "frame_sha256": "b"},
                ]
            )
        with self.assertRaisesRegex(ValueError, "image hash crosses"):
            VERIFIER.validate_split_isolation(
                [
                    {"session_id": "a", "split": "train", "frame_sha256": "x"},
                    {"session_id": "b", "split": "calibration", "frame_sha256": "x"},
                ]
            )

    def test_fit_rejects_nontraining_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-training"):
            VERIFIER.fit_verifier(
                [{"split": "calibration", "sample_id": "x", "proxy_state": VERIFIER.POSITIVE}],
                {"x": __import__("numpy").zeros(3)},
                {"n_estimators": 1},
            )

    def test_threshold_selection_preserves_recall_and_reduces_ambiguity(self) -> None:
        baseline = {"tp": 10, "fp": 2, "selected_ambiguous_images": 4}
        candidates = {
            0.5: {"tp": 9, "fp": 2, "selected_ambiguous_images": 3, "selected_positive_images": 9},
            0.6: {"tp": 8, "fp": 0, "selected_ambiguous_images": 0, "selected_positive_images": 8},
        }
        gate = {
            "minimum_positive_recall_retention_fraction": 0.9,
            "maximum_proxy_negative_false_positive_delta": 0,
            "minimum_ambiguous_selection_reduction": 1,
        }
        self.assertEqual(VERIFIER.choose_threshold(baseline, candidates, gate), 0.5)

    def test_training_selection_uses_train_only_and_one_row_per_image_state(self) -> None:
        rows = [
            {"split": "train", "frame_sha256": "a", "proxy_state": VERIFIER.POSITIVE,
             "primary_accepted": True, "primary_probability": 0.6, "sample_id": "low"},
            {"split": "train", "frame_sha256": "a", "proxy_state": VERIFIER.POSITIVE,
             "primary_accepted": True, "primary_probability": 0.8, "sample_id": "high"},
            {"split": "train", "frame_sha256": "b", "proxy_state": VERIFIER.NEGATIVE,
             "primary_accepted": True, "primary_probability": 0.7, "sample_id": "n"},
            {"split": "train", "frame_sha256": "c", "proxy_state": VERIFIER.AMBIGUOUS,
             "primary_accepted": True, "primary_probability": 0.7, "sample_id": "u"},
            {"split": "calibration", "frame_sha256": "d", "proxy_state": VERIFIER.POSITIVE,
             "primary_accepted": True, "primary_probability": 0.9, "sample_id": "cal"},
        ]
        selected = VERIFIER.select_training_rows(rows)
        self.assertEqual({item["sample_id"] for item in selected}, {"high", "n", "u"})

    def test_runtime_feature_path_has_no_geometry_inputs(self) -> None:
        import inspect

        source = inspect.getsource(VERIFIER.decisions_by_image)
        self.assertNotIn("geometry", source)
        self.assertNotIn("uncertain_points", source)

    def test_missing_proposal_is_not_a_negative_candidate(self) -> None:
        frame = {
            "proposal_row": None,
            "proposal_column": None,
            "proposal_track_id": None,
        }
        self.assertFalse(VERIFIER.proposal_available(frame))
        self.assertTrue(
            VERIFIER.proposal_available(
                {"proposal_row": 100.0, "proposal_column": 60.0, "proposal_track_id": "H0"}
            )
        )

    def test_proposal_metrics_keep_ambiguity_separate(self) -> None:
        metrics = VERIFIER.summarize_proposal_decisions(
            [
                {"proxy_state": VERIFIER.POSITIVE, "predicted_positive": True,
                 "positive_probability": 0.8},
                {"proxy_state": VERIFIER.NEGATIVE, "predicted_positive": True,
                 "positive_probability": 0.7},
                {"proxy_state": VERIFIER.AMBIGUOUS, "predicted_positive": True,
                 "positive_probability": 0.9},
            ]
        )
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp_on_clear_negative"], 1)
        self.assertEqual(metrics["ambiguous_activation_count"], 1)
        self.assertEqual(metrics["precision_excluding_ambiguous"], 0.5)

    def test_session_metrics_do_not_pool_sessions(self) -> None:
        records = [
            {"session_id": "a", "truth_state": VERIFIER.POSITIVE,
             "predicted_positive": True, "selected_state": VERIFIER.POSITIVE,
             "positive_available": True, "top3_primary_contains_positive": True},
            {"session_id": "b", "truth_state": VERIFIER.NEGATIVE,
             "predicted_positive": True, "selected_state": VERIFIER.AMBIGUOUS,
             "positive_available": False, "top3_primary_contains_positive": False},
        ]
        by_session = VERIFIER.image_metrics_by_session(records)
        self.assertEqual(by_session["a"]["tp"], 1)
        self.assertEqual(by_session["b"]["fp"], 1)

    def test_verifier_cannot_revive_primary_rejected_proposal(self) -> None:
        self.assertFalse(
            VERIFIER.combined_verifier_pass(
                primary_probability=0.49,
                primary_threshold=0.50,
                verifier_prediction=VERIFIER.STATE_ID[VERIFIER.POSITIVE],
                verifier_probability=0.99,
                verifier_threshold=0.65,
            )
        )
        self.assertTrue(
            VERIFIER.combined_verifier_pass(
                primary_probability=0.50,
                primary_threshold=0.50,
                verifier_prediction=VERIFIER.STATE_ID[VERIFIER.POSITIVE],
                verifier_probability=0.65,
                verifier_threshold=0.65,
            )
        )


if __name__ == "__main__":
    unittest.main()
