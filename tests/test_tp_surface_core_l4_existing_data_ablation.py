#!/usr/bin/env python3
"""CPU tests for the L4 existing-data ablation."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "workflows/two_stage_navigation/run_tp_surface_core_l4_existing_data_ablation.py"
SPEC = importlib.util.spec_from_file_location("l4_existing_data_ablation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ABLATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ABLATION
SPEC.loader.exec_module(ABLATION)


def selected_row(class_id: int, *, split: str = "train", frame: str = "frame"):
    return {
        "sample_id": f"sample-{class_id}-{frame}",
        "split": split,
        "proxy_class_id": str(class_id),
        "frame_sha256": frame,
    }


class FakeEstimator:
    classes_ = np.asarray([0, 1, 2])

    def __init__(self, values: list[list[float]]):
        self.values = np.asarray(values)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.values[: len(features)]


class ExistingDataAblationTests(unittest.TestCase):
    def test_l5_positive_is_l4_negative_in_one_vs_rest(self) -> None:
        arm = {"target": "l4_one_vs_rest", "hard_negatives": "exclude"}
        rows = ABLATION.build_training_rows(
            [selected_row(1, frame="l4"), selected_row(2, frame="l5")], [], arm
        )
        self.assertEqual([row["target"] for row in rows], [1, 0])

    def test_ambiguous_state_is_excluded_from_primary_metric(self) -> None:
        rows = [
            {"sample_id": "a", "split": "calibration", "frame_sha256": "a", "session_id": "s"},
            {"sample_id": "b", "split": "calibration", "frame_sha256": "b", "session_id": "s"},
        ]
        features = {"a": np.zeros(2), "b": np.zeros(2)}
        metrics, _ = ABLATION.evaluate_l4(
            rows,
            features,
            {"a": ABLATION.TRISTATE.PROXY_NEGATIVE, "b": ABLATION.TRISTATE.PROXY_AMBIGUOUS},
            FakeEstimator([[0.2, 0.7, 0.1], [0.2, 0.7, 0.1]]),
            0.5,
            target="three_class",
            split="calibration",
            arm_id="test",
        )
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["ambiguous_activation_images"], 1)

    def test_non_training_rows_cannot_reach_fit(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-training"):
            ABLATION.fit_arm(
                [{**selected_row(1, split="calibration"), "target": 1, "sample_weight": 1.0}],
                {"sample-1-frame": np.zeros(2)},
                {"n_estimators": 1, "random_state": 1, "n_jobs": 1},
            )

    def test_hard_negative_must_be_train_and_clear(self) -> None:
        arm = {
            "target": "l4_one_vs_rest",
            "hard_negatives": "include_as_l4_negative",
            "hard_negative_weight": 4.0,
        }
        hard = {
            "sample_id": "hard",
            "split": "train",
            "l4_proxy_state": ABLATION.TRISTATE.PROXY_NEGATIVE,
            "frame_sha256": "hard-frame",
        }
        rows = ABLATION.build_training_rows([], [hard], arm)
        self.assertEqual(rows[0]["target"], 0)
        self.assertEqual(rows[0]["sample_weight"], 4.0)
        with self.assertRaisesRegex(ValueError, "inadmissible"):
            ABLATION.build_training_rows(
                [], [{**hard, "l4_proxy_state": ABLATION.TRISTATE.PROXY_AMBIGUOUS}], arm
            )

    def test_multiclass_gate_requires_argmax(self) -> None:
        accepted, scores = ABLATION.l4_accepts(
            FakeEstimator([[0.6, 0.55, 0.0]]), np.zeros((1, 2)), 0.5, target="three_class"
        )
        self.assertFalse(bool(accepted[0]))
        self.assertEqual(float(scores[0]), 0.55)

    def test_candidate_selection_uses_calibration_only(self) -> None:
        protocol = {
            "arms": [{"arm_id": "a"}, {"arm_id": "b"}],
            "calibration_selection": {
                "minimum_true_positive_images": 9,
                "maximum_clean_negative_false_positive_images": 0,
            },
        }
        calibration = {
            "a": {"tp": 9, "fp": 0, "clean_negative_false_detections": 0},
            "b": {"tp": 10, "fp": 0, "clean_negative_false_detections": 0},
        }
        self.assertEqual(ABLATION.choose_candidate(calibration, protocol), "b")

    def test_source_does_not_include_privileged_feature_fields(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        feature_call = source[source.index("def load_features") : source.index("def assign_states")]
        for forbidden in ("core_points", "uncertain_points", "proxy_state"):
            self.assertNotIn(forbidden, feature_call)


if __name__ == "__main__":
    unittest.main()
