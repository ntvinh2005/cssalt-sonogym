#!/usr/bin/env python3
"""CPU tests for Group A Stage 1 evaluation behavior."""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = REPOSITORY_ROOT / "workflows" / "two_stage_navigation"
sys.path.insert(0, str(WORKFLOW_ROOT))

from train_stage1_group_a import (  # noqa: E402
    calibrate_coordinate_offsets,
    evaluate_results,
    gate_pass,
)


CLASS_ORDER = ["sacrum", "S1", "L5", "L4", "L3", "L2", "L1"]


def protocol() -> dict:
    return {
        "evaluation": {
            "minimum_event_precision": 0.90,
            "minimum_event_recall": 0.90,
            "minimum_complete_map_session_rate": 0.80,
            "maximum_median_localization_error_mm": 10.0,
            "require_every_class_session_success_rate": 0.80,
            "maximum_wrong_actor_stops": 0,
        }
    }


def successful_session(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "ordered_map": True,
        "predictions": [
            {
                "anatomy": name,
                "actor_status": "SUCCESS",
                "localization_error_mm_evaluation_only": 2.0,
            }
            for name in CLASS_ORDER
        ],
    }


class GroupAEvaluationTests(unittest.TestCase):
    def test_v3_protocol_is_sealed_and_requires_complete_maps(self) -> None:
        path = (
            REPOSITORY_ROOT
            / "experiments"
            / "unity_two_stage_transfer"
            / "stage1_group_a_protocol_v3.yaml"
        )
        expected = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(observed, expected)
        specification = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(specification["resource_bound"]["gpu_count"], 0)
        self.assertEqual(specification["resource_bound"]["cpu_cores"], 4)
        self.assertEqual(
            specification["ordered_map"]["selection_mode"],
            "joint_ordered_path",
        )
        self.assertEqual(
            specification["spatial_map"].get(
                "passing_threshold_tie_break", "highest"
            ),
            "highest",
        )
        self.assertEqual(
            specification["evaluation"]["minimum_complete_map_session_rate"],
            1.0,
        )

    def test_v4_protocol_uses_lowest_passing_threshold_tie_break(self) -> None:
        path = (
            REPOSITORY_ROOT
            / "experiments"
            / "unity_two_stage_transfer"
            / "stage1_group_a_protocol_v4.yaml"
        )
        expected = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(observed, expected)
        specification = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(
            specification["spatial_map"]["passing_threshold_tie_break"],
            "lowest",
        )

    def test_v5_protocol_uses_event_support_and_anatomical_end_anchors(self) -> None:
        path = (
            REPOSITORY_ROOT
            / "experiments"
            / "unity_two_stage_transfer"
            / "stage1_group_a_protocol_v5.yaml"
        )
        expected = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(observed, expected)
        specification = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(
            specification["spatial_map"]["direction_support_radius_mm"],
            5.0,
        )
        self.assertTrue(
            specification["ordered_map"][
                "first_event_must_be_minimum_coordinate"
            ]
        )
        self.assertTrue(
            specification["ordered_map"][
                "last_event_must_be_maximum_coordinate"
            ]
        )

    def test_complete_ordered_maps_pass(self) -> None:
        results = [successful_session(f"session_{index}") for index in range(4)]
        metrics = evaluate_results(results, CLASS_ORDER, 10.0)
        self.assertEqual(metrics["correct_event_count"], 28)
        self.assertEqual(metrics["complete_map_session_count"], 4)
        self.assertTrue(gate_pass(metrics, protocol()))

    def test_wrong_stop_prevents_gate_pass(self) -> None:
        results = [successful_session(f"session_{index}") for index in range(4)]
        results[0]["predictions"][0]["localization_error_mm_evaluation_only"] = 20.0
        metrics = evaluate_results(results, CLASS_ORDER, 10.0)
        self.assertEqual(metrics["wrong_actor_stop_count"], 1)
        self.assertFalse(gate_pass(metrics, protocol()))

    def test_incomplete_map_is_not_counted_as_complete(self) -> None:
        results = [successful_session(f"session_{index}") for index in range(4)]
        results[0]["predictions"][0]["actor_status"] = "ABSTAIN"
        results[0]["predictions"][0]["localization_error_mm_evaluation_only"] = None
        metrics = evaluate_results(results, CLASS_ORDER, 10.0)
        self.assertEqual(metrics["abstention_count"], 1)
        self.assertEqual(metrics["complete_map_session_count"], 3)
        self.assertFalse(gate_pass(metrics, protocol()))

    def test_per_class_midrange_offset_minimizes_worst_development_error(self) -> None:
        results = []
        for index, coordinate in enumerate((-10.0, -20.0, -14.0, -16.0)):
            predictions = [
                {
                    "anatomy": name,
                    "actor_status": "SUCCESS",
                    "actor_coordinate_mm": coordinate if name == "L4" else 0.0,
                    "reference_coordinate_mm_evaluation_only": (
                        -15.0 if name == "L4" else 1.0
                    ),
                }
                for name in CLASS_ORDER
            ]
            results.append({"session_id": f"session_{index}", "predictions": predictions})
        offsets = calibrate_coordinate_offsets(
            results,
            CLASS_ORDER,
            "per_class_midrange_minimax_offset",
        )
        self.assertEqual(offsets["L4"], 0.0)
        self.assertEqual(offsets["sacrum"], 1.0)


if __name__ == "__main__":
    unittest.main()
