#!/usr/bin/env python3
"""CPU tests for Group A Stage 1 evaluation behavior."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = REPOSITORY_ROOT / "workflows" / "two_stage_navigation"
sys.path.insert(0, str(WORKFLOW_ROOT))

from train_stage1_group_a import evaluate_results, gate_pass  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
