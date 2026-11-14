#!/usr/bin/env python3
"""CPU tests for the physical orientation-convergence result gate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.stage2_settling import (  # noqa: E402
    SettlingCriteria,
    evaluate_settled_frames,
)


def record(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "producer_sequence": 0,
        "technical_image_valid": True,
        "command_position_error_mm": 1.0,
        "command_orientation_error_deg": 0.5,
        "goal_position_error_mm": 1.0,
        "goal_orientation_error_deg": 8.5,
        "l4_visible_fraction": 0.10,
    }
    row.update(overrides)
    return row


class SettlingGateTests(unittest.TestCase):
    @staticmethod
    def records(count: int = 10) -> list[dict[str, object]]:
        return [record(producer_sequence=index + 1) for index in range(count)]

    def test_ten_jointly_valid_frames_pass(self) -> None:
        result = evaluate_settled_frames(self.records(), SettlingCriteria())
        self.assertEqual(result["decision"], "PASS")
        self.assertEqual(result["joint_pass_count"], 10)

    def test_one_orientation_failure_fails(self) -> None:
        rows = self.records()
        rows[-1] = record(producer_sequence=10, command_orientation_error_deg=1.01)
        result = evaluate_settled_frames(rows, SettlingCriteria())
        self.assertEqual(result["decision"], "FAIL")
        self.assertEqual(result["command_pose_pass_count"], 9)

    def test_goal_and_visibility_are_joint_requirements(self) -> None:
        rows = self.records()
        rows[0] = record(producer_sequence=1, goal_orientation_error_deg=10.01)
        rows[1] = record(producer_sequence=2, l4_visible_fraction=0.049)
        result = evaluate_settled_frames(rows, SettlingCriteria())
        self.assertEqual(result["decision"], "FAIL")
        self.assertEqual(result["goal_pose_pass_count"], 9)
        self.assertEqual(result["l4_visibility_pass_count"], 9)

    def test_missing_frame_fails(self) -> None:
        result = evaluate_settled_frames(self.records(9), SettlingCriteria())
        self.assertEqual(result["decision"], "FAIL")

    def test_repeated_producer_sequence_fails(self) -> None:
        rows = self.records()
        rows[-1]["producer_sequence"] = 9
        result = evaluate_settled_frames(rows, SettlingCriteria())
        self.assertEqual(result["decision"], "FAIL")
        self.assertFalse(result["producer_sequence_strictly_increasing"])


if __name__ == "__main__":
    unittest.main()
