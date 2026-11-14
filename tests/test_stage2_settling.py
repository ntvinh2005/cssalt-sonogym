#!/usr/bin/env python3
"""CPU tests for the physical orientation-convergence result gate."""

from __future__ import annotations

import sys
import unittest
import hashlib
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
sys.path.insert(0, str(EVALUATION_ROOT))
PROTOCOL = (
    REPOSITORY_ROOT
    / "experiments"
    / "unity_two_stage_transfer"
    / "stage2_orientation_smoke_protocol_v1.yaml"
)
PROTOCOL_V2 = PROTOCOL.with_name("stage2_orientation_smoke_protocol_v2.yaml")

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


class SettlingProtocolTests(unittest.TestCase):
    def test_protocol_is_sealed_and_uses_the_approved_resources(self) -> None:
        expected = PROTOCOL.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), expected)
        protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
        resources = protocol["resource_bound"]
        self.assertEqual(resources["account"], "anest-dept")
        self.assertEqual(resources["gpu_count"], 1)
        self.assertEqual(resources["cpu_cores"], 4)
        self.assertEqual(resources["memory"], "30gb")
        self.assertEqual(resources["walltime"], "00:15:00")

    def test_protocol_preserves_the_scientific_gate_and_forbids_training(self) -> None:
        protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
        criteria = protocol["criteria"]
        self.assertEqual(criteria["required_frame_count"], 10)
        self.assertEqual(criteria["maximum_goal_position_error_mm"], 10.0)
        self.assertEqual(criteria["maximum_goal_orientation_error_deg"], 10.0)
        self.assertEqual(criteria["minimum_l4_visible_fraction"], 0.05)
        self.assertEqual(protocol["scope"]["model_training"], "forbidden")
        self.assertEqual(protocol["scope"]["dataset_recapture"], "forbidden")
        forbidden = set(protocol["information_boundaries"]["actor_record_forbidden"])
        self.assertIn("raw_labels", forbidden)
        self.assertIn("goal_cmd_pose", forbidden)

    def test_v2_changes_motion_sequence_but_not_acceptance_criteria(self) -> None:
        expected = PROTOCOL_V2.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V2.read_bytes()).hexdigest(), expected)
        first = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
        second = yaml.safe_load(PROTOCOL_V2.read_text(encoding="utf-8"))
        self.assertEqual(second["amendment"]["evidence_job"], 51274046)
        self.assertEqual(first["session"], second["session"])
        self.assertEqual(first["criteria"], second["criteria"])
        self.assertEqual(first["resource_bound"]["account"], second["resource_bound"]["account"])
        self.assertEqual(first["resource_bound"]["gpu_count"], second["resource_bound"]["gpu_count"])
        self.assertEqual(first["resource_bound"]["cpu_cores"], second["resource_bound"]["cpu_cores"])
        self.assertEqual(first["resource_bound"]["memory"], second["resource_bound"]["memory"])
        self.assertEqual(first["resource_bound"]["walltime"], second["resource_bound"]["walltime"])
        self.assertEqual(second["motion"]["maximum_orientation_steps"], 120)
        self.assertEqual(second["motion"]["maximum_positioning_steps"], 120)
        self.assertIn("single researcher-approved retry", second["amendment"]["retry_policy"])


if __name__ == "__main__":
    unittest.main()
