#!/usr/bin/env python3
"""CPU tests for the Stage 2 active-acquisition pilot planning boundary."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.stage2_active_pilot import (  # noqa: E402
    offset_poses,
    phase1_handoff_pose,
    relative_pose_from_start,
    serpentine_offsets,
    transition_is_valid,
)


def actor_result(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema": "sonogym-stage1-confirmation-actor-v1",
        "training_or_threshold_selection_performed": False,
        "results": [
            {
                "session_id": "cf_00",
                "actor_status": "SUCCESS",
                "actor_coordinate_mm": -75.5,
                "declared_home": {"patient_x_index": 133.0, "angle_rad": 2.75},
            }
        ],
    }
    row.update(changes)
    return row


def action(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "finite": True,
        "surface_proxy_ok": True,
        "command_mapping_valid": True,
        "terminated": False,
        "truncated": False,
        "command_position_clamped": False,
        "command_angle_clamped": True,
    }
    row.update(changes)
    return row


class ActivePilotTests(unittest.TestCase):
    def test_phase1_handoff_uses_actor_coordinate_and_declared_home(self) -> None:
        pose = phase1_handoff_pose(
            actor_result(),
            session_id="cf_00",
            home_z_index=195.0,
            label_resolution_mm=1.5,
            stage2_angle_rad=2.85,
        )
        np.testing.assert_allclose(pose, [133.0, 144.66666666666666, 2.85])

    def test_phase1_handoff_rejects_evaluator_leakage(self) -> None:
        leaked = actor_result(goal_cmd_pose=[120.0, 140.0, 2.9])
        with self.assertRaisesRegex(ValueError, "privileged actor field"):
            phase1_handoff_pose(
                leaked,
                session_id="cf_00",
                home_z_index=195.0,
                label_resolution_mm=1.5,
                stage2_angle_rad=2.85,
            )

    def test_offsets_are_deterministic_and_do_not_mutate_center(self) -> None:
        center = np.asarray([10.0, 20.0, 2.8])
        rows = offset_poses(center, [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.1]])
        np.testing.assert_allclose(rows[0], [9.0, 20.0, 2.8])
        np.testing.assert_allclose(rows[1], [10.0, 21.0, 2.9])
        np.testing.assert_allclose(center, [10.0, 20.0, 2.8])

    def test_serpentine_search_has_expected_count_and_order(self) -> None:
        rows = serpentine_offsets([-2.0, 0.0], [-1.0, 1.0], [-0.1, 0.1])
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[:4], ((-2.0, -1.0, -0.1), (-2.0, -1.0, 0.1), (0.0, -1.0, 0.1), (0.0, -1.0, -0.1)))
        self.assertEqual(rows[4][0], 0.0)

    def test_relative_pose_uses_millimetres_and_wrapped_angle(self) -> None:
        pose = relative_pose_from_start([11.0, 18.0, -3.1], [10.0, 20.0, 3.1], 1.5)
        np.testing.assert_allclose(pose[:2], [1.5, -3.0])
        self.assertAlmostEqual(pose[2], (0.08318530717958605), places=12)

    def test_transition_allows_angle_only_but_rejects_position_and_safety_failures(self) -> None:
        self.assertTrue(transition_is_valid([action()]))
        for changes in (
            {"command_position_clamped": True},
            {"finite": False},
            {"surface_proxy_ok": False},
            {"command_mapping_valid": False},
            {"terminated": True},
            {"truncated": True},
        ):
            self.assertFalse(transition_is_valid([action(**changes)]))
        self.assertFalse(transition_is_valid([]))

    def test_nonfinite_plans_fail(self) -> None:
        with self.assertRaises(ValueError):
            offset_poses([0.0, 0.0, 0.0], [[0.0, math.nan, 0.0]])


if __name__ == "__main__":
    unittest.main()
