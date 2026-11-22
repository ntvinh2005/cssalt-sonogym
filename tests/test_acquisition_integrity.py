#!/usr/bin/env python3
"""CPU tests for synchronized acquisition and waypoint integrity."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PATH = (
    Path(__file__).resolve().parents[1]
    / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/acquisition_integrity.py"
)
SPEC = importlib.util.spec_from_file_location("acquisition_integrity", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def view(**changes):
    values = {
        "acquisition_id": "session:reset:1",
        "session_id": "session",
        "reset_id": "reset",
        "acquisition_sequence": 1,
        "producer_sequence": 10,
        "realized_pose_index": np.asarray([100.0, 120.0, 2.8]),
        "frame_sha256": "a" * 64,
        "action_count": 2,
        "technical_image_valid": True,
    }
    values.update(changes)
    return MODULE.AcquiredView(**values)


class SnapshotTests(unittest.TestCase):
    def test_snapshot_copy_is_immutable_after_source_mutation(self) -> None:
        source = np.asarray([1.0, 2.0, 3.0])
        frozen = MODULE.immutable_array(source)
        source[:] = 9.0
        np.testing.assert_array_equal(frozen, [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            frozen[0] = 4.0

    def test_shifted_frame_pose_pairing_fails_join(self) -> None:
        with self.assertRaisesRegex(ValueError, "one acquisition identity"):
            MODULE.validate_acquisition_join(
                [{"acquisition_id": "a"}, {"acquisition_id": "b"}]
            )

    def test_privileged_mapping_cannot_enter_detector_input(self) -> None:
        MODULE.assert_detector_input_is_actor_visible({"pixels": [1, 2]})
        with self.assertRaisesRegex(ValueError, "privileged"):
            MODULE.assert_detector_input_is_actor_visible(
                {"pixels": [1, 2], "human_img_coords": [0, 1]}
            )

    def test_actor_6dof_record_accepts_robot_pose_and_joint_state(self) -> None:
        MODULE.validate_actor_6dof_record(
            {
                "actor_base_to_ee_position_m": [0.1, 0.2, 0.3],
                "actor_base_to_ee_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "actor_base_to_ee_rotation_columns": np.eye(3).tolist(),
                "actor_joint_position_rad": [0.0] * 7,
                "simulation_step": 12,
                "simulation_time_s": 0.1,
            }
        )

    def test_actor_6dof_record_rejects_nonrigid_rotation(self) -> None:
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            MODULE.validate_actor_6dof_record(
                {
                    "actor_base_to_ee_position_m": [0.1, 0.2, 0.3],
                    "actor_base_to_ee_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "actor_base_to_ee_rotation_columns": np.diag([1.0, 1.0, 2.0]),
                    "actor_joint_position_rad": [0.0] * 7,
                    "simulation_step": 12,
                    "simulation_time_s": 0.1,
                }
            )


class WaypointTests(unittest.TestCase):
    def test_waypoints_swallowed_by_tolerance_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "swallowed"):
            MODULE.validate_waypoint_bank(
                [[100.0, 100.0, 2.8], [102.0, 100.0, 2.81]],
                command_minimum=[90.0, 90.0, 2.0],
                command_maximum=[110.0, 110.0, 3.0],
                label_resolution_mm=1.5,
                position_tolerance_mm=2.0,
                orientation_tolerance_deg=1.0,
            )

    def test_post_clipping_duplicate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.validate_waypoint_bank(
                [[100.0, 100.0, 2.8], [120.0, 100.0, 2.8], [130.0, 100.0, 2.8]],
                command_minimum=[90.0, 90.0, 2.0],
                command_maximum=[110.0, 110.0, 3.0],
                label_resolution_mm=1.5,
                position_tolerance_mm=2.0,
                orientation_tolerance_deg=1.0,
            )

    def test_distinct_bank_passes_twice_tolerance_rule(self) -> None:
        result = MODULE.validate_waypoint_bank(
            [[100.0, 100.0, 2.8], [106.0, 100.0, 2.8], [106.0, 106.0, 2.9]],
            command_minimum=[90.0, 90.0, 2.0],
            command_maximum=[110.0, 110.0, 3.0],
            label_resolution_mm=1.5,
            position_tolerance_mm=2.0,
            orientation_tolerance_deg=1.0,
        )
        self.assertEqual(result.shape, (3, 3))

    def test_continuous_dataset_budget_reaches_episode_horizon(self) -> None:
        with self.assertRaisesRegex(ValueError, "reaches episode horizon"):
            MODULE.validate_segment_action_budget(
                expected_frame_count=50,
                trajectory_start_frames=[0],
                maximum_initial_positioning_steps=350,
                maximum_action_steps_per_waypoint=100,
                maximum_episode_steps=600,
            )

    def test_three_frame_segments_fit_episode_horizon(self) -> None:
        maximum = MODULE.validate_segment_action_budget(
            expected_frame_count=50,
            trajectory_start_frames=list(range(0, 50, 3)),
            maximum_initial_positioning_steps=350,
            maximum_action_steps_per_waypoint=100,
            maximum_episode_steps=600,
        )
        self.assertEqual(maximum, 550)

    def test_episode_length_override_is_explicit_and_positive(self) -> None:
        self.assertIsNone(MODULE.resolve_episode_length_s({}))
        self.assertEqual(
            MODULE.resolve_episode_length_s({"episode_length_s_override": 400}),
            400.0,
        )
        for value in (False, 0, float("nan"), "invalid"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MODULE.resolve_episode_length_s({"episode_length_s_override": value})


class FreshnessTests(unittest.TestCase):
    def test_first_frame_after_reset_is_fresh_without_cross_reset_comparison(self) -> None:
        record = {
            "action_steps_since_previous_frame": 25,
            "trajectory_frame_sequence": 0,
            "action_accepted": True,
            "action_completed": True,
            "image_valid": True,
            "physical_viewpoint_check": None,
        }
        self.assertTrue(MODULE.recorded_view_is_fresh(record))

    def test_first_frame_after_reset_requires_real_action_and_valid_image(self) -> None:
        record = {
            "action_steps_since_previous_frame": 0,
            "trajectory_frame_sequence": 0,
            "action_accepted": True,
            "action_completed": True,
            "image_valid": True,
            "physical_viewpoint_check": None,
        }
        self.assertFalse(MODULE.recorded_view_is_fresh(record))
        record["action_steps_since_previous_frame"] = 25
        record["image_valid"] = False
        self.assertFalse(MODULE.recorded_view_is_fresh(record))

    def test_noninitial_frame_requires_accepted_viewpoint_check(self) -> None:
        record = {
            "action_steps_since_previous_frame": 25,
            "trajectory_frame_sequence": 1,
            "action_accepted": True,
            "action_completed": True,
            "image_valid": True,
            "physical_viewpoint_check": None,
        }
        self.assertFalse(MODULE.recorded_view_is_fresh(record))
        record["physical_viewpoint_check"] = {"accepted": True}
        self.assertTrue(MODULE.recorded_view_is_fresh(record))

    def test_repeated_producer_cannot_create_new_view(self) -> None:
        result = MODULE.validate_new_view(
            view(),
            view(
                acquisition_id="session:reset:2",
                acquisition_sequence=2,
                producer_sequence=10,
                realized_pose_index=np.asarray([106.0, 120.0, 2.8]),
                frame_sha256="b" * 64,
            ),
            label_resolution_mm=1.5,
            minimum_translation_mm=2.0,
            minimum_orientation_deg=1.0,
        )
        self.assertFalse(result["accepted"])

    def test_stale_pose_is_rejected_even_with_new_hash(self) -> None:
        result = MODULE.validate_new_view(
            view(),
            view(
                acquisition_id="session:reset:2",
                acquisition_sequence=2,
                producer_sequence=11,
                frame_sha256="b" * 64,
                action_count=1,
            ),
            label_resolution_mm=1.5,
            minimum_translation_mm=2.0,
            minimum_orientation_deg=1.0,
        )
        self.assertFalse(result["physical_viewpoint_changed"])
        self.assertFalse(result["accepted"])

    def test_valid_moved_snapshot_is_accepted(self) -> None:
        result = MODULE.validate_new_view(
            view(),
            view(
                acquisition_id="session:reset:2",
                acquisition_sequence=2,
                producer_sequence=11,
                realized_pose_index=np.asarray([106.0, 120.0, 2.8]),
                frame_sha256="b" * 64,
            ),
            label_resolution_mm=1.5,
            minimum_translation_mm=2.0,
            minimum_orientation_deg=1.0,
        )
        self.assertTrue(result["accepted"])


if __name__ == "__main__":
    unittest.main()
