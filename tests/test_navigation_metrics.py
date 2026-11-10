#!/usr/bin/env python3
"""Focused CPU tests for Phase 5A navigation metrics."""

from __future__ import annotations

import math
import unittest

import numpy as np

from scripts.navigation_metrics import (
    EpisodeMetricTracker,
    SuccessThresholds,
    classify_completion,
    command_pose_errors,
    frontal_plane_position_error_mm,
    is_success,
    meters_to_millimeters,
    orientation_error_deg,
    world_to_patient_points,
    wrap_to_pi,
)


class NavigationMetricTests(unittest.TestCase):
    def test_meter_to_millimeter_conversion(self) -> None:
        np.testing.assert_allclose(
            meters_to_millimeters([0.0, 0.0015, 0.01]), [0.0, 1.5, 10.0]
        )

    def test_world_to_patient_frame_transform(self) -> None:
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        patient_point = np.array([2.0, 3.0, 4.0])
        origin = np.array([10.0, -2.0, 1.0])
        world_point = origin + rotation @ patient_point
        np.testing.assert_allclose(
            world_to_patient_points(world_point, origin, rotation), patient_point
        )

    def test_world_to_patient_rejects_nonrigid_matrix(self) -> None:
        with self.assertRaises(ValueError):
            world_to_patient_points(
                [0, 0, 0], [0, 0, 0], np.diag([1.0, 1.0, 2.0])
            )

    def test_frontal_plane_projection_ignores_normal_axis(self) -> None:
        error = frontal_plane_position_error_mm(
            [0.010, 99.0, 0.020], [0.0, -99.0, 0.0]
        )
        self.assertAlmostEqual(error, math.sqrt(10.0**2 + 20.0**2))

    def test_angle_wrap_around(self) -> None:
        self.assertAlmostEqual(float(wrap_to_pi(math.pi)), -math.pi)
        self.assertAlmostEqual(
            orientation_error_deg(math.radians(179), math.radians(-179)), 2.0
        )

    def test_native_command_pose_units(self) -> None:
        position_mm, orientation_deg = command_pose_errors(
            [103.0, 106.0, math.radians(10)],
            [100.0, 102.0, 0.0],
            label_resolution_m=0.0015,
        )
        self.assertAlmostEqual(position_mm, 7.5)
        self.assertAlmostEqual(orientation_deg, 10.0)

    def test_success_requires_position_and_orientation(self) -> None:
        threshold = SuccessThresholds(10.0, 10.0)
        self.assertTrue(is_success(10.0, 10.0, threshold))
        self.assertFalse(is_success(10.1, 9.0, threshold))
        self.assertFalse(is_success(9.0, 10.1, threshold))

    def test_reset_and_episode_final_extraction(self) -> None:
        tracker = EpisodeMetricTracker(
            [0.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            label_resolution_m=0.0015,
        )
        tracker.record(
            step=1,
            pose=[2.0, 0.0, 0.0],
            reward=0.1,
            observation_finite=True,
            transition_finite=True,
        )
        final = tracker.finalize(
            terminated=False,
            native_truncated=False,
            protocol_horizon_reached=True,
            out_of_bounds=False,
        )
        self.assertAlmostEqual(final["initial_position_error_mm"], 18.0)
        self.assertAlmostEqual(final["final_position_error_mm"], 15.0)
        self.assertIsNone(final["first_successful_step"])
        self.assertEqual(final["completion_reason"], "protocol_horizon")
        self.assertTrue(final["protocol_truncated"])

    def test_timeout_termination_and_truncation_classification(self) -> None:
        native_timeout = classify_completion(
            terminated=False,
            native_truncated=True,
            protocol_horizon_reached=True,
            out_of_bounds=False,
        )
        self.assertEqual(native_timeout["completion_reason"], "native_time_limit")
        self.assertFalse(native_timeout["protocol_truncated"])
        boundary = classify_completion(
            terminated=True,
            native_truncated=False,
            protocol_horizon_reached=False,
            out_of_bounds=True,
        )
        self.assertEqual(boundary["completion_reason"], "out_of_bounds")
        with self.assertRaises(ValueError):
            classify_completion(
                terminated=False,
                native_truncated=False,
                protocol_horizon_reached=False,
                out_of_bounds=True,
            )

    def test_time_limit_is_not_navigation_success(self) -> None:
        tracker = EpisodeMetricTracker(
            [20.0, 20.0, 1.0], [0.0, 0.0, 0.0], label_resolution_m=0.0015
        )
        tracker.record(
            step=1,
            pose=[20.0, 20.0, 1.0],
            reward=0.0,
            observation_finite=True,
            transition_finite=True,
        )
        final = tracker.finalize(
            terminated=False,
            native_truncated=True,
            protocol_horizon_reached=False,
            out_of_bounds=False,
        )
        self.assertFalse(final["final_success"])
        self.assertEqual(final["completion_reason"], "native_time_limit")

    def test_invalid_or_missing_state_handling(self) -> None:
        with self.assertRaises(ValueError):
            command_pose_errors(
                [1.0, 2.0], [1.0, 2.0, 3.0], label_resolution_m=0.0015
            )
        with self.assertRaises(ValueError):
            command_pose_errors(
                [1.0, np.nan, 3.0],
                [1.0, 2.0, 3.0],
                label_resolution_m=0.0015,
            )
        tracker = EpisodeMetricTracker(
            [0, 0, 0], [1, 1, 0], label_resolution_m=0.0015
        )
        with self.assertRaises(ValueError):
            tracker.record(
                step=2,
                pose=[0, 0, 0],
                reward=0,
                observation_finite=True,
                transition_finite=True,
            )


if __name__ == "__main__":
    unittest.main()
