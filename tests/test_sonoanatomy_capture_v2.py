"""Focused checks for the additional level-specific development capture."""

from __future__ import annotations

import unittest

import numpy as np

from workflows.two_stage_navigation.prepare_s0010_sonoanatomy_capture_v2 import (
    local_sweep,
    pose_is_valid,
    poses_are_separated,
)


class SonoanatomyCaptureV2Test(unittest.TestCase):
    def test_local_sweep_is_valid_and_separated(self) -> None:
        poses = local_sweep(np.asarray([150.0, 130.0, 2.4]))
        self.assertEqual(len(poses), 8)
        self.assertTrue(all(pose_is_valid(pose) for pose in poses))
        self.assertTrue(
            all(poses_are_separated(first, second) for first, second in zip(poses, poses[1:]))
        )

    def test_high_angle_sweep_stays_below_controller_limit(self) -> None:
        poses = local_sweep(np.asarray([150.0, 130.0, 2.95]))
        self.assertLess(max(float(pose[2]) for pose in poses), 2.99)
        self.assertIn(2.95, [float(pose[2]) for pose in poses])

    def test_angle_steps_remain_distinct_after_opposed_tracking_errors(self) -> None:
        poses = local_sweep(np.asarray([150.0, 130.0, 2.4]))
        same_position = [
            np.degrees(abs(float(second[2] - first[2])))
            for first, second in zip(poses, poses[1:])
            if np.allclose(first[:2], second[:2])
        ]
        self.assertTrue(all(value - 2.0 * 1.2 > 1.0 for value in same_position))


if __name__ == "__main__":
    unittest.main()
