"""Focused checks for the additional level-specific development capture."""

from __future__ import annotations

import unittest

import numpy as np

from workflows.two_stage_navigation.prepare_s0010_sonoanatomy_capture_v2 import (
    local_sweep,
    pose_is_valid,
    poses_are_separated,
)
from workflows.two_stage_navigation.prepare_s0010_sonoanatomy_hard_negative_capture import (
    choose_session_anchors,
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

    def test_hard_negative_selection_pairs_error_with_positive_controls(self) -> None:
        def row(
            pose: tuple[float, float, float],
            *,
            has_tp: bool,
            has_sap: bool,
            tp_pixels: int,
        ) -> dict[str, object]:
            value = np.asarray(pose, dtype=np.float64)
            return {
                "level": "L5",
                "pose": value,
                "sweep": local_sweep(value),
                "tp_predicted": True,
                "sap_predicted": True,
                "has_tp": has_tp,
                "has_sap": has_sap,
                "tp_predicted_pixels": tp_pixels,
                "sap_predicted_pixels": 20,
                "tp_pixels": int(has_tp) * 10,
                "sap_pixels": int(has_sap) * 10,
            }

        rows = [
            row((120.0, 90.0, 2.0), has_tp=False, has_sap=True, tp_pixels=30),
            row((140.0, 130.0, 2.3), has_tp=True, has_sap=True, tp_pixels=20),
            row((165.0, 170.0, 2.6), has_tp=True, has_sap=True, tp_pixels=15),
        ]

        selected = choose_session_anchors(rows, "L5", session_count=1)

        self.assertEqual(len(selected), 1)
        self.assertFalse(selected[0][0]["has_tp"])
        self.assertTrue(all(item["has_tp"] for item in selected[0][1:]))


if __name__ == "__main__":
    unittest.main()
