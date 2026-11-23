from __future__ import annotations

import unittest

import numpy as np

import prepare_s0010_clinician_scan as scan


class ClinicianScanTests(unittest.TestCase):
    def test_longitudinal_sweep_moves_laterally_at_fixed_level(self) -> None:
        poses = scan.longitudinal_sweep(np.asarray([140.0, 130.0, 2.0]))
        self.assertEqual(len(poses), 8)
        self.assertTrue(all(row[1] == 130.0 for row in poses))
        self.assertTrue(
            all(np.isclose(row[2], scan.LONG_AXIS_ANGLE_RAD) for row in poses)
        )
        steps_mm = [
            np.linalg.norm(after[:2] - before[:2]) * scan.LABEL_RESOLUTION_MM
            for before, after in zip(poses, poses[1:])
        ]
        self.assertTrue(
            all(value > scan.CAPTURE_POSITION_SEPARATION_MM for value in steps_mm)
        )
        scan.validate_sequence(poses)

    def test_transition_moves_forward_and_then_reverses(self) -> None:
        def evaluator(pose: np.ndarray) -> dict[str, int]:
            return {"tp_first_visible_pixels": 20 if pose[1] <= 130.0 else 0}

        poses = scan.cranial_caudal_transition(
            np.asarray([140.0, 130.0, 2.95]), evaluator
        )
        self.assertEqual(len(poses), 8)
        self.assertGreater(poses[4][1], poses[0][1])
        self.assertLess(poses[-1][1], poses[4][1])
        self.assertFalse(np.isclose(poses[5][0], poses[4][0]))
        scan.validate_pose_bank(poses)
        scan.validate_sequence(poses)

    def test_alternative_views_never_use_untrackable_ceiling(self) -> None:
        poses = scan.alternative_view_sweep(np.asarray([140.0, 130.0, 2.95]))
        self.assertTrue(all(row[2] < 2.99 for row in poses))
        self.assertGreater(max(row[2] for row in poses) - min(row[2] for row in poses), 1.3)
        scan.validate_pose_bank(poses)
        scan.validate_sequence(poses)

    def test_combined_bank_has_no_exact_pose_reuse(self) -> None:
        def evaluator(pose: np.ndarray) -> dict[str, int]:
            return {"tp_first_visible_pixels": 20 if pose[1] <= 130.0 else 0}

        longitudinal = scan.longitudinal_sweep(np.asarray([140.0, 120.0, 2.0]))
        transition = scan.cranial_caudal_transition(
            np.asarray([150.0, 130.0, 2.95]), evaluator
        )
        alternative = scan.alternative_view_sweep(
            np.asarray([150.0, 130.0, 2.95])
        )
        combined = longitudinal + transition + alternative
        scan.validate_pose_bank(combined)
        self.assertEqual(len(combined), len({tuple(row) for row in combined}))

    def test_anchor_selection_requires_spatial_diversity(self) -> None:
        rows = [
            {
                "pose": np.asarray([x, 130.0, scan.TRANSVERSE_ANGLE_RAD]),
                "sap": 10,
                "tp": 10,
                "margin": 10,
            }
            for x in (130.0, 131.0, 134.0, 140.0)
        ]
        selected = scan.distinct_anchor_rows(
            rows, angle_rad=scan.TRANSVERSE_ANGLE_RAD, count=2
        )
        distance = np.linalg.norm(selected[0]["pose"][:2] - selected[1]["pose"][:2]) * 1.5
        self.assertGreaterEqual(distance, 6.0)


if __name__ == "__main__":
    unittest.main()
