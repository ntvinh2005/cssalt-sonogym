from __future__ import annotations

import unittest

import numpy as np

from two_stage_navigation.sonoanatomy_active_scan import (
    ClinicianSweepController,
    ScanConfig,
    ScanEvidence,
    ScanState,
    pair_confirmation_requirement_met,
    staged_orientation_targets,
)


class ClinicianSweepControllerTests(unittest.TestCase):
    def evidence(
        self,
        controller: ClinicianSweepController,
        frame: int,
        *,
        tp: bool,
        sap: bool = False,
        valid: bool = True,
    ) -> ScanEvidence:
        pose = controller.next_pose()
        assert pose is not None
        return ScanEvidence(
            frame_sequence=frame,
            pose_index=tuple(float(value) for value in pose),
            tp_predicted=tp,
            sap_predicted=sap,
            other_bone_predicted=False,
            tp_predicted_pixels=20 if tp else 0,
            sap_predicted_pixels=10 if sap else 0,
            image_valid=valid,
            contact_ok=True,
            producer_advanced=True,
        )

    def test_scan_is_conditioned_on_loss_return_and_sap_evidence(self) -> None:
        controller = ClinicianSweepController([100.0, 180.0, 2.9], ScanConfig())
        for frame in range(5):
            controller.observe(self.evidence(controller, frame, tp=frame in {1, 2, 3}))
        self.assertEqual(controller.state, ScanState.LONGITUDINAL_LATERAL_SURVEY)
        controller.observe(self.evidence(controller, 5, tp=True))
        controller.observe(self.evidence(controller, 6, tp=False))
        self.assertEqual(controller.state, ScanState.REVERSE_MEDIAL_SURVEY)
        controller.observe(self.evidence(controller, 7, tp=True))
        controller.observe(self.evidence(controller, 8, tp=True, sap=True))
        self.assertEqual(controller.state, ScanState.ACQUIRE_TRANSVERSE_VIEW)
        for frame in range(9, 17):
            controller.observe(self.evidence(controller, frame, tp=frame % 2 == 0))
        self.assertEqual(controller.state, ScanState.SAP_TP_IDENTITY_INFERENCE)
        transition = controller.transition_evidence()
        self.assertTrue(transition["tp_lost_laterally"])
        self.assertTrue(transition["tp_reappeared_medially"])
        self.assertTrue(transition["sap_after_tp_reappearance"])

    def test_invalid_acquisition_abstains(self) -> None:
        controller = ClinicianSweepController([100.0, 180.0, 2.9], ScanConfig())
        controller.observe(self.evidence(controller, 0, tp=True, valid=False))
        self.assertEqual(controller.state, ScanState.ABSTAIN)
        self.assertIsNone(controller.next_pose())

    def test_reference_search_uses_longitudinal_orientation(self) -> None:
        config = ScanConfig(longitudinal_angle_rad=1.7)
        controller = ClinicianSweepController([100.0, 180.0, 2.9], config)
        poses = []
        for frame in range(5):
            pose = controller.next_pose()
            assert pose is not None
            poses.append(pose)
            controller.observe(self.evidence(controller, frame, tp=frame == 2))
        self.assertTrue(all(np.isclose(row[2], 1.7) for row in poses))

    def test_large_orientation_change_is_split_without_translation(self) -> None:
        rows = staged_orientation_targets(
            [147.0, 142.0, 2.8],
            [141.0, 142.0, 1.57],
            maximum_step_rad=0.35,
        )
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(np.allclose(row[:2], [147.0, 142.0]) for row in rows))
        angles = [2.8, *(float(row[2]) for row in rows), 1.57]
        self.assertLessEqual(max(abs(right - left) for left, right in zip(angles, angles[1:])), 0.35)

    def test_pair_confirmation_accepts_two_of_three_fresh_views(self) -> None:
        rows = [
            {
                "frame_sha256": str(index),
                "image_valid": True,
                "producer_advanced": True,
                "tp_predicted": index != 1,
                "sap_predicted": True,
            }
            for index in range(3)
        ]
        self.assertTrue(
            pair_confirmation_requirement_met(
                rows,
                required_count=3,
                minimum_positive_count=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
