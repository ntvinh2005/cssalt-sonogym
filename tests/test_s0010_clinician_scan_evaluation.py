from __future__ import annotations

import unittest

from evaluate_s0010_clinician_scan import (
    longitudinal_transition_evidence,
    transition_evidence,
)


class ClinicianScanEvaluationTests(unittest.TestCase):
    @staticmethod
    def rows(
        tp_pixels: list[int],
        *,
        shadows: list[float] | None = None,
        deep: list[float] | None = None,
    ) -> list[dict[str, object]]:
        z = [100.0, 103.0, 106.0, 109.0, 112.0, 109.0, 106.0, 103.0]
        shadows = shadows or [0.4, 0.4, 0.4, 0.6, 0.6, 0.5, 0.4, 0.4]
        deep = deep or [0.2, 0.2, 0.2, 0.1, 0.1, 0.15, 0.2, 0.2]
        return [
            {
                "tp_interface_pixels": pixels,
                "realized_pose_index": [130.0 + (3.0 if index >= 5 else 0.0), z[index], 2.95],
                "acquisition_id": f"acquisition-{index}",
                "frame_sha256": f"hash-{index}",
                "producer_sequence": 100 + index,
                "contact_ok": True,
                "image_valid": True,
                "action_accepted": True,
                "posterior_shadow_fraction": shadows[index],
                "deep_window_visibility": deep[index],
            }
            for index, pixels in enumerate(tp_pixels)
        ]

    def test_transition_requires_disappearance_and_reverse_reappearance(self) -> None:
        result = transition_evidence(self.rows([8, 7, 5, 0, 0, 0, 4, 6]))
        self.assertTrue(result["passed"])
        self.assertEqual(result["disappearance_frame_in_trajectory"], 3)

    def test_always_visible_does_not_pass(self) -> None:
        self.assertFalse(
            transition_evidence(self.rows([8, 7, 6, 5, 4, 5, 6, 7]))["passed"]
        )

    def test_disappearance_without_return_does_not_pass(self) -> None:
        self.assertFalse(
            transition_evidence(self.rows([8, 7, 0, 0, 0, 0, 0, 0]))["passed"]
        )

    def test_transition_requires_actor_visible_morphology(self) -> None:
        rows = self.rows(
            [8, 7, 5, 0, 0, 0, 4, 6],
            shadows=[0.4, 0.4, 0.4, 0.35, 0.35, 0.4, 0.5, 0.5],
            deep=[0.1, 0.1, 0.1, 0.2, 0.2, 0.15, 0.1, 0.1],
        )
        result = transition_evidence(rows)
        self.assertTrue(result["privileged_tp_pattern_passed"])
        self.assertFalse(result["actor_morphology_passed"])
        self.assertFalse(result["passed"])

    def test_transition_requires_contact_and_fresh_acquisition(self) -> None:
        rows = self.rows([8, 7, 5, 0, 0, 0, 4, 6])
        rows[4]["contact_ok"] = False
        rows[6]["acquisition_id"] = rows[5]["acquisition_id"]
        result = transition_evidence(rows)
        self.assertFalse(result["freshness_passed"])
        self.assertFalse(result["technical_validity_passed"])
        self.assertFalse(result["passed"])

    def test_transition_requires_cranial_then_caudal_motion(self) -> None:
        rows = self.rows([8, 7, 5, 0, 0, 0, 4, 6])
        rows[6]["realized_pose_index"] = [133.0, 115.0, 2.95]
        result = transition_evidence(rows)
        self.assertFalse(result["reverse_caudal_motion_passed"])
        self.assertFalse(result["passed"])

    def test_longitudinal_transition_requires_ordered_regions_and_motion(self) -> None:
        rows = self.rows([8, 6, 0, 0, 5, 7, 0, 0])
        x = [140.0, 143.0, 146.0, 146.0, 143.0, 140.0, 137.0, 134.0]
        z = [130.0, 130.0, 130.0, 133.0, 133.0, 133.0, 133.0, 133.0]
        for index, row in enumerate(rows):
            row["realized_pose_index"] = [x[index], z[index], 1.57]
            row["sap_interface_pixels"] = 6 if index >= 6 else 0
        rows[0]["posterior_shadow_fraction"] = 0.3
        rows[2]["posterior_shadow_fraction"] = 0.5
        rows[5]["deep_window_visibility"] = 0.2
        rows[6]["deep_window_visibility"] = 0.1
        result = longitudinal_transition_evidence(rows)
        self.assertTrue(result["longitudinal_transition_passed"])

    def test_longitudinal_transition_rejects_missing_ap_region(self) -> None:
        rows = self.rows([8, 6, 0, 0, 5, 7, 0, 0])
        x = [140.0, 143.0, 146.0, 146.0, 143.0, 140.0, 137.0, 134.0]
        for index, row in enumerate(rows):
            row["realized_pose_index"] = [x[index], 130.0 + (3.0 if index >= 3 else 0.0), 1.57]
            row["sap_interface_pixels"] = 0
        result = longitudinal_transition_evidence(rows)
        self.assertFalse(result["longitudinal_medial_ap_candidate_supported"])
        self.assertFalse(result["longitudinal_transition_passed"])


if __name__ == "__main__":
    unittest.main()
