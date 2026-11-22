"""Checks for frozen SAP/TP pair confirmation pose selection."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from evaluate_s0010_sap_tp_pair_confirmation import target_margin
from prepare_s0010_sap_tp_pair_confirmation import (
    distinct,
    nearby,
    select_pairs,
)
from verify_exact_pose_capture import requires_synchronized_capture
from two_stage_navigation.coordinates import (
    is_supported_stage2_dataset_protocol,
    uses_coupled_stage2_dataset_controller,
    uses_segmented_stage2_dataset_capture,
)
from two_stage_navigation.acquisition_integrity import validate_waypoint_bank


ROOT = Path(__file__).parents[1]


class TestSapTpPairConfirmation(unittest.TestCase):
    def test_protocol_uses_segmented_coupled_capture(self) -> None:
        protocol_id = "sonogym-s0010-sap-tp-pair-confirmation-v1r1"
        self.assertTrue(is_supported_stage2_dataset_protocol(protocol_id))
        self.assertTrue(uses_coupled_stage2_dataset_controller(protocol_id))
        self.assertTrue(uses_segmented_stage2_dataset_capture(protocol_id))
        self.assertTrue(requires_synchronized_capture(protocol_id))

    def test_frozen_protocol_waypoints_pass_real_capture_validator(self) -> None:
        protocol = yaml.safe_load(
            (
                ROOT
                / "experiments/unity_two_stage_transfer/"
                "s0010_sap_tp_pair_confirmation_v1r1.yaml"
            ).read_text(encoding="utf-8")
        )
        sampling = protocol["sampling"]
        for bank in sampling["absolute_waypoint_banks_index"].values():
            resolved = validate_waypoint_bank(
                bank,
                command_minimum=[-1.0e9, -1.0e9, -1.0e9],
                command_maximum=[1.0e9, 1.0e9, 1.0e9],
                label_resolution_mm=1.5,
                position_tolerance_mm=sampling["physical_tracking_position_tolerance_mm"],
                orientation_tolerance_deg=sampling[
                    "physical_tracking_orientation_tolerance_deg"
                ],
                separation_multiplier=sampling["waypoint_separation_multiplier"],
            )
            self.assertEqual(resolved.shape, (14, 3))

    def test_target_margin_uses_core_evidence(self) -> None:
        projected = {
            "sap": np.zeros((8, 10), dtype=bool),
            "tp": np.zeros((8, 10), dtype=bool),
            "sap_core": np.zeros((8, 10), dtype=bool),
            "tp_core": np.zeros((8, 10), dtype=bool),
        }
        projected["sap"][0, 0] = True
        projected["tp"][7, 9] = True
        projected["sap_core"][2, 3] = True
        projected["tp_core"][5, 6] = True

        self.assertEqual(target_margin(projected), 2)

    def test_pose_pairs_are_new_and_neighboring(self) -> None:
        rows = [
            {
                "pose_index_rad": [float(index * 4), 100.0, 2.0],
                "sap_first_visible_pixels": 10,
                "tp_first_visible_pixels": 10,
                "target_margin_pixels": 20,
            }
            for index in range(8)
        ]
        chosen = select_pairs(rows, 2, [{"pose_index_rad": [-20.0, 100.0, 2.0]}])
        self.assertEqual(len(chosen), 4)
        self.assertTrue(distinct(chosen[0], chosen[1]))
        self.assertTrue(nearby(chosen[0], chosen[1]))

    def test_duplicate_old_pose_is_excluded(self) -> None:
        rows = [
            {
                "pose_index_rad": [float(index * 4), 100.0, 2.0],
                "sap_first_visible_pixels": 10,
                "tp_first_visible_pixels": 10,
                "target_margin_pixels": 20,
            }
            for index in range(6)
        ]
        chosen = select_pairs(rows, 1, [{"pose_index_rad": [0.0, 100.0, 2.0]}])
        self.assertNotEqual(chosen[0]["pose_index_rad"], [0.0, 100.0, 2.0])

    def test_exact_tolerance_boundary_is_not_a_distinct_view(self) -> None:
        first = {"pose_index_rad": [100.0, 100.0, 2.0]}
        second = {"pose_index_rad": [102.0, 100.0, 2.0]}
        self.assertFalse(distinct(first, second))


if __name__ == "__main__":
    unittest.main()
