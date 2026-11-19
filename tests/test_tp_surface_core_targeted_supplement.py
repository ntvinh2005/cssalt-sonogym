#!/usr/bin/env python3
"""CPU contract tests for the targeted L4 supplement."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v12.yaml"
SEAL = PROTOCOL.with_suffix(".sha256")
SCRIPT = ROOT / "workflows/two_stage_navigation/train_tp_surface_core_l4_targeted_supplement.py"
SPEC = importlib.util.spec_from_file_location("targeted_supplement", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TARGETED = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TARGETED
SPEC.loader.exec_module(TARGETED)


class TargetedSupplementTests(unittest.TestCase):
    def test_protocol_is_sealed_and_split_before_capture(self) -> None:
        expected = SEAL.read_text().split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), expected)
        protocol = yaml.safe_load(PROTOCOL.read_text())
        self.assertRegex(protocol["source"]["implementation_commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(protocol["source"]["superseded_failed_job"], 51992362)
        self.assertEqual(
            [(row["split"], row["expected_frame_count"]) for row in protocol["sessions"]],
            [("train", 48), ("calibration", 24), ("development_evaluation", 24)],
        )

    def test_waypoint_pairs_exceed_motion_tolerance(self) -> None:
        protocol = yaml.safe_load(PROTOCOL.read_text())
        for rows in protocol["sampling"]["absolute_waypoint_banks_index"].values():
            self.assertEqual(len(rows), len({tuple(row) for row in rows}))
            self.assertLessEqual(max(float(row[2]) for row in rows), 2.94)
            for first, second in zip(rows[::2], rows[1::2]):
                translation_mm = math.hypot(second[0] - first[0], second[1] - first[1]) * 1.5
                angle_deg = abs(second[2] - first[2]) * 180.0 / math.pi
                self.assertTrue(translation_mm > 3.0 or angle_deg > 2.0)

    def test_uncertainty_is_never_negative(self) -> None:
        row = {
            "projection_status": "VERIFIED",
            "candidate_id": "BACKGROUND_OR_OTHER_STRUCTURE",
            "review_target": "REVIEW_HARD_NEGATIVE",
            "l4_core_points_in_patch": "0",
            "l5_core_points_in_patch": "0",
            "l4_uncertain_points_in_patch": "1",
            "l5_uncertain_points_in_patch": "0",
            "adequate_projected_support": "False",
        }
        state, _ = TARGETED.new_proposal_state(row)
        self.assertEqual(state, TARGETED.ABLATION.TRISTATE.PROXY_AMBIGUOUS)

    def test_l5_core_is_l4_negative_when_clear(self) -> None:
        row = {
            "projection_status": "VERIFIED",
            "candidate_id": "L5_RIGHT_TP_SURFACE_CORE",
            "review_target": "REVIEW_POSITIVE",
            "l4_core_points_in_patch": "0",
            "l5_core_points_in_patch": "12",
            "l4_uncertain_points_in_patch": "0",
            "l5_uncertain_points_in_patch": "0",
            "adequate_projected_support": "True",
        }
        state, _ = TARGETED.new_proposal_state(row)
        self.assertEqual(state, TARGETED.ABLATION.TRISTATE.PROXY_NEGATIVE)

    def test_l5_control_keeps_l4_core_as_clear_negative(self) -> None:
        row = {
            "projection_status": "VERIFIED",
            "candidate_id": "L4_RIGHT_TP_SURFACE_CORE",
            "review_target": "REVIEW_POSITIVE",
            "l4_core_points_in_patch": "12",
            "l5_core_points_in_patch": "0",
            "l4_uncertain_points_in_patch": "0",
            "l5_uncertain_points_in_patch": "0",
            "adequate_projected_support": "True",
        }
        self.assertEqual(
            TARGETED.new_l5_proposal_state(row),
            TARGETED.ABLATION.TRISTATE.PROXY_NEGATIVE,
        )

    def test_fit_guard_is_explicit(self) -> None:
        source = SCRIPT.read_text()
        self.assertIn('row["split"] != "train"', source)
        self.assertIn('development_used_for_fit_or_calibration": False', source)
        self.assertIn('"development_feature_rows_loaded_before_calibration": 0', source)
        self.assertIn('if calibration_pass:', source)
        self.assertLess(
            source.index('if calibration_pass:'),
            source.index('evaluation_features = ABLATION.load_features'),
        )


if __name__ == "__main__":
    unittest.main()
