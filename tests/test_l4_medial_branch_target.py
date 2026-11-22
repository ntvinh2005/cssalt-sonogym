"""Tests for the RIGHT L5 bony-surrogate evaluator and actor evidence."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


EVALUATION_DIR = (
    Path(__file__).resolve().parents[1]
    / "source/spinal_surgery/spinal_surgery/evaluation"
)
sys.path.insert(0, str(EVALUATION_DIR))
PATH = EVALUATION_DIR / "two_stage_navigation/l4_medial_branch_target.py"
SPEC = importlib.util.spec_from_file_location(
    "two_stage_navigation.l4_medial_branch_target", PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class L4MedialBranchTargetTests(unittest.TestCase):
    def test_contract_uses_physical_spacing_and_keeps_masks_disjoint(self) -> None:
        sap = np.zeros((12, 12, 12), dtype=bool)
        tp = np.zeros_like(sap)
        sap[2:5, 4:7, 4:7] = True
        tp[7:10, 4:7, 4:7] = True
        affine = np.diag([1.5, 1.5, 1.5, 1.0])
        result = MODULE.build_target_contract_geometry(
            sap_mask=sap, tp_mask=tp, affine=affine
        )
        self.assertAlmostEqual(result.minimum_surface_separation_mm, 4.5)
        self.assertAlmostEqual(result.core_separation_limit_mm, 6.0)
        self.assertGreater(np.count_nonzero(result.sap_core), 0)
        self.assertGreater(np.count_nonzero(result.tp_core), 0)
        self.assertFalse(np.any(result.sap_core & result.tp_core))
        self.assertEqual(result.midpoint_world_mm.shape[1], 3)

    def test_coordinate_index_follows_native_width_depth_flattening(self) -> None:
        self.assertEqual(
            MODULE.projected_coordinate_index(100, 40, depth=200, width=150),
            8100,
        )

    def test_distance_uses_native_coordinate_map_without_actor_leakage(self) -> None:
        coordinates = np.zeros((200 * 150, 3), dtype=np.float64)
        coordinates[40 * 200 + 100] = [2.0, 3.0, 4.0]
        distance = MODULE.distance_to_target_region_mm(
            coordinate_map_index=coordinates,
            row=100,
            column=40,
            affine=np.diag([1.5, 1.5, 1.5, 1.0]),
            midpoint_world_mm=np.asarray([[3.0, 4.5, 6.0]]),
        )
        self.assertAlmostEqual(distance, 0.0)

    def test_image_hypothesis_uses_only_image_and_tp_proposal(self) -> None:
        image = np.zeros((200, 150), dtype=np.float64)
        columns = np.arange(18, 132)
        rows = np.rint(126 + 2 * np.sin(columns / 12)).astype(int)
        for row, column in zip(rows, columns):
            image[max(0, row - 1) : row + 2, column] = 1.0
            if 85 <= column <= 105:
                image[row + 3 : row + 10, column] = 0.0
        result = MODULE.image_target_hypothesis(
            image,
            proposal_row=float(rows[48 - 18]),
            proposal_column=48.0,
            proposal_track_id="H0_BASELINE",
            minimum_column_offset_pixels=20,
            maximum_column_offset_pixels=65,
            smoothing_width_pixels=5,
            minimum_evidence_score=0.4,
            maximum_proposal_track_distance_pixels=6.0,
            minimum_image_margin_pixels=4,
        )
        self.assertEqual(result.status, "ACCEPT")
        self.assertIsNotNone(result.column)
        self.assertGreaterEqual(float(result.column), 68.0)
        self.assertLessEqual(float(result.column), 113.0)

    def test_image_hypothesis_abstains_when_proposal_misses_interface(self) -> None:
        image = np.zeros((200, 150), dtype=np.float64)
        image[120:123, 18:132] = 1.0
        result = MODULE.image_target_hypothesis(
            image,
            proposal_row=160.0,
            proposal_column=48.0,
            proposal_track_id="H0_BASELINE",
            minimum_column_offset_pixels=20,
            maximum_column_offset_pixels=65,
            smoothing_width_pixels=5,
            minimum_evidence_score=0.4,
            maximum_proposal_track_distance_pixels=6.0,
            minimum_image_margin_pixels=4,
        )
        self.assertEqual(result.status, "ABSTAIN")
        self.assertEqual(result.reason, "TP_PROPOSAL_NOT_ON_INTERFACE")


if __name__ == "__main__":
    unittest.main()
