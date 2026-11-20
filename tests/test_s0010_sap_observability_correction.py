"""Tests for the offline SAP observability correction."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

WORKFLOW_DIR = Path(__file__).resolve().parents[1] / "workflows/two_stage_navigation"
sys.path.insert(0, str(WORKFLOW_DIR))
PATH = WORKFLOW_DIR / "correct_s0010_sap_observability_v1.py"
SPEC = importlib.util.spec_from_file_location("sap_observability_correction", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SapObservabilityCorrectionTests(unittest.TestCase):
    def test_first_bone_surface_separates_visible_and_occluded_columns(self) -> None:
        raw = np.zeros((200, 150), dtype=np.uint8)
        raw[20, 10] = 28
        raw[30, 20] = 28
        pseudo = np.zeros_like(raw, dtype=bool)
        pseudo[20:25, 10] = True
        pseudo[50:55, 20] = True
        result = MODULE.first_bone_surface(
            raw_labels=raw,
            corrected_projection={"pseudo": pseudo},
            bone_ids={28},
        )
        self.assertTrue(bool(result["visible_pseudo_surface"][20, 10]))
        self.assertTrue(bool(result["occluded_pseudo"][50, 20]))
        self.assertEqual(result["visible_pseudo_surface_pixel_count"], 1)
        self.assertEqual(result["occluded_pseudo_column_count"], 1)

    def test_polyline_rasterization_and_distance_use_row_column_order(self) -> None:
        points = np.asarray([[10, 20], [10, 24]], dtype=np.int64)
        pixels = MODULE.rasterize_polyline(points)
        self.assertEqual(len(pixels), 5)
        mask = np.zeros((200, 150), dtype=bool)
        mask[13, 24] = True
        self.assertEqual(MODULE.minimum_distance(pixels, mask), 3.0)

    def test_surface_status_requires_clean_alignment(self) -> None:
        self.assertEqual(
            MODULE.frame_status(
                role=MODULE.TARGET_ROLE,
                clean_decision="VISIBLE",
                visible_pseudo_surface_count=3,
                occluded_columns=0,
                minimum_polyline_distance=4.0,
            ),
            "SAP_SURFACE_CANDIDATE_VISIBLE",
        )
        self.assertEqual(
            MODULE.frame_status(
                role=MODULE.TARGET_ROLE,
                clean_decision="VISIBLE",
                visible_pseudo_surface_count=0,
                occluded_columns=5,
                minimum_polyline_distance=None,
            ),
            "OCCLUDED_BY_OTHER_BONE",
        )

    def test_structure_verdict_requires_projection_gate_in_two_sessions(self) -> None:
        targets = [
            {
                "session_id": "session_00",
                "corrected_surface_status": "SAP_SURFACE_CANDIDATE_VISIBLE",
                "corrected_projection_gate_passed": False,
            },
            {
                "session_id": "session_01",
                "corrected_surface_status": "SAP_SURFACE_CANDIDATE_VISIBLE",
                "corrected_projection_gate_passed": True,
            },
            {
                "session_id": "session_01",
                "corrected_surface_status": "SAP_SURFACE_CANDIDATE_VISIBLE",
                "corrected_projection_gate_passed": True,
            },
        ]
        result = MODULE.summarize_structure(targets)
        self.assertEqual(result["verdict"], "INCONCLUSIVE")
        self.assertEqual(result["qualifying_visible_session_count"], 1)


if __name__ == "__main__":
    unittest.main()
