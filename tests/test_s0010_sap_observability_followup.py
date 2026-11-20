"""Tests for corrected SAP follow-up waypoint selection."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).parents[1] / "workflows/two_stage_navigation/plan_s0010_sap_observability_followup.py"
SPEC = importlib.util.spec_from_file_location("sap_followup", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FollowupPlanTests(unittest.TestCase):
    def test_physical_distinction_uses_translation_or_angle(self) -> None:
        base = np.asarray([100.0, 120.0, 2.8])
        self.assertFalse(MODULE.physically_distinct(base, base + [3.0, 0.0, 0.0]))
        self.assertTrue(MODULE.physically_distinct(base, base + [3.1, 0.0, 0.0]))
        self.assertTrue(MODULE.physically_distinct(base, base + [0.0, 0.0, np.deg2rad(2.5)]))

    def test_selection_does_not_duplicate_nearby_poses(self) -> None:
        rows = []
        for index, x in enumerate((100.0, 101.0, 104.0)):
            rows.append(
                {
                    "session_id": "s",
                    "frame_sequence": index,
                    "surface_gradient_median": 3.0 - index,
                    "visible_pseudo_surface_pixel_count": 10,
                    "core_margin_pixels": 5,
                    "requested_relative_pose_index": [x, 120.0, 2.8],
                }
            )
        selected = MODULE.select_diverse(rows, 2)
        self.assertEqual([row["frame_sequence"] for row in selected], [0, 2])


if __name__ == "__main__":
    unittest.main()
