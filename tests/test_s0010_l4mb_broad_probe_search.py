"""Tests for evaluator-only broad RIGHT L5 SAP/TP probe search."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PATH = (
    Path(__file__).parents[1]
    / "workflows/two_stage_navigation/search_s0010_l4mb_target_viewpoints.py"
)
SPEC = importlib.util.spec_from_file_location("l4mb_broad_probe_search", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BroadProbeSearchTests(unittest.TestCase):
    def test_native_flatten_order_matches_width_major_slicer_order(self) -> None:
        coordinates = np.zeros((MODULE.IMAGE_DEPTH, MODULE.IMAGE_WIDTH, 3))
        for row in range(MODULE.IMAGE_DEPTH):
            for column in range(MODULE.IMAGE_WIDTH):
                coordinates[row, column] = [column, 0, row]
        flattened = MODULE.native_flat_coordinates(coordinates)
        np.testing.assert_array_equal(flattened[0], [0, 0, 0])
        np.testing.assert_array_equal(flattened[1], [0, 0, 1])
        np.testing.assert_array_equal(flattened[MODULE.IMAGE_DEPTH], [1, 0, 0])

    def test_diverse_selection_prefers_positive_and_rejects_duplicate_pose(self) -> None:
        common = {
            "evaluator_reason": "test",
            "sap_core_first_visible_pixels": 1,
            "tp_core_first_visible_pixels": 1,
            "sap_first_visible_pixels": 3,
            "tp_first_visible_pixels": 4,
            "target_margin_pixels": 5,
        }
        rows = [
            {**common, "pose_index_rad": [120, 120, 2.0], "evaluator_label": "PROXY_AMBIGUOUS"},
            {**common, "pose_index_rad": [120, 120, 2.0], "evaluator_label": "PROXY_POSITIVE"},
            {**common, "pose_index_rad": [124, 120, 2.0], "evaluator_label": "PROXY_POSITIVE"},
        ]
        selected = MODULE.select_diverse(rows, 2)
        self.assertEqual([row["evaluator_label"] for row in selected], ["PROXY_POSITIVE"] * 2)
        self.assertEqual([row["pose_index_rad"][0] for row in selected], [120, 124])


if __name__ == "__main__":
    unittest.main()
