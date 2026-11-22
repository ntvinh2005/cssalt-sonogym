"""Tests for evaluator-only RIGHT L5 SAP/TP pose-atlas selection."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PATH = Path(__file__).parents[1] / "workflows/two_stage_navigation/plan_s0010_l4mb_target_observability.py"
SPEC = importlib.util.spec_from_file_location("l4mb_observability_plan", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PoseAtlasTests(unittest.TestCase):
    def test_target_geometry_loader_uses_frozen_semantic_volumes(self) -> None:
        self.assertEqual(
            tuple(MODULE.target_workflow.load_target_geometry.__annotations__),
            ("prediction_path", "segmentation_path", "return"),
        )

    def test_pose_diversity_requires_translation_or_rotation(self) -> None:
        base = np.asarray([100.0, 120.0, 2.8])
        self.assertFalse(MODULE.physically_distinct(base, base + [3.0, 0.0, 0.0]))
        self.assertTrue(MODULE.physically_distinct(base, base + [3.1, 0.0, 0.0]))
        self.assertTrue(
            MODULE.physically_distinct(base, base + [0.0, 0.0, np.deg2rad(2.5)])
        )

    def test_selection_does_not_count_near_duplicate_poses(self) -> None:
        rows = [
            {
                "session_id": "s",
                "frame_sequence": index,
                "requested_waypoint_pose_index": [x, 120.0, 2.8],
                "score": 10.0 - index,
            }
            for index, x in enumerate((100.0, 101.0, 104.0))
        ]
        selected = MODULE.select_diverse(rows, 2, ranking_fields=("score",))
        self.assertEqual([row["frame_sequence"] for row in selected], [0, 2])

    def test_visible_score_requires_both_endpoint_masks(self) -> None:
        gradient = np.ones((4, 5), dtype=np.float64)
        first = np.zeros((4, 5), dtype=bool)
        first[1, 1] = True
        first[2, 3] = True
        projected = {
            "sap_core": np.zeros((4, 5), dtype=bool),
            "tp_core": np.zeros((4, 5), dtype=bool),
        }
        projected["sap_core"][1, 1] = True
        projected["tp_core"][2, 3] = True
        self.assertEqual(MODULE.visible_score(gradient, projected, first), (1, 1, 1.0))

    def test_ambiguous_seed_is_not_relabelled_positive(self) -> None:
        row = {
            "evaluator_label": "PROXY_AMBIGUOUS",
            "evaluator_reason": "TP_AND_SAP_VISIBLE_OUTSIDE_TARGET_CORE",
            "sap_core_projected_pixels": 4,
            "tp_core_projected_pixels": 8,
        }
        self.assertEqual(row["evaluator_label"], "PROXY_AMBIGUOUS")


if __name__ == "__main__":
    unittest.main()
