"""Focused tests for the public-6DoF landmark-graph confirmation."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows/two_stage_navigation"
if str(WORKFLOWS) not in sys.path:
    sys.path.insert(0, str(WORKFLOWS))

import evaluate_s0010_landmark_graph_confirmation as evaluation  # noqa: E402
from two_stage_navigation.ultrasound_spatial_map import (  # noqa: E402
    SpatialMapConfig,
)


def candidate(frame: int, position_mm: float, score: float = 0.8) -> dict:
    position = np.asarray([position_mm / 1000.0, 0.0, 0.0])
    return {
        "graph_score": score,
        "tp_frame": frame,
        "sap_frame": frame,
        "tp_pose_position_m": position,
        "sap_pose_position_m": position,
        "tp_pose_rotation": np.eye(3),
        "sap_pose_rotation": np.eye(3),
        "tp_coordinate_mm": np.asarray([position_mm, 0.0, 0.0]),
        "sap_coordinate_mm": np.asarray([position_mm, 2.0, 0.0]),
    }


class LandmarkGraphConfirmationTest(unittest.TestCase):
    def test_support_requires_distinct_public_poses(self) -> None:
        config = SpatialMapConfig(
            minimum_view_translation_mm=2.0, minimum_view_angle_deg=1.0
        )
        near_duplicate = [candidate(0, 0.0), candidate(1, 0.5)]
        distinct = near_duplicate + [candidate(2, 3.0)]
        self.assertEqual(evaluation.diverse_support(near_duplicate, "tp", config), 1)
        self.assertEqual(evaluation.diverse_support(distinct, "tp", config), 2)

    def test_consensus_ranking_prioritizes_independent_support(self) -> None:
        rows = [
            candidate(0, 0.0, 0.99),
            candidate(1, 0.5, 0.98),
            candidate(2, 10.0, 0.80),
            candidate(3, 12.5, 0.79),
        ]
        config = {
            "cluster_radius_mm": 3.0,
            "cluster_candidate_limit": 250,
            "minimum_view_translation_mm": 2.0,
            "minimum_view_angle_deg": 1.0,
        }
        ranked = evaluation.consensus_ranking(rows, config)
        self.assertEqual(ranked[0]["independent_support"], 2)
        self.assertAlmostEqual(
            ranked[0]["candidate"]["tp_coordinate_mm"][0], 10.0
        )


if __name__ == "__main__":
    unittest.main()
