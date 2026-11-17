#!/usr/bin/env python3
"""Focused CPU tests for the RIGHT SAP/TP review exporter."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPOSITORY_ROOT
    / "workflows"
    / "two_stage_navigation"
    / "review_s0010_right_landmarks.py"
)
SPEC = importlib.util.spec_from_file_location("right_landmark_review", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CandidateSelectionTests(unittest.TestCase):
    def test_quantile_candidate_is_deterministic(self) -> None:
        vertices = np.asarray(
            [
                [float(x), float(y), float(z)]
                for x in range(5)
                for y in range(5)
                for z in range(5)
            ]
        )
        selector = {
            "x_quantile_fraction": [0.75, 1.0],
            "y_quantile_fraction": [0.0, 0.25],
            "z_quantile_fraction": [0.25, 0.75],
        }
        first = MODULE.quantile_candidate(vertices, selector)
        second = MODULE.quantile_candidate(vertices, selector)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[:, 0] >= 3.0))
        self.assertTrue(np.all(first[:, 1] <= 1.0))
        self.assertTrue(np.all((first[:, 2] >= 1.0) & (first[:, 2] <= 3.0)))

    def test_source_candidate_requires_explicit_right_match(self) -> None:
        rows = [
            {
                "anatomy_level": "L4",
                "landmark_name": "transverse_process",
                "laterality": "RIGHT",
            },
            {
                "anatomy_level": "L4",
                "landmark_name": "transverse_process",
                "laterality": "LEFT",
            },
        ]
        result = MODULE.source_candidate(rows, "L4", "transverse_process")
        self.assertEqual(result["laterality"], "RIGHT")

    def test_duplicate_right_candidate_fails_closed(self) -> None:
        row = {
            "anatomy_level": "L5",
            "landmark_name": "superior_articular_process",
            "laterality": "RIGHT",
        }
        with self.assertRaisesRegex(RuntimeError, "expected one"):
            MODULE.source_candidate([row, dict(row)], "L5", "superior_articular_process")

    def test_geometry_frames_preserve_units_and_point_count(self) -> None:
        points = np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        geometry = MODULE.geometry_frames(points, np.eye(4))
        self.assertEqual(geometry["mesh_index"]["point_count"], 2)
        np.testing.assert_allclose(
            geometry["sonogym_patient_local_m"]["centroid"],
            np.mean(points, axis=0) * 0.0015,
        )

    def test_projected_points_do_not_fabricate_distant_surface(self) -> None:
        points = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 8.0]])
        selected = MODULE.projected_points(points, axis=2, coordinate=3, tolerance=1.0)
        np.testing.assert_array_equal(selected, points[:1])


if __name__ == "__main__":
    unittest.main()
