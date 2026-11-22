"""Tests for calibrated evaluator-only scan-plane viewpoint solving."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PATH = Path(__file__).parents[1] / "workflows/two_stage_navigation/solve_s0010_l4mb_target_viewpoints.py"
SPEC = importlib.util.spec_from_file_location("l4mb_viewpoint_solver", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ViewpointSolverTests(unittest.TestCase):
    def test_native_flatten_order_round_trip(self) -> None:
        parameters = np.asarray([10, 20, 30, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        coordinates = MODULE.coordinates_from_parameters(parameters)
        np.testing.assert_allclose(MODULE.plane_parameters(coordinates), parameters)
        expected = parameters[:3] + 100 * parameters[3:6] + 40 * parameters[6:9]
        np.testing.assert_allclose(coordinates[40 * 200 + 100], expected)

    def test_point_projection_returns_row_column_and_residual(self) -> None:
        parameters = np.asarray([10, 20, 30, 1, 0, 0, 0, 1, 0], dtype=np.float64)
        points = np.asarray([[15, 27, 30], [15, 27, 32]], dtype=np.float64)
        rc, residual = MODULE.point_projection(points, parameters)
        np.testing.assert_allclose(rc, [[5, 7], [5, 7]])
        np.testing.assert_allclose(residual, [0, 2])

    def test_polynomial_pose_model_recovers_exact_mapping(self) -> None:
        poses = np.asarray(
            [[x, z, angle] for x in (100, 110, 120) for z in (120, 130) for angle in (2.7, 2.9, 3.1)],
            dtype=np.float64,
        )
        coefficient = np.arange(12 * 9, dtype=np.float64).reshape(12, 9) / 100.0
        parameters = MODULE.features(poses) @ coefficient
        fitted = MODULE.fit_model(poses, parameters)
        np.testing.assert_allclose(MODULE.predict_parameters(poses, fitted), parameters, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
