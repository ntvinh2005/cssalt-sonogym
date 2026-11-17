"""CPU parity tests for the native SonoGym image-plane convention."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

PATH = (
    Path(__file__).resolve().parents[1]
    / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/ultrasound_plane_geometry.py"
)
SPEC = importlib.util.spec_from_file_location("ultrasound_plane_geometry", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
native_local_coordinates = MODULE.native_local_coordinates
plane_from_full_pose = MODULE.plane_from_full_pose
plane_from_reduced_pose = MODULE.plane_from_reduced_pose
project_points_to_pixels = MODULE.project_points_to_pixels


class NativeCoordinateTests(unittest.TestCase):
    def test_row_is_depth_column_is_width_and_elevation_is_zero(self) -> None:
        local = native_local_coordinates(width=4, depth=3, resolution_m=0.5)
        np.testing.assert_array_equal(local[0, 2], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(local[1, 2], [0.0, 0.0, 0.5])
        np.testing.assert_array_equal(local[0, 3], [0.5, 0.0, 0.0])
        self.assertTrue(np.all(local[..., 1] == 0.0))

    def test_full_pose_known_coordinates_and_nonidentity_rotation(self) -> None:
        # Independent expected values: local x maps to world y, elevation to -world x,
        # and local depth maps to world z for this +90 degree z rotation.
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        local = np.asarray([[[0.0, 0.0, 0.0], [0.002, 0.0, 0.003]]])
        plane = plane_from_full_pose(
            end_effector_position_m=np.asarray([0.01, 0.02, 0.03]),
            rotation_columns=rotation,
            image_origin_height_m=0.004,
            local_coordinates_m=local,
            label_resolution_m=0.001,
        )
        np.testing.assert_allclose(plane.origin_index, [10.0, 20.0, 34.0])
        np.testing.assert_allclose(plane.coordinates_index[0, 1], [10.0, 22.0, 37.0])

    def test_downward_normal_flips_only_image_origin_direction(self) -> None:
        rotation = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        local = np.asarray([[[0.0, 0.0, 0.002]]])
        plane = plane_from_full_pose(
            end_effector_position_m=np.zeros(3),
            rotation_columns=rotation,
            image_origin_height_m=0.003,
            local_coordinates_m=local,
            label_resolution_m=0.001,
        )
        # Origin is moved along +z after the native sign check; local depth still uses -z.
        np.testing.assert_allclose(plane.origin_index, [0.0, 3.0, 0.0])
        np.testing.assert_allclose(plane.coordinates_index[0, 0], [0.0, 1.0, 0.0])

    def test_reduced_pose_flat_surface_matches_independent_reference(self) -> None:
        surface_y = np.full((8, 9), 10.0)
        normals = np.zeros((8, 9, 3))
        normals[..., 1] = -1.0
        plane = plane_from_reduced_pose(
            realized_pose_index=np.asarray([3.0, 4.0, np.pi]),
            surface_y_index=surface_y,
            surface_normals=normals,
            width=4,
            depth=3,
            image_resolution_m=0.001,
            label_resolution_m=0.001,
            probe_height_m=0.010,
            image_origin_height_m=0.013,
        )
        # The controller keeps the end effector 10 mm opposite the -y normal, then
        # the native origin-sign rule adds another 13 mm along +y.
        np.testing.assert_allclose(plane.origin_index, [3.0, 33.0, 4.0], atol=1e-12)
        self.assertEqual(plane.coordinates_index.shape, (3, 4, 3))

    def test_invalid_rotation_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            plane_from_full_pose(
                end_effector_position_m=np.zeros(3),
                rotation_columns=np.ones((3, 3)),
                image_origin_height_m=0.0,
                local_coordinates_m=np.zeros((1, 1, 3)),
                label_resolution_m=0.001,
            )


class InverseProjectionTests(unittest.TestCase):
    def test_known_points_map_back_to_correct_pixels(self) -> None:
        points = np.asarray(
            [
                [-2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0],
                [1.0, 2.0, 1.0],
            ]
        )
        result = project_points_to_pixels(
            points_index=points,
            image_origin_index=np.zeros(3),
            rotation_columns=np.eye(3),
            width=6,
            depth=5,
            image_resolution_m=0.001,
            label_resolution_m=0.001,
            plane_tolerance_mm=0.5,
        )
        np.testing.assert_array_equal(result["columns"], [1, 3, 4])
        np.testing.assert_array_equal(result["rows"], [0, 3, 1])
        np.testing.assert_array_equal(result["inside"], [True, True, False])

    def test_old_width_elevation_mapping_is_detected(self) -> None:
        # A point two millimetres in depth must map to row 2, not be rejected as
        # two millimetres out of plane as the historical X-Y helper did.
        result = project_points_to_pixels(
            points_index=np.asarray([[0.0, 0.0, 2.0]]),
            image_origin_index=np.zeros(3),
            rotation_columns=np.eye(3),
            width=3,
            depth=4,
            image_resolution_m=0.001,
            label_resolution_m=0.001,
            plane_tolerance_mm=0.1,
        )
        self.assertEqual(int(result["rows"][0]), 2)
        self.assertTrue(bool(result["inside"][0]))


if __name__ == "__main__":
    unittest.main()
