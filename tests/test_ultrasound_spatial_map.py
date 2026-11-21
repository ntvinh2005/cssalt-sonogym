from __future__ import annotations

import inspect
import unittest

import numpy as np

from two_stage_navigation.ultrasound_spatial_map import (
    CorticalMapPoint,
    SpatialMapConfig,
    associate_spatial_points,
    cortical_map_points,
    frame_support_counts,
)


def interface_image(row: int = 120) -> np.ndarray:
    image = np.zeros((200, 150), dtype=np.float64)
    image[row - 1 : row + 2] = 1.0
    image[row + 4 :] = 0.04
    return image


class UltrasoundSpatialMapTests(unittest.TestCase):
    def test_known_pose_maps_lateral_axis_and_depth(self) -> None:
        points = cortical_map_points(
            interface_image(),
            executed_pose_index=(100.0, 200.0, 0.0),
            frame_sequence=0,
            frame_sha256="a",
        )
        point = next(value for value in points if value.column == 72)
        self.assertAlmostEqual(point.coordinate_mm[0], 148.5)
        self.assertAlmostEqual(point.coordinate_mm[1], 300.0)
        self.assertAlmostEqual(point.coordinate_mm[2], 59.5, places=1)

    def test_motion_compensation_associates_same_surface(self) -> None:
        config = SpatialMapConfig(include_depth=True)
        points = (
            CorticalMapPoint(0, "a", "t", 120, 75, (0.0, 0.0, 60.0), 1.0),
            CorticalMapPoint(1, "b", "t", 120, 72, (0.5, 0.0, 60.0), 1.0),
        )
        edges = associate_spatial_points(
            points,
            poses_by_frame={0: (0, 0, 0), 1: (1, 0, 0)},
            config=config,
        )
        self.assertEqual(len(edges), 1)
        self.assertEqual(frame_support_counts(points, edges), {0: 1, 1: 1})

    def test_same_hash_cannot_supply_independent_support(self) -> None:
        points = (
            CorticalMapPoint(0, "same", "t", 120, 75, (0.0, 0.0, 60.0), 1.0),
            CorticalMapPoint(1, "same", "t", 120, 75, (0.0, 0.0, 60.0), 1.0),
        )
        with self.assertRaisesRegex(ValueError, "duplicate image hash"):
            associate_spatial_points(
                points, poses_by_frame={0: (0, 0, 0), 1: (1, 0, 0)}
            )

    def test_same_pose_does_not_supply_support(self) -> None:
        points = (
            CorticalMapPoint(0, "a", "t", 120, 75, (0.0, 0.0, 60.0), 1.0),
            CorticalMapPoint(1, "b", "t", 120, 75, (0.0, 0.0, 60.0), 1.0),
        )
        edges = associate_spatial_points(
            points, poses_by_frame={0: (0, 0, 0), 1: (0, 0, 0)}
        )
        self.assertEqual(edges, ())

    def test_actor_api_has_no_privileged_inputs(self) -> None:
        signature = " ".join(inspect.signature(cortical_map_points).parameters)
        source = inspect.getsource(cortical_map_points) + inspect.getsource(
            associate_spatial_points
        )
        for forbidden in (
            "raw_label",
            "segmentation",
            "native_coordinate",
            "target_coordinate",
            "full_pose",
        ):
            self.assertNotIn(forbidden, signature)
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
