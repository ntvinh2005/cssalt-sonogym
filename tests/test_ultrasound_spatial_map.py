from __future__ import annotations

import inspect
import unittest

import numpy as np

from two_stage_navigation.ultrasound_spatial_map import (
    CorticalMapPoint,
    SpatialLocalizerConfig,
    SpatialMapConfig,
    SpatialViewCandidate,
    associate_spatial_points,
    calibrated_image_coordinates_mm,
    cortical_map_points,
    frame_support_counts,
    rank_spatial_target_views,
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

    def test_calibrated_coordinates_preserve_native_axes_and_units(self) -> None:
        coordinates = calibrated_image_coordinates_mm(
            end_effector_position_m=(0.1, 0.2, 0.3),
            rotation_columns=np.eye(3),
            width=5,
            depth=4,
            pixel_resolution_mm=0.5,
            image_origin_height_mm=100.0,
        )
        np.testing.assert_allclose(coordinates[0, 2], [100.0, 200.0, 400.0])
        np.testing.assert_allclose(coordinates[3, 4], [101.0, 200.0, 401.5])

    def test_calibrated_coordinates_reject_left_handed_rotation(self) -> None:
        rotation = np.diag([-1.0, 1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "right-handed"):
            calibrated_image_coordinates_mm(
                end_effector_position_m=(0, 0, 0),
                rotation_columns=rotation,
                width=5,
                depth=4,
                pixel_resolution_mm=0.5,
                image_origin_height_mm=100.0,
            )

    def test_spatial_support_changes_target_ranking(self) -> None:
        points = []
        for frame in (1, 2, 3, 4):
            points.append(
                CorticalMapPoint(frame, str(frame), "t", 120, 75, (0.0, 0.0, 60.0), 1.0)
            )
        candidates = (
            SpatialViewCandidate(0, "a", 0.9, 120, 75),
            SpatialViewCandidate(1, "1", 0.7, 120, 75),
        )
        poses = {frame: (0.0, 0.0, 0.0) for frame in range(5)}
        decision = rank_spatial_target_views(
            points,
            candidates,
            poses_by_frame=poses,
            stage1_prior_pose_index=(0.0, 0.0, 0.0),
            localizer_config=SpatialLocalizerConfig(
                hypothesis="support", minimum_margin=0.0
            ),
        )
        self.assertEqual(decision.status, "ACCEPT")
        self.assertEqual(decision.selected.frame_sequence, 0)
        self.assertEqual(len(decision.selected.supporting_frames), 4)

    def test_localizer_abstains_without_distinct_multiview_support(self) -> None:
        decision = rank_spatial_target_views(
            (),
            (SpatialViewCandidate(0, "a", 0.99, 120, 75),),
            poses_by_frame={0: (0.0, 0.0, 0.0)},
            stage1_prior_pose_index=(0.0, 0.0, 0.0),
            localizer_config=SpatialLocalizerConfig(hypothesis="support"),
        )
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIsNone(decision.selected)
        self.assertEqual(decision.reason, "NO_ELIGIBLE_SPATIAL_VIEW")

    def test_localizer_api_has_no_evaluator_inputs(self) -> None:
        signature = " ".join(inspect.signature(rank_spatial_target_views).parameters)
        source = inspect.getsource(rank_spatial_target_views)
        for forbidden in (
            "raw_label",
            "segmentation",
            "native_coordinate",
            "target_coordinate",
            "full_pose",
            "geometry",
        ):
            self.assertNotIn(forbidden, signature)
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
