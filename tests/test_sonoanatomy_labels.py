from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from two_stage_navigation.sonoanatomy_labels import (
    BACKGROUND,
    IGNORE,
    OTHER_BONE_INTERFACE,
    SAP_INTERFACE,
    TP_INTERFACE,
    first_bone_surface,
    project_mask,
    right_level_masks,
    visible_interface_labels,
)
from two_stage_navigation.stage1_interfaces import InterfaceTrack


def coordinate_map(depth: int, width: int) -> np.ndarray:
    result = np.zeros((depth * width, 3), dtype=np.float64)
    for column in range(width):
        for row in range(depth):
            result[column * depth + row] = (row, column, 0)
    return result


class SonoanatomyLabelTests(unittest.TestCase):
    def test_right_level_identity_surface_is_not_its_own_uncertainty(self) -> None:
        prediction = np.zeros((5, 5, 5), dtype=np.uint8)
        segmentation = np.zeros_like(prediction)
        segmentation[1:5, 1:5, 1:5] = 28
        prediction[3, 2, 2] = 5
        prediction[3, 3, 2] = 6
        masks = right_level_masks(prediction, segmentation, np.eye(4), "L4")
        self.assertFalse(np.any(masks["tp"] & masks["uncertainty"]))
        self.assertFalse(np.any(masks["sap"] & masks["uncertainty"]))
        self.assertTrue(masks["uncertainty"][2, 2, 2])

    def test_project_mask_preserves_native_width_major_pixel_positions(self) -> None:
        depth, width = 4, 3
        mask = np.zeros((5, 5, 1), dtype=bool)
        mask[2, 1, 0] = True
        projected = project_mask(
            coordinate_map(depth, width), mask, image_shape=(depth, width)
        )
        self.assertEqual(projected.shape, (depth, width))
        self.assertEqual(np.argwhere(projected).tolist(), [[2, 1]])

    def test_first_bone_surface_returns_one_nearest_hit_per_column(self) -> None:
        raw = np.zeros((5, 3), dtype=np.uint8)
        raw[3:, 0] = 28
        raw[1:4, 2] = 27
        first = first_bone_surface(raw, {27, 28})
        self.assertEqual(np.argwhere(first).tolist(), [[1, 2], [3, 0]])

    def test_visible_labels_require_first_bone_and_clean_image_support(self) -> None:
        depth, width = 5, 3
        image = np.zeros((depth, width), dtype=np.float32)
        raw = np.zeros((depth, width), dtype=np.uint8)
        raw[2:, :] = 28
        coordinates = coordinate_map(depth, width)
        tp = np.zeros((6, 6, 1), dtype=bool)
        sap = np.zeros_like(tp)
        uncertainty = np.zeros_like(tp)
        tp[2, 0, 0] = True
        sap[2, 1, 0] = True
        uncertainty[2, 2, 0] = True
        track = InterfaceTrack(
            track_id="track_0",
            columns=np.asarray([0, 1, 2]),
            rows=np.asarray([2.0, 2.0, 2.0]),
            mean_ridge_score=1.0,
            maximum_jump_pixels=0.0,
            valid=True,
        )
        with mock.patch(
            "two_stage_navigation.sonoanatomy_labels.track_bright_interfaces",
            return_value=[track],
        ):
            result = visible_interface_labels(
                image=image,
                raw_labels=raw,
                coordinate_map_index=coordinates,
                masks={"tp": tp, "sap": sap, "uncertainty": uncertainty},
                bone_source_ids={28},
                maximum_track_distance_pixels=0.0,
            )
        self.assertEqual(result.labels[2, 0], TP_INTERFACE)
        self.assertEqual(result.labels[2, 1], SAP_INTERFACE)
        self.assertEqual(result.labels[2, 2], IGNORE)
        self.assertEqual(result.labels[0, 0], BACKGROUND)

    def test_visible_non_target_bone_is_an_explicit_other_bone_class(self) -> None:
        depth, width = 5, 2
        image = np.zeros((depth, width), dtype=np.float32)
        raw = np.zeros((depth, width), dtype=np.uint8)
        raw[3:, :] = 28
        empty = np.zeros((6, 6, 1), dtype=bool)
        track = InterfaceTrack(
            track_id="track_0",
            columns=np.asarray([0, 1]),
            rows=np.asarray([3.0, 3.0]),
            mean_ridge_score=1.0,
            maximum_jump_pixels=0.0,
            valid=True,
        )
        with mock.patch(
            "two_stage_navigation.sonoanatomy_labels.track_bright_interfaces",
            return_value=[track],
        ):
            result = visible_interface_labels(
                image=image,
                raw_labels=raw,
                coordinate_map_index=coordinate_map(depth, width),
                masks={"tp": empty, "sap": empty, "uncertainty": empty},
                bone_source_ids={28},
                maximum_track_distance_pixels=0.0,
            )
        self.assertTrue(np.all(result.labels[3] == OTHER_BONE_INTERFACE))


if __name__ == "__main__":
    unittest.main()
