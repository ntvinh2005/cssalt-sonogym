from __future__ import annotations

import inspect
import unittest

import numpy as np

from two_stage_navigation.ultrasound_multiview import (
    EvidenceConfig,
    accumulate_evidence,
    correspondence_matrix,
    interface_correspondence,
    interface_descriptors,
    multiview_feature_matrix,
    pose_distance_matrices,
)


def synthetic_interface(row: float, slope: float = 0.0) -> np.ndarray:
    image = np.zeros((200, 150), dtype=np.float64)
    columns = np.arange(150)
    rows = np.clip(np.round(row + slope * (columns - 75)), 0, 199).astype(int)
    for column, value in enumerate(rows):
        image[max(0, value - 1) : min(200, value + 2), column] = 1.0
        image[min(199, value + 3) :, column] += 0.05
    return image


class UltrasoundMultiviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EvidenceConfig(
            aggregation="mean",
            maximum_neighbor_translation_mm=25.0,
            maximum_neighbor_angle_deg=20.0,
            minimum_view_translation_mm=2.0,
            minimum_view_angle_deg=1.0,
            minimum_correspondence=0.3,
            maximum_neighbors=3,
            neighbor_weight=0.5,
            minimum_support=1,
        )

    def test_identical_interfaces_have_stronger_correspondence(self) -> None:
        first = interface_descriptors(synthetic_interface(120, 0.05))
        same = interface_descriptors(synthetic_interface(120, 0.05))
        different = interface_descriptors(synthetic_interface(150, -0.25))
        self.assertGreater(interface_correspondence(first, same), 0.99)
        self.assertGreater(
            interface_correspondence(first, same),
            interface_correspondence(first, different),
        )

    def test_same_pose_records_do_not_supply_viewpoint_support(self) -> None:
        images = [synthetic_interface(120), synthetic_interface(120)]
        correspondence = correspondence_matrix(images)
        translation, angle = pose_distance_matrices(np.zeros((2, 3)))
        accumulated, counts, indices = accumulate_evidence(
            np.asarray([0.8, 0.9]), correspondence, translation, angle, self.config
        )
        np.testing.assert_allclose(accumulated, [0.8, 0.9])
        np.testing.assert_array_equal(counts, [0, 0])
        self.assertEqual(indices, [[], []])

    def test_distinct_corresponding_view_changes_evidence(self) -> None:
        images = [synthetic_interface(120), synthetic_interface(121)]
        correspondence = correspondence_matrix(images)
        translation, angle = pose_distance_matrices(
            np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        )
        accumulated, counts, indices = accumulate_evidence(
            np.asarray([0.2, 0.8]), correspondence, translation, angle, self.config
        )
        self.assertGreater(accumulated[0], 0.2)
        self.assertLess(accumulated[1], 0.8)
        np.testing.assert_array_equal(counts, [1, 1])
        self.assertEqual(indices, [[1], [0]])

    def test_actor_functions_do_not_accept_privileged_geometry(self) -> None:
        signature = " ".join(inspect.signature(accumulate_evidence).parameters)
        for forbidden in ("ct", "segmentation", "target", "pseudo", "raw_label"):
            self.assertNotIn(forbidden, signature)

    def test_multiview_features_require_distinct_support(self) -> None:
        images = [synthetic_interface(120), synthetic_interface(121)]
        correspondence = correspondence_matrix(images)
        translation, angle = pose_distance_matrices(
            np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        )
        matrix, counts, support = multiview_feature_matrix(
            np.asarray([0.2, 0.8]), correspondence, translation, angle, self.config
        )
        self.assertEqual(matrix.shape, (2, 13))
        np.testing.assert_array_equal(counts, [1, 1])
        self.assertEqual(support, [[1], [0]])


if __name__ == "__main__":
    unittest.main()
