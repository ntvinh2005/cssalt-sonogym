"""Focused checks for image-only SAP/TP interface-pair evidence."""

from __future__ import annotations

import unittest

import numpy as np

from two_stage_navigation.ultrasound_sap_tp_configuration import (
    CorticalSegment,
    SegmentConfig,
    cortical_segments,
    pair_candidates,
    pair_feature,
    track_window_pair_candidates,
)


class TestUltrasoundSapTpConfiguration(unittest.TestCase):
    def test_two_separated_arcs_create_ordered_pairs(self) -> None:
        image = np.zeros((200, 150), dtype=np.float64)
        for column in range(30, 55):
            row = 126 - int(round(5 * np.cos((column - 30) / 24 * np.pi)))
            image[row - 1 : row + 2, column] = 1.0
        for column in range(90, 120):
            row = 110 - int(round(7 * np.cos((column - 90) / 29 * np.pi)))
            image[row - 1 : row + 2, column] = 0.9
        segments = cortical_segments(
            image,
            SegmentConfig(ridge_quantile=0.80, minimum_column_span=4),
        )
        self.assertGreaterEqual(len(segments), 2)
        pairs = pair_candidates(
            image,
            SegmentConfig(ridge_quantile=0.80, minimum_column_span=4),
        )
        self.assertGreaterEqual(len(pairs), 2)
        self.assertTrue(all(feature.shape == (21,) for _, _, feature in pairs))

    def test_one_segment_cannot_create_pair(self) -> None:
        image = np.zeros((200, 150), dtype=np.float64)
        image[120:123, 40:70] = 1.0
        self.assertEqual(
            pair_candidates(
                image,
                SegmentConfig(ridge_quantile=0.80, minimum_column_span=4),
            ),
            (),
        )

    def test_pair_feature_contains_no_geometry_or_pose_argument(self) -> None:
        first = CorticalSegment(
            rows=np.asarray([100.0, 101.0]),
            columns=np.asarray([20.0, 21.0]),
            ridge_score=0.8,
            shadow_contrast=0.2,
            intensity_contrast=0.3,
        )
        second = CorticalSegment(
            rows=np.asarray([90.0, 91.0]),
            columns=np.asarray([80.0, 81.0]),
            ridge_score=0.9,
            shadow_contrast=0.4,
            intensity_contrast=0.5,
        )
        feature = pair_feature(first, second, (200, 150))
        self.assertEqual(feature.shape, (21,))
        self.assertTrue(np.all(np.isfinite(feature)))

    def test_complete_track_is_partitioned_into_left_to_right_pairs(self) -> None:
        image = np.zeros((200, 150), dtype=np.float64)
        columns = np.arange(18, 132)
        rows = 120 - 15 * np.exp(-((columns - 70) / 20) ** 2)
        for row, column in zip(np.rint(rows).astype(int), columns):
            image[row - 1 : row + 2, column] = 1.0
        pairs = track_window_pair_candidates(
            image,
            window_columns=15,
            stride_columns=5,
            minimum_center_separation_columns=8,
            maximum_tracks=1,
        )
        self.assertGreater(len(pairs), 0)
        self.assertTrue(all(first.center[1] < second.center[1] for first, second, _ in pairs))


if __name__ == "__main__":
    unittest.main()
