from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from two_stage_navigation.sonoanatomy_multiview import (
    MultiviewConfig,
    SemanticObservation,
    cluster_observations,
    decide_tp_sap_pair,
    semantic_observations,
)


def observation(kind: str, frame: int, x: float) -> SemanticObservation:
    return SemanticObservation(
        semantic_class=kind,
        frame_sequence=frame,
        frame_sha256=f"{frame:064d}",
        viewpoint_id=f"view-{frame}",
        viewpoint_position_m=(0.003 * frame, 0.0, 0.0),
        viewpoint_rotation_columns=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        row=100.0,
        column=75.0,
        coordinate_mm=(x, 0.0, 0.0),
        confidence=0.9,
        pixel_count=5,
    )


def stale_observation(kind: str, frame: int, x: float) -> SemanticObservation:
    row = observation(kind, frame, x)
    return SemanticObservation(
        semantic_class=row.semantic_class,
        frame_sequence=row.frame_sequence,
        frame_sha256="stale-image",
        viewpoint_id=row.viewpoint_id,
        viewpoint_position_m=row.viewpoint_position_m,
        viewpoint_rotation_columns=row.viewpoint_rotation_columns,
        row=row.row,
        column=row.column,
        coordinate_mm=row.coordinate_mm,
        confidence=row.confidence,
        pixel_count=row.pixel_count,
    )


class SonoanatomyMultiviewTests(unittest.TestCase):
    def test_semantic_observation_requires_class_to_win_the_pixel(self) -> None:
        probabilities = np.zeros((4, 4, 4), dtype=np.float32)
        probabilities[1, 1, 1] = 0.8
        probabilities[2, 1, 1] = 0.9
        thresholds = {
            name: {"threshold": 0.7, "minimum_pixels": 1}
            for name in ("TP", "SAP", "OTHER_BONE")
        }

        rows = semantic_observations(
            probabilities,
            thresholds=thresholds,
            end_effector_position_m=(0.0, 0.0, 0.0),
            rotation_columns=np.eye(3),
            frame_sequence=0,
            frame_sha256="0" * 64,
        )

        self.assertEqual([row.semantic_class for row in rows], ["SAP"])

    def test_duplicate_frame_does_not_create_independent_cluster_support(self) -> None:
        rows = (observation("TP", 0, 0.0), observation("TP", 0, 0.5))
        clusters = cluster_observations(rows, association_radius_mm=2.0)
        self.assertEqual(len(clusters), 2)
        self.assertTrue(all(len(cluster.supporting_frames) == 1 for cluster in clusters))

    def test_stale_image_does_not_create_independent_cluster_support(self) -> None:
        rows = (
            stale_observation("TP", 0, 0.0),
            stale_observation("TP", 1, 0.5),
        )
        clusters = cluster_observations(rows, association_radius_mm=2.0)
        self.assertEqual(len(clusters), 2)
        self.assertTrue(all(len(cluster.supporting_frames) == 1 for cluster in clusters))

    def test_duplicate_viewpoint_does_not_create_independent_cluster_support(self) -> None:
        first = observation("TP", 0, 0.0)
        second = replace(observation("TP", 1, 0.5), viewpoint_id=first.viewpoint_id)
        clusters = cluster_observations((first, second), association_radius_mm=2.0)
        self.assertEqual(len(clusters), 2)
        self.assertTrue(all(len(cluster.supporting_frames) == 1 for cluster in clusters))

    def test_near_duplicate_viewpoint_does_not_add_support(self) -> None:
        first = observation("TP", 0, 0.0)
        second = replace(
            observation("TP", 1, 0.5),
            viewpoint_id="different-record",
            viewpoint_position_m=(0.001, 0.0, 0.0),
        )
        clusters = cluster_observations((first, second), association_radius_mm=2.0)
        self.assertEqual(len(clusters), 2)
        self.assertTrue(all(len(cluster.supporting_frames) == 1 for cluster in clusters))

    def test_supported_pair_accepts_with_distinct_views(self) -> None:
        rows = (
            observation("TP", 0, 0.0),
            observation("TP", 1, 0.5),
            observation("SAP", 2, 8.0),
            observation("SAP", 3, 8.5),
        )
        clusters = cluster_observations(rows, association_radius_mm=2.0)
        decision = decide_tp_sap_pair(
            clusters,
            config=MultiviewConfig(
                minimum_pair_distance_mm=4.0,
                maximum_pair_distance_mm=12.0,
                minimum_pair_score=0.5,
                minimum_score_margin=0.01,
            ),
        )
        self.assertEqual(decision.status, "ACCEPT")
        self.assertEqual(len(decision.selected.supporting_frames), 4)

    def test_pair_abstains_when_sap_has_one_view(self) -> None:
        rows = (
            observation("TP", 0, 0.0),
            observation("TP", 1, 0.5),
            observation("SAP", 2, 8.0),
        )
        clusters = cluster_observations(rows, association_radius_mm=2.0)
        decision = decide_tp_sap_pair(clusters, config=MultiviewConfig())
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertEqual(decision.reason, "INSUFFICIENT_SUPPORTED_STRUCTURES")


if __name__ == "__main__":
    unittest.main()
