from __future__ import annotations

import unittest

from two_stage_navigation.sonoanatomy_multiview import (
    MultiviewConfig,
    SemanticObservation,
    cluster_observations,
    decide_tp_sap_pair,
)


def observation(kind: str, frame: int, x: float) -> SemanticObservation:
    return SemanticObservation(
        semantic_class=kind,
        frame_sequence=frame,
        frame_sha256=f"{frame:064d}",
        row=100.0,
        column=75.0,
        coordinate_mm=(x, 0.0, 0.0),
        confidence=0.9,
        pixel_count=5,
    )


class SonoanatomyMultiviewTests(unittest.TestCase):
    def test_duplicate_frame_does_not_create_independent_cluster_support(self) -> None:
        rows = (observation("TP", 0, 0.0), observation("TP", 0, 0.5))
        clusters = cluster_observations(rows, association_radius_mm=2.0)
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
