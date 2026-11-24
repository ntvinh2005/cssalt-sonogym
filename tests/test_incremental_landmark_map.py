from __future__ import annotations

from dataclasses import replace
import unittest

from two_stage_navigation.incremental_landmark_map import (
    IncrementalAnatomicalLandmarkMap,
    IncrementalLandmarkMapConfig,
    select_landmark_trajectory,
)
from two_stage_navigation.sonoanatomy_multiview import SemanticObservation


def observation(kind: str, frame: int, coordinate: tuple[float, float, float]) -> SemanticObservation:
    return SemanticObservation(
        semantic_class=kind,
        frame_sequence=frame,
        frame_sha256=f"{frame:064d}",
        viewpoint_id=f"view-{frame}",
        viewpoint_position_m=(0.003 * frame, 0.0, 0.0),
        viewpoint_rotation_columns=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        row=90.0,
        column=70.0,
        coordinate_mm=coordinate,
        confidence=0.9,
        pixel_count=20,
    )


class IncrementalLandmarkMapTests(unittest.TestCase):
    def test_tp_anchor_resolves_competing_supported_clusters(self) -> None:
        mapping = IncrementalAnatomicalLandmarkMap(
            IncrementalLandmarkMapConfig(minimum_hypothesis_score_margin=0.05)
        )
        mapping.set_actor_anchor("RIGHT_L4_TP", (0.0, 0.0, 20.0))
        mapping.add_observations(
            "L4",
            (
                observation("TP", 0, (0.0, 0.0, 20.0)),
                observation("TP", 1, (0.5, 0.0, 20.0)),
                observation("TP", 2, (12.0, 0.0, 20.0)),
                observation("TP", 3, (12.5, 0.0, 20.0)),
            ),
        )

        hypothesis = mapping.landmark("RIGHT_L4_TP")
        self.assertEqual(hypothesis.status, "CONFIRMED")
        self.assertLess(hypothesis.position_mm[0], 2.0)

    def test_trajectory_selection_does_not_require_same_frame_pair(self) -> None:
        decision = select_landmark_trajectory(
            "L4",
            {
                "weak": (observation("TP", 0, (0.0, 0.0, 20.0)),),
                "supported": (
                    observation("TP", 1, (0.0, 0.0, 20.0)),
                    observation("TP", 2, (0.5, 0.0, 20.0)),
                    observation("SAP", 3, (12.0, 0.0, 20.0)),
                    observation("SAP", 4, (12.5, 0.0, 20.0)),
                ),
            },
            IncrementalLandmarkMapConfig(),
        )

        self.assertEqual(decision.status, "ACCEPT")
        self.assertEqual(decision.trajectory_id, "supported")
        self.assertEqual(decision.confirmed_landmark_count, 2)
        self.assertEqual(decision.relation_status, "CONSISTENT")

    def test_landmarks_are_confirmed_independently_without_same_frame(self) -> None:
        mapping = IncrementalAnatomicalLandmarkMap(IncrementalLandmarkMapConfig())
        mapping.add_observations(
            "L4",
            (
                observation("TP", 0, (0.0, 0.0, 20.0)),
                observation("TP", 1, (0.5, 0.0, 20.0)),
                observation("SAP", 2, (12.0, 0.0, 20.0)),
                observation("SAP", 3, (12.5, 0.0, 20.0)),
            ),
        )

        self.assertEqual(mapping.landmark("RIGHT_L4_TP").status, "CONFIRMED")
        self.assertEqual(mapping.landmark("RIGHT_L4_SAP").status, "CONFIRMED")
        self.assertEqual(mapping.relation("L4").status, "CONSISTENT")

    def test_shared_map_retains_both_levels(self) -> None:
        mapping = IncrementalAnatomicalLandmarkMap(IncrementalLandmarkMapConfig())
        for level, offset, start in (("L4", 0.0, 0), ("L5", 25.0, 4)):
            mapping.add_observations(
                level,
                (
                    observation("TP", start, (offset, 0.0, 20.0)),
                    observation("TP", start + 1, (offset + 0.5, 0.0, 20.0)),
                    observation("SAP", start + 2, (offset + 10.0, 0.0, 20.0)),
                    observation("SAP", start + 3, (offset + 10.5, 0.0, 20.0)),
                ),
            )

        self.assertTrue(mapping.complete())
        self.assertEqual(
            set(mapping.snapshot()["landmarks"]),
            {
                "RIGHT_L4_TP",
                "RIGHT_L4_SAP",
                "RIGHT_L5_TP",
                "RIGHT_L5_SAP",
            },
        )

    def test_near_duplicate_pose_does_not_confirm_landmark(self) -> None:
        mapping = IncrementalAnatomicalLandmarkMap(IncrementalLandmarkMapConfig())
        first = observation("TP", 0, (0.0, 0.0, 20.0))
        second = replace(
            observation("TP", 1, (0.5, 0.0, 20.0)),
            viewpoint_position_m=(0.001, 0.0, 0.0),
        )
        mapping.add_observations("L4", (first, second))

        self.assertEqual(mapping.landmark("RIGHT_L4_TP").status, "CANDIDATE")

    def test_competing_supported_clusters_remain_ambiguous(self) -> None:
        mapping = IncrementalAnatomicalLandmarkMap(
            IncrementalLandmarkMapConfig(minimum_hypothesis_score_margin=0.05)
        )
        mapping.add_observations(
            "L4",
            (
                observation("TP", 0, (0.0, 0.0, 20.0)),
                observation("TP", 1, (0.5, 0.0, 20.0)),
                observation("TP", 2, (12.0, 0.0, 20.0)),
                observation("TP", 3, (12.5, 0.0, 20.0)),
            ),
        )

        self.assertEqual(mapping.landmark("RIGHT_L4_TP").status, "AMBIGUOUS")

    def test_inconsistent_relative_geometry_blocks_level_confirmation(self) -> None:
        mapping = IncrementalAnatomicalLandmarkMap(
            IncrementalLandmarkMapConfig(maximum_sap_tp_distance_mm=20.0)
        )
        mapping.add_observations(
            "L4",
            (
                observation("TP", 0, (0.0, 0.0, 20.0)),
                observation("TP", 1, (0.5, 0.0, 20.0)),
                observation("SAP", 2, (40.0, 0.0, 20.0)),
                observation("SAP", 3, (40.5, 0.0, 20.0)),
            ),
        )

        self.assertEqual(mapping.relation("L4").status, "INCONSISTENT")
        self.assertFalse(mapping.level_confirmed("L4"))


if __name__ == "__main__":
    unittest.main()
