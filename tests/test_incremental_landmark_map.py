from __future__ import annotations

from dataclasses import replace
import math
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
    def test_l5_tp_fresh_views_can_confirm_without_counting_source(self) -> None:
        def revisit(frame: int, angle_deg: float, x: float) -> SemanticObservation:
            angle = math.radians(angle_deg)
            c, s = math.cos(angle), math.sin(angle)
            return replace(
                observation("TP", frame, (x, 0.0, 20.0)),
                viewpoint_position_m=(0.0, 0.0, 0.0),
                viewpoint_rotation_columns=((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)),
            )

        source = revisit(0, 0.0, 0.0)
        left = revisit(1, -0.8, -0.2)
        right = revisit(2, 0.8, 0.3)
        duplicate = replace(right, frame_sequence=3)
        distant = revisit(4, 4.0, 2.5)
        for fresh, expected in (
            ((left, right, duplicate), (1, 2)),
            ((left, distant), ()),
            ((left, replace(right, viewpoint_rotation_columns=left.viewpoint_rotation_columns)), ()),
        ):
            with self.subTest(fresh=fresh):
                mapping = IncrementalAnatomicalLandmarkMap(
                    IncrementalLandmarkMapConfig(l5_tp_pairwise_reacquisition=True)
                )
                mapping.add_observations("L5", (source,))
                mapping.add_reacquisition_observations(
                    "L5",
                    "TP",
                    reference_position_mm=source.coordinate_mm,
                    reference_frame_hashes=(source.frame_sha256,),
                    observations=fresh,
                    preserve_frozen_surface_extent=True,
                )
                landmark = mapping.landmark("RIGHT_L5_TP")
                self.assertEqual(landmark.status, "CONFIRMED" if expected else "CANDIDATE")
                self.assertEqual(set(landmark.supporting_frame_sequences), set(expected or (0,)))
                self.assertEqual(mapping.selected_reacquisition_frame_sequences("RIGHT_L5_TP"), expected)

    def test_sap_prediction_cannot_anchor_itself(self) -> None:
        mapping = IncrementalAnatomicalLandmarkMap(IncrementalLandmarkMapConfig())
        with self.assertRaisesRegex(ValueError, "only independently confirmed TP"):
            mapping.set_actor_anchor("RIGHT_L4_SAP", (0.0, 0.0, 20.0))

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

    def test_two_additional_motion_views_resolve_close_competitor(self) -> None:
        mapping = IncrementalAnatomicalLandmarkMap(
            IncrementalLandmarkMapConfig(minimum_hypothesis_score_margin=0.05)
        )
        mapping.add_observations(
            "L4",
            (
                observation("SAP", 0, (0.0, 0.0, 20.0)),
                observation("SAP", 1, (0.5, 0.0, 20.0)),
                observation("SAP", 2, (0.2, 0.0, 20.0)),
                observation("SAP", 3, (0.3, 0.0, 20.0)),
                observation("SAP", 4, (12.0, 0.0, 20.0)),
                observation("SAP", 5, (12.5, 0.0, 20.0)),
            ),
        )

        hypothesis = mapping.landmark("RIGHT_L4_SAP")
        self.assertEqual(hypothesis.status, "CONFIRMED")
        self.assertEqual(len(hypothesis.supporting_frame_sequences), 4)
        self.assertLess(hypothesis.position_mm[0], 2.0)

    def test_sap_prefers_three_pose_support_over_compact_two_pose_cluster(self) -> None:
        mapping = IncrementalAnatomicalLandmarkMap(
            IncrementalLandmarkMapConfig(
                minimum_preferred_sap_supporting_views=3
            )
        )
        mapping.add_observations(
            "L4",
            (
                replace(observation("SAP", 0, (0.0, 0.0, 20.0)), confidence=0.99),
                replace(observation("SAP", 1, (0.1, 0.0, 20.0)), confidence=0.99),
                observation("SAP", 2, (8.0, 0.0, 20.0)),
                observation("SAP", 3, (10.0, 0.0, 20.0)),
                observation("SAP", 4, (12.0, 0.0, 20.0)),
            ),
        )

        hypothesis = mapping.landmark("RIGHT_L4_SAP")
        self.assertEqual(hypothesis.status, "CONFIRMED")
        self.assertEqual(len(hypothesis.supporting_frame_sequences), 3)
        self.assertGreater(hypothesis.position_mm[0], 5.0)

    def test_fresh_reacquisition_extends_frozen_surface_hypothesis(self) -> None:
        mapping = IncrementalAnatomicalLandmarkMap(IncrementalLandmarkMapConfig())
        source = (
            observation("SAP", 0, (0.0, 0.0, 20.0)),
            observation("SAP", 1, (0.5, 0.0, 20.0)),
        )
        mapping.add_observations("L4", source)
        reference = mapping.landmark("RIGHT_L4_SAP")
        fresh = (
            observation("SAP", 2, (-3.0, 0.0, 20.0)),
            observation("SAP", 3, (3.0, 0.0, 20.0)),
        )

        accepted = mapping.add_reacquisition_observations(
            "L4",
            "SAP",
            reference_position_mm=reference.position_mm,
            reference_frame_hashes=reference.supporting_frame_hashes,
            observations=fresh,
        )

        hypothesis = mapping.landmark("RIGHT_L4_SAP")
        self.assertEqual(accepted, 2)
        self.assertEqual(hypothesis.status, "CONFIRMED")
        self.assertEqual(len(hypothesis.supporting_frame_sequences), 4)

    def test_reacquisition_outside_frozen_radius_is_not_associated(self) -> None:
        mapping = IncrementalAnatomicalLandmarkMap(IncrementalLandmarkMapConfig())
        source = (
            observation("SAP", 0, (0.0, 0.0, 20.0)),
            observation("SAP", 1, (0.5, 0.0, 20.0)),
        )
        mapping.add_observations("L4", source)
        reference = mapping.landmark("RIGHT_L4_SAP")

        accepted = mapping.add_reacquisition_observations(
            "L4",
            "SAP",
            reference_position_mm=reference.position_mm,
            reference_frame_hashes=reference.supporting_frame_hashes,
            observations=(observation("SAP", 2, (8.0, 0.0, 20.0)),),
        )

        self.assertEqual(accepted, 0)
        self.assertEqual(mapping.landmark("RIGHT_L4_SAP").observation_count, 2)

    def test_anchor_candidate_is_confirmed_by_linked_fresh_motion_view(self) -> None:
        mapping = IncrementalAnatomicalLandmarkMap(IncrementalLandmarkMapConfig())
        mapping.set_actor_anchor("RIGHT_L5_TP", (0.0, 0.0, 20.0))
        mapping.add_observations(
            "L5",
            (
                observation("TP", 0, (4.0, 0.0, 20.0)),
                observation("TP", 1, (4.5, 0.0, 20.0)),
                observation("TP", 2, (0.1, 0.0, 20.0)),
            ),
        )

        candidate = mapping.landmark("RIGHT_L5_TP")
        self.assertEqual(candidate.status, "CANDIDATE")
        self.assertEqual(candidate.supporting_frame_sequences, (2,))

        mapping.add_reacquisition_observations(
            "L5",
            "TP",
            reference_position_mm=candidate.position_mm,
            reference_frame_hashes=candidate.supporting_frame_hashes,
            observations=(observation("TP", 3, (0.2, 0.0, 20.0)),),
        )

        hypothesis = mapping.landmark("RIGHT_L5_TP")
        self.assertEqual(hypothesis.status, "CONFIRMED")
        self.assertEqual(len(hypothesis.supporting_frame_sequences), 2)

    def test_tp_detector_reacquisition_can_span_the_frozen_anchor_gate(self) -> None:
        mapping = IncrementalAnatomicalLandmarkMap(IncrementalLandmarkMapConfig())
        mapping.set_actor_anchor("RIGHT_L5_TP", (0.0, 0.0, 20.0))
        source = observation("TP", 0, (0.0, 0.0, 20.0))
        mapping.add_observations("L5", (source,))

        accepted = mapping.add_reacquisition_observations(
            "L5",
            "TP",
            reference_position_mm=source.coordinate_mm,
            reference_frame_hashes=(source.frame_sha256,),
            observations=(
                observation("TP", 1, (7.0, 0.0, 20.0)),
                observation("TP", 2, (8.0, 0.0, 20.0)),
            ),
            association_radius_mm=10.0,
        )

        hypothesis = mapping.landmark("RIGHT_L5_TP")
        self.assertEqual(accepted, 2)
        self.assertEqual(hypothesis.status, "CONFIRMED")
        self.assertEqual(len(hypothesis.supporting_frame_sequences), 3)

    def test_sap_reacquisition_cannot_expand_beyond_surface_radius(self) -> None:
        mapping = IncrementalAnatomicalLandmarkMap(IncrementalLandmarkMapConfig())
        source = observation("SAP", 0, (0.0, 0.0, 20.0))
        mapping.add_observations("L4", (source,))

        with self.assertRaisesRegex(ValueError, "outside its actor gate"):
            mapping.add_reacquisition_observations(
                "L4",
                "SAP",
                reference_position_mm=source.coordinate_mm,
                reference_frame_hashes=(source.frame_sha256,),
                observations=(observation("SAP", 1, (8.0, 0.0, 20.0)),),
                association_radius_mm=10.0,
            )

    def test_reacquisition_keeps_frozen_historical_hypothesis_selected(self) -> None:
        mapping = IncrementalAnatomicalLandmarkMap(
            IncrementalLandmarkMapConfig(minimum_hypothesis_score_margin=0.01)
        )
        mapping.add_observations(
            "L4",
            (
                replace(observation("SAP", 0, (0.0, 0.0, 20.0)), confidence=0.99),
                replace(observation("SAP", 1, (0.5, 0.0, 20.0)), confidence=0.99),
                observation("SAP", 2, (8.0, 0.0, 20.0)),
                observation("SAP", 3, (8.5, 0.0, 20.0)),
            ),
        )
        reference = mapping.landmark("RIGHT_L4_SAP")
        mapping.add_reacquisition_observations(
            "L4",
            "SAP",
            reference_position_mm=reference.position_mm,
            reference_frame_hashes=reference.supporting_frame_hashes,
            observations=(
                replace(
                    observation("SAP", 4, (0.2, 0.0, 20.0)),
                    confidence=0.5,
                ),
            ),
        )

        hypothesis = mapping.landmark("RIGHT_L4_SAP")
        self.assertEqual(hypothesis.status, "CONFIRMED")
        self.assertLess(hypothesis.position_mm[0], 2.0)

    def test_frozen_surface_extent_rejects_revisit_drift(self) -> None:
        mapping = IncrementalAnatomicalLandmarkMap(IncrementalLandmarkMapConfig())
        mapping.add_observations(
            "L5",
            (
                observation("SAP", 0, (0.0, 0.0, 20.0)),
                observation("SAP", 1, (1.0, 0.0, 20.0)),
                observation("SAP", 2, (2.0, 0.0, 20.0)),
            ),
        )
        reference = mapping.landmark("RIGHT_L5_SAP")

        accepted = mapping.add_reacquisition_observations(
            "L5",
            "SAP",
            reference_position_mm=reference.position_mm,
            reference_frame_hashes=reference.supporting_frame_hashes,
            observations=(observation("SAP", 3, (4.9, 0.0, 20.0)),),
            preserve_frozen_surface_extent=True,
        )

        self.assertEqual(accepted, 0)
        self.assertEqual(
            mapping.selected_reacquisition_frame_sequences("RIGHT_L5_SAP"), ()
        )

    def test_frozen_candidate_hypothesis_can_be_reacquired(self) -> None:
        mapping = IncrementalAnatomicalLandmarkMap(IncrementalLandmarkMapConfig())
        source = observation("SAP", 0, (0.0, 0.0, 20.0))
        mapping.add_observations("L5", (source,))
        reference = mapping.landmark("RIGHT_L5_SAP")
        self.assertEqual(reference.status, "CANDIDATE")

        accepted = mapping.add_reacquisition_observations(
            "L5",
            "SAP",
            reference_position_mm=reference.position_mm,
            reference_frame_hashes=reference.supporting_frame_hashes,
            observations=(observation("SAP", 1, (1.0, 0.0, 20.0)),),
            preserve_frozen_surface_extent=True,
        )

        self.assertEqual(accepted, 1)
        self.assertEqual(
            mapping.selected_reacquisition_frame_sequences("RIGHT_L5_SAP"), (1,)
        )

    def test_only_selected_distinct_reacquisition_views_count(self) -> None:
        mapping = IncrementalAnatomicalLandmarkMap(IncrementalLandmarkMapConfig())
        source = (
            observation("SAP", 0, (0.0, 0.0, 20.0)),
            observation("SAP", 1, (1.0, 0.0, 20.0)),
        )
        mapping.add_observations("L4", source)
        reference = mapping.landmark("RIGHT_L4_SAP")
        distinct = observation("SAP", 2, (0.5, 0.0, 20.0))
        duplicate_view = replace(
            observation("SAP", 3, (0.6, 0.0, 20.0)),
            viewpoint_id=distinct.viewpoint_id,
            viewpoint_position_m=distinct.viewpoint_position_m,
        )
        unrelated = observation("SAP", 4, (12.0, 0.0, 20.0))

        mapping.add_reacquisition_observations(
            "L4",
            "SAP",
            reference_position_mm=reference.position_mm,
            reference_frame_hashes=reference.supporting_frame_hashes,
            observations=(distinct, duplicate_view),
            preserve_frozen_surface_extent=True,
        )
        mapping.add_observations("L4", (unrelated,))

        self.assertEqual(
            mapping.selected_reacquisition_frame_sequences("RIGHT_L4_SAP"), (2,)
        )

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
