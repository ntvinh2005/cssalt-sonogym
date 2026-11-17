#!/usr/bin/env python3
"""CPU-only tests for the MBB-oriented Stage 2 research contracts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.stage2_mbb_landmarks import (  # noqa: E402
    AnatomicalConsistencyRecord,
    BranchIdentity,
    CalibrationRecord,
    DecisionStatus,
    FacetLevel,
    LandmarkCandidateMemory,
    LandmarkHypothesis,
    LandmarkReadiness,
    LandmarkType,
    Laterality,
    ObservationEvidence,
    TargetRegionMapping,
    evaluate_landmark_evidence,
    required_sites,
    validate_mbb_actor_payload,
)


def observation(
    sequence: int,
    *,
    session: str = "session-a",
    digest: str | None = None,
    pose: tuple[float, float, float] | None = None,
    advanced: bool = True,
    transition_valid: bool = True,
) -> ObservationEvidence:
    return ObservationEvidence(
        session_id=session,
        frame_sequence=sequence,
        producer_sequence=sequence,
        frame_sha256=digest or f"{sequence + 1:064x}",
        executed_relative_pose=pose or (float(sequence), 0.0, 0.0),
        transition_valid=transition_valid,
        image_valid=True,
        producer_advanced=advanced,
    )


def hypothesis(
    level: str,
    kind: LandmarkType,
    *,
    side: Laterality = Laterality.LEFT,
    evidence: tuple[ObservationEvidence, ...] | None = None,
    calibrated: bool = True,
) -> LandmarkHypothesis:
    rows = evidence or (observation(0), observation(1))
    return LandmarkHypothesis(
        landmark_type=kind,
        vertebral_level=level,
        laterality=side,
        image_xy_px=(12.0, 18.0),
        patient_relative_xyz_mm=(1.0, 2.0, 3.0),
        covariance_mm2=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        confidence=0.7,
        evidence=rows,
        transform_validated=True,
        calibration=(CalibrationRecord("calibration-v1", True, False) if calibrated else None),
    )


def ready_map(side: Laterality = Laterality.LEFT) -> dict[tuple, LandmarkReadiness]:
    return {
        (level, kind, side): LandmarkReadiness.READY
        for level in ("L4", "L5")
        for kind in (
            LandmarkType.SUPERIOR_ARTICULAR_PROCESS,
            LandmarkType.TRANSVERSE_PROCESS,
        )
    }


def consistency_records() -> dict[BranchIdentity, AnatomicalConsistencyRecord]:
    return {
        branch: AnatomicalConsistencyRecord(branch, "review-v1", True, True)
        for branch in (BranchIdentity.L3_MEDIAL_BRANCH, BranchIdentity.L4_MEDIAL_BRANCH)
    }


class TaxonomyTests(unittest.TestCase):
    def test_l4_l5_maps_to_two_distinct_branch_sites(self) -> None:
        sites = required_sites(FacetLevel.L4_L5, Laterality.RIGHT)
        self.assertEqual(
            [site.branch for site in sites],
            [BranchIdentity.L3_MEDIAL_BRANCH, BranchIdentity.L4_MEDIAL_BRANCH],
        )
        self.assertEqual(
            [[item.vertebral_level for item in site.required_landmarks] for site in sites],
            [["L4", "L4"], ["L5", "L5"]],
        )
        self.assertTrue(
            all(
                item.laterality is Laterality.RIGHT
                for site in sites
                for item in site.required_landmarks
            )
        )

    def test_laterality_is_explicit(self) -> None:
        with self.assertRaisesRegex(TypeError, "explicit Laterality"):
            required_sites(FacetLevel.L4_L5, "LEFT")  # type: ignore[arg-type]

    def test_l5_dorsal_ramification_is_not_generic_medial_branch(self) -> None:
        initial = required_sites(FacetLevel.L4_L5, Laterality.LEFT)
        self.assertNotIn(BranchIdentity.L5_DORSAL_RAMUS, [site.branch for site in initial])
        transition = required_sites(FacetLevel.L5_S1, Laterality.LEFT)
        l5_site = next(site for site in transition if site.branch is BranchIdentity.L5_DORSAL_RAMUS)
        self.assertEqual(
            {item.landmark_type for item in l5_site.required_landmarks},
            {LandmarkType.SUPERIOR_ARTICULAR_PROCESS, LandmarkType.SACRAL_ALA},
        )

    def test_centroid_cannot_be_target_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "centroid"):
            TargetRegionMapping(
                BranchIdentity.L3_MEDIAL_BRANCH,
                "vertebral-centroid",
                validated=True,
                uses_vertebral_centroid=True,
            )


class InformationBoundaryTests(unittest.TestCase):
    def test_privileged_fields_are_rejected_recursively(self) -> None:
        allowed = {"ultrasound": [[0.0]], "history": [{"executed_relative_pose": [0, 0, 0]}]}
        validate_mbb_actor_payload(allowed)
        for key in (
            "ct_volume",
            "raw_segmentation",
            "evaluator_goal_pose",
            "ground_truth_target_coordinates",
            "future_observations",
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "privileged"):
                validate_mbb_actor_payload({"ultrasound": [[0.0]], "nested": [{key: [1]}]})

    def test_memory_rejects_reset_boundary_and_future_reference(self) -> None:
        memory = LandmarkCandidateMemory("session-a")
        memory.add_observation(observation(0))
        with self.assertRaisesRegex(ValueError, "boundary"):
            memory.add_observation(observation(1, session="session-b"))
        with self.assertRaisesRegex(ValueError, "future"):
            memory.add_hypothesis(
                hypothesis(
                    "L4",
                    LandmarkType.TRANSVERSE_PROCESS,
                    evidence=(observation(1), observation(2)),
                )
            )

    def test_memory_requires_causal_frame_order(self) -> None:
        memory = LandmarkCandidateMemory("session-a")
        memory.add_observation(observation(2))
        with self.assertRaisesRegex(ValueError, "future/reordered"):
            memory.add_observation(observation(1))


class MultiViewAndDecisionTests(unittest.TestCase):
    def test_duplicate_hashes_cannot_create_independent_views(self) -> None:
        duplicate = "a" * 64
        row = hypothesis(
            "L4",
            LandmarkType.TRANSVERSE_PROCESS,
            evidence=(
                observation(0, digest=duplicate, pose=(0.0, 0.0, 0.0)),
                observation(1, digest=duplicate, pose=(1.0, 0.0, 0.0)),
            ),
        )
        self.assertEqual(row.distinct_viewpoint_count, 1)

    def test_distinct_hashes_at_same_pose_are_one_viewpoint(self) -> None:
        row = hypothesis(
            "L4",
            LandmarkType.TRANSVERSE_PROCESS,
            evidence=(
                observation(0, pose=(0.0, 0.0, 0.0)),
                observation(1, pose=(0.0, 0.0, 0.0)),
            ),
        )
        self.assertEqual(row.distinct_viewpoint_count, 1)

    def test_invalid_or_stale_evidence_does_not_count(self) -> None:
        row = hypothesis(
            "L4",
            LandmarkType.TRANSVERSE_PROCESS,
            evidence=(
                observation(0),
                observation(1, transition_valid=False),
                observation(2, advanced=False),
            ),
        )
        self.assertEqual(row.distinct_viewpoint_count, 1)

    def test_missing_transform_prevents_3d_claim(self) -> None:
        with self.assertRaisesRegex(ValueError, "validated coordinate transform"):
            LandmarkHypothesis(
                LandmarkType.TRANSVERSE_PROCESS,
                "L4",
                Laterality.LEFT,
                (1.0, 2.0),
                0.5,
                (observation(0),),
                patient_relative_xyz_mm=(1.0, 2.0, 3.0),
            )

    def test_current_unresolved_sap_blocks_ground_truth(self) -> None:
        readiness = ready_map()
        readiness[("L4", LandmarkType.SUPERIOR_ARTICULAR_PROCESS, Laterality.LEFT)] = (
            LandmarkReadiness.UNRESOLVED
        )
        result = evaluate_landmark_evidence(
            facet_level=FacetLevel.L4_L5,
            laterality=Laterality.LEFT,
            hypotheses=(),
            ground_truth_readiness=readiness,
            observability_validated=False,
        )
        self.assertEqual(result.status, DecisionStatus.BLOCKED_GROUND_TRUTH)
        self.assertFalse(result.target_region_localized)

    def test_ready_labels_without_observability_fail_closed(self) -> None:
        result = evaluate_landmark_evidence(
            facet_level=FacetLevel.L4_L5,
            laterality=Laterality.LEFT,
            hypotheses=(),
            ground_truth_readiness=ready_map(),
            observability_validated=False,
        )
        self.assertEqual(result.status, DecisionStatus.BLOCKED_OBSERVABILITY)

    def test_wrong_level_and_side_do_not_satisfy_requirements(self) -> None:
        rows = (
            hypothesis("L3", LandmarkType.SUPERIOR_ARTICULAR_PROCESS),
            hypothesis("L4", LandmarkType.TRANSVERSE_PROCESS, side=Laterality.RIGHT),
        )
        result = evaluate_landmark_evidence(
            facet_level=FacetLevel.L4_L5,
            laterality=Laterality.LEFT,
            hypotheses=rows,
            ground_truth_readiness=ready_map(),
            observability_validated=True,
        )
        self.assertEqual(result.status, DecisionStatus.PARTIAL_LANDMARKS)

    def test_uncalibrated_uncertainty_abstains(self) -> None:
        rows = (
            hypothesis("L4", LandmarkType.SUPERIOR_ARTICULAR_PROCESS, calibrated=False),
            hypothesis("L4", LandmarkType.TRANSVERSE_PROCESS),
            hypothesis("L5", LandmarkType.SUPERIOR_ARTICULAR_PROCESS),
            hypothesis("L5", LandmarkType.TRANSVERSE_PROCESS),
        )
        result = evaluate_landmark_evidence(
            facet_level=FacetLevel.L4_L5,
            laterality=Laterality.LEFT,
            hypotheses=rows,
            ground_truth_readiness=ready_map(),
            observability_validated=True,
            anatomical_consistency=consistency_records(),
        )
        self.assertEqual(result.status, DecisionStatus.ABSTAIN)
        self.assertTrue(any("uncalibrated" in reason for reason in result.failure_reasons))

    def test_missing_mapping_allows_landmarks_but_not_target_region(self) -> None:
        rows = (
            hypothesis("L4", LandmarkType.SUPERIOR_ARTICULAR_PROCESS),
            hypothesis("L4", LandmarkType.TRANSVERSE_PROCESS),
            hypothesis("L5", LandmarkType.SUPERIOR_ARTICULAR_PROCESS),
            hypothesis("L5", LandmarkType.TRANSVERSE_PROCESS),
        )
        result = evaluate_landmark_evidence(
            facet_level=FacetLevel.L4_L5,
            laterality=Laterality.LEFT,
            hypotheses=rows,
            ground_truth_readiness=ready_map(),
            observability_validated=True,
            anatomical_consistency=consistency_records(),
        )
        self.assertEqual(result.status, DecisionStatus.LANDMARKS_LOCALIZED)
        self.assertFalse(result.target_region_localized)
        self.assertEqual(result.target_regions, {})

    def test_even_declared_mapping_cannot_fabricate_geometry(self) -> None:
        rows = (
            hypothesis("L4", LandmarkType.SUPERIOR_ARTICULAR_PROCESS),
            hypothesis("L4", LandmarkType.TRANSVERSE_PROCESS),
            hypothesis("L5", LandmarkType.SUPERIOR_ARTICULAR_PROCESS),
            hypothesis("L5", LandmarkType.TRANSVERSE_PROCESS),
        )
        mappings = {
            branch: TargetRegionMapping(branch, "reviewed-reference", True)
            for branch in (BranchIdentity.L3_MEDIAL_BRANCH, BranchIdentity.L4_MEDIAL_BRANCH)
        }
        result = evaluate_landmark_evidence(
            facet_level=FacetLevel.L4_L5,
            laterality=Laterality.LEFT,
            hypotheses=rows,
            ground_truth_readiness=ready_map(),
            observability_validated=True,
            anatomical_consistency=consistency_records(),
            target_mappings=mappings,
        )
        self.assertEqual(result.status, DecisionStatus.BLOCKED_GROUND_TRUTH)
        self.assertFalse(result.target_region_localized)

    def test_conflicting_hypotheses_abstain(self) -> None:
        first = hypothesis("L4", LandmarkType.SUPERIOR_ARTICULAR_PROCESS)
        conflicting = LandmarkHypothesis(
            landmark_type=LandmarkType.SUPERIOR_ARTICULAR_PROCESS,
            vertebral_level="L4",
            laterality=Laterality.LEFT,
            image_xy_px=(99.0, 99.0),
            confidence=0.8,
            evidence=(observation(0), observation(1)),
            patient_relative_xyz_mm=(8.0, 9.0, 10.0),
            covariance_mm2=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            transform_validated=True,
            calibration=CalibrationRecord("calibration-v1", True, False),
        )
        result = evaluate_landmark_evidence(
            facet_level=FacetLevel.L4_L5,
            laterality=Laterality.LEFT,
            hypotheses=(first, conflicting),
            ground_truth_readiness=ready_map(),
            observability_validated=True,
            anatomical_consistency=consistency_records(),
        )
        self.assertEqual(result.status, DecisionStatus.ABSTAIN)
        self.assertTrue(any("conflicting" in reason for reason in result.failure_reasons))

    def test_missing_anatomical_consistency_abstains(self) -> None:
        rows = (
            hypothesis("L4", LandmarkType.SUPERIOR_ARTICULAR_PROCESS),
            hypothesis("L4", LandmarkType.TRANSVERSE_PROCESS),
            hypothesis("L5", LandmarkType.SUPERIOR_ARTICULAR_PROCESS),
            hypothesis("L5", LandmarkType.TRANSVERSE_PROCESS),
        )
        result = evaluate_landmark_evidence(
            facet_level=FacetLevel.L4_L5,
            laterality=Laterality.LEFT,
            hypotheses=rows,
            ground_truth_readiness=ready_map(),
            observability_validated=True,
        )
        self.assertEqual(result.status, DecisionStatus.ABSTAIN)
        self.assertFalse(result.anatomical_consistency)


if __name__ == "__main__":
    unittest.main()
