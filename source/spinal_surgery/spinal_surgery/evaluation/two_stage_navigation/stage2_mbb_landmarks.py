"""Fail-closed contracts for experimental MBB-oriented Stage 2 research.

This module does not perform semantic landmark inference.  It defines the actor-visible
evidence, anatomy taxonomy, sufficiency checks, and target-region boundary needed before a
validated detector can be integrated.  Privileged CT, segmentation, and evaluator data must
remain in offline annotation/evaluation code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import assert_no_privileged_fields


class Laterality(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class FacetLevel(str, Enum):
    L4_L5 = "L4-L5"
    L5_S1 = "L5-S1"


class BranchIdentity(str, Enum):
    L3_MEDIAL_BRANCH = "L3_MEDIAL_BRANCH"
    L4_MEDIAL_BRANCH = "L4_MEDIAL_BRANCH"
    L5_DORSAL_RAMUS = "L5_DORSAL_RAMUS"


class LandmarkType(str, Enum):
    SUPERIOR_ARTICULAR_PROCESS = "SUPERIOR_ARTICULAR_PROCESS"
    TRANSVERSE_PROCESS = "TRANSVERSE_PROCESS"
    SACRAL_ALA = "SACRAL_ALA"


class LandmarkReadiness(str, Enum):
    READY = "READY"
    PROVISIONAL = "PROVISIONAL"
    UNRESOLVED = "UNRESOLVED"
    MISSING = "MISSING"


class DecisionStatus(str, Enum):
    INFORMATIVE_VIEW = "INFORMATIVE_VIEW"
    LANDMARKS_LOCALIZED = "LANDMARKS_LOCALIZED"
    PARTIAL_LANDMARKS = "PARTIAL_LANDMARKS"
    TARGET_REGION_LOCALIZED = "TARGET_REGION_LOCALIZED"
    ABSTAIN = "ABSTAIN"
    BLOCKED_GROUND_TRUTH = "BLOCKED_GROUND_TRUTH"
    BLOCKED_OBSERVABILITY = "BLOCKED_OBSERVABILITY"
    INVALID_ACQUISITION = "INVALID_ACQUISITION"


@dataclass(frozen=True)
class RequiredLandmark:
    branch: BranchIdentity
    vertebral_level: str
    landmark_type: LandmarkType
    laterality: Laterality


@dataclass(frozen=True)
class AnatomicalSite:
    """Procedure-related bony site; not a nerve coordinate or injection target."""

    branch: BranchIdentity
    laterality: Laterality
    required_landmarks: tuple[RequiredLandmark, ...]


def required_sites(facet_level: FacetLevel, laterality: Laterality) -> tuple[AnatomicalSite, ...]:
    """Return the branch-specific anatomy for a unilateral facet-joint task.

    L4-L5 is supplied by the L3 and L4 medial branches.  The L5 dorsal ramus is
    intentionally represented only in the distinct L5-S1 taxonomy.
    """

    if not isinstance(laterality, Laterality):
        raise TypeError("laterality must be an explicit Laterality")
    if facet_level is FacetLevel.L4_L5:
        definitions = (
            (
                BranchIdentity.L3_MEDIAL_BRANCH,
                "L4",
                (
                    LandmarkType.SUPERIOR_ARTICULAR_PROCESS,
                    LandmarkType.TRANSVERSE_PROCESS,
                ),
            ),
            (
                BranchIdentity.L4_MEDIAL_BRANCH,
                "L5",
                (
                    LandmarkType.SUPERIOR_ARTICULAR_PROCESS,
                    LandmarkType.TRANSVERSE_PROCESS,
                ),
            ),
        )
    elif facet_level is FacetLevel.L5_S1:
        definitions = (
            (
                BranchIdentity.L4_MEDIAL_BRANCH,
                "L5",
                (
                    LandmarkType.SUPERIOR_ARTICULAR_PROCESS,
                    LandmarkType.TRANSVERSE_PROCESS,
                ),
            ),
            (
                BranchIdentity.L5_DORSAL_RAMUS,
                "S1",
                (
                    LandmarkType.SUPERIOR_ARTICULAR_PROCESS,
                    LandmarkType.SACRAL_ALA,
                ),
            ),
        )
    else:  # pragma: no cover - Enum currently exhausts this branch.
        raise ValueError(f"unsupported facet level {facet_level!r}")
    return tuple(
        AnatomicalSite(
            branch=branch,
            laterality=laterality,
            required_landmarks=tuple(
                RequiredLandmark(branch, level, landmark_type, laterality)
                for landmark_type in landmark_types
            ),
        )
        for branch, level, landmark_types in definitions
    )

@dataclass(frozen=True)
class ObservationEvidence:
    """Causal actor-visible provenance for one acquired image."""

    session_id: str
    frame_sequence: int
    producer_sequence: int
    frame_sha256: str
    executed_relative_pose: tuple[float, float, float]
    transition_valid: bool
    image_valid: bool
    producer_advanced: bool

    def __post_init__(self) -> None:
        if not self.session_id or self.frame_sequence < 0 or self.producer_sequence < 0:
            raise ValueError("invalid observation identity")
        if len(self.frame_sha256) != 64:
            raise ValueError("frame_sha256 must contain 64 hexadecimal characters")
        try:
            int(self.frame_sha256, 16)
        except ValueError as exc:
            raise ValueError("frame_sha256 must be hexadecimal") from exc
        if len(self.executed_relative_pose) != 3 or not all(
            math.isfinite(value) for value in self.executed_relative_pose
        ):
            raise ValueError("executed_relative_pose must contain three finite values")


@dataclass(frozen=True)
class CalibrationRecord:
    reference_id: str
    coverage_tested: bool
    patient_disjoint: bool

    @property
    def valid_for_uncertainty_gate(self) -> bool:
        return bool(self.reference_id and self.coverage_tested)


@dataclass(frozen=True)
class LandmarkHypothesis:
    landmark_type: LandmarkType
    vertebral_level: str
    laterality: Laterality
    image_xy_px: tuple[float, float]
    confidence: float
    evidence: tuple[ObservationEvidence, ...]
    patient_relative_xyz_mm: tuple[float, float, float] | None = None
    covariance_mm2: tuple[tuple[float, float, float], ...] | None = None
    transform_validated: bool = False
    calibration: CalibrationRecord | None = None

    def __post_init__(self) -> None:
        if len(self.image_xy_px) != 2 or not all(math.isfinite(v) for v in self.image_xy_px):
            raise ValueError("image coordinates must be finite")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not self.evidence:
            raise ValueError("a landmark hypothesis requires actor-visible evidence")
        if self.patient_relative_xyz_mm is not None:
            if not self.transform_validated:
                raise ValueError("3-D localization requires a validated coordinate transform")
            if len(self.patient_relative_xyz_mm) != 3 or not all(
                math.isfinite(v) for v in self.patient_relative_xyz_mm
            ):
                raise ValueError("patient-relative coordinates must be finite")
        if self.covariance_mm2 is not None:
            covariance = np.asarray(self.covariance_mm2, dtype=np.float64)
            if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
                raise ValueError("covariance must be a finite 3x3 matrix")
            if not np.allclose(covariance, covariance.T, atol=1e-9):
                raise ValueError("covariance must be symmetric")
            if np.min(np.linalg.eigvalsh(covariance)) < -1e-9:
                raise ValueError("covariance must be positive semidefinite")

    @property
    def key(self) -> tuple[str, LandmarkType, Laterality]:
        return self.vertebral_level, self.landmark_type, self.laterality

    @property
    def has_calibrated_spatial_uncertainty(self) -> bool:
        return bool(
            self.patient_relative_xyz_mm is not None
            and self.covariance_mm2 is not None
            and self.calibration is not None
            and self.calibration.valid_for_uncertainty_gate
        )

    @property
    def distinct_viewpoint_count(self) -> int:
        poses_by_hash: dict[str, tuple[float, float, float]] = {}
        for item in self.evidence:
            if not (
                item.transition_valid
                and item.image_valid
                and item.producer_advanced
            ):
                continue
            poses_by_hash.setdefault(item.frame_sha256, item.executed_relative_pose)
        return len(set(poses_by_hash.values()))


@dataclass
class LandmarkCandidateMemory:
    """Retain multiple causal views without selecting one global best image."""

    session_id: str
    observations: list[ObservationEvidence] = field(default_factory=list)
    hypotheses: list[LandmarkHypothesis] = field(default_factory=list)

    def add_observation(self, observation: ObservationEvidence) -> None:
        if observation.session_id != self.session_id:
            raise ValueError("reset or trajectory boundary crossed")
        if self.observations and observation.frame_sequence <= self.observations[-1].frame_sequence:
            raise ValueError("future/reordered frames cannot enter causal memory")
        self.observations.append(observation)

    def add_hypothesis(self, hypothesis: LandmarkHypothesis) -> None:
        known = {(row.frame_sequence, row.frame_sha256) for row in self.observations}
        if any((row.frame_sequence, row.frame_sha256) not in known for row in hypothesis.evidence):
            raise ValueError("hypothesis refers to an unavailable or future observation")
        if any(row.session_id != self.session_id for row in hypothesis.evidence):
            raise ValueError("hypothesis crosses a reset boundary")
        self.hypotheses.append(hypothesis)


@dataclass(frozen=True)
class TargetRegionMapping:
    branch: BranchIdentity
    anatomical_reference: str
    validated: bool
    uses_vertebral_centroid: bool = False

    def __post_init__(self) -> None:
        if self.uses_vertebral_centroid:
            raise ValueError("a vertebral centroid cannot substitute for an MBB target region")


@dataclass(frozen=True)
class AnatomicalConsistencyRecord:
    branch: BranchIdentity
    reference_id: str
    validated: bool
    consistent: bool


@dataclass(frozen=True)
class Stage2MBBResult:
    facet_level: FacetLevel
    laterality: Laterality
    branches: tuple[BranchIdentity, ...]
    landmarks: tuple[LandmarkHypothesis, ...]
    supporting_frame_ids: tuple[str, ...]
    supporting_viewpoint_count: int
    anatomical_consistency: bool
    target_regions: Mapping[BranchIdentity, Any]
    target_region_localized: bool
    status: DecisionStatus
    failure_reasons: tuple[str, ...]


STRICT_FORBIDDEN_MBB_ACTOR_FIELDS = frozenset(
    {
        "ct",
        "ct_volume",
        "raw_segmentation",
        "oracle_landmark_positions",
        "evaluator_goal_pose",
        "evaluator_geometric_errors",
        "ground_truth_target_coordinates",
        "ground_truth_validity",
        "future_observations",
    }
)


def validate_mbb_actor_payload(payload: Mapping[str, Any]) -> None:
    """Apply the shared information boundary plus MBB-specific forbidden fields."""

    assert_no_privileged_fields(payload)

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                if key_text in STRICT_FORBIDDEN_MBB_ACTOR_FIELDS:
                    raise ValueError(f"privileged MBB actor field: {path}.{key_text}")
                visit(child, f"{path}.{key_text}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(payload, "payload")


def _required_readiness(
    sites: Iterable[AnatomicalSite],
    readiness: Mapping[tuple[str, LandmarkType, Laterality], LandmarkReadiness],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for site in sites:
        for required in site.required_landmarks:
            status = readiness.get(
                (required.vertebral_level, required.landmark_type, required.laterality),
                LandmarkReadiness.MISSING,
            )
            if status is not LandmarkReadiness.READY:
                reasons.append(
                    f"{required.vertebral_level} {required.laterality.value} "
                    f"{required.landmark_type.value}: {status.value}"
                )
    return not reasons, reasons


def evaluate_landmark_evidence(
    *,
    facet_level: FacetLevel,
    laterality: Laterality,
    hypotheses: Iterable[LandmarkHypothesis],
    ground_truth_readiness: Mapping[
        tuple[str, LandmarkType, Laterality], LandmarkReadiness
    ],
    observability_validated: bool,
    anatomical_consistency: Mapping[
        BranchIdentity, AnatomicalConsistencyRecord
    ] | None = None,
    target_mappings: Mapping[BranchIdentity, TargetRegionMapping] | None = None,
) -> Stage2MBBResult:
    """Evaluate evidence without inventing numerical confidence thresholds."""

    sites = required_sites(facet_level, laterality)
    branches = tuple(site.branch for site in sites)
    rows = tuple(hypotheses)
    readiness_ok, readiness_reasons = _required_readiness(sites, ground_truth_readiness)
    if not readiness_ok:
        return Stage2MBBResult(
            facet_level, laterality, branches, rows, (), 0, False, {}, False,
            DecisionStatus.BLOCKED_GROUND_TRUTH, tuple(readiness_reasons),
        )
    if not observability_validated:
        return Stage2MBBResult(
            facet_level, laterality, branches, rows, (), 0, False, {}, False,
            DecisionStatus.BLOCKED_OBSERVABILITY,
            ("actor-visible SAP/TP observability has not been validated",),
        )

    expected = {
        (item.vertebral_level, item.landmark_type, item.laterality)
        for site in sites
        for item in site.required_landmarks
    }
    grouped: dict[
        tuple[str, LandmarkType, Laterality], list[LandmarkHypothesis]
    ] = {}
    for row in rows:
        if row.key in expected:
            grouped.setdefault(row.key, []).append(row)
    conflicts = [
        key
        for key, candidates in grouped.items()
        if len(
            {
                (candidate.image_xy_px, candidate.patient_relative_xyz_mm)
                for candidate in candidates
            }
        )
        > 1
    ]
    if conflicts:
        return Stage2MBBResult(
            facet_level, laterality, branches, rows, (), 0, False, {}, False,
            DecisionStatus.ABSTAIN,
            tuple(
                f"conflicting hypotheses: {level} {side.value} {kind.value}"
                for level, kind, side in conflicts
            ),
        )
    provided = {key: candidates[0] for key, candidates in grouped.items()}
    missing = sorted(expected - set(provided), key=lambda item: (item[0], item[1].value))
    if missing:
        return Stage2MBBResult(
            facet_level, laterality, branches, rows, (), 0, False, {}, False,
            DecisionStatus.PARTIAL_LANDMARKS,
            tuple(f"missing {level} {side.value} {kind.value}" for level, kind, side in missing),
        )

    uncalibrated = [
        f"{row.vertebral_level} {row.landmark_type.value}"
        for row in provided.values()
        if not row.has_calibrated_spatial_uncertainty
    ]
    insufficient_views = [
        f"{row.vertebral_level} {row.landmark_type.value}"
        for row in provided.values()
        if row.distinct_viewpoint_count < 2
    ]
    if uncalibrated or insufficient_views:
        reasons = [f"uncalibrated spatial uncertainty: {item}" for item in uncalibrated]
        reasons.extend(f"insufficient distinct viewpoints: {item}" for item in insufficient_views)
        return Stage2MBBResult(
            facet_level, laterality, branches, rows, (), 0, False, {}, False,
            DecisionStatus.ABSTAIN, tuple(reasons),
        )

    consistency = anatomical_consistency or {}
    missing_consistency = [
        branch.value
        for branch in branches
        if branch not in consistency or not consistency[branch].validated
    ]
    contradicted = [
        branch.value
        for branch in branches
        if branch in consistency
        and consistency[branch].validated
        and not consistency[branch].consistent
    ]
    if missing_consistency or contradicted:
        reasons = [
            f"anatomical consistency not validated: {branch}"
            for branch in missing_consistency
        ]
        reasons.extend(
            f"anatomical consistency contradicted: {branch}" for branch in contradicted
        )
        return Stage2MBBResult(
            facet_level, laterality, branches, rows, (), 0, False, {}, False,
            DecisionStatus.ABSTAIN, tuple(reasons),
        )

    frame_ids = tuple(
        sorted(
            {
                item.frame_sha256
                for row in provided.values()
                for item in row.evidence
            }
        )
    )
    viewpoint_count = len(
        {item.executed_relative_pose for row in provided.values() for item in row.evidence}
    )
    mappings = target_mappings or {}
    missing_mappings = [
        branch.value
        for branch in branches
        if branch not in mappings or not mappings[branch].validated
    ]
    if missing_mappings:
        return Stage2MBBResult(
            facet_level, laterality, branches, rows, frame_ids, viewpoint_count, True,
            {}, False, DecisionStatus.LANDMARKS_LOCALIZED,
            tuple(f"target-region mapping not validated: {branch}" for branch in missing_mappings),
        )

    # Mapping geometry is deliberately not implemented until independently reviewed
    # patient-specific anatomical relationships exist.
    return Stage2MBBResult(
        facet_level, laterality, branches, rows, frame_ids, viewpoint_count, True,
        {}, False, DecisionStatus.BLOCKED_GROUND_TRUTH,
        ("validated target-region geometry implementation is unavailable",),
    )
