"""Actor-owned incremental 3D map for the fixed-s0010 Stage 2 landmarks.

The map consumes only semantic observations derived from clean ultrasound and
the synchronized executed probe pose.  Anatomical level is supplied by the
frozen Stage 1 handoff.  No evaluator geometry is used here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal, Mapping, Sequence

import numpy as np

from .sonoanatomy_multiview import (
    SemanticCluster,
    SemanticObservation,
    cluster_observations,
)


LandmarkIdentity = Literal[
    "RIGHT_L4_TP",
    "RIGHT_L4_SAP",
    "RIGHT_L5_TP",
    "RIGHT_L5_SAP",
]
LandmarkStatus = Literal[
    "NOT_OBSERVED",
    "CANDIDATE",
    "AMBIGUOUS",
    "CONFIRMED",
    "REJECTED",
]

LANDMARK_IDENTITIES: tuple[LandmarkIdentity, ...] = (
    "RIGHT_L4_TP",
    "RIGHT_L4_SAP",
    "RIGHT_L5_TP",
    "RIGHT_L5_SAP",
)


@dataclass(frozen=True)
class IncrementalLandmarkMapConfig:
    association_radius_mm: float = 4.0
    minimum_supporting_views: int = 2
    minimum_view_translation_mm: float = 2.0
    minimum_view_rotation_deg: float = 1.0
    minimum_semantic_confidence: float = 0.5
    maximum_rms_uncertainty_mm: float = 4.0
    minimum_hypothesis_score_margin: float = 0.01
    minimum_sap_tp_distance_mm: float = 1.0
    maximum_sap_tp_distance_mm: float = 30.0
    maximum_tp_anchor_distance_mm: float = 10.0
    minimum_supporting_view_advantage: int = 2

    def __post_init__(self) -> None:
        positive = (
            self.association_radius_mm,
            self.minimum_supporting_views,
            self.minimum_view_translation_mm,
            self.minimum_view_rotation_deg,
            self.maximum_rms_uncertainty_mm,
            self.minimum_sap_tp_distance_mm,
            self.maximum_sap_tp_distance_mm,
            self.maximum_tp_anchor_distance_mm,
            self.minimum_supporting_view_advantage,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in positive):
            raise ValueError("landmark-map positive settings must be finite")
        if not 0.0 <= self.minimum_semantic_confidence <= 1.0:
            raise ValueError("semantic confidence must lie in [0, 1]")
        if self.minimum_hypothesis_score_margin < 0.0:
            raise ValueError("hypothesis score margin must be nonnegative")
        if self.minimum_sap_tp_distance_mm >= self.maximum_sap_tp_distance_mm:
            raise ValueError("SAP/TP distance interval is empty")


@dataclass(frozen=True)
class LandmarkSupportObservation:
    frame_sequence: int
    frame_sha256: str
    viewpoint_id: str
    row: float
    column: float
    coordinate_mm: tuple[float, float, float]
    confidence: float


@dataclass(frozen=True)
class LandmarkHypothesis:
    identity: LandmarkIdentity
    status: LandmarkStatus
    position_mm: tuple[float, float, float] | None
    rms_uncertainty_mm: float | None
    maximum_residual_mm: float | None
    semantic_confidence: float
    hypothesis_score: float
    supporting_frame_sequences: tuple[int, ...]
    supporting_frame_hashes: tuple[str, ...]
    supporting_viewpoint_ids: tuple[str, ...]
    supporting_observations: tuple[LandmarkSupportObservation, ...]
    observation_count: int
    competing_hypothesis_count: int
    reason: str
    level_source: str = "FROZEN_STAGE1_HANDOFF"


@dataclass(frozen=True)
class LandmarkRelation:
    level: Literal["L4", "L5"]
    status: Literal["INCOMPLETE", "CONSISTENT", "INCONSISTENT"]
    sap_tp_distance_mm: float | None
    reason: str


@dataclass(frozen=True)
class LandmarkTrajectoryDecision:
    status: Literal["ACCEPT", "ABSTAIN"]
    trajectory_id: str | None
    confirmed_landmark_count: int
    relation_status: Literal["INCOMPLETE", "CONSISTENT", "INCONSISTENT"]
    score: float
    reason: str


ObservationKey = tuple[int, str, float, float]


@dataclass
class _ReacquisitionAssociation:
    """Actor-visible link from a frozen hypothesis to fresh revisit evidence."""

    reference_position_mm: tuple[float, float, float]
    source_observation_keys: set[ObservationKey]
    fresh_observation_keys: set[ObservationKey]


def _identity(level: str, semantic_class: str) -> LandmarkIdentity:
    value = f"RIGHT_{level}_{semantic_class}"
    if value not in LANDMARK_IDENTITIES:
        raise ValueError(f"unsupported landmark identity: {value}")
    return value  # type: ignore[return-value]


def _cluster_quality(
    cluster: SemanticCluster,
    observations: Sequence[SemanticObservation],
    config: IncrementalLandmarkMapConfig,
) -> tuple[float, float, float]:
    coordinates = np.asarray(
        [observations[index].coordinate_mm for index in cluster.observation_indices],
        dtype=np.float64,
    )
    centroid = np.asarray(cluster.centroid_mm, dtype=np.float64)
    residuals = np.linalg.norm(coordinates - centroid, axis=1)
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    maximum = float(np.max(residuals))
    support = min(
        len(cluster.supporting_frames) / float(config.minimum_supporting_views),
        1.0,
    )
    compactness = max(0.0, 1.0 - rms / config.maximum_rms_uncertainty_mm)
    score = float(0.55 * cluster.confidence + 0.25 * support + 0.20 * compactness)
    return score, rms, maximum


def _observation_key(observation: SemanticObservation) -> ObservationKey:
    return (
        observation.frame_sequence,
        observation.frame_sha256,
        observation.row,
        observation.column,
    )


def _viewpoints_are_distinct(
    left: SemanticObservation,
    right: SemanticObservation,
    config: IncrementalLandmarkMapConfig,
) -> bool:
    translation_mm = 1000.0 * float(
        np.linalg.norm(
            np.asarray(left.viewpoint_position_m)
            - np.asarray(right.viewpoint_position_m)
        )
    )
    left_rotation = np.asarray(left.viewpoint_rotation_columns)
    right_rotation = np.asarray(right.viewpoint_rotation_columns)
    cosine = np.clip(
        (np.trace(left_rotation.T @ right_rotation) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    rotation_deg = math.degrees(math.acos(float(cosine)))
    return bool(
        translation_mm >= config.minimum_view_translation_mm
        or rotation_deg >= config.minimum_view_rotation_deg
    )


class IncrementalAnatomicalLandmarkMap:
    """Persistent shared map for RIGHT L4/L5 SAP and TP hypotheses."""

    def __init__(self, config: IncrementalLandmarkMapConfig) -> None:
        self.config = config
        self._observations: dict[LandmarkIdentity, list[SemanticObservation]] = {
            identity: [] for identity in LANDMARK_IDENTITIES
        }
        self._rejections: dict[LandmarkIdentity, str] = {}
        self._anchors: dict[LandmarkIdentity, tuple[float, float, float]] = {}
        self._reacquisition_associations: dict[
            LandmarkIdentity, list[_ReacquisitionAssociation]
        ] = {identity: [] for identity in LANDMARK_IDENTITIES}

    def set_actor_anchor(
        self,
        identity: LandmarkIdentity,
        coordinate_mm: Sequence[float],
    ) -> None:
        if not identity.endswith("_TP"):
            raise ValueError("only independently confirmed TP may anchor the map")
        coordinate = np.asarray(coordinate_mm, dtype=np.float64)
        if coordinate.shape != (3,) or not np.all(np.isfinite(coordinate)):
            raise ValueError("actor anchor must contain three finite coordinates")
        self._anchors[identity] = tuple(float(value) for value in coordinate)

    def add_observations(
        self,
        level: Literal["L4", "L5"],
        observations: Sequence[SemanticObservation],
    ) -> None:
        """Add fresh actor observations associated with a Stage 1 level."""

        for observation in observations:
            if observation.semantic_class not in {"TP", "SAP"}:
                continue
            identity = _identity(level, observation.semantic_class)
            key = (observation.frame_sequence, observation.frame_sha256, observation.row, observation.column)
            existing = {
                (row.frame_sequence, row.frame_sha256, row.row, row.column)
                for row in self._observations[identity]
            }
            if key not in existing:
                self._observations[identity].append(observation)

    def add_reacquisition_observations(
        self,
        level: Literal["L4", "L5"],
        semantic_class: Literal["TP", "SAP"],
        *,
        reference_position_mm: Sequence[float],
        reference_frame_hashes: Sequence[str],
        observations: Sequence[SemanticObservation],
    ) -> int:
        """Associate fresh revisit evidence with one frozen actor hypothesis.

        The reference must already exist in the actor map.  Fresh predictions
        outside the frozen association radius are ignored, and the reference
        is never moved or promoted to an actor anchor by this operation.
        """

        identity = _identity(level, semantic_class)
        reference = np.asarray(reference_position_mm, dtype=np.float64)
        if reference.shape != (3,) or not np.all(np.isfinite(reference)):
            raise ValueError("reacquisition reference must contain three finite values")
        reference_hashes = set(reference_frame_hashes)
        if not reference_hashes:
            raise ValueError("reacquisition requires historical supporting frames")

        historical = self._observations[identity]
        source_keys: set[ObservationKey] = set()
        for frame_hash in sorted(reference_hashes):
            candidates = [
                row for row in historical if row.frame_sha256 == frame_hash
            ]
            if not candidates:
                raise ValueError("reacquisition source is absent from the actor map")
            closest = min(
                candidates,
                key=lambda row: float(
                    np.linalg.norm(np.asarray(row.coordinate_mm) - reference)
                ),
            )
            source_keys.add(_observation_key(closest))

        accepted_by_frame: dict[tuple[int, str], SemanticObservation] = {}
        for observation in observations:
            if observation.semantic_class != semantic_class:
                raise ValueError("reacquisition semantic class does not match")
            distance = float(
                np.linalg.norm(np.asarray(observation.coordinate_mm) - reference)
            )
            if distance > self.config.association_radius_mm:
                continue
            frame_key = (observation.frame_sequence, observation.frame_sha256)
            previous = accepted_by_frame.get(frame_key)
            if previous is None or distance < float(
                np.linalg.norm(np.asarray(previous.coordinate_mm) - reference)
            ):
                accepted_by_frame[frame_key] = observation

        if not accepted_by_frame:
            return 0
        self.add_observations(level, tuple(accepted_by_frame.values()))
        fresh_keys = {_observation_key(row) for row in accepted_by_frame.values()}
        associations = self._reacquisition_associations[identity]
        matching = next(
            (
                row
                for row in associations
                if row.source_observation_keys == source_keys
                and np.allclose(row.reference_position_mm, reference, atol=1e-6)
            ),
            None,
        )
        if matching is None:
            associations.append(
                _ReacquisitionAssociation(
                    reference_position_mm=tuple(float(value) for value in reference),
                    source_observation_keys=source_keys,
                    fresh_observation_keys=fresh_keys,
                )
            )
        else:
            matching.fresh_observation_keys.update(fresh_keys)
        return len(fresh_keys)

    def _clusters(
        self,
        identity: LandmarkIdentity,
        observations: Sequence[SemanticObservation],
    ) -> tuple[SemanticCluster, ...]:
        clusters = list(
            cluster_observations(
                observations,
                association_radius_mm=self.config.association_radius_mm,
                minimum_view_translation_mm=self.config.minimum_view_translation_mm,
                minimum_view_rotation_deg=self.config.minimum_view_rotation_deg,
            )
        )
        key_to_index = {
            _observation_key(observation): index
            for index, observation in enumerate(observations)
        }
        for association in self._reacquisition_associations[identity]:
            source_indices = [
                key_to_index[key]
                for key in association.source_observation_keys
                if key in key_to_index
            ]
            fresh_indices = [
                key_to_index[key]
                for key in association.fresh_observation_keys
                if key in key_to_index
            ]
            accepted = list(source_indices)
            reference = np.asarray(association.reference_position_mm)
            for index in sorted(
                fresh_indices,
                key=lambda item: (
                    float(
                        np.linalg.norm(
                            np.asarray(observations[item].coordinate_mm) - reference
                        )
                    ),
                    -observations[item].confidence,
                    observations[item].frame_sequence,
                ),
            ):
                observation = observations[index]
                if any(
                    observation.frame_sequence == observations[item].frame_sequence
                    or observation.frame_sha256 == observations[item].frame_sha256
                    or observation.viewpoint_id == observations[item].viewpoint_id
                    or not _viewpoints_are_distinct(
                        observation, observations[item], self.config
                    )
                    for item in accepted
                ):
                    continue
                accepted.append(index)
            associated_indices = set(source_indices + fresh_indices)
            if len(accepted) < 2:
                continue
            clusters = [
                cluster
                for cluster in clusters
                if not associated_indices.intersection(cluster.observation_indices)
            ]
            coordinates = np.asarray(
                [observations[index].coordinate_mm for index in accepted]
            )
            clusters.append(
                SemanticCluster(
                    semantic_class=observations[accepted[0]].semantic_class,
                    observation_indices=tuple(accepted),
                    supporting_frames=tuple(
                        sorted({observations[index].frame_sequence for index in accepted})
                    ),
                    centroid_mm=tuple(
                        float(value) for value in np.mean(coordinates, axis=0)
                    ),
                    confidence=float(
                        np.mean([observations[index].confidence for index in accepted])
                    ),
                )
            )
        return tuple(clusters)

    def reject(self, identity: LandmarkIdentity, reason: str) -> None:
        if not reason:
            raise ValueError("rejection reason is required")
        self._rejections[identity] = reason

    def observations(self, identity: LandmarkIdentity) -> tuple[SemanticObservation, ...]:
        return tuple(self._observations[identity])

    def _hypothesis(self, identity: LandmarkIdentity) -> LandmarkHypothesis:
        observations = self._observations[identity]
        if identity in self._rejections:
            return LandmarkHypothesis(
                identity=identity,
                status="REJECTED",
                position_mm=None,
                rms_uncertainty_mm=None,
                maximum_residual_mm=None,
                semantic_confidence=0.0,
                hypothesis_score=0.0,
                supporting_frame_sequences=(),
                supporting_frame_hashes=(),
                supporting_viewpoint_ids=(),
                supporting_observations=(),
                observation_count=len(observations),
                competing_hypothesis_count=0,
                reason=self._rejections[identity],
            )
        if not observations:
            return LandmarkHypothesis(
                identity=identity,
                status="NOT_OBSERVED",
                position_mm=None,
                rms_uncertainty_mm=None,
                maximum_residual_mm=None,
                semantic_confidence=0.0,
                hypothesis_score=0.0,
                supporting_frame_sequences=(),
                supporting_frame_hashes=(),
                supporting_viewpoint_ids=(),
                supporting_observations=(),
                observation_count=0,
                competing_hypothesis_count=0,
                reason="NO_SEMANTIC_OBSERVATION",
            )
        clusters = self._clusters(identity, observations)
        ranked = sorted(
            (
                (*_cluster_quality(cluster, observations, self.config), index, cluster)
                for index, cluster in enumerate(clusters)
            ),
            key=lambda row: (-row[0], -len(row[4].supporting_frames), row[1], row[3]),
        )
        anchor = self._anchors.get(identity)
        if anchor is not None:
            anchor_coordinate = np.asarray(anchor, dtype=np.float64)
            ranked = sorted(
                ranked,
                key=lambda row: (
                    int(
                        float(
                            np.linalg.norm(
                                np.asarray(row[4].centroid_mm) - anchor_coordinate
                            )
                        )
                        > self.config.maximum_tp_anchor_distance_mm
                    ),
                    int(
                        len(row[4].supporting_frames)
                        < self.config.minimum_supporting_views
                    ),
                    float(
                        np.linalg.norm(
                            np.asarray(row[4].centroid_mm) - anchor_coordinate
                        )
                    ),
                    -row[0],
                    -len(row[4].supporting_frames),
                    row[3],
                ),
            )
        elif len(ranked) > 1:
            top_score = ranked[0][0]
            close = [
                row
                for row in ranked
                if top_score - row[0] < self.config.minimum_hypothesis_score_margin
            ]
            by_support = sorted(
                close,
                key=lambda row: (-len(row[4].supporting_frames), -row[0], row[3]),
            )
            if (
                len(by_support) > 1
                and len(by_support[0][4].supporting_frames)
                >= len(by_support[1][4].supporting_frames)
                + self.config.minimum_supporting_view_advantage
            ):
                selected_row = by_support[0]
                ranked = [selected_row] + [
                    row for row in ranked if row is not selected_row
                ]
        score, rms, maximum, _, selected = ranked[0]
        support = len(selected.supporting_frames)
        eligible = bool(
            support >= self.config.minimum_supporting_views
            and selected.confidence >= self.config.minimum_semantic_confidence
            and rms <= self.config.maximum_rms_uncertainty_mm
        )
        anchor_distance = (
            None
            if anchor is None
            else float(
                np.linalg.norm(
                    np.asarray(selected.centroid_mm) - np.asarray(anchor)
                )
            )
        )
        if anchor_distance is not None:
            eligible = bool(
                eligible
                and anchor_distance
                <= self.config.maximum_tp_anchor_distance_mm
            )
        competing = [
            row
            for row in ranked[1:]
            if len(row[4].supporting_frames) >= self.config.minimum_supporting_views
        ]
        close_competitor = bool(
            competing
            and score - competing[0][0] < self.config.minimum_hypothesis_score_margin
            and support
            < len(competing[0][4].supporting_frames)
            + self.config.minimum_supporting_view_advantage
        )
        if (
            anchor_distance is not None
            and anchor_distance > self.config.maximum_tp_anchor_distance_mm
        ):
            status: LandmarkStatus = "REJECTED"
            reason = "NO_SPATIAL_CLUSTER_NEAR_ACTOR_TP_ANCHOR"
        elif selected.confidence < self.config.minimum_semantic_confidence:
            status: LandmarkStatus = "REJECTED"
            reason = "SEMANTIC_CONFIDENCE_BELOW_GATE"
        elif close_competitor and anchor is None:
            status = "AMBIGUOUS"
            reason = "COMPETING_SPATIAL_HYPOTHESES"
        elif eligible:
            status = "CONFIRMED"
            reason = (
                "INDEPENDENT_SUPPORT_CONSISTENT_WITH_ACTOR_TP_ANCHOR"
                if anchor is not None
                else "INDEPENDENT_DISTINCT_POSE_SUPPORT"
            )
        elif support >= self.config.minimum_supporting_views:
            status = "AMBIGUOUS"
            reason = "SPATIAL_UNCERTAINTY_ABOVE_GATE"
        else:
            status = "CANDIDATE"
            reason = "ADDITIONAL_DISTINCT_POSE_REQUIRED"
        members = [observations[index] for index in selected.observation_indices]
        return LandmarkHypothesis(
            identity=identity,
            status=status,
            position_mm=tuple(float(value) for value in selected.centroid_mm),
            rms_uncertainty_mm=rms,
            maximum_residual_mm=maximum,
            semantic_confidence=float(selected.confidence),
            hypothesis_score=score,
            supporting_frame_sequences=tuple(row.frame_sequence for row in members),
            supporting_frame_hashes=tuple(row.frame_sha256 for row in members),
            supporting_viewpoint_ids=tuple(row.viewpoint_id for row in members),
            supporting_observations=tuple(
                LandmarkSupportObservation(
                    frame_sequence=row.frame_sequence,
                    frame_sha256=row.frame_sha256,
                    viewpoint_id=row.viewpoint_id,
                    row=row.row,
                    column=row.column,
                    coordinate_mm=row.coordinate_mm,
                    confidence=row.confidence,
                )
                for row in members
            ),
            observation_count=len(observations),
            competing_hypothesis_count=len(competing),
            reason=reason,
        )

    def landmark(self, identity: LandmarkIdentity) -> LandmarkHypothesis:
        return self._hypothesis(identity)

    def relation(self, level: Literal["L4", "L5"]) -> LandmarkRelation:
        tp = self._hypothesis(_identity(level, "TP"))
        sap = self._hypothesis(_identity(level, "SAP"))
        if tp.status != "CONFIRMED" or sap.status != "CONFIRMED":
            return LandmarkRelation(level, "INCOMPLETE", None, "LANDMARKS_NOT_BOTH_CONFIRMED")
        assert tp.position_mm is not None and sap.position_mm is not None
        distance = float(
            np.linalg.norm(np.asarray(tp.position_mm) - np.asarray(sap.position_mm))
        )
        if not self.config.minimum_sap_tp_distance_mm <= distance <= self.config.maximum_sap_tp_distance_mm:
            return LandmarkRelation(level, "INCONSISTENT", distance, "SAP_TP_DISTANCE_OUTSIDE_GATE")
        return LandmarkRelation(level, "CONSISTENT", distance, "CONFIRMED_IN_SHARED_3D_MAP")

    def level_confirmed(self, level: Literal["L4", "L5"]) -> bool:
        return self.relation(level).status == "CONSISTENT"

    def complete(self) -> bool:
        return all(self._hypothesis(identity).status == "CONFIRMED" for identity in LANDMARK_IDENTITIES) and all(
            self.relation(level).status == "CONSISTENT" for level in ("L4", "L5")
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": "sonogym-incremental-anatomical-landmark-map-v1",
            "coordinate_frame": "ACTOR_BASE_WORLD_MM",
            "actor_owned": True,
            "config": asdict(self.config),
            "landmarks": {
                identity: asdict(self._hypothesis(identity))
                for identity in LANDMARK_IDENTITIES
            },
            "relations": {
                level: asdict(self.relation(level)) for level in ("L4", "L5")
            },
            "actor_anchors_mm": dict(self._anchors),
            "reacquisition_association_count": {
                identity: len(self._reacquisition_associations[identity])
                for identity in LANDMARK_IDENTITIES
            },
            "complete": self.complete(),
        }


def select_landmark_trajectory(
    level: Literal["L4", "L5"],
    observations_by_trajectory: Mapping[str, Sequence[SemanticObservation]],
    config: IncrementalLandmarkMapConfig,
    *,
    anchors: Mapping[LandmarkIdentity, Sequence[float]] | None = None,
) -> LandmarkTrajectoryDecision:
    """Select the motion trajectory with the strongest independent landmarks."""

    ranked = []
    for trajectory_id, observations in observations_by_trajectory.items():
        mapping = IncrementalAnatomicalLandmarkMap(config)
        for identity, coordinate in (anchors or {}).items():
            mapping.set_actor_anchor(identity, coordinate)
        mapping.add_observations(level, observations)
        tp = mapping.landmark(_identity(level, "TP"))
        sap = mapping.landmark(_identity(level, "SAP"))
        relation = mapping.relation(level)
        confirmed = sum(row.status == "CONFIRMED" for row in (tp, sap))
        score = float(tp.hypothesis_score + sap.hypothesis_score)
        support = len(tp.supporting_frame_sequences) + len(sap.supporting_frame_sequences)
        ranked.append(
            (
                int(relation.status == "CONSISTENT"),
                confirmed,
                score,
                support,
                trajectory_id,
                relation,
            )
        )
    if not ranked:
        return LandmarkTrajectoryDecision(
            "ABSTAIN", None, 0, "INCOMPLETE", 0.0, "NO_SEMANTIC_TRAJECTORY"
        )
    best = max(ranked, key=lambda row: row[:5])
    if best[1] == 0:
        return LandmarkTrajectoryDecision(
            "ABSTAIN",
            None,
            0,
            best[5].status,
            best[2],
            "NO_INDEPENDENT_LANDMARK_CONFIRMED",
        )
    return LandmarkTrajectoryDecision(
        "ACCEPT",
        best[4],
        best[1],
        best[5].status,
        best[2],
        (
            "BOTH_LANDMARKS_AND_RELATION_SUPPORTED"
            if best[5].status == "CONSISTENT"
            else "PARTIAL_LANDMARK_SUPPORT_REQUIRES_RESCAN"
        ),
    )
