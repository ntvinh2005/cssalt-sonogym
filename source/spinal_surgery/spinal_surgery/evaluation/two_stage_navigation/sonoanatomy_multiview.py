"""Actor-side semantic cortical observations and multi-view pair decisions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Literal, Mapping, Sequence

import numpy as np
from scipy import ndimage

from .ultrasound_spatial_map import calibrated_image_coordinates_mm


SemanticClass = Literal["TP", "SAP", "OTHER_BONE"]


@dataclass(frozen=True)
class SemanticObservation:
    semantic_class: SemanticClass
    frame_sequence: int
    frame_sha256: str
    viewpoint_id: str
    viewpoint_position_m: tuple[float, float, float]
    viewpoint_rotation_columns: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    row: float
    column: float
    coordinate_mm: tuple[float, float, float]
    confidence: float
    pixel_count: int


@dataclass(frozen=True)
class SemanticCluster:
    semantic_class: SemanticClass
    observation_indices: tuple[int, ...]
    supporting_frames: tuple[int, ...]
    centroid_mm: tuple[float, float, float]
    confidence: float


@dataclass(frozen=True)
class PairHypothesis:
    tp_cluster_index: int
    sap_cluster_index: int
    surface_distance_mm: float
    score: float
    supporting_frames: tuple[int, ...]


@dataclass(frozen=True)
class PairDecision:
    status: Literal["ACCEPT", "ABSTAIN"]
    selected: PairHypothesis | None
    hypotheses: tuple[PairHypothesis, ...]
    reason: str


@dataclass(frozen=True)
class MultiviewConfig:
    association_radius_mm: float = 5.0
    minimum_supporting_frames_per_structure: int = 2
    minimum_pair_distance_mm: float = 1.0
    maximum_pair_distance_mm: float = 30.0
    minimum_pair_score: float = 0.55
    minimum_score_margin: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.association_radius_mm,
            self.minimum_supporting_frames_per_structure,
            self.minimum_pair_distance_mm,
            self.maximum_pair_distance_mm,
            self.minimum_pair_score,
            self.minimum_score_margin,
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
            raise ValueError("multi-view settings must be finite and nonnegative")
        if self.minimum_pair_distance_mm >= self.maximum_pair_distance_mm:
            raise ValueError("pair distance interval is empty")


def semantic_observations(
    probabilities: np.ndarray,
    *,
    thresholds: Mapping[str, Mapping[str, float | int]],
    end_effector_position_m: Sequence[float],
    rotation_columns: np.ndarray,
    frame_sequence: int,
    frame_sha256: str,
    pixel_resolution_mm: float = 0.5,
    image_origin_height_mm: float = 133.0,
) -> tuple[SemanticObservation, ...]:
    """Convert image-only semantic maps into actor-frame surface observations."""

    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.ndim != 3 or probability.shape[0] != 4:
        raise ValueError("probabilities must be 4 x H x W")
    if frame_sequence < 0 or not frame_sha256:
        raise ValueError("frame identity is incomplete")
    viewpoint_values = np.concatenate(
        (
            np.asarray(end_effector_position_m, dtype="<f8").reshape(-1),
            np.asarray(rotation_columns, dtype="<f8").reshape(-1),
        )
    )
    if viewpoint_values.shape != (12,) or not np.all(np.isfinite(viewpoint_values)):
        raise ValueError("actor viewpoint must contain finite position and rotation")
    viewpoint_id = hashlib.sha256(
        np.round(viewpoint_values, decimals=9).tobytes()
    ).hexdigest()
    coordinates = calibrated_image_coordinates_mm(
        end_effector_position_m=end_effector_position_m,
        rotation_columns=rotation_columns,
        width=probability.shape[2],
        depth=probability.shape[1],
        pixel_resolution_mm=pixel_resolution_mm,
        image_origin_height_mm=image_origin_height_mm,
        home_up_axis=np.asarray(rotation_columns, dtype=np.float64)[:, 2],
    )
    result: list[SemanticObservation] = []
    for class_id, name in ((1, "TP"), (2, "SAP"), (3, "OTHER_BONE")):
        specification = thresholds[name]
        mask = probability[class_id] >= float(specification["threshold"])
        components, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        minimum_pixels = int(specification["minimum_pixels"])
        for component_id in range(1, count + 1):
            component = components == component_id
            pixel_count = int(np.count_nonzero(component))
            if pixel_count < minimum_pixels:
                continue
            weights = probability[class_id][component]
            rows, columns = np.where(component)
            total = max(float(np.sum(weights)), 1.0e-12)
            coordinate = np.sum(coordinates[component] * weights[:, None], axis=0) / total
            result.append(
                SemanticObservation(
                    semantic_class=name,
                    frame_sequence=frame_sequence,
                    frame_sha256=frame_sha256,
                    viewpoint_id=viewpoint_id,
                    viewpoint_position_m=tuple(
                        float(value) for value in viewpoint_values[:3]
                    ),
                    viewpoint_rotation_columns=tuple(
                        tuple(float(value) for value in column)
                        for column in np.asarray(rotation_columns, dtype=np.float64)
                    ),
                    row=float(np.sum(rows * weights) / total),
                    column=float(np.sum(columns * weights) / total),
                    coordinate_mm=tuple(float(value) for value in coordinate),
                    confidence=float(np.mean(weights)),
                    pixel_count=pixel_count,
                )
            )
    return tuple(result)


def cluster_observations(
    observations: Sequence[SemanticObservation],
    *,
    association_radius_mm: float,
    minimum_view_translation_mm: float = 2.0,
    minimum_view_rotation_deg: float = 1.0,
) -> tuple[SemanticCluster, ...]:
    """Cluster same-class observations without counting a frame twice."""

    if (
        association_radius_mm <= 0.0
        or minimum_view_translation_mm < 0.0
        or minimum_view_rotation_deg < 0.0
    ):
        raise ValueError("association and viewpoint separation settings are invalid")
    if len({(row.frame_sequence, row.frame_sha256) for row in observations}) != len(
        {row.frame_sequence for row in observations}
    ):
        raise ValueError("one frame sequence cannot have multiple image hashes")
    clusters: list[list[int]] = []
    for index, observation in enumerate(observations):
        coordinate = np.asarray(observation.coordinate_mm)
        eligible = []
        for cluster_index, members in enumerate(clusters):
            first = observations[members[0]]
            if first.semantic_class != observation.semantic_class:
                continue
            if any(
                observations[item].frame_sequence == observation.frame_sequence
                or observations[item].frame_sha256 == observation.frame_sha256
                or observations[item].viewpoint_id == observation.viewpoint_id
                for item in members
            ):
                continue
            viewpoint_is_distinct = True
            for item in members:
                prior = observations[item]
                translation_mm = 1000.0 * float(
                    np.linalg.norm(
                        np.asarray(prior.viewpoint_position_m)
                        - np.asarray(observation.viewpoint_position_m)
                    )
                )
                left_rotation = np.asarray(prior.viewpoint_rotation_columns)
                right_rotation = np.asarray(observation.viewpoint_rotation_columns)
                cosine = np.clip(
                    (np.trace(left_rotation.T @ right_rotation) - 1.0) / 2.0,
                    -1.0,
                    1.0,
                )
                rotation_deg = math.degrees(math.acos(float(cosine)))
                if (
                    translation_mm < minimum_view_translation_mm
                    and rotation_deg < minimum_view_rotation_deg
                ):
                    viewpoint_is_distinct = False
                    break
            if not viewpoint_is_distinct:
                continue
            distances = [
                float(
                    np.linalg.norm(
                        coordinate - np.asarray(observations[item].coordinate_mm)
                    )
                )
                for item in members
            ]
            if max(distances) <= association_radius_mm:
                eligible.append((float(np.mean(distances)), cluster_index))
        if eligible:
            clusters[min(eligible)[1]].append(index)
        else:
            clusters.append([index])
    result = []
    for members in clusters:
        coordinates = np.asarray([observations[index].coordinate_mm for index in members])
        confidence = np.asarray([observations[index].confidence for index in members])
        result.append(
            SemanticCluster(
                semantic_class=observations[members[0]].semantic_class,
                observation_indices=tuple(members),
                supporting_frames=tuple(
                    sorted({observations[index].frame_sequence for index in members})
                ),
                centroid_mm=tuple(float(value) for value in np.mean(coordinates, axis=0)),
                confidence=float(np.mean(confidence)),
            )
        )
    return tuple(result)


def decide_tp_sap_pair(
    clusters: Sequence[SemanticCluster],
    *,
    config: MultiviewConfig,
) -> PairDecision:
    """Accept one supported TP/SAP relation or abstain on weak/competing evidence."""

    hypotheses = []
    for tp_index, tp in enumerate(clusters):
        if tp.semantic_class != "TP" or len(tp.supporting_frames) < config.minimum_supporting_frames_per_structure:
            continue
        for sap_index, sap in enumerate(clusters):
            if sap.semantic_class != "SAP" or len(sap.supporting_frames) < config.minimum_supporting_frames_per_structure:
                continue
            distance = float(
                np.linalg.norm(np.asarray(tp.centroid_mm) - np.asarray(sap.centroid_mm))
            )
            if not config.minimum_pair_distance_mm <= distance <= config.maximum_pair_distance_mm:
                continue
            support = len(set(tp.supporting_frames) | set(sap.supporting_frames))
            support_score = min(support / 6.0, 1.0)
            confidence = 0.5 * (tp.confidence + sap.confidence)
            hypotheses.append(
                PairHypothesis(
                    tp_cluster_index=tp_index,
                    sap_cluster_index=sap_index,
                    surface_distance_mm=distance,
                    score=float(0.65 * confidence + 0.35 * support_score),
                    supporting_frames=tuple(
                        sorted(set(tp.supporting_frames) | set(sap.supporting_frames))
                    ),
                )
            )
    ordered = tuple(
        sorted(
            hypotheses,
            key=lambda row: (-row.score, -len(row.supporting_frames), row.surface_distance_mm),
        )
    )
    if not ordered:
        return PairDecision("ABSTAIN", None, ordered, "INSUFFICIENT_SUPPORTED_STRUCTURES")
    if ordered[0].score < config.minimum_pair_score:
        return PairDecision("ABSTAIN", None, ordered, "PAIR_SCORE_BELOW_GATE")
    if len(ordered) > 1 and ordered[0].score - ordered[1].score < config.minimum_score_margin:
        return PairDecision("ABSTAIN", None, ordered, "COMPETING_PAIR_HYPOTHESES")
    return PairDecision("ACCEPT", ordered[0], ordered, "SUPPORTED_TP_SAP_PAIR")
