"""Leakage-separated patch sampling and Stage 1 landmark reconstruction helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .stage1_features import PatchSpec, extract_interface_patch, extract_patch_feature
from .stage1_interfaces import InterfaceTrackerConfig, track_bright_interfaces
from .stage1_map import EvidenceObservation, LandmarkEvent, SpatialEvidenceMap, SpatialMapConfig


Direction = Literal["FORWARD", "REVERSE"]
LUMBAR_LABELS = (26, 27, 28, 29, 30, 31)
GROUP_A_LABELS = (25, 26, 27, 28, 29, 30, 31)


@dataclass(frozen=True)
class PatchLabelConfig:
    minimum_lumbar_fraction: float = 0.01
    l4_label: int = 28

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_lumbar_fraction <= 1.0:
            raise ValueError("minimum_lumbar_fraction must lie in (0, 1]")
        if self.l4_label not in LUMBAR_LABELS:
            raise ValueError("l4_label must be a declared lumbar label")


@dataclass(frozen=True)
class ActorPatchCandidate:
    feature: np.ndarray
    coordinate_mm: float
    frame_sequence: int
    direction: Direction
    track_id: str
    interface_row: float
    column: float

    def __post_init__(self) -> None:
        feature = np.asarray(self.feature, dtype=np.float64)
        if feature.ndim != 1 or not np.all(np.isfinite(feature)):
            raise ValueError("feature must be a finite vector")
        if not all(
            math.isfinite(value)
            for value in (self.coordinate_mm, self.interface_row, self.column)
        ):
            raise ValueError("candidate geometry must be finite")
        if self.frame_sequence < 0 or self.direction not in ("FORWARD", "REVERSE"):
            raise ValueError("invalid frame identity or direction")
        if not self.track_id:
            raise ValueError("track_id must be nonempty")


@dataclass(frozen=True)
class PatchAnnotation:
    primary_label: int
    verifier_label: int | None
    dominant_lumbar_label: int | None
    lumbar_fraction: float
    dominant_fraction: float


@dataclass(frozen=True)
class GroupAPatchAnnotation:
    primary_label: int
    anatomy_class: int | None
    anatomy_fraction: float
    dominant_fraction: float


@dataclass(frozen=True)
class LandmarkInference:
    status: Literal["SUCCESS", "ABSTAIN"]
    coordinate_mm: float | None
    uncertainty_interval_mm: tuple[float, float] | None
    accepted_event_count: int
    reason: str | None


def sample_columns(image_width: int, spec: PatchSpec, stride_pixels: int) -> np.ndarray:
    if image_width <= 0 or stride_pixels <= 0:
        raise ValueError("image width and stride must be positive")
    half_width = spec.width_pixels / 2.0
    minimum = int(math.ceil(half_width))
    maximum = int(math.floor(image_width - half_width))
    if maximum < minimum:
        return np.empty(0, dtype=np.float64)
    return np.arange(minimum, maximum + 1, stride_pixels, dtype=np.float64)


def project_column_to_home_longitudinal_mm(
    *,
    probe_longitudinal_home_mm: float,
    column: float,
    width_pixels: int,
    pixel_resolution_mm: float,
    declared_home_angle_rad: float,
    relative_angle_rad: float,
) -> float:
    """Project the image lateral coordinate onto patient/Home longitudinal z.

    SonoGym constructs image columns along the probe x-axis. The surface-motion planner
    defines that axis in patient x-z with projected z component ``sin(angle)``.
    """
    values = (
        probe_longitudinal_home_mm,
        column,
        pixel_resolution_mm,
        declared_home_angle_rad,
        relative_angle_rad,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("projection inputs must be finite")
    if width_pixels <= 1 or pixel_resolution_mm <= 0.0:
        raise ValueError("invalid image geometry")
    if not 0.0 <= column <= width_pixels - 1:
        raise ValueError("column lies outside image")
    center = 0.5 * (width_pixels - 1.0)
    lateral_mm = (column - center) * pixel_resolution_mm
    angle = declared_home_angle_rad + relative_angle_rad
    return float(probe_longitudinal_home_mm + math.sin(angle) * lateral_mm)


def extract_actor_patch_candidates(
    image: np.ndarray,
    *,
    probe_longitudinal_home_mm: float,
    relative_angle_rad: float,
    declared_home_angle_rad: float,
    frame_sequence: int,
    direction: Direction,
    patch_spec: PatchSpec,
    tracker_config: InterfaceTrackerConfig,
    column_stride_pixels: int,
    pixel_resolution_mm: float,
) -> list[ActorPatchCandidate]:
    """Extract actor-visible candidates without accepting any privileged annotation."""
    array = np.asarray(image, dtype=np.float64)
    columns = sample_columns(array.shape[1], patch_spec, column_stride_pixels)
    candidates: list[ActorPatchCandidate] = []
    for track in track_bright_interfaces(array, tracker_config):
        for column in columns:
            if column < track.columns[0] or column > track.columns[-1]:
                continue
            interface_row = track.row_at(float(column))
            feature = extract_patch_feature(array, interface_row, float(column), patch_spec)
            if feature is None:
                continue
            coordinate_mm = project_column_to_home_longitudinal_mm(
                probe_longitudinal_home_mm=probe_longitudinal_home_mm,
                column=float(column),
                width_pixels=array.shape[1],
                pixel_resolution_mm=pixel_resolution_mm,
                declared_home_angle_rad=declared_home_angle_rad,
                relative_angle_rad=relative_angle_rad,
            )
            candidates.append(
                ActorPatchCandidate(
                    feature=feature,
                    coordinate_mm=coordinate_mm,
                    frame_sequence=frame_sequence,
                    direction=direction,
                    track_id=track.track_id,
                    interface_row=interface_row,
                    column=float(column),
                )
            )
    return candidates


def annotate_patch_candidate(
    raw_labels: np.ndarray,
    candidate: ActorPatchCandidate,
    patch_spec: PatchSpec,
    config: PatchLabelConfig,
) -> PatchAnnotation:
    """Create offline labels; this function is forbidden from actor inference paths."""
    patch = extract_interface_patch(
        np.asarray(raw_labels), candidate.interface_row, candidate.column, patch_spec
    )
    if patch is None:
        raise ValueError("actor candidate did not produce the corresponding label patch")
    counts = np.asarray([np.count_nonzero(patch == label) for label in LUMBAR_LABELS])
    fractions = counts.astype(np.float64) / float(patch.size)
    lumbar_fraction = float(np.sum(fractions))
    if lumbar_fraction < config.minimum_lumbar_fraction:
        return PatchAnnotation(0, None, None, lumbar_fraction, 0.0)
    maximum = int(np.max(counts))
    winners = np.flatnonzero(counts == maximum)
    if maximum <= 0 or len(winners) != 1:
        return PatchAnnotation(1, None, None, lumbar_fraction, 0.0)
    dominant_label = int(LUMBAR_LABELS[int(winners[0])])
    dominant_fraction = float(fractions[int(winners[0])])
    verifier_label = int(dominant_label == config.l4_label)
    return PatchAnnotation(
        1,
        verifier_label,
        dominant_label,
        lumbar_fraction,
        dominant_fraction,
    )


def annotate_group_a_patch_candidate(
    raw_labels: np.ndarray,
    candidate: ActorPatchCandidate,
    patch_spec: PatchSpec,
    minimum_anatomy_fraction: float,
) -> GroupAPatchAnnotation:
    """Create privileged Group A labels outside actor inference."""
    if not 0.0 < minimum_anatomy_fraction <= 1.0:
        raise ValueError("minimum_anatomy_fraction must lie in (0, 1]")
    patch = extract_interface_patch(
        np.asarray(raw_labels), candidate.interface_row, candidate.column, patch_spec
    )
    if patch is None:
        raise ValueError("actor candidate did not produce the corresponding label patch")
    counts = np.asarray([np.count_nonzero(patch == label) for label in GROUP_A_LABELS])
    fractions = counts.astype(np.float64) / float(patch.size)
    anatomy_fraction = float(np.sum(fractions))
    if anatomy_fraction < minimum_anatomy_fraction:
        return GroupAPatchAnnotation(0, None, anatomy_fraction, 0.0)
    maximum = int(np.max(counts))
    winners = np.flatnonzero(counts == maximum)
    if maximum <= 0 or len(winners) != 1:
        return GroupAPatchAnnotation(1, None, anatomy_fraction, 0.0)
    anatomy_class = int(winners[0])
    return GroupAPatchAnnotation(
        1,
        anatomy_class,
        anatomy_fraction,
        float(fractions[anatomy_class]),
    )


def reconstruct_landmark(
    candidates: list[ActorPatchCandidate],
    primary_probabilities: np.ndarray,
    verifier_probabilities: np.ndarray,
    *,
    map_config: SpatialMapConfig,
    coordinate_offset_mm: float = 0.0,
    minimum_top_score_margin: float = 0.0,
) -> tuple[LandmarkInference, list[LandmarkEvent]]:
    if not math.isfinite(coordinate_offset_mm):
        raise ValueError("coordinate_offset_mm must be finite")
    if not math.isfinite(minimum_top_score_margin) or minimum_top_score_margin < 0.0:
        raise ValueError("minimum_top_score_margin must be finite and nonnegative")
    if len(candidates) == 0:
        return LandmarkInference("ABSTAIN", None, None, 0, "NO_PATCH_CANDIDATES"), []
    primary = np.asarray(primary_probabilities, dtype=np.float64)
    verifier = np.asarray(verifier_probabilities, dtype=np.float64)
    if primary.shape != (len(candidates),) or verifier.shape != (len(candidates),):
        raise ValueError("probability vectors must align with candidates")
    if not (
        np.all(np.isfinite(primary))
        and np.all(np.isfinite(verifier))
        and np.all((0.0 <= primary) & (primary <= 1.0))
        and np.all((0.0 <= verifier) & (verifier <= 1.0))
    ):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    coordinates = np.asarray([item.coordinate_mm for item in candidates])
    minimum = math.floor(float(np.min(coordinates)) / map_config.grid_step_mm) * map_config.grid_step_mm
    maximum = math.ceil(float(np.max(coordinates)) / map_config.grid_step_mm) * map_config.grid_step_mm
    if maximum <= minimum:
        maximum = minimum + map_config.grid_step_mm
    spatial_map = SpatialEvidenceMap(minimum, maximum, map_config)
    for candidate, probability in zip(candidates, np.minimum(primary, verifier)):
        spatial_map.add(
            EvidenceObservation(
                coordinate_mm=candidate.coordinate_mm,
                probability=float(probability),
                direction=candidate.direction,
                frame_sequence=candidate.frame_sequence,
                track_id=candidate.track_id,
            )
        )
    events = spatial_map.events()
    accepted = [event for event in events if event.accepted]
    if not accepted:
        return (
            LandmarkInference(
                "ABSTAIN",
                None,
                None,
                0,
                "NO_ACCEPTED_EVENT",
            ),
            events,
        )
    ranked = sorted(accepted, key=lambda event: (-event.score, event.coordinate_mm))
    if len(ranked) > 1 and ranked[0].score - ranked[1].score < minimum_top_score_margin:
        return (
            LandmarkInference(
                "ABSTAIN",
                None,
                None,
                len(accepted),
                "AMBIGUOUS_TOP_EVENT_SCORE",
            ),
            events,
        )
    target = ranked[0]
    coordinate = target.coordinate_mm + coordinate_offset_mm
    uncertainty = (
        target.support_interval_mm[0] + coordinate_offset_mm,
        target.support_interval_mm[1] + coordinate_offset_mm,
    )
    return (
        LandmarkInference(
            "SUCCESS",
            coordinate,
            uncertainty,
            len(accepted),
            None,
        ),
        events,
    )


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if truth.shape != scores.shape or truth.ndim != 1 or not np.all(np.isin(truth, (0, 1))):
        raise ValueError("labels and probabilities must be aligned binary vectors")
    predicted = scores >= threshold
    tp = int(np.count_nonzero(predicted & (truth == 1)))
    fp = int(np.count_nonzero(predicted & (truth == 0)))
    fn = int(np.count_nonzero(~predicted & (truth == 1)))
    tn = int(np.count_nonzero(~predicted & (truth == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
    }

