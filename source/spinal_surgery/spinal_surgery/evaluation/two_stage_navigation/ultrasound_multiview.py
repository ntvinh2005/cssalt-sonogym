"""Associate actor-visible cortical interfaces across recorded probe viewpoints."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .stage1_interfaces import InterfaceTrackerConfig, track_bright_interfaces
from .ultrasound_surface_features import normalized_image


@dataclass(frozen=True)
class InterfaceDescriptor:
    rows: np.ndarray
    intensity: np.ndarray
    shadow: np.ndarray
    ridge_score: float


@dataclass(frozen=True)
class EvidenceConfig:
    aggregation: str
    maximum_neighbor_translation_mm: float
    maximum_neighbor_angle_deg: float
    minimum_view_translation_mm: float
    minimum_view_angle_deg: float
    minimum_correspondence: float
    maximum_neighbors: int
    neighbor_weight: float
    minimum_support: int

    def __post_init__(self) -> None:
        if self.aggregation not in {"mean", "log_odds"}:
            raise ValueError("aggregation must be mean or log_odds")
        values = (
            self.maximum_neighbor_translation_mm,
            self.maximum_neighbor_angle_deg,
            self.minimum_view_translation_mm,
            self.minimum_view_angle_deg,
            self.minimum_correspondence,
            self.neighbor_weight,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("evidence settings must be finite and nonnegative")
        if not 0.0 <= self.minimum_correspondence <= 1.0:
            raise ValueError("minimum correspondence must not exceed one")
        if not 0.0 <= self.neighbor_weight <= 1.0:
            raise ValueError("neighbor weight must not exceed one")
        if self.maximum_neighbors < 1 or self.minimum_support < 1:
            raise ValueError("neighbor and support counts must be positive")


def interface_descriptors(
    image: np.ndarray,
    *,
    sample_columns: int = 12,
    tracker_config: InterfaceTrackerConfig = InterfaceTrackerConfig(),
) -> tuple[InterfaceDescriptor, ...]:
    """Describe complete interfaces without evaluator geometry."""

    array = normalized_image(image)
    columns = np.linspace(
        tracker_config.x_margin_pixels,
        array.shape[1] - tracker_config.x_margin_pixels - 1,
        sample_columns,
    )
    result: list[InterfaceDescriptor] = []
    for track in track_bright_interfaces(array, tracker_config):
        rows = np.asarray([track.row_at(float(column)) for column in columns])
        intensity: list[float] = []
        shadow: list[float] = []
        for row_value, column_value in zip(rows, columns):
            row = int(np.clip(round(float(row_value)), 0, array.shape[0] - 1))
            column = int(np.clip(round(float(column_value)), 0, array.shape[1] - 1))
            proximal = array[max(0, row - 2) : min(array.shape[0], row + 2), column]
            distal = array[min(array.shape[0], row + 3) : min(array.shape[0], row + 11), column]
            proximal_mean = float(np.mean(proximal)) if proximal.size else 0.0
            distal_mean = float(np.mean(distal)) if distal.size else proximal_mean
            intensity.append(float(array[row, column]))
            shadow.append(proximal_mean - distal_mean)
        result.append(
            InterfaceDescriptor(
                rows=rows / array.shape[0],
                intensity=np.asarray(intensity, dtype=np.float64),
                shadow=np.asarray(shadow, dtype=np.float64),
                ridge_score=float(track.mean_ridge_score),
            )
        )
    return tuple(result)


def _pair_score(left: InterfaceDescriptor, right: InterfaceDescriptor) -> float:
    left_shape = left.rows - np.median(left.rows)
    right_shape = right.rows - np.median(right.rows)
    shape_error = float(np.mean(np.abs(left_shape - right_shape)))
    depth_error = abs(float(np.median(left.rows) - np.median(right.rows)))
    intensity_error = float(np.mean(np.abs(left.intensity - right.intensity)))
    shadow_error = float(np.mean(np.abs(left.shadow - right.shadow)))
    ridge_error = abs(left.ridge_score - right.ridge_score) / 2.0
    distance = (
        12.0 * shape_error
        + depth_error
        + 2.0 * intensity_error
        + 2.0 * shadow_error
        + ridge_error
    )
    return float(np.exp(-distance))


def interface_correspondence(
    left: tuple[InterfaceDescriptor, ...],
    right: tuple[InterfaceDescriptor, ...],
) -> float:
    """Match complete interface arrangements with unmatched tracks scored as zero."""

    if not left or not right:
        return 0.0
    scores = np.asarray(
        [[_pair_score(first, second) for second in right] for first in left],
        dtype=np.float64,
    )
    rows, columns = linear_sum_assignment(-scores)
    return float(np.sum(scores[rows, columns]) / max(len(left), len(right)))


def correspondence_matrix(images: list[np.ndarray]) -> np.ndarray:
    descriptors = [interface_descriptors(image) for image in images]
    result = np.eye(len(images), dtype=np.float64)
    for first in range(len(images)):
        for second in range(first + 1, len(images)):
            value = interface_correspondence(descriptors[first], descriptors[second])
            result[first, second] = value
            result[second, first] = value
    return result


def _wrap_degrees(angle_radians: float) -> float:
    return abs(math.degrees((angle_radians + math.pi) % (2.0 * math.pi) - math.pi))


def pose_distance_matrices(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("poses must be finite x/z/angle rows")
    count = len(values)
    translation = np.zeros((count, count), dtype=np.float64)
    angle = np.zeros((count, count), dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            translation_value = float(np.linalg.norm(values[first, :2] - values[second, :2]))
            angle_value = _wrap_degrees(values[first, 2] - values[second, 2])
            translation[first, second] = translation[second, first] = translation_value
            angle[first, second] = angle[second, first] = angle_value
    return translation, angle


def accumulate_evidence(
    base_scores: np.ndarray,
    correspondences: np.ndarray,
    translation_mm: np.ndarray,
    angle_deg: np.ndarray,
    config: EvidenceConfig,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Accumulate support from distinct, nearby, image-corresponding viewpoints."""

    scores = np.asarray(base_scores, dtype=np.float64)
    count = len(scores)
    shape = (count, count)
    for name, matrix in (
        ("correspondence", correspondences),
        ("translation", translation_mm),
        ("angle", angle_deg),
    ):
        values = np.asarray(matrix, dtype=np.float64)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} matrix must be finite and square")
    if scores.shape != (count,) or not np.all(np.isfinite(scores)):
        raise ValueError("base scores must be a finite vector")
    scores = np.clip(scores, 1e-6, 1.0 - 1e-6)
    accumulated = scores.copy()
    support_counts = np.zeros(count, dtype=np.int64)
    support_indices: list[list[int]] = [[] for _ in range(count)]
    for index in range(count):
        candidates: list[tuple[float, int]] = []
        for other in range(count):
            if other == index:
                continue
            distinct = bool(
                translation_mm[index, other] >= config.minimum_view_translation_mm
                or angle_deg[index, other] >= config.minimum_view_angle_deg
            )
            nearby = bool(
                translation_mm[index, other] <= config.maximum_neighbor_translation_mm
                and angle_deg[index, other] <= config.maximum_neighbor_angle_deg
            )
            if (
                not distinct
                or not nearby
                or correspondences[index, other] < config.minimum_correspondence
            ):
                continue
            distance_weight = math.exp(
                -translation_mm[index, other]
                / max(config.maximum_neighbor_translation_mm, 1e-6)
                -angle_deg[index, other]
                / max(config.maximum_neighbor_angle_deg, 1e-6)
            )
            candidates.append(
                (float(correspondences[index, other] * distance_weight), other)
            )
        chosen = sorted(candidates, reverse=True)[: config.maximum_neighbors]
        support_indices[index] = [other for _, other in chosen]
        support_counts[index] = len(chosen)
        if not chosen:
            continue
        weights = np.asarray([weight for weight, _ in chosen], dtype=np.float64)
        neighbor_scores = np.asarray([scores[other] for _, other in chosen])
        neighbor_mean = float(np.average(neighbor_scores, weights=weights))
        if config.aggregation == "mean":
            accumulated[index] = (
                (1.0 - config.neighbor_weight) * scores[index]
                + config.neighbor_weight * neighbor_mean
            )
        else:
            current_log_odds = math.log(scores[index] / (1.0 - scores[index]))
            neighbor_log_odds = np.log(neighbor_scores / (1.0 - neighbor_scores))
            combined = current_log_odds + config.neighbor_weight * float(
                np.average(neighbor_log_odds, weights=weights)
            )
            accumulated[index] = 1.0 / (1.0 + math.exp(-combined))
    return accumulated, support_counts, support_indices


MULTIVIEW_FEATURE_NAMES = (
    "base_score",
    "support_fraction",
    "maximum_correspondence",
    "mean_correspondence",
    "neighbor_score_mean",
    "neighbor_score_maximum",
    "neighbor_score_minimum",
    "neighbor_score_std",
    "mean_translation_fraction",
    "maximum_translation_fraction",
    "mean_angle_fraction",
    "maximum_angle_fraction",
    "base_neighbor_difference",
)


def multiview_feature_matrix(
    base_scores: np.ndarray,
    correspondences: np.ndarray,
    translation_mm: np.ndarray,
    angle_deg: np.ndarray,
    config: EvidenceConfig,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Summarize image correspondence and executed motion for each candidate view."""

    _, support_counts, support_indices = accumulate_evidence(
        base_scores, correspondences, translation_mm, angle_deg, config
    )
    scores = np.asarray(base_scores, dtype=np.float64)
    rows: list[np.ndarray] = []
    for index, support in enumerate(support_indices):
        if support:
            corr = correspondences[index, support]
            neighbors = scores[support]
            translations = translation_mm[index, support]
            angles = angle_deg[index, support]
            neighbor_mean = float(np.mean(neighbors))
            values = np.asarray(
                [
                    scores[index],
                    len(support) / config.maximum_neighbors,
                    float(np.max(corr)),
                    float(np.mean(corr)),
                    neighbor_mean,
                    float(np.max(neighbors)),
                    float(np.min(neighbors)),
                    float(np.std(neighbors)),
                    float(np.mean(translations))
                    / max(config.maximum_neighbor_translation_mm, 1e-6),
                    float(np.max(translations))
                    / max(config.maximum_neighbor_translation_mm, 1e-6),
                    float(np.mean(angles)) / max(config.maximum_neighbor_angle_deg, 1e-6),
                    float(np.max(angles)) / max(config.maximum_neighbor_angle_deg, 1e-6),
                    abs(float(scores[index]) - neighbor_mean),
                ],
                dtype=np.float64,
            )
        else:
            values = np.zeros(len(MULTIVIEW_FEATURE_NAMES), dtype=np.float64)
            values[0] = scores[index]
        rows.append(values)
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (len(scores), len(MULTIVIEW_FEATURE_NAMES)):
        raise AssertionError("invalid multi-view feature shape")
    if not np.all(np.isfinite(matrix)):
        raise AssertionError("multi-view features must be finite")
    return matrix, support_counts, support_indices
