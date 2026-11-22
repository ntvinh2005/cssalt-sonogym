"""Image-only evidence for a pair of local cortical interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .ultrasound_surface_features import normalized_image


@dataclass(frozen=True)
class SegmentConfig:
    minimum_depth_fraction: float = 0.48
    maximum_depth_fraction: float = 0.96
    minimum_column_span: int = 5
    minimum_component_pixels: int = 8
    maximum_segments: int = 8
    ridge_quantile: float = 0.86

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_depth_fraction < self.maximum_depth_fraction <= 1.0:
            raise ValueError("invalid depth fractions")
        if min(
            self.minimum_column_span,
            self.minimum_component_pixels,
            self.maximum_segments,
        ) < 1:
            raise ValueError("segment limits must be positive")
        if not 0.0 < self.ridge_quantile < 1.0:
            raise ValueError("ridge quantile must lie between zero and one")


@dataclass(frozen=True)
class CorticalSegment:
    rows: np.ndarray
    columns: np.ndarray
    ridge_score: float
    shadow_contrast: float
    intensity_contrast: float

    @property
    def center(self) -> tuple[float, float]:
        return float(np.median(self.rows)), float(np.median(self.columns))


def _ridge_field(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertical_gradient = np.abs(np.gradient(image, axis=0))
    gradient_scale = max(float(np.quantile(vertical_gradient, 0.95)), 1e-6)
    gradient = np.clip(vertical_gradient / gradient_scale, 0.0, 2.0)
    proximal = ndimage.uniform_filter1d(image, size=3, axis=0, origin=-1)
    distal = ndimage.uniform_filter1d(image, size=9, axis=0, origin=4)
    shadow = np.clip(proximal - distal, 0.0, 1.0)
    return 0.45 * image + 0.40 * gradient + 0.15 * shadow, shadow


def cortical_segments(
    image: np.ndarray,
    config: SegmentConfig = SegmentConfig(),
) -> tuple[CorticalSegment, ...]:
    """Extract spatially bounded bright-interface candidates from one clean image."""

    clean = normalized_image(image)
    height, width = clean.shape
    first_row = int(round(height * config.minimum_depth_fraction))
    last_row = int(round(height * config.maximum_depth_fraction))
    ridge, shadow = _ridge_field(clean)
    region = ridge[first_row:last_row]
    threshold = max(float(np.quantile(region, config.ridge_quantile)), 0.55)
    local_maximum = region == ndimage.maximum_filter(region, size=(3, 3), mode="nearest")
    active = local_maximum & (region >= threshold)
    active = ndimage.binary_dilation(active, structure=np.ones((3, 3), dtype=bool))
    labels, count = ndimage.label(active, structure=np.ones((3, 3), dtype=np.int8))
    result: list[CorticalSegment] = []
    for label in range(1, count + 1):
        component_rows, component_columns = np.where(labels == label)
        if len(component_rows) < config.minimum_component_pixels:
            continue
        unique_columns = np.unique(component_columns)
        if len(unique_columns) < config.minimum_column_span:
            continue
        rows: list[float] = []
        scores: list[float] = []
        for column in unique_columns:
            candidates = component_rows[component_columns == column]
            absolute_rows = candidates + first_row
            candidate_scores = ridge[absolute_rows, column]
            best = int(np.argmax(candidate_scores))
            rows.append(float(absolute_rows[best]))
            scores.append(float(candidate_scores[best]))
        row_array = np.asarray(rows, dtype=np.float64)
        column_array = unique_columns.astype(np.float64)
        pixel_rows = np.clip(np.rint(row_array).astype(int), 0, height - 1)
        pixel_columns = np.clip(np.rint(column_array).astype(int), 0, width - 1)
        distal_rows = np.clip(pixel_rows + 7, 0, height - 1)
        result.append(
            CorticalSegment(
                rows=row_array,
                columns=column_array,
                ridge_score=float(np.mean(scores)),
                shadow_contrast=float(np.mean(shadow[pixel_rows, pixel_columns])),
                intensity_contrast=float(
                    np.mean(clean[pixel_rows, pixel_columns] - clean[distal_rows, pixel_columns])
                ),
            )
        )
    result.sort(
        key=lambda segment: (
            segment.ridge_score * len(segment.columns),
            segment.shadow_contrast,
        ),
        reverse=True,
    )
    return tuple(result[: config.maximum_segments])


def segment_feature(segment: CorticalSegment, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    centered_columns = segment.columns - float(np.mean(segment.columns))
    if float(np.sum(centered_columns * centered_columns)) > 0.0:
        slope = float(
            np.sum(centered_columns * (segment.rows - float(np.mean(segment.rows))))
            / np.sum(centered_columns * centered_columns)
        )
    else:
        slope = 0.0
    centered_rows = segment.rows - (slope * centered_columns + float(np.mean(segment.rows)))
    return np.asarray(
        [
            len(segment.columns) / width,
            (float(np.max(segment.columns)) - float(np.min(segment.columns)) + 1.0) / width,
            float(np.median(segment.rows)) / height,
            float(np.median(segment.columns)) / width,
            slope,
            float(np.mean(np.abs(centered_rows))) / height,
            segment.ridge_score,
            segment.shadow_contrast,
            segment.intensity_contrast,
        ],
        dtype=np.float64,
    )


def pair_feature(
    first: CorticalSegment,
    second: CorticalSegment,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Describe two image-derived candidates without geometry or absolute robot pose."""

    height, width = image_shape
    first_feature = segment_feature(first, image_shape)
    second_feature = segment_feature(second, image_shape)
    first_center = np.asarray(first.center, dtype=np.float64)
    second_center = np.asarray(second.center, dtype=np.float64)
    delta = (second_center - first_center) / np.asarray([height, width], dtype=np.float64)
    gap = float(np.linalg.norm(delta))
    result = np.concatenate((first_feature, second_feature, delta, [gap]))
    if result.shape != (21,) or not np.all(np.isfinite(result)):
        raise AssertionError("invalid interface-pair feature")
    return result


def pair_candidates(
    image: np.ndarray,
    config: SegmentConfig = SegmentConfig(),
) -> tuple[tuple[CorticalSegment, CorticalSegment, np.ndarray], ...]:
    segments = cortical_segments(image, config)
    result: list[tuple[CorticalSegment, CorticalSegment, np.ndarray]] = []
    for first_index, first in enumerate(segments):
        for second in segments[first_index + 1 :]:
            result.append((first, second, pair_feature(first, second, image.shape)))
            result.append((second, first, pair_feature(second, first, image.shape)))
    return tuple(result)
