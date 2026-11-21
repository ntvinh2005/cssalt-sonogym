"""Actor-visible full-image bone-interface features for recorded ultrasound."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stage1_features import bilinear_resize
from .stage1_interfaces import InterfaceTrackerConfig, track_bright_interfaces


@dataclass(frozen=True)
class SurfaceFeatureSpec:
    sample_columns: int = 12
    thumbnail_rows: int = 12
    thumbnail_columns: int = 12
    maximum_tracks: int = 3

    def __post_init__(self) -> None:
        if min(
            self.sample_columns,
            self.thumbnail_rows,
            self.thumbnail_columns,
            self.maximum_tracks,
        ) <= 0:
            raise ValueError("surface feature dimensions must be positive")

    @property
    def per_track_width(self) -> int:
        return 3 + 3 * self.sample_columns

    @property
    def pair_width(self) -> int:
        return self.sample_columns

    @property
    def topology_width(self) -> int:
        pairs = self.maximum_tracks * (self.maximum_tracks - 1) // 2
        return self.maximum_tracks * self.per_track_width + pairs * self.pair_width

    @property
    def single_view_width(self) -> int:
        thumbnail = self.thumbnail_rows * self.thumbnail_columns
        return self.topology_width + 2 * thumbnail

    @property
    def causal_width(self) -> int:
        return self.single_view_width + 2 * self.topology_width + 5


DEFAULT_SURFACE_FEATURE_SPEC = SurfaceFeatureSpec()


def normalized_image(image: np.ndarray) -> np.ndarray:
    """Normalize one clean image with the existing five-to-95 percent rule."""

    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 4 or not np.all(np.isfinite(array)):
        raise ValueError("image must be a finite nontrivial 2-D array")
    low, high = np.quantile(array, (0.05, 0.95))
    return np.clip((array - low) / max(float(high - low), 1e-6), 0.0, 1.0)


def _sample_track(
    image: np.ndarray,
    track: object,
    columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height = image.shape[0]
    rows = np.asarray([track.row_at(float(column)) for column in columns])
    intensity: list[float] = []
    shadow: list[float] = []
    for row_value, column_value in zip(rows, columns):
        row = int(np.clip(round(float(row_value)), 0, height - 1))
        column = int(np.clip(round(float(column_value)), 0, image.shape[1] - 1))
        proximal = image[max(0, row - 2) : min(height, row + 2), column]
        distal = image[min(height, row + 3) : min(height, row + 11), column]
        proximal_mean = float(np.mean(proximal)) if proximal.size else 0.0
        distal_mean = float(np.mean(distal)) if distal.size else proximal_mean
        intensity.append(float(image[row, column]))
        shadow.append(proximal_mean - distal_mean)
    values = np.concatenate(
        (
            np.asarray(
                [1.0, track.mean_ridge_score, track.maximum_jump_pixels / height],
                dtype=np.float64,
            ),
            rows / height,
            np.asarray(intensity, dtype=np.float64),
            np.asarray(shadow, dtype=np.float64),
        )
    )
    return values, rows / height


def topology_feature(
    image: np.ndarray,
    spec: SurfaceFeatureSpec = DEFAULT_SURFACE_FEATURE_SPEC,
    tracker_config: InterfaceTrackerConfig = InterfaceTrackerConfig(),
) -> np.ndarray:
    """Encode complete tracked interfaces and their relative depth arrangement."""

    normalized = normalized_image(image)
    tracks = track_bright_interfaces(normalized, tracker_config)[: spec.maximum_tracks]
    columns = np.linspace(
        tracker_config.x_margin_pixels,
        normalized.shape[1] - tracker_config.x_margin_pixels - 1,
        spec.sample_columns,
    )
    values: list[np.ndarray] = []
    sampled_rows: list[np.ndarray] = []
    for track_index in range(spec.maximum_tracks):
        if track_index < len(tracks):
            feature, rows = _sample_track(normalized, tracks[track_index], columns)
        else:
            feature = np.zeros(spec.per_track_width, dtype=np.float64)
            rows = np.zeros(spec.sample_columns, dtype=np.float64)
        values.append(feature)
        sampled_rows.append(rows)
    for first in range(spec.maximum_tracks):
        for second in range(first + 1, spec.maximum_tracks):
            both_present = values[first][0] > 0.0 and values[second][0] > 0.0
            values.append(
                sampled_rows[second] - sampled_rows[first]
                if both_present
                else np.zeros(spec.sample_columns, dtype=np.float64)
            )
    result = np.concatenate(values)
    if result.shape != (spec.topology_width,) or not np.all(np.isfinite(result)):
        raise AssertionError("invalid topology feature")
    return result


def single_view_feature(
    image: np.ndarray,
    spec: SurfaceFeatureSpec = DEFAULT_SURFACE_FEATURE_SPEC,
    tracker_config: InterfaceTrackerConfig = InterfaceTrackerConfig(),
) -> np.ndarray:
    """Combine interface arrangement with low-resolution image context."""

    normalized = normalized_image(image)
    topology = topology_feature(normalized, spec, tracker_config)
    intensity = bilinear_resize(
        normalized, spec.thumbnail_rows, spec.thumbnail_columns
    )
    vertical_gradient = np.abs(np.gradient(normalized, axis=0))
    gradient_scale = max(float(np.quantile(vertical_gradient, 0.95)), 1e-6)
    gradient = bilinear_resize(
        np.clip(vertical_gradient / gradient_scale, 0.0, 2.0),
        spec.thumbnail_rows,
        spec.thumbnail_columns,
    )
    result = np.concatenate((topology, intensity.ravel(), gradient.ravel()))
    if result.shape != (spec.single_view_width,) or not np.all(np.isfinite(result)):
        raise AssertionError("invalid single-view feature")
    return result


def causal_feature(
    image: np.ndarray,
    previous_image: np.ndarray | None,
    executed_delta_relative_mm: tuple[float, float] | list[float] | np.ndarray,
    executed_delta_angle_rad: float,
    action_steps_since_previous_frame: int,
    spec: SurfaceFeatureSpec = DEFAULT_SURFACE_FEATURE_SPEC,
    tracker_config: InterfaceTrackerConfig = InterfaceTrackerConfig(),
) -> np.ndarray:
    """Add only the preceding trajectory observation and executed motion."""

    current = single_view_feature(image, spec, tracker_config)
    current_topology = current[: spec.topology_width]
    has_previous = previous_image is not None
    previous_topology = (
        topology_feature(previous_image, spec, tracker_config)
        if has_previous
        else np.zeros(spec.topology_width, dtype=np.float64)
    )
    delta = np.asarray(executed_delta_relative_mm, dtype=np.float64)
    if delta.shape != (2,) or not np.all(np.isfinite(delta)):
        raise ValueError("executed translation must contain two finite values")
    motion = np.asarray(
        [
            float(has_previous),
            delta[0] / 50.0,
            delta[1] / 50.0,
            float(executed_delta_angle_rad) / np.pi,
            min(max(int(action_steps_since_previous_frame), 0), 500) / 500.0,
        ],
        dtype=np.float64,
    )
    result = np.concatenate(
        (
            current,
            current_topology - previous_topology,
            np.abs(current_topology - previous_topology),
            motion,
        )
    )
    if result.shape != (spec.causal_width,) or not np.all(np.isfinite(result)):
        raise AssertionError("invalid causal feature")
    return result
