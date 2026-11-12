"""Bounded bright-interface hypotheses for Stage 1 ultrasound patches.

This is a small Python transfer of the verified Unity multi-interface idea: score intensity and
vertical-gradient ridges, solve smooth paths, and retain only spatially diverse hypotheses.
The G0 pilot must validate its depth band and score thresholds on SonoGym images.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterfaceTrackerConfig:
    x_margin_pixels: int = 18
    y_min_fraction: float = 0.50
    y_max_fraction: float = 0.96
    maximum_path_step_pixels: int = 6
    maximum_smoothed_path_jump_pixels: float = 4.0
    path_smoothness_penalty: float = 0.014
    minimum_mean_ridge_score: float = 0.18
    maximum_hypotheses: int = 3
    minimum_median_separation_pixels: float = 2.0
    minimum_separated_column_fraction: float = 0.15

    def __post_init__(self) -> None:
        if self.x_margin_pixels < 0 or self.maximum_path_step_pixels < 0:
            raise ValueError("pixel margins and steps must be nonnegative")
        if self.maximum_smoothed_path_jump_pixels < 0.0:
            raise ValueError("smoothed path jump must be nonnegative")
        if not 0.0 <= self.y_min_fraction < self.y_max_fraction <= 1.0:
            raise ValueError("invalid depth fractions")
        if self.maximum_hypotheses < 1:
            raise ValueError("maximum_hypotheses must be positive")


@dataclass(frozen=True)
class InterfaceTrack:
    track_id: str
    columns: np.ndarray
    rows: np.ndarray
    mean_ridge_score: float
    maximum_jump_pixels: float
    valid: bool

    def row_at(self, column: float) -> float:
        if column < self.columns[0] or column > self.columns[-1]:
            raise ValueError("column lies outside tracked interface")
        return float(np.interp(column, self.columns, self.rows))


def _normalize_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 4:
        raise ValueError(f"image must be a nontrivial 2-D array, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("image contains nonfinite values")
    low, high = np.quantile(array, (0.05, 0.95))
    return np.clip((array - low) / max(float(high - low), 1e-6), 0.0, 1.0)


def _solve_path(
    ridge: np.ndarray,
    columns: np.ndarray,
    y_offset: int,
    config: InterfaceTrackerConfig,
    track_id: str,
) -> InterfaceTrack:
    row_count = ridge.shape[0]
    cost = -ridge[:, columns[0]].copy()
    back = np.zeros((len(columns), row_count), dtype=np.int16)
    shifts = np.arange(
        -config.maximum_path_step_pixels,
        config.maximum_path_step_pixels + 1,
        dtype=np.int16,
    )
    for column_index, column in enumerate(columns[1:], start=1):
        candidates = np.full((len(shifts), row_count), np.inf, dtype=np.float64)
        for shift_index, shift in enumerate(shifts):
            if shift < 0:
                candidates[shift_index, :shift] = cost[-shift:]
            elif shift > 0:
                candidates[shift_index, shift:] = cost[:-shift]
            else:
                candidates[shift_index] = cost
            candidates[shift_index] += config.path_smoothness_penalty * float(shift * shift)
        best = np.argmin(candidates, axis=0)
        back[column_index] = shifts[best]
        cost = candidates[best, np.arange(row_count)] - ridge[:, column]

    path = np.zeros(len(columns), dtype=np.int32)
    path[-1] = int(np.argmin(cost))
    for column_index in range(len(columns) - 1, 0, -1):
        path[column_index - 1] = path[column_index] - int(back[column_index, path[column_index]])
    path = np.clip(path, 0, row_count - 1)
    raw_rows = path.astype(np.float64) + y_offset
    padded = np.pad(raw_rows, (3, 3), mode="edge")
    rows = np.convolve(padded, np.ones(7, dtype=np.float64) / 7.0, mode="valid")
    samples = ridge[path, columns]
    jumps = np.abs(np.diff(rows))
    mean_score = float(np.mean(samples))
    maximum_jump = float(np.max(jumps, initial=0.0))
    return InterfaceTrack(
        track_id=track_id,
        columns=columns.astype(np.float64),
        rows=rows,
        mean_ridge_score=mean_score,
        maximum_jump_pixels=maximum_jump,
        valid=bool(
            mean_score >= config.minimum_mean_ridge_score
            and maximum_jump <= config.maximum_smoothed_path_jump_pixels
        ),
    )


def _diverse(
    candidate: InterfaceTrack,
    accepted: list[InterfaceTrack],
    config: InterfaceTrackerConfig,
) -> bool:
    for other in accepted:
        separation = np.abs(candidate.rows - other.rows)
        if (
            float(np.median(separation)) < config.minimum_median_separation_pixels
            or float(np.mean(separation >= config.minimum_median_separation_pixels))
            < config.minimum_separated_column_fraction
        ):
            return False
    return True


def track_bright_interfaces(
    image: np.ndarray,
    config: InterfaceTrackerConfig = InterfaceTrackerConfig(),
) -> list[InterfaceTrack]:
    """Return up to three valid, diverse smooth bright-interface hypotheses."""
    normalized = _normalize_image(image)
    height, width = normalized.shape
    if 2 * config.x_margin_pixels >= width - 1:
        raise ValueError("x margin leaves no trackable columns")
    y_min = int(round(height * config.y_min_fraction))
    y_max = int(round(height * config.y_max_fraction))
    if y_max - y_min < 2:
        raise ValueError("depth band is too small")
    region = normalized[y_min:y_max]
    vertical_gradient = np.abs(np.gradient(region, axis=0))
    gradient_scale = max(float(np.quantile(vertical_gradient, 0.95)), 1e-6)
    gradient = np.clip(vertical_gradient / gradient_scale, 0.0, 2.0)
    depth_bias = np.linspace(0.0, 1.0, region.shape[0], dtype=np.float64)[:, None]
    ridge_fields = (
        ("H0_BASELINE", 0.65 * region + 0.35 * gradient),
        ("H1_EDGE_DEEP", 0.40 * region + 0.60 * gradient + 0.08 * depth_bias),
        ("H2_INTENSITY_DEEP", 0.75 * region + 0.25 * gradient + 0.06 * depth_bias),
    )
    columns = np.arange(config.x_margin_pixels, width - config.x_margin_pixels, dtype=np.int32)
    tracks: list[InterfaceTrack] = []
    for track_id, ridge in ridge_fields[: config.maximum_hypotheses]:
        candidate = _solve_path(ridge, columns, y_min, config, track_id)
        if candidate.valid and _diverse(candidate, tracks, config):
            tracks.append(candidate)
    return tracks
