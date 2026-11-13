"""Labels and actor-side selection for supervised best-view acquisition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .stage2_features import extract_stage2_feature


@dataclass(frozen=True)
class Stage2LabelConfig:
    maximum_position_error_mm: float = 10.0
    maximum_orientation_error_deg: float = 10.0
    minimum_l4_visible_fraction: float = 0.05
    visibility_normalization_fraction: float = 0.20

    def __post_init__(self) -> None:
        values = (
            self.maximum_position_error_mm,
            self.maximum_orientation_error_deg,
            self.minimum_l4_visible_fraction,
            self.visibility_normalization_fraction,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("label thresholds must be finite and positive")
        if self.minimum_l4_visible_fraction > self.visibility_normalization_fraction:
            raise ValueError("visibility normalization must cover the validity threshold")


def supervised_targets(
    *,
    technical_image_valid: bool,
    position_error_mm: float,
    orientation_error_deg: float,
    l4_visible_fraction: float,
    config: Stage2LabelConfig,
) -> tuple[int, float]:
    """Build offline validity and quality targets from privileged annotations."""
    values = (position_error_mm, orientation_error_deg, l4_visible_fraction)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("annotation values must be finite and nonnegative")
    if l4_visible_fraction > 1.0:
        raise ValueError("l4_visible_fraction must not exceed one")
    validity = int(
        technical_image_valid
        and position_error_mm <= config.maximum_position_error_mm
        and orientation_error_deg <= config.maximum_orientation_error_deg
        and l4_visible_fraction >= config.minimum_l4_visible_fraction
    )
    visibility = min(l4_visible_fraction / config.visibility_normalization_fraction, 1.0)
    proximity = math.exp(
        -0.5 * (position_error_mm / config.maximum_position_error_mm) ** 2
        + (-0.5 * (orientation_error_deg / config.maximum_orientation_error_deg) ** 2)
    )
    quality = float(visibility * proximity) if technical_image_valid else 0.0
    return validity, quality


@dataclass(frozen=True)
class ScoredView:
    candidate_id: str
    relative_pose: tuple[float, float, float]
    validity_probability: float
    quality: float
    transition_valid: bool
    image_valid: bool
    frame_sha256: str

    def __post_init__(self) -> None:
        if not self.candidate_id or len(self.frame_sha256) != 64:
            raise ValueError("candidate identity is invalid")
        if len(self.relative_pose) != 3 or not all(
            math.isfinite(value) for value in self.relative_pose
        ):
            raise ValueError("relative_pose must contain three finite values")
        if not 0.0 <= self.validity_probability <= 1.0:
            raise ValueError("validity probability must be in [0, 1]")
        if not math.isfinite(self.quality):
            raise ValueError("quality must be finite")


def score_ultrasound(
    image: np.ndarray,
    classifier: Any,
    regressor: Any,
) -> tuple[float, float]:
    """Score one clean ultrasound image without accepting metadata inputs."""
    feature = extract_stage2_feature(image).reshape(1, -1)
    columns = {int(value): index for index, value in enumerate(classifier.classes_)}
    if 1 not in columns:
        raise RuntimeError("validity classifier has no positive class")
    probability = float(classifier.predict_proba(feature)[0, columns[1]])
    quality = float(regressor.predict(feature)[0])
    if not math.isfinite(probability) or not math.isfinite(quality):
        raise RuntimeError("nonfinite model output")
    return probability, quality


def select_best_view(
    candidates: Iterable[ScoredView], threshold: float
) -> ScoredView | None:
    """Rank quality only among physically valid classifier-accepted candidates."""
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    eligible = [
        item
        for item in candidates
        if item.transition_valid
        and item.image_valid
        and item.validity_probability >= threshold
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda item: (item.quality, item.validity_probability))


def fresh_confirmation_passes(
    remembered: ScoredView,
    confirmations: Iterable[ScoredView],
    *,
    threshold: float,
    maximum_quality_drop: float,
    required_count: int,
) -> bool:
    """Require new accepted frames whose predicted quality remains near the selected view."""
    if maximum_quality_drop < 0.0 or required_count <= 0:
        raise ValueError("confirmation limits are invalid")
    rows = list(confirmations)
    if len(rows) < required_count:
        return False
    if any(row.frame_sha256 == remembered.frame_sha256 for row in rows):
        return False
    accepted = [
        row
        for row in rows
        if row.transition_valid
        and row.image_valid
        and row.validity_probability >= threshold
        and row.quality >= remembered.quality - maximum_quality_drop
    ]
    return len(accepted) >= required_count
