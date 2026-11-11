"""Home-relative spatial evidence aggregation and abstaining target inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


Direction = Literal["FORWARD", "REVERSE"]


@dataclass(frozen=True)
class EvidenceObservation:
    coordinate_mm: float
    probability: float
    direction: Direction
    frame_sequence: int
    track_id: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.coordinate_mm):
            raise ValueError("coordinate_mm must be finite")
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        if self.direction not in ("FORWARD", "REVERSE"):
            raise ValueError("direction must be FORWARD or REVERSE")
        if self.frame_sequence < 0 or not self.track_id:
            raise ValueError("invalid evidence identity")


@dataclass(frozen=True)
class SpatialMapConfig:
    grid_step_mm: float
    minimum_event_separation_mm: float
    raw_peak_minimum_score: float
    acceptance_score: float
    minimum_forward_frames: int
    minimum_reverse_frames: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.grid_step_mm) or self.grid_step_mm <= 0:
            raise ValueError("grid_step_mm must be finite and positive")
        if not math.isfinite(self.minimum_event_separation_mm) or self.minimum_event_separation_mm <= 0:
            raise ValueError("minimum_event_separation_mm must be finite and positive")
        if not 0.0 <= self.raw_peak_minimum_score <= self.acceptance_score <= 1.0:
            raise ValueError("invalid score thresholds")
        if min(self.minimum_forward_frames, self.minimum_reverse_frames) < 1:
            raise ValueError("direction support must be positive")


@dataclass(frozen=True)
class GridRecord:
    coordinate_mm: float
    score_q75: float
    score_median: float
    distinct_frame_count: int
    forward_count: int
    reverse_count: int


@dataclass(frozen=True)
class LandmarkEvent:
    coordinate_mm: float
    score: float
    forward_count: int
    reverse_count: int
    support_interval_mm: tuple[float, float]
    accepted: bool
    edge_clipped: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class TargetEstimate:
    status: Literal["SUCCESS", "ABSTAIN"]
    coordinate_mm: float | None
    uncertainty_interval_mm: tuple[float, float] | None
    evidence_event_count: int
    anchor_id: str
    reason: str | None


class SpatialEvidenceMap:
    """Aggregate per-frame image likelihoods on an explicit Home-relative grid."""

    def __init__(self, minimum_mm: float, maximum_mm: float, config: SpatialMapConfig) -> None:
        if not (math.isfinite(minimum_mm) and math.isfinite(maximum_mm)) or maximum_mm <= minimum_mm:
            raise ValueError("invalid map bounds")
        self.minimum_mm = float(minimum_mm)
        self.maximum_mm = float(maximum_mm)
        self.config = config
        self.grid = np.arange(
            self.minimum_mm,
            self.maximum_mm + 0.5 * config.grid_step_mm,
            config.grid_step_mm,
            dtype=np.float64,
        )
        self._evidence: list[EvidenceObservation] = []

    def add(self, evidence: EvidenceObservation) -> None:
        if evidence.coordinate_mm < self.minimum_mm or evidence.coordinate_mm > self.maximum_mm:
            raise ValueError("evidence coordinate lies outside the frozen map")
        self._evidence.append(evidence)

    def records(self) -> list[GridRecord]:
        # Multiple interface hypotheses from one frame may address the same grid location. Only
        # the strongest one counts, preventing hypothesis multiplicity from inflating support.
        by_bin_frame_direction: dict[tuple[int, int, str], float] = {}
        for evidence in self._evidence:
            bin_index = int(round((evidence.coordinate_mm - self.minimum_mm) / self.config.grid_step_mm))
            bin_index = int(np.clip(bin_index, 0, len(self.grid) - 1))
            key = (bin_index, evidence.frame_sequence, evidence.direction)
            by_bin_frame_direction[key] = max(by_bin_frame_direction.get(key, 0.0), evidence.probability)

        values: list[list[float]] = [[] for _ in self.grid]
        forward: list[set[int]] = [set() for _ in self.grid]
        reverse: list[set[int]] = [set() for _ in self.grid]
        for (bin_index, frame_sequence, direction), probability in by_bin_frame_direction.items():
            values[bin_index].append(probability)
            (forward if direction == "FORWARD" else reverse)[bin_index].add(frame_sequence)

        records: list[GridRecord] = []
        for index, coordinate in enumerate(self.grid):
            scores = values[index]
            records.append(
                GridRecord(
                    coordinate_mm=float(coordinate),
                    score_q75=float(np.quantile(scores, 0.75)) if scores else 0.0,
                    score_median=float(np.median(scores)) if scores else 0.0,
                    distinct_frame_count=len(forward[index] | reverse[index]),
                    forward_count=len(forward[index]),
                    reverse_count=len(reverse[index]),
                )
            )
        return records

    def events(self) -> list[LandmarkEvent]:
        records = self.records()
        scores = np.asarray([record.score_q75 for record in records], dtype=np.float64)
        if not np.any(scores >= self.config.raw_peak_minimum_score):
            return []
        local_candidates: list[int] = []
        for index, score in enumerate(scores):
            left = scores[index - 1] if index > 0 else -np.inf
            right = scores[index + 1] if index + 1 < len(scores) else -np.inf
            if score >= self.config.raw_peak_minimum_score and score >= left and score >= right:
                local_candidates.append(index)

        separation_bins = max(
            1,
            int(round(self.config.minimum_event_separation_mm / self.config.grid_step_mm)),
        )
        selected: list[int] = []
        for index in sorted(local_candidates, key=lambda item: (-scores[item], item)):
            if all(abs(index - other) >= separation_bins for other in selected):
                selected.append(index)
        selected.sort()

        events: list[LandmarkEvent] = []
        for index in selected:
            record = records[index]
            reasons: list[str] = []
            if record.score_q75 < self.config.acceptance_score:
                reasons.append("SCORE_BELOW_ACCEPTANCE")
            if record.forward_count < self.config.minimum_forward_frames:
                reasons.append("INSUFFICIENT_FORWARD_SUPPORT")
            if record.reverse_count < self.config.minimum_reverse_frames:
                reasons.append("INSUFFICIENT_REVERSE_SUPPORT")
            support_level = max(self.config.raw_peak_minimum_score, 0.8 * record.score_q75)
            left = index
            right = index
            while left > 0 and scores[left - 1] >= support_level:
                left -= 1
            while right + 1 < len(scores) and scores[right + 1] >= support_level:
                right += 1
            edge_clipped = left == 0 or right == len(scores) - 1
            if edge_clipped:
                reasons.append("EDGE_CLIPPED")
            events.append(
                LandmarkEvent(
                    coordinate_mm=record.coordinate_mm,
                    score=record.score_q75,
                    forward_count=record.forward_count,
                    reverse_count=record.reverse_count,
                    support_interval_mm=(records[left].coordinate_mm, records[right].coordinate_mm),
                    accepted=not reasons,
                    edge_clipped=edge_clipped,
                    rejection_reasons=tuple(reasons),
                )
            )
        return events

    def infer_ordered_target(
        self,
        *,
        expected_event_count: int,
        target_event_index: int,
        anchor_id: str,
    ) -> TargetEstimate:
        """Infer a target only under an explicit complete-order prior; otherwise abstain."""
        if expected_event_count <= 0 or not 0 <= target_event_index < expected_event_count:
            raise ValueError("invalid ordered-target specification")
        if not anchor_id:
            raise ValueError("anchor_id must explicitly declare the ordering prior")
        accepted = [event for event in self.events() if event.accepted]
        if len(accepted) != expected_event_count:
            return TargetEstimate(
                status="ABSTAIN",
                coordinate_mm=None,
                uncertainty_interval_mm=None,
                evidence_event_count=len(accepted),
                anchor_id=anchor_id,
                reason="INCOMPLETE_OR_AMBIGUOUS_ORDERED_MAP",
            )
        target = accepted[target_event_index]
        return TargetEstimate(
            status="SUCCESS",
            coordinate_mm=target.coordinate_mm,
            uncertainty_interval_mm=target.support_interval_mm,
            evidence_event_count=len(accepted),
            anchor_id=anchor_id,
            reason=None,
        )
