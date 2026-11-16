"""Pure helpers for the bounded Stage 2 movement-contract audit."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .stage2_temporal_decision import (
    MovementContract,
    ReplayObservation,
    TemporalDecisionConfig,
    transition_valid_for_contract,
)


@dataclass(frozen=True)
class CandidateSupportProfile:
    potential_candidate_count: int
    maximum_neighborhood_observations: int
    maximum_quality_consistent_observations: int
    maximum_distinct_hashes: int
    maximum_distinct_viewpoints: int
    qualifying_frame_sequences: tuple[int, ...]


def original_transition_valid(record: Mapping[str, Any]) -> bool:
    """Reproduce the immutable temporal replay movement check."""
    return transition_valid_for_contract(
        record,
        (),
        contract=MovementContract.ORIGINAL_FROZEN,
        angle_ceiling_exception_declared=False,
    )


def clamp_consistent_transition_valid(
    record: Mapping[str, Any],
    action_rows: Sequence[Mapping[str, Any]],
    *,
    angle_ceiling_exception_declared: bool,
) -> bool:
    """Apply only the capture protocol's narrow angle-ceiling exception.

    This diagnostic never changes the deployed actor contract. An angle-only clamp is retained
    only for a source whose frozen acquisition protocol declared the exception and whose complete
    low-level movement record remains finite, mapping-valid, surface-valid, nonterminal, and free
    of position clamps.
    """
    return transition_valid_for_contract(
        record,
        action_rows,
        contract=MovementContract.CLAMP_CONSISTENT_DIAGNOSTIC,
        angle_ceiling_exception_declared=angle_ceiling_exception_declared,
    )


def observations_with_contract(
    observations: Sequence[ReplayObservation],
    records: Sequence[Mapping[str, Any]],
    actions_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    contract: MovementContract,
    angle_ceiling_exception_declared: bool,
) -> tuple[ReplayObservation, ...]:
    if len(observations) != len(records):
        raise ValueError("observations and records must align")
    output: list[ReplayObservation] = []
    for observation, record in zip(observations, records):
        if observation.frame_sequence != int(record["frame_sequence"]):
            raise ValueError("observation and record frame identities do not align")
        if contract is MovementContract.ORIGINAL_FROZEN:
            valid = original_transition_valid(record)
        elif contract is MovementContract.CLAMP_CONSISTENT_DIAGNOSTIC:
            valid = clamp_consistent_transition_valid(
                record,
                actions_by_frame.get(observation.frame_sequence, ()),
                angle_ceiling_exception_declared=angle_ceiling_exception_declared,
            )
        else:  # pragma: no cover - Enum makes this unreachable for typed callers.
            raise ValueError(f"unsupported movement contract {contract!r}")
        output.append(replace(observation, transition_valid=valid))
    return tuple(output)


def _wrap_to_pi(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _pose_distance(
    left: ReplayObservation, right: ReplayObservation
) -> tuple[float, float]:
    translation_mm = math.hypot(
        left.relative_pose[0] - right.relative_pose[0],
        left.relative_pose[1] - right.relative_pose[1],
    )
    angle_deg = (
        abs(_wrap_to_pi(left.relative_pose[2] - right.relative_pose[2]))
        * 180.0
        / math.pi
    )
    return translation_mm, angle_deg


def _candidate_eligible(
    observation: ReplayObservation, config: TemporalDecisionConfig
) -> bool:
    return bool(
        observation.producer_advanced
        and observation.transition_valid
        and observation.image_valid
        and observation.validity_probability >= config.validity_probability_threshold
    )


def candidate_support_profile(
    observations: Sequence[ReplayObservation], config: TemporalDecisionConfig
) -> CandidateSupportProfile:
    """Return maxima at each candidate-evidence filter without evaluator fields."""
    candidates = [row for row in observations if _candidate_eligible(row, config)]
    maximum_neighborhood = 0
    maximum_quality = 0
    maximum_hashes = 0
    maximum_viewpoints = 0
    qualifying_sequences: set[int] = set()
    for candidate in candidates:
        neighborhood = []
        for row in candidates:
            translation_mm, angle_deg = _pose_distance(candidate, row)
            if (
                translation_mm <= config.maximum_neighbor_translation_mm
                and angle_deg <= config.maximum_neighbor_angle_deg
            ):
                neighborhood.append(row)
        quality_rows = [
            row
            for row in neighborhood
            if row.quality >= candidate.quality - config.maximum_quality_drop
        ]
        unique_hashes: list[ReplayObservation] = []
        seen_hashes: set[str] = set()
        for row in quality_rows:
            if row.frame_sha256 not in seen_hashes:
                unique_hashes.append(row)
                seen_hashes.add(row.frame_sha256)
        distinct_views: list[ReplayObservation] = [candidate]
        for row in unique_hashes:
            if row.frame_sha256 == candidate.frame_sha256:
                continue
            if all(
                (
                    _pose_distance(row, prior)[0]
                    >= config.minimum_distinct_view_translation_mm
                    or _pose_distance(row, prior)[1]
                    >= config.minimum_distinct_view_angle_deg
                )
                for prior in distinct_views
            ):
                distinct_views.append(row)
        maximum_neighborhood = max(maximum_neighborhood, len(neighborhood))
        maximum_quality = max(maximum_quality, len(quality_rows))
        maximum_hashes = max(maximum_hashes, len(unique_hashes))
        maximum_viewpoints = max(maximum_viewpoints, len(distinct_views))
        qualifying_sequences.update(row.frame_sequence for row in quality_rows)
    return CandidateSupportProfile(
        potential_candidate_count=len(candidates),
        maximum_neighborhood_observations=maximum_neighborhood,
        maximum_quality_consistent_observations=maximum_quality,
        maximum_distinct_hashes=maximum_hashes,
        maximum_distinct_viewpoints=maximum_viewpoints,
        qualifying_frame_sequences=tuple(sorted(qualifying_sequences)),
    )
