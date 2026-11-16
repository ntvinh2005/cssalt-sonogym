"""Pure causal decision logic for exploratory Stage 2 temporal replay."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .contracts import assert_no_privileged_fields


class DecisionState(str, Enum):
    SEARCH = "SEARCH"
    CANDIDATE_FOUND = "CANDIDATE_FOUND"
    GATHER_EVIDENCE = "GATHER_EVIDENCE"
    SELECT_BEST = "SELECT_BEST"
    RETURN_BEST = "RETURN_BEST"
    FRESH_CONFIRM = "FRESH_CONFIRM"
    ACCEPT = "ACCEPT"
    ABSTAIN = "ABSTAIN"
    FAILURE = "FAILURE"


class MovementContract(str, Enum):
    ORIGINAL_FROZEN = "original_frozen"
    CLAMP_CONSISTENT_DIAGNOSTIC = "clamp_consistent_diagnostic"


def transition_valid_for_contract(
    record: Mapping[str, Any],
    action_rows: Sequence[Mapping[str, Any]],
    *,
    contract: MovementContract,
    angle_ceiling_exception_declared: bool,
) -> bool:
    """Validate movement under an explicitly selected replay contract."""
    assert_no_privileged_fields(record)
    for row in action_rows:
        assert_no_privileged_fields(row)
    original = bool(
        record["action_accepted"]
        and record["action_completed"]
        and not record["action_clamped"]
        and record["contact_ok"]
        and record["workspace_ok"]
    )
    if contract is MovementContract.ORIGINAL_FROZEN:
        return original
    if contract is not MovementContract.CLAMP_CONSISTENT_DIAGNOSTIC:
        raise ValueError(f"unsupported movement contract {contract!r}")

    delta = record.get("executed_delta_relative_mm")
    try:
        finite_motion = bool(
            isinstance(delta, (list, tuple))
            and len(delta) == 2
            and all(math.isfinite(float(value)) for value in delta)
            and math.isfinite(float(record["executed_delta_angle_rad"]))
        )
    except (KeyError, TypeError, ValueError):
        return False
    if not bool(
        record.get("action_accepted", False)
        and record.get("action_completed", False)
        and record.get("contact_ok", False)
        and record.get("workspace_ok", False)
        and finite_motion
    ):
        return False
    if not bool(record.get("action_clamped", False)):
        return True
    if (
        not angle_ceiling_exception_declared
        or bool(record.get("action_position_clamped", False))
        or not bool(record.get("action_angle_clamped", False))
        or not action_rows
        or int(record.get("action_steps_since_previous_frame", -1))
        != len(action_rows)
    ):
        return False
    return bool(
        all(
            bool(row.get("finite", False))
            and bool(row.get("surface_proxy_ok", False))
            and bool(row.get("command_mapping_valid", False))
            and not bool(row.get("terminated", False))
            and not bool(row.get("truncated", False))
            and not bool(row.get("command_position_clamped", False))
            for row in action_rows
        )
        and any(bool(row.get("command_angle_clamped", False)) for row in action_rows)
    )


@dataclass(frozen=True)
class TemporalDecisionConfig:
    validity_probability_threshold: float
    maximum_neighbor_translation_mm: float
    maximum_neighbor_angle_deg: float
    maximum_quality_drop: float
    required_distinct_hash_observations: int
    minimum_distinct_view_translation_mm: float
    minimum_distinct_view_angle_deg: float

    def __post_init__(self) -> None:
        values = (
            self.validity_probability_threshold,
            self.maximum_neighbor_translation_mm,
            self.maximum_neighbor_angle_deg,
            self.maximum_quality_drop,
            self.minimum_distinct_view_translation_mm,
            self.minimum_distinct_view_angle_deg,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("decision limits must be finite and nonnegative")
        if self.validity_probability_threshold > 1.0:
            raise ValueError("validity threshold must not exceed one")
        if self.required_distinct_hash_observations < 2:
            raise ValueError("temporal evidence requires at least two distinct hashes")
        if (
            self.minimum_distinct_view_translation_mm == 0.0
            and self.minimum_distinct_view_angle_deg == 0.0
        ):
            raise ValueError("a distinct viewpoint must require physical separation")


@dataclass(frozen=True)
class ReplayObservation:
    session_id: str
    trajectory_id: str
    frame_sequence: int
    trajectory_frame_sequence: int
    producer_sequence: int
    frame_sha256: str
    relative_pose: tuple[float, float, float]
    validity_probability: float
    quality: float
    transition_valid: bool
    image_valid: bool
    producer_advanced: bool


    def __post_init__(self) -> None:
        if not self.session_id or not self.trajectory_id:
            raise ValueError("session and trajectory identities must be nonempty")
        if (
            min(
                self.frame_sequence,
                self.trajectory_frame_sequence,
                self.producer_sequence,
            )
            < 0
        ):
            raise ValueError("sequences must be nonnegative")
        if len(self.frame_sha256) != 64:
            raise ValueError("frame_sha256 must contain 64 characters")
        try:
            int(self.frame_sha256, 16)
        except ValueError as error:
            raise ValueError("frame_sha256 must be hexadecimal") from error
        if len(self.relative_pose) != 3 or not all(
            math.isfinite(value) for value in self.relative_pose
        ):
            raise ValueError("relative_pose must contain x mm, z mm and angle rad")
        if not 0.0 <= self.validity_probability <= 1.0:
            raise ValueError("validity_probability must be in [0, 1]")
        if not math.isfinite(self.quality):
            raise ValueError("quality must be finite")

    @classmethod
    def from_actor_record(
        cls,
        record: Mapping[str, Any],
        *,
        relative_pose: tuple[float, float, float],
        validity_probability: float,
        quality: float,
        producer_advanced: bool,
        action_rows: Sequence[Mapping[str, Any]] = (),
        movement_contract: MovementContract = MovementContract.ORIGINAL_FROZEN,
        angle_ceiling_exception_declared: bool = False,
    ) -> "ReplayObservation":
        """Construct one replay observation after a fail-closed leakage check."""
        assert_no_privileged_fields(record)
        transition_valid = transition_valid_for_contract(
            record,
            action_rows,
            contract=movement_contract,
            angle_ceiling_exception_declared=angle_ceiling_exception_declared,
        )
        return cls(
            session_id=str(record["session_id"]),
            trajectory_id=str(record["trajectory_id"]),
            frame_sequence=int(record["frame_sequence"]),
            trajectory_frame_sequence=int(record["trajectory_frame_sequence"]),
            producer_sequence=int(record["producer_sequence"]),
            frame_sha256=str(record["frame_sha256"]),
            relative_pose=relative_pose,
            validity_probability=float(validity_probability),
            quality=float(quality),
            transition_valid=transition_valid,
            image_valid=bool(record["image_valid"]),
            producer_advanced=bool(producer_advanced),
        )


def reindex_trajectory_subset(
    observations: Sequence[ReplayObservation],
) -> tuple[ReplayObservation, ...]:
    """Create a replay-only contiguous view without changing global frame identity."""
    if not observations:
        return ()
    if len({row.session_id for row in observations}) != 1 or len(
        {row.trajectory_id for row in observations}
    ) != 1:
        raise ValueError("replay subset cannot cross a session or trajectory boundary")
    if any(
        current.frame_sequence <= previous.frame_sequence
        or current.producer_sequence < previous.producer_sequence
        for previous, current in zip(observations, observations[1:])
    ):
        raise ValueError("replay subset must retain causal global ordering")
    return tuple(
        replace(row, trajectory_frame_sequence=index)
        for index, row in enumerate(observations)
    )


@dataclass(frozen=True)
class CandidateEvidence:
    candidate: ReplayObservation
    supporting_observations: tuple[ReplayObservation, ...]
    distinct_hash_count: int
    distinct_viewpoint_count: int


@dataclass(frozen=True)
class ReplayDecision:
    state: DecisionState
    selected: ReplayObservation | None
    evidence: CandidateEvidence | None
    observed_frame_count: int
    state_trace: tuple[DecisionState, ...]


def _wrap_to_pi(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _pose_distance(
    left: ReplayObservation, right: ReplayObservation
) -> tuple[float, float]:
    translation = math.hypot(
        left.relative_pose[0] - right.relative_pose[0],
        left.relative_pose[1] - right.relative_pose[1],
    )
    angle_deg = (
        abs(_wrap_to_pi(left.relative_pose[2] - right.relative_pose[2]))
        * 180.0
        / math.pi
    )
    return translation, angle_deg


def _physically_admissible(observation: ReplayObservation) -> bool:
    return observation.transition_valid and observation.image_valid


def _classifier_accepted(
    observation: ReplayObservation, config: TemporalDecisionConfig
) -> bool:
    return bool(
        _physically_admissible(observation)
        and observation.validity_probability >= config.validity_probability_threshold
    )


def _validate_sequence(observations: Sequence[ReplayObservation]) -> None:
    if not observations:
        return
    session_ids = {row.session_id for row in observations}
    trajectory_ids = {row.trajectory_id for row in observations}
    if len(session_ids) != 1 or len(trajectory_ids) != 1:
        raise ValueError("evidence cannot cross a session or trajectory boundary")
    ordered = sorted(observations, key=lambda row: row.trajectory_frame_sequence)
    if list(observations) != ordered:
        raise ValueError("observations must be in causal trajectory order")
    expected = list(range(len(observations)))
    observed = [row.trajectory_frame_sequence for row in observations]
    if observed != expected:
        raise ValueError(
            "trajectory_frame_sequence must restart at zero and be contiguous"
        )
    if any(
        current.frame_sequence <= previous.frame_sequence
        or current.producer_sequence < previous.producer_sequence
        for previous, current in zip(observations, observations[1:])
    ):
        raise ValueError(
            "frame sequence must increase and producer sequence cannot decrease"
        )


def reconstruct_trajectory_observations(
    actor_records: Sequence[Mapping[str, Any]],
    validity_probabilities: Sequence[float],
    quality_predictions: Sequence[float],
    *,
    actions_by_frame: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
    movement_contract: MovementContract = MovementContract.ORIGINAL_FROZEN,
    angle_ceiling_exception_declared: bool = False,
) -> tuple[ReplayObservation, ...]:
    """Reconstruct actor-relative pose from executed deltas within one trajectory."""
    if not (
        len(actor_records) == len(validity_probabilities) == len(quality_predictions)
    ):
        raise ValueError("actor records and frozen predictions must align")
    x_mm = 0.0
    z_mm = 0.0
    angle_rad = 0.0
    previous_producer_sequence: int | None = None
    observations: list[ReplayObservation] = []
    for record, probability, quality in zip(
        actor_records, validity_probabilities, quality_predictions
    ):
        assert_no_privileged_fields(record)
        delta = record.get("executed_delta_relative_mm")
        if not isinstance(delta, (list, tuple)) or len(delta) != 2:
            raise ValueError("executed_delta_relative_mm is missing or invalid")
        delta_angle = float(record["executed_delta_angle_rad"])
        if not all(math.isfinite(float(value)) for value in (*delta, delta_angle)):
            raise ValueError("executed relative motion must be finite")
        x_mm += float(delta[0])
        z_mm += float(delta[1])
        angle_rad = _wrap_to_pi(angle_rad + delta_angle)
        observations.append(
            ReplayObservation.from_actor_record(
                record,
                relative_pose=(x_mm, z_mm, angle_rad),
                validity_probability=float(probability),
                quality=float(quality),
                producer_advanced=(
                    previous_producer_sequence is None
                    or int(record["producer_sequence"]) > previous_producer_sequence
                ),
                action_rows=(
                    ()
                    if actions_by_frame is None
                    else actions_by_frame.get(int(record["frame_sequence"]), ())
                ),
                movement_contract=movement_contract,
                angle_ceiling_exception_declared=angle_ceiling_exception_declared,
            )
        )
        previous_producer_sequence = int(record["producer_sequence"])
    _validate_sequence(observations)
    return tuple(observations)


def _in_candidate_neighborhood(
    candidate: ReplayObservation,
    other: ReplayObservation,
    config: TemporalDecisionConfig,
) -> bool:
    translation_mm, angle_deg = _pose_distance(candidate, other)
    return bool(
        translation_mm <= config.maximum_neighbor_translation_mm
        and angle_deg <= config.maximum_neighbor_angle_deg
    )


def _is_distinct_viewpoint(
    left: ReplayObservation, right: ReplayObservation, config: TemporalDecisionConfig
) -> bool:
    translation_mm, angle_deg = _pose_distance(left, right)
    return bool(
        translation_mm >= config.minimum_distinct_view_translation_mm
        or angle_deg >= config.minimum_distinct_view_angle_deg
    )


def _evidence_for_candidate(
    observations: Sequence[ReplayObservation],
    candidate: ReplayObservation,
    config: TemporalDecisionConfig,
    *,
    require_distinct_viewpoints: bool,
) -> CandidateEvidence | None:
    if not candidate.producer_advanced or not _classifier_accepted(candidate, config):
        return None
    eligible = [
        row
        for row in observations
        if row.producer_advanced
        and _classifier_accepted(row, config)
        and _in_candidate_neighborhood(candidate, row, config)
        and row.quality >= candidate.quality - config.maximum_quality_drop
    ]
    unique_hashes: list[ReplayObservation] = []
    seen_hashes: set[str] = set()
    for row in eligible:
        if row.frame_sha256 not in seen_hashes:
            unique_hashes.append(row)
            seen_hashes.add(row.frame_sha256)
    if len(unique_hashes) < config.required_distinct_hash_observations:
        return None

    distinct_views: list[ReplayObservation] = [candidate]
    for row in unique_hashes:
        if row.frame_sha256 == candidate.frame_sha256:
            continue
        if all(_is_distinct_viewpoint(row, prior, config) for prior in distinct_views):
            distinct_views.append(row)
    if require_distinct_viewpoints:
        if len(distinct_views) < config.required_distinct_hash_observations:
            return None
        support = tuple(distinct_views)
    else:
        support = tuple(unique_hashes)
    return CandidateEvidence(
        candidate=candidate,
        supporting_observations=support,
        distinct_hash_count=len({row.frame_sha256 for row in support}),
        distinct_viewpoint_count=len(distinct_views),
    )


def replay_decision(
    observations: Sequence[ReplayObservation],
    *,
    policy: str,
    config: TemporalDecisionConfig,
    available_until_frame_sequence: int | None = None,
) -> ReplayDecision:
    """Replay one declared policy using only the available causal prefix."""
    _validate_sequence(observations)
    available = tuple(
        row
        for row in observations
        if available_until_frame_sequence is None
        or row.frame_sequence <= available_until_frame_sequence
    )
    if not available:
        return ReplayDecision(
            state=DecisionState.ABSTAIN,
            selected=None,
            evidence=None,
            observed_frame_count=0,
            state_trace=(DecisionState.SEARCH, DecisionState.ABSTAIN),
        )

    selected: ReplayObservation | None = None
    evidence: CandidateEvidence | None = None
    observed_frame_count = len(available)
    if policy == "baseline_a_single_frame":
        selected = next(
            (row for row in available if _classifier_accepted(row, config)), None
        )
        if selected is not None:
            observed_frame_count = available.index(selected) + 1
    elif policy == "baseline_b_quality_ranked":
        eligible = [row for row in available if _classifier_accepted(row, config)]
        selected = max(eligible, key=_selection_key) if eligible else None
    elif policy == "diagnostic_quality_only":
        eligible = [row for row in available if _physically_admissible(row)]
        selected = max(eligible, key=_selection_key) if eligible else None
    elif policy in {"candidate_c_temporal", "candidate_d_multi_view"}:
        candidates: list[CandidateEvidence] = []
        for row in available:
            current = _evidence_for_candidate(
                available,
                row,
                config,
                require_distinct_viewpoints=policy == "candidate_d_multi_view",
            )
            if current is not None:
                candidates.append(current)
        if candidates:
            evidence = max(candidates, key=lambda item: _selection_key(item.candidate))
            selected = evidence.candidate
    else:
        raise ValueError(f"unknown replay policy {policy!r}")

    if selected is None:
        return ReplayDecision(
            state=DecisionState.ABSTAIN,
            selected=None,
            evidence=None,
            observed_frame_count=len(available),
            state_trace=(
                DecisionState.SEARCH,
                DecisionState.GATHER_EVIDENCE,
                DecisionState.ABSTAIN,
            ),
        )
    trace = [DecisionState.SEARCH, DecisionState.CANDIDATE_FOUND]
    if policy in {"candidate_c_temporal", "candidate_d_multi_view"}:
        trace.append(DecisionState.GATHER_EVIDENCE)
    trace.extend((DecisionState.SELECT_BEST, DecisionState.ACCEPT))
    return ReplayDecision(
        state=DecisionState.ACCEPT,
        selected=selected,
        evidence=evidence,
        observed_frame_count=observed_frame_count,
        state_trace=tuple(trace),
    )


def _selection_key(observation: ReplayObservation) -> tuple[float, float, int, str]:
    return (
        observation.quality,
        observation.validity_probability,
        -observation.frame_sequence,
        observation.frame_sha256,
    )


def fresh_confirmation_passes_distinct(
    remembered: ReplayObservation,
    confirmations: Iterable[ReplayObservation],
    *,
    config: TemporalDecisionConfig,
    required_count: int,
) -> bool:
    """Check fresh same-pose repeatability without claiming multi-view evidence."""
    rows = tuple(confirmations)
    if required_count <= 0 or len(rows) < required_count:
        return False
    if any(
        row.session_id != remembered.session_id
        or row.frame_sequence <= remembered.frame_sequence
        or row.producer_sequence <= remembered.producer_sequence
        for row in rows
    ):
        return False
    hashes = [row.frame_sha256 for row in rows]
    if remembered.frame_sha256 in hashes or len(set(hashes)) != len(hashes):
        return False
    ordered = sorted(rows, key=lambda row: row.frame_sequence)
    if list(rows) != ordered or any(
        current.producer_sequence <= previous.producer_sequence
        for previous, current in zip(rows, rows[1:])
    ):
        return False
    accepted = [
        row
        for row in rows
        if _classifier_accepted(row, config)
        and row.quality >= remembered.quality - config.maximum_quality_drop
    ]
    return len(accepted) >= required_count
