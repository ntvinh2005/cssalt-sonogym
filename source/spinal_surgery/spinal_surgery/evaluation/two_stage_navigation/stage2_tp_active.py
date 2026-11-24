"""Actor-only TP surface-core scoring and causal view selection.

The runtime code in this module consumes clean ultrasound, executed probe poses,
and frozen image models.  Simulator geometry is deliberately absent; it belongs
to the offline evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import assert_no_privileged_fields
from .stage1_features import PatchSpec
from .tp_surface_core_dataset import PatchProposal, actor_visible_patch_proposals


TARGETS = ("L4_RIGHT_TP_SURFACE_CORE", "L5_RIGHT_TP_SURFACE_CORE")
POC_PROTOCOL_IDS = frozenset(
    {
        "sonogym-s0010-mbb-single-anatomy-poc-v1",
        "sonogym-s0010-mbb-single-anatomy-poc-v2",
        "sonogym-s0010-mbb-single-anatomy-poc-v3",
        "sonogym-s0010-mbb-single-anatomy-poc-v4",
        "sonogym-s0010-mbb-single-anatomy-poc-v5",
        "sonogym-s0010-mbb-single-anatomy-poc-v5r1",
        "sonogym-s0010-mbb-single-anatomy-poc-v5r2",
        "sonogym-s0010-mbb-single-anatomy-poc-v6",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r1",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r2",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r3",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r4",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r5",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r6",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r7",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r8",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r9",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r10",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r11",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r12",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r13",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r14",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r15",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r16",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r17",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r18",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r19",
        "sonogym-s0010-mbb-single-anatomy-poc-v6r20",
    }
)


def perception_mode(protocol: Mapping[str, Any]) -> str:
    """Return the declared Stage 2 perception path."""

    mode = str(protocol.get("perception", {}).get("mode", "legacy_dual_model"))
    if mode not in {"legacy_dual_model", "unet_only"}:
        raise ValueError(f"unsupported Stage 2 perception mode: {mode}")
    return mode


def tp_soft_evidence_score(
    probabilities: np.ndarray, *, top_k_pixels: int = 20
) -> float:
    """Rank actor-visible TP near-misses without changing acceptance masks."""

    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.ndim != 3 or probability.shape[0] != 4:
        raise ValueError("probabilities must be 4 x H x W")
    if top_k_pixels < 1:
        raise ValueError("top_k_pixels must be positive")
    values = probability[1].ravel()
    count = min(top_k_pixels, values.size)
    strongest = np.partition(values, values.size - count)[-count:]
    return float(np.mean(strongest))


def tp_reacquisition_offsets(
    confirmation_spec: Mapping[str, Any],
    *,
    level: str,
    supporting_view_count: int,
    minimum_supporting_views: int,
) -> tuple[tuple[float, float, float], ...]:
    """Choose the bounded TP rescan without widening the general sweep.

    A single-view L5 TP uses the declared small orientation changes first.
    Other levels and already-supported hypotheses keep the established local
    translation plan.
    """

    key = (
        "l5_single_view_tp_orientation_offsets_index"
        if level == "L5" and supporting_view_count < minimum_supporting_views
        else "tp_local_search_offsets_index"
    )
    raw = confirmation_spec.get(key, ())
    offsets: list[tuple[float, float, float]] = []
    for row in raw:
        values = tuple(float(value) for value in row)
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"invalid TP reacquisition offset in {key}")
        offsets.append(values)
    return tuple(offsets)


def proxy_confirmation_requirement_met(
    states: Sequence[str], *, required_count: int, minimum_positive_count: int
) -> bool:
    """Apply a frozen evaluator confirmation count without changing actor inputs."""

    if required_count < 1 or not 1 <= minimum_positive_count <= required_count:
        raise ValueError("invalid proxy-positive fresh-confirmation requirement")
    return bool(
        len(states) == required_count
        and sum(state == "PROXY_POSITIVE" for state in states) >= minimum_positive_count
    )


def actor_confirmation_requirement_met(
    views: Sequence["TargetView"],
    *,
    probability_threshold: float,
    required_count: int,
    minimum_positive_count: int,
) -> bool:
    """Require fresh eligible views and a bounded count above the actor threshold."""

    if (
        required_count < 1
        or not 1 <= minimum_positive_count <= required_count
        or not math.isfinite(probability_threshold)
    ):
        raise ValueError("invalid actor fresh-confirmation requirement")
    return bool(
        len(views) == required_count
        and len({row.frame_sha256 for row in views}) == len(views)
        and all(row.eligible for row in views)
        and sum(row.probability >= probability_threshold for row in views)
        >= minimum_positive_count
    )


def select_tp_refinement_anchor(
    rows: Sequence[Mapping[str, Any]],
    public_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Choose one bounded TP search center from actor-visible evidence.

    The selected row must already be linked to the historical TP hypothesis.
    A large, confident semantic interface is preferred over a single detector
    score spike.  The caller may search around this pose once, but must not
    recursively promote refinement predictions into new anchors.
    """

    candidates: list[tuple[tuple[float, ...], Mapping[str, Any]]] = []
    for row in rows:
        frame_sequence = int(row.get("frame_sequence", -1))
        observations = row.get("semantic_observations", ())
        if (
            not bool(row.get("predicted"))
            or not bool(row.get("image_valid"))
            or not bool(row.get("producer_advanced"))
            or not bool(row.get("fresh_relative_to_source"))
            or frame_sequence < 0
            or frame_sequence >= len(public_rows)
            or not observations
        ):
            continue
        probability = float(public_rows[frame_sequence].get("target_probability", 0.0))
        if not math.isfinite(probability):
            continue
        pixel_count = max(float(item.get("pixel_count", 0.0)) for item in observations)
        confidence = max(float(item.get("confidence", 0.0)) for item in observations)
        if pixel_count <= 0.0 or not math.isfinite(confidence):
            continue
        candidates.append(
            (
                (pixel_count, probability, confidence, -float(frame_sequence)),
                row,
            )
        )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def tp_refinement_is_required(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_predicted_count: int,
    large_interface_pixel_count: int,
) -> bool:
    """Trigger one bounded TP refinement only when current evidence is unstable."""

    if minimum_predicted_count < 1 or large_interface_pixel_count < 1:
        raise ValueError("invalid TP refinement trigger")
    predicted = [
        row
        for row in rows
        if bool(row.get("predicted"))
        and bool(row.get("image_valid"))
        and bool(row.get("producer_advanced"))
        and bool(row.get("fresh_relative_to_source"))
        and row.get("semantic_observations")
    ]
    maximum_pixels = max(
        (
            int(observation.get("pixel_count", 0))
            for row in predicted
            for observation in row["semantic_observations"]
        ),
        default=0,
    )
    return bool(
        len(predicted) < minimum_predicted_count
        or maximum_pixels >= large_interface_pixel_count
    )


def observation_evidence_key(row: Mapping[str, Any]) -> tuple[int, str, float, float]:
    return (
        int(row["frame_sequence"]), str(row["frame_sha256"]),
        float(row["row"]), float(row["column"]),
    )


def valid_evidence_frame(frame: Mapping[str, Any], sequence: int, frame_hash: str) -> bool:
    return (
        frame["frame_sequence"] == sequence
        and frame["frame_sha256"] == frame_hash
        and all(frame.get(field, False) for field in (
            "transition_valid", "image_valid", "contact_ok", "producer_advanced"
        ))
    )


def select_landmark_return_sources(
    hypothesis: Mapping[str, Any],
    public_rows: Sequence[Mapping[str, Any]],
    *,
    maximum_poses: int,
) -> list[Mapping[str, Any]]:
    """Return the valid independent observations closest to the reconstructed landmark."""
    if hypothesis["position_mm"] is None:
        return []
    center = np.asarray(hypothesis["position_mm"])
    return sorted(
        (
            row for row in hypothesis["supporting_observations"]
            if valid_evidence_frame(
                public_rows[row["frame_sequence"]], row["frame_sequence"], row["frame_sha256"]
            )
        ),
        key=lambda row: (
            float(np.linalg.norm(np.asarray(row["coordinate_mm"]) - center)),
            -row["confidence"],
            row["frame_sequence"],
        ),
    )[:maximum_poses]


def canonical_landmark_confirmations(
    attempts: Sequence[Mapping[str, Any]],
    public_rows: Sequence[Mapping[str, Any]],
    *,
    identity: str,
    evidence_keys: set[tuple[int, str, float, float]],
) -> list[dict[str, Any]]:
    """Retain only the exact observations credited by the final hypothesis."""
    result = {}
    for attempt in attempts:
        if attempt["identity"] != identity or not all(attempt.get(field, False) for field in (
            "predicted", "image_valid", "producer_advanced", "fresh_relative_to_source"
        )):
            continue
        sequence = int(attempt["frame_sequence"])
        frame_hash = str(attempt["frame_sha256"])
        if not valid_evidence_frame(public_rows[sequence], sequence, frame_hash):
            continue
        observations = [
            row for row in attempt["semantic_observations"]
            if row["frame_sequence"] == sequence
            and row["frame_sha256"] == frame_hash
            and row["semantic_class"] == identity.rsplit("_", 1)[1]
            and observation_evidence_key(row) in evidence_keys
        ]
        if observations:
            result.setdefault(sequence, {**attempt, "semantic_observations": observations})
    return list(result.values())


def select_tp_reacquisition_confirmations(
    rows: Sequence[Mapping[str, Any]],
    public_rows: Sequence[Mapping[str, Any]],
    *,
    required_count: int,
    probability_threshold: float,
) -> list[Mapping[str, Any]]:
    """Select stable fresh TP confirmations without evaluator information."""

    if required_count < 1 or not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("invalid TP reacquisition confirmation requirement")
    eligible: list[tuple[tuple[float, ...], Mapping[str, Any]]] = []
    seen_hashes: set[str] = set()
    for row in rows:
        frame_sequence = int(row.get("frame_sequence", -1))
        frame_hash = str(row.get("frame_sha256", ""))
        observations = row.get("semantic_observations", ())
        if (
            not bool(row.get("predicted"))
            or not bool(row.get("image_valid"))
            or not bool(row.get("producer_advanced"))
            or not bool(row.get("fresh_relative_to_source"))
            or frame_sequence < 0
            or frame_sequence >= len(public_rows)
            or not observations
            or not frame_hash
            or frame_hash in seen_hashes
        ):
            continue
        probability = float(public_rows[frame_sequence].get("target_probability", 0.0))
        if not math.isfinite(probability) or probability < probability_threshold:
            continue
        pixel_count = max(float(item.get("pixel_count", 0.0)) for item in observations)
        confidence = max(float(item.get("confidence", 0.0)) for item in observations)
        if pixel_count <= 0.0 or not math.isfinite(confidence):
            continue
        seen_hashes.add(frame_hash)
        eligible.append(
            (
                (pixel_count, probability, confidence, -float(frame_sequence)),
                row,
            )
        )
    refinement = [
        item
        for item in eligible
        if str(item[1].get("confirmation_kind", "")).startswith("REFINEMENT_")
    ]
    if len(refinement) >= required_count:
        ranked = sorted(refinement, key=lambda item: item[0], reverse=True)
    else:
        ranked = sorted(
            eligible,
            key=lambda item: (
                -item[0][1],
                int(item[1]["frame_sequence"]),
            ),
        )
    return [row for _, row in ranked[:required_count]]


@dataclass(frozen=True)
class TargetView:
    """One actor-visible observation summarized for target-specific selection."""

    target: str
    frame_sequence: int
    producer_sequence: int
    frame_sha256: str
    pose_index: tuple[float, float, float]
    probability: float
    proposal_row: float | None
    proposal_column: float | None
    proposal_track_id: str | None
    transition_valid: bool
    image_valid: bool
    producer_advanced: bool

    @property
    def eligible(self) -> bool:
        return bool(
            self.transition_valid
            and self.image_valid
            and self.producer_advanced
            and self.proposal_row is not None
            and self.proposal_column is not None
        )


@dataclass(frozen=True)
class TargetDecision:
    """Causal target-view decision and its physically distinct support."""

    status: str
    selected: TargetView | None
    supporting_frame_sequences: tuple[int, ...]
    distinct_hash_count: int
    distinct_viewpoint_count: int
    reason: str


def group_a_target_handoff_pose(
    actor_result: Mapping[str, Any],
    *,
    session_id: str,
    target_level: str,
    home_z_index: float,
    label_resolution_mm: float,
    stage2_angle_rad: float,
) -> np.ndarray:
    """Convert a frozen seven-landmark actor result into a target-level pose."""

    assert_no_privileged_fields(actor_result)
    if actor_result.get("schema") != "sonogym-stage1-group-a-confirmation-actor-v1":
        raise ValueError("unexpected Group A actor schema")
    if actor_result.get("training_or_threshold_selection_performed") is not False:
        raise ValueError("handoff requires frozen Stage 1 inference")
    sessions = [
        row for row in actor_result.get("results", ()) if row.get("session_id") == session_id
    ]
    if len(sessions) != 1:
        raise ValueError("Stage 1 session is missing or duplicated")
    session = sessions[0]
    if session.get("map_actor_status") != "SUCCESS":
        raise ValueError("Stage 1 anatomical map did not succeed")
    predictions = [
        row for row in session.get("predictions", ()) if row.get("anatomy") == target_level
    ]
    if len(predictions) != 1:
        raise ValueError("target-level prediction is missing or duplicated")
    prediction = predictions[0]
    if prediction.get("actor_status") != "SUCCESS":
        raise ValueError("target-level Stage 1 prediction abstained")
    home = session.get("declared_home")
    if not isinstance(home, Mapping):
        raise ValueError("declared Home is missing")
    values = np.asarray(
        [
            float(home["patient_x_index"]),
            float(home_z_index) + float(prediction["actor_coordinate_mm"]) / label_resolution_mm,
            float(stage2_angle_rad),
        ],
        dtype=np.float64,
    )
    if label_resolution_mm <= 0.0 or not np.all(np.isfinite(values)):
        raise ValueError("handoff values must be finite")
    return values


def positive_probability(estimator: Any, features: np.ndarray, class_id: int) -> np.ndarray:
    """Read one declared class probability from a fitted scikit-learn estimator."""

    columns = {int(value): index for index, value in enumerate(estimator.classes_)}
    if class_id not in columns:
        raise ValueError(f"model has no class {class_id}")
    return np.asarray(estimator.predict_proba(features)[:, columns[class_id]], dtype=np.float64)


def score_target_image(
    image: np.ndarray,
    *,
    target: str,
    estimator: Any,
    class_id: int,
    patch_spec: PatchSpec,
    column_stride_pixels: int,
) -> tuple[float, PatchProposal | None, int]:
    """Return the maximum target probability across actor-visible proposals."""

    if target not in TARGETS:
        raise ValueError(f"unsupported target {target}")
    proposals = actor_visible_patch_proposals(
        image,
        patch_spec=patch_spec,
        column_stride_pixels=column_stride_pixels,
    )
    if not proposals:
        return 0.0, None, 0
    features = np.stack([proposal.feature for proposal in proposals])
    probabilities = positive_probability(estimator, features, class_id)
    best = int(np.argmax(probabilities))
    return float(probabilities[best]), proposals[best], len(proposals)


def _pose_separation(
    left: TargetView, right: TargetView, *, label_resolution_mm: float
) -> tuple[float, float]:
    left_pose = np.asarray(left.pose_index, dtype=np.float64)
    right_pose = np.asarray(right.pose_index, dtype=np.float64)
    translation = float(np.linalg.norm(left_pose[:2] - right_pose[:2]) * label_resolution_mm)
    angle = abs(float((left_pose[2] - right_pose[2] + math.pi) % (2.0 * math.pi) - math.pi))
    return translation, math.degrees(angle)


def select_target_view(
    views: Sequence[TargetView],
    *,
    target: str,
    probability_threshold: float,
    required_support: int,
    neighborhood_translation_mm: float,
    neighborhood_angle_deg: float,
    minimum_view_translation_mm: float,
    minimum_view_angle_deg: float,
    label_resolution_mm: float,
) -> TargetDecision:
    """Select the strongest causally supported target view or abstain.

    Support must be both local to the candidate and physically distinct.  Exact
    duplicate image hashes never add support.
    """

    if target not in TARGETS:
        raise ValueError(f"unsupported target {target}")
    if required_support < 1 or not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("invalid target decision settings")
    candidates = [
        view
        for view in views
        if view.target == target and view.eligible and view.probability >= probability_threshold
    ]
    if not candidates:
        return TargetDecision("ABSTAIN", None, (), 0, 0, "NO_ELIGIBLE_TARGET_VIEW")

    best_supported: tuple[tuple[float, int, int], TargetView, tuple[TargetView, ...]] | None = None
    best_insufficient: tuple[tuple[int, float, int], TargetView, tuple[TargetView, ...]] | None = None
    for candidate in candidates:
        local = []
        for view in candidates:
            translation, angle = _pose_separation(
                candidate, view, label_resolution_mm=label_resolution_mm
            )
            if translation <= neighborhood_translation_mm and angle <= neighborhood_angle_deg:
                local.append(view)
        # Deterministically retain one observation per exact image hash.
        unique_by_hash: dict[str, TargetView] = {}
        for view in sorted(local, key=lambda row: (-row.probability, row.frame_sequence)):
            unique_by_hash.setdefault(view.frame_sha256, view)
        distinct: list[TargetView] = []
        for view in sorted(unique_by_hash.values(), key=lambda row: row.frame_sequence):
            if not distinct:
                distinct.append(view)
                continue
            if all(
                (lambda separation: separation[0] >= minimum_view_translation_mm
                 or separation[1] >= minimum_view_angle_deg)(
                    _pose_separation(view, prior, label_resolution_mm=label_resolution_mm)
                )
                for prior in distinct
            ):
                distinct.append(view)
        if len(distinct) >= required_support:
            # Evidence is a gate.  Among supported candidates, preserve the
            # Unity best-valid-view philosophy and rank target utility first.
            rank = (candidate.probability, len(distinct), -candidate.frame_sequence)
            record = (rank, candidate, tuple(distinct))
            if best_supported is None or record[0] > best_supported[0]:
                best_supported = record
        else:
            rank = (len(distinct), candidate.probability, -candidate.frame_sequence)
            record = (rank, candidate, tuple(distinct))
            if best_insufficient is None or record[0] > best_insufficient[0]:
                best_insufficient = record

    if best_supported is None:
        assert best_insufficient is not None
        _, _, support = best_insufficient
        return TargetDecision(
            "ABSTAIN",
            None,
            tuple(view.frame_sequence for view in support),
            len({view.frame_sha256 for view in support}),
            len(support),
            "INSUFFICIENT_DISTINCT_SUPPORT",
        )
    _, selected, support = best_supported
    result = TargetDecision(
        "ACCEPT",
        selected,
        tuple(view.frame_sequence for view in support),
        len({view.frame_sha256 for view in support}),
        len(support),
        "SUPPORTED_TARGET_VIEW",
    )
    assert_no_privileged_fields(
        {
            "target_name": target,
            "status": result.status,
            "supporting_frame_sequences": list(result.supporting_frame_sequences),
        }
    )
    return result

