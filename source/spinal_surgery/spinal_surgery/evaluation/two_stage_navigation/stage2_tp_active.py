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
    }
)


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

