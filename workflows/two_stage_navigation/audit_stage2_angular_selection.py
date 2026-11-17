#!/usr/bin/env python3
"""Audit angular reachability and frozen candidate selection from recorded Stage 2 runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = REPO_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.stage2_temporal_decision import (
    ReplayObservation,
    TemporalDecisionConfig,
    _evidence_for_candidate,
    reindex_trajectory_subset,
)

SEARCH_PHASES = ("coarse", "fine", "micro", "orientation_refine")
EARLY_PHASES = ("coarse", "fine", "micro")
EVALUATED_STARTS = ("confirm_new_01", "confirm_new_02", "confirm_new_03")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def orientation_error_deg(realized: float, target: float) -> float:
    # Keep the canonical coordinates.py convention: [-pi, pi).
    wrapped = (realized - target + math.pi) % (2.0 * math.pi) - math.pi
    return abs(wrapped) * 180.0 / math.pi


def decision_config(protocol: Mapping[str, Any]) -> TemporalDecisionConfig:
    values = protocol["decision"]
    return TemporalDecisionConfig(
        validity_probability_threshold=float(values["validity_probability_threshold"]),
        maximum_neighbor_translation_mm=float(values["maximum_neighbor_translation_mm"]),
        maximum_neighbor_angle_deg=float(values["maximum_neighbor_angle_deg"]),
        maximum_quality_drop=float(values["maximum_quality_drop"]),
        required_distinct_hash_observations=int(values["required_distinct_hash_observations"]),
        minimum_distinct_view_translation_mm=float(values["minimum_distinct_view_translation_mm"]),
        minimum_distinct_view_angle_deg=float(values["minimum_distinct_view_angle_deg"]),
    )


def associate_actions(
    frames: Sequence[Mapping[str, Any]], actions: Sequence[Mapping[str, Any]]
) -> dict[int, tuple[Mapping[str, Any], ...]]:
    """Associate sequential action rows with observations without crossing phase boundaries."""
    cursor = 0
    grouped: dict[int, tuple[Mapping[str, Any], ...]] = {}
    for frame in frames:
        phase = str(frame["phase"])
        count = int(frame["action_step_count"])
        while cursor < len(actions) and str(actions[cursor].get("phase")) != phase:
            cursor += 1
        if count == 0:
            grouped[int(frame["frame_sequence"])] = ()
            continue
        selected = tuple(actions[cursor : cursor + count])
        if len(selected) != count or any(str(row.get("phase")) != phase for row in selected):
            raise ValueError(
                f"cannot associate {count} actions with frame {frame['frame_sequence']} phase {phase}"
            )
        grouped[int(frame["frame_sequence"])] = selected
        cursor += count
    return grouped


def make_observations(
    frames: Sequence[Mapping[str, Any]], start_id: str
) -> tuple[ReplayObservation, ...]:
    rows = []
    for index, frame in enumerate(frames):
        rows.append(
            ReplayObservation(
                session_id=str(frame["session_id"]),
                trajectory_id=start_id,
                frame_sequence=int(frame["frame_sequence"]),
                trajectory_frame_sequence=index,
                producer_sequence=int(frame["producer_sequence"]),
                frame_sha256=str(frame["frame_sha256"]),
                relative_pose=tuple(float(value) for value in frame["relative_pose"]),
                validity_probability=float(frame["validity_probability"]),
                quality=float(frame["quality_prediction"]),
                transition_valid=bool(frame["transition_valid"]),
                image_valid=bool(frame["image_valid"]),
                producer_advanced=bool(frame["producer_advanced"]),
            )
        )
    return tuple(rows)


def eligible_sequences(
    observations: Sequence[ReplayObservation], config: TemporalDecisionConfig
) -> tuple[set[int], dict[int, dict[str, int]]]:
    if not observations:
        return set(), {}
    pool = reindex_trajectory_subset(observations)
    eligible: set[int] = set()
    support: dict[int, dict[str, int]] = {}
    for row in pool:
        evidence = _evidence_for_candidate(
            pool, row, config, require_distinct_viewpoints=True
        )
        if evidence is not None:
            eligible.add(row.frame_sequence)
            support[row.frame_sequence] = {
                "hashes": evidence.distinct_hash_count,
                "viewpoints": evidence.distinct_viewpoint_count,
            }
    return eligible, support


def candidate_filter_diagnostics(
    observations: Sequence[ReplayObservation], config: TemporalDecisionConfig
) -> dict[int, dict[str, int | bool]]:
    """Expose the frozen evidence filters without changing their decision order."""
    if not observations:
        return {}
    pool = reindex_trajectory_subset(observations)
    output: dict[int, dict[str, int | bool]] = {}
    for candidate in pool:
        technically_valid = bool(
            candidate.transition_valid and candidate.image_valid and candidate.producer_advanced
        )
        classifier_accepted = bool(
            technically_valid
            and candidate.validity_probability >= config.validity_probability_threshold
        )
        neighborhood: list[ReplayObservation] = []
        for row in pool:
            translation = math.hypot(
                candidate.relative_pose[0] - row.relative_pose[0],
                candidate.relative_pose[1] - row.relative_pose[1],
            )
            angle = orientation_error_deg(candidate.relative_pose[2], row.relative_pose[2])
            if (
                row.transition_valid
                and row.image_valid
                and row.producer_advanced
                and row.validity_probability >= config.validity_probability_threshold
                and translation <= config.maximum_neighbor_translation_mm
                and angle <= config.maximum_neighbor_angle_deg
            ):
                neighborhood.append(row)
        quality_rows = [
            row
            for row in neighborhood
            if row.quality >= candidate.quality - config.maximum_quality_drop
        ]
        unique_rows: list[ReplayObservation] = []
        seen: set[str] = set()
        for row in quality_rows:
            if row.frame_sha256 not in seen:
                seen.add(row.frame_sha256)
                unique_rows.append(row)
        distinct = [candidate]
        if classifier_accepted:
            for row in unique_rows:
                if row.frame_sha256 == candidate.frame_sha256:
                    continue
                if all(
                    math.hypot(
                        row.relative_pose[0] - prior.relative_pose[0],
                        row.relative_pose[1] - prior.relative_pose[1],
                    )
                    >= config.minimum_distinct_view_translation_mm
                    or orientation_error_deg(row.relative_pose[2], prior.relative_pose[2])
                    >= config.minimum_distinct_view_angle_deg
                    for prior in distinct
                ):
                    distinct.append(row)
        output[candidate.frame_sequence] = {
            "technically_valid": technically_valid,
            "classifier_accepted": classifier_accepted,
            "neighborhood_support": len(neighborhood),
            "quality_support": len(quality_rows),
            "distinct_hashes": len(unique_rows),
            "distinct_viewpoints": len(distinct) if classifier_accepted else 0,
        }
    return output


def best(rows: Iterable[Mapping[str, Any]], key) -> Mapping[str, Any] | None:
    values = list(rows)
    return None if not values else max(values, key=key)


def compact_frame(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    keys = (
        "frame_sequence",
        "phase",
        "requested_angle_rad",
        "realized_angle_rad",
        "angular_tracking_error_deg",
        "evaluator_orientation_error_deg",
        "position_error_mm",
        "l4_visible_fraction",
        "validity_probability",
        "quality_prediction",
        "actor_eligible",
        "full_gate",
    )
    return {key: row.get(key) for key in keys}


def audit_start(
    start_dir: Path,
    start_id: str,
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    public_path = start_dir / "public" / "frames.jsonl"
    partial = False
    if not public_path.exists():
        public_path = start_dir / "public" / "frames_partial.jsonl"
        partial = True
    action_path = start_dir / "privileged" / ("actions_partial.jsonl" if partial else "actions.jsonl")
    annotation_path = start_dir / "privileged" / "annotations.jsonl"
    actor_path = start_dir / "results" / "actor_result.json"
    evaluation_path = start_dir / "results" / "evaluation_report.json"
    frames = read_jsonl(public_path)
    actions = read_jsonl(action_path)
    annotations = read_jsonl(annotation_path) if annotation_path.exists() else []
    by_annotation = {int(row["frame_sequence"]): row for row in annotations}
    actions_by_frame = associate_actions(frames, actions)
    actor = json.loads(actor_path.read_text(encoding="utf-8")) if actor_path.exists() else None
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation_path.exists()
        else None
    )

    config = decision_config(protocol)
    observations = make_observations(frames, start_id)
    early_obs = tuple(row for row, frame in zip(observations, frames) if frame["phase"] in EARLY_PHASES)
    refine_obs = tuple(
        row for row, frame in zip(observations, frames) if frame["phase"] == "orientation_refine"
    )
    search_obs = tuple(row for row, frame in zip(observations, frames) if frame["phase"] in SEARCH_PHASES)
    early_eligible, early_support = eligible_sequences(early_obs, config)
    refine_eligible, refine_support = eligible_sequences(refine_obs, config)
    all_eligible, all_support = eligible_sequences(search_obs, config)
    early_filters = candidate_filter_diagnostics(early_obs, config)
    refine_filters = candidate_filter_diagnostics(refine_obs, config)

    handoff_angle = float(protocol["geometry"]["stage2_handoff_angle_rad"])
    gate = protocol["evaluation_gate"]
    threshold = float(protocol["decision"]["validity_probability_threshold"])
    selected_sequence = None if actor is None else actor.get("selected_frame_sequence")
    preliminary_sequence = None if actor is None else actor.get("preliminary_selected_frame_sequence")
    joined: list[dict[str, Any]] = []
    for frame in frames:
        sequence = int(frame["frame_sequence"])
        annotation = by_annotation.get(sequence)
        action_rows = actions_by_frame[sequence]
        requested_relative = [float(value) for value in frame["requested_relative_pose"]]
        requested_angle = handoff_angle + requested_relative[2]
        realized_angle = (
            None if annotation is None else float(annotation["realized_pose_index"][2])
        )
        target_angle = None if annotation is None else float(annotation["goal_cmd_pose"][2])
        evaluator_error = (
            None
            if realized_angle is None or target_angle is None
            else orientation_error_deg(realized_angle, target_angle)
        )
        if annotation is not None and not math.isclose(
            evaluator_error,
            float(annotation["orientation_error_deg"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"orientation join mismatch for {start_id} frame {sequence}")
        phase = str(frame["phase"])
        actor_eligible = bool(
            sequence in (refine_eligible if phase == "orientation_refine" else early_eligible)
        )
        position_error = None if annotation is None else float(annotation["position_error_mm"])
        visibility = None if annotation is None else float(annotation["l4_visible_fraction"])
        technical = bool(
            frame["transition_valid"] and frame["image_valid"] and frame["producer_advanced"]
        )
        full_gate = bool(
            technical
            and position_error is not None
            and evaluator_error is not None
            and visibility is not None
            and position_error <= float(gate["maximum_position_error_mm"])
            and evaluator_error <= float(gate["maximum_orientation_error_deg"])
            and visibility >= float(gate["minimum_l4_visible_fraction"])
        )
        support = (
            refine_support.get(sequence, {})
            if phase == "orientation_refine"
            else early_support.get(sequence, {})
        )
        filters = (
            refine_filters.get(sequence, {})
            if phase == "orientation_refine"
            else early_filters.get(sequence, {})
        )
        joined.append(
            {
                "start_id": start_id,
                "artifact_complete": not partial,
                "frame_sequence": sequence,
                "phase": phase,
                "requested_angle_rad": requested_angle,
                "requested_angle_deg": math.degrees(requested_angle),
                "realized_angle_rad": realized_angle,
                "realized_angle_deg": None if realized_angle is None else math.degrees(realized_angle),
                "angular_tracking_error_deg": (
                    None
                    if realized_angle is None
                    else orientation_error_deg(realized_angle, requested_angle)
                ),
                "evaluator_target_angle_rad": target_angle,
                "evaluator_target_angle_deg": None if target_angle is None else math.degrees(target_angle),
                "evaluator_orientation_error_deg": evaluator_error,
                "position_error_mm": position_error,
                "l4_visible_fraction": visibility,
                "validity_probability": float(frame["validity_probability"]),
                "quality_prediction": float(frame["quality_prediction"]),
                "classifier_accepted": float(frame["validity_probability"]) >= threshold,
                "actor_eligible": actor_eligible,
                "all_history_actor_eligible_hindsight": sequence in all_eligible,
                "supporting_hash_count": support.get("hashes", 0),
                "supporting_viewpoint_count": support.get("viewpoints", 0),
                "candidate_neighborhood_support_count": filters.get("neighborhood_support", 0),
                "candidate_quality_support_count": filters.get("quality_support", 0),
                "candidate_distinct_hash_count": filters.get("distinct_hashes", 0),
                "candidate_distinct_viewpoint_count": filters.get("distinct_viewpoints", 0),
                "producer_advanced": bool(frame["producer_advanced"]),
                "action_count": int(frame["action_step_count"]),
                "position_clamped": any(bool(row.get("command_position_clamped")) for row in action_rows),
                "angle_clamped": any(bool(row.get("command_angle_clamped")) for row in action_rows),
                "image_hash": str(frame["frame_sha256"]),
                "preliminary_selected": sequence == preliminary_sequence,
                "selected": sequence == selected_sequence,
                "technical_valid": technical,
                "full_gate": full_gate,
                "available_before_final_selection": phase in SEARCH_PHASES,
            }
        )

    search_rows = [row for row in joined if row["phase"] in SEARCH_PHASES]
    valid_rows = [row for row in search_rows if row["full_gate"]]
    eligible_rows = [row for row in search_rows if row["actor_eligible"]]
    all_history_eligible_rows = [
        row for row in search_rows if row["all_history_actor_eligible_hindsight"]
    ]
    selected_row = next((row for row in joined if row["selected"]), None)
    preliminary_row = next((row for row in joined if row["preliminary_selected"]), None)
    final_confirmation = (
        None
        if actor is None or not actor.get("fresh_confirmation_frame_sequences")
        else next(
            row
            for row in joined
            if row["frame_sequence"] == actor["fresh_confirmation_frame_sequences"][-1]
        )
    )
    best_orientation = (
        None
        if not search_rows or any(row["evaluator_orientation_error_deg"] is None for row in search_rows)
        else min(search_rows, key=lambda row: row["evaluator_orientation_error_deg"])
    )
    best_quality_eligible = best(eligible_rows, lambda row: (row["quality_prediction"], row["validity_probability"], -row["frame_sequence"]))
    best_probability_eligible = best(eligible_rows, lambda row: (row["validity_probability"], row["quality_prediction"], -row["frame_sequence"]))
    best_orientation_eligible = (
        None
        if not eligible_rows or any(row["evaluator_orientation_error_deg"] is None for row in eligible_rows)
        else min(eligible_rows, key=lambda row: row["evaluator_orientation_error_deg"])
    )
    best_all_history_quality = best(
        all_history_eligible_rows,
        lambda row: (row["quality_prediction"], row["validity_probability"], -row["frame_sequence"]),
    )

    if partial:
        primary_cause = "TECHNICAL_FRESHNESS_FAILURE"
    elif not valid_rows:
        primary_cause = "SEARCH_COVERAGE_FAILURE"
    elif not any(row["actor_eligible"] for row in valid_rows):
        primary_cause = "ACTOR_ELIGIBILITY_FAILURE"
    elif selected_row is None or not selected_row["full_gate"]:
        primary_cause = "SELECTION_FAILURE"
    elif final_confirmation is not None and not final_confirmation["full_gate"]:
        primary_cause = "POST_RETURN_DEGRADATION"
    else:
        primary_cause = "INCONCLUSIVE"

    exclusion = {
        "classifier_threshold": sum(not row["classifier_accepted"] for row in search_rows),
        "invalid_motion_or_image": sum(not row["technical_valid"] for row in search_rows),
        "non_advancing_producer": sum(not row["producer_advanced"] for row in search_rows),
        "multi_view_evidence": sum(row["classifier_accepted"] and row["technical_valid"] and not row["actor_eligible"] for row in search_rows),
        "neighborhood_support": sum(
            row["classifier_accepted"]
            and row["technical_valid"]
            and row["candidate_neighborhood_support_count"] < config.required_distinct_hash_observations
            for row in search_rows
        ),
        "quality_consistency": sum(
            row["classifier_accepted"]
            and row["technical_valid"]
            and row["candidate_neighborhood_support_count"] >= config.required_distinct_hash_observations
            and row["candidate_quality_support_count"] < config.required_distinct_hash_observations
            for row in search_rows
        ),
        "distinct_hashes": sum(
            row["classifier_accepted"]
            and row["technical_valid"]
            and row["candidate_quality_support_count"] >= config.required_distinct_hash_observations
            and row["candidate_distinct_hash_count"] < config.required_distinct_hash_observations
            for row in search_rows
        ),
        "distinct_viewpoints": sum(
            row["classifier_accepted"]
            and row["technical_valid"]
            and row["candidate_distinct_hash_count"] >= config.required_distinct_hash_observations
            and row["candidate_distinct_viewpoint_count"] < config.required_distinct_hash_observations
            for row in search_rows
        ),
        "refinement_only_pool": sum(row["phase"] in EARLY_PHASES for row in search_rows),
    }
    refinement_rows = [row for row in search_rows if row["phase"] == "orientation_refine"]
    sequential = {
        "refinement_candidates": len(refinement_rows),
        "technical_and_fresh": sum(row["technical_valid"] for row in refinement_rows),
        "classifier_accepted": sum(
            row["technical_valid"] and row["classifier_accepted"] for row in refinement_rows
        ),
        "minimum_neighborhood_support": sum(
            row["technical_valid"]
            and row["classifier_accepted"]
            and row["candidate_neighborhood_support_count"] >= config.required_distinct_hash_observations
            for row in refinement_rows
        ),
        "minimum_quality_consistent_support": sum(
            row["technical_valid"]
            and row["classifier_accepted"]
            and row["candidate_neighborhood_support_count"] >= config.required_distinct_hash_observations
            and row["candidate_quality_support_count"] >= config.required_distinct_hash_observations
            for row in refinement_rows
        ),
        "minimum_distinct_hashes": sum(
            row["technical_valid"]
            and row["classifier_accepted"]
            and row["candidate_quality_support_count"] >= config.required_distinct_hash_observations
            and row["candidate_distinct_hash_count"] >= config.required_distinct_hash_observations
            for row in refinement_rows
        ),
        "actor_eligible_distinct_viewpoints": sum(row["actor_eligible"] for row in refinement_rows),
    }
    summary = {
        "start_id": start_id,
        "artifact_complete": not partial,
        "frame_count": len(joined),
        "search_frame_count": len(search_rows),
        "fully_valid_observed_count": len(valid_rows),
        "fully_valid_frame_sequences": [row["frame_sequence"] for row in valid_rows],
        "fully_valid_actor_eligible_count": sum(row["actor_eligible"] for row in valid_rows),
        "fully_valid_actor_eligible_sequences": [row["frame_sequence"] for row in valid_rows if row["actor_eligible"]],
        "fully_valid_in_final_pool_count": sum(row["phase"] == "orientation_refine" for row in valid_rows),
        "selected_fully_valid": None if selected_row is None else selected_row["full_gate"],
        "primary_cause": primary_cause,
        "best_physically_observed": compact_frame(best_orientation),
        "highest_quality_actor_eligible": compact_frame(best_quality_eligible),
        "highest_probability_actor_eligible": compact_frame(best_probability_eligible),
        "smallest_orientation_error_actor_eligible": compact_frame(best_orientation_eligible),
        "best_all_history_model_ranked": compact_frame(best_all_history_quality),
        "preliminary_selected": compact_frame(preliminary_row),
        "selected": compact_frame(selected_row),
        "final_confirmation": compact_frame(final_confirmation),
        "return_orientation_error_deg": None if actor is None else actor.get("return_orientation_error_deg"),
        "final_evaluator_orientation_error_deg": None if evaluation is None else evaluation.get("final_orientation_error_deg"),
        "refinement_only_discarded_higher_model_ranked_earlier_candidate": bool(
            selected_row is not None
            and best_all_history_quality is not None
            and best_all_history_quality["phase"] in EARLY_PHASES
            and best_all_history_quality["frame_sequence"] != selected_row["frame_sequence"]
            and best_all_history_quality["quality_prediction"] > selected_row["quality_prediction"]
        ),
        "refinement_only_excluded_fully_valid_earlier_candidate": any(
            row["phase"] in EARLY_PHASES for row in valid_rows
        ),
        "exclusion_nonexclusive_counts": exclusion,
        "refinement_sequential_filter_counts": sequential,
    }
    inputs = {
        str(path.relative_to(start_dir)): file_sha256(path)
        for path in (public_path, action_path, annotation_path, actor_path, evaluation_path)
        if path.exists()
    }
    return joined, summary, inputs


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_angular_coverage(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    phase_colors = {"coarse": "#4c78a8", "fine": "#f58518", "micro": "#54a24b", "orientation_refine": "#e45756"}
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False, constrained_layout=True)
    for axis, start_id in zip(axes, EVALUATED_STARTS):
        subset = [row for row in rows if row["start_id"] == start_id and row["phase"] in SEARCH_PHASES]
        axis.plot([row["frame_sequence"] for row in subset], [row["requested_angle_rad"] for row in subset], color="black", linewidth=1, alpha=0.55, label="requested")
        axis.plot([row["frame_sequence"] for row in subset], [row["realized_angle_rad"] for row in subset], color="#666666", linewidth=1, alpha=0.8, label="realized")
        for phase, color in phase_colors.items():
            selected = [row for row in subset if row["phase"] == phase]
            axis.scatter([row["frame_sequence"] for row in selected], [row["realized_angle_rad"] for row in selected], s=24, color=color, label=phase)
        selected = [row for row in subset if row["selected"]]
        if selected:
            axis.scatter([selected[0]["frame_sequence"]], [selected[0]["realized_angle_rad"]], marker="*", s=180, color="gold", edgecolor="black", zorder=5, label="selected")
        target = float(subset[0]["evaluator_target_angle_rad"])
        axis.axhspan(target - math.radians(10), target + math.radians(10), color="#72b7b2", alpha=0.15, label="evaluator-only <=10 deg")
        axis.axhline(target, color="#72b7b2", linestyle="--", linewidth=1, label="evaluator-only target")
        axis.axhline(3.00, color="#b279a2", linestyle=":", linewidth=1, label="native command ceiling")
        axis.axhline(2.97, color="#ff9da6", linestyle="-.", linewidth=1, label="refinement ceiling")
        axis.set_title(start_id)
        axis.set_ylabel("Angle (rad)")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Causal frame sequence")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(unique.values(), unique.keys(), loc="outside upper center", ncol=5, fontsize=8)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_quality_orientation(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    colors = {"confirm_new_01": "#4c78a8", "confirm_new_02": "#f58518", "confirm_new_03": "#54a24b"}
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for start_id, color in colors.items():
        subset = [row for row in rows if row["start_id"] == start_id and row["phase"] in SEARCH_PHASES]
        for accepted, marker in ((True, "o"), (False, "x")):
            values = [row for row in subset if row["classifier_accepted"] is accepted]
            axis.scatter([row["evaluator_orientation_error_deg"] for row in values], [row["quality_prediction"] for row in values], color=color, marker=marker, alpha=0.7, label=f"{start_id} {'accepted' if accepted else 'rejected'}")
        selected = [row for row in subset if row["selected"]]
        if selected:
            axis.scatter([selected[0]["evaluator_orientation_error_deg"]], [selected[0]["quality_prediction"]], marker="*", s=220, color="gold", edgecolor="black", zorder=5)
    axis.axvline(10.0, color="black", linestyle="--", linewidth=1, label="10 deg evaluator gate")
    axis.set_xlabel("Evaluator-only orientation error (deg)")
    axis.set_ylabel("Frozen predicted quality")
    axis.set_title("Image quality ranking versus geometric orientation (evaluator-only join)")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, ncol=2)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_failure_decomposition(summaries: Sequence[Mapping[str, Any]], output: Path) -> None:
    evaluated = [row for row in summaries if row["start_id"] in EVALUATED_STARTS]
    columns = ("Fully valid observed", "Actor-eligible", "Final pool", "Selected", "Preserved after return")
    matrix = []
    for row in evaluated:
        matrix.append(
            [
                row["fully_valid_observed_count"] > 0,
                row["fully_valid_actor_eligible_count"] > 0,
                row["fully_valid_in_final_pool_count"] > 0,
                bool(row["selected_fully_valid"]),
                bool(row["final_confirmation"] and row["final_confirmation"]["full_gate"]),
            ]
        )
    figure, axis = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    axis.axis("off")
    text = [["YES" if value else "NO" for value in line] for line in matrix]
    table = axis.table(cellText=text, rowLabels=[row["start_id"] for row in evaluated], colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for (row, column), cell in table.get_celld().items():
        if row > 0 and column >= 0:
            cell.set_facecolor("#d9ead3" if cell.get_text().get_text() == "YES" else "#f4cccc")
    axis.set_title("Stage 2 failure decomposition (evaluator-only retrospective audit)", pad=16)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    input_hashes: dict[str, dict[str, str]] = {}
    for start_id in ("confirm_new_00", *EVALUATED_STARTS):
        rows, summary, hashes = audit_start(args.run_root / "starts" / start_id, start_id, protocol)
        all_rows.extend(rows)
        summaries.append(summary)
        input_hashes[start_id] = hashes
    write_csv(args.output_dir / "per_frame_audit.csv", all_rows)
    flat_summaries = [
        {
            "start_id": row["start_id"],
            "artifact_complete": row["artifact_complete"],
            "search_frame_count": row["search_frame_count"],
            "fully_valid_observed_count": row["fully_valid_observed_count"],
            "fully_valid_actor_eligible_count": row["fully_valid_actor_eligible_count"],
            "fully_valid_in_final_pool_count": row["fully_valid_in_final_pool_count"],
            "selected_fully_valid": row["selected_fully_valid"],
            "primary_cause": row["primary_cause"],
            "selected_frame": None if row["selected"] is None else row["selected"]["frame_sequence"],
            "selected_orientation_error_deg": None if row["selected"] is None else row["selected"]["evaluator_orientation_error_deg"],
            "best_observed_frame": None if row["best_physically_observed"] is None else row["best_physically_observed"]["frame_sequence"],
            "best_observed_orientation_error_deg": None if row["best_physically_observed"] is None else row["best_physically_observed"]["evaluator_orientation_error_deg"],
            "final_orientation_error_deg": row["final_evaluator_orientation_error_deg"],
        }
        for row in summaries
    ]
    write_csv(args.output_dir / "per_start_summary.csv", flat_summaries)
    plot_angular_coverage(all_rows, args.output_dir / "angular_coverage.png")
    plot_quality_orientation(all_rows, args.output_dir / "quality_vs_orientation.png")
    plot_failure_decomposition(summaries, args.output_dir / "failure_decomposition.png")
    result = {
        "schema": "sonogym-stage2-angular-selection-audit-v1",
        "historical_verdict": "CONFIRMATION_NO_GO",
        "diagnostic_recommendation": "SELECTION_REPAIR_RECOMMENDED",
        "run_root": str(args.run_root),
        "protocol": str(args.protocol),
        "protocol_sha256": file_sha256(args.protocol),
        "input_hashes": input_hashes,
        "starts": summaries,
        "preserved_results": {
            "V16": "CONFIRMATION_NO_GO",
            "V17": "CONFIRMATION_NO_GO",
            "V18": "CONFIRMATION_NO_GO",
            "new_start_confirmation_v1": "CONFIRMATION_NO_GO",
        },
        "gpu_jobs_run": 0,
        "simulator_runs": 0,
    }
    (args.output_dir / "audit_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"recommendation": result["diagnostic_recommendation"], "starts": flat_summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
