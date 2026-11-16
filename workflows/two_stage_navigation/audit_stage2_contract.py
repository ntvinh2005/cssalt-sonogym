#!/usr/bin/env python3
"""Audit Stage 2 clamp semantics and the 143/144 abstention waterfall."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.contracts import assert_no_privileged_fields  # noqa: E402
from two_stage_navigation.stage2_contract_diagnostic import (  # noqa: E402
    MovementContract,
    candidate_support_profile,
    observations_with_contract,
    original_transition_valid,
)
from two_stage_navigation.stage2_temporal_decision import (  # noqa: E402
    DecisionState,
    TemporalDecisionConfig,
    reconstruct_trajectory_observations,
    replay_decision,
)


POLICIES = ("candidate_c_temporal", "candidate_d_multi_view")
CONTRACTS = tuple(contract.value for contract in MovementContract)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def seal(path: Path) -> str:
    digest = file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def load_frame_scores(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["source"], row["session_id"], int(row["frame_sequence"]))
            if key in output:
                raise RuntimeError(f"duplicate frozen score identity {key}")
            output[key] = {
                **row,
                "frame_sequence": int(row["frame_sequence"]),
                "trajectory_frame_sequence": int(row["trajectory_frame_sequence"]),
                "validity_probability": float(row["validity_probability"]),
                "quality_prediction": float(row["quality_prediction"]),
                "image_validity_target": int(row["image_validity_target"]),
                "geometric_validity_target": int(row["geometric_validity_target"]),
            }
    return output


def trajectory_groups(rows: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    grouped: dict[str, list[int]] = {}
    order: list[str] = []
    for index, row in enumerate(rows):
        identity = str(row["trajectory_id"])
        if identity not in grouped:
            grouped[identity] = []
            order.append(identity)
        grouped[identity].append(index)
    return [grouped[identity] for identity in order]


def action_groups(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "frame_sequence" not in row:
            raise RuntimeError("action row has no frame_sequence")
        grouped[int(row["frame_sequence"])].append(dict(row))
    return grouped


def clamp_category(row: Mapping[str, Any]) -> str:
    position = bool(row["action_position_clamped"])
    angle = bool(row["action_angle_clamped"])
    if position and angle:
        return "both_position_and_angle"
    if position:
        return "position_only"
    if angle:
        return "angle_only"
    return "no_clamp"


def zero_executed_movement(row: Mapping[str, Any]) -> bool:
    delta = [float(value) for value in row["executed_delta_relative_mm"]]
    return bool(
        math.hypot(*delta) <= 1e-12
        and abs(float(row["executed_delta_angle_rad"])) <= 1e-12
    )


def actor_collector_accepted(row: Mapping[str, Any]) -> bool:
    return bool(
        row["action_accepted"]
        and row["action_completed"]
        and row["contact_ok"]
        and row["workspace_ok"]
        and row["image_valid"]
    )


def aggregate_frame_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    categories = Counter(str(row["clamp_category"]) for row in rows)
    category_hashes = {
        category: len(
            {
                (str(row["source"]), str(row["frame_sha256"]))
                for row in rows
                if row["clamp_category"] == category
            }
        )
        for category in (
            "no_clamp",
            "angle_only",
            "position_only",
            "both_position_and_angle",
        )
    }
    boolean_fields = (
        "unaccepted_or_incomplete",
        "accepted_angle_only_rejected_original",
        "zero_executed_movement",
        "nonadvancing_producer",
        "nonadvancing_zero_action",
        "nonadvancing_identical_previous_hash",
        "nonadvancing_nonzero_movement",
        "fresh_same_pose",
        "fresh_same_pose_distinct_hash",
        "ceiling_limited_capture_flag",
    )
    return {
        "frame_count": len(rows),
        "distinct_source_hash_count": len(
            {(str(row["source"]), str(row["frame_sha256"])) for row in rows}
        ),
        "clamp_category_frame_counts": dict(categories),
        "clamp_category_distinct_source_hash_counts": category_hashes,
        **{
            field: {
                "frame_count": sum(int(bool(row[field])) for row in rows),
                "distinct_source_hash_count": len(
                    {
                        (str(row["source"]), str(row["frame_sha256"]))
                        for row in rows
                        if row[field]
                    }
                ),
            }
            for field in boolean_fields
        },
    }


def aggregate_decisions(
    rows: Sequence[Mapping[str, Any]], *, contract: str, policy: str
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row[f"{contract}_{policy}_outcome"] == DecisionState.ACCEPT.value
    ]
    false_count = sum(
        int(row[f"{contract}_{policy}_selected_image_validity"] == 0)
        for row in selected
    )
    return {
        "trajectory_count": len(rows),
        "accepted_count": len(selected),
        "false_accepted_count": false_count,
        "correct_accepted_count": len(selected) - false_count,
        "accepted_precision": None
        if not selected
        else (len(selected) - false_count) / len(selected),
        "coverage": None if not rows else len(selected) / len(rows),
        "abstention_count": len(rows) - len(selected),
    }


def waterfall(rows: Sequence[Mapping[str, Any]], *, contract: str) -> list[dict[str, Any]]:
    stages = (
        ("all_trajectories", lambda row: True),
        ("collector_accepted_observation", lambda row: row["collector_accepted_count"] >= 1),
        ("collector_classifier_candidate", lambda row: row["collector_classifier_count"] >= 1),
        ("at_least_two_raw_distinct_hashes", lambda row: row["raw_distinct_hash_count"] >= 2),
        ("at_least_three_raw_distinct_hashes", lambda row: row["raw_distinct_hash_count"] >= 3),
        ("at_least_two_producer_advanced", lambda row: row["producer_advanced_count"] >= 2),
        ("at_least_three_producer_advanced", lambda row: row["producer_advanced_count"] >= 3),
        ("three_movement_valid_candidates", lambda row: row[f"{contract}_potential_candidate_count"] >= 3),
        ("three_in_candidate_neighborhood", lambda row: row[f"{contract}_maximum_neighborhood_observations"] >= 3),
        ("three_quality_consistent", lambda row: row[f"{contract}_maximum_quality_consistent_observations"] >= 3),
        ("three_distinct_qualifying_hashes", lambda row: row[f"{contract}_maximum_distinct_hashes"] >= 3),
        ("three_distinct_qualifying_viewpoints", lambda row: row[f"{contract}_maximum_distinct_viewpoints"] >= 3),
        ("final_temporal_accept", lambda row: row[f"{contract}_candidate_c_temporal_outcome"] == "ACCEPT"),
        ("final_multi_view_accept", lambda row: row[f"{contract}_candidate_d_multi_view_outcome"] == "ACCEPT"),
    )
    surviving = list(rows)
    output = []
    for stage, predicate in stages:
        before = len(surviving)
        surviving = [row for row in surviving if predicate(row)]
        output.append(
            {
                "stage": stage,
                "input_trajectory_count": before,
                "surviving_trajectory_count": len(surviving),
                "eliminated_at_stage": before - len(surviving),
                "fixed_denominator": len(rows),
            }
        )
    return output


def nonexclusive_failures(
    rows: Sequence[Mapping[str, Any]], *, contract: str
) -> dict[str, int]:
    conditions = {
        "no_collector_accepted_observation": lambda row: row["collector_accepted_count"] < 1,
        "no_collector_classifier_candidate": lambda row: row["collector_classifier_count"] < 1,
        "fewer_than_two_raw_distinct_hashes": lambda row: row["raw_distinct_hash_count"] < 2,
        "fewer_than_three_raw_distinct_hashes": lambda row: row["raw_distinct_hash_count"] < 3,
        "fewer_than_two_producer_advanced": lambda row: row["producer_advanced_count"] < 2,
        "fewer_than_three_producer_advanced": lambda row: row["producer_advanced_count"] < 3,
        "fewer_than_three_movement_valid_candidates": lambda row: row[f"{contract}_potential_candidate_count"] < 3,
        "no_candidate_with_three_neighborhood_observations": lambda row: row[f"{contract}_maximum_neighborhood_observations"] < 3,
        "no_candidate_with_three_quality_consistent_observations": lambda row: row[f"{contract}_maximum_quality_consistent_observations"] < 3,
        "no_candidate_with_three_distinct_hashes": lambda row: row[f"{contract}_maximum_distinct_hashes"] < 3,
        "no_candidate_with_three_distinct_viewpoints": lambda row: row[f"{contract}_maximum_distinct_viewpoints"] < 3,
    }
    return {
        name: sum(int(predicate(row)) for row in rows)
        for name, predicate in conditions.items()
    }


def plot_waterfall(report: Mapping[str, Any], output: Path) -> None:
    original = report["waterfalls"][MovementContract.ORIGINAL_FROZEN.value]
    diagnostic = report["waterfalls"][
        MovementContract.CLAMP_CONSISTENT_DIAGNOSTIC.value
    ]
    labels = [row["stage"] for row in original]
    x = list(range(len(labels)))
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        x,
        [row["surviving_trajectory_count"] for row in original],
        marker="o",
        label="original frozen",
    )
    axis.plot(
        x,
        [row["surviving_trajectory_count"] for row in diagnostic],
        marker="o",
        label="clamp-consistent diagnostic",
    )
    axis.set_xticks(x, labels, rotation=60, ha="right")
    axis.set_ylabel("trajectories remaining / 144")
    axis.set_ylim(0, 150)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()

    protocol_hash = file_sha256(args.protocol)
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "sonogym-stage2-contract-audit-v1":
        raise RuntimeError("unexpected audit protocol")
    prior = protocol["prior_replay"]
    score_path = Path(prior["frame_scores"])
    trajectory_path = Path(prior["trajectory_results"])
    if file_sha256(score_path) != prior["frame_scores_sha256"]:
        raise RuntimeError("frozen frame score hash mismatch")
    if file_sha256(trajectory_path) != prior["trajectory_results_sha256"]:
        raise RuntimeError("frozen trajectory result hash mismatch")
    scores = load_frame_scores(score_path)
    prior_trajectory_rows = read_jsonl(trajectory_path)
    prior_lookup = {
        (row["source"], row["trajectory_id"], row["policy"]): row
        for row in prior_trajectory_rows
        if row["policy"] in POLICIES
    }

    rule = protocol["decision_rules"]
    decision_config = TemporalDecisionConfig(
        validity_probability_threshold=float(rule["validity_probability_threshold"]),
        maximum_neighbor_translation_mm=float(rule["maximum_neighbor_translation_mm"]),
        maximum_neighbor_angle_deg=float(rule["maximum_neighbor_angle_deg"]),
        maximum_quality_drop=float(rule["maximum_quality_drop"]),
        required_distinct_hash_observations=int(rule["required_distinct_hash_observations"]),
        minimum_distinct_view_translation_mm=float(rule["minimum_distinct_view_translation_mm"]),
        minimum_distinct_view_angle_deg=float(rule["minimum_distinct_view_angle_deg"]),
    )
    frame_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    original_reproduction_mismatch_count = 0
    dataset_audit: list[dict[str, Any]] = []

    for dataset in protocol["datasets"]:
        source = str(dataset["name"])
        capture_root = Path(dataset["root"])
        summary = capture_root / "results" / "capture_summary.json"
        if file_sha256(summary) != dataset["capture_summary_sha256"]:
            raise RuntimeError(f"capture summary mismatch for {source}")
        capture_protocol = REPOSITORY_ROOT / dataset["capture_protocol"]
        if file_sha256(capture_protocol) != dataset["capture_protocol_sha256"]:
            raise RuntimeError(f"capture protocol mismatch for {source}")
        capture_config = yaml.safe_load(capture_protocol.read_text(encoding="utf-8"))
        angle_exception = bool(
            capture_config["sampling"]["physical_controller"]
            ["capture_angle_ceiling_limited_samples"]
        )
        session_count = 0
        for session_root in sorted((capture_root / "sessions").iterdir()):
            if not session_root.is_dir() or not (session_root / "summary.json").exists():
                continue
            session_summary = json.loads(
                (session_root / "summary.json").read_text(encoding="utf-8")
            )
            if session_summary.get("status") != "PASSED":
                raise RuntimeError(f"incomplete source session {session_root.name}")
            session_count += 1
            actors = read_jsonl(session_root / "public" / "frames.jsonl")
            actions = read_jsonl(session_root / "privileged" / "actions.jsonl")
            actions_by_frame = action_groups(actions)
            for actor in actors:
                assert_no_privileged_fields(actor)
                if len(actions_by_frame.get(int(actor["frame_sequence"]), ())) != int(
                    actor["action_steps_since_previous_frame"]
                ):
                    raise RuntimeError("actor/action step count mismatch")

            for indexes in trajectory_groups(actors):
                actor_rows = [actors[index] for index in indexes]
                probability = []
                quality = []
                score_rows = []
                for actor in actor_rows:
                    identity = (
                        source,
                        session_root.name,
                        int(actor["frame_sequence"]),
                    )
                    score = scores.get(identity)
                    if score is None or score["frame_sha256"] != actor["frame_sha256"]:
                        raise RuntimeError(f"missing or mismatched frozen score {identity}")
                    probability.append(float(score["validity_probability"]))
                    quality.append(float(score["quality_prediction"]))
                    score_rows.append(score)
                base_observations = reconstruct_trajectory_observations(
                    actor_rows, probability, quality
                )
                contract_observations = {
                    contract.value: observations_with_contract(
                        base_observations,
                        actor_rows,
                        actions_by_frame,
                        contract=contract,
                        angle_ceiling_exception_declared=angle_exception,
                    )
                    for contract in MovementContract
                }
                trajectory_row: dict[str, Any] = {
                    "source": source,
                    "session_id": session_root.name,
                    "trajectory_id": actor_rows[0]["trajectory_id"],
                    "frame_count": len(actor_rows),
                    "collector_accepted_count": sum(
                        int(actor_collector_accepted(row)) for row in actor_rows
                    ),
                    "collector_classifier_count": sum(
                        int(
                            actor_collector_accepted(row)
                            and float(score["validity_probability"])
                            >= decision_config.validity_probability_threshold
                        )
                        for row, score in zip(actor_rows, score_rows)
                    ),
                    "raw_distinct_hash_count": len(
                        {str(row["frame_sha256"]) for row in actor_rows}
                    ),
                    "producer_advanced_count": sum(
                        int(row.producer_advanced) for row in base_observations
                    ),
                }
                for contract, observations in contract_observations.items():
                    profile = candidate_support_profile(observations, decision_config)
                    for name, value in asdict(profile).items():
                        trajectory_row[f"{contract}_{name}"] = value
                    for policy in POLICIES:
                        decision = replay_decision(
                            observations, policy=policy, config=decision_config
                        )
                        prefix = f"{contract}_{policy}"
                        trajectory_row[f"{prefix}_outcome"] = decision.state.value
                        trajectory_row[f"{prefix}_selected_frame_sequence"] = (
                            None if decision.selected is None else decision.selected.frame_sequence
                        )
                        selected_score = (
                            None
                            if decision.selected is None
                            else scores[
                                (
                                    source,
                                    session_root.name,
                                    decision.selected.frame_sequence,
                                )
                            ]
                        )
                        trajectory_row[f"{prefix}_selected_image_validity"] = (
                            None
                            if selected_score is None
                            else selected_score["image_validity_target"]
                        )
                        if contract == MovementContract.ORIGINAL_FROZEN.value:
                            previous = prior_lookup[(source, actor_rows[0]["trajectory_id"], policy)]
                            mismatch = bool(
                                previous["outcome"] != decision.state.value
                                or previous["selected_frame_sequence"]
                                != trajectory_row[f"{prefix}_selected_frame_sequence"]
                            )
                            original_reproduction_mismatch_count += int(mismatch)
                original_profile = candidate_support_profile(
                    contract_observations[MovementContract.ORIGINAL_FROZEN.value],
                    decision_config,
                )
                diagnostic_profile = candidate_support_profile(
                    contract_observations[
                        MovementContract.CLAMP_CONSISTENT_DIAGNOSTIC.value
                    ],
                    decision_config,
                )
                original_valid_by_sequence = {
                    row.frame_sequence: row.transition_valid
                    for row in contract_observations[MovementContract.ORIGINAL_FROZEN.value]
                }
                diagnostic_valid_by_sequence = {
                    row.frame_sequence: row.transition_valid
                    for row in contract_observations[
                        MovementContract.CLAMP_CONSISTENT_DIAGNOSTIC.value
                    ]
                }
                trajectory_row["clamp_only_qualifying_evidence_count"] = sum(
                    int(
                        sequence in diagnostic_profile.qualifying_frame_sequences
                        and diagnostic_valid_by_sequence[sequence]
                        and not original_valid_by_sequence[sequence]
                    )
                    for sequence in diagnostic_valid_by_sequence
                )
                trajectory_row["original_qualifying_evidence_count"] = len(
                    original_profile.qualifying_frame_sequences
                )
                trajectory_rows.append(trajectory_row)

                previous_actor: Mapping[str, Any] | None = None
                for actor, score in zip(actor_rows, score_rows):
                    sequence = int(actor["frame_sequence"])
                    action_rows = actions_by_frame.get(sequence, [])
                    nonadvancing = bool(
                        previous_actor is not None
                        and int(actor["producer_sequence"])
                        == int(previous_actor["producer_sequence"])
                    )
                    zero_movement = zero_executed_movement(actor)
                    frame_rows.append(
                        {
                            "source": source,
                            "session_id": session_root.name,
                            "trajectory_id": actor["trajectory_id"],
                            "frame_sequence": sequence,
                            "frame_sha256": actor["frame_sha256"],
                            "clamp_category": clamp_category(actor),
                            "unaccepted_or_incomplete": not bool(
                                actor["action_accepted"] and actor["action_completed"]
                            ),
                            "accepted_angle_only_rejected_original": bool(
                                clamp_category(actor) == "angle_only"
                                and actor_collector_accepted(actor)
                                and not original_transition_valid(actor)
                                and diagnostic_valid_by_sequence[sequence]
                            ),
                            "zero_executed_movement": zero_movement,
                            "nonadvancing_producer": nonadvancing,
                            "nonadvancing_zero_action": bool(
                                nonadvancing
                                and int(actor["action_steps_since_previous_frame"]) == 0
                            ),
                            "nonadvancing_identical_previous_hash": bool(
                                nonadvancing
                                and previous_actor is not None
                                and actor["frame_sha256"] == previous_actor["frame_sha256"]
                            ),
                            "nonadvancing_nonzero_movement": bool(
                                nonadvancing and not zero_movement
                            ),
                            "fresh_same_pose": bool(
                                previous_actor is not None
                                and int(actor["producer_sequence"])
                                > int(previous_actor["producer_sequence"])
                                and zero_movement
                            ),
                            "fresh_same_pose_distinct_hash": bool(
                                previous_actor is not None
                                and int(actor["producer_sequence"])
                                > int(previous_actor["producer_sequence"])
                                and zero_movement
                                and actor["frame_sha256"] != previous_actor["frame_sha256"]
                            ),
                            "ceiling_limited_capture_flag": any(
                                bool(row.get("ceiling_limited_capture", False))
                                for row in action_rows
                            ),
                            "validity_probability": score["validity_probability"],
                            "image_validity_target": score["image_validity_target"],
                        }
                    )
                    previous_actor = actor
        dataset_audit.append(
            {
                "source": source,
                "session_count": session_count,
                "capture_summary_sha256": dataset["capture_summary_sha256"],
                "capture_protocol_sha256": dataset["capture_protocol_sha256"],
                "angle_ceiling_exception_declared": angle_exception,
            }
        )

    if len(scores) != len(frame_rows):
        raise RuntimeError("not every frozen score was consumed")
    if original_reproduction_mismatch_count:
        raise RuntimeError("original frozen replay was not exactly reproduced")

    source_names = sorted({row["source"] for row in trajectory_rows})
    session_names = sorted({row["session_id"] for row in trajectory_rows})
    scope_rows = {
        "overall": trajectory_rows,
        **{
            f"source:{source}": [
                row for row in trajectory_rows if row["source"] == source
            ]
            for source in source_names
        },
        **{
            f"session:{session}": [
                row for row in trajectory_rows if row["session_id"] == session
            ]
            for session in session_names
        },
    }
    waterfalls = {
        contract: waterfall(trajectory_rows, contract=contract)
        for contract in CONTRACTS
    }
    source_waterfalls = {
        source: {
            contract: waterfall(scope_rows[f"source:{source}"], contract=contract)
            for contract in CONTRACTS
        }
        for source in source_names
    }
    nonexclusive = {
        contract: nonexclusive_failures(trajectory_rows, contract=contract)
        for contract in CONTRACTS
    }
    metrics = {
        contract: {
            policy: aggregate_decisions(
                trajectory_rows, contract=contract, policy=policy
            )
            for policy in POLICIES
        }
        for contract in CONTRACTS
    }
    source_metrics = {
        source: {
            contract: {
                policy: aggregate_decisions(
                    scope_rows[f"source:{source}"], contract=contract, policy=policy
                )
                for policy in POLICIES
            }
            for contract in CONTRACTS
        }
        for source in source_names
    }
    session_metrics = {
        session: {
            contract: {
                policy: aggregate_decisions(
                    scope_rows[f"session:{session}"], contract=contract, policy=policy
                )
                for policy in POLICIES
            }
            for contract in CONTRACTS
        }
        for session in session_names
    }
    frame_aggregates = {
        "overall": aggregate_frame_rows(frame_rows),
        "by_source": {
            source: aggregate_frame_rows(
                [row for row in frame_rows if row["source"] == source]
            )
            for source in source_names
        },
        "by_session": {
            session: aggregate_frame_rows(
                [row for row in frame_rows if row["session_id"] == session]
            )
            for session in session_names
        },
    }
    rescued_trajectories = sum(
        int(row["clamp_only_qualifying_evidence_count"] > 0)
        for row in trajectory_rows
    )
    diagnostic_multi = metrics[MovementContract.CLAMP_CONSISTENT_DIAGNOSTIC.value][
        "candidate_d_multi_view"
    ]
    material = bool(
        diagnostic_multi["accepted_count"]
        >= int(protocol["interpretation_gate"]["minimum_material_accept_count"])
        and sum(
            int(
                source_metrics[source][
                    MovementContract.CLAMP_CONSISTENT_DIAGNOSTIC.value
                ]["candidate_d_multi_view"]["accepted_count"]
                > 0
            )
            for source in source_names
        )
        >= int(protocol["interpretation_gate"]["minimum_sources_with_acceptance"])
    )
    report = {
        "schema": "sonogym-stage2-contract-audit-report-v1",
        "decision": "OPTION_1_CONTRACT_CORRECTION_FIRST"
        if material
        else "OPTION_2_ACTIVE_ACQUISITION_PILOT",
        "protocol_sha256": protocol_hash,
        "source_commit": protocol["source_commit"],
        "prior_replay_reproduced": True,
        "original_reproduction_mismatch_count": original_reproduction_mismatch_count,
        "dataset_audit": dataset_audit,
        "frame_audit": frame_aggregates,
        "waterfalls": waterfalls,
        "source_waterfalls": source_waterfalls,
        "nonexclusive_failure_counts": nonexclusive,
        "decision_metrics": metrics,
        "source_decision_metrics": source_metrics,
        "session_decision_metrics": session_metrics,
        "clamp_only_potentially_useful_trajectory_count": rescued_trajectories,
        "trajectory_length_distribution": dict(
            Counter(str(row["frame_count"]) for row in trajectory_rows)
        ),
        "interpretation_gate": {
            **protocol["interpretation_gate"],
            "materially_restored": material,
        },
        "preserved_results": {
            "temporal_replay": "EXPLORATORY_NO_GO",
            "V16": "CONFIRMATION_NO_GO",
            "V17": "CONFIRMATION_NO_GO",
            "V18": "CONFIRMATION_NO_GO",
            "FULL_STAGE2_CLOSED_LOOP": "NOT_RUN",
        },
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "contract_audit_report.json"
    trajectory_path_out = output / "trajectory_contract_diagnostics.jsonl"
    frame_path_out = output / "frame_clamp_audit.csv"
    write_json(report_path, report)
    write_jsonl(trajectory_path_out, trajectory_rows)
    with frame_path_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    plot_waterfall(report, output / "abstention_waterfall.png")
    for path in (report_path, trajectory_path_out, frame_path_out):
        seal(path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
