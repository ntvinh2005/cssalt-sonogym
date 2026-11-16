#!/usr/bin/env python3
"""Replay Stage 2 with the explicitly selected movement contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.contracts import assert_no_privileged_fields  # noqa: E402
from two_stage_navigation.stage2_temporal_decision import (  # noqa: E402
    DecisionState,
    MovementContract,
    TemporalDecisionConfig,
    reconstruct_trajectory_observations,
    replay_decision,
)


POLICIES = ("candidate_c_temporal", "candidate_d_multi_view")
ACTION_FEEDBACK_FIELDS = (
    "finite",
    "surface_proxy_ok",
    "command_mapping_valid",
    "terminated",
    "truncated",
    "command_position_clamped",
    "command_angle_clamped",
)


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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def seal(path: Path) -> str:
    digest = file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def load_scores(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["source"], row["session_id"], int(row["frame_sequence"]))
            if key in output:
                raise RuntimeError(f"duplicate frozen score {key}")
            output[key] = {
                "frame_sha256": row["frame_sha256"],
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
        sequence = int(row["frame_sequence"])
        feedback = {field: row[field] for field in ACTION_FEEDBACK_FIELDS}
        assert_no_privileged_fields(feedback)
        grouped[sequence].append(feedback)
    return grouped


def aggregate(rows: Sequence[Mapping[str, Any]], policy: str) -> dict[str, Any]:
    selected = [row for row in rows if row["policy"] == policy]
    accepted = [row for row in selected if row["outcome"] == DecisionState.ACCEPT.value]
    false_accepts = sum(int(row["selected_image_validity"] == 0) for row in accepted)
    return {
        "trajectory_count": len(selected),
        "accepted_trajectory_count": len(accepted),
        "abstention_count": len(selected) - len(accepted),
        "false_accepted_candidate_count": false_accepts,
        "acceptance_coverage": len(accepted) / len(selected),
        "accepted_candidate_precision": (
            None if not accepted else (len(accepted) - false_accepts) / len(accepted)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()

    protocol_hash = file_sha256(args.protocol)
    expected_hash = args.protocol.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if protocol_hash != expected_hash:
        raise RuntimeError("protocol seal mismatch")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "sonogym-stage2-contract-corrected-replay-v1":
        raise RuntimeError("unexpected protocol")

    score_path = Path(protocol["frozen_inputs"]["frame_scores"])
    if file_sha256(score_path) != protocol["frozen_inputs"]["frame_scores_sha256"]:
        raise RuntimeError("frozen frame score mismatch")
    scores = load_scores(score_path)
    prior_path = Path(protocol["frozen_inputs"]["original_trajectory_results"])
    if file_sha256(prior_path) != protocol["frozen_inputs"]["original_trajectory_results_sha256"]:
        raise RuntimeError("original trajectory result mismatch")
    prior_rows = read_jsonl(prior_path)
    prior = {
        (row["source"], row["trajectory_id"], row["policy"]): row
        for row in prior_rows
        if row["policy"] in POLICIES
    }

    rule = protocol["decision_rules"]
    config = TemporalDecisionConfig(
        validity_probability_threshold=float(rule["validity_probability_threshold"]),
        maximum_neighbor_translation_mm=float(rule["maximum_neighbor_translation_mm"]),
        maximum_neighbor_angle_deg=float(rule["maximum_neighbor_angle_deg"]),
        maximum_quality_drop=float(rule["maximum_quality_drop"]),
        required_distinct_hash_observations=int(rule["required_distinct_hash_observations"]),
        minimum_distinct_view_translation_mm=float(rule["minimum_distinct_view_translation_mm"]),
        minimum_distinct_view_angle_deg=float(rule["minimum_distinct_view_angle_deg"]),
    )
    result_rows: list[dict[str, Any]] = []
    original_mismatch_count = 0
    consumed_scores = 0

    for dataset in protocol["datasets"]:
        source = str(dataset["name"])
        root = Path(dataset["root"])
        summary = root / "results" / "capture_summary.json"
        if file_sha256(summary) != dataset["capture_summary_sha256"]:
            raise RuntimeError(f"capture summary mismatch for {source}")
        source_protocol = REPOSITORY_ROOT / dataset["capture_protocol"]
        if file_sha256(source_protocol) != dataset["capture_protocol_sha256"]:
            raise RuntimeError(f"capture protocol mismatch for {source}")
        source_config = yaml.safe_load(source_protocol.read_text(encoding="utf-8"))
        angle_exception = bool(
            source_config["sampling"]["physical_controller"]
            ["capture_angle_ceiling_limited_samples"]
        )
        for session_root in sorted((root / "sessions").iterdir()):
            if not session_root.is_dir() or not (session_root / "summary.json").exists():
                continue
            session_summary = json.loads((session_root / "summary.json").read_text(encoding="utf-8"))
            if session_summary.get("status") != "PASSED":
                raise RuntimeError(f"incomplete session {session_root.name}")
            actor_rows = read_jsonl(session_root / "public" / "frames.jsonl")
            for row in actor_rows:
                assert_no_privileged_fields(row)
            actions = action_groups(
                read_jsonl(session_root / "privileged" / "actions.jsonl")
            )
            for indexes in trajectory_groups(actor_rows):
                records = [actor_rows[index] for index in indexes]
                probabilities: list[float] = []
                qualities: list[float] = []
                labels: dict[int, dict[str, int]] = {}
                for record in records:
                    identity = (source, session_root.name, int(record["frame_sequence"]))
                    score = scores.get(identity)
                    if score is None or score["frame_sha256"] != record["frame_sha256"]:
                        raise RuntimeError(f"missing or mismatched score {identity}")
                    probabilities.append(score["validity_probability"])
                    qualities.append(score["quality_prediction"])
                    labels[int(record["frame_sequence"])] = {
                        "image": score["image_validity_target"],
                        "geometric": score["geometric_validity_target"],
                    }
                    consumed_scores += 1
                original = reconstruct_trajectory_observations(records, probabilities, qualities)
                corrected = reconstruct_trajectory_observations(
                    records,
                    probabilities,
                    qualities,
                    actions_by_frame=actions,
                    movement_contract=MovementContract.CLAMP_CONSISTENT_DIAGNOSTIC,
                    angle_ceiling_exception_declared=angle_exception,
                )
                for policy in POLICIES:
                    original_decision = replay_decision(original, policy=policy, config=config)
                    previous = prior[(source, records[0]["trajectory_id"], policy)]
                    original_mismatch_count += int(
                        previous["outcome"] != original_decision.state.value
                        or previous["selected_frame_sequence"]
                        != (
                            None
                            if original_decision.selected is None
                            else original_decision.selected.frame_sequence
                        )
                    )
                    decision = replay_decision(corrected, policy=policy, config=config)
                    selected = decision.selected
                    selected_labels = None if selected is None else labels[selected.frame_sequence]
                    result_rows.append(
                        {
                            "source": source,
                            "session_id": session_root.name,
                            "trajectory_id": records[0]["trajectory_id"],
                            "policy": policy,
                            "outcome": decision.state.value,
                            "selected_frame_sequence": None if selected is None else selected.frame_sequence,
                            "selected_frame_sha256": None if selected is None else selected.frame_sha256,
                            "selected_image_validity": None if selected_labels is None else selected_labels["image"],
                            "selected_geometric_validity": None if selected_labels is None else selected_labels["geometric"],
                            "supporting_hash_count": 0 if decision.evidence is None else decision.evidence.distinct_hash_count,
                            "supporting_viewpoint_count": 0 if decision.evidence is None else decision.evidence.distinct_viewpoint_count,
                        }
                    )

    if original_mismatch_count:
        raise RuntimeError("original frozen replay was not reproduced")
    if consumed_scores != len(scores):
        raise RuntimeError("not every frozen score was consumed")

    policy_metrics = {policy: aggregate(result_rows, policy) for policy in POLICIES}
    source_metrics = {
        source: {
            policy: aggregate(
                [row for row in result_rows if row["source"] == source], policy
            )
            for policy in POLICIES
        }
        for source in sorted({str(row["source"]) for row in result_rows})
    }
    expected = protocol["expected_opened_data_reproduction"]
    reproduced = bool(
        policy_metrics["candidate_c_temporal"]["accepted_trajectory_count"]
        == int(expected["temporal_accepts"])
        and policy_metrics["candidate_d_multi_view"]["accepted_trajectory_count"]
        == int(expected["multi_view_accepts"])
        and policy_metrics["candidate_d_multi_view"]["false_accepted_candidate_count"]
        == int(expected["multi_view_false_accepts"])
    )
    report = {
        "schema": "sonogym-stage2-contract-corrected-replay-v1",
        "decision": "EXPLORATORY_NO_GO",
        "contract_correction_reproduced": reproduced,
        "protocol_sha256": protocol_hash,
        "source_commit": protocol["source_commit"],
        "frame_count": consumed_scores,
        "trajectory_count": len(result_rows) // len(POLICIES),
        "original_reproduction_mismatch_count": original_mismatch_count,
        "policy_metrics": policy_metrics,
        "source_metrics": source_metrics,
        "model_retrained": False,
        "decision_parameters_changed": False,
        "opened_data_only": True,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "preserved_results": protocol["preserved_results"],
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "trajectory_results.jsonl"
    report_path = output / "replay_report.json"
    write_jsonl(trajectory_path, result_rows)
    write_json(report_path, report)
    seal(trajectory_path)
    seal(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if reproduced else 3


if __name__ == "__main__":
    raise SystemExit(main())
