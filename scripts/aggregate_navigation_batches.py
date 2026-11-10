#!/usr/bin/env python3
"""Aggregate process-isolated Phase 5A batches and verify paired replay."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def angle_delta_deg(left: float, right: float) -> float:
    return float(abs((left - right + np.pi) % (2.0 * np.pi) - np.pi) * 180.0 / np.pi)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--execution-protocol", required=True)
    parser.add_argument("--batches-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol).resolve()
    execution_protocol_path = Path(args.execution_protocol).resolve()
    batches_dir = Path(args.batches_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = YAML(typ="safe").load(protocol_path.read_text(encoding="utf-8"))
    execution_protocol = YAML(typ="safe").load(
        execution_protocol_path.read_text(encoding="utf-8")
    )
    expected_base_hash = execution_protocol["base_protocol"]["sha256"]
    if sha256_file(protocol_path) != expected_base_hash:
        raise RuntimeError("base protocol hash does not match protocol-v2")

    controllers = protocol["controllers"]["order"]
    num_envs = int(protocol["execution"]["num_parallel_envs"])
    batch_count = int(protocol["execution"]["batches"])
    starts_count = int(protocol["initial_conditions"]["count"])
    expected_start_ids = [f"start_{index:03d}" for index in range(starts_count)]
    controller_reports: dict[str, dict] = {}
    initial_states: dict[str, dict[str, dict]] = {}
    batch_report_paths: list[str] = []
    environment = None
    manifest = None
    checkpoint = None
    all_finite = True
    step_count = 0
    trajectory_path = output_dir / "per_step.jsonl.gz"

    with gzip.open(trajectory_path, "wt", encoding="utf-8") as merged_steps:
        for controller in controllers:
            episodes: list[dict] = []
            controller_initials: dict[str, dict] = {}
            elapsed_seconds = 0.0
            for batch_index in range(batch_count):
                batch_dir = batches_dir / controller / f"batch_{batch_index:02d}"
                report_path = batch_dir / "evaluation.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                batch_report_paths.append(str(report_path))
                if report["status"] != "PASSED":
                    raise RuntimeError(f"batch report failed: {report_path}")
                execution = report["execution"]
                if execution["controllers"] != [controller]:
                    raise RuntimeError(f"wrong controller in {report_path}")
                if execution["batch_index"] != batch_index:
                    raise RuntimeError(f"wrong batch index in {report_path}")
                if not execution["process_isolated_batch"]:
                    raise RuntimeError(f"batch was not process isolated: {report_path}")
                batch_episodes = report["controllers"][controller]["episodes"]
                expected_batch_ids = expected_start_ids[
                    batch_index * num_envs : (batch_index + 1) * num_envs
                ]
                actual_batch_ids = [item["start_id"] for item in batch_episodes]
                if actual_batch_ids != expected_batch_ids:
                    raise RuntimeError(f"wrong or reordered starts in {report_path}")
                if sorted(report["initial_states"]) != sorted(expected_batch_ids):
                    raise RuntimeError(f"incomplete initial states in {report_path}")
                episodes.extend(batch_episodes)
                controller_initials.update(report["initial_states"])
                elapsed_seconds += float(execution["elapsed_seconds"])
                all_finite = (
                    all_finite
                    and bool(report["all_observations_rewards_poses_finite"])
                )
                if environment is None:
                    environment = report["environment"]
                    manifest = report["manifest"]
                    checkpoint = report["checkpoint"]
                elif (
                    report["environment"] != environment
                    or report["manifest"] != manifest
                    or report["checkpoint"] != checkpoint
                ):
                    raise RuntimeError("batch environment or frozen artifact identity changed")

                with gzip.open(batch_dir / "per_step.jsonl.gz", "rt", encoding="utf-8") as source:
                    for line in source:
                        record = json.loads(line)
                        if (
                            record["controller"] != controller
                            or record["start_id"] not in expected_batch_ids
                        ):
                            raise RuntimeError(f"invalid step provenance in {batch_dir}")
                        merged_steps.write(
                            json.dumps(record, allow_nan=False, sort_keys=True) + "\n"
                        )
                        step_count += 1
            if [item["start_id"] for item in episodes] != expected_start_ids:
                raise RuntimeError(f"controller {controller} is not a complete ordered replay")
            controller_reports[controller] = {
                "seed": int(protocol["seeds"][controller]),
                "elapsed_seconds": elapsed_seconds,
                "episodes": episodes,
            }
            initial_states[controller] = controller_initials

    all_episodes = [
        episode
        for report in controller_reports.values()
        for episode in report["episodes"]
    ]
    expected_steps = sum(item["episode_length"] for item in all_episodes)
    if step_count != expected_steps:
        raise RuntimeError(
            f"per-step records {step_count} do not match episode steps {expected_steps}"
        )

    oracle_initials = initial_states["oracle"]
    label_resolution_m = float(environment["label_resolution_m"])
    pairing_deltas: list[dict] = []
    for controller in controllers:
        if controller == "oracle":
            continue
        for start_id in expected_start_ids:
            oracle_state = oracle_initials[start_id]
            current_state = initial_states[controller][start_id]
            oracle_pose = np.asarray(oracle_state["realized_pose"], dtype=np.float64)
            current_pose = np.asarray(current_state["realized_pose"], dtype=np.float64)
            oracle_goal = np.asarray(oracle_state["goal_pose"], dtype=np.float64)
            current_goal = np.asarray(current_state["goal_pose"], dtype=np.float64)
            pairing_deltas.append(
                {
                    "controller": controller,
                    "start_id": start_id,
                    "realized_position_delta_mm_from_oracle_reset": float(
                        np.linalg.norm(current_pose[:2] - oracle_pose[:2])
                        * label_resolution_m
                        * 1000.0
                    ),
                    "realized_orientation_delta_deg_from_oracle_reset": angle_delta_deg(
                        current_pose[2], oracle_pose[2]
                    ),
                    "goal_pose_max_abs_delta": float(
                        np.max(np.abs(current_goal - oracle_goal))
                    ),
                }
            )

    tolerance = protocol["initial_conditions"]["realized_pairing_tolerance"]
    max_position_delta = max(
        item["realized_position_delta_mm_from_oracle_reset"]
        for item in pairing_deltas
    )
    max_angle_delta = max(
        item["realized_orientation_delta_deg_from_oracle_reset"]
        for item in pairing_deltas
    )
    max_goal_delta = max(item["goal_pose_max_abs_delta"] for item in pairing_deltas)
    replay_verified = (
        max_position_delta <= float(tolerance["position_mm"])
        and max_angle_delta <= float(tolerance["orientation_deg"])
        and max_goal_delta <= float(tolerance["goal_pose_abs"])
    )
    finite_and_valid = all_finite and all(
        item["nonfinite_observations"] == 0 and item["invalid_transitions"] == 0
        for item in all_episodes
    )
    report = {
        "status": "PASSED" if replay_verified and finite_and_valid else "FAILED",
        "protocol_id": execution_protocol["protocol_id"],
        "base_protocol_id": protocol["protocol_id"],
        "protocol_v2": {
            "path": str(execution_protocol_path),
            "sha256": sha256_file(execution_protocol_path),
            "base_protocol_sha256": expected_base_hash,
        },
        "task": protocol["task"]["gym_id"],
        "manifest": manifest,
        "checkpoint": checkpoint,
        "environment": environment,
        "execution": {
            "num_starts": starts_count,
            "num_parallel_envs": num_envs,
            "max_action_steps": int(protocol["execution"]["max_action_steps"]),
            "controllers": controllers,
            "process_isolated_batches": batch_count * len(controllers),
            "fresh_isaac_process_per_four_start_batch": True,
            "inter_batch_reset_method": "none_fresh_process_per_batch",
            "transitions": expected_steps,
            "per_step_records": step_count,
            "elapsed_seconds_sum_across_processes": sum(
                item["elapsed_seconds"] for item in controller_reports.values()
            ),
        },
        "paired_replay": {
            "verified": replay_verified,
            "verification_scope": "cross_process_protocol_v2_aggregation",
            "max_realized_position_delta_mm": max_position_delta,
            "max_realized_orientation_delta_deg": max_angle_delta,
            "max_goal_pose_abs_delta": max_goal_delta,
            "per_start_deltas": pairing_deltas,
        },
        "all_observations_rewards_poses_finite": finite_and_valid,
        "controllers": controller_reports,
        "per_step_path": str(trajectory_path),
        "batch_report_paths": batch_report_paths,
    }
    report_path = output_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "process_isolated_batches": report["execution"][
                    "process_isolated_batches"
                ],
                "transitions": expected_steps,
                "paired_replay": report["paired_replay"],
                "output": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
