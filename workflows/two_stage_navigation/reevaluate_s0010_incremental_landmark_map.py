#!/usr/bin/env python3
"""Reevaluate opened Stage 2 actor records with the shared 3D landmark map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from two_stage_navigation.incremental_landmark_map import (
    IncrementalAnatomicalLandmarkMap,
    IncrementalLandmarkMapConfig,
)
from two_stage_navigation.sonoanatomy_multiview import SemanticObservation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configuration(protocol: dict[str, Any]) -> IncrementalLandmarkMapConfig:
    return IncrementalLandmarkMapConfig(**protocol["landmark_map"])


def observation_rows(
    actor: dict[str, Any], *, selected_trajectory_only: bool
) -> list[tuple[int, str, str, SemanticObservation]]:
    rows = []
    for target in actor["targets"]:
        level = str(target["stage1_level"])
        sonoanatomy = target.get("sonoanatomy", {})
        selected_trajectory = sonoanatomy.get("selected_trajectory_id")
        for trajectory in sonoanatomy.get("trajectory_decisions", []):
            trajectory_id = str(trajectory["trajectory_id"])
            if selected_trajectory_only and trajectory_id != selected_trajectory:
                continue
            for value in trajectory["semantic_observations"]:
                observation = SemanticObservation(**value)
                rows.append(
                    (observation.frame_sequence, level, trajectory_id, observation)
                )
    return sorted(rows, key=lambda row: (row[0], row[1], row[2]))


def evaluator_summary(session_dir: Path) -> dict[str, Any] | None:
    candidates = (
        session_dir / "results/evaluation_report.json",
        session_dir / "results/evaluation_report_inf_fix_development.json",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return None
    report = json.loads(path.read_text())
    return {
        "path": str(path),
        "sha256": sha256(path),
        "milestone_verdict": report["milestone_verdict"],
        "targets": [
            {
                "target_name": row["target_name"],
                "selected_pair_correct": row.get("semantic_pair_identity", {}).get(
                    "pair_correct", False
                ),
                "localization_error_mm": row.get("localization_error_mm"),
                "same_frame_semantic_pair_success": row.get("semantic_pair_success"),
            }
            for row in report["target_reports"]
        ],
    }


def assess_session(
    session_dir: Path,
    config: IncrementalLandmarkMapConfig,
    *,
    selected_trajectory_only: bool,
) -> dict[str, Any]:
    actor_path = session_dir / "results/actor_result.json"
    actor = json.loads(actor_path.read_text())
    mapping = IncrementalAnatomicalLandmarkMap(config)
    timeline = []
    previous_states = None
    rows = observation_rows(
        actor, selected_trajectory_only=selected_trajectory_only
    )
    for sequence, level, trajectory_id, observation in rows:
        mapping.add_observations(level, (observation,))
        snapshot = mapping.snapshot()
        states = tuple(
            value["status"] for value in snapshot["landmarks"].values()
        )
        if states != previous_states:
            timeline.append(
                {
                    "frame_sequence": sequence,
                    "trajectory_id": trajectory_id,
                    "landmark_statuses": {
                        name: value["status"]
                        for name, value in snapshot["landmarks"].items()
                    },
                }
            )
            previous_states = states
    final = mapping.snapshot()
    return {
        "session_id": actor["session_id"],
        "actor_result_path": str(actor_path),
        "actor_result_sha256": sha256(actor_path),
        "historical_actor_outcome": actor["outcome"],
        "observation_count": len(rows),
        "incremental_map": final,
        "status_transition_timeline": timeline,
        "evaluator": evaluator_summary(session_dir),
    }


def main() -> int:
    args = parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol.get("protocol_id") != "sonogym-s0010-incremental-landmark-map-v1":
        raise RuntimeError("unexpected protocol")
    config = configuration(protocol)
    observation_scope = protocol.get("observation_scope")
    if observation_scope != "selected_actor_motion_trajectory":
        raise RuntimeError("unsupported observation scope")
    session_dirs = sorted(
        path.parent.parent
        for path in args.run_root.glob("*/results/actor_result.json")
    )
    if not session_dirs:
        raise RuntimeError("no sealed actor results found")
    sessions = [
        assess_session(path, config, selected_trajectory_only=True)
        for path in session_dirs
    ]
    report = {
        "schema": "sonogym-s0010-incremental-landmark-map-development-v1",
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": sha256(args.protocol),
        "source_run_root": str(args.run_root.resolve()),
        "source_job": int(protocol["source"]["opened_stage2_job"]),
        "observation_scope": observation_scope,
        "development_only": True,
        "independent_confirmation": False,
        "session_count": len(sessions),
        "complete_map_session_count": sum(
            bool(row["incremental_map"]["complete"]) for row in sessions
        ),
        "sessions": sessions,
        "verdict": (
            "DEVELOPMENT_MAP_COMPLETE"
            if all(row["incremental_map"]["complete"] for row in sessions)
            else "ACTIVE_RESCAN_REQUIRED"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    checksum = sha256(args.output)
    args.output.with_suffix(".json.sha256").write_text(
        f"{checksum}  {args.output.name}\n"
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "complete_map_sessions": report["complete_map_session_count"],
                "session_count": report["session_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
