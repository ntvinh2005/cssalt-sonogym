#!/usr/bin/env python3
"""Freeze an L4 capture around opened, physically realized SAP/TP views."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import prepare_s0010_sap_tp_6dof_confirmation as previous
from two_stage_navigation.acquisition_integrity import validate_waypoint_bank


PROTOCOL_ID = "sonogym-s0010-l4-landmark-focused-development-v1r1"
LABEL_RESOLUTION_MM = 1.5


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def focused_pose_candidates(centers: list[list[float]]) -> list[list[float]]:
    """Generate deterministic non-duplicated command poses around opened centers."""

    offsets = sorted(
        (
            (dx, dz, da)
            for dx in (-4.0, -2.0, 0.0, 2.0, 4.0)
            for dz in (-2.0, 0.0, 2.0)
            for da in (-0.10, -0.05, 0.0, 0.05, 0.10)
            if (dx, dz, da) != (0.0, 0.0, 0.0)
        ),
        key=lambda value: (
            abs(value[0]) + abs(value[1]) + 20.0 * abs(value[2]),
            value,
        ),
    )
    candidates: list[list[float]] = []
    seen: set[tuple[float, float, float]] = set()
    for offset_index in range(len(offsets)):
        for center_index, center in enumerate(centers):
            dx, dz, da = offsets[(offset_index + center_index) % len(offsets)]
            pose = [center[0] + dx, center[1] + dz, center[2] + da]
            key = tuple(round(value, 6) for value in pose)
            if key in seen or not (1.0 <= pose[2] <= 3.0):
                continue
            seen.add(key)
            candidates.append(pose)
    return candidates


def separated_sequence(
    candidates: list[list[float]], count: int, *, start_index: int
) -> list[list[float]]:
    """Choose a sequence whose adjacent commands exceed tracking tolerances."""

    remaining = candidates[start_index:] + candidates[:start_index]
    selected: list[list[float]] = []
    while remaining and len(selected) < count:
        next_index = next(
            (
                index
                for index, pose in enumerate(remaining)
                if not selected
                or np.linalg.norm(
                    np.asarray(pose[:2]) - np.asarray(selected[-1][:2])
                )
                * LABEL_RESOLUTION_MM
                > 4.0
                or abs(
                    float(
                        (pose[2] - selected[-1][2] + np.pi) % (2 * np.pi) - np.pi
                    )
                )
                * 180.0
                / np.pi
                > 2.4
            ),
            None,
        )
        if next_index is None:
            break
        selected.append(remaining.pop(next_index))
    if len(selected) != count:
        raise ValueError(f"could not choose {count} separated focused poses")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base = yaml.safe_load(args.base_protocol.read_text(encoding="utf-8"))
    focus_keys = {
        ("s0010l4graphdev_01", 7),
        ("s0010l4graphdev_03", 4),
        ("s0010l4graphdev_03", 5),
    }
    centers = []
    for session_id, frame_sequence in sorted(focus_keys):
        rows = read_jsonl(
            args.source_run / "sessions" / session_id / "privileged/annotations.jsonl"
        )
        row = next(item for item in rows if item["frame_sequence"] == frame_sequence)
        centers.append([float(value) for value in row["requested_waypoint_pose_index"]])
    candidates = focused_pose_candidates(centers)

    controls: list[list[float]] = []
    for bank_id, bank in base["sampling"]["absolute_waypoint_banks_index"].items():
        evidence = base["sampling"]["planned_waypoint_evidence_by_bank"][bank_id]
        controls.extend(
            [float(value) for value in pose]
            for pose, role in zip(bank, evidence)
            if role["declared_role"] == "CLEAR_NEGATIVE"
        )
    if len(controls) < 16:
        raise ValueError("base protocol lacks sixteen clear controls")

    protocol = deepcopy(base)
    protocol["protocol_id"] = PROTOCOL_ID
    protocol["status"] = "FROZEN_OPENED_FOCUSED_DEVELOPMENT_CAPTURE"
    protocol["purpose"] = (
        "Collect L4 SAP/TP image evidence around three opened, evaluator-confirmed "
        "surface views; never use the absolute bank at runtime."
    )
    protocol["source"] = {
        "implementation_commit": args.implementation_commit,
        "base_protocol_sha256": previous.sha256(args.base_protocol),
        "source_run": str(args.source_run.resolve()),
        "source_focus_frames": [f"{session}:{frame}" for session, frame in sorted(focus_keys)],
        "evidence_status": "privileged_opened_development_capture_design",
    }
    sampling = protocol["sampling"]
    sampling["absolute_waypoint_banks_index"] = {}
    sampling["planned_waypoint_evidence_by_bank"] = {}
    sampling["trajectory_start_frames"] = list(range(0, 14, 2))
    sampling["maximum_initial_positioning_steps"] = 700
    sampling["maximum_action_steps_per_waypoint"] = 200
    sampling["episode_length_s_override"] = 500.0
    sessions = []
    used_targets: set[tuple[float, float, float]] = set()
    for session_index in range(4):
        available = [
            pose
            for pose in candidates
            if tuple(round(value, 6) for value in pose) not in used_targets
        ]
        targets = separated_sequence(
            available, 10, start_index=(session_index * 7) % len(available)
        )
        used_targets.update(tuple(round(value, 6) for value in pose) for pose in targets)
        negatives = controls[session_index * 4 : (session_index + 1) * 4]
        rows = targets + negatives
        validate_waypoint_bank(
            rows,
            command_minimum=[-1.0e9] * 3,
            command_maximum=[1.0e9] * 3,
            label_resolution_mm=LABEL_RESOLUTION_MM,
            position_tolerance_mm=2.0,
            orientation_tolerance_deg=1.2,
            separation_multiplier=2.0,
        )
        bank_id = f"bank_{session_index:02d}"
        sampling["absolute_waypoint_banks_index"][bank_id] = rows
        sampling["planned_waypoint_evidence_by_bank"][bank_id] = [
            {
                "declared_role": (
                    "L4_FOCUSED_TARGET_CANDIDATE" if index < 10 else "CLEAR_NEGATIVE"
                ),
                "source": "opened_development_focus_neighborhood",
                "waypoint_index": index,
            }
            for index in range(14)
        ]
        session_id = f"s0010l4focusdev_{session_index:02d}"
        sessions.append(
            {
                "session_id": session_id,
                "split": "opened_l4_landmark_focused_development",
                "seed": 2026120400 + session_index,
                "reset_id": session_id,
                "absolute_waypoint_bank_id": bank_id,
                "expected_frame_count": 14,
            }
        )
    protocol["sessions"] = sessions
    protocol["change_policy"] = [
        "These frames are opened development evidence, never independent confirmation.",
        "The focus centers and absolute waypoints are privileged capture-design inputs.",
        "Runtime landmark inference may use only clean ultrasound and public robot pose.",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = args.output_dir / "s0010_l4_landmark_focused_development_v1r1.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    protocol_path.with_suffix(".sha256").write_text(
        f"{previous.sha256(protocol_path)}  {protocol_path.name}\n", encoding="utf-8"
    )
    manifest = {
        "protocol": str(protocol_path),
        "protocol_sha256": previous.sha256(protocol_path),
        "session_count": 4,
        "focused_target_pose_count": 40,
        "clear_control_pose_count": 16,
        "focus_center_count": len(centers),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
