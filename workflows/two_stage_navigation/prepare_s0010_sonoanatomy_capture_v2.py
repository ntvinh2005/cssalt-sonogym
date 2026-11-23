#!/usr/bin/env python3
"""Prepare extra L4/L5 development views from observed SAP/TP and confuser poses."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from two_stage_navigation.acquisition_integrity import validate_waypoint_bank


PROTOCOL_ID = "sonogym-s0010-sonoanatomy-development-v2"
LABEL_RESOLUTION_MM = 1.5
POSE_MINIMUM = np.asarray([100.0, 70.0, 1.0], dtype=np.float64)
POSE_MAXIMUM = np.asarray([180.0, 200.0, 3.0], dtype=np.float64)
POSITION_SEPARATION_MM = 3.0
ORIENTATION_SEPARATION_DEG = 2.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def pose_is_valid(pose: np.ndarray) -> bool:
    return bool(
        pose.shape == (3,)
        and np.all(np.isfinite(pose))
        and np.all(pose >= POSE_MINIMUM)
        and np.all(pose <= POSE_MAXIMUM)
        and pose[2] < 2.99
    )


def poses_are_separated(first: np.ndarray, second: np.ndarray) -> bool:
    translation_mm = float(np.linalg.norm(first[:2] - second[:2]) * LABEL_RESOLUTION_MM)
    rotation_deg = math.degrees(abs(float(first[2] - second[2])))
    return bool(
        translation_mm > POSITION_SEPARATION_MM
        or rotation_deg > ORIENTATION_SEPARATION_DEG
    )


def local_sweep(anchor: np.ndarray) -> list[np.ndarray]:
    """Acquire distinct local translations and angles around one observed pose."""

    if anchor[2] > 2.84:
        offsets = (-0.15, -0.10, -0.05, 0.0)
    else:
        offsets = (-0.10, -0.05, 0.0, 0.05)
    x, z, angle = (float(value) for value in anchor)
    result = [
        np.asarray([x, z, angle + offsets[0]]),
        np.asarray([x, z, angle + offsets[1]]),
        np.asarray([x, z, angle + offsets[2]]),
        np.asarray([x, z, angle + offsets[3]]),
        np.asarray([x + 3.0, z, angle + offsets[3]]),
        np.asarray([x + 3.0, z + 3.0, angle + offsets[2]]),
        np.asarray([x, z + 3.0, angle + offsets[1]]),
        np.asarray([x - 3.0, z + 3.0, angle + offsets[2]]),
    ]
    if not all(pose_is_valid(pose) for pose in result):
        raise ValueError(f"local sweep leaves the command domain: {anchor.tolist()}")
    if not all(poses_are_separated(before, after) for before, after in zip(result, result[1:])):
        raise ValueError("local sweep contains consecutive views below the separation gate")
    return result


def empirical_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    annotations: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    result = []
    for row in manifest:
        session_key = (str(row["source_root"]), str(row["session_id"]))
        if session_key not in annotations:
            source = Path(session_key[0]) / "sessions" / session_key[1]
            annotations[session_key] = {
                str(item["acquisition_id"]): item
                for item in read_jsonl(source / "privileged/annotations.jsonl")
            }
        annotation = annotations[session_key].get(str(row["acquisition_id"]))
        if annotation is None:
            raise ValueError(f"missing annotation for {row['acquisition_id']}")
        pose_value = annotation.get("requested_waypoint_pose_index")
        if pose_value is None:
            pose_value = annotation.get("controller_target_pose_index")
        if pose_value is None:
            continue
        pose = np.asarray(pose_value, dtype=np.float64)
        if not pose_is_valid(pose):
            continue
        result.append(
            {
                "level": str(row["level"]),
                "source_role": str(row["source_role"]),
                "session_id": str(row["session_id"]),
                "frame_sequence": int(row["frame_sequence"]),
                "frame_sha256": str(row["frame_sha256"]),
                "pose": pose,
                "tp_pixels": int(row["tp_interface_pixels"]),
                "sap_pixels": int(row["sap_interface_pixels"]),
                "other_pixels": int(row["other_bone_interface_pixels"]),
                "has_tp": row["has_tp_interface"] == "True",
                "has_sap": row["has_sap_interface"] == "True",
            }
        )
    return result


def anchor_rank(row: dict[str, Any], *, positive: bool) -> tuple[Any, ...]:
    if positive:
        return (
            min(row["tp_pixels"], row["sap_pixels"]),
            row["tp_pixels"] + row["sap_pixels"],
            -row["other_pixels"],
        )
    return (
        row["other_pixels"] + row["tp_pixels"],
        row["other_pixels"],
        row["tp_pixels"],
    )


def sweeps_are_distinct(rows: list[dict[str, Any]]) -> bool:
    poses = [pose for row in rows for pose in row["sweep"]]
    return all(
        poses_are_separated(poses[first], poses[second])
        for first in range(len(poses))
        for second in range(first + 1, len(poses))
    )


def choose_level_anchors(
    rows: list[dict[str, Any]], level: str
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    level_rows = [row for row in rows if row["level"] == level]
    positives_by_pose: dict[tuple[float, float, float], dict[str, Any]] = {}
    negatives_by_pose: dict[tuple[float, float, float], dict[str, Any]] = {}
    for row in level_rows:
        key = tuple(float(value) for value in np.round(row["pose"], 6))
        if row["has_tp"] and row["has_sap"] and row["sap_pixels"] >= 3:
            if key not in positives_by_pose or anchor_rank(row, positive=True) > anchor_rank(
                positives_by_pose[key], positive=True
            ):
                positives_by_pose[key] = row
        if not row["has_sap"] and row["other_pixels"] >= 10:
            if key not in negatives_by_pose or anchor_rank(row, positive=False) > anchor_rank(
                negatives_by_pose[key], positive=False
            ):
                negatives_by_pose[key] = row
    positives = [
        {**row, "sweep": local_sweep(row["pose"])}
        for row in sorted(
            positives_by_pose.values(),
            key=lambda row: anchor_rank(row, positive=True),
            reverse=True,
        )
    ]
    negatives = [
        {**row, "sweep": local_sweep(row["pose"])}
        for row in sorted(
            negatives_by_pose.values(),
            key=lambda row: anchor_rank(row, positive=False),
            reverse=True,
        )
    ]
    selected: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    used_positive: set[tuple[float, float, float]] = set()
    used_negative: set[tuple[float, float, float]] = set()
    for _ in range(4):
        choice = None
        for first_index, first in enumerate(positives):
            first_key = tuple(float(value) for value in first["pose"])
            if first_key in used_positive:
                continue
            for second in positives[first_index + 1 :]:
                second_key = tuple(float(value) for value in second["pose"])
                if second_key in used_positive or not sweeps_are_distinct([first, second]):
                    continue

                def matched_negative_rank(row: dict[str, Any]) -> tuple[Any, ...]:
                    distance = min(
                        float(
                            np.linalg.norm(row["pose"][:2] - positive["pose"][:2])
                            * LABEL_RESOLUTION_MM
                        )
                        + 5.0
                        * math.degrees(
                            abs(float(row["pose"][2] - positive["pose"][2]))
                        )
                        for positive in (first, second)
                    )
                    return (-distance, *anchor_rank(row, positive=False))

                for negative in sorted(negatives, key=matched_negative_rank, reverse=True):
                    negative_key = tuple(float(value) for value in negative["pose"])
                    if negative_key in used_negative:
                        continue
                    if sweeps_are_distinct([first, second, negative]):
                        choice = (first, second, negative)
                        break
                if choice is not None:
                    break
            if choice is not None:
                break
        if choice is None:
            raise ValueError(f"{level} cannot form four separated session banks")
        selected.append(choice)
        used_positive.update(tuple(float(value) for value in row["pose"]) for row in choice[:2])
        used_negative.add(tuple(float(value) for value in choice[2]["pose"]))
    return selected


def evidence(anchor: dict[str, Any], role: str, offset: int) -> dict[str, Any]:
    return {
        "declared_role": role,
        "anchor_source_role": anchor["source_role"],
        "anchor_session_id": anchor["session_id"],
        "anchor_frame_sequence": anchor["frame_sequence"],
        "anchor_frame_sha256": anchor["frame_sha256"],
        "anchor_tp_interface_pixels": anchor["tp_pixels"],
        "anchor_sap_interface_pixels": anchor["sap_pixels"],
        "anchor_other_bone_interface_pixels": anchor["other_pixels"],
        "local_sweep_offset": offset,
    }


def main() -> int:
    args = parse_args()
    base = yaml.safe_load(args.base_protocol.read_text(encoding="utf-8"))
    rows = empirical_rows(args.dataset_manifest)
    protocol = deepcopy(base)
    protocol["protocol_id"] = PROTOCOL_ID
    protocol["status"] = "FROZEN_OPENED_SIMULATOR_SUPERVISED_DEVELOPMENT_CAPTURE"
    protocol["purpose"] = (
        "Acquire level-specific local views around previously observed TP/SAP interfaces "
        "and nearby other-bone examples; source anatomy remains simulator-proxy."
    )
    protocol["source"] = {
        "implementation_commit": args.implementation_commit,
        "base_protocol": str(args.base_protocol.resolve()),
        "base_protocol_sha256": sha256(args.base_protocol),
        "dataset_manifest": str(args.dataset_manifest.resolve()),
        "dataset_manifest_sha256": sha256(args.dataset_manifest),
        "evidence_status": "opened_simulator_proxy_capture_design",
    }
    sessions = []
    banks: dict[str, list[list[float]]] = {}
    evidence_banks: dict[str, list[dict[str, Any]]] = {}
    selection: dict[str, Any] = {}
    for level_index, level in enumerate(("L4", "L5")):
        session_anchors = choose_level_anchors(rows, level)
        positive = [row for anchors in session_anchors for row in anchors[:2]]
        negative = [anchors[2] for anchors in session_anchors]
        selection[level] = {
            "positive_anchor_count": len(positive),
            "negative_anchor_count": len(negative),
            "positive_anchor_poses": [row["pose"].tolist() for row in positive],
            "negative_anchor_poses": [row["pose"].tolist() for row in negative],
        }
        for session_index in range(4):
            first_positive, second_positive, session_negative = session_anchors[session_index]
            anchors = (
                ("OBSERVED_TP_SAP_INTERFACE", first_positive),
                ("OBSERVED_TP_SAP_INTERFACE", second_positive),
                ("OBSERVED_OTHER_BONE_WITHOUT_SAP", session_negative),
            )
            poses: list[np.ndarray] = []
            evidence_rows = []
            for role, anchor in anchors:
                poses.extend(anchor["sweep"])
                evidence_rows.extend(evidence(anchor, role, offset) for offset in range(8))
            validate_waypoint_bank(
                poses,
                command_minimum=POSE_MINIMUM,
                command_maximum=POSE_MAXIMUM,
                label_resolution_mm=LABEL_RESOLUTION_MM,
                position_tolerance_mm=float(
                    base["sampling"]["physical_tracking_position_tolerance_mm"]
                ),
                orientation_tolerance_deg=float(
                    base["sampling"]["physical_tracking_orientation_tolerance_deg"]
                ),
                separation_multiplier=float(
                    base["sampling"].get("waypoint_separation_multiplier", 1.0)
                ),
            )
            bank_id = f"{level.lower()}_bank_{session_index:02d}"
            session_id = f"s0010sonodevv2{level.lower()}_{session_index:02d}"
            banks[bank_id] = [pose.tolist() for pose in poses]
            evidence_banks[bank_id] = evidence_rows
            sessions.append(
                {
                    "session_id": session_id,
                    "split": "opened_sonoanatomy_development_v2",
                    "seed": 2026112310 + level_index * 10 + session_index,
                    "reset_id": session_id,
                    "vertebral_level": level,
                    "absolute_waypoint_bank_id": bank_id,
                    "expected_frame_count": 24,
                }
            )
    sampling = protocol["sampling"]
    sampling["absolute_waypoint_banks_index"] = banks
    sampling["planned_waypoint_evidence_by_bank"] = evidence_banks
    sampling["trajectory_start_frames"] = [0, 8, 16]
    sampling["expected_frames_per_session"] = 24
    sampling["maximum_initial_positioning_steps"] = 700
    sampling["maximum_action_steps_per_waypoint"] = 250
    sampling["maximum_total_action_steps_per_session"] = 7000
    sampling["episode_length_s_override"] = 700.0
    sampling["physical_controller"]["ceiling_assisted_minimum_requested_angle_rad"] = 2.99
    sampling["physical_controller"]["capture_angle_ceiling_limited_samples"] = False
    protocol["sessions"] = sessions
    protocol["visualization"] = {"record_robot_video": False, "actor_input": False}
    protocol["change_policy"] = [
        "This is opened development data and cannot be independent confirmation.",
        "Absolute waypoints and source labels are evaluator-only capture-design inputs.",
        "The capture must be relabeled from synchronized native coordinates after execution.",
        "No SAP boundary or MBB target is promoted beyond simulator-proxy status.",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = args.output_dir / "s0010_sonoanatomy_development_v2.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    protocol_path.with_suffix(".sha256").write_text(
        f"{sha256(protocol_path)}  {protocol_path.name}\n", encoding="utf-8"
    )
    result = {
        "schema": "s0010-sonoanatomy-development-capture-preparation-v2",
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256(protocol_path),
        "session_count": len(sessions),
        "frame_count": sum(int(item["expected_frame_count"]) for item in sessions),
        "level_frame_count": {"L4": 96, "L5": 96},
        "selection": selection,
        "anatomical_label_status": "SIMULATOR_WEAK_ANATOMICAL_LABEL",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
