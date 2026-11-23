#!/usr/bin/env python3
"""Prepare opened TP hard-negative sweeps from actor false-positive frames."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from prepare_s0010_sonoanatomy_capture_v2 import (
    evidence,
    local_sweep,
    pose_is_valid,
    sweeps_are_distinct,
)
from two_stage_navigation.acquisition_integrity import validate_waypoint_bank


PROTOCOL_ID = "sonogym-s0010-sonoanatomy-development-v2"
POSE_MINIMUM = np.asarray([100.0, 70.0, 1.0], dtype=np.float64)
POSE_MAXIMUM = np.asarray([180.0, 200.0, 3.0], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--actor-frames", type=Path, required=True)
    parser.add_argument("--source-capture-root", type=Path, required=True)
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


def candidates(
    *,
    dataset_manifest: Path,
    actor_frames: Path,
    source_capture_root: Path,
) -> list[dict[str, Any]]:
    with dataset_manifest.open(newline="", encoding="utf-8") as handle:
        manifest = {
            (row["session_id"], int(row["frame_sequence"])): row
            for row in csv.DictReader(handle)
            if Path(row["source_root"]).resolve() == source_capture_root.resolve()
        }
    annotations = {}
    for path in sorted(source_capture_root.glob("sessions/*/privileged/annotations.jsonl")):
        for row in read_jsonl(path):
            annotations[(row["session_id"], int(row["frame_sequence"]))] = row
    result = []
    for actor in read_jsonl(actor_frames):
        key = (str(actor["session_id"]), int(actor["frame_sequence"]))
        truth = manifest.get(key)
        annotation = annotations.get(key)
        if truth is None or annotation is None:
            raise ValueError(f"missing synchronized evidence for {key}")
        pose = np.asarray(annotation["requested_waypoint_pose_index"], dtype=np.float64)
        if not pose_is_valid(pose):
            continue
        result.append(
            {
                "level": str(actor["vertebral_level"]),
                "session_id": key[0],
                "frame_sequence": key[1],
                "frame_sha256": str(actor["frame_sha256"]),
                "pose": pose,
                "sweep": local_sweep(pose),
                "has_tp": truth["has_tp_interface"] == "True",
                "has_sap": truth["has_sap_interface"] == "True",
                "tp_predicted": bool(actor["tp_predicted"]),
                "sap_predicted": bool(actor["sap_predicted"]),
                "tp_predicted_pixels": int(actor["tp_predicted_pixels"]),
                "sap_predicted_pixels": int(actor["sap_predicted_pixels"]),
                "tp_pixels": int(truth["tp_interface_pixels"]),
                "sap_pixels": int(truth["sap_interface_pixels"]),
                "other_pixels": int(truth["other_bone_interface_pixels"]),
                "source_role": "opened_actor_false_positive_mining",
            }
        )
    return result


def choose_session_anchors(
    rows: list[dict[str, Any]], level: str, *, session_count: int = 3
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    level_rows = [row for row in rows if row["level"] == level]
    negatives = sorted(
        [row for row in level_rows if row["tp_predicted"] and not row["has_tp"]],
        key=lambda row: (row["tp_predicted_pixels"], row["sap_predicted_pixels"]),
        reverse=True,
    )
    positives = sorted(
        [
            row
            for row in level_rows
            if row["tp_predicted"]
            and row["sap_predicted"]
            and row["has_tp"]
            and row["has_sap"]
        ],
        key=lambda row: (
            min(row["tp_predicted_pixels"], row["sap_predicted_pixels"]),
            row["tp_pixels"] + row["sap_pixels"],
        ),
        reverse=True,
    )
    selected = []
    used = set()
    selected_negatives: list[dict[str, Any]] = []
    for negative in negatives:
        negative_key = tuple(float(value) for value in negative["pose"])
        if negative_key in used:
            continue
        if any(
            not sweeps_are_distinct([negative, prior])
            for prior in selected_negatives
        ):
            continue
        choice = None
        for first_index, first in enumerate(positives):
            first_key = tuple(float(value) for value in first["pose"])
            if first_key in used:
                continue
            for second in positives[first_index + 1 :]:
                second_key = tuple(float(value) for value in second["pose"])
                if second_key in used:
                    continue
                if sweeps_are_distinct([negative, first, second]):
                    choice = (negative, first, second)
                    break
            if choice is not None:
                break
        if choice is None:
            continue
        selected.append(choice)
        selected_negatives.append(negative)
        used.update(tuple(float(value) for value in row["pose"]) for row in choice)
        if len(selected) == session_count:
            return selected
    raise ValueError(f"{level} cannot form {session_count} hard-negative session banks")


def main() -> int:
    args = parse_args()
    base = yaml.safe_load(args.base_protocol.read_text(encoding="utf-8"))
    rows = candidates(
        dataset_manifest=args.dataset_manifest,
        actor_frames=args.actor_frames,
        source_capture_root=args.source_capture_root,
    )
    protocol = deepcopy(base)
    protocol["protocol_id"] = PROTOCOL_ID
    protocol["status"] = "FROZEN_OPENED_TP_HARD_NEGATIVE_DEVELOPMENT_CAPTURE"
    protocol["purpose"] = (
        "Acquire local sweeps around actor TP false positives and matched true "
        "TP/SAP controls; all source identities remain simulator-proxy labels."
    )
    protocol["source"] = {
        "implementation_commit": args.implementation_commit,
        "base_protocol": str(args.base_protocol.resolve()),
        "base_protocol_sha256": sha256(args.base_protocol),
        "dataset_manifest": str(args.dataset_manifest.resolve()),
        "dataset_manifest_sha256": sha256(args.dataset_manifest),
        "actor_frames": str(args.actor_frames.resolve()),
        "actor_frames_sha256": sha256(args.actor_frames),
        "source_capture_root": str(args.source_capture_root.resolve()),
        "evidence_status": "opened_actor_error_mining_with_privileged_relabeling",
    }
    sessions = []
    banks = {}
    evidence_banks = {}
    selection = {}
    for level_index, level in enumerate(("L4", "L5")):
        anchors = choose_session_anchors(rows, level)
        selection[level] = {
            "hard_negative_anchor_count": len(anchors),
            "positive_control_anchor_count": 2 * len(anchors),
            "hard_negative_anchor_poses": [row[0]["pose"].tolist() for row in anchors],
        }
        for session_index, (negative, first, second) in enumerate(anchors):
            specified = (
                ("ACTOR_TP_FALSE_POSITIVE", negative),
                ("TRUE_TP_SAP_CONTROL", first),
                ("TRUE_TP_SAP_CONTROL", second),
            )
            poses = []
            evidence_rows = []
            for role, anchor in specified:
                poses.extend(anchor["sweep"])
                evidence_rows.extend(evidence(anchor, role, offset) for offset in range(8))
            validate_waypoint_bank(
                poses,
                command_minimum=POSE_MINIMUM,
                command_maximum=POSE_MAXIMUM,
                label_resolution_mm=1.5,
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
            bank_id = f"{level.lower()}_hard_negative_bank_{session_index:02d}"
            session_id = f"s0010sonohardv1{level.lower()}_{session_index:02d}"
            banks[bank_id] = [pose.tolist() for pose in poses]
            evidence_banks[bank_id] = evidence_rows
            sessions.append(
                {
                    "session_id": session_id,
                    "split": "opened_sonoanatomy_hard_negative_development_v1",
                    "seed": 2026112340 + level_index * 10 + session_index,
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
    protocol["sessions"] = sessions
    protocol["visualization"] = {"record_robot_video": False, "actor_input": False}
    protocol["change_policy"] = [
        "This is opened development data and cannot be independent confirmation.",
        "Actor errors select scan regions; privileged labels only relabel captured frames.",
        "No source role or anchor declaration is available to an online actor.",
        "No SAP boundary or MBB target is promoted beyond simulator-proxy status.",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "s0010_sonoanatomy_hard_negative_v1.yaml"
    output.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    digest = sha256(output)
    (args.output_dir / "s0010_sonoanatomy_hard_negative_v1.sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    manifest = {
        "schema": "s0010-sonoanatomy-hard-negative-capture-preparation-v1",
        "protocol": str(output.resolve()),
        "protocol_sha256": digest,
        "session_count": len(sessions),
        "frame_count": sum(row["expected_frame_count"] for row in sessions),
        "selection": selection,
        "anatomical_label_status": "SIMULATOR_WEAK_ANATOMICAL_LABEL",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
