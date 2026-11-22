#!/usr/bin/env python3
"""Freeze diverse L4 TP/SAP/other-bone training trajectories from a broad search."""

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


PROTOCOL_ID = "sonogym-s0010-sonoanatomy-development-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_candidates(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result.append(
                {
                    **row,
                    "pose": np.asarray(json.loads(row["pose_index_rad"]), dtype=np.float64),
                    "sap_core": int(row["sap_core_first_visible_pixels"]),
                    "tp_core": int(row["tp_core_first_visible_pixels"]),
                    "sap": int(row["sap_first_visible_pixels"]),
                    "tp": int(row["tp_first_visible_pixels"]),
                    "margin": int(row["target_margin_pixels"]),
                }
            )
    return result


def pose_distance(first: np.ndarray, second: np.ndarray) -> float:
    translation_mm = float(np.linalg.norm(first[:2] - second[:2]) * 1.5)
    angle_deg = abs(float((first[2] - second[2] + math.pi) % (2.0 * math.pi) - math.pi))
    return translation_mm + 5.0 * math.degrees(angle_deg)


def categories(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    positive = [row for row in rows if row["evaluator_label"] == "PROXY_POSITIVE"]
    transition = [
        row
        for row in rows
        if row["evaluator_label"] == "PROXY_AMBIGUOUS"
        and row["sap"] > 0
        and row["tp"] > 0
    ]
    negative = [row for row in rows if row["evaluator_label"] == "PROXY_NEGATIVE"]
    if min(len(positive), len(transition), len(negative)) < 32:
        raise ValueError("broad search lacks 32 rows in a required capture category")
    return {"positive": positive, "transition": transition, "negative": negative}


def diverse(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            -min(row["sap_core"], row["tp_core"]),
            -min(row["sap"], row["tp"]),
            -row["margin"],
            tuple(row["pose"]),
        ),
    )
    selected = [ranked.pop(0)]
    while ranked and len(selected) < count:
        choice = max(
            ranked,
            key=lambda row: (
                min(pose_distance(row["pose"], item["pose"]) for item in selected),
                min(row["sap"], row["tp"]),
                row["margin"],
            ),
        )
        selected.append(choice)
        ranked.remove(choice)
    if len(selected) != count:
        raise ValueError(f"could not select {count} diverse poses")
    return selected


def nearest_distinct(
    anchor: dict[str, Any],
    pool: list[dict[str, Any]],
    used: set[tuple[float, float, float]],
) -> dict[str, Any]:
    eligible = [
        row
        for row in pool
        if tuple(row["pose"]) not in used and pose_distance(anchor["pose"], row["pose"]) >= 4.0
    ]
    if not eligible:
        raise ValueError("no distinct transition candidate remains")
    return min(eligible, key=lambda row: (pose_distance(anchor["pose"], row["pose"]), tuple(row["pose"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--exact-candidates", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base = yaml.safe_load(args.base_protocol.read_text(encoding="utf-8"))
    grouped = categories(read_candidates(args.exact_candidates))
    anchors = diverse(grouped["positive"], 32)
    used: set[tuple[float, float, float]] = set()
    triplets = []
    for anchor in anchors:
        positive_key = tuple(anchor["pose"])
        if positive_key in used:
            raise ValueError("duplicate positive anchor")
        used.add(positive_key)
        transition = nearest_distinct(anchor, grouped["transition"], used)
        used.add(tuple(transition["pose"]))
        negative = nearest_distinct(anchor, grouped["negative"], used)
        used.add(tuple(negative["pose"]))
        triplets.append((anchor, transition, negative))

    protocol = deepcopy(base)
    protocol["protocol_id"] = PROTOCOL_ID
    protocol["status"] = "FROZEN_OPENED_SIMULATOR_SUPERVISED_DEVELOPMENT_CAPTURE"
    protocol["purpose"] = (
        "Collect physically executed L4 TP, SAP, transition and other-bone evidence "
        "from a broad evaluator-only capture design; absolute poses are never actor inputs."
    )
    protocol["source"] = {
        "implementation_commit": args.implementation_commit,
        "base_protocol_sha256": sha256(args.base_protocol),
        "exact_candidates": str(args.exact_candidates.resolve()),
        "exact_candidates_sha256": sha256(args.exact_candidates),
        "evidence_status": "privileged_opened_development_capture_design",
    }
    protocol["scope"]["privileged_pose_design"] = True
    sampling = protocol["sampling"]
    sampling["absolute_waypoint_banks_index"] = {}
    sampling["planned_waypoint_evidence_by_bank"] = {}
    sampling["trajectory_start_frames"] = list(range(0, 24, 3))
    sampling["expected_frames_per_session"] = 24
    sampling["maximum_total_action_steps_per_session"] = 7000
    sampling["episode_length_s_override"] = 700.0
    sessions = []
    for session_index in range(4):
        current = triplets[session_index * 8 : (session_index + 1) * 8]
        rows: list[list[float]] = []
        evidence: list[dict[str, Any]] = []
        for trajectory_index, (positive, transition, negative) in enumerate(current):
            for role, candidate in (
                ("L4_PROXY_POSITIVE", positive),
                ("L4_BOUNDARY_OR_TRANSITION", transition),
                ("L4_CLEAR_NEGATIVE_OR_OTHER_BONE", negative),
            ):
                rows.append([float(value) for value in candidate["pose"]])
                evidence.append(
                    {
                        "declared_role": role,
                        "source": "broad_geometry_search_opened_development",
                        "trajectory_index": trajectory_index,
                        "evaluator_label": candidate["evaluator_label"],
                        "evaluator_reason": candidate["evaluator_reason"],
                    }
                )
        validate_waypoint_bank(
            rows,
            command_minimum=[-1.0e9] * 3,
            command_maximum=[1.0e9] * 3,
            label_resolution_mm=1.5,
            position_tolerance_mm=2.0,
            orientation_tolerance_deg=1.2,
            separation_multiplier=1.0,
        )
        bank_id = f"bank_{session_index:02d}"
        sampling["absolute_waypoint_banks_index"][bank_id] = rows
        sampling["planned_waypoint_evidence_by_bank"][bank_id] = evidence
        session_id = f"s0010sonodev_{session_index:02d}"
        sessions.append(
            {
                "session_id": session_id,
                "split": "opened_sonoanatomy_development",
                "seed": 2026112200 + session_index,
                "reset_id": session_id,
                "absolute_waypoint_bank_id": bank_id,
                "expected_frame_count": 24,
            }
        )
    protocol["sessions"] = sessions
    protocol["change_policy"] = [
        "The capture is opened development evidence, never independent confirmation.",
        "Geometry and absolute waypoints are capture-design and evaluator-only inputs.",
        "The segmentation model receives clean ultrasound only.",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / "s0010_sonoanatomy_development_v1.yaml"
    path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    path.with_suffix(".sha256").write_text(f"{sha256(path)}  {path.name}\n")
    result = {
        "protocol": str(path),
        "protocol_sha256": sha256(path),
        "session_count": 4,
        "frame_count": 96,
        "trajectory_count": 32,
        "category_counts": {"positive": 32, "transition": 32, "negative": 32},
        "unique_pose_count": len(used),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
