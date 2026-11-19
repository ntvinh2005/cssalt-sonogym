#!/usr/bin/env python3
"""Plan a targeted L4 TP proxy capture from verified prior acquisitions.

The planner uses privileged geometry only to construct an offline development
dataset.  Its outputs are waypoint banks and provenance diagnostics; none of
the privileged fields are actor-visible inference inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


L4 = "L4_RIGHT_TP_SURFACE_CORE"
POSITIVE = "L4_PROXY_POSITIVE_ANCHOR"
FALSE_TRIGGER = "FROZEN_L4_FALSE_TRIGGER"
COMPETING_L5 = "COMPETING_L5_CORE"
MIN_UNIQUE_CORE_PIXELS = 12
SESSION_SPECS = {
    "train_a": {"split": "train", "offset_scale": 0.45},
    "train_b": {"split": "train", "offset_scale": 0.70},
    "calibration": {"split": "calibration", "offset_scale": 0.95},
    "development_evaluation": {
        "split": "development_evaluation",
        "offset_scale": 1.20,
    },
}
BASE_OFFSETS = (
    (-1.5, -0.5, -0.018),
    (1.5, 0.5, 0.018),
    (-0.5, 1.5, 0.028),
    (0.5, -1.5, -0.028),
    (-2.0, 1.0, 0.012),
    (2.0, -1.0, -0.012),
    (-1.0, -2.0, 0.022),
    (1.0, 2.0, -0.022),
)
SAFE_BOUNDS = ((136.0, 165.0), (121.0, 158.0), (2.80, 2.95))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def is_l4_positive(row: dict[str, str]) -> bool:
    return bool(
        row["projection_status"] == "VERIFIED"
        and row["candidate_id"] == L4
        and row["review_target"] == "REVIEW_POSITIVE"
        and int(row["l4_core_points_in_patch"]) > 0
        and int(row["l5_core_points_in_patch"]) == 0
        and int(row["l4_uncertain_points_in_patch"]) == 0
        and int(row["l5_uncertain_points_in_patch"]) == 0
        and row["adequate_projected_support"].lower() == "true"
    )


def load_annotations(reference_run: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted((reference_run / "sessions").glob("*/privileged/annotations.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            result[str(row["frame_sha256"])] = row
    if not result:
        raise ValueError("reference run contains no privileged annotations")
    return result


def select_positive_anchors(
    rows: list[dict[str, str]], annotations: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    by_frame: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_frame[row["frame_sha256"]].append(row)
    selected: list[dict[str, Any]] = []
    for frame_hash, proposals in sorted(by_frame.items()):
        eligible = [row for row in proposals if is_l4_positive(row)]
        if not eligible:
            continue
        best = max(
            eligible,
            key=lambda row: (
                int(row["unique_core_pixels"]),
                int(row["interior_core_feature_cells"]),
                row["sample_id"],
            ),
        )
        support = int(best["unique_core_pixels"])
        if support < MIN_UNIQUE_CORE_PIXELS:
            continue
        annotation = annotations[frame_hash]
        selected.append(
            {
                "anchor_id": best["sample_id"],
                "anchor_type": POSITIVE,
                "frame_sha256": frame_hash,
                "requested_pose_index": [
                    float(value) for value in annotation["requested_waypoint_pose_index"]
                ],
                "realized_pose_index": [
                    float(value) for value in annotation["realized_pose_index"]
                ],
                "unique_core_pixels": support,
                "interior_core_feature_cells": int(best["interior_core_feature_cells"]),
            }
        )
    if len(selected) < 6:
        raise ValueError(f"only {len(selected)} strong L4-positive anchors were found")
    return selected


def select_confuser_anchors(
    rows: list[dict[str, str]], annotations: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        role = row["source_role"]
        if role not in {FALSE_TRIGGER, COMPETING_L5}:
            continue
        frame_hash = row["frame_sha256"]
        if frame_hash in seen:
            continue
        seen.add(frame_hash)
        annotation = annotations[frame_hash]
        selected.append(
            {
                "anchor_id": row["sample_id"],
                "anchor_type": role,
                "frame_sha256": frame_hash,
                "requested_pose_index": [
                    float(value) for value in annotation["requested_waypoint_pose_index"]
                ],
                "realized_pose_index": [
                    float(value) for value in annotation["realized_pose_index"]
                ],
                "frozen_l4_probability": float(row.get("frozen_l4_probability") or 0.0),
            }
        )
    roles = {row["anchor_type"] for row in selected}
    if roles != {FALSE_TRIGGER, COMPETING_L5}:
        raise ValueError(f"confuser anchors are incomplete: {sorted(roles)}")
    return selected


def offset_pose(pose: list[float], offset: tuple[float, float, float], scale: float) -> list[float]:
    values = [pose[index] + scale * offset[index] for index in range(3)]
    return [
        round(min(max(value, lower), upper), 6)
        for value, (lower, upper) in zip(values, SAFE_BOUNDS)
    ]


def pose_separation(first: list[float], second: list[float]) -> tuple[float, float]:
    translation_mm = math.hypot(second[0] - first[0], second[1] - first[1]) * 1.5
    angle_deg = abs(second[2] - first[2]) * 180.0 / math.pi
    return translation_mm, angle_deg


def build_bank(
    positives: list[dict[str, Any]],
    confusers: list[dict[str, Any]],
    *,
    session_name: str,
    scale: float,
) -> list[dict[str, Any]]:
    bank: list[dict[str, Any]] = []
    # Each two-frame trajectory contains one positive-targeted view and one
    # confuser-targeted view.  This yields 24 declared views of each type while
    # keeping the within-trajectory movement comfortably executable.
    session_shift = list(SESSION_SPECS).index(session_name)
    for pair_index in range(24):
        positive_index = pair_index % len(positives)
        positive_cycle = pair_index // len(positives)
        offset_index = (positive_cycle + 2 * session_shift) % len(BASE_OFFSETS)
        positive = positives[positive_index]
        confuser_index = pair_index % len(confusers)
        confuser_cycle = pair_index // len(confusers)
        confuser_offset_index = (
            confuser_cycle + 3 + 2 * session_shift
        ) % len(BASE_OFFSETS)
        confuser = confusers[confuser_index]
        positive_pose = offset_pose(
            positive["requested_pose_index"], BASE_OFFSETS[offset_index], scale
        )
        confuser_pose = offset_pose(
            confuser["requested_pose_index"],
            BASE_OFFSETS[confuser_offset_index],
            scale,
        )
        translation_mm, angle_deg = pose_separation(positive_pose, confuser_pose)
        if translation_mm <= 3.0 and angle_deg <= 2.0:
            raise ValueError("planned waypoint pair can be swallowed by motion tolerance")
        bank.extend(
            [
                {
                    "pose_index": positive_pose,
                    "target_role": POSITIVE,
                    "anchor_id": positive["anchor_id"],
                    "anchor_frame_sha256": positive["frame_sha256"],
                    "offset_index": offset_index,
                },
                {
                    "pose_index": confuser_pose,
                    "target_role": confuser["anchor_type"],
                    "anchor_id": confuser["anchor_id"],
                    "anchor_frame_sha256": confuser["frame_sha256"],
                    "offset_index": confuser_offset_index,
                },
            ]
        )
    poses = [tuple(row["pose_index"]) for row in bank]
    if len(poses) != 48 or len(set(poses)) != len(poses):
        raise ValueError(f"{session_name} does not contain 48 unique poses")
    return bank


def build_plan(
    candidate_rows: list[dict[str, str]],
    training_rows: list[dict[str, str]],
    annotations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    positives = select_positive_anchors(candidate_rows, annotations)
    confusers = select_confuser_anchors(training_rows, annotations)
    banks = {
        name: build_bank(
            positives, confusers, session_name=name, scale=float(spec["offset_scale"])
        )
        for name, spec in SESSION_SPECS.items()
    }
    split_poses: dict[str, set[tuple[float, ...]]] = defaultdict(set)
    for name, rows in banks.items():
        split = str(SESSION_SPECS[name]["split"])
        poses = {tuple(row["pose_index"]) for row in rows}
        if split_poses[split] & poses:
            raise ValueError(f"duplicate pose within split {split}")
        split_poses[split].update(poses)
    split_names = sorted(split_poses)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            if split_poses[first] & split_poses[second]:
                raise ValueError(f"cross-split waypoint overlap: {first}, {second}")
    return {
        "schema": "sonogym-l4-tp-proxy-waypoint-plan-v1",
        "status": "CPU_PREFLIGHT_PASS",
        "minimum_unique_core_pixels_for_anchor": MIN_UNIQUE_CORE_PIXELS,
        "positive_anchor_count": len(positives),
        "confuser_anchor_count": len(confusers),
        "positive_anchors": positives,
        "confuser_anchors": confusers,
        "banks": banks,
        "bank_summary": {
            name: {
                "split": SESSION_SPECS[name]["split"],
                "frames": len(rows),
                "positive_targeted": sum(row["target_role"] == POSITIVE for row in rows),
                "false_trigger_targeted": sum(
                    row["target_role"] == FALSE_TRIGGER for row in rows
                ),
                "competing_l5_targeted": sum(
                    row["target_role"] == COMPETING_L5 for row in rows
                ),
            }
            for name, rows in banks.items()
        },
    }


def main() -> int:
    args = parse_args()
    reference = args.reference_run.resolve()
    candidate_manifest = reference / "dataset/candidate_sample_manifest.csv"
    supplement_manifest = reference / "training/supplement_training_samples.csv"
    annotations = load_annotations(reference)
    plan = build_plan(
        read_csv(candidate_manifest), read_csv(supplement_manifest), annotations
    )
    plan["reference"] = {
        "run": str(reference),
        "candidate_manifest_sha256": sha256(candidate_manifest),
        "supplement_manifest_sha256": sha256(supplement_manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": plan["status"], **plan["bank_summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
