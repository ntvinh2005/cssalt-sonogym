#!/usr/bin/env python3
"""Freeze SAP capture sessions around supported poses and clear controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml


LEVELS = ("L4", "L5")
SESSION_DESIGN = (
    ("s0010sapval_train_06", "train", 12, 8),
    ("s0010sapval_train_07", "train", 12, 8),
    ("s0010sapval_calibration_03", "calibration", 18, 12),
    ("s0010sapval_evaluation_04", "evaluation", 12, 8),
    ("s0010sapval_evaluation_05", "evaluation", 12, 8),
)
POSITIVE_OFFSETS = (
    (0.0, 0.0, 0.0),
    (4.0, 0.0, 0.0),
    (-4.0, 0.0, 0.0),
    (0.0, 4.0, 0.0),
    (0.0, -4.0, 0.0),
    (0.0, 0.0, 0.08),
    (0.0, 0.0, -0.08),
    (4.0, 4.0, 0.0),
    (-4.0, 4.0, 0.0),
    (4.0, -4.0, 0.0),
    (-4.0, -4.0, 0.0),
)
BOUNDS = {
    "L4": ((143.0, 165.5), (135.0, 152.5), (2.74, 2.99)),
    "L5": ((143.0, 165.5), (118.0, 135.5), (2.74, 2.99)),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def as_bool(value: str) -> bool:
    return value.lower() == "true"


def frame_key(row: dict[str, str]) -> tuple[str, int, str]:
    return row["session_id"], int(row["frame_sequence"]), row["conditioned_level"]


def positive_score(row: dict[str, str]) -> tuple[float, ...]:
    return (
        float(as_bool(row["passes"])),
        float(row["surface_columns"]),
        float(row["longest_aligned_run_columns"]),
        float(row["aligned_fraction"]),
        float(row["overlapping_surface_fraction"]),
        -float(row["distance_p90_pixels"]),
        float(row["local_gradient_median"]),
        float(row["local_contrast_median"]),
        float(row["posterior_shadow_median"]),
    )


def negative_score(row: dict[str, str]) -> tuple[float, ...]:
    return (
        float(row["feature_gradient_p90"]),
        float(row["feature_intensity_std"]),
    )


def round_robin_sessions(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in rows:
        grouped[str(row["session_id"])].append(row)
    ordered: list[dict[str, Any]] = []
    session_ids = sorted(grouped)
    while any(grouped.values()):
        for session_id in session_ids:
            if grouped[session_id]:
                ordered.append(grouped[session_id].popleft())
    return ordered


def source_pose_lookup(protocol: dict[str, Any]) -> dict[tuple[str, int], np.ndarray]:
    result: dict[tuple[str, int], np.ndarray] = {}
    for session in protocol["sessions"]:
        session_id = str(session["session_id"])
        root = Path(session["source_session_path"])
        for row in read_jsonl(root / "privileged/annotations.jsonl"):
            result[(session_id, int(row["frame_sequence"]))] = np.asarray(
                row["realized_pose_index"], dtype=np.float64
            )
    return result


def candidate_rows(
    manifest: Path, source_protocol: dict[str, Any]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    poses = source_pose_lookup(source_protocol)
    output = {
        level: {"SAP_RELATED_CORTICAL_INTERFACE": [], "OTHER_OR_BACKGROUND": []}
        for level in LEVELS
    }
    seen: set[tuple[str, int, str, str]] = set()
    for row in read_csv(manifest):
        level = row["conditioned_level"]
        label = row["provisional_label"]
        if level not in output or label not in output[level]:
            continue
        if not as_bool(row["selected_for_dataset"]) or not as_bool(
            row["training_eligible"]
        ):
            continue
        key = (*frame_key(row), label)
        if key in seen:
            continue
        if label == "SAP_RELATED_CORTICAL_INTERFACE" and not as_bool(row["passes"]):
            continue
        if label == "OTHER_OR_BACKGROUND" and any(
            int(row[name]) != 0
            for name in (
                "surface_pixels_in_patch",
                "pseudo_pixels_in_patch",
                "uncertainty_pixels_in_patch",
                "other_level_surface_pixels_in_patch",
            )
        ):
            continue
        seen.add(key)
        pose = poses[(row["session_id"], int(row["frame_sequence"]))]
        output[level][label].append(
            {
                "session_id": row["session_id"],
                "frame_sequence": int(row["frame_sequence"]),
                "frame_sha256": row["frame_sha256"],
                "pose": pose,
                "score": positive_score(row)
                if label == "SAP_RELATED_CORTICAL_INTERFACE"
                else negative_score(row),
            }
        )
    for level in LEVELS:
        for label in output[level]:
            rows = sorted(
                output[level][label],
                key=lambda item: (item["score"], item["frame_sha256"]),
                reverse=True,
            )
            output[level][label] = round_robin_sessions(rows)
    return output


def clipped_pose(level: str, pose: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.clip(pose[index], bounds[0], bounds[1])
            for index, bounds in enumerate(BOUNDS[level])
        ],
        dtype=np.float64,
    )


def pose_key(pose: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.round(pose, 6))


def positive_pool(level: str, bases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    for offset in POSITIVE_OFFSETS:
        for base in bases:
            pose = clipped_pose(level, base["pose"] + np.asarray(offset))
            key = pose_key(pose)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "pose": pose,
                    "level": level,
                    "planned_label": "SAP_RELATED_CORTICAL_INTERFACE",
                    "source_session_id": base["session_id"],
                    "source_frame_sequence": base["frame_sequence"],
                    "source_frame_sha256": base["frame_sha256"],
                    "offset_index": [float(value) for value in offset],
                }
            )
    return rows


def negative_pool(level: str, bases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    for base in bases:
        pose = clipped_pose(level, base["pose"])
        key = pose_key(pose)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "pose": pose,
                "level": level,
                "planned_label": "OTHER_OR_BACKGROUND",
                "source_session_id": base["session_id"],
                "source_frame_sequence": base["frame_sequence"],
                "source_frame_sha256": base["frame_sha256"],
                "offset_index": [0.0, 0.0, 0.0],
            }
        )
    return rows


def planned_separation(
    left: Sequence[float], right: Sequence[float], label_resolution_mm: float
) -> tuple[float, float]:
    translation = float(
        np.linalg.norm(np.asarray(left[:2]) - np.asarray(right[:2]))
        * label_resolution_mm
    )
    orientation = float((right[2] - left[2] + math.pi) % (2.0 * math.pi) - math.pi)
    return translation, abs(orientation) * 180.0 / math.pi


def has_tracking_margin(
    left: Sequence[float], right: Sequence[float], margin: dict[str, float]
) -> bool:
    translation, orientation = planned_separation(
        left, right, margin["label_resolution_mm"]
    )
    return bool(
        translation >= margin["required_translation_mm"]
        or orientation >= margin["required_orientation_deg"]
    )


def reorder_records(
    records: list[dict[str, Any]],
    margin: dict[str, float],
    previous: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    rows = list(records)
    adjacency = {
        index: {
            other
            for other in range(len(rows))
            if other != index
            and has_tracking_margin(rows[index]["pose"], rows[other]["pose"], margin)
        }
        for index in range(len(rows))
    }
    starts = [
        index
        for index, row in enumerate(rows)
        if previous is None or has_tracking_margin(previous, row["pose"], margin)
    ]
    starts.sort(key=lambda index: (len(adjacency[index]), index))
    visits = 0

    def search(path: list[int], remaining: set[int]) -> list[int] | None:
        nonlocal visits
        visits += 1
        if visits > 100_000:
            return None
        if not remaining:
            return path
        candidates = adjacency[path[-1]] & remaining
        ordered = sorted(
            candidates,
            key=lambda index: (len(adjacency[index] & remaining), index),
        )
        for index in ordered:
            result = search(path + [index], remaining - {index})
            if result is not None:
                return result
        return None

    for start in starts:
        result = search([start], set(range(len(rows))) - {start})
        if result is not None:
            return [rows[index] for index in result]
    raise ValueError("waypoint block cannot satisfy the tracking margin")


def take_unique(
    pool: list[dict[str, Any]], count: int, used: set[tuple[float, float, float]]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    while pool and len(selected) < count:
        row = pool.pop(0)
        key = pose_key(row["pose"])
        if key in used:
            continue
        used.add(key)
        selected.append(row)
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} / {count} distinct planned poses available")
    return selected


def build_protocol(
    source_protocol: dict[str, Any],
    source_protocol_path: Path,
    source_dataset_dir: Path,
    resume_protocol: dict[str, Any],
    resume_protocol_path: Path,
    implementation_commit: str,
) -> dict[str, Any]:
    candidates = candidate_rows(
        source_dataset_dir / "sample_manifest.csv", source_protocol
    )
    pools = {
        level: {
            "SAP_RELATED_CORTICAL_INTERFACE": positive_pool(
                level, candidates[level]["SAP_RELATED_CORTICAL_INTERFACE"]
            ),
            "OTHER_OR_BACKGROUND": negative_pool(
                level, candidates[level]["OTHER_OR_BACKGROUND"]
            ),
        }
        for level in LEVELS
    }
    sampling = resume_protocol["sampling"]
    margin = {
        "label_resolution_mm": float(resume_protocol["exact_pose_replay"]["label_resolution_m"])
        * 1000.0,
        "required_translation_mm": float(
            sampling["minimum_realized_viewpoint_translation_mm"]
        )
        + 2.0 * float(sampling["physical_tracking_position_tolerance_mm"])
        + 1.0,
        "required_orientation_deg": float(
            sampling["minimum_realized_viewpoint_orientation_deg"]
        )
        + 2.0 * float(sampling["physical_tracking_orientation_tolerance_deg"])
        + 0.5,
    }
    sessions: list[dict[str, Any]] = []
    banks: dict[str, list[list[float]]] = {}
    level_orders: dict[str, list[dict[str, Any]]] = {}
    start_frames: dict[str, list[int]] = {}
    roles: dict[str, list[dict[str, Any]]] = {}
    used: set[tuple[float, float, float]] = set()
    for session_index, (session_id, split, positive_count, negative_count) in enumerate(
        SESSION_DESIGN
    ):
        session_records: list[dict[str, Any]] = []
        level_orders[session_id] = []
        for level in LEVELS:
            level_rows = take_unique(
                pools[level]["SAP_RELATED_CORTICAL_INTERFACE"], positive_count, used
            ) + take_unique(
                pools[level]["OTHER_OR_BACKGROUND"], negative_count, used
            )
            start = len(session_records)
            level_rows = reorder_records(
                level_rows,
                margin,
                previous=session_records[-1]["pose"] if session_records else None,
            )
            session_records.extend(level_rows)
            level_orders[session_id].append(
                {
                    "level": level,
                    "start_index": start,
                    "stop_index_exclusive": len(session_records),
                }
            )
        banks[session_id] = [
            [float(value) for value in row["pose"]] for row in session_records
        ]
        roles[session_id] = [
            {
                "waypoint_index": index,
                "level": row["level"],
                "planned_label": row["planned_label"],
                "not_a_training_label": True,
                "source_session_id": row["source_session_id"],
                "source_frame_sequence": row["source_frame_sequence"],
                "source_frame_sha256": row["source_frame_sha256"],
                "offset_index": row["offset_index"],
            }
            for index, row in enumerate(session_records)
        ]
        count = len(session_records)
        start_frames[session_id] = list(range(0, count, 2))
        sessions.append(
            {
                "session_id": session_id,
                "split": split,
                "seed": 2026112401 + session_index,
                "reset_id": session_id,
                "absolute_waypoint_bank_id": session_id,
                "expected_frame_count": count,
            }
        )
    document = dict(resume_protocol)
    document["protocol_id"] = "sonogym-s0010-sap-validation-resume-v2"
    document["status"] = "PRE_GPU_FROZEN"
    document["purpose"] = (
        "Collect session-separated SAP views around supported archived poses and clear "
        "other-bone or background controls after the first resume capture remained sparse."
    )
    document["source"] = {
        **resume_protocol["source"],
        "implementation_commit": implementation_commit,
        "source_existing_capture_protocol": str(source_protocol_path),
        "source_existing_capture_protocol_sha256": file_sha256(source_protocol_path),
        "source_dataset_manifest": str(source_dataset_dir / "sample_manifest.csv"),
        "source_dataset_manifest_sha256": file_sha256(
            source_dataset_dir / "sample_manifest.csv"
        ),
        "source_resume_protocol": str(resume_protocol_path),
        "source_resume_protocol_sha256": file_sha256(resume_protocol_path),
        "prior_capture_job": 52182100,
        "target_selection": "ranked_supported_frames_and_clear_controls_with_disjoint_pose_allocation",
    }
    document["sessions"] = sessions
    document["sampling"] = {
        **sampling,
        "absolute_waypoint_banks_index": banks,
        "waypoint_level_order_by_session": level_orders,
        "trajectory_start_frames_by_session": start_frames,
        "planned_waypoint_evidence_by_session": roles,
        "waypoint_tracking_margin": margin,
    }
    document["dataset_gate"] = {
        **document["dataset_gate"],
        "require_all_expected_frames": True,
        "minimum_distinct_frame_hashes_by_session": {
            row["session_id"]: row["expected_frame_count"] for row in sessions
        },
    }
    document["change_policy"] = [
        "The planned label records coverage intent only and is not a training label.",
        "Rebuild simulator-proxy labels from synchronized native coordinates after capture.",
        "Do not inspect evaluation images before checkpoints and thresholds are frozen.",
        "Do not expose privileged geometry to classifier features.",
        "Preserve all earlier protocols and results as historical evidence.",
    ]
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-existing-protocol", type=Path, required=True)
    parser.add_argument("--source-dataset-dir", type=Path, required=True)
    parser.add_argument("--source-resume-protocol", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_protocol = yaml.safe_load(
        args.source_existing_protocol.read_text(encoding="utf-8")
    )
    resume_protocol = yaml.safe_load(
        args.source_resume_protocol.read_text(encoding="utf-8")
    )
    document = build_protocol(
        source_protocol,
        args.source_existing_protocol,
        args.source_dataset_dir,
        resume_protocol,
        args.source_resume_protocol,
        args.implementation_commit,
    )
    args.output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    digest = file_sha256(args.output)
    args.output.with_suffix(".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "protocol_id": document["protocol_id"],
                "sessions": len(document["sessions"]),
                "frames": sum(row["expected_frame_count"] for row in document["sessions"]),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
