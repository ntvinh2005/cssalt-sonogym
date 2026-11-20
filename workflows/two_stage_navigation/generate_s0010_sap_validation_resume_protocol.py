#!/usr/bin/env python3
"""Create a remaining-session SAP capture protocol with robust waypoint spacing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

REMAINING_SESSIONS = (
    ("s0010sapval_calibration_01", "s0010sapval_calibration_02", 2026112301),
    ("s0010sapval_evaluation_00", "s0010sapval_evaluation_02", 2026112302),
    ("s0010sapval_evaluation_01", "s0010sapval_evaluation_03", 2026112303),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def planned_separation(
    left: Sequence[float], right: Sequence[float], *, label_resolution_mm: float
) -> tuple[float, float]:
    translation_mm = float(
        np.linalg.norm(np.asarray(left[:2]) - np.asarray(right[:2]))
        * label_resolution_mm
    )
    orientation_rad = float((right[2] - left[2] + math.pi) % (2.0 * math.pi) - math.pi)
    return translation_mm, abs(orientation_rad) * 180.0 / math.pi


def transition_has_tracking_margin(
    left: Sequence[float],
    right: Sequence[float],
    *,
    label_resolution_mm: float,
    tracking_position_tolerance_mm: float,
    tracking_orientation_tolerance_deg: float,
    minimum_realized_translation_mm: float,
    minimum_realized_orientation_deg: float,
    extra_translation_margin_mm: float,
    extra_orientation_margin_deg: float,
) -> bool:
    translation_mm, orientation_deg = planned_separation(
        left, right, label_resolution_mm=label_resolution_mm
    )
    required_translation = (
        minimum_realized_translation_mm
        + 2.0 * tracking_position_tolerance_mm
        + extra_translation_margin_mm
    )
    required_orientation = (
        minimum_realized_orientation_deg
        + 2.0 * tracking_orientation_tolerance_deg
        + extra_orientation_margin_deg
    )
    return bool(
        translation_mm >= required_translation or orientation_deg >= required_orientation
    )


def reorder_waypoints(
    waypoints: Sequence[Sequence[float]],
    *,
    previous: Sequence[float] | None = None,
    **margin: float,
) -> list[list[float]]:
    if not waypoints:
        return []
    ordered: list[list[float]] = []
    remaining = [[float(value) for value in row] for row in waypoints]
    if previous is None:
        ordered.append(remaining.pop(0))
    while remaining:
        left = ordered[-1] if ordered else previous
        assert left is not None
        next_index = next(
            (
                index
                for index, row in enumerate(remaining)
                if transition_has_tracking_margin(left, row, **margin)
            ),
            None,
        )
        if next_index is None:
            raise ValueError("waypoint bank cannot satisfy the tracking-margin requirement")
        ordered.append(remaining.pop(next_index))
    return ordered


def build_resume_protocol(
    source: dict[str, Any], source_path: Path, implementation_commit: str
) -> dict[str, Any]:
    document = deepcopy(source)
    sampling = document["sampling"]
    replay = document["exact_pose_replay"]
    margin = {
        "label_resolution_mm": float(replay["label_resolution_m"]) * 1000.0,
        "tracking_position_tolerance_mm": float(
            sampling["physical_tracking_position_tolerance_mm"]
        ),
        "tracking_orientation_tolerance_deg": float(
            sampling["physical_tracking_orientation_tolerance_deg"]
        ),
        "minimum_realized_translation_mm": float(
            sampling["minimum_realized_viewpoint_translation_mm"]
        ),
        "minimum_realized_orientation_deg": float(
            sampling["minimum_realized_viewpoint_orientation_deg"]
        ),
        "extra_translation_margin_mm": 1.0,
        "extra_orientation_margin_deg": 0.5,
    }
    source_sessions = {row["session_id"]: row for row in document["sessions"]}
    source_banks = sampling["absolute_waypoint_banks_index"]
    source_levels = sampling["waypoint_level_order_by_session"]
    source_trajectories = sampling["trajectory_start_frames_by_session"]
    sessions: list[dict[str, Any]] = []
    banks: dict[str, list[list[float]]] = {}
    levels: dict[str, list[dict[str, Any]]] = {}
    trajectories: dict[str, list[int]] = {}
    for source_id, session_id, seed in REMAINING_SESSIONS:
        source_session = source_sessions[source_id]
        session = deepcopy(source_session)
        session.update(
            {
                "session_id": session_id,
                "seed": seed,
                "reset_id": session_id,
                "absolute_waypoint_bank_id": session_id,
            }
        )
        sessions.append(session)
        ordered_bank: list[list[float]] = []
        levels[session_id] = deepcopy(source_levels[source_id])
        for level_range in levels[session_id]:
            start = int(level_range["start_index"])
            stop = int(level_range["stop_index_exclusive"])
            block = reorder_waypoints(
                source_banks[source_id][start:stop],
                previous=ordered_bank[-1] if ordered_bank else None,
                **margin,
            )
            ordered_bank.extend(block)
        banks[session_id] = ordered_bank
        trajectories[session_id] = deepcopy(source_trajectories[source_id])
    document["protocol_id"] = "sonogym-s0010-sap-validation-resume-v1"
    document["status"] = "PRE_GPU_FROZEN"
    document["purpose"] = (
        "Resume the failed calibration and unopened evaluation sessions with "
        "waypoint transitions that retain realized-viewpoint margin."
    )
    document["source"] = {
        **document["source"],
        "implementation_commit": implementation_commit,
        "source_protocol": str(source_path),
        "source_protocol_sha256": file_sha256(source_path),
        "failed_capture_job": 52131349,
        "failed_realized_translation_mm": 1.9915416010134313,
        "failed_realized_orientation_deg": 0.33978825082979436,
    }
    sampling["waypoint_tracking_margin"] = margin
    sampling["absolute_waypoint_banks_index"] = banks
    sampling["waypoint_level_order_by_session"] = levels
    sampling["trajectory_start_frames_by_session"] = trajectories
    document["sessions"] = sessions
    document["change_policy"] = [
        *document["change_policy"],
        "Preserve the failed follow-up protocol and capture as historical evidence.",
        "Do not inspect the evaluation images before the classifier and threshold are frozen.",
    ]
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = yaml.safe_load(args.source_protocol.read_text(encoding="utf-8"))
    document = build_resume_protocol(
        source, args.source_protocol, args.implementation_commit
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
