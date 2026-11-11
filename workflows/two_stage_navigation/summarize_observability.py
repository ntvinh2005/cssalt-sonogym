#!/usr/bin/env python3
"""Validate and summarize all frozen SonoGym G0/G1 capture sessions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.annotations import (  # noqa: E402
    dataset_summary,
    file_sha256,
    validate_privileged_rows,
    validate_public_rows,
    validate_split_assignments,
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sessions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    assignments = validate_split_assignments(protocol["sessions"])
    expected_session_ids = [item["session_id"] for item in protocol["sessions"]]
    public_rows: list[dict[str, Any]] = []
    privileged_rows: list[dict[str, Any]] = []
    session_summaries: list[dict[str, Any]] = []
    failures: list[str] = []

    for session_id in expected_session_ids:
        session_dir = args.sessions_dir / session_id
        summary_path = session_dir / "summary.json"
        if not summary_path.is_file():
            failures.append(f"MISSING_SESSION:{session_id}")
            continue
        summary = read_json(summary_path)
        session_summaries.append(summary)
        if summary.get("status") != "PASSED":
            failures.append(f"SESSION_FAILED:{session_id}")
            continue
        public_rows.extend(read_jsonl(session_dir / "public" / "frames.jsonl"))
        privileged_rows.extend(
            read_jsonl(session_dir / "privileged" / "annotations.jsonl")
        )

    if not failures:
        public_rows = validate_public_rows(public_rows, assignments)
        privileged_rows = validate_privileged_rows(privileged_rows)
        provenance = dataset_summary(public_rows, privileged_rows, file_sha256(args.protocol))
    else:
        provenance = {
            "schema": "sonogym-two-stage-dataset-summary-v1",
            "protocol_sha256": file_sha256(args.protocol),
            "one_to_one_join": False,
            "actor_privileged_leakage": None,
        }

    public_by_key = {
        (row["session_id"], row["frame_sequence"], row["frame_sha256"]): row
        for row in public_rows
    }
    support: dict[str, dict[str, set[str] | int]] = defaultdict(
        lambda: {
            "l4_frames": 0,
            "adjacent_negative_frames": 0,
            "l4_sessions": set(),
            "adjacent_negative_sessions": set(),
            "l4_directions": set(),
            "adjacent_negative_directions": set(),
        }
    )
    total_mismatches = 0
    for row in privileged_rows:
        key = (row["session_id"], row["frame_sequence"], row["frame_sha256"])
        public = public_by_key[key]
        split = str(public["split"])
        phase = str(public["phase"])
        total_mismatches += int(row["raw_to_acoustic_bone_mismatch_pixels"])
        if int(row["raw_label_pixel_counts"]["28"]) > 0:
            support[split]["l4_frames"] += 1  # type: ignore[operator]
            support[split]["l4_sessions"].add(str(row["session_id"]))  # type: ignore[union-attr]
            support[split]["l4_directions"].add(phase)  # type: ignore[union-attr]
        if bool(row["adjacent_negative_without_l4"]):
            support[split]["adjacent_negative_frames"] += 1  # type: ignore[operator]
            support[split]["adjacent_negative_sessions"].add(str(row["session_id"]))  # type: ignore[union-attr]
            support[split]["adjacent_negative_directions"].add(phase)  # type: ignore[union-attr]

    required_directions = {"OUTBOUND_CAUDAL", "RETURN_CRANIAL"}
    support_json: dict[str, dict[str, Any]] = {}
    for split in ("train_dev", "calibration_dev", "untouched_audit"):
        values = support[split]
        support_json[split] = {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in values.items()
        }
        if int(values["l4_frames"]) <= 0:
            failures.append(f"NO_L4_SUPPORT:{split}")
        if int(values["adjacent_negative_frames"]) <= 0:
            failures.append(f"NO_ADJACENT_NEGATIVE_SUPPORT:{split}")
        if len(values["l4_sessions"]) < 2:  # type: ignore[arg-type]
            failures.append(f"L4_SUPPORT_FEWER_THAN_TWO_SESSIONS:{split}")
        if len(values["adjacent_negative_sessions"]) < 2:  # type: ignore[arg-type]
            failures.append(f"NEGATIVE_SUPPORT_FEWER_THAN_TWO_SESSIONS:{split}")
        if not required_directions.issubset(values["l4_directions"]):  # type: ignore[arg-type]
            failures.append(f"L4_DIRECTION_SUPPORT_MISSING:{split}")
        if not required_directions.issubset(values["adjacent_negative_directions"]):  # type: ignore[arg-type]
            failures.append(f"NEGATIVE_DIRECTION_SUPPORT_MISSING:{split}")
    if total_mismatches:
        failures.append("RAW_TO_ACOUSTIC_PROJECTION_MISMATCH")

    output_dir = args.output_dir.resolve()
    if public_rows:
        write_jsonl(output_dir / "public_frames.jsonl", public_rows)
        write_jsonl(output_dir / "privileged_annotations.jsonl", privileged_rows)
    report = {
        **provenance,
        "protocol_id": protocol["protocol_id"],
        "status": "CAPTURE_PASS" if not failures else "CAPTURE_FAIL",
        "g0_decision": "DATA_READY_RECOGNITION_PENDING" if not failures else "BLOCKED",
        "g1_decision": "PASS" if not failures else "FAIL",
        "scientific_note": (
            "Positive/negative support makes supervised recognition testable; image-only "
            "L4 identifiability remains unproven until G2."
        ),
        "expected_session_count": len(expected_session_ids),
        "completed_session_count": len(session_summaries),
        "session_status_counts": dict(Counter(item.get("status") for item in session_summaries)),
        "support_by_split": support_json,
        "raw_to_acoustic_bone_mismatch_pixels": total_mismatches,
        "failures": sorted(set(failures)),
        "session_summaries": session_summaries,
    }
    write_json(output_dir / "capture_summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
