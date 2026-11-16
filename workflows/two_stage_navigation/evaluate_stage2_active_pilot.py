#!/usr/bin/env python3
"""Evaluate a sealed Stage 2 active-acquisition actor result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from two_stage_navigation.stage2_active_pilot import (
    ACTIVE_PILOT_PROTOCOL_IDS,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--protocol", type=Path, required=True)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


FORBIDDEN = {
    "goal_cmd_pose",
    "position_error_mm",
    "orientation_error_deg",
    "l4_visible_fraction",
    "realized_pose_index",
    "target_pose",
    "evaluator_score",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(recursive_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(recursive_keys(nested))
    return keys


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    protocol_hash = file_sha256(args.protocol)
    expected_protocol_hash = args.protocol.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if protocol_hash != expected_protocol_hash:
        raise RuntimeError("protocol seal mismatch")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") not in ACTIVE_PILOT_PROTOCOL_IDS:
        raise RuntimeError("unexpected protocol")
    actor_path = args.run_dir / "results" / "actor_result.json"
    actor_hash = file_sha256(actor_path)
    actor_seal_path = actor_path.with_suffix(actor_path.suffix + ".sha256")
    expected_actor_hash = actor_seal_path.read_text(encoding="utf-8").split()[0]
    if actor_hash != expected_actor_hash:
        raise RuntimeError("actor result seal mismatch")
    actor = json.loads(actor_path.read_text(encoding="utf-8"))
    leaks = FORBIDDEN & recursive_keys(actor)
    if leaks:
        raise RuntimeError(f"privileged fields entered actor result: {sorted(leaks)}")
    annotations = read_jsonl(args.run_dir / "privileged" / "annotations.jsonl")
    by_sequence = {int(row["frame_sequence"]): row for row in annotations}
    confirmation_sequences = [int(value) for value in actor["fresh_confirmation_frame_sequences"]]
    final = None if not confirmation_sequences else by_sequence[confirmation_sequences[-1]]
    gate = protocol["evaluation_gate"]
    geometric_success = bool(
        final is not None
        and float(final["position_error_mm"]) <= float(gate["maximum_position_error_mm"])
        and float(final["orientation_error_deg"]) <= float(gate["maximum_orientation_error_deg"])
    )
    image_success = bool(
        final is not None
        and float(final["l4_visible_fraction"]) >= float(gate["minimum_l4_visible_fraction"])
    )
    return_success = bool(
        actor["return_position_error_mm"] is not None
        and float(actor["return_position_error_mm"]) <= float(gate["maximum_return_position_error_mm"])
        and float(actor["return_orientation_error_deg"]) <= float(gate["maximum_return_orientation_error_deg"])
    )
    passed = bool(
        actor["outcome"] == "ACCEPT"
        and actor["supporting_viewpoint_count"] >= int(gate["minimum_supporting_viewpoints"])
        and actor["fresh_confirmation_passed"]
        and return_success
        and geometric_success
        and image_success
    )
    report = {
        "schema": "sonogym-stage2-active-pilot-evaluation-v1",
        "decision": "DEVELOPMENT_PASS" if passed else "DEVELOPMENT_NO_GO",
        "protocol_sha256": protocol_hash,
        "actor_result_sha256": actor_hash,
        "actor_outcome": actor["outcome"],
        "supporting_viewpoint_count": actor["supporting_viewpoint_count"],
        "fresh_confirmation_passed": actor["fresh_confirmation_passed"],
        "return_success": return_success,
        "geometric_success": geometric_success,
        "image_success": image_success,
        "final_position_error_mm": None if final is None else final["position_error_mm"],
        "final_orientation_error_deg": None if final is None else final["orientation_error_deg"],
        "final_l4_visible_fraction": None if final is None else final["l4_visible_fraction"],
        "fixed_anatomy_only": True,
        "independent_confirmation": False,
        "full_stage2_closed_loop": "DEVELOPMENT_PASS" if passed else "NOT_ESTABLISHED",
    }
    write_json(args.output, report)
    digest = file_sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
