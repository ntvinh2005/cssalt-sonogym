#!/usr/bin/env python3
"""Validate and summarize the six-session Stage 2 development capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_hash = file_sha256(args.protocol)
    expected_hash = args.protocol.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if protocol_hash != expected_hash:
        raise RuntimeError("Stage 2 protocol seal mismatch")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    summaries = []
    public_hashes: set[str] = set()
    trajectory_ids: set[str] = set()
    for session in protocol["sessions"]:
        session_id = str(session["session_id"])
        root = args.capture_root / "sessions" / session_id
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "PASSED":
            raise RuntimeError(f"failed Stage 2 dataset session {session_id}")
        public = read_jsonl(root / "public" / "frames.jsonl")
        privileged = read_jsonl(root / "privileged" / "annotations.jsonl")
        images = np.load(root / "public" / "ultrasound.npz")["ultrasound"]
        if not len(public) == len(privileged) == len(images) == 75:
            raise RuntimeError(f"incomplete Stage 2 dataset session {session_id}")
        for actor, annotation in zip(public, privileged):
            actor_key = (
                actor["session_id"],
                int(actor["frame_sequence"]),
                actor["frame_sha256"],
            )
            annotation_key = (
                annotation["session_id"],
                int(annotation["frame_sequence"]),
                annotation["frame_sha256"],
            )
            if actor_key != annotation_key:
                raise RuntimeError(f"public/privileged join mismatch in {session_id}")
            if actor.get("trajectory_id") != annotation.get("trajectory_id"):
                raise RuntimeError(f"trajectory join mismatch in {session_id}")
            trajectory_ids.add(str(actor["trajectory_id"]))
            public_hashes.add(str(actor["frame_sha256"]))
        summaries.append(summary)
    expected_trajectories = sum(int(row.get("trajectory_count", 1)) for row in summaries)
    if len(trajectory_ids) != expected_trajectories:
        raise RuntimeError("unexpected Stage 2 trajectory count")
    report = {
        "schema": "sonogym-stage2-dataset-summary-v1",
        "decision": "DATA_READY",
        "protocol_sha256": protocol_hash,
        "session_count": len(summaries),
        "trajectory_count": len(trajectory_ids),
        "frame_count": sum(int(row["frame_count"]) for row in summaries),
        "valid_frame_count": sum(int(row["valid_frame_count"]) for row in summaries),
        "distinct_frame_hash_count": len(public_hashes),
        "public_privileged_one_to_one": True,
        "sessions": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
