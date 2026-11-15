#!/usr/bin/env python3
"""Validate and summarize the six-session Stage 2 development capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
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


def dataset_gate_passes(
    gate: dict[str, Any] | None,
    split_positive_hashes: dict[str, set[str]],
    split_positive_trajectories: dict[str, set[str]],
) -> bool:
    if gate is None:
        return True
    return bool(
        all(
            len(split_positive_hashes.get(split, set())) >= int(minimum)
            for split, minimum in gate["minimum_unique_positive_hashes"].items()
        )
        and all(
            len(split_positive_trajectories.get(split, set())) >= int(minimum)
            for split, minimum in gate["minimum_positive_trajectories"].items()
        )
    )


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
    split_hashes: dict[str, set[str]] = defaultdict(set)
    split_positive_hashes: dict[str, set[str]] = defaultdict(set)
    split_positive_trajectories: dict[str, set[str]] = defaultdict(set)
    split_frame_counts: dict[str, int] = defaultdict(int)
    split_positive_frame_counts: dict[str, int] = defaultdict(int)
    expected_frames = int(protocol["sampling"]["expected_frames_per_session"])
    for session in protocol["sessions"]:
        session_id = str(session["session_id"])
        root = args.capture_root / "sessions" / session_id
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "PASSED":
            raise RuntimeError(f"failed Stage 2 dataset session {session_id}")
        public = read_jsonl(root / "public" / "frames.jsonl")
        privileged = read_jsonl(root / "privileged" / "annotations.jsonl")
        images = np.load(root / "public" / "ultrasound.npz")["ultrasound"]
        if not len(public) == len(privileged) == len(images) == expected_frames:
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
            split = str(actor["split"])
            trajectory_id = str(actor["trajectory_id"])
            frame_hash = str(actor["frame_sha256"])
            trajectory_ids.add(trajectory_id)
            public_hashes.add(frame_hash)
            split_hashes[split].add(frame_hash)
            split_frame_counts[split] += 1
            if int(annotation["validity_label"]) == 1:
                split_positive_hashes[split].add(frame_hash)
                split_positive_trajectories[split].add(trajectory_id)
                split_positive_frame_counts[split] += 1
        summaries.append(summary)
    expected_trajectories = sum(int(row.get("trajectory_count", 1)) for row in summaries)
    if len(trajectory_ids) != expected_trajectories:
        raise RuntimeError("unexpected Stage 2 trajectory count")
    split_names = tuple(split_hashes)
    if any(
        split_hashes[left] & split_hashes[right]
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    ):
        raise RuntimeError("duplicate image hash crosses a Stage 2 split")
    gate_passed = dataset_gate_passes(
        protocol.get("dataset_gate"),
        split_positive_hashes,
        split_positive_trajectories,
    )
    declared_splits = sorted({str(row["split"]) for row in protocol["sessions"]})
    report = {
        "schema": "sonogym-stage2-dataset-summary-v1",
        "decision": "DATA_READY" if gate_passed else "DATA_NO_GO",
        "protocol_sha256": protocol_hash,
        "session_count": len(summaries),
        "trajectory_count": len(trajectory_ids),
        "frame_count": sum(int(row["frame_count"]) for row in summaries),
        "valid_frame_count": sum(int(row["valid_frame_count"]) for row in summaries),
        "distinct_frame_hash_count": len(public_hashes),
        "public_privileged_one_to_one": True,
        "split_frame_counts": dict(split_frame_counts),
        "split_positive_frame_counts": dict(split_positive_frame_counts),
        "split_unique_positive_hash_counts": {
            split: len(split_positive_hashes.get(split, set()))
            for split in declared_splits
        },
        "split_positive_trajectory_counts": {
            split: len(split_positive_trajectories.get(split, set()))
            for split in declared_splits
        },
        "dataset_gate_requirements": protocol.get("dataset_gate"),
        "dataset_gate_passed": gate_passed,
        "sessions": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if gate_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
