#!/usr/bin/env python3
"""Validate separate public and privileged SonoGym capture manifests.

This workflow never merges privileged labels into actor records. It validates a one-to-one
join by immutable frame identity and writes only aggregate provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--public-jsonl", type=Path, required=True)
    parser.add_argument("--privileged-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    assignments = [
        {"session_id": session["session_id"], "split": session["split"]}
        for session in protocol["sessions"]
    ]
    session_to_split = validate_split_assignments(assignments)
    public_rows = validate_public_rows(read_jsonl(args.public_jsonl), session_to_split)
    privileged_rows = validate_privileged_rows(read_jsonl(args.privileged_jsonl))
    summary = dataset_summary(public_rows, privileged_rows, file_sha256(args.protocol))
    summary["protocol_id"] = protocol["protocol_id"]
    summary["public_manifest"] = str(args.public_jsonl.resolve())
    summary["privileged_manifest"] = str(args.privileged_jsonl.resolve())
    write_json_atomic(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
