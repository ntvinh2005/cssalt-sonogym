#!/usr/bin/env python3
"""Aggregate sealed Stage 2 evaluator reports across protocol-declared new starts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from two_stage_navigation.stage2_active_pilot import summarize_confirmation_runs


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--protocol", type=Path, required=True)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    expected = args.protocol.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    observed = file_sha256(args.protocol)
    if observed != expected:
        raise RuntimeError("protocol seal mismatch")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    starts = protocol["execution"]["confirmation_starts"]
    rows = []
    for start in starts:
        start_id = str(start["start_id"])
        report_path = args.run_dir / "starts" / start_id / "results" / "evaluation_report.json"
        if not report_path.is_file():
            raise RuntimeError(f"missing evaluator report for {start_id}")
        row = json.loads(report_path.read_text(encoding="utf-8"))
        if row["protocol_sha256"] != observed:
            raise RuntimeError(f"protocol mismatch for {start_id}")
        if row["start_id"] != start_id or int(row["seed"]) != int(start["seed"]):
            raise RuntimeError(f"execution identity mismatch for {start_id}")
        rows.append(row)
    aggregate = summarize_confirmation_runs(
        rows, required_count=int(protocol["confirmation"]["required_pass_count"])
    )
    poses = np.asarray([row["initial_reset_pose_index"] for row in rows], dtype=np.float64)
    pairwise_translation_mm = []
    pairwise_orientation_deg = []
    label_resolution_mm = float(protocol["geometry"]["label_resolution_mm"])
    for left in range(len(poses)):
        for right in range(left + 1, len(poses)):
            pairwise_translation_mm.append(
                float(np.linalg.norm(poses[left, :2] - poses[right, :2]) * label_resolution_mm)
            )
            delta = (poses[left, 2] - poses[right, 2] + np.pi) % (2.0 * np.pi) - np.pi
            pairwise_orientation_deg.append(abs(float(np.degrees(delta))))
    result = {
        "schema": "sonogym-stage2-new-start-confirmation-result-v1",
        "protocol_sha256": observed,
        **aggregate,
        "patient_id": protocol["scope"]["patient_id"],
        "fixed_anatomy_only": True,
        "independent_start_confirmation": True,
        "patient_generalization": False,
        "minimum_pairwise_initial_translation_mm": min(pairwise_translation_mm),
        "maximum_pairwise_initial_translation_mm": max(pairwise_translation_mm),
        "minimum_pairwise_initial_orientation_deg": min(pairwise_orientation_deg),
        "maximum_pairwise_initial_orientation_deg": max(pairwise_orientation_deg),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    digest = file_sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] == "CONFIRMATION_PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
