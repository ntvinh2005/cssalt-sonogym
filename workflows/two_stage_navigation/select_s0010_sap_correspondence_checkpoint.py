#!/usr/bin/env python3
"""Select one correspondence checkpoint using development calibration only."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


LEVELS = ("L4", "L5")


def selection_key(result: dict[str, Any]) -> tuple[float, ...]:
    calibration = result["calibration"]
    return (
        float(result["calibration_gate_pass"]),
        calibration["matched_acceptance_rate"]
        - calibration["mismatched_acceptance_rate"],
        result["calibration_top_one_rate"],
        calibration["matched_acceptance_rate"],
        -calibration["mismatched_acceptance_rate"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.candidate]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected: dict[str, Any] = {}
    for level in LEVELS:
        ranked = sorted(
            (
                (selection_key(report["levels"][level]), report)
                for report in reports
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        _, report = ranked[0]
        source = Path(report["levels"][level]["checkpoint"])
        destination = output / f"s0010_{level.lower()}_sap_correspondence.joblib"
        shutil.copy2(source, destination)
        selected[level] = {
            "estimator_name": report["estimator_name"],
            "feature_set": report["feature_set"],
            "development_calibration": report["levels"][level],
            "checkpoint": str(destination),
        }
    result = {
        "schema": "s0010-sap-correspondence-development-v1",
        "selection_split": "calibration",
        "evaluation_opened": False,
        "candidates": [str(path.resolve()) for path in args.candidate],
        "levels": selected,
    }
    (output / "selection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
