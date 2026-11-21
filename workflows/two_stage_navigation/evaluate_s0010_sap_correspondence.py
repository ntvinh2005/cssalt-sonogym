#!/usr/bin/env python3
"""Evaluate frozen SAP correspondence checkpoints without fitting or threshold selection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

import validate_s0010_sap_correspondence as validation


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_level(
    rows: list[dict[str, str]], estimator: Any, threshold: float
) -> tuple[dict[str, Any], np.ndarray]:
    labels = np.asarray(
        [row["comparison"] == "MATCHED" for row in rows], dtype=np.int64
    )
    scores = estimator.predict_proba(validation.feature_matrix(rows))[:, 1]
    summary = validation.rates(labels, scores, threshold)
    wins, total, ranking = validation.top_one_rate(rows, scores)
    by_comparison = validation.acceptance_by_comparison(rows, scores, threshold)
    distinct_pose_rate = by_comparison.get(
        "DISTINCT_POSE", {"acceptance_rate": 1.0}
    )["acceptance_rate"]
    gate_pass = bool(
        summary["matched_acceptance_rate"]
        >= validation.MINIMUM_MATCHED_ACCEPTANCE
        and summary["mismatched_acceptance_rate"]
        <= validation.MAXIMUM_MISMATCHED_ACCEPTANCE
        and distinct_pose_rate <= validation.MAXIMUM_MISMATCHED_ACCEPTANCE
        and ranking >= validation.MINIMUM_TOP_ONE_RANKING
    )
    return (
        {
            "calibration_gate_pass": gate_pass,
            "frozen_threshold": threshold,
            "calibration": summary,
            "calibration_top_one_wins": wins,
            "calibration_top_one_total": total,
            "calibration_top_one_rate": ranking,
            "calibration_acceptance_by_comparison": by_comparison,
        },
        scores,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset_dir.resolve()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    session_split = {row["session_id"]: row["split"] for row in protocol["sessions"]}
    input_path = dataset / "correspondence_controls.csv"
    all_rows = read_csv(input_path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for level in validation.LEVELS:
        rows = [
            row
            for row in all_rows
            if row["level"] == level
            and session_split.get(row["session_id"]) == args.split
        ]
        if not rows:
            raise ValueError(f"missing {args.split} correspondence rows for {level}")
        checkpoint = (
            args.checkpoint_dir / f"s0010_{level.lower()}_sap_correspondence.joblib"
        )
        bundle = joblib.load(checkpoint)
        if bundle.get("schema") != "s0010-sap-correspondence-development-v1":
            raise ValueError("unsupported correspondence checkpoint")
        if not bundle.get("evaluator_only") or tuple(bundle.get("features", ())) != tuple(
            validation.FEATURES
        ):
            raise ValueError("correspondence checkpoint contract mismatch")
        threshold = bundle.get("threshold")
        if threshold is None:
            raise ValueError(f"frozen correspondence threshold is absent for {level}")
        summary, scores = evaluate_level(rows, bundle["estimator"], float(threshold))
        summary.update(
            checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=validation.sha256(checkpoint),
        )
        results[level] = summary
        for row, score in zip(rows, scores):
            prediction_rows.append(
                {
                    "level": level,
                    "session_id": row["session_id"],
                    "frame_sequence": row["frame_sequence"],
                    "comparison": row["comparison"],
                    "projection_source": row["projection_source"],
                    "score": float(score),
                    "accepted": bool(score >= float(threshold)),
                }
            )
    prediction_path = output / "correspondence_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    report = {
        "schema": "s0010-sap-correspondence-development-v1",
        "mode": "FROZEN_CHECKPOINT_EVALUATION",
        "split": args.split,
        "input": str(input_path.resolve()),
        "input_sha256": validation.sha256(input_path),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": validation.sha256(args.protocol),
        "evaluator_only": True,
        "fit_called": False,
        "threshold_selected": False,
        "actor_feature_privileged_fields": [],
        "criteria": {
            "minimum_matched_acceptance_rate": validation.MINIMUM_MATCHED_ACCEPTANCE,
            "maximum_mismatched_acceptance_rate": validation.MAXIMUM_MISMATCHED_ACCEPTANCE,
            "minimum_top_one_rate": validation.MINIMUM_TOP_ONE_RANKING,
        },
        "levels": results,
    }
    (output / "correspondence_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if all(row["calibration_gate_pass"] for row in results.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
