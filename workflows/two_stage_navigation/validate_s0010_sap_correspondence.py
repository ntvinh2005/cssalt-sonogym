#!/usr/bin/env python3
"""Fit a development-only pair verifier and evaluate session-separated calibration pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LEVELS = ("L4", "L5")
FEATURES = (
    "surface_columns",
    "aligned_columns",
    "aligned_fraction",
    "longest_aligned_run_columns",
    "median_distance_pixels",
    "local_gradient_median",
    "local_contrast_median",
    "posterior_shadow_median",
    "maximum_track_jump_pixels",
)
RANDOM_SEED = 20261120
MINIMUM_MATCHED_ACCEPTANCE = 0.50
MAXIMUM_MISMATCHED_ACCEPTANCE = 0.10
MINIMUM_TOP_ONE_RANKING = 0.70


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def feature_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    matrix = np.asarray([[float(row[name]) for name in FEATURES] for row in rows])
    return np.nan_to_num(matrix, nan=0.0, posinf=200.0, neginf=-200.0)


def rates(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    matched = labels == 1
    mismatched = labels == 0
    return {
        "threshold": float(threshold),
        "matched_acceptance_rate": float(np.mean(scores[matched] >= threshold)),
        "mismatched_acceptance_rate": float(np.mean(scores[mismatched] >= threshold)),
        "matched_count": int(np.count_nonzero(matched)),
        "mismatched_count": int(np.count_nonzero(mismatched)),
    }


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float | None, list[dict[str, Any]]]:
    rows = [rates(labels, scores, float(value)) for value in np.linspace(0.05, 0.95, 181)]
    passing = [
        row
        for row in rows
        if row["matched_acceptance_rate"] >= MINIMUM_MATCHED_ACCEPTANCE
        and row["mismatched_acceptance_rate"] <= MAXIMUM_MISMATCHED_ACCEPTANCE
    ]
    if not passing:
        return None, rows
    selected = max(
        passing,
        key=lambda row: (
            row["matched_acceptance_rate"] - row["mismatched_acceptance_rate"],
            row["matched_acceptance_rate"],
            row["threshold"],
        ),
    )
    return float(selected["threshold"]), rows


def top_one_rate(
    rows: list[dict[str, str]], scores: np.ndarray
) -> tuple[int, int, float]:
    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault((row["session_id"], row["frame_sequence"]), []).append(index)
    wins = 0
    for indices in groups.values():
        matched = [index for index in indices if rows[index]["comparison"] == "MATCHED"]
        if len(matched) != 1:
            raise ValueError("each frame requires exactly one matched comparison")
        wins += int(max(indices, key=lambda index: scores[index]) == matched[0])
    total = len(groups)
    return wins, total, wins / total if total else 0.0


def acceptance_by_comparison(
    rows: list[dict[str, str]], scores: np.ndarray, threshold: float
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in sorted({row["comparison"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["comparison"] == name]
        accepted = int(np.count_nonzero(scores[indices] >= threshold))
        result[name] = {
            "accepted": accepted,
            "total": len(indices),
            "acceptance_rate": accepted / len(indices) if indices else 0.0,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
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
    for level in LEVELS:
        rows = [row for row in all_rows if row["level"] == level]
        if any(row["session_id"] not in session_split for row in rows):
            raise ValueError("correspondence row is absent from the sealed protocol")
        train_rows = [row for row in rows if session_split[row["session_id"]] == "train"]
        calibration_rows = [
            row for row in rows if session_split[row["session_id"]] == "calibration"
        ]
        if not train_rows or not calibration_rows:
            raise ValueError(f"missing session-separated rows for {level}")
        train_labels = np.asarray(
            [row["comparison"] == "MATCHED" for row in train_rows], dtype=np.int64
        )
        calibration_labels = np.asarray(
            [row["comparison"] == "MATCHED" for row in calibration_rows], dtype=np.int64
        )
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
        )
        estimator.fit(feature_matrix(train_rows), train_labels)
        train_scores = estimator.predict_proba(feature_matrix(train_rows))[:, 1]
        calibration_scores = estimator.predict_proba(feature_matrix(calibration_rows))[:, 1]
        threshold, threshold_rows = select_threshold(train_labels, train_scores)
        train_result = rates(train_labels, train_scores, threshold or 1.0)
        calibration_result = rates(
            calibration_labels, calibration_scores, threshold or 1.0
        )
        wins, total, ranking = top_one_rate(calibration_rows, calibration_scores)
        by_comparison = acceptance_by_comparison(
            calibration_rows, calibration_scores, threshold or 1.0
        )
        distinct_pose_rate = by_comparison.get(
            "DISTINCT_POSE", {"acceptance_rate": 1.0}
        )["acceptance_rate"]
        gate_pass = bool(
            threshold is not None
            and calibration_result["matched_acceptance_rate"]
            >= MINIMUM_MATCHED_ACCEPTANCE
            and calibration_result["mismatched_acceptance_rate"]
            <= MAXIMUM_MISMATCHED_ACCEPTANCE
            and distinct_pose_rate <= MAXIMUM_MISMATCHED_ACCEPTANCE
            and ranking >= MINIMUM_TOP_ONE_RANKING
        )
        checkpoint = output / f"s0010_{level.lower()}_sap_correspondence.joblib"
        joblib.dump(
            {
                "schema": "s0010-sap-correspondence-development-v1",
                "level": level,
                "features": FEATURES,
                "estimator": estimator,
                "threshold": threshold,
                "input_sha256": sha256(input_path),
                "protocol_sha256": sha256(args.protocol),
                "random_seed": RANDOM_SEED,
                "evaluator_only": True,
            },
            checkpoint,
        )
        reloaded = joblib.load(checkpoint)
        if not np.array_equal(
            calibration_scores,
            reloaded["estimator"].predict_proba(feature_matrix(calibration_rows))[:, 1],
        ):
            raise AssertionError("checkpoint reload changed correspondence scores")
        results[level] = {
            "calibration_gate_pass": gate_pass,
            "threshold_selected_on_train": threshold,
            "train": train_result,
            "calibration": calibration_result,
            "calibration_top_one_wins": wins,
            "calibration_top_one_total": total,
            "calibration_top_one_rate": ranking,
            "calibration_acceptance_by_comparison": by_comparison,
            "threshold_candidates": threshold_rows,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
        }
    report = {
        "schema": "s0010-sap-correspondence-development-v1",
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256(args.protocol),
        "evaluator_only": True,
        "actor_feature_privileged_fields": [],
        "criteria": {
            "minimum_matched_acceptance_rate": MINIMUM_MATCHED_ACCEPTANCE,
            "maximum_mismatched_acceptance_rate": MAXIMUM_MISMATCHED_ACCEPTANCE,
            "minimum_top_one_rate": MINIMUM_TOP_ONE_RANKING,
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
