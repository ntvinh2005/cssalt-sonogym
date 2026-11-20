#!/usr/bin/env python3
"""Preflight or fit the bounded level-conditioned SAP ExtraTrees baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

LEVELS = ("L4", "L5")
POSITIVE = "SAP_RELATED_CORTICAL_INTERFACE"
NEGATIVE = "OTHER_OR_BACKGROUND"
LABEL_ID = {NEGATIVE: 0, POSITIVE: 1}
MINIMUM = {
    "train": {POSITIVE: 30, NEGATIVE: 30},
    "calibration": {POSITIVE: 10, NEGATIVE: 10},
}
N_ESTIMATORS = 240
RANDOM_SEED = 20261121
MINIMUM_PRECISION = 0.80
MINIMUM_RECALL = 0.80
MAXIMUM_FALSE_POSITIVE_RATE = 0.20


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = scores >= threshold
    positive = labels == 1
    negative = labels == 0
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & negative))
    fn = int(np.count_nonzero(~predicted & positive))
    tn = int(np.count_nonzero(~predicted & negative))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "gate_pass": bool(
            precision >= MINIMUM_PRECISION
            and recall >= MINIMUM_RECALL
            and fpr <= MAXIMUM_FALSE_POSITIVE_RATE
        ),
    }


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float | None, list[dict[str, Any]]]:
    rows = [metrics(labels, scores, float(value)) for value in np.linspace(0.05, 0.95, 19)]
    passing = [row for row in rows if row["gate_pass"]]
    if not passing:
        return None, rows
    selected = max(
        passing,
        key=lambda row: (row["f1"], row["recall"], row["precision"], row["threshold"]),
    )
    return float(selected["threshold"]), rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument(
        "--level", action="append", choices=LEVELS, dest="levels",
        help="Level to preflight or fit; repeat to select both (default: both).",
    )
    args = parser.parse_args()
    requested_levels = tuple(args.levels or LEVELS)
    dataset = args.dataset_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((dataset / "readiness_report.json").read_text())
    if report.get("phase") != "development":
        raise ValueError("training requires a development dataset")
    rows = read_csv(dataset / "sample_manifest.csv")
    arrays = np.load(dataset / "dataset_arrays.npz")
    features = arrays["features"]
    if features.ndim != 2 or features.shape[1] != 774 or not np.all(np.isfinite(features)):
        raise ValueError("invalid production feature matrix")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate sample IDs")
    if any(row["training_eligible"].lower() != "true" for row in rows):
        raise ValueError("sample manifest contains an ineligible row")
    if any(row["provisional_label"] not in LABEL_ID for row in rows):
        raise ValueError("ambiguous labels entered the binary manifest")
    for row in rows:
        index = int(row["feature_index"])
        if index < 0 or index >= len(features):
            raise ValueError("invalid feature index")
    train_hashes = {row["frame_sha256"] for row in rows if row["split"] == "train"}
    calibration_hashes = {
        row["frame_sha256"] for row in rows if row["split"] == "calibration"
    }
    if train_hashes & calibration_hashes:
        raise ValueError("exact image leakage across train and calibration")
    train_sessions = {row["session_id"] for row in rows if row["split"] == "train"}
    calibration_sessions = {
        row["session_id"] for row in rows if row["split"] == "calibration"
    }
    if len(train_sessions) < 3 or len(calibration_sessions) < 1:
        raise ValueError("insufficient session-disjoint development splits")

    level_results: dict[str, Any] = {}
    blocked = False
    fitted = False
    for level in requested_levels:
        level_rows = [row for row in rows if row["conditioned_level"] == level]
        deficits: list[str] = []
        if report["readiness"][level]["dataset_status"] != "READY":
            deficits.extend(report["readiness"][level]["deficits"])
        for split, requirements in MINIMUM.items():
            for label, minimum in requirements.items():
                count = len(
                    {
                        row["frame_sha256"]
                        for row in level_rows
                        if row["split"] == split and row["provisional_label"] == label
                    }
                )
                if count < minimum:
                    deficits.append(f"{split}:{label}={count}<{minimum}")
        if deficits:
            blocked = True
            level_results[level] = {"status": "SAP_TRAINING_BLOCKED_DATA", "deficits": sorted(set(deficits))}
            continue
        partitions: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, str]]]] = {}
        for split in ("train", "calibration"):
            part_rows = [row for row in level_rows if row["split"] == split]
            matrix = np.asarray([features[int(row["feature_index"])] for row in part_rows])
            labels = np.asarray([LABEL_ID[row["provisional_label"]] for row in part_rows], dtype=np.int64)
            partitions[split] = (matrix, labels, part_rows)
        estimator = ExtraTreesClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=4,
        )
        result: dict[str, Any] = {
            "status": "TRAINING_READY" if not args.fit else "FIT_PENDING",
            "deficits": [],
            "train_sample_ids": [row["sample_id"] for row in partitions["train"][2]],
            "calibration_sample_ids": [row["sample_id"] for row in partitions["calibration"][2]],
            "development_evaluation_used": False,
        }
        if args.fit:
            estimator.fit(partitions["train"][0], partitions["train"][1])
            fitted = True
            calibration_scores = estimator.predict_proba(partitions["calibration"][0])[:, list(estimator.classes_).index(1)]
            threshold, diagnostics = select_threshold(partitions["calibration"][1], calibration_scores)
            result["calibration_thresholds"] = diagnostics
            if threshold is None:
                result["status"] = "SAP_TRAINING_INCONCLUSIVE"
                result["deficits"] = ["no calibration threshold met frozen precision/recall/FPR gates"]
                blocked = True
            else:
                selected_metrics = metrics(partitions["calibration"][1], calibration_scores, threshold)
                bundle = {
                    "schema": "s0010-sap-level-conditioned-extratrees-v1",
                    "level": level,
                    "estimator": estimator,
                    "threshold": threshold,
                    "feature_width": int(features.shape[1]),
                    "patch_spec": report["preprocessing"],
                    "label_ids": LABEL_ID,
                    "dataset_manifest_sha256": file_sha256(dataset / "sample_manifest.csv"),
                    "dataset_arrays_sha256": file_sha256(dataset / "dataset_arrays.npz"),
                    "random_seed": RANDOM_SEED,
                }
                checkpoint = output / f"s0010_{level.lower()}_sap_classifier_v1.joblib"
                joblib.dump(bundle, checkpoint)
                reloaded = joblib.load(checkpoint)
                reloaded_scores = reloaded["estimator"].predict_proba(partitions["calibration"][0])[:, list(reloaded["estimator"].classes_).index(1)]
                if not np.array_equal(calibration_scores, reloaded_scores):
                    raise AssertionError("checkpoint save/reload changed predictions")
                result.update(
                    status="TRAINED_CALIBRATED_HOLDOUT_UNOPENED",
                    threshold=threshold,
                    calibration_metrics=selected_metrics,
                    checkpoint=str(checkpoint),
                    checkpoint_sha256=file_sha256(checkpoint),
                )
        level_results[level] = result

    result = {
        "schema": "s0010-sap-training-preflight-v1",
        "mode": "FIT" if args.fit else "NO_FIT",
        "fit_called": fitted,
        "dataset_manifest_sha256": file_sha256(dataset / "sample_manifest.csv"),
        "dataset_arrays_sha256": file_sha256(dataset / "dataset_arrays.npz"),
        "frozen_gates": {
            "minimum_precision": MINIMUM_PRECISION,
            "minimum_recall": MINIMUM_RECALL,
            "maximum_false_positive_rate": MAXIMUM_FALSE_POSITIVE_RATE,
        },
        "levels": level_results,
        "requested_levels": list(requested_levels),
        "status": (
            "ENGINEERING_PREFLIGHT_ONLY_BLOCKED"
            if blocked and not args.fit
            else "TRAINING_BLOCKED"
            if blocked
            else "TRAINING_READY_FOR_EXECUTION"
            if not args.fit
            else "TRAINED_CALIBRATED_HOLDOUT_UNOPENED"
        ),
    }
    (output / ("training_result.json" if args.fit else "preflight_result.json")).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if not blocked else 3


if __name__ == "__main__":
    raise SystemExit(main())
