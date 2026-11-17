#!/usr/bin/env python3
"""Validate or fit the bounded CPU ExtraTrees TP_SURFACE_CORE patch pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

FEATURE_PATH = Path(__file__).resolve().parents[2] / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/stage1_features.py"
FEATURE_SPEC = importlib.util.spec_from_file_location("tp_surface_core_features", FEATURE_PATH)
if FEATURE_SPEC is None or FEATURE_SPEC.loader is None:
    raise RuntimeError("cannot load CPU patch features")
FEATURES = importlib.util.module_from_spec(FEATURE_SPEC)
sys.modules[FEATURE_SPEC.name] = FEATURES
FEATURE_SPEC.loader.exec_module(FEATURES)
UNITY_REFERENCE_PATCH_SPEC = FEATURES.UNITY_REFERENCE_PATCH_SPEC
extract_patch_feature = FEATURES.extract_patch_feature

SPLITS = ("train", "calibration", "development_evaluation")
CLASS_NAMES = {0: "BACKGROUND_OR_OTHER_STRUCTURE", 1: "L4_RIGHT_TP_SURFACE_CORE", 2: "L5_RIGHT_TP_SURFACE_CORE"}
MINIMUM_UNIQUE_IMAGES = {
    class_id: {"train": 40, "calibration": 10, "development_evaluation": 10}
    for class_id in CLASS_NAMES
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-fit", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ultrasound_sha256(image: np.ndarray) -> str:
    array = np.asarray(image, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def partition_eligible_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    partitions = {split: [] for split in SPLITS}
    for row in rows:
        split = str(row["split"])
        if split not in partitions:
            raise ValueError(f"eligible sample has unsupported split: {split}")
        partitions[split].append(row)
    identifiers = {split: {str(row["sample_id"]) for row in values} for split, values in partitions.items()}
    if any(
        identifiers[left].intersection(identifiers[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    ):
        raise ValueError("sample identifiers cross split boundaries")
    return partitions


def class_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_count": int(len(targets)),
        "accuracy": float(np.mean(targets == predictions)) if len(targets) else None,
        "per_class": {},
    }
    for class_id, name in CLASS_NAMES.items():
        tp = int(np.count_nonzero((targets == class_id) & (predictions == class_id)))
        fp = int(np.count_nonzero((targets != class_id) & (predictions == class_id)))
        fn = int(np.count_nonzero((targets == class_id) & (predictions != class_id)))
        result["per_class"][name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": None if tp + fp == 0 else tp / (tp + fp),
            "recall": None if tp + fn == 0 else tp / (tp + fn),
        }
    return result


def calibrate_thresholds(targets: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> tuple[dict[int, float], dict[str, Any]]:
    thresholds: dict[int, float] = {}
    diagnostics: dict[str, Any] = {}
    for class_id in (1, 2):
        class_column = int(np.flatnonzero(classes == class_id)[0])
        scores = probabilities[:, class_column]
        candidates = []
        for threshold in np.linspace(0.50, 0.95, 10):
            accepted = scores >= threshold
            tp = int(np.count_nonzero(accepted & (targets == class_id)))
            fp = int(np.count_nonzero(accepted & (targets != class_id)))
            fn = int(np.count_nonzero((~accepted) & (targets == class_id)))
            candidates.append(
                {
                    "threshold": float(threshold),
                    "precision": None if tp + fp == 0 else tp / (tp + fp),
                    "recall": None if tp + fn == 0 else tp / (tp + fn),
                }
            )
        passing = [item for item in candidates if item["precision"] is not None and item["precision"] >= 0.90]
        selected = max(
            passing or candidates,
            key=lambda item: (-1.0 if item["recall"] is None else item["recall"], -item["threshold"]),
        )
        thresholds[class_id] = float(selected["threshold"])
        diagnostics[CLASS_NAMES[class_id]] = {"selected": selected, "grid": candidates}
    return thresholds, diagnostics


def apply_thresholds(probabilities: np.ndarray, classes: np.ndarray, thresholds: dict[int, float]) -> np.ndarray:
    predictions = np.zeros(len(probabilities), dtype=np.int64)
    for index, row in enumerate(probabilities):
        best_column = int(np.argmax(row))
        class_id = int(classes[best_column])
        if class_id in thresholds and row[best_column] >= thresholds[class_id]:
            predictions[index] = class_id
    return predictions


def main() -> int:
    args = parse_args()
    dataset = args.dataset_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(dataset / "candidate_sample_manifest.csv")
    if not rows:
        raise RuntimeError("candidate manifest is empty")
    dataset_summary = json.loads((dataset / "dataset_summary.json").read_text())
    failures: list[str] = []
    if dataset_summary.get("status") != "ENGINEERING_READY_LABEL_REVIEW_PENDING":
        failures.append(f"dataset status is {dataset_summary.get('status')}")

    archives: dict[Path, np.ndarray] = {}
    feature_by_id: dict[str, np.ndarray] = {}
    valid_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("proposal_source") != "ACTOR_VISIBLE_BRIGHT_INTERFACE":
            failures.append(f"unsupported proposal source: {row.get('sample_id')}")
            continue
        path = Path(row["image_archive"])
        if not path.is_file():
            failures.append(f"missing archive: {path}")
            continue
        if path not in archives:
            archives[path] = np.load(path)["ultrasound"]
        image = archives[path][int(row["image_archive_index"])]
        if image.shape != (200, 150) or ultrasound_sha256(image) != row["frame_sha256"]:
            failures.append(f"image shape or hash mismatch: {row['sample_id']}")
            continue
        feature = extract_patch_feature(image, float(row["patch_row"]), float(row["patch_column"]), UNITY_REFERENCE_PATCH_SPEC)
        if feature is None or not np.all(np.isfinite(feature)):
            failures.append(f"invalid patch feature: {row['sample_id']}")
            continue
        feature_by_id[row["sample_id"]] = feature
        valid_rows.append(row)
    if not valid_rows:
        failures.append("no resolvable actor-visible patch proposals")

    split_hashes = {split: {row["frame_sha256"] for row in valid_rows if row["split"] == split} for split in SPLITS}
    cross_split_duplicates = sorted(
        set().union(
            *(split_hashes[left].intersection(split_hashes[right]) for index, left in enumerate(SPLITS) for right in SPLITS[index + 1 :])
        )
    )
    if cross_split_duplicates:
        failures.append(f"cross-split exact image duplicates: {len(cross_split_duplicates)}")

    eligible = [row for row in valid_rows if row["training_eligible"].lower() == "true"]
    for row in eligible:
        class_id = int(row["class_id"])
        if class_id not in CLASS_NAMES:
            failures.append(f"unsupported class id: {row['sample_id']}")
        elif class_id == 0:
            if row["training_label"] != "VERIFIED_NEGATIVE" or row["semantic_identity"] != "REJECTED_AS_TP":
                failures.append(f"unsupported negative label: {row['sample_id']}")
        elif (
            row["training_label"] != "VERIFIED_POSITIVE"
            or row["semantic_identity"] != "SUPPORTED"
            or row["acoustic_visibility"] != "VISIBLE"
            or row["anatomical_approval"] != "APPROVED"
        ):
            failures.append(f"unsupported positive label: {row['sample_id']}")

    unique_counts = {
        CLASS_NAMES[class_id]: {
            split: len({row["frame_sha256"] for row in eligible if int(row["class_id"]) == class_id and row["split"] == split})
            for split in SPLITS
        }
        for class_id in CLASS_NAMES
    }
    for class_id, minimums in MINIMUM_UNIQUE_IMAGES.items():
        for split, minimum in minimums.items():
            count = unique_counts[CLASS_NAMES[class_id]][split]
            if count < minimum:
                failures.append(f"{CLASS_NAMES[class_id]} {split}: {count} unique images < {minimum}")

    partitions = partition_eligible_rows(eligible)
    estimator = ExtraTreesClassifier(n_estimators=240, class_weight="balanced", random_state=20261117, n_jobs=4)
    report: dict[str, Any] = {
        "schema": "sonogym-tp-surface-core-training-result-v2",
        "mode": "NO_FIT" if args.no_fit else "FIT",
        "candidate_rows": len(rows),
        "resolved_feature_rows": len(valid_rows),
        "feature_width": UNITY_REFERENCE_PATCH_SPEC.feature_width,
        "eligible_unique_image_counts": unique_counts,
        "cross_split_duplicate_count": len(cross_split_duplicates),
        "fit_sample_ids": [row["sample_id"] for row in partitions["train"]],
        "calibration_sample_ids": [row["sample_id"] for row in partitions["calibration"]],
        "development_sample_ids": [row["sample_id"] for row in partitions["development_evaluation"]],
        "estimator": estimator.get_params(),
        "blocking_conditions": sorted(set(failures)),
        "fit_called": False,
        "checkpoint_saved": False,
        "checkpoint_reload_verified": False,
    }
    if args.no_fit:
        report["status"] = "ENGINEERING_PREFLIGHT_ONLY" if failures else "TRAINING_READY_FOR_APPROVAL"
    elif failures:
        report["status"] = "TRAINING_BLOCKED"
    else:
        def matrix(split: str) -> tuple[np.ndarray, np.ndarray]:
            selected = partitions[split]
            return np.stack([feature_by_id[row["sample_id"]] for row in selected]), np.asarray([int(row["class_id"]) for row in selected], dtype=np.int64)

        train_x, train_y = matrix("train")
        calibration_x, calibration_y = matrix("calibration")
        development_x, development_y = matrix("development_evaluation")
        estimator.fit(train_x, train_y)
        report["fit_called"] = True
        thresholds, calibration_diagnostics = calibrate_thresholds(calibration_y, estimator.predict_proba(calibration_x), estimator.classes_)
        development_probabilities = estimator.predict_proba(development_x)
        development_predictions = apply_thresholds(development_probabilities, estimator.classes_, thresholds)
        bundle = {
            "schema": "sonogym-tp-surface-core-model-v1",
            "estimator": estimator,
            "class_names": CLASS_NAMES,
            "thresholds": thresholds,
            "patch_spec": UNITY_REFERENCE_PATCH_SPEC,
            "fit_sample_ids": report["fit_sample_ids"],
        }
        checkpoint = output / "tp_surface_core_extratrees.joblib"
        joblib.dump(bundle, checkpoint)
        loaded = joblib.load(checkpoint)
        if not np.array_equal(development_probabilities, loaded["estimator"].predict_proba(development_x)):
            raise RuntimeError("checkpoint reload changed development predictions")
        report.update(
            status="TRAINING_COMPLETED",
            checkpoint=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            checkpoint_saved=True,
            checkpoint_reload_verified=True,
            thresholds={str(key): value for key, value in thresholds.items()},
            calibration=calibration_diagnostics,
            development_evaluation=class_metrics(development_y, development_predictions),
        )
    (output / "preflight_result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "resolved_feature_rows", "fit_called")}, sort_keys=True))
    return 3 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
