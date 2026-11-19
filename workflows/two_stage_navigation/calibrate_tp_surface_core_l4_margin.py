#!/usr/bin/env python3
"""Calibrate one frozen-checkpoint L4 decision margin without fitting or reading development images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = (
    ROOT
    / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/stage1_features.py"
)
FEATURE_SPEC = importlib.util.spec_from_file_location("tp_proxy_features", FEATURE_PATH)
if FEATURE_SPEC is None or FEATURE_SPEC.loader is None:
    raise RuntimeError("cannot load the frozen checkpoint feature module")
FEATURES = importlib.util.module_from_spec(FEATURE_SPEC)
sys.modules[FEATURE_SPEC.name] = FEATURES
FEATURE_SPEC.loader.exec_module(FEATURES)
PATCH_SPEC = FEATURES.UNITY_REFERENCE_PATCH_SPEC
extract_patch_feature = FEATURES.extract_patch_feature

CLASS_BACKGROUND = 0
CLASS_L4 = 1
CLASS_L5 = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def ultrasound_sha256(image: np.ndarray) -> str:
    array = np.asarray(image, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def class_column(classes: np.ndarray, class_id: int) -> int:
    columns = np.flatnonzero(classes == class_id)
    if len(columns) != 1:
        raise ValueError(f"checkpoint does not contain class {class_id}")
    return int(columns[0])


def l4_proposal_passes(
    probabilities: np.ndarray,
    classes: np.ndarray,
    *,
    base_threshold: float,
    margin: float,
) -> bool:
    l4_column = class_column(classes, CLASS_L4)
    background_column = class_column(classes, CLASS_BACKGROUND)
    l5_column = class_column(classes, CLASS_L5)
    predicted_class = int(classes[int(np.argmax(probabilities))])
    l4_probability = float(probabilities[l4_column])
    competing_probability = max(
        float(probabilities[background_column]), float(probabilities[l5_column])
    )
    return bool(
        predicted_class == CLASS_L4
        and l4_probability >= base_threshold
        and l4_probability - competing_probability >= margin
    )


def l5_proposal_passes(
    probabilities: np.ndarray, classes: np.ndarray, *, base_threshold: float
) -> bool:
    l5_column = class_column(classes, CLASS_L5)
    predicted_class = int(classes[int(np.argmax(probabilities))])
    return bool(
        predicted_class == CLASS_L5
        and float(probabilities[l5_column]) >= base_threshold
    )


def accepted_false_positive_strata(rows: list[dict[str, Any]]) -> dict[str, bool]:
    has_uncertainty = any(
        int(row["l4_uncertain_points_in_patch"]) > 0
        or int(row["l5_uncertain_points_in_patch"]) > 0
        for row in rows
    )
    has_competing_l5 = any(int(row["l5_core_points_in_patch"]) > 0 for row in rows)
    geometry_clean = all(
        int(row[field]) == 0
        for row in rows
        for field in (
            "l4_core_points_in_patch",
            "l5_core_points_in_patch",
            "l4_uncertain_points_in_patch",
            "l5_uncertain_points_in_patch",
        )
    )
    return {
        "clean_negative": geometry_clean,
        "competing_l5": has_competing_l5,
        "uncertainty": has_uncertainty,
        "other_proxy_exclusion": not (
            geometry_clean or has_competing_l5 or has_uncertainty
        ),
    }


def evaluate_margin(
    frames: dict[str, list[dict[str, Any]]],
    truths: dict[str, set[int]],
    classes: np.ndarray,
    *,
    base_threshold: float,
    margin: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tp = fp = fn = tn = 0
    strata = {
        "clean_negative": 0,
        "competing_l5": 0,
        "uncertainty": 0,
        "other_proxy_exclusion": 0,
    }
    image_rows: list[dict[str, Any]] = []
    for frame_hash, proposals in sorted(frames.items()):
        accepted = [
            row
            for row in proposals
            if l4_proposal_passes(
                row["probabilities"],
                classes,
                base_threshold=base_threshold,
                margin=margin,
            )
        ]
        truth = CLASS_L4 in truths.get(frame_hash, set())
        predicted = bool(accepted)
        tp += int(truth and predicted)
        fp += int((not truth) and predicted)
        fn += int(truth and (not predicted))
        tn += int((not truth) and (not predicted))
        flags = (
            accepted_false_positive_strata(accepted)
            if predicted and not truth
            else {key: False for key in strata}
        )
        if predicted and not truth:
            for key, value in flags.items():
                strata[key] += int(value)
        image_rows.append(
            {
                "frame_sha256": frame_hash,
                "session_id": proposals[0]["session_id"],
                "truth_l4": truth,
                "predicted_l4": predicted,
                "accepted_l4_proposal_count": len(accepted),
                **{f"false_positive_{key}": value for key, value in flags.items()},
            }
        )
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    return (
        {
            "margin": margin,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "clean_negative_fp": strata["clean_negative"],
            "competing_l5_fp": strata["competing_l5"],
            "uncertainty_fp": strata["uncertainty"],
            "other_proxy_exclusion_fp": strata["other_proxy_exclusion"],
        },
        image_rows,
    )


def select_margin(
    grid: list[dict[str, Any]], *, retain_fraction: float
) -> tuple[dict[str, Any] | None, int]:
    baseline = grid[0]
    required_tp = math.ceil(float(baseline["tp"]) * retain_fraction)
    passing = [
        row
        for row in grid[1:]
        if int(row["tp"]) >= required_tp
        and int(row["clean_negative_fp"]) < int(baseline["clean_negative_fp"])
    ]
    if not passing:
        return None, required_tp
    return (
        min(
            passing,
            key=lambda row: (
                int(row["clean_negative_fp"]),
                -int(row["tp"]),
                int(row["fp"]),
                float(row["margin"]),
            ),
        ),
        required_tp,
    )


def main() -> int:
    args = parse_args()
    training_dir = args.training_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol["allowed_split"] != "calibration":
        raise ValueError("this experiment is calibration-only")
    if protocol["checkpoint_fitting"] != "forbidden":
        raise ValueError("checkpoint fitting must remain forbidden")
    if protocol["l5_modification"] != "forbidden":
        raise ValueError("L5 modification must remain forbidden")

    checkpoint = training_dir / "tp_surface_core_simulator_proxy_extratrees.joblib"
    audit_path = training_dir / "proxy_label_audit.csv"
    selected_path = training_dir / "proxy_sample_manifest.csv"
    expected_hashes = {
        checkpoint: protocol["source_checkpoint_sha256"],
        audit_path: protocol["source_proxy_label_audit_sha256"],
        selected_path: protocol["source_proxy_sample_manifest_sha256"],
    }
    for path, expected in expected_hashes.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"frozen input SHA-256 mismatch for {path}: {actual}")

    bundle = joblib.load(checkpoint)
    estimator = bundle["estimator"]
    classes = np.asarray(estimator.classes_)
    thresholds = {int(key): float(value) for key, value in bundle["thresholds"].items()}
    base_threshold = float(protocol["l4_base_threshold"])
    if thresholds != {CLASS_L4: base_threshold, CLASS_L5: 0.5}:
        raise ValueError(f"unexpected frozen thresholds: {thresholds}")

    audit_rows = [
        row for row in read_csv(audit_path) if row["split"] == protocol["allowed_split"]
    ]
    selected_rows = [
        row for row in read_csv(selected_path) if row["split"] == protocol["allowed_split"]
    ]
    if not audit_rows or not selected_rows:
        raise ValueError("calibration rows are missing")

    archives: dict[Path, np.ndarray] = {}
    frames: dict[str, list[dict[str, Any]]] = defaultdict(list)
    proposal_scores: list[dict[str, Any]] = []
    opened_sessions: set[str] = set()
    for row in audit_rows:
        if row["split"] != "calibration":
            raise ValueError("a non-calibration proposal reached scoring")
        archive_path = Path(row["image_archive"])
        if archive_path not in archives:
            archives[archive_path] = np.load(archive_path)["ultrasound"]
        index = int(row["image_archive_index"])
        image = np.asarray(archives[archive_path][index])
        if image.shape != (200, 150) or ultrasound_sha256(image) != row["frame_sha256"]:
            raise ValueError(f"source image mismatch: {row['sample_id']}")
        feature = extract_patch_feature(
            image, float(row["patch_row"]), float(row["patch_column"]), PATCH_SPEC
        )
        if feature is None or not np.all(np.isfinite(feature)):
            raise ValueError(f"invalid feature: {row['sample_id']}")
        if array_sha256(feature) != row["feature_sha256"]:
            raise ValueError(f"feature hash mismatch: {row['sample_id']}")
        probabilities = estimator.predict_proba(np.asarray(feature)[None, :])[0]
        scored = dict(row)
        scored["probabilities"] = probabilities
        frames[row["frame_sha256"]].append(scored)
        opened_sessions.add(row["session_id"])
        proposal_scores.append(
            {
                "sample_id": row["sample_id"],
                "session_id": row["session_id"],
                "frame_sha256": row["frame_sha256"],
                "probability_background": float(probabilities[class_column(classes, 0)]),
                "probability_l4": float(probabilities[class_column(classes, 1)]),
                "probability_l5": float(probabilities[class_column(classes, 2)]),
                "l4_margin": float(probabilities[class_column(classes, 1)])
                - max(
                    float(probabilities[class_column(classes, 0)]),
                    float(probabilities[class_column(classes, 2)]),
                ),
                "l4_core_points_in_patch": row["l4_core_points_in_patch"],
                "l5_core_points_in_patch": row["l5_core_points_in_patch"],
                "l4_uncertain_points_in_patch": row["l4_uncertain_points_in_patch"],
                "l5_uncertain_points_in_patch": row["l5_uncertain_points_in_patch"],
            }
        )

    truths: dict[str, set[int]] = defaultdict(set)
    for row in selected_rows:
        truths[row["frame_sha256"]].add(int(row["proxy_class_id"]))
    margin_spec = protocol["l4_margin"]
    margins = np.arange(
        float(margin_spec["grid_start"]),
        float(margin_spec["grid_stop"]) + float(margin_spec["grid_step"]) / 2.0,
        float(margin_spec["grid_step"]),
    )
    grid: list[dict[str, Any]] = []
    image_rows_by_margin: dict[float, list[dict[str, Any]]] = {}
    for margin_value in margins:
        value = float(round(float(margin_value), 10))
        metrics, image_rows = evaluate_margin(
            frames,
            truths,
            classes,
            base_threshold=base_threshold,
            margin=value,
        )
        grid.append(metrics)
        image_rows_by_margin[value] = image_rows

    selected, required_tp = select_margin(
        grid,
        retain_fraction=float(
            protocol["selection"]["retain_baseline_true_positive_fraction"]
        ),
    )
    baseline = grid[0]
    selected_margin = None if selected is None else float(selected["margin"])
    status = "PASS" if selected is not None else "FAIL"
    chosen_rows = image_rows_by_margin[0.0 if selected_margin is None else selected_margin]

    l5_truth_and_prediction: list[tuple[bool, bool]] = []
    for frame_hash, proposals in sorted(frames.items()):
        truth = CLASS_L5 in truths.get(frame_hash, set())
        predicted = any(
            l5_proposal_passes(
                row["probabilities"], classes, base_threshold=thresholds[CLASS_L5]
            )
            for row in proposals
        )
        l5_truth_and_prediction.append((truth, predicted))
    l5_metrics = {
        "tp": sum(truth and predicted for truth, predicted in l5_truth_and_prediction),
        "fp": sum((not truth) and predicted for truth, predicted in l5_truth_and_prediction),
        "fn": sum(truth and (not predicted) for truth, predicted in l5_truth_and_prediction),
        "tn": sum((not truth) and (not predicted) for truth, predicted in l5_truth_and_prediction),
    }

    write_csv(output / "calibration_l4_margin_grid.csv", grid)
    write_csv(output / "calibration_proposal_scores.csv", proposal_scores)
    write_csv(output / "selected_margin_image_decisions.csv", chosen_rows)
    result = {
        "schema": "sonogym-tp-surface-core-proxy-l4-margin-calibration-v1",
        "experiment_id": protocol["experiment_id"],
        "status": status,
        "scientific_scope": protocol["scientific_scope"],
        "fit_called": False,
        "development_images_opened": False,
        "opened_sessions": sorted(opened_sessions),
        "calibration_image_count": len(frames),
        "calibration_proposal_count": len(audit_rows),
        "checkpoint_sha256": sha256(checkpoint),
        "base_threshold_unchanged": base_threshold,
        "l5_modified": False,
        "l5_calibration_metrics_unchanged": l5_metrics,
        "baseline": baseline,
        "required_retained_true_positives": required_tp,
        "selected": selected,
        "selected_margin": selected_margin,
        "margin_grid": [float(value) for value in margins],
        "stop_without_new_capture": selected is None,
        "future_development_evaluation_run": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if selected is not None:
        frozen = {
            "schema": "sonogym-tp-surface-core-proxy-l4-margin-frozen-v1",
            "source_calibration_experiment": protocol["experiment_id"],
            "checkpoint_sha256": sha256(checkpoint),
            "l4_base_threshold": base_threshold,
            "l4_margin": selected_margin,
            "l5_threshold": thresholds[CLASS_L5],
            "l5_modified": False,
            "development_evaluation_status": "NOT_RUN",
        }
        (output / "frozen_margin.json").write_text(
            json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"status": status, "selected_margin": selected_margin}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
