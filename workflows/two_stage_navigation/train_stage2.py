#!/usr/bin/env python3
"""Fit and evaluate the bounded Stage 2 supervised development models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
import sys

if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.contracts import (  # noqa: E402
    assert_no_privileged_fields,
    ultrasound_sha256,
)
from two_stage_navigation.stage2_features import extract_stage2_feature  # noqa: E402
from two_stage_navigation.stage2_models import (  # noqa: E402
    Stage2LabelConfig,
    supervised_targets,
)


PROTOCOL_IDS = frozenset(
    {
        "sonogym-stage2-dataset-v1",
        "sonogym-stage2-dataset-v2",
        "sonogym-stage2-dataset-v3",
    }
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_seal(path: Path) -> str:
    digest = file_sha256(path)
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    truth = labels.astype(bool)
    predicted = predictions.astype(bool)
    true_positive = int(np.count_nonzero(truth & predicted))
    false_positive = int(np.count_nonzero(~truth & predicted))
    false_negative = int(np.count_nonzero(truth & ~predicted))
    true_negative = int(np.count_nonzero(~truth & ~predicted))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": float(precision),
        "recall": float(recall),
    }


def positive_probability(model: ExtraTreesClassifier, features: np.ndarray) -> np.ndarray:
    columns = {int(value): index for index, value in enumerate(model.classes_)}
    if 1 not in columns:
        raise RuntimeError("validity classifier has no positive class")
    return model.predict_proba(features)[:, columns[1]]


def select_threshold(
    labels: np.ndarray, probabilities: np.ndarray, minimum_precision: float
) -> tuple[float, dict[str, float | int]]:
    candidates = sorted({float(value) for value in probabilities}, reverse=True)
    qualified = []
    for threshold in candidates:
        metrics = binary_metrics(labels, probabilities >= threshold)
        if metrics["precision"] >= minimum_precision and metrics["true_positive"] > 0:
            qualified.append((float(metrics["recall"]), float(metrics["precision"]), -threshold, threshold, metrics))
    if not qualified:
        raise RuntimeError("no calibration threshold meets the precision requirement")
    _, _, _, threshold, metrics = max(qualified)
    return float(threshold), metrics


def load_session(
    capture_root: Path,
    session: dict[str, Any],
    label_config: Stage2LabelConfig,
) -> dict[str, Any]:
    session_id = str(session["session_id"])
    root = capture_root / "sessions" / session_id
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "PASSED":
        raise RuntimeError(f"incomplete Stage 2 dataset session {session_id}")
    public = read_jsonl(root / "public" / "frames.jsonl")
    privileged = read_jsonl(root / "privileged" / "annotations.jsonl")
    images = np.load(root / "public" / "ultrasound.npz")["ultrasound"]
    expected = 75
    if not len(public) == len(privileged) == len(images) == expected:
        raise RuntimeError(f"unexpected sample count in {session_id}")
    features = []
    validity = []
    quality = []
    hashes = []
    for index, (actor, annotation, image) in enumerate(zip(public, privileged, images)):
        assert_no_privileged_fields(actor)
        identity = (session_id, index, ultrasound_sha256(image))
        if identity != (
            actor["session_id"],
            int(actor["frame_sequence"]),
            actor["frame_sha256"],
        ) or identity != (
            annotation["session_id"],
            int(annotation["frame_sequence"]),
            annotation["frame_sha256"],
        ):
            raise RuntimeError(f"frame identity mismatch in {session_id}:{index}")
        expected_validity, expected_quality = supervised_targets(
            technical_image_valid=bool(annotation["technical_image_valid"]),
            position_error_mm=float(annotation["position_error_mm"]),
            orientation_error_deg=float(annotation["orientation_error_deg"]),
            l4_visible_fraction=float(annotation["l4_visible_fraction"]),
            config=label_config,
        )
        if expected_validity != int(annotation["validity_label"]) or not math.isclose(
            expected_quality, float(annotation["quality_label"]), abs_tol=1e-12
        ):
            raise RuntimeError(f"stored label mismatch in {session_id}:{index}")
        features.append(extract_stage2_feature(image))
        validity.append(expected_validity)
        quality.append(expected_quality)
        hashes.append(identity[2])
    return {
        "session_id": session_id,
        "split": str(session["split"]),
        "features": np.stack(features).astype(np.float32),
        "validity": np.asarray(validity, dtype=np.int8),
        "quality": np.asarray(quality, dtype=np.float32),
        "hashes": hashes,
    }


def concatenate(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.concatenate([np.asarray(row[key]) for row in rows])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    protocol_hash = file_sha256(args.protocol)
    expected_hash = args.protocol.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if protocol_hash != expected_hash:
        raise RuntimeError("Stage 2 protocol seal mismatch")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol["protocol_id"] not in PROTOCOL_IDS:
        raise RuntimeError("unexpected Stage 2 protocol")
    label = protocol["labels"]
    label_config = Stage2LabelConfig(
        maximum_position_error_mm=float(label["validity"]["maximum_position_error_mm"]),
        maximum_orientation_error_deg=float(label["validity"]["maximum_orientation_error_deg"]),
        minimum_l4_visible_fraction=float(label["validity"]["minimum_l4_visible_fraction"]),
        visibility_normalization_fraction=float(label["quality"]["visibility_normalization_fraction"]),
    )
    sessions = [load_session(args.capture_root, row, label_config) for row in protocol["sessions"]]
    training = [row for row in sessions if row["split"] == "train"]
    calibration = [row for row in sessions if row["split"] == "calibration"]
    development = [row for row in sessions if row["split"] == "development_evaluation"]
    if not (len(training) == 4 and len(calibration) == 1 and len(development) == 1):
        raise RuntimeError("unexpected Stage 2 session split")
    split_hashes = [set(concatenate(rows, "hashes")) for rows in (training, calibration, development)]
    if any(left & right for index, left in enumerate(split_hashes) for right in split_hashes[index + 1 :]):
        raise RuntimeError("duplicate image hash crosses a session split")

    model = protocol["model"]
    train_features = concatenate(training, "features")
    train_validity = concatenate(training, "validity")
    train_quality = concatenate(training, "quality")
    classifier = ExtraTreesClassifier(
        n_estimators=int(model["n_estimators"]),
        max_depth=int(model["max_depth"]),
        min_samples_leaf=int(model["min_samples_leaf"]),
        max_features=float(model["max_features"]),
        class_weight=str(model["classifier_class_weight"]),
        n_jobs=int(model["n_jobs"]),
        random_state=int(model["random_seed"]),
    )
    classifier.fit(train_features, train_validity)
    calibration_features = concatenate(calibration, "features")
    calibration_validity = concatenate(calibration, "validity")
    threshold, calibration_metrics = select_threshold(
        calibration_validity,
        positive_probability(classifier, calibration_features),
        float(protocol["development_gate"]["minimum_validity_precision"]),
    )

    regressor_mask = train_quality >= float(model["regressor_training_minimum_quality"])
    if np.count_nonzero(regressor_mask) < 20:
        raise RuntimeError("insufficient nontrivial quality examples")
    regressor = ExtraTreesRegressor(
        n_estimators=int(model["n_estimators"]),
        max_depth=int(model["max_depth"]),
        min_samples_leaf=int(model["min_samples_leaf"]),
        max_features=float(model["max_features"]),
        n_jobs=int(model["n_jobs"]),
        random_state=int(model["random_seed"]) + 1,
    )
    regressor.fit(train_features[regressor_mask], train_quality[regressor_mask])

    dev_features = concatenate(development, "features")
    dev_validity = concatenate(development, "validity")
    dev_quality = concatenate(development, "quality")
    dev_probability = positive_probability(classifier, dev_features)
    dev_metrics = binary_metrics(dev_validity, dev_probability >= threshold)
    dev_quality_prediction = regressor.predict(dev_features)
    quality_mask = dev_quality >= float(model["regressor_training_minimum_quality"])
    if np.count_nonzero(quality_mask) < 3:
        raise RuntimeError("insufficient development quality support")
    correlation = spearmanr(
        dev_quality[quality_mask], dev_quality_prediction[quality_mask]
    ).statistic
    if not np.isfinite(correlation):
        correlation = 0.0
    mse = float(np.mean((dev_quality[quality_mask] - dev_quality_prediction[quality_mask]) ** 2))
    constant = float(np.mean(train_quality[regressor_mask]))
    constant_mse = float(np.mean((dev_quality[quality_mask] - constant) ** 2))
    gate = protocol["development_gate"]
    passed = bool(
        float(dev_metrics["precision"]) >= float(gate["minimum_validity_precision"])
        and float(dev_metrics["recall"]) >= float(gate["minimum_validity_recall"])
        and float(correlation) >= float(gate["minimum_quality_spearman"])
        and mse < constant_mse
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle_path = output / "stage2_models.joblib"
    bundle = {
        "schema": "sonogym-stage2-model-bundle-v1",
        "protocol_sha256": protocol_hash,
        "classifier": classifier,
        "regressor": regressor,
        "validity_threshold": threshold,
        "label_config": label_config.__dict__,
        "feature_width": int(protocol["features"]["feature_width"]),
        "training_session_ids": [row["session_id"] for row in training],
        "calibration_session_ids": [row["session_id"] for row in calibration],
        "development_session_ids": [row["session_id"] for row in development],
        "development_gate_passed": passed,
    }
    joblib.dump(bundle, bundle_path, compress=3)
    bundle_hash = file_sha256(bundle_path)
    report = {
        "schema": "sonogym-stage2-development-report-v1",
        "decision": "DEVELOPMENT_PASS" if passed else "DEVELOPMENT_NO_GO",
        "protocol_sha256": protocol_hash,
        "session_count": len(sessions),
        "frame_count": sum(len(row["validity"]) for row in sessions),
        "patient_count": 1,
        "split_frame_counts": {
            "train": int(len(train_validity)),
            "calibration": int(len(calibration_validity)),
            "development_evaluation": int(len(dev_validity)),
        },
        "split_positive_counts": {
            "train": int(np.count_nonzero(train_validity)),
            "calibration": int(np.count_nonzero(calibration_validity)),
            "development_evaluation": int(np.count_nonzero(dev_validity)),
        },
        "validity_threshold": threshold,
        "calibration_metrics": calibration_metrics,
        "development_validity_metrics": dev_metrics,
        "development_quality_metrics": {
            "support": int(np.count_nonzero(quality_mask)),
            "spearman": float(correlation),
            "mse": mse,
            "training_mean_constant_mse": constant_mse,
        },
        "bundle_path": str(bundle_path),
        "bundle_sha256": bundle_hash,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "limitations": {
            "fixed_anatomy_only": True,
            "development_only": True,
            "closed_loop_not_established": True,
            "patient_generalization_not_established": True,
        },
    }
    report_path = output / "development_report.json"
    write_json(report_path, report)
    write_seal(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
