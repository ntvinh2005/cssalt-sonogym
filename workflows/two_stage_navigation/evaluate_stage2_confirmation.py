#!/usr/bin/env python3
"""Evaluate the frozen Stage 2 model on independent fixed-anatomy sessions."""

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
    Stage2ImageValidityConfig,
    Stage2LabelConfig,
    image_validity_target,
    supervised_targets,
)


PROTOCOL_IDS = frozenset(
    {
        "sonogym-stage2-confirmation-v16",
        "sonogym-stage2-confirmation-v17",
        "sonogym-stage2-confirmation-v18",
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
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": float(precision),
        "recall": float(recall),
    }


def confirmation_gate_passes(
    validity_metrics: dict[str, float | int],
    quality_spearman: float,
    quality_mse: float,
    constant_mse: float,
    gate: dict[str, Any],
) -> bool:
    return bool(
        float(validity_metrics["precision"]) >= float(gate["minimum_validity_precision"])
        and float(validity_metrics["recall"]) >= float(gate["minimum_validity_recall"])
        and quality_spearman >= float(gate["minimum_quality_spearman"])
        and quality_mse < constant_mse
    )


def reference_hashes(root: Path) -> set[str]:
    hashes: set[str] = set()
    for manifest in sorted((root / "sessions").glob("*/public/frames.jsonl")):
        hashes.update(str(row["frame_sha256"]) for row in read_jsonl(manifest))
    if not hashes:
        raise RuntimeError("reference capture has no frame hashes")
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--reference-capture-root", type=Path, required=True)
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()

    protocol_hash = file_sha256(args.protocol)
    expected_hash = args.protocol.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if protocol_hash != expected_hash:
        raise RuntimeError("Stage 2 confirmation protocol seal mismatch")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol["protocol_id"] not in PROTOCOL_IDS:
        raise RuntimeError("unexpected Stage 2 confirmation protocol")

    frozen = protocol["model_confirmation"]
    if file_sha256(args.model_bundle) != frozen["model_bundle_sha256"]:
        raise RuntimeError("frozen Stage 2 model bundle hash mismatch")
    bundle = joblib.load(args.model_bundle)
    if bundle.get("protocol_sha256") != frozen["source_protocol_sha256"]:
        raise RuntimeError("model source protocol mismatch")
    if bundle.get("classifier_target") != "l4_image_visibility":
        raise RuntimeError("model classifier target mismatch")
    if not bool(bundle.get("development_gate_passed")):
        raise RuntimeError("model bundle did not pass its development gate")
    calibration = frozen.get("threshold_calibration")
    if calibration is None:
        if not math.isclose(
            float(bundle["validity_threshold"]),
            float(frozen["validity_probability_threshold"]),
            abs_tol=1e-15,
        ):
            raise RuntimeError("model validity threshold mismatch")
    else:
        calibration_path = Path(calibration["report"])
        if file_sha256(calibration_path) != calibration["report_sha256"]:
            raise RuntimeError("threshold-calibration report hash mismatch")
        calibration_report = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration_report.get("decision") != "CALIBRATION_PASS":
            raise RuntimeError("threshold calibration did not pass")
        if not math.isclose(
            float(calibration_report["selected"]["threshold"]),
            float(frozen["validity_probability_threshold"]),
            abs_tol=1e-15,
        ):
            raise RuntimeError("recalibrated validity threshold mismatch")
    if int(bundle["feature_width"]) != int(protocol["features"]["feature_width"]):
        raise RuntimeError("model feature width mismatch")

    label = protocol["labels"]
    label_config = Stage2LabelConfig(
        maximum_position_error_mm=float(label["validity"]["maximum_position_error_mm"]),
        maximum_orientation_error_deg=float(
            label["validity"]["maximum_orientation_error_deg"]
        ),
        minimum_l4_visible_fraction=float(label["validity"]["minimum_l4_visible_fraction"]),
        visibility_normalization_fraction=float(
            label["quality"]["visibility_normalization_fraction"]
        ),
    )
    image_config = Stage2ImageValidityConfig(
        minimum_l4_visible_fraction=float(
            frozen["classifier_target"]["minimum_l4_visible_fraction"]
        ),
        training_ambiguous_half_width=float(
            frozen["classifier_target"]["training_ambiguous_half_width"]
        ),
    )
    if bundle.get("image_validity_config") != image_config.__dict__:
        raise RuntimeError("model image-validity configuration mismatch")

    classifier = bundle["classifier"]
    regressor = bundle["regressor"]
    class_columns = {int(value): index for index, value in enumerate(classifier.classes_)}
    if 1 not in class_columns:
        raise RuntimeError("validity classifier has no positive class")

    declared_reference_roots = frozen.get("reference_capture_roots")
    if declared_reference_roots is None:
        declared_reference_roots = [str(args.reference_capture_root)]
    elif str(args.reference_capture_root) != str(Path(declared_reference_roots[0])):
        raise RuntimeError("primary reference capture root mismatch")
    old_hashes: set[str] = set()
    for reference_root in declared_reference_roots:
        old_hashes.update(reference_hashes(Path(reference_root)))
    features: list[np.ndarray] = []
    image_validity: list[int] = []
    evaluator_validity: list[int] = []
    quality: list[float] = []
    hashes: list[str] = []
    trajectories: set[str] = set()
    session_rows: list[dict[str, Any]] = []
    expected_frames = int(protocol["sampling"]["expected_frames_per_session"])
    for session in protocol["sessions"]:
        session_id = str(session["session_id"])
        root = args.capture_root / "sessions" / session_id
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "PASSED":
            raise RuntimeError(f"incomplete confirmation session {session_id}")
        public = read_jsonl(root / "public" / "frames.jsonl")
        privileged = read_jsonl(root / "privileged" / "annotations.jsonl")
        images = np.load(root / "public" / "ultrasound.npz")["ultrasound"]
        if not len(public) == len(privileged) == len(images) == expected_frames:
            raise RuntimeError(f"unexpected confirmation sample count in {session_id}")
        session_hashes: set[str] = set()
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
            if identity[2] in old_hashes:
                raise RuntimeError(f"reference image hash reused in {session_id}:{index}")
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
            image_target, _ = image_validity_target(
                technical_image_valid=bool(annotation["technical_image_valid"]),
                l4_visible_fraction=float(annotation["l4_visible_fraction"]),
                config=image_config,
            )
            features.append(extract_stage2_feature(image))
            image_validity.append(image_target)
            evaluator_validity.append(expected_validity)
            quality.append(expected_quality)
            hashes.append(identity[2])
            session_hashes.add(identity[2])
            trajectories.add(str(actor["trajectory_id"]))
        session_rows.append(
            {
                "session_id": session_id,
                "frame_count": len(public),
                "distinct_frame_hash_count": len(session_hashes),
            }
        )

    feature_array = np.stack(features).astype(np.float32)
    image_validity_array = np.asarray(image_validity, dtype=np.int8)
    evaluator_validity_array = np.asarray(evaluator_validity, dtype=np.int8)
    quality_array = np.asarray(quality, dtype=np.float32)
    probabilities = classifier.predict_proba(feature_array)[:, class_columns[1]]
    predictions = probabilities >= float(frozen["validity_probability_threshold"])
    validity_metrics = binary_metrics(image_validity_array, predictions)
    quality_prediction = regressor.predict(feature_array)
    if not np.all(np.isfinite(probabilities)) or not np.all(
        np.isfinite(quality_prediction)
    ):
        raise RuntimeError("nonfinite frozen-model output")
    quality_mask = quality_array >= float(frozen["regressor_evaluation_minimum_quality"])
    if np.count_nonzero(quality_mask) < int(frozen["minimum_quality_support"]):
        raise RuntimeError("insufficient confirmation quality support")
    correlation = spearmanr(
        quality_array[quality_mask], quality_prediction[quality_mask]
    ).statistic
    if not np.isfinite(correlation):
        correlation = 0.0
    mse = float(
        np.mean((quality_array[quality_mask] - quality_prediction[quality_mask]) ** 2)
    )
    constant = float(frozen["quality_training_mean_constant"])
    constant_mse = float(np.mean((quality_array[quality_mask] - constant) ** 2))
    passed = confirmation_gate_passes(
        validity_metrics,
        float(correlation),
        mse,
        constant_mse,
        frozen["gate"],
    )

    output = args.output_dir.resolve()
    report = {
        "schema": "sonogym-stage2-confirmation-report-v1",
        "decision": "CONFIRMATION_PASS" if passed else "CONFIRMATION_NO_GO",
        "protocol_sha256": protocol_hash,
        "model_bundle_sha256": file_sha256(args.model_bundle),
        "model_retrained": False,
        "session_count": len(session_rows),
        "trajectory_count": len(trajectories),
        "frame_count": len(features),
        "distinct_frame_hash_count": len(set(hashes)),
        "reference_hash_overlap_count": 0,
        "reference_capture_count": len(declared_reference_roots),
        "classifier_target_positive_count": int(np.count_nonzero(image_validity_array)),
        "geometric_evaluator_positive_count": int(
            np.count_nonzero(evaluator_validity_array)
        ),
        "validity_probability_threshold": float(
            frozen["validity_probability_threshold"]
        ),
        "validity_metrics": validity_metrics,
        "quality_metrics": {
            "support": int(np.count_nonzero(quality_mask)),
            "spearman": float(correlation),
            "mse": mse,
            "training_mean_constant": constant,
            "training_mean_constant_mse": constant_mse,
        },
        "sessions": session_rows,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "limitations": {
            "fixed_anatomy_only": True,
            "closed_loop_not_established": True,
            "patient_generalization_not_established": True,
        },
    }
    report_path = output / "confirmation_report.json"
    write_json(report_path, report)
    write_seal(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
