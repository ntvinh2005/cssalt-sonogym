#!/usr/bin/env python3
"""Select a session-robust threshold for the frozen Stage 2 classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml


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
    image_validity_target,
)


PROTOCOL_ID = "sonogym-stage2-threshold-calibration-v17"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()

    protocol_hash = file_sha256(args.protocol)
    expected_hash = args.protocol.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if protocol_hash != expected_hash:
        raise RuntimeError("threshold-calibration protocol seal mismatch")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("unexpected threshold-calibration protocol")

    frozen = protocol["frozen_model"]
    bundle_path = Path(frozen["bundle"])
    if file_sha256(bundle_path) != frozen["bundle_sha256"]:
        raise RuntimeError("frozen model bundle hash mismatch")
    bundle = joblib.load(bundle_path)
    if bundle.get("protocol_sha256") != frozen["source_protocol_sha256"]:
        raise RuntimeError("frozen model source protocol mismatch")
    if bundle.get("classifier_target") != "l4_image_visibility":
        raise RuntimeError("frozen classifier target mismatch")
    classifier = bundle["classifier"]
    columns = {int(value): index for index, value in enumerate(classifier.classes_)}
    if 1 not in columns:
        raise RuntimeError("validity classifier has no positive class")

    target = protocol["classifier_target"]
    image_config = Stage2ImageValidityConfig(
        minimum_l4_visible_fraction=float(target["minimum_l4_visible_fraction"]),
        training_ambiguous_half_width=float(target["training_ambiguous_half_width"]),
    )
    if bundle.get("image_validity_config") != image_config.__dict__:
        raise RuntimeError("image-validity configuration mismatch")

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    hashes: list[str] = []
    session_names: list[str] = []
    source_hashes: dict[str, set[str]] = {}
    session_counts: dict[str, int] = {}
    for source in protocol["calibration_sources"]:
        source_id = str(source["source_id"])
        root = Path(source["capture_root"])
        current_source_hashes: set[str] = set()
        for session_id in source["session_ids"]:
            session_root = root / "sessions" / str(session_id)
            public = read_jsonl(session_root / "public" / "frames.jsonl")
            privileged = read_jsonl(session_root / "privileged" / "annotations.jsonl")
            images = np.load(session_root / "public" / "ultrasound.npz")["ultrasound"]
            if not len(public) == len(privileged) == len(images) == int(source["frames_per_session"]):
                raise RuntimeError(f"incomplete calibration session {session_id}")
            session_counts[str(session_id)] = len(public)
            for index, (actor, annotation, image) in enumerate(zip(public, privileged, images)):
                assert_no_privileged_fields(actor)
                identity = (str(session_id), index, ultrasound_sha256(image))
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
                label, _ = image_validity_target(
                    technical_image_valid=bool(annotation["technical_image_valid"]),
                    l4_visible_fraction=float(annotation["l4_visible_fraction"]),
                    config=image_config,
                )
                feature_rows.append(extract_stage2_feature(image))
                labels.append(label)
                hashes.append(identity[2])
                session_names.append(str(session_id))
                current_source_hashes.add(identity[2])
        source_hashes[source_id] = current_source_hashes

    source_ids = list(source_hashes)
    if any(
        source_hashes[left] & source_hashes[right]
        for index, left in enumerate(source_ids)
        for right in source_ids[index + 1 :]
    ):
        raise RuntimeError("image hash crosses calibration sources")

    features = np.stack(feature_rows).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int8)
    probabilities = classifier.predict_proba(features)[:, columns[1]]
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("nonfinite classifier probabilities")
    session_array = np.asarray(session_names, dtype=object)

    first_hash_indices: list[int] = []
    seen_hashes: set[str] = set()
    for index, digest in enumerate(hashes):
        if digest not in seen_hashes:
            seen_hashes.add(digest)
            first_hash_indices.append(index)
    unique_indices = np.asarray(first_hash_indices, dtype=np.int64)

    rule = protocol["selection_rule"]
    candidates: list[dict[str, Any]] = []
    for threshold in sorted({float(value) for value in probabilities}, reverse=True):
        frame_metrics = binary_metrics(label_array, probabilities >= threshold)
        unique_metrics = binary_metrics(
            label_array[unique_indices], probabilities[unique_indices] >= threshold
        )
        per_session = {
            session_id: binary_metrics(
                label_array[session_array == session_id],
                probabilities[session_array == session_id] >= threshold,
            )
            for session_id in sorted(session_counts)
        }
        if (
            frame_metrics["precision"] >= float(rule["minimum_pooled_precision"])
            and unique_metrics["precision"] >= float(rule["minimum_unique_hash_precision"])
            and all(row["true_positive"] > 0 for row in per_session.values())
        ):
            recalls = [float(row["recall"]) for row in per_session.values()]
            candidates.append(
                {
                    "threshold": threshold,
                    "frame_metrics": frame_metrics,
                    "unique_hash_metrics": unique_metrics,
                    "per_session_metrics": per_session,
                    "minimum_session_recall": min(recalls),
                    "median_session_recall": float(statistics.median(recalls)),
                }
            )
    if not candidates:
        raise RuntimeError("no threshold meets the frozen calibration precision constraints")
    selected = max(
        candidates,
        key=lambda row: (
            row["minimum_session_recall"],
            row["median_session_recall"],
            row["frame_metrics"]["recall"],
            row["unique_hash_metrics"]["recall"],
            row["threshold"],
        ),
    )
    passed = bool(
        selected["frame_metrics"]["recall"] >= float(rule["minimum_pooled_recall"])
        and selected["unique_hash_metrics"]["recall"]
        >= float(rule["minimum_unique_hash_recall"])
        and selected["minimum_session_recall"] >= float(rule["minimum_session_recall"])
        and selected["median_session_recall"] >= float(rule["minimum_median_session_recall"])
    )
    report = {
        "schema": "sonogym-stage2-threshold-calibration-report-v1",
        "decision": "CALIBRATION_PASS" if passed else "CALIBRATION_NO_GO",
        "protocol_sha256": protocol_hash,
        "model_bundle_sha256": file_sha256(bundle_path),
        "model_retrained": False,
        "source_count": len(source_ids),
        "session_count": len(session_counts),
        "frame_count": len(labels),
        "distinct_frame_hash_count": len(seen_hashes),
        "cross_source_hash_overlap_count": 0,
        "selected": selected,
        "selection_rule": rule,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "boundary": {
            "development_and_prior_confirmation_data_only": True,
            "future_confirmation_data_used": False,
            "closed_loop_not_authorized": True,
        },
    }
    output = args.output_dir.resolve()
    report_path = output / "calibration_report.json"
    write_json(report_path, report)
    digest = file_sha256(report_path)
    report_path.with_suffix(".sha256").write_text(
        f"{digest}  {report_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
