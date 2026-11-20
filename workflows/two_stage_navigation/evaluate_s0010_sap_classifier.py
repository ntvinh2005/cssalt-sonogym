#!/usr/bin/env python3
"""Evaluate frozen SAP classifiers once on session-disjoint evaluation captures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

import train_s0010_sap_classifier as training


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def top_k_localization(
    *,
    estimator: Any,
    level: str,
    sample_rows: list[dict[str, str]],
    proposal_rows: list[dict[str, str]],
    proposal_features: np.ndarray,
) -> dict[str, Any]:
    positive_hashes = {
        row["frame_sha256"]
        for row in sample_rows
        if row["conditioned_level"] == level
        and row["split"] == "evaluation"
        and row["provisional_label"] == training.POSITIVE
    }
    groups: dict[str, list[dict[str, str]]] = {}
    for row in proposal_rows:
        if row["conditioned_level"] == level and row["split"] == "evaluation":
            groups.setdefault(row["frame_sha256"], []).append(row)
    top1 = 0
    top3 = 0
    positive_column = list(estimator.classes_).index(1)
    for frame_hash in positive_hashes:
        rows = groups[frame_hash]
        matrix = np.asarray(
            [proposal_features[int(row["proposal_feature_index"])] for row in rows]
        )
        scores = estimator.predict_proba(matrix)[:, positive_column]
        order = np.argsort(-scores)
        top1 += int(rows[int(order[0])]["provisional_label"] == training.POSITIVE)
        top3 += int(
            any(
                rows[int(index)]["provisional_label"] == training.POSITIVE
                for index in order[:3]
            )
        )
    total = len(positive_hashes)
    return {
        "positive_images": total,
        "top1_correct": top1,
        "top1_accuracy": top1 / total if total else 0.0,
        "top3_correct": top3,
        "top3_accuracy": top3 / total if total else 0.0,
    }


def session_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    hashes: list[str],
    rows: list[dict[str, str]],
    level: str,
    threshold: float,
) -> dict[str, Any]:
    hash_session = {
        row["frame_sha256"]: row["session_id"]
        for row in rows
        if row["conditioned_level"] == level and row["split"] == "evaluation"
    }
    result: dict[str, Any] = {}
    for session in sorted(set(hash_session.values())):
        indices = [index for index, value in enumerate(hashes) if hash_session[value] == session]
        result[session] = training.metrics(labels[indices], scores[indices], threshold)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset_dir.resolve()
    training_dir = args.training_dir.resolve()
    readiness = json.loads((dataset / "readiness_report.json").read_text())
    if readiness.get("phase") != "evaluation":
        raise ValueError("held-out evaluation requires an evaluation-phase dataset")
    rows = read_csv(dataset / "sample_manifest.csv")
    proposal_rows = read_csv(dataset / "proposal_manifest.csv")
    arrays = np.load(dataset / "dataset_arrays.npz")
    features = arrays["features"]
    proposal_features = arrays["proposal_features"]
    results: dict[str, Any] = {}
    failed = False
    for level in training.LEVELS:
        if readiness["readiness"][level]["dataset_status"] != "READY":
            results[level] = {
                "status": "SAP_TRAINING_INCONCLUSIVE",
                "reason": "evaluation simulator-proxy labels failed the frozen readiness gate",
                "deficits": readiness["readiness"][level]["deficits"],
            }
            failed = True
            continue
        checkpoint = training_dir / f"s0010_{level.lower()}_sap_classifier_v1.joblib"
        if not checkpoint.exists():
            results[level] = {"status": "SAP_TRAINING_BLOCKED_DATA", "reason": "checkpoint absent"}
            failed = True
            continue
        bundle = joblib.load(checkpoint)
        estimator = bundle["estimator"]
        threshold = float(bundle["threshold"])
        selected = [
            row
            for row in rows
            if row["conditioned_level"] == level and row["split"] == "evaluation"
        ]
        selected_x = np.asarray([features[int(row["feature_index"])] for row in selected])
        selected_y = np.asarray(
            [training.LABEL_ID[row["provisional_label"]] for row in selected], dtype=np.int64
        )
        positive_column = list(estimator.classes_).index(1)
        patch_scores = estimator.predict_proba(selected_x)[:, positive_column]
        image_labels, image_scores, hashes = training.full_image_scores(
            estimator=estimator,
            level=level,
            split="evaluation",
            sample_rows=rows,
            proposal_rows=proposal_rows,
            proposal_features=proposal_features,
        )
        image_metrics = training.metrics(image_labels, image_scores, threshold)
        hash_metadata = {
            row["frame_sha256"]: {
                "session_id": row["session_id"],
                "frame_sequence": int(row["frame_sequence"]),
            }
            for row in selected
        }
        image_predictions = []
        for frame_hash, label, score in zip(hashes, image_labels, image_scores):
            predicted = int(score >= threshold)
            image_predictions.append(
                {
                    "frame_sha256": frame_hash,
                    **hash_metadata[frame_hash],
                    "label": int(label),
                    "score": float(score),
                    "predicted": predicted,
                    "outcome": (
                        "TP" if label == 1 and predicted == 1
                        else "FN" if label == 1
                        else "FP" if predicted == 1
                        else "TN"
                    ),
                }
            )
        heuristic_threshold = bundle.get("heuristic_gradient_threshold")
        heuristic: dict[str, Any]
        if heuristic_threshold is None:
            heuristic = {"status": "CALIBRATION_NO_THRESHOLD"}
        else:
            heuristic_labels, heuristic_scores = training.full_image_heuristic_scores(
                level=level,
                split="evaluation",
                sample_rows=rows,
                proposal_rows=proposal_rows,
            )
            heuristic = {
                "status": "EVALUATED",
                "threshold": float(heuristic_threshold),
                "metrics": training.metrics(
                    heuristic_labels, heuristic_scores, float(heuristic_threshold)
                ),
            }
        status = (
            "SAP_TRAINED_HELDOUT_PASS"
            if image_metrics["gate_pass"]
            else "SAP_TRAINED_HELDOUT_FAIL"
        )
        failed |= status != "SAP_TRAINED_HELDOUT_PASS"
        results[level] = {
            "status": status,
            "threshold": threshold,
            "full_image_metrics": image_metrics,
            "patch_metrics": training.metrics(selected_y, patch_scores, threshold),
            "per_session": session_metrics(
                image_labels, image_scores, hashes, rows, level, threshold
            ),
            "candidate_ranking": top_k_localization(
                estimator=estimator,
                level=level,
                sample_rows=rows,
                proposal_rows=proposal_rows,
                proposal_features=proposal_features,
            ),
            "gradient_baseline": heuristic,
            "evaluated_image_hashes": hashes,
            "full_image_predictions": image_predictions,
            "geometry_used_as_model_input": False,
        }
    report = {
        "schema": "s0010-sap-heldout-evaluation-v1",
        "evaluation_opened_once": True,
        "levels": results,
        "overall_status": "PASS" if not failed else "FAIL_OR_INCONCLUSIVE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if not failed else 3


if __name__ == "__main__":
    raise SystemExit(main())
