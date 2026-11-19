#!/usr/bin/env python3
"""Fit and evaluate the frozen L4 one-vs-rest targeted supplement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


ROOT = Path(__file__).resolve().parents[2]
ABLATION_PATH = ROOT / "workflows/two_stage_navigation/run_tp_surface_core_l4_existing_data_ablation.py"
SPEC = importlib.util.spec_from_file_location("targeted_supplement_ablation", ABLATION_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the frozen ablation implementation")
ABLATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ABLATION
SPEC.loader.exec_module(ABLATION)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-training-dir", type=Path, required=True)
    parser.add_argument("--old-hard-negatives", type=Path, required=True)
    parser.add_argument("--new-dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def new_proposal_state(row: dict[str, str]) -> tuple[str, str]:
    if row["projection_status"] != "VERIFIED":
        return ABLATION.TRISTATE.PROXY_AMBIGUOUS, "projection_not_verified"
    l4 = int(row["l4_core_points_in_patch"])
    l5 = int(row["l5_core_points_in_patch"])
    l4u = int(row["l4_uncertain_points_in_patch"])
    l5u = int(row["l5_uncertain_points_in_patch"])
    positive = bool(
        row["candidate_id"] == ABLATION.CLASS_NAMES[1]
        and row["review_target"] == "REVIEW_POSITIVE"
        and l4 > 0
        and l5 == 0
        and l4u == 0
        and l5u == 0
        and row["adequate_projected_support"].lower() == "true"
    )
    if positive:
        return ABLATION.TRISTATE.PROXY_POSITIVE, "complete_l4_proxy_contract"
    if l4u > 0 or (l4 > 0 and not positive):
        return ABLATION.TRISTATE.PROXY_AMBIGUOUS, "l4_evidence_not_decisive"
    return ABLATION.TRISTATE.PROXY_NEGATIVE, "zero_l4_core_and_uncertainty"


def add_states(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    by_frame: dict[str, list[str]] = defaultdict(list)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        state, reason = new_proposal_state(row)
        item = dict(row, l4_proxy_state=state, l4_proxy_state_reason=reason)
        enriched.append(item)
        by_frame[row["frame_sha256"]].append(state)
    states = {
        frame: ABLATION.TRISTATE.image_state(values) for frame, values in by_frame.items()
    }
    return enriched, states


def best_by_frame(rows: list[dict[str, Any]], key) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["frame_sha256"]].append(row)
    return [min(values, key=key) for _, values in sorted(grouped.items())]


def select_new_training_rows(
    rows: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    frozen: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    train = [row for row in rows if row["split"] == "train"]
    positives = best_by_frame(
        [row for row in train if row["l4_proxy_state"] == ABLATION.TRISTATE.PROXY_POSITIVE],
        lambda row: (
            -int(row["interior_core_feature_cells"]),
            -int(row["unique_core_pixels"]),
            row["sample_id"],
        ),
    )
    negatives = [row for row in train if row["l4_proxy_state"] == ABLATION.TRISTATE.PROXY_NEGATIVE]
    matrix = np.stack([features[row["sample_id"]] for row in negatives])
    accepted, scores = ABLATION.l4_accepts(frozen, matrix, 0.5, target="three_class")
    scored = []
    for row, is_accepted, score in zip(negatives, accepted, scores):
        l5_positive = bool(
            int(row["l5_core_points_in_patch"]) > 0
            and int(row["l4_core_points_in_patch"]) == 0
            and int(row["l4_uncertain_points_in_patch"]) == 0
            and int(row["l5_uncertain_points_in_patch"]) == 0
            and row["adequate_projected_support"].lower() == "true"
        )
        if bool(is_accepted) or l5_positive:
            scored.append(
                {
                    **row,
                    "frozen_l4_probability": float(score),
                    "negative_subtype": "FROZEN_L4_FALSE_TRIGGER"
                    if bool(is_accepted)
                    else "COMPETING_L5_CORE",
                }
            )
    chosen_negatives = best_by_frame(
        scored,
        lambda row: (
            0 if row["negative_subtype"] == "FROZEN_L4_FALSE_TRIGGER" else 1,
            -float(row["frozen_l4_probability"]),
            row["sample_id"],
        ),
    )
    selected = [
        {**row, "target": 1, "sample_weight": 1.0, "source_role": "new_l4_positive"}
        for row in positives
    ] + [
        {**row, "target": 0, "sample_weight": 1.0, "source_role": row["negative_subtype"]}
        for row in chosen_negatives
    ]
    return selected, {
        "new_positive_images": len(positives),
        "new_negative_images": len(chosen_negatives),
        "new_false_trigger_images": sum(
            row["negative_subtype"] == "FROZEN_L4_FALSE_TRIGGER" for row in chosen_negatives
        ),
        "new_competing_l5_images": sum(
            row["negative_subtype"] == "COMPETING_L5_CORE" for row in chosen_negatives
        ),
    }


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    old_dir = args.old_training_dir.resolve()
    old_audit = read_csv(old_dir / "proxy_label_audit.csv")
    old_selected = read_csv(old_dir / "proxy_sample_manifest.csv")
    old_hard = read_csv(args.old_hard_negatives.resolve())
    new_rows_raw = read_csv(args.new_dataset_manifest.resolve())
    new_rows, new_states = add_states(new_rows_raw)

    old_features = ABLATION.load_features(old_audit)
    new_features = ABLATION.load_features(new_rows_raw)
    frozen_bundle = joblib.load(old_dir / "tp_surface_core_simulator_proxy_extratrees.joblib")
    frozen = frozen_bundle["estimator"]
    old_arm = {"target": "l4_one_vs_rest", "hard_negatives": "include_as_l4_negative", "hard_negative_weight": 1.0}
    old_train = ABLATION.build_training_rows(old_selected, old_hard, old_arm)
    new_train, supplement_counts = select_new_training_rows(new_rows, new_features, frozen)
    training_rows = old_train + new_train
    if any(row["split"] != "train" for row in training_rows):
        raise RuntimeError("non-training sample reached fit")
    features = {**old_features, **new_features}
    estimator = ExtraTreesClassifier(
        n_estimators=240,
        class_weight="balanced",
        random_state=20261119,
        n_jobs=4,
    )
    matrix = np.stack([features[row["sample_id"]] for row in training_rows])
    targets = np.asarray([int(row["target"]) for row in training_rows])
    weights = np.asarray([float(row["sample_weight"]) for row in training_rows])
    estimator.fit(matrix, targets, sample_weight=weights)

    baseline_cal, _ = ABLATION.evaluate_l4(
        new_rows, new_features, new_states, frozen, 0.5,
        target="three_class", split="calibration", arm_id="frozen_v1"
    )
    candidate_cal, _ = ABLATION.evaluate_l4(
        new_rows, new_features, new_states, estimator, 0.5,
        target="l4_one_vs_rest", split="calibration", arm_id="targeted_candidate"
    )
    calibration_pass = bool(
        candidate_cal["tp"] >= baseline_cal["tp"]
        and candidate_cal["fp"] <= baseline_cal["fp"]
        and candidate_cal["ambiguous_activation_images"]
        <= baseline_cal["ambiguous_activation_images"]
    )
    result: dict[str, Any] = {
        "schema": "sonogym-tp-surface-core-proxy-l4-targeted-supplement-v1",
        "status": "CALIBRATION_NO_GO",
        "fit_sample_count": len(training_rows),
        "fit_unique_source_images": len({row["frame_sha256"] for row in training_rows}),
        "supplement_training": supplement_counts,
        "baseline_calibration": baseline_cal,
        "candidate_calibration": candidate_cal,
        "calibration_pass": calibration_pass,
        "development_used_for_fit_or_calibration": False,
        "threshold": 0.5,
        "l5_modified": False,
    }
    checkpoint = output / "l4_targeted_supplement_candidate.joblib"
    joblib.dump(
        {
            "schema": result["schema"],
            "estimator": estimator,
            "threshold": 0.5,
            "patch_spec": asdict(ABLATION.PATCH_SPEC),
            "fit_sample_ids": [row["sample_id"] for row in training_rows],
            "frozen_l5_checkpoint_sha256": ABLATION.sha256(
                old_dir / "tp_surface_core_simulator_proxy_extratrees.joblib"
            ),
        },
        checkpoint,
    )
    reloaded = joblib.load(checkpoint)["estimator"]
    if not np.array_equal(estimator.predict_proba(matrix[:32]), reloaded.predict_proba(matrix[:32])):
        raise RuntimeError("checkpoint reload changed probabilities")
    result["checkpoint_sha256"] = ABLATION.sha256(checkpoint)
    result["checkpoint_reload_verified"] = True

    if calibration_pass:
        baseline_eval, baseline_records = ABLATION.evaluate_l4(
            new_rows, new_features, new_states, frozen, 0.5,
            target="three_class", split="development_evaluation", arm_id="frozen_v1"
        )
        candidate_eval, candidate_records = ABLATION.evaluate_l4(
            new_rows, new_features, new_states, estimator, 0.5,
            target="l4_one_vs_rest", split="development_evaluation", arm_id="targeted_candidate"
        )
        required_tp = math.ceil(0.9 * baseline_eval["tp"])
        enough_evidence = baseline_eval["tp"] >= 3 and baseline_eval["fp"] >= 2
        success = bool(
            enough_evidence
            and candidate_eval["tp"] >= required_tp
            and candidate_eval["fp"] < baseline_eval["fp"]
            and candidate_eval["ambiguous_activation_images"]
            <= baseline_eval["ambiguous_activation_images"]
        )
        write_csv(output / "sealed_evaluation_image_decisions.csv", baseline_records + candidate_records)
        result.update(
            status="IMPROVEMENT_DEMONSTRATED" if success else "SCIENTIFIC_NO_GO_WITH_DIAGNOSIS",
            baseline_sealed_evaluation=baseline_eval,
            candidate_sealed_evaluation=candidate_eval,
            minimum_required_candidate_tp=required_tp,
            sealed_evaluation_evidence_sufficient=enough_evidence,
            improvement_demonstrated=success,
        )
    write_csv(output / "supplement_training_samples.csv", new_train)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "calibration_pass": calibration_pass}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
