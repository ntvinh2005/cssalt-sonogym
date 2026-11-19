#!/usr/bin/env python3
"""Fit the frozen L4 coverage-repair ablation and evaluate it fail-closed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesClassifier


ROOT = Path(__file__).resolve().parents[2]
TARGETED_PATH = ROOT / "workflows/two_stage_navigation/train_tp_surface_core_l4_targeted_supplement.py"
SPEC = importlib.util.spec_from_file_location("l4_targeted_supplement", TARGETED_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load targeted supplement implementation")
TARGETED = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TARGETED
SPEC.loader.exec_module(TARGETED)
ABLATION = TARGETED.ABLATION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-training-dir", type=Path, required=True)
    parser.add_argument("--old-hard-negatives", type=Path, required=True)
    parser.add_argument("--new-dataset-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def proxy_image_counts(
    states: dict[str, str], rows: list[dict[str, Any]], split: str
) -> dict[str, int]:
    frame_hashes = {
        row["frame_sha256"] for row in rows if str(row["split"]) == split
    }
    counts = Counter(states[frame] for frame in frame_hashes)
    return {
        "images": len(frame_hashes),
        "positive": counts[ABLATION.TRISTATE.PROXY_POSITIVE],
        "negative": counts[ABLATION.TRISTATE.PROXY_NEGATIVE],
        "ambiguous": counts[ABLATION.TRISTATE.PROXY_AMBIGUOUS],
    }


def coverage_passes(counts: dict[str, dict[str, int]], protocol: dict[str, Any]) -> bool:
    gate = protocol["dataset_gate"]
    return all(
        counts[split]["positive"] >= int(gate["minimum_l4_proxy_positive_images"][split])
        and counts[split]["negative"] >= int(gate["minimum_clear_confuser_images"][split])
        for split in counts
    )


def fit_variant(
    rows: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    *,
    arm_id: str,
    positive_weight: float,
) -> ExtraTreesClassifier:
    if any(row["split"] != "train" for row in rows):
        raise RuntimeError("non-training sample reached fit")
    matrix = np.stack([features[row["sample_id"]] for row in rows])
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    weights = np.asarray(
        [
            float(row["sample_weight"]) * (positive_weight if int(row["target"]) == 1 else 1.0)
            for row in rows
        ],
        dtype=np.float64,
    )
    estimator = ExtraTreesClassifier(
        n_estimators=240,
        class_weight="balanced",
        random_state=20261123,
        n_jobs=4,
    )
    estimator.fit(matrix, targets, sample_weight=weights)
    estimator.coverage_repair_arm_id = arm_id
    return estimator


def calibration_eligible(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    return bool(
        candidate["tp"] >= baseline["tp"]
        and candidate["fp"] <= baseline["fp"]
        and candidate["ambiguous_activation_images"]
        <= baseline["ambiguous_activation_images"]
    )


def choose_candidate(
    baseline: dict[str, Any], candidates: dict[str, dict[str, Any]]
) -> str | None:
    eligible = [
        arm_id
        for arm_id, metrics in candidates.items()
        if calibration_eligible(baseline, metrics)
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda arm_id: (
            -int(candidates[arm_id]["tp"]),
            int(candidates[arm_id]["fp"]),
            int(candidates[arm_id]["ambiguous_activation_images"]),
            arm_id,
        ),
    )


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.protocol.resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != "sonogym-s0010-tp-surface-core-dataset-v13":
        raise ValueError("coverage repair requires the sealed v13 protocol")
    expected_seal = protocol_path.with_suffix(".sha256").read_text().split()[0]
    if sha256(protocol_path) != expected_seal:
        raise ValueError("v13 protocol seal mismatch")

    old_dir = args.old_training_dir.resolve()
    old_audit = TARGETED.read_csv(old_dir / "proxy_label_audit.csv")
    old_selected = TARGETED.read_csv(old_dir / "proxy_sample_manifest.csv")
    old_hard = TARGETED.read_csv(args.old_hard_negatives.resolve())
    new_rows_raw = TARGETED.read_csv(args.new_dataset_manifest.resolve())
    allowed_splits = {"train", "calibration", "development_evaluation"}
    if {row["split"] for row in new_rows_raw} - allowed_splits:
        raise ValueError("new dataset contains an unexpected split")

    design_rows_raw = [
        row for row in new_rows_raw if row["split"] in {"train", "calibration"}
    ]
    design_rows, design_states = TARGETED.add_states(design_rows_raw)
    design_counts = {
        split: proxy_image_counts(design_states, design_rows, split)
        for split in ("train", "calibration")
    }
    result: dict[str, Any] = {
        "schema": "sonogym-tp-surface-core-proxy-l4-coverage-repair-v1",
        "status": "DATASET_COVERAGE_NO_GO",
        "protocol_sha256": expected_seal,
        "design_coverage": design_counts,
        "threshold": float(protocol["model_comparison_gate"]["threshold"]),
        "development_used_for_fit_or_calibration": False,
        "development_feature_rows_loaded_before_calibration": 0,
        "l5_modified": False,
    }
    if not coverage_passes(design_counts, protocol):
        (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": result["status"]}))
        return 0

    old_features = ABLATION.load_features(old_audit)
    design_features = ABLATION.load_features(design_rows_raw)
    frozen_bundle = joblib.load(old_dir / "tp_surface_core_simulator_proxy_extratrees.joblib")
    frozen = frozen_bundle["estimator"]
    old_arm = {
        "target": "l4_one_vs_rest",
        "hard_negatives": "include_as_l4_negative",
        "hard_negative_weight": 1.0,
    }
    old_train = ABLATION.build_training_rows(old_selected, old_hard, old_arm)
    new_train, supplement_counts = TARGETED.select_new_training_rows(
        design_rows, design_features, frozen
    )
    training_rows = old_train + new_train
    features = {**old_features, **design_features}
    variants = {
        "coverage_balanced": fit_variant(
            training_rows, features, arm_id="coverage_balanced", positive_weight=1.0
        ),
        "coverage_positive_1p5": fit_variant(
            training_rows, features, arm_id="coverage_positive_1p5", positive_weight=1.5
        ),
    }
    threshold = float(protocol["model_comparison_gate"]["threshold"])
    baseline_cal, _ = ABLATION.evaluate_l4(
        design_rows,
        design_features,
        design_states,
        frozen,
        threshold,
        target="three_class",
        split="calibration",
        arm_id="frozen_v1",
    )
    candidate_calibration: dict[str, dict[str, Any]] = {}
    for arm_id, estimator in variants.items():
        candidate_calibration[arm_id], _ = ABLATION.evaluate_l4(
            design_rows,
            design_features,
            design_states,
            estimator,
            threshold,
            target="l4_one_vs_rest",
            split="calibration",
            arm_id=arm_id,
        )
    selected_arm = choose_candidate(baseline_cal, candidate_calibration)
    result.update(
        status="CALIBRATION_NO_GO" if selected_arm is None else "CALIBRATION_PASS",
        fit_sample_count=len(training_rows),
        fit_unique_source_images=len({row["frame_sha256"] for row in training_rows}),
        supplement_training=supplement_counts,
        baseline_calibration=baseline_cal,
        candidate_calibration=candidate_calibration,
        selected_arm=selected_arm,
        l5_frozen_control_calibration=TARGETED.evaluate_frozen_l5(
            design_rows_raw, design_features, frozen, split="calibration"
        ),
    )
    TARGETED.write_csv(output / "supplement_training_samples.csv", new_train)
    if selected_arm is None:
        (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": result["status"]}))
        return 0

    estimator = variants[selected_arm]
    checkpoint = output / "l4_coverage_repair_candidate.joblib"
    joblib.dump(
        {
            "schema": result["schema"],
            "estimator": estimator,
            "threshold": threshold,
            "selected_arm": selected_arm,
            "patch_spec": asdict(ABLATION.PATCH_SPEC),
            "fit_sample_ids": [row["sample_id"] for row in training_rows],
            "frozen_l5_checkpoint_sha256": ABLATION.sha256(
                old_dir / "tp_surface_core_simulator_proxy_extratrees.joblib"
            ),
        },
        checkpoint,
    )
    matrix = np.stack([features[row["sample_id"]] for row in training_rows[:32]])
    reloaded = joblib.load(checkpoint)["estimator"]
    if not np.array_equal(estimator.predict_proba(matrix), reloaded.predict_proba(matrix)):
        raise RuntimeError("checkpoint reload changed probabilities")
    result["checkpoint_sha256"] = sha256(checkpoint)
    result["checkpoint_reload_verified"] = True

    # Sealed development is first materialized after candidate selection.
    evaluation_rows_raw = [
        row for row in new_rows_raw if row["split"] == "development_evaluation"
    ]
    evaluation_rows, evaluation_states = TARGETED.add_states(evaluation_rows_raw)
    evaluation_counts = {
        "development_evaluation": proxy_image_counts(
            evaluation_states, evaluation_rows, "development_evaluation"
        )
    }
    result["sealed_coverage"] = evaluation_counts["development_evaluation"]
    if not coverage_passes(evaluation_counts, protocol):
        result["status"] = "SEALED_DATASET_COVERAGE_NO_GO"
        (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": result["status"]}))
        return 0
    evaluation_features = ABLATION.load_features(evaluation_rows_raw)
    baseline_eval, baseline_records = ABLATION.evaluate_l4(
        evaluation_rows,
        evaluation_features,
        evaluation_states,
        frozen,
        threshold,
        target="three_class",
        split="development_evaluation",
        arm_id="frozen_v1",
    )
    candidate_eval, candidate_records = ABLATION.evaluate_l4(
        evaluation_rows,
        evaluation_features,
        evaluation_states,
        estimator,
        threshold,
        target="l4_one_vs_rest",
        split="development_evaluation",
        arm_id=selected_arm,
    )
    gate = protocol["model_comparison_gate"]["sealed_evaluation"]
    enough_evidence = bool(
        baseline_eval["tp"] >= int(gate["minimum_baseline_tp"])
        and baseline_eval["fp"] >= int(gate["minimum_baseline_fp"])
    )
    required_tp = math.ceil(
        float(gate["minimum_tp_retention_fraction"]) * baseline_eval["tp"]
    )
    success = bool(
        enough_evidence
        and candidate_eval["tp"] >= required_tp
        and candidate_eval["fp"] < baseline_eval["fp"]
        and candidate_eval["ambiguous_activation_images"]
        <= baseline_eval["ambiguous_activation_images"]
    )
    result.update(
        status="IMPROVEMENT_DEMONSTRATED" if success else "SCIENTIFIC_NO_GO_WITH_DIAGNOSIS",
        baseline_sealed_evaluation=baseline_eval,
        candidate_sealed_evaluation=candidate_eval,
        minimum_required_candidate_tp=required_tp,
        sealed_evaluation_evidence_sufficient=enough_evidence,
        improvement_demonstrated=success,
        l5_frozen_control_sealed_evaluation=TARGETED.evaluate_frozen_l5(
            evaluation_rows_raw,
            evaluation_features,
            frozen,
            split="development_evaluation",
        ),
        development_feature_rows_loaded_after_calibration=len(evaluation_rows_raw),
    )
    TARGETED.write_csv(
        output / "sealed_evaluation_image_decisions.csv",
        baseline_records + candidate_records,
    )
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "selected_arm": selected_arm}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
