#!/usr/bin/env python3
"""Run the frozen L4 existing-data ablation and opened-development diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesClassifier


ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = (
    ROOT
    / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/stage1_features.py"
)
FEATURE_SPEC = importlib.util.spec_from_file_location("tp_proxy_features", FEATURE_PATH)
if FEATURE_SPEC is None or FEATURE_SPEC.loader is None:
    raise RuntimeError("cannot load production patch features")
FEATURES = importlib.util.module_from_spec(FEATURE_SPEC)
sys.modules[FEATURE_SPEC.name] = FEATURES
FEATURE_SPEC.loader.exec_module(FEATURES)
PATCH_SPEC = FEATURES.UNITY_REFERENCE_PATCH_SPEC
extract_patch_feature = FEATURES.extract_patch_feature

TRISTATE_PATH = ROOT / "workflows/two_stage_navigation/prepare_tp_surface_core_proxy_tristate_hard_negatives.py"
TRISTATE_SPEC = importlib.util.spec_from_file_location("tp_proxy_tristate", TRISTATE_PATH)
if TRISTATE_SPEC is None or TRISTATE_SPEC.loader is None:
    raise RuntimeError("cannot load tri-state definitions")
TRISTATE = importlib.util.module_from_spec(TRISTATE_SPEC)
sys.modules[TRISTATE_SPEC.name] = TRISTATE
TRISTATE_SPEC.loader.exec_module(TRISTATE)

CLASS_L4 = 1
CLASS_L5 = 2
CLASS_NAMES = {
    0: "BACKGROUND_OR_OTHER_STRUCTURE",
    1: "L4_RIGHT_TP_SURFACE_CORE",
    2: "L5_RIGHT_TP_SURFACE_CORE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--hard-negative-manifest", type=Path, required=True)
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
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_inputs(
    protocol: dict[str, Any],
    training_dir: Path,
    dataset_manifest: Path,
    hard_negative_manifest: Path,
) -> None:
    paths = {
        training_dir / "tp_surface_core_simulator_proxy_extratrees.joblib": protocol[
            "source_checkpoint_sha256"
        ],
        dataset_manifest: protocol["source_dataset_manifest_sha256"],
        training_dir / "proxy_label_audit.csv": protocol[
            "source_proxy_label_audit_sha256"
        ],
        training_dir / "proxy_sample_manifest.csv": protocol[
            "source_proxy_sample_manifest_sha256"
        ],
        hard_negative_manifest: protocol["source_hard_negative_manifest_sha256"],
    }
    for path, expected in paths.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"frozen input SHA-256 mismatch for {path}: {actual}")


def load_features(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    archives: dict[Path, np.ndarray] = {}
    features: dict[str, np.ndarray] = {}
    for row in rows:
        archive = Path(row["image_archive"])
        if archive not in archives:
            archives[archive] = np.load(archive)["ultrasound"]
        image = np.asarray(archives[archive][int(row["image_archive_index"])])
        if image.shape != (200, 150) or ultrasound_sha256(image) != row["frame_sha256"]:
            raise ValueError(f"source image mismatch: {row['sample_id']}")
        feature = extract_patch_feature(
            image, float(row["patch_row"]), float(row["patch_column"]), PATCH_SPEC
        )
        if feature is None or not np.all(np.isfinite(feature)):
            raise ValueError(f"invalid feature: {row['sample_id']}")
        expected = row.get("feature_sha256", "")
        if expected and array_sha256(feature) != expected:
            raise ValueError(f"feature hash mismatch: {row['sample_id']}")
        features[row["sample_id"]] = feature
    return features


def assign_states(rows: list[dict[str, str]]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    by_frame: dict[str, list[str]] = defaultdict(list)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        state, reason = TRISTATE.proposal_state(row)
        item = dict(row, l4_proxy_state=state, l4_proxy_state_reason=reason)
        enriched.append(item)
        by_frame[row["frame_sha256"]].append(state)
    return (
        {frame: TRISTATE.image_state(states) for frame, states in by_frame.items()},
        enriched,
    )


def build_training_rows(
    selected: list[dict[str, str]],
    hard_negatives: list[dict[str, str]],
    arm: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in selected:
        if row["split"] != "train":
            continue
        class_id = int(row["proxy_class_id"])
        target = class_id if arm["target"] == "three_class" else int(class_id == CLASS_L4)
        rows.append({**row, "target": target, "sample_weight": 1.0, "source_role": "selected_v1"})
    if arm["hard_negatives"] != "exclude":
        for row in hard_negatives:
            if row["split"] != "train" or row["l4_proxy_state"] != TRISTATE.PROXY_NEGATIVE:
                raise ValueError("hard-negative manifest contains an inadmissible row")
            target = 0
            rows.append(
                {
                    **row,
                    "target": target,
                    "sample_weight": float(arm["hard_negative_weight"]),
                    "source_role": "hard_negative",
                }
            )
    return rows


def fit_arm(
    rows: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    estimator_spec: dict[str, Any],
) -> ExtraTreesClassifier:
    if any(row["split"] != "train" for row in rows):
        raise ValueError("non-training row reached fit")
    estimator = ExtraTreesClassifier(**estimator_spec)
    x = np.stack([features[row["sample_id"]] for row in rows])
    y = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    weights = np.asarray([float(row["sample_weight"]) for row in rows], dtype=np.float64)
    estimator.fit(x, y, sample_weight=weights)
    return estimator


def l4_probabilities(estimator: Any, features: np.ndarray) -> np.ndarray:
    probabilities = estimator.predict_proba(features)
    columns = np.flatnonzero(np.asarray(estimator.classes_) == CLASS_L4)
    if len(columns) != 1:
        raise ValueError("estimator does not contain L4 class")
    return probabilities[:, int(columns[0])]


def l4_accepts(
    estimator: Any,
    features: np.ndarray,
    threshold: float,
    *,
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = estimator.predict_proba(features)
    classes = np.asarray(estimator.classes_)
    column = int(np.flatnonzero(classes == CLASS_L4)[0])
    scores = probabilities[:, column]
    if target == "three_class":
        accepted = (classes[np.argmax(probabilities, axis=1)] == CLASS_L4) & (
            scores >= threshold
        )
    else:
        accepted = scores >= threshold
    return accepted, scores


def evaluate_l4(
    rows: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    states: dict[str, str],
    estimator: Any,
    threshold: float,
    *,
    target: str,
    split: str,
    arm_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            by_frame[row["frame_sha256"]].append(row)
    records: list[dict[str, Any]] = []
    for frame, proposals in sorted(by_frame.items()):
        matrix = np.stack([features[row["sample_id"]] for row in proposals])
        accepted, scores = l4_accepts(estimator, matrix, threshold, target=target)
        records.append(
            {
                "arm_id": arm_id,
                "split": split,
                "session_id": proposals[0]["session_id"],
                "frame_sha256": frame,
                "l4_proxy_state": states[frame],
                "predicted_l4": bool(np.any(accepted)),
                "accepted_l4_proposal_count": int(np.count_nonzero(accepted)),
                "maximum_l4_probability": float(np.max(scores)),
            }
        )
    primary = [row for row in records if row["l4_proxy_state"] != TRISTATE.PROXY_AMBIGUOUS]
    tp = sum(row["l4_proxy_state"] == TRISTATE.PROXY_POSITIVE and row["predicted_l4"] for row in primary)
    fp = sum(row["l4_proxy_state"] == TRISTATE.PROXY_NEGATIVE and row["predicted_l4"] for row in primary)
    fn = sum(row["l4_proxy_state"] == TRISTATE.PROXY_POSITIVE and not row["predicted_l4"] for row in primary)
    tn = sum(row["l4_proxy_state"] == TRISTATE.PROXY_NEGATIVE and not row["predicted_l4"] for row in primary)
    ambiguous = [row for row in records if row["l4_proxy_state"] == TRISTATE.PROXY_AMBIGUOUS]
    metrics = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
        "clean_negative_false_detections": sum(
            row["accepted_l4_proposal_count"]
            for row in primary
            if row["l4_proxy_state"] == TRISTATE.PROXY_NEGATIVE
        ),
        "ambiguous_image_count": len(ambiguous),
        "ambiguous_activation_images": sum(row["predicted_l4"] for row in ambiguous),
        "image_count": len(records),
    }
    return metrics, records


def choose_candidate(
    calibration: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
) -> str | None:
    gate = protocol["calibration_selection"]
    eligible = [
        arm["arm_id"]
        for arm in protocol["arms"]
        if calibration[arm["arm_id"]]["tp"] >= int(gate["minimum_true_positive_images"])
        and calibration[arm["arm_id"]]["fp"]
        <= int(gate["maximum_clean_negative_false_positive_images"])
    ]
    if not eligible:
        return None
    arm_order = {arm["arm_id"]: index for index, arm in enumerate(protocol["arms"])}
    return min(
        eligible,
        key=lambda arm_id: (
            -calibration[arm_id]["tp"],
            calibration[arm_id]["fp"],
            calibration[arm_id]["clean_negative_false_detections"],
            arm_order[arm_id],
        ),
    )


def l5_control_metrics(
    rows: list[dict[str, Any]],
    selected: list[dict[str, str]],
    features: dict[str, np.ndarray],
    estimator: Any,
    threshold: float,
    split: str,
) -> dict[str, int]:
    truth = {
        row["frame_sha256"]
        for row in selected
        if row["split"] == split and int(row["proxy_class_id"]) == CLASS_L5
    }
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            by_frame[row["frame_sha256"]].append(row)
    predicted: set[str] = set()
    classes = np.asarray(estimator.classes_)
    column = int(np.flatnonzero(classes == CLASS_L5)[0])
    for frame, proposals in by_frame.items():
        probabilities = estimator.predict_proba(
            np.stack([features[row["sample_id"]] for row in proposals])
        )
        accepted = (classes[np.argmax(probabilities, axis=1)] == CLASS_L5) & (
            probabilities[:, column] >= threshold
        )
        if np.any(accepted):
            predicted.add(frame)
    frames = set(by_frame)
    return {
        "tp": len(truth & predicted),
        "fp": len(predicted - truth),
        "fn": len(truth - predicted),
        "tn": len(frames - truth - predicted),
    }


def session_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_session[row["session_id"]].append(row)
    for session, rows in sorted(by_session.items()):
        primary = [row for row in rows if row["l4_proxy_state"] != TRISTATE.PROXY_AMBIGUOUS]
        output.append(
            {
                "arm_id": rows[0]["arm_id"],
                "session_id": session,
                "image_count": len(rows),
                "tp": sum(row["l4_proxy_state"] == TRISTATE.PROXY_POSITIVE and row["predicted_l4"] for row in primary),
                "fp": sum(row["l4_proxy_state"] == TRISTATE.PROXY_NEGATIVE and row["predicted_l4"] for row in primary),
                "fn": sum(row["l4_proxy_state"] == TRISTATE.PROXY_POSITIVE and not row["predicted_l4"] for row in primary),
                "ambiguous_activations": sum(row["l4_proxy_state"] == TRISTATE.PROXY_AMBIGUOUS and row["predicted_l4"] for row in rows),
            }
        )
    return output


def main() -> int:
    args = parse_args()
    training_dir = args.training_dir.resolve()
    dataset_manifest = args.dataset_manifest.resolve()
    hard_negative_manifest = args.hard_negative_manifest.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    validate_inputs(protocol, training_dir, dataset_manifest, hard_negative_manifest)

    audit_rows = read_csv(training_dir / "proxy_label_audit.csv")
    dataset_rows = read_csv(dataset_manifest)
    if [row["sample_id"] for row in audit_rows] != [row["sample_id"] for row in dataset_rows]:
        raise ValueError("proxy audit and source dataset row identities differ")
    selected = read_csv(training_dir / "proxy_sample_manifest.csv")
    hard_negatives = read_csv(hard_negative_manifest)
    states, enriched = assign_states(audit_rows)
    all_rows_by_id = {row["sample_id"]: row for row in audit_rows}
    feature_rows = list(audit_rows)
    for row in hard_negatives:
        if row["sample_id"] not in all_rows_by_id:
            raise ValueError("hard-negative row is absent from the frozen audit")
    features = load_features(feature_rows)

    frozen_bundle = joblib.load(
        training_dir / "tp_surface_core_simulator_proxy_extratrees.joblib"
    )
    frozen = frozen_bundle["estimator"]
    threshold = float(protocol["base_l4_threshold"])
    baseline_calibration, baseline_calibration_records = evaluate_l4(
        enriched,
        features,
        states,
        frozen,
        threshold,
        target="three_class",
        split="calibration",
        arm_id="frozen_v1",
    )
    expected_calibration = protocol["expected_frozen_baseline"]["calibration"]
    observed_calibration = {
        "tp": baseline_calibration["tp"],
        "fp": baseline_calibration["fp"],
        "fn": baseline_calibration["fn"],
        "tn": baseline_calibration["tn"],
        "ambiguous_activations": baseline_calibration["ambiguous_activation_images"],
    }
    if observed_calibration != expected_calibration:
        raise RuntimeError(f"frozen calibration baseline changed: {observed_calibration}")

    estimators: dict[str, ExtraTreesClassifier] = {}
    training_rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    calibration: dict[str, dict[str, Any]] = {}
    calibration_records = list(baseline_calibration_records)
    for arm in protocol["arms"]:
        arm_id = arm["arm_id"]
        training_rows = build_training_rows(selected, hard_negatives, arm)
        estimator = fit_arm(training_rows, features, protocol["estimator"])
        metrics, records = evaluate_l4(
            enriched,
            features,
            states,
            estimator,
            threshold,
            target=arm["target"],
            split="calibration",
            arm_id=arm_id,
        )
        estimators[arm_id] = estimator
        training_rows_by_arm[arm_id] = training_rows
        calibration[arm_id] = metrics
        calibration_records.extend(records)

    chosen = choose_candidate(calibration, protocol)
    result: dict[str, Any] = {
        "schema": "sonogym-tp-surface-core-proxy-l4-existing-data-ablation-v1",
        "experiment_id": protocol["experiment_id"],
        "status": "CALIBRATION_NO_GO" if chosen is None else "OPENED_DEVELOPMENT_EVALUATED",
        "frozen_checkpoint_sha256": protocol["source_checkpoint_sha256"],
        "base_l4_threshold": threshold,
        "fit_split": "train",
        "selection_split": "calibration",
        "opened_diagnostic_split": "development_evaluation",
        "baseline_calibration": baseline_calibration,
        "candidate_calibration": calibration,
        "selected_arm": chosen,
        "fit_sample_counts": {
            arm_id: {
                "rows": len(rows),
                "unique_source_images": len({row["frame_sha256"] for row in rows}),
                "l4_positive_rows": sum(int(row["target"]) == CLASS_L4 for row in rows),
                "hard_negative_rows": sum(row["source_role"] == "hard_negative" for row in rows),
            }
            for arm_id, rows in training_rows_by_arm.items()
        },
        "development_used_for_selection": False,
        "new_capture_run": False,
        "l5_modified": False,
    }
    write_csv(output / "calibration_image_decisions.csv", calibration_records)

    if chosen is not None:
        arm = next(item for item in protocol["arms"] if item["arm_id"] == chosen)
        chosen_estimator = estimators[chosen]
        baseline_development, baseline_records = evaluate_l4(
            enriched,
            features,
            states,
            frozen,
            threshold,
            target="three_class",
            split="development_evaluation",
            arm_id="frozen_v1",
        )
        expected_development = protocol["expected_frozen_baseline"][
            "opened_development"
        ]
        observed_development = {
            "tp": baseline_development["tp"],
            "fp": baseline_development["fp"],
            "fn": baseline_development["fn"],
            "tn": baseline_development["tn"],
            "ambiguous_activations": baseline_development[
                "ambiguous_activation_images"
            ],
        }
        if observed_development != expected_development:
            raise RuntimeError(
                f"frozen opened-development baseline changed: {observed_development}"
            )
        candidate_development, candidate_records = evaluate_l4(
            enriched,
            features,
            states,
            chosen_estimator,
            threshold,
            target=arm["target"],
            split="development_evaluation",
            arm_id=chosen,
        )
        gate = protocol["opened_development_success"]
        improvement = bool(
            candidate_development["tp"] >= int(gate["minimum_true_positive_images"])
            and candidate_development["fp"]
            <= int(gate["maximum_clean_negative_false_positive_images"])
            and candidate_development["ambiguous_activation_images"]
            <= int(gate["maximum_ambiguous_activation_images"])
        )
        l5_control = l5_control_metrics(
            enriched,
            selected,
            features,
            frozen,
            float(frozen_bundle["thresholds"][CLASS_L5]),
            "development_evaluation",
        )
        if l5_control != {"tp": 16, "fp": 4, "fn": 3, "tn": 80}:
            raise RuntimeError(f"frozen L5 control changed: {l5_control}")
        checkpoint = output / "selected_l4_candidate.joblib"
        bundle = {
            "schema": "sonogym-tp-surface-core-proxy-l4-ablation-candidate-v1",
            "experiment_id": protocol["experiment_id"],
            "arm": arm,
            "estimator": chosen_estimator,
            "threshold": threshold,
            "patch_spec": PATCH_SPEC,
            "fit_sample_ids": [row["sample_id"] for row in training_rows_by_arm[chosen]],
            "frozen_l5_checkpoint_sha256": protocol["source_checkpoint_sha256"],
        }
        joblib.dump(bundle, checkpoint)
        reloaded = joblib.load(checkpoint)
        probe_rows = [row for row in enriched if row["split"] == "calibration"][:32]
        probe = np.stack([features[row["sample_id"]] for row in probe_rows])
        if not np.array_equal(
            chosen_estimator.predict_proba(probe), reloaded["estimator"].predict_proba(probe)
        ):
            raise RuntimeError("checkpoint reload changed probabilities")
        all_records = baseline_records + candidate_records
        write_csv(output / "opened_development_image_decisions.csv", all_records)
        write_csv(
            output / "opened_development_session_metrics.csv",
            session_metrics(baseline_records) + session_metrics(candidate_records),
        )
        result.update(
            status="IMPROVEMENT_DEMONSTRATED_OPENED_DEVELOPMENT"
            if improvement
            else "OPENED_DEVELOPMENT_NO_GO",
            baseline_opened_development=baseline_development,
            candidate_opened_development=candidate_development,
            opened_development_success=improvement,
            l5_opened_development_control=l5_control,
            selected_checkpoint=str(checkpoint),
            selected_checkpoint_sha256=sha256(checkpoint),
            checkpoint_reload_verified=True,
        )

    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "selected_arm": chosen}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
