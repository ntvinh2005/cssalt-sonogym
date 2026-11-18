#!/usr/bin/env python3
"""Run the isolated SIMULATOR_GEOMETRY_PROXY TP surface-core pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


FEATURE_PATH = (
    Path(__file__).resolve().parents[2]
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

EXPERIMENT_ID = "s0010_tp_surface_core_simulator_proxy_exploratory_v1"
LABEL_MODE = "SIMULATOR_GEOMETRY_PROXY"
SPLITS = ("train", "calibration", "development_evaluation")
CLASS_NAMES = {
    0: "BACKGROUND_OR_OTHER_STRUCTURE",
    1: "L4_RIGHT_TP_SURFACE_CORE",
    2: "L5_RIGHT_TP_SURFACE_CORE",
}
GEOMETRY_ARCHIVES = {
    "L4": "L4_RIGHT_transverse_process_revision.npz",
    "L5": "L5_RIGHT_transverse_process_revision.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--label-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-fit", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def as_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text not in {"true", "false"}:
        raise ValueError(f"invalid boolean value: {value}")
    return text == "true"


def load_contract(path: Path) -> dict[int, dict[str, Any]]:
    rows = read_csv(path)
    if len(rows) != 3:
        raise ValueError("proxy label contract must contain exactly three classes")
    contract: dict[int, dict[str, Any]] = {}
    for raw in rows:
        if raw["experiment_id"] != EXPERIMENT_ID or raw["label_mode"] != LABEL_MODE:
            raise ValueError("unexpected experiment identity or label mode")
        class_id = int(raw["class_id"])
        if class_id in contract or CLASS_NAMES.get(class_id) != raw["class_name"]:
            raise ValueError("invalid or duplicate proxy class")
        row: dict[str, Any] = dict(raw)
        for field in (
            "other_core_must_be_zero",
            "all_uncertain_must_be_zero",
            "adequate_projected_support_required",
        ):
            row[field] = as_bool(row[field])
        row["minimums"] = {
            "train": int(row["min_train_unique_images"]),
            "calibration": int(row["min_calibration_unique_images"]),
            "development_evaluation": int(row["min_development_unique_images"]),
        }
        contract[class_id] = row
    source_hashes = {row["source_zip_sha256"] for row in contract.values()}
    if set(contract) != set(CLASS_NAMES) or len(source_hashes) != 1:
        raise ValueError("proxy label contract is incomplete or references multiple packages")
    return contract


def zip_member(archive: zipfile.ZipFile, basename: str) -> str:
    matches = [name for name in archive.namelist() if Path(name).name == basename]
    if len(matches) != 1:
        raise ValueError(f"expected one {basename} in source package, found {len(matches)}")
    return matches[0]


def package_geometry_hashes(source_zip: Path) -> tuple[dict[str, str], bytes]:
    with zipfile.ZipFile(source_zip) as archive:
        manifest_bytes = archive.read(zip_member(archive, "candidate_sample_manifest.csv"))
        geometry = {
            level: bytes_sha256(archive.read(zip_member(archive, basename)))
            for level, basename in GEOMETRY_ARCHIVES.items()
        }
    return geometry, manifest_bytes


def validate_sample_id(row: dict[str, str]) -> bool:
    prefix = f"{row['session_id']}:{int(row['frame_sequence'])}:{row['track_id']}:"
    suffix = row["sample_id"][len(prefix) :] if row["sample_id"].startswith(prefix) else ""
    return suffix.isdigit()


def geometry_map(row: dict[str, str]) -> dict[str, str]:
    value = json.loads(row["geometry_version_sha256"])
    expected = set(CLASS_NAMES.values()) - {CLASS_NAMES[0]}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("geometry checksum map does not contain exactly L4 and L5")
    return {"L4": value[CLASS_NAMES[1]], "L5": value[CLASS_NAMES[2]]}


def proxy_condition(row: dict[str, str], rule: dict[str, Any]) -> tuple[bool, str]:
    if row["split"] == rule["excluded_split"]:
        return False, "excluded_split"
    if row["split"] not in SPLITS:
        return False, "unsupported_split"
    if row["candidate_id"] != rule["source_candidate_id"]:
        return False, "candidate_mismatch"
    if row["review_target"] != rule["required_review_target"]:
        return False, "review_target_mismatch"
    if row["projection_status"] != rule["projection_status_required"]:
        return False, "projection_unverified"

    l4_core = int(row["l4_core_points_in_patch"])
    l5_core = int(row["l5_core_points_in_patch"])
    required = rule["required_core"]
    if required == "NONE" and (l4_core != 0 or l5_core != 0):
        return False, "projected_core_present"
    if required == "L4" and l4_core <= 0:
        return False, "required_l4_core_missing"
    if required == "L5" and l5_core <= 0:
        return False, "required_l5_core_missing"
    if rule["other_core_must_be_zero"]:
        other = l5_core if required == "L4" else l4_core if required == "L5" else 0
        if other != 0:
            return False, "competing_core_present"
    if rule["all_uncertain_must_be_zero"] and (
        int(row["l4_uncertain_points_in_patch"]) != 0
        or int(row["l5_uncertain_points_in_patch"]) != 0
    ):
        return False, "uncertain_region_present"
    if rule["adequate_projected_support_required"] and not as_bool(
        row["adequate_projected_support"]
    ):
        return False, "projected_support_inadequate"
    return True, "eligible"


def selection_rank(row: dict[str, str], class_id: int) -> tuple[Any, ...]:
    if class_id == 0:
        return (
            -float(row["patch_gradient_p90"]),
            -float(row["patch_normalized_std"]),
            row["sample_id"],
        )
    return (
        -int(row["interior_core_feature_cells"]),
        -int(row["near_interface_core_feature_cells"]),
        -int(row["core_feature_cells"]),
        float(row["nearest_core_distance_feature_cells"]),
        float(row["border_core_feature_cell_fraction"]),
        -int(row["unique_core_pixels"]),
        row["sample_id"],
    )


def choose_one_per_source_image(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["proxy_class_id"]), row["split"], row["frame_sha256"])].append(row)
    return [
        min(candidates, key=lambda row: selection_rank(row, class_id))
        for (class_id, _split, _frame), candidates in sorted(grouped.items())
    ]


def class_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_count": int(len(targets)),
        "accuracy": float(np.mean(targets == predictions)),
        "per_class": {},
    }
    for class_id, name in CLASS_NAMES.items():
        tp = int(np.count_nonzero((targets == class_id) & (predictions == class_id)))
        fp = int(np.count_nonzero((targets != class_id) & (predictions == class_id)))
        fn = int(np.count_nonzero((targets == class_id) & (predictions != class_id)))
        tn = int(np.count_nonzero((targets != class_id) & (predictions != class_id)))
        precision = None if tp + fp == 0 else tp / (tp + fp)
        recall = None if tp + fn == 0 else tp / (tp + fn)
        result["per_class"][name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": None
            if precision is None or recall is None or precision + recall == 0
            else 2 * precision * recall / (precision + recall),
        }
    return result


def calibrate_thresholds(
    targets: np.ndarray, probabilities: np.ndarray, classes: np.ndarray
) -> tuple[dict[int, float], dict[str, Any]]:
    thresholds: dict[int, float] = {}
    diagnostics: dict[str, Any] = {}
    for class_id in (1, 2):
        column = int(np.flatnonzero(classes == class_id)[0])
        scores = probabilities[:, column]
        grid = []
        for threshold in np.linspace(0.50, 0.95, 10):
            accepted = scores >= threshold
            tp = int(np.count_nonzero(accepted & (targets == class_id)))
            fp = int(np.count_nonzero(accepted & (targets != class_id)))
            fn = int(np.count_nonzero((~accepted) & (targets == class_id)))
            grid.append(
                {
                    "threshold": float(threshold),
                    "precision": None if tp + fp == 0 else tp / (tp + fp),
                    "recall": None if tp + fn == 0 else tp / (tp + fn),
                }
            )
        passing = [row for row in grid if row["precision"] is not None and row["precision"] >= 0.90]
        selected = max(
            passing or grid,
            key=lambda row: (-1.0 if row["recall"] is None else row["recall"], -row["threshold"]),
        )
        thresholds[class_id] = selected["threshold"]
        diagnostics[CLASS_NAMES[class_id]] = {"selected": selected, "grid": grid}
    return thresholds, diagnostics


def apply_thresholds(
    probabilities: np.ndarray, classes: np.ndarray, thresholds: dict[int, float]
) -> np.ndarray:
    predictions = np.zeros(len(probabilities), dtype=np.int64)
    for index, row in enumerate(probabilities):
        column = int(np.argmax(row))
        class_id = int(classes[column])
        if class_id in thresholds and row[column] >= thresholds[class_id]:
            predictions[index] = class_id
    return predictions


def full_image_results(
    development_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
    feature_by_id: dict[str, np.ndarray],
    estimator: ExtraTreesClassifier,
    thresholds: dict[int, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frame_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in development_rows:
        frame_rows[row["frame_sha256"]].append(row)
    truths: dict[str, set[int]] = defaultdict(set)
    for row in proxy_rows:
        if row["split"] == "development_evaluation":
            truths[row["frame_sha256"]].add(int(row["proxy_class_id"]))

    records: list[dict[str, Any]] = []
    binary: dict[int, list[tuple[bool, bool]]] = {class_id: [] for class_id in CLASS_NAMES}
    for frame_hash, rows in sorted(frame_rows.items()):
        probabilities = estimator.predict_proba(
            np.stack([feature_by_id[row["sample_id"]] for row in rows])
        )
        predictions = apply_thresholds(probabilities, estimator.classes_, thresholds)
        predicted_targets = {int(value) for value in predictions if int(value) != 0}
        true_targets = truths.get(frame_hash, set()) - {0}
        predicted = predicted_targets or {0}
        truth = true_targets or {0}
        for class_id in CLASS_NAMES:
            binary[class_id].append((class_id in truth, class_id in predicted))
        best_l4 = float(np.max(probabilities[:, int(np.flatnonzero(estimator.classes_ == 1)[0])]))
        best_l5 = float(np.max(probabilities[:, int(np.flatnonzero(estimator.classes_ == 2)[0])]))
        records.append(
            {
                "frame_sha256": frame_hash,
                "session_id": rows[0]["session_id"],
                "split": "development_evaluation",
                "proposal_count": len(rows),
                "proxy_truth_class_ids": ";".join(map(str, sorted(truth))),
                "predicted_class_ids": ";".join(map(str, sorted(predicted))),
                "maximum_l4_probability": best_l4,
                "maximum_l5_probability": best_l5,
            }
        )
    metrics: dict[str, Any] = {"image_count": len(records), "per_class": {}}
    for class_id, values in binary.items():
        tp = sum(truth and pred for truth, pred in values)
        fp = sum((not truth) and pred for truth, pred in values)
        fn = sum(truth and (not pred) for truth, pred in values)
        tn = sum((not truth) and (not pred) for truth, pred in values)
        metrics["per_class"][CLASS_NAMES[class_id]] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": None if tp + fp == 0 else tp / (tp + fp),
            "recall": None if tp + fn == 0 else tp / (tp + fn),
        }
    return metrics, records


def main() -> int:
    args = parse_args()
    dataset = args.dataset_dir.resolve()
    source_zip = args.source_zip.resolve()
    contract_path = args.label_contract.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    contract = load_contract(contract_path)
    expected_zip_hash = next(iter(contract.values()))["source_zip_sha256"]
    actual_zip_hash = sha256(source_zip)
    if actual_zip_hash != expected_zip_hash:
        failures.append(f"source ZIP SHA-256 mismatch: {actual_zip_hash}")

    manifest_path = dataset / "candidate_sample_manifest.csv"
    rows = read_csv(manifest_path)
    geometry_hashes, packaged_manifest = package_geometry_hashes(source_zip)
    if bytes_sha256(packaged_manifest) != sha256(manifest_path):
        failures.append("dataset manifest differs from the manifest sealed in the source ZIP")
    if not rows:
        failures.append("source manifest is empty")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        failures.append("source manifest has duplicate sample identifiers")

    archives: dict[Path, np.ndarray] = {}
    images: dict[tuple[Path, int], np.ndarray] = {}
    features: dict[str, np.ndarray] = {}
    audited_rows: list[dict[str, Any]] = []
    proxy_candidates: list[dict[str, Any]] = []
    for row in rows:
        row_failures: list[str] = []
        if not validate_sample_id(row):
            row_failures.append("invalid_sample_id")
        try:
            row_geometry = geometry_map(row)
            if row_geometry != geometry_hashes:
                row_failures.append("geometry_checksum_mismatch")
        except (ValueError, KeyError, json.JSONDecodeError):
            row_failures.append("invalid_geometry_checksum_map")
        path = Path(row["image_archive"])
        index = int(row["image_archive_index"])
        if not path.is_file():
            row_failures.append("missing_source_archive")
        else:
            if path not in archives:
                archives[path] = np.load(path)["ultrasound"]
            if index < 0 or index >= len(archives[path]):
                row_failures.append("invalid_source_array_index")
            else:
                image = np.asarray(archives[path][index])
                images[(path, index)] = image
                if image.shape != (200, 150) or ultrasound_sha256(image) != row["frame_sha256"]:
                    row_failures.append("source_frame_hash_or_shape_mismatch")
                feature = extract_patch_feature(
                    image, float(row["patch_row"]), float(row["patch_column"]), PATCH_SPEC
                )
                if feature is None or not np.all(np.isfinite(feature)):
                    row_failures.append("invalid_actor_visible_feature")
                else:
                    features[row["sample_id"]] = feature
        matched: list[tuple[int, str]] = []
        for class_id, rule in contract.items():
            eligible, reason = proxy_condition(row, rule)
            if eligible:
                matched.append((class_id, reason))
        if len(matched) > 1:
            row_failures.append("multiple_proxy_classes")
        proxy_class_id = matched[0][0] if len(matched) == 1 else None
        if row_failures:
            failures.extend(f"{row['sample_id']}: {reason}" for reason in row_failures)
        eligible = proxy_class_id is not None and not row_failures
        audit = dict(row)
        audit.update(
            proxy_experiment_id=EXPERIMENT_ID,
            proxy_label_mode=LABEL_MODE,
            proxy_class_id="" if proxy_class_id is None else proxy_class_id,
            generated_proxy_label="" if proxy_class_id is None else contract[proxy_class_id]["generated_proxy_label"],
            proxy_condition_eligible=eligible,
            proxy_exclusion_reason=";".join(row_failures) if row_failures else ("" if eligible else "proxy_conditions_not_met"),
        )
        audited_rows.append(audit)
        if eligible:
            proxy_candidates.append(audit)

    selected_rows = choose_one_per_source_image(proxy_candidates)
    selected_ids = {row["sample_id"] for row in selected_rows}
    for row in audited_rows:
        row["proxy_selected_for_experiment"] = row["sample_id"] in selected_ids
    write_csv(output / "proxy_label_audit.csv", audited_rows)
    write_csv(output / "proxy_sample_manifest.csv", selected_rows)

    split_hashes = {
        split: {row["frame_sha256"] for row in selected_rows if row["split"] == split}
        for split in SPLITS
    }
    duplicate_hashes = set().union(
        *(
            split_hashes[left] & split_hashes[right]
            for index, left in enumerate(SPLITS)
            for right in SPLITS[index + 1 :]
        )
    )
    if duplicate_hashes:
        failures.append(f"cross-split exact source-image duplicates: {len(duplicate_hashes)}")
    unique_counts = {
        CLASS_NAMES[class_id]: {
            split: len(
                {
                    row["frame_sha256"]
                    for row in selected_rows
                    if int(row["proxy_class_id"]) == class_id and row["split"] == split
                }
            )
            for split in SPLITS
        }
        for class_id in CLASS_NAMES
    }
    for class_id, rule in contract.items():
        for split, minimum in rule["minimums"].items():
            count = unique_counts[CLASS_NAMES[class_id]][split]
            if count < minimum:
                failures.append(f"{CLASS_NAMES[class_id]} {split}: {count} unique images < {minimum}")

    partitions = {
        split: [row for row in selected_rows if row["split"] == split] for split in SPLITS
    }
    estimator = ExtraTreesClassifier(
        n_estimators=240,
        class_weight="balanced",
        random_state=20261117,
        n_jobs=4,
    )
    try:
        git_commit = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "UNKNOWN"
    report: dict[str, Any] = {
        "schema": "sonogym-tp-surface-core-simulator-proxy-result-v1",
        "experiment_id": EXPERIMENT_ID,
        "label_mode": LABEL_MODE,
        "scientific_scope": "simulator geometry proxy performance; not anatomical recognition",
        "mode": "NO_FIT" if args.no_fit else "FIT",
        "source_commit": git_commit,
        "source_zip": str(source_zip),
        "source_zip_sha256": actual_zip_hash,
        "label_contract_sha256": sha256(contract_path),
        "source_manifest_sha256": sha256(manifest_path),
        "source_rows_audited": len(rows),
        "proxy_condition_candidate_rows": len(proxy_candidates),
        "selected_proxy_rows": len(selected_rows),
        "selected_unique_image_counts": unique_counts,
        "cross_split_duplicate_count": len(duplicate_hashes),
        "feature_width": PATCH_SPEC.feature_width,
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
        report["status"] = "PROXY_TRAINING_BLOCKED" if failures else "PROXY_TRAINING_READY"
    elif failures:
        report["status"] = "PROXY_TRAINING_BLOCKED"
    else:
        def matrix(split: str) -> tuple[np.ndarray, np.ndarray]:
            chosen = partitions[split]
            return (
                np.stack([features[row["sample_id"]] for row in chosen]),
                np.asarray([int(row["proxy_class_id"]) for row in chosen], dtype=np.int64),
            )

        train_x, train_y = matrix("train")
        calibration_x, calibration_y = matrix("calibration")
        development_x, development_y = matrix("development_evaluation")
        estimator.fit(train_x, train_y)
        report["fit_called"] = True
        calibration_probabilities = estimator.predict_proba(calibration_x)
        thresholds, calibration = calibrate_thresholds(
            calibration_y, calibration_probabilities, estimator.classes_
        )
        development_probabilities = estimator.predict_proba(development_x)
        development_predictions = apply_thresholds(
            development_probabilities, estimator.classes_, thresholds
        )
        failures_rows = []
        for row, target, predicted, probabilities in zip(
            partitions["development_evaluation"],
            development_y,
            development_predictions,
            development_probabilities,
        ):
            if int(target) != int(predicted):
                failures_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "frame_sha256": row["frame_sha256"],
                        "target_class_id": int(target),
                        "predicted_class_id": int(predicted),
                        **{
                            f"probability_{CLASS_NAMES[int(class_id)]}": float(probabilities[column])
                            for column, class_id in enumerate(estimator.classes_)
                        },
                    }
                )
        write_csv(output / "development_patch_failures.csv", failures_rows)
        development_all_rows = [
            row for row in audited_rows if row["split"] == "development_evaluation" and row["sample_id"] in features
        ]
        image_metrics, image_records = full_image_results(
            development_all_rows, selected_rows, features, estimator, thresholds
        )
        write_csv(output / "development_full_image_proposals.csv", image_records)
        bundle = {
            "schema": "sonogym-tp-surface-core-simulator-proxy-model-v1",
            "experiment_id": EXPERIMENT_ID,
            "label_mode": LABEL_MODE,
            "scientific_scope": "simulator geometry proxy performance; not anatomical recognition",
            "estimator": estimator,
            "class_names": CLASS_NAMES,
            "thresholds": thresholds,
            "patch_spec": PATCH_SPEC,
            "source_zip_sha256": actual_zip_hash,
            "source_manifest_sha256": sha256(manifest_path),
            "label_contract_sha256": sha256(contract_path),
            "fit_sample_ids": report["fit_sample_ids"],
        }
        checkpoint = output / "tp_surface_core_simulator_proxy_extratrees.joblib"
        joblib.dump(bundle, checkpoint)
        loaded = joblib.load(checkpoint)
        if not np.array_equal(
            development_probabilities, loaded["estimator"].predict_proba(development_x)
        ):
            raise RuntimeError("checkpoint reload changed development probabilities")
        report.update(
            status="EXPLORATORY_PROXY_TRAINING_COMPLETED",
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha256(checkpoint),
            checkpoint_saved=True,
            checkpoint_reload_verified=True,
            thresholds={str(key): value for key, value in thresholds.items()},
            calibration=calibration,
            patch_level_development=class_metrics(
                development_y, development_predictions
            ),
            full_image_proposal_development=image_metrics,
            development_patch_failure_count=len(failures_rows),
        )
    (output / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "source_rows_audited": report["source_rows_audited"],
                "selected_proxy_rows": report["selected_proxy_rows"],
                "fit_called": report["fit_called"],
            },
            sort_keys=True,
        )
    )
    return 3 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
