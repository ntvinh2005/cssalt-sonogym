#!/usr/bin/env python3
"""Build L4 proxy tri-state records and mine training-only hard negatives."""

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

PROXY_POSITIVE = "PROXY_POSITIVE"
PROXY_NEGATIVE = "PROXY_NEGATIVE"
PROXY_AMBIGUOUS = "PROXY_AMBIGUOUS"
CLASS_L4 = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--failure-audit-csv", type=Path, required=True)
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


def as_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text not in {"true", "false"}:
        raise ValueError(f"invalid boolean value: {value}")
    return text == "true"


def proposal_state(row: dict[str, Any]) -> tuple[str, str]:
    if row["projection_status"] != "VERIFIED":
        return PROXY_AMBIGUOUS, "projection_not_verified"
    class_text = str(row.get("proxy_class_id", "")).strip()
    if (
        class_text == str(CLASS_L4)
        and as_bool(row["proxy_condition_eligible"])
    ):
        return PROXY_POSITIVE, "complete_l4_proxy_contract"
    if int(row["l4_uncertain_points_in_patch"]) > 0:
        return PROXY_AMBIGUOUS, "l4_uncertainty_intersection"
    if int(row["l4_core_points_in_patch"]) > 0:
        return PROXY_AMBIGUOUS, "l4_core_without_complete_positive_contract"
    return PROXY_NEGATIVE, "zero_l4_core_and_zero_l4_uncertainty"


def image_state(states: list[str]) -> str:
    if PROXY_POSITIVE in states:
        return PROXY_POSITIVE
    if PROXY_AMBIGUOUS in states:
        return PROXY_AMBIGUOUS
    if states and set(states) == {PROXY_NEGATIVE}:
        return PROXY_NEGATIVE
    raise ValueError("cannot resolve image state")


def class_column(classes: np.ndarray, class_id: int) -> int:
    columns = np.flatnonzero(classes == class_id)
    if len(columns) != 1:
        raise ValueError(f"checkpoint does not contain class {class_id}")
    return int(columns[0])


def l4_is_frozen_positive(
    probabilities: np.ndarray, classes: np.ndarray, *, threshold: float
) -> bool:
    predicted = int(classes[int(np.argmax(probabilities))])
    return bool(
        predicted == CLASS_L4
        and float(probabilities[class_column(classes, CLASS_L4)]) >= threshold
    )


def hard_negative_geometry_is_eligible(
    row: dict[str, Any], *, source_split: str, source_proposal: str
) -> bool:
    return bool(
        row["split"] == source_split
        and row["l4_proxy_state"] == PROXY_NEGATIVE
        and row["proposal_source"] == source_proposal
        and int(row["l4_uncertain_points_in_patch"]) == 0
        and int(row["l5_uncertain_points_in_patch"]) == 0
    )


def select_one_per_frame(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["frame_sha256"]].append(row)
    return [
        sorted(
            candidates,
            key=lambda row: (-float(row["probability_l4"]), row["sample_id"]),
        )[0]
        for _, candidates in sorted(grouped.items())
    ]


def main() -> int:
    args = parse_args()
    training_dir = args.training_dir.resolve()
    failure_audit = args.failure_audit_csv.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol["model_fitting"] != "forbidden":
        raise ValueError("model fitting must remain forbidden")
    if protocol["new_capture"] != "forbidden":
        raise ValueError("new capture must remain forbidden")

    checkpoint = training_dir / "tp_surface_core_simulator_proxy_extratrees.joblib"
    audit_path = training_dir / "proxy_label_audit.csv"
    selected_path = training_dir / "proxy_sample_manifest.csv"
    expected_hashes = {
        checkpoint: protocol["source_checkpoint_sha256"],
        audit_path: protocol["source_proxy_label_audit_sha256"],
        selected_path: protocol["source_proxy_sample_manifest_sha256"],
        failure_audit: protocol["source_l4_failure_audit_sha256"],
    }
    for path, expected in expected_hashes.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"frozen input SHA-256 mismatch for {path}: {actual}")

    bundle = joblib.load(checkpoint)
    estimator = bundle["estimator"]
    classes = np.asarray(estimator.classes_)
    thresholds = {int(key): float(value) for key, value in bundle["thresholds"].items()}
    minimum_l4_probability = float(
        protocol["hard_negative_mining"]["minimum_l4_probability"]
    )
    if thresholds.get(CLASS_L4) != minimum_l4_probability:
        raise ValueError("hard-negative probability gate differs from frozen L4 threshold")
    if {int(key): float(value) for key, value in bundle["thresholds"].items()} != {
        1: 0.5,
        2: 0.5,
    }:
        raise ValueError("frozen thresholds changed")

    rows = read_csv(audit_path)
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tri_rows: list[dict[str, Any]] = []
    for row in rows:
        state, reason = proposal_state(row)
        tri = dict(row)
        tri["l4_proxy_state"] = state
        tri["l4_proxy_state_reason"] = reason
        tri_rows.append(tri)
        by_frame[row["frame_sha256"]].append(tri)

    frame_states: dict[str, str] = {}
    image_rows: list[dict[str, Any]] = []
    for frame_hash, proposals in sorted(by_frame.items()):
        state = image_state([row["l4_proxy_state"] for row in proposals])
        frame_states[frame_hash] = state
        reason_counts = Counter(row["l4_proxy_state_reason"] for row in proposals)
        image_rows.append(
            {
                "frame_sha256": frame_hash,
                "session_id": proposals[0]["session_id"],
                "split": proposals[0]["split"],
                "l4_proxy_state": state,
                "proposal_count": len(proposals),
                "positive_proposal_count": sum(
                    row["l4_proxy_state"] == PROXY_POSITIVE for row in proposals
                ),
                "ambiguous_proposal_count": sum(
                    row["l4_proxy_state"] == PROXY_AMBIGUOUS for row in proposals
                ),
                "negative_proposal_count": sum(
                    row["l4_proxy_state"] == PROXY_NEGATIVE for row in proposals
                ),
                "state_reason_counts": json.dumps(reason_counts, sort_keys=True),
            }
        )

    archives: dict[Path, np.ndarray] = {}
    hard_candidates: list[dict[str, Any]] = []
    for row in tri_rows:
        if not hard_negative_geometry_is_eligible(
            row,
            source_split=protocol["hard_negative_mining"]["source_split"],
            source_proposal=protocol["hard_negative_mining"]["source_proposal"],
        ):
            continue
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
        if not l4_is_frozen_positive(
            probabilities, classes, threshold=minimum_l4_probability
        ):
            continue
        subtype = (
            "COMPETING_L5_CORE"
            if int(row["l5_core_points_in_patch"]) > 0
            else "OTHER_BONE_INTERFACE"
        )
        hard_candidates.append(
            {
                "sample_id": row["sample_id"],
                "session_id": row["session_id"],
                "split": row["split"],
                "frame_sequence": row["frame_sequence"],
                "frame_sha256": row["frame_sha256"],
                "image_archive": row["image_archive"],
                "image_archive_index": row["image_archive_index"],
                "track_id": row["track_id"],
                "patch_row": row["patch_row"],
                "patch_column": row["patch_column"],
                "feature_sha256": row["feature_sha256"],
                "geometry_version_sha256": row["geometry_version_sha256"],
                "l4_proxy_state": row["l4_proxy_state"],
                "l4_source_image_state": frame_states[row["frame_sha256"]],
                "hard_negative_subtype": subtype,
                "probability_background": float(
                    probabilities[class_column(classes, 0)]
                ),
                "probability_l4": float(probabilities[class_column(classes, 1)]),
                "probability_l5": float(probabilities[class_column(classes, 2)]),
                "l4_core_points_in_patch": row["l4_core_points_in_patch"],
                "l5_core_points_in_patch": row["l5_core_points_in_patch"],
                "l4_uncertain_points_in_patch": row["l4_uncertain_points_in_patch"],
                "l5_uncertain_points_in_patch": row["l5_uncertain_points_in_patch"],
            }
        )

    hard_selected = select_one_per_frame(hard_candidates)
    minimum = int(protocol["data_sufficiency"]["minimum_unique_training_images"])
    unique_frames = len({row["frame_sha256"] for row in hard_selected})
    sufficient = unique_frames >= minimum

    failure_by_hash = {row["frame_sha256"]: row for row in read_csv(failure_audit)}
    reclassified: list[dict[str, Any]] = []
    for frame_hash, failure in sorted(failure_by_hash.items()):
        reclassified.append(
            {
                "failure_id": failure["failure_id"],
                "frame_sha256": frame_hash,
                "session_id": failure["session_id"],
                "historical_primary_classification": failure[
                    "primary_evidence_classification"
                ],
                "new_l4_proxy_state": frame_states[frame_hash],
                "historical_result_changed": False,
                "training_eligible": False,
            }
        )

    proposal_counts = Counter(row["l4_proxy_state"] for row in tri_rows)
    image_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "calibration", "development_evaluation"):
        counts = Counter(
            row["l4_proxy_state"] for row in image_rows if row["split"] == split
        )
        image_counts[split] = {
            state: counts[state]
            for state in (PROXY_POSITIVE, PROXY_NEGATIVE, PROXY_AMBIGUOUS)
        }
    subtype_counts = Counter(row["hard_negative_subtype"] for row in hard_selected)
    session_counts = Counter(row["session_id"] for row in hard_selected)

    write_csv(output / "l4_proxy_tristate_proposals.csv", tri_rows)
    write_csv(output / "l4_proxy_tristate_images.csv", image_rows)
    write_csv(output / "l4_hard_negative_candidates.csv", hard_candidates)
    write_csv(output / "l4_hard_negative_selected.csv", hard_selected)
    write_csv(output / "opened_development_fp_reclassification.csv", reclassified)
    selected_manifest_hash = sha256(output / "l4_hard_negative_selected.csv")
    result = {
        "schema": "sonogym-tp-surface-core-proxy-tristate-hard-negative-v1",
        "experiment_id": protocol["experiment_id"],
        "status": "DATA_SUFFICIENT" if sufficient else "DATA_INSUFFICIENT",
        "fit_called": False,
        "new_capture_run": False,
        "development_images_opened": False,
        "development_rows_used_for_training": 0,
        "historical_results_modified": False,
        "source_row_count": len(rows),
        "proposal_state_counts": dict(proposal_counts),
        "image_state_counts_by_split": image_counts,
        "training_hard_negative_candidate_count": len(hard_candidates),
        "training_hard_negative_selected_count": len(hard_selected),
        "training_hard_negative_unique_image_count": unique_frames,
        "training_hard_negative_subtype_counts": dict(subtype_counts),
        "training_hard_negative_session_counts": dict(session_counts),
        "minimum_unique_training_images": minimum,
        "data_sufficiency_passed": sufficient,
        "hard_negative_selected_manifest_sha256": selected_manifest_hash,
        "opened_development_false_positive_count": len(reclassified),
        "opened_development_reclassification_counts": dict(
            Counter(row["new_l4_proxy_state"] for row in reclassified)
        ),
        "l5_checkpoint_role": protocol["future_l4_v2"]["l5_checkpoint"],
        "future_l4_v2_protocol_preparation_allowed": sufficient,
        "future_l4_v2_fit_run": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if sufficient:
        inputs = {
            "schema": "sonogym-tp-surface-core-proxy-l4-v2-inputs-v1",
            "source_checkpoint_sha256": sha256(checkpoint),
            "hard_negative_manifest": str(
                output / "l4_hard_negative_selected.csv"
            ),
            "hard_negative_manifest_sha256": selected_manifest_hash,
            "hard_negative_unique_training_images": unique_frames,
            "l5_checkpoint_role": "frozen_control",
            "fit_status": "NOT_RUN",
        }
        (output / "l4_v2_protocol_inputs.json").write_text(
            json.dumps(inputs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_hard_negatives": len(hard_selected),
                "fit_called": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
