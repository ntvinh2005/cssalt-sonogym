#!/usr/bin/env python3
"""Run the bounded CPU Stage 1 perception, control, and L4 mapping pilot."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesClassifier


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
WORKFLOW_ROOT = Path(__file__).resolve().parent
for path in (EVALUATION_ROOT, WORKFLOW_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_stage1 import (  # noqa: E402
    dataclass_kwargs,
    file_sha256,
    positive_probability,
    read_json,
    read_jsonl,
    validate_capture,
    validate_protocol_seal,
    write_json,
)
from train_stage1_group_a import (  # noqa: E402
    GroupASessionData,
    anatomy_patch_metrics,
    load_session,
)
from two_stage_navigation.contracts import ultrasound_sha256  # noqa: E402
from two_stage_navigation.stage1_features import PatchSpec  # noqa: E402
from two_stage_navigation.stage1_interfaces import InterfaceTrackerConfig  # noqa: E402
from two_stage_navigation.stage1_map import SpatialMapConfig  # noqa: E402
from two_stage_navigation.stage1_pipeline import reconstruct_landmark  # noqa: E402


@dataclass
class PilotSession:
    source: GroupASessionData
    position_history_features: np.ndarray
    label_audit: dict[str, Any]


def wrap_to_pi(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def binary_metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(prediction, dtype=np.int64)
    if truth.shape != predicted.shape or not np.all(np.isin(truth, (0, 1))):
        raise ValueError("binary labels and predictions must align")
    tp = int(np.count_nonzero((truth == 1) & (predicted == 1)))
    fp = int(np.count_nonzero((truth == 0) & (predicted == 1)))
    fn = int(np.count_nonzero((truth == 1) & (predicted == 0)))
    tn = int(np.count_nonzero((truth == 0) & (predicted == 0)))
    return {
        "accuracy": float(np.mean(truth == predicted)),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": int(len(truth)),
    }


def fit_model(
    features: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    seed_offset: int,
) -> ExtraTreesClassifier:
    observed = np.unique(labels)
    if len(observed) < 2:
        raise RuntimeError("supervised model requires at least two classes")
    model = ExtraTreesClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        class_weight=str(config["class_weight"]),
        n_jobs=int(config["n_jobs"]),
        random_state=int(config["random_seed"]) + seed_offset,
    )
    model.fit(features, labels)
    return model


def validate_annotations(protocol: dict[str, Any]) -> None:
    specification = protocol["annotations"]
    path = REPOSITORY_ROOT / str(specification["path"])
    if file_sha256(path) != str(specification["sha256"]):
        raise RuntimeError("annotation manifest seal mismatch")
    manifest = read_json(path)
    records = manifest["annotations"][str(specification["group"])]
    class_order = [str(value) for value in protocol["offline_labels"]["class_order"]]
    if [str(record["anatomy_level"]) for record in records] != class_order:
        raise RuntimeError("Group A anatomy order does not match the protocol")
    if bool(specification["group_b_permitted"]):
        raise RuntimeError("Group B must remain excluded from this pilot")
    l4 = next(record for record in records if record["anatomy_level"] == "L4")
    observed = float(l4["geometry"]["reference_centroid_home_relative_mm"])
    expected = float(protocol["evaluation"]["l4_reference_coordinate_mm"])
    if not math.isclose(observed, expected, abs_tol=1e-9):
        raise RuntimeError("L4 evaluator reference does not match annotation v2")


def audit_and_load_session(
    session_specification: dict[str, Any],
    *,
    capture_root: Path,
    patch_spec: PatchSpec,
    tracker_config: InterfaceTrackerConfig,
    minimum_anatomy_fraction: float,
    column_stride_pixels: int,
    pixel_resolution_mm: float,
    forbidden_fields: set[str],
    load_candidates: bool,
) -> tuple[PilotSession | None, dict[str, Any]]:
    session_id = str(session_specification["session_id"])
    root = capture_root / "sessions" / session_id
    public_rows = read_jsonl(root / "public" / "frames.jsonl")
    privileged_rows = read_jsonl(root / "privileged" / "annotations.jsonl")
    images = np.load(root / "public" / "ultrasound.npz")["ultrasound"]
    raw_labels = np.load(root / "privileged" / "raw_labels.npz")["raw_labels"]
    if not (
        len(public_rows) == len(privileged_rows) == len(images) == len(raw_labels) == 211
    ):
        raise RuntimeError(f"incomplete arrays or manifests in {session_id}")

    cumulative_position = np.zeros(2, dtype=np.float64)
    cumulative_angle = 0.0
    maximum_position_error = 0.0
    maximum_angle_error = 0.0
    phase_counts: dict[str, int] = {}
    hashes: set[str] = set()
    for index, (public, privileged, image) in enumerate(
        zip(public_rows, privileged_rows, images)
    ):
        identity = (session_id, index, ultrasound_sha256(image))
        public_identity = (
            str(public["session_id"]),
            int(public["frame_sequence"]),
            str(public["frame_sha256"]),
        )
        privileged_identity = (
            str(privileged["session_id"]),
            int(privileged["frame_sequence"]),
            str(privileged["frame_sha256"]),
        )
        if identity != public_identity or identity != privileged_identity:
            raise RuntimeError(f"frame identity mismatch in {session_id}:{index}")
        leaked = forbidden_fields & set(public)
        if leaked:
            raise RuntimeError(f"privileged fields in public row: {sorted(leaked)}")
        if image.shape != (200, 150) or not np.all(np.isfinite(image)):
            raise RuntimeError(f"invalid ultrasound frame in {session_id}:{index}")
        cumulative_position += np.asarray(
            public["executed_delta_home_mm"], dtype=np.float64
        )
        cumulative_angle = wrap_to_pi(
            cumulative_angle + float(public["executed_delta_angle_rad"])
        )
        position_error = float(
            np.max(
                np.abs(
                    cumulative_position
                    - np.asarray(public["relative_position_home_mm"], dtype=np.float64)
                )
            )
        )
        angle_error = abs(
            wrap_to_pi(cumulative_angle - float(public["relative_angle_home_rad"]))
        )
        maximum_position_error = max(maximum_position_error, position_error)
        maximum_angle_error = max(maximum_angle_error, angle_error)
        phase = str(public["phase"])
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        hashes.add(identity[2])
        if not all(
            bool(public[field])
            for field in (
                "action_accepted",
                "action_completed",
                "contact_ok",
                "workspace_ok",
                "image_valid",
            )
        ) or bool(public["action_clamped"]):
            raise RuntimeError(f"invalid transition in {session_id}:{index}")
    expected_phases = (
        ["HOME_BASELINE"]
        + ["OUTBOUND_CAUDAL"] * 105
        + ["RETURN_CRANIAL"] * 105
    )
    if [str(row["phase"]) for row in public_rows] != expected_phases:
        raise RuntimeError(f"unexpected forward/reverse history in {session_id}")
    if maximum_position_error > 1e-9 or maximum_angle_error > 1e-9:
        raise RuntimeError(f"executed-motion reconstruction mismatch in {session_id}")

    audit = {
        "session_id": session_id,
        "split": str(session_specification["split"]),
        "frame_count": len(public_rows),
        "unique_frame_hash_count": len(hashes),
        "phase_counts": phase_counts,
        "maximum_odometry_position_reconstruction_error_mm": maximum_position_error,
        "maximum_odometry_angle_reconstruction_error_rad": maximum_angle_error,
        "actor_privileged_field_intersection": [],
        "final_home_relative_position_mm": cumulative_position.tolist(),
        "final_home_relative_angle_rad": cumulative_angle,
        "labels_loaded_for_modeling": load_candidates,
    }
    if not load_candidates:
        return None, audit

    source = load_session(
        session_specification,
        capture_root=capture_root,
        patch_spec=patch_spec,
        tracker_config=tracker_config,
        minimum_anatomy_fraction=minimum_anatomy_fraction,
        column_stride_pixels=column_stride_pixels,
        pixel_resolution_mm=pixel_resolution_mm,
    )
    position_rows = []
    for candidate in source.candidates:
        public = public_rows[candidate.frame_sequence]
        relative = np.asarray(public["relative_position_home_mm"], dtype=np.float64)
        position_rows.append(
            [
                float(relative[0]),
                float(relative[1]),
                float(public["relative_angle_home_rad"]),
                1.0 if candidate.direction == "FORWARD" else -1.0,
                float(candidate.frame_sequence) / 210.0,
            ]
        )
    position_features = np.asarray(position_rows, dtype=np.float32)
    if position_features.shape != (len(source.candidates), 5):
        raise RuntimeError(f"invalid position/history features in {session_id}")
    unique_anatomy = source.anatomy_classes >= 0
    label_audit = {
        "candidate_count": len(source.candidates),
        "shared_positive_count": int(np.count_nonzero(source.primary_labels == 1)),
        "shared_negative_count": int(np.count_nonzero(source.primary_labels == 0)),
        "unique_anatomy_label_count": int(np.count_nonzero(unique_anatomy)),
        "ambiguous_positive_count": int(
            np.count_nonzero((source.primary_labels == 1) & ~unique_anatomy)
        ),
        "class_counts": {
            str(index): int(np.count_nonzero(source.anatomy_classes == index))
            for index in range(7)
        },
    }
    audit["label_audit"] = label_audit
    return PilotSession(source, position_features, label_audit), audit


def arrays(
    sessions: list[PilotSession], *, anatomy_only: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_rows: list[np.ndarray] = []
    position_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    for session in sessions:
        labels = (
            session.source.anatomy_classes
            if anatomy_only
            else session.source.primary_labels
        )
        mask = labels >= 0
        image_rows.append(session.source.features[mask])
        position_rows.append(session.position_history_features[mask])
        label_rows.append(labels[mask])
    return (
        np.concatenate(image_rows),
        np.concatenate(position_rows),
        np.concatenate(label_rows),
    )


def shuffled_images(
    sessions: list[PilotSession], *, anatomy_only: bool, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    for session in sessions:
        labels = (
            session.source.anatomy_classes
            if anatomy_only
            else session.source.primary_labels
        )
        permutation = rng.permutation(len(session.source.candidates))
        rows.append(session.source.features[permutation][labels >= 0])
    return np.concatenate(rows)


def classification_arms(
    training: list[PilotSession],
    development: list[PilotSession],
    *,
    anatomy_only: bool,
    protocol: dict[str, Any],
    seed_base: int,
) -> tuple[dict[str, ExtraTreesClassifier], dict[str, Any]]:
    train_image, train_position, train_labels = arrays(
        training, anatomy_only=anatomy_only
    )
    dev_image, dev_position, dev_labels = arrays(
        development, anatomy_only=anatomy_only
    )
    shuffled = shuffled_images(
        development,
        anatomy_only=anatomy_only,
        seed=int(protocol["controls"]["shuffled_image_seed"]),
    )
    train_combined = np.concatenate([train_image, train_position], axis=1)
    dev_combined = np.concatenate([dev_image, dev_position], axis=1)
    shuffled_combined = np.concatenate([shuffled, dev_position], axis=1)
    models = {
        "image_only": fit_model(
            train_image, train_labels, protocol["models"], seed_base
        ),
        "position_history_only": fit_model(
            train_position, train_labels, protocol["models"], seed_base + 1
        ),
        "image_plus_position_history": fit_model(
            train_combined, train_labels, protocol["models"], seed_base + 2
        ),
    }
    predictions = {
        "image_only": models["image_only"].predict(dev_image),
        "position_history_only": models["position_history_only"].predict(dev_position),
        "image_plus_position_history": models["image_plus_position_history"].predict(
            dev_combined
        ),
        "shuffled_image": models["image_only"].predict(shuffled),
        "shuffled_image_plus_position_history": models[
            "image_plus_position_history"
        ].predict(shuffled_combined),
    }
    if anatomy_only:
        class_order = [
            str(value) for value in protocol["offline_labels"]["class_order"]
        ]
        metrics = {
            name: anatomy_patch_metrics(
                models[
                    "image_only"
                    if name == "shuffled_image"
                    else (
                        "image_plus_position_history"
                        if name == "shuffled_image_plus_position_history"
                        else name
                    )
                ],
                (
                    shuffled
                    if name == "shuffled_image"
                    else (
                        shuffled_combined
                        if name == "shuffled_image_plus_position_history"
                        else (
                            dev_image
                            if name == "image_only"
                            else (
                                dev_position
                                if name == "position_history_only"
                                else dev_combined
                            )
                        )
                    )
                ),
                dev_labels,
                class_order,
            )
            for name in predictions
        }
    else:
        metrics = {
            name: binary_metrics(dev_labels, prediction)
            for name, prediction in predictions.items()
        }
    sample_counts = {
        "training": int(len(train_labels)),
        "development": int(len(dev_labels)),
        "training_class_counts": {
            str(value): int(np.count_nonzero(train_labels == value))
            for value in np.unique(train_labels)
        },
        "development_class_counts": {
            str(value): int(np.count_nonzero(dev_labels == value))
            for value in np.unique(dev_labels)
        },
    }
    return models, {"metrics": metrics, "sample_counts": sample_counts}


def directional_peak(
    candidates: list[Any], scores: np.ndarray, direction: str, grid_step_mm: float
) -> dict[str, Any] | None:
    values: dict[tuple[int, int], float] = {}
    coordinates = [candidate.coordinate_mm for candidate in candidates]
    minimum = math.floor(min(coordinates) / grid_step_mm) * grid_step_mm
    for candidate, score in zip(candidates, scores):
        if candidate.direction != direction:
            continue
        bin_index = int(round((candidate.coordinate_mm - minimum) / grid_step_mm))
        key = (bin_index, int(candidate.frame_sequence))
        values[key] = max(values.get(key, 0.0), float(score))
    by_bin: dict[int, list[float]] = {}
    for (bin_index, _), score in values.items():
        by_bin.setdefault(bin_index, []).append(score)
    eligible = {
        index: observations
        for index, observations in by_bin.items()
        if len(observations) >= 2
    }
    if not eligible:
        return None
    selected = max(
        eligible,
        key=lambda index: (float(np.quantile(eligible[index], 0.75)), -index),
    )
    return {
        "coordinate_mm": float(minimum + selected * grid_step_mm),
        "score_q75": float(np.quantile(eligible[selected], 0.75)),
        "distinct_frame_count": len(eligible[selected]),
    }


def spatial_metrics(results: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    tolerance = float(protocol["evaluation"]["localization_tolerance_mm"])
    emitted = [item for item in results if item["actor_status"] == "SUCCESS"]
    correct = [
        item
        for item in emitted
        if float(item["localization_error_mm_evaluation_only"]) <= tolerance
    ]
    wrong = len(emitted) - len(correct)
    errors = [float(item["localization_error_mm_evaluation_only"]) for item in correct]
    differences = [
        float(item["forward_reverse_peak_difference_mm"])
        for item in results
        if item["forward_reverse_peak_difference_mm"] is not None
    ]
    return {
        "session_count": len(results),
        "emitted_count": len(emitted),
        "correct_count": len(correct),
        "wrong_stop_count": wrong,
        "abstention_count": len(results) - len(emitted),
        "precision": len(correct) / len(emitted) if emitted else 0.0,
        "recall": len(correct) / len(results) if results else 0.0,
        "median_localization_error_mm": float(np.median(errors)) if errors else None,
        "p90_localization_error_mm": (
            float(np.quantile(errors, 0.90)) if errors else None
        ),
        "median_forward_reverse_peak_difference_mm": (
            float(np.median(differences)) if differences else None
        ),
    }


def spatial_gate(metrics: dict[str, Any], protocol: dict[str, Any]) -> bool:
    evaluation = protocol["evaluation"]
    median_error = metrics["median_localization_error_mm"]
    median_direction = metrics["median_forward_reverse_peak_difference_mm"]
    return bool(
        metrics["precision"] >= float(evaluation["minimum_l4_precision"])
        and metrics["recall"] >= float(evaluation["minimum_l4_recall"])
        and metrics["wrong_stop_count"] <= int(evaluation["maximum_wrong_stops"])
        and median_error is not None
        and median_error
        <= float(evaluation["maximum_median_localization_error_mm"])
        and median_direction is not None
        and median_direction
        <= float(evaluation["maximum_median_forward_reverse_peak_difference_mm"])
    )


def run_spatial_development(
    sessions: list[PilotSession],
    shared_model: ExtraTreesClassifier,
    anatomy_model: ExtraTreesClassifier,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    spatial = protocol["spatial_map"]
    class_index = int(protocol["offline_labels"]["l4_class_index"])
    class_column = list(anatomy_model.classes_).index(class_index)
    reference = float(protocol["evaluation"]["l4_reference_coordinate_mm"])
    threshold_rows = []
    for threshold in spatial["calibration_acceptance_thresholds"]:
        config = SpatialMapConfig(
            grid_step_mm=float(spatial["grid_step_mm"]),
            minimum_event_separation_mm=float(spatial["minimum_event_separation_mm"]),
            raw_peak_minimum_score=float(spatial["raw_peak_minimum_score"]),
            acceptance_score=float(threshold),
            minimum_forward_frames=int(spatial["minimum_forward_frames"]),
            minimum_reverse_frames=int(spatial["minimum_reverse_frames"]),
        )
        results = []
        for session in sessions:
            features = session.source.features
            shared = positive_probability(shared_model, features)
            anatomy = anatomy_model.predict_proba(features)[:, class_column]
            score = np.minimum(shared, anatomy)
            inference, events = reconstruct_landmark(
                session.source.candidates,
                shared,
                anatomy,
                map_config=config,
                coordinate_offset_mm=0.0,
                minimum_top_score_margin=float(
                    spatial["minimum_top_event_score_margin"]
                ),
            )
            error = (
                abs(float(inference.coordinate_mm) - reference)
                if inference.coordinate_mm is not None
                else None
            )
            forward = directional_peak(
                session.source.candidates,
                score,
                "FORWARD",
                float(spatial["grid_step_mm"]),
            )
            reverse = directional_peak(
                session.source.candidates,
                score,
                "REVERSE",
                float(spatial["grid_step_mm"]),
            )
            difference = (
                abs(float(forward["coordinate_mm"]) - float(reverse["coordinate_mm"]))
                if forward is not None and reverse is not None
                else None
            )
            results.append(
                {
                    "session_id": session.source.session_id,
                    "actor_status": inference.status,
                    "actor_coordinate_mm": inference.coordinate_mm,
                    "actor_uncertainty_interval_mm": inference.uncertainty_interval_mm,
                    "actor_reason": inference.reason,
                    "accepted_event_count": inference.accepted_event_count,
                    "reference_coordinate_mm_evaluation_only": reference,
                    "localization_error_mm_evaluation_only": error,
                    "forward_peak": forward,
                    "reverse_peak": reverse,
                    "forward_reverse_peak_difference_mm": difference,
                    "events": [asdict(event) for event in events],
                }
            )
        metrics = spatial_metrics(results, protocol)
        threshold_rows.append(
            {
                "acceptance_threshold": float(threshold),
                "metrics": metrics,
                "gate_pass": spatial_gate(metrics, protocol),
                "results": results,
            }
        )
    selected = max(
        threshold_rows,
        key=lambda row: (
            bool(row["gate_pass"]),
            int(row["metrics"]["correct_count"]),
            -int(row["metrics"]["wrong_stop_count"]),
            float(row["metrics"]["precision"]),
            float(row["acceptance_threshold"]),
        ),
    )
    return {
        "status": "PASS" if selected["gate_pass"] else "FAIL",
        "selected": selected,
        "threshold_rows": threshold_rows,
        "coordinate_calibration_applied": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_hash = validate_protocol_seal(args.protocol)
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != "sonogym-stage1-supervised-anatomical-discovery-v1":
        raise RuntimeError("unexpected supervised pilot protocol identity")
    capture_summary = validate_capture(protocol, args.capture_root)
    validate_annotations(protocol)
    capture_protocol = yaml.safe_load(
        (args.capture_root / "context" / "protocol.yaml").read_text(encoding="utf-8")
    )
    by_id = {str(item["session_id"]): item for item in capture_protocol["sessions"]}
    training_ids = [str(value) for value in protocol["split_policy"]["training_session_ids"]]
    development_ids = [
        str(value) for value in protocol["split_policy"]["development_session_ids"]
    ]
    unused_ids = [str(value) for value in protocol["split_policy"]["unused_session_ids"]]
    if set(training_ids) & set(development_ids) or set(training_ids) & set(unused_ids):
        raise RuntimeError("session splits overlap")
    if set(development_ids) & set(unused_ids):
        raise RuntimeError("session splits overlap")
    if set(training_ids + development_ids + unused_ids) != set(by_id):
        raise RuntimeError("session split coverage differs from capture protocol")

    patch_spec = PatchSpec(
        **dataclass_kwargs(PatchSpec, protocol["patch"])
    )
    tracker_config = InterfaceTrackerConfig(
        **dataclass_kwargs(InterfaceTrackerConfig, protocol["tracker"])
    )
    forbidden = set(protocol["information_boundary"]["actor_forbidden"])
    sessions: dict[str, PilotSession] = {}
    audits = []
    for session_id in training_ids + development_ids + unused_ids:
        session, audit = audit_and_load_session(
            by_id[session_id],
            capture_root=args.capture_root,
            patch_spec=patch_spec,
            tracker_config=tracker_config,
            minimum_anatomy_fraction=float(
                protocol["offline_labels"]["minimum_group_a_fraction"]
            ),
            column_stride_pixels=int(protocol["patch"]["column_stride_pixels"]),
            pixel_resolution_mm=float(protocol["patch"]["pixel_resolution_mm"]),
            forbidden_fields=forbidden,
            load_candidates=session_id not in unused_ids,
        )
        if session is not None:
            sessions[session_id] = session
        audits.append(audit)
    training = [sessions[value] for value in training_ids]
    development = [sessions[value] for value in development_ids]
    shared_models, shared_results = classification_arms(
        training,
        development,
        anatomy_only=False,
        protocol=protocol,
        seed_base=0,
    )
    anatomy_models, anatomy_results = classification_arms(
        training,
        development,
        anatomy_only=True,
        protocol=protocol,
        seed_base=10,
    )
    shared_image = shared_results["metrics"]["image_only"]
    anatomy_image = anatomy_results["metrics"]["image_only"]
    anatomy_shuffled = anatomy_results["metrics"]["shuffled_image"]
    anatomy_position = anatomy_results["metrics"]["position_history_only"]
    anatomy_combined = anatomy_results["metrics"]["image_plus_position_history"]
    gate = protocol["image_evidence_gate"]
    image_minus_shuffled = float(anatomy_image["macro_recall"]) - float(
        anatomy_shuffled["macro_recall"]
    )
    combined_minus_position = float(anatomy_combined["macro_recall"]) - float(
        anatomy_position["macro_recall"]
    )
    image_evidence_pass = bool(
        float(shared_image["precision"]) >= float(gate["minimum_shared_precision"])
        and float(shared_image["recall"]) >= float(gate["minimum_shared_recall"])
        and float(anatomy_image["macro_recall"])
        >= float(gate["minimum_image_only_macro_recall"])
        and image_minus_shuffled
        >= float(gate["minimum_image_only_minus_shuffled_macro_recall"])
    )
    meaningful_increment = bool(
        combined_minus_position
        >= float(
            gate["minimum_combined_minus_position_macro_recall_for_meaningful_increment"]
        )
    )
    spatial = (
        run_spatial_development(
            development,
            shared_models["image_only"],
            anatomy_models["image_only"],
            protocol,
        )
        if image_evidence_pass
        else {
            "status": "NOT_RUN_IMAGE_EVIDENCE_GATE",
            "selected": None,
            "threshold_rows": [],
            "coordinate_calibration_applied": False,
        }
    )
    overall = "PASS" if image_evidence_pass and spatial["status"] == "PASS" else "FAIL"
    model_path = args.output_dir / "development_models.joblib"
    joblib.dump(
        {
            "schema": "sonogym-stage1-supervised-development-models-v1",
            "protocol_sha256": protocol_hash,
            "deployment_authorized": False,
            "shared_models": shared_models,
            "anatomy_models": anatomy_models,
            "class_order": protocol["offline_labels"]["class_order"],
            "patch_spec": patch_spec,
            "tracker_config": tracker_config,
            "position_history_fields": protocol["position_history_control"]["fields"],
        },
        model_path,
        compress=3,
    )
    report = {
        "schema": "sonogym-stage1-supervised-pilot-report-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "capture_job_id": int(protocol["capture"]["job_id"]),
        "capture_summary": {
            "frame_count": int(capture_summary["frame_count"]),
            "session_count": int(capture_summary["session_count"]),
            "status": capture_summary["status"],
        },
        "training_session_ids": training_ids,
        "development_session_ids": development_ids,
        "unused_session_ids": unused_ids,
        "patient_count": 1,
        "patient_id_development_only": "s0010",
        "data_integrity": {
            "status": "PASS",
            "audited_session_count": len(audits),
            "audited_frame_count": sum(item["frame_count"] for item in audits),
            "sessions": audits,
            "renderer_source": "US_slicer.us_img_tensor",
            "raw_identity_in_actor_features": False,
            "label_definition": "dominant raw identity in a tracked-interface-centered patch",
            "label_limitation": (
                "Raw identity overlap supplies offline supervision; it does not prove "
                "that acoustic class 13 contains unique level morphology."
            ),
        },
        "shared_structure_detection": shared_results,
        "anatomical_level_diagnostic": anatomy_results,
        "control_differences": {
            "image_only_minus_shuffled_macro_recall": image_minus_shuffled,
            "image_plus_position_minus_position_macro_recall": combined_minus_position,
            "meaningful_increment_beyond_position": meaningful_increment,
        },
        "image_evidence_gate": {
            "status": "PASS" if image_evidence_pass else "FAIL",
            "criteria": gate,
        },
        "l4_spatial_development": spatial,
        "decision": overall,
        "checkpoint": {
            "path": str(model_path.resolve()),
            "sha256": file_sha256(model_path),
            "deployment_authorized": False,
        },
        "evaluation_boundary": {
            "development_evidence_only": True,
            "locked_test_claim": False,
            "patient_generalization_established": False,
            "phase2_started": False,
            "gpu_used": False,
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(args.output_dir / "pilot_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if overall == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
