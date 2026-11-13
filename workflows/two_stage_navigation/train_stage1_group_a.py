#!/usr/bin/env python3
"""Fit and evaluate Group A Stage 1 recognition and spatial reconstruction."""

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
    fit_model,
    phase_direction,
    positive_probability,
    read_json,
    read_jsonl,
    select_patch_threshold,
    validate_capture,
    validate_protocol_seal,
    write_json,
)
from two_stage_navigation.contracts import ultrasound_sha256  # noqa: E402
from two_stage_navigation.stage1_features import PatchSpec  # noqa: E402
from two_stage_navigation.stage1_interfaces import InterfaceTrackerConfig  # noqa: E402
from two_stage_navigation.stage1_map import (  # noqa: E402
    SpatialMapConfig,
    select_ordered_event_path,
)
from two_stage_navigation.stage1_pipeline import (  # noqa: E402
    ActorPatchCandidate,
    annotate_group_a_patch_candidate,
    extract_actor_patch_candidates,
    reconstruct_landmark,
)


@dataclass
class GroupASessionData:
    session_id: str
    split: str
    candidates: list[ActorPatchCandidate]
    primary_labels: np.ndarray
    anatomy_classes: np.ndarray

    @property
    def features(self) -> np.ndarray:
        if not self.candidates:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([item.feature for item in self.candidates]).astype(np.float32)


def validate_annotations(protocol: dict[str, Any]) -> dict[str, float]:
    specification = protocol["annotations"]
    path = REPOSITORY_ROOT / str(specification["path"])
    if file_sha256(path) != str(specification["sha256"]):
        raise RuntimeError("Group A annotation manifest seal mismatch")
    manifest = read_json(path)
    records = manifest["annotations"]["group_A_unity_method_reproduction"]
    observed = {
        str(record["anatomy_level"]): float(
            record["geometry"]["reference_centroid_home_relative_mm"]
        )
        for record in records
    }
    expected = {
        str(name): float(value)
        for name, value in protocol["evaluation"]["reference_coordinates_mm"].items()
    }
    if observed.keys() != expected.keys():
        raise RuntimeError("Group A anatomy identities do not match the frozen protocol")
    for name in expected:
        if not math.isclose(observed[name], expected[name], abs_tol=1e-9):
            raise RuntimeError(f"reference coordinate mismatch for {name}")
    return expected


def load_session(
    session: dict[str, Any],
    *,
    capture_root: Path,
    patch_spec: PatchSpec,
    tracker_config: InterfaceTrackerConfig,
    minimum_anatomy_fraction: float,
    column_stride_pixels: int,
    pixel_resolution_mm: float,
) -> GroupASessionData:
    session_id = str(session["session_id"])
    root = capture_root / "sessions" / session_id
    public_rows = read_jsonl(root / "public" / "frames.jsonl")
    privileged_rows = read_jsonl(root / "privileged" / "annotations.jsonl")
    images = np.load(root / "public" / "ultrasound.npz")["ultrasound"]
    raw_labels = np.load(root / "privileged" / "raw_labels.npz")["raw_labels"]
    if not (
        len(public_rows) == len(privileged_rows) == len(images) == len(raw_labels) == 211
    ):
        raise RuntimeError(f"incomplete session {session_id}")
    candidates: list[ActorPatchCandidate] = []
    primary_labels: list[int] = []
    anatomy_classes: list[int] = []
    for index, (public, privileged, image, raw) in enumerate(
        zip(public_rows, privileged_rows, images, raw_labels)
    ):
        identity = (session_id, index, ultrasound_sha256(image))
        if identity != (
            public["session_id"],
            int(public["frame_sequence"]),
            public["frame_sha256"],
        ) or identity != (
            privileged["session_id"],
            int(privileged["frame_sequence"]),
            privileged["frame_sha256"],
        ):
            raise RuntimeError(f"frame identity mismatch in {session_id}:{index}")
        direction = phase_direction(str(public["phase"]))
        if direction is None:
            continue
        actor_candidates = extract_actor_patch_candidates(
            image,
            probe_longitudinal_home_mm=float(public["relative_position_home_mm"][1]),
            relative_angle_rad=float(public["relative_angle_home_rad"]),
            declared_home_angle_rad=float(session["angle_rad"]),
            frame_sequence=index,
            direction=direction,  # type: ignore[arg-type]
            patch_spec=patch_spec,
            tracker_config=tracker_config,
            column_stride_pixels=column_stride_pixels,
            pixel_resolution_mm=pixel_resolution_mm,
        )
        for candidate in actor_candidates:
            annotation = annotate_group_a_patch_candidate(
                raw,
                candidate,
                patch_spec,
                minimum_anatomy_fraction,
            )
            candidates.append(candidate)
            primary_labels.append(annotation.primary_label)
            anatomy_classes.append(
                -1 if annotation.anatomy_class is None else annotation.anatomy_class
            )
    return GroupASessionData(
        session_id=session_id,
        split=str(session["split"]),
        candidates=candidates,
        primary_labels=np.asarray(primary_labels, dtype=np.int8),
        anatomy_classes=np.asarray(anatomy_classes, dtype=np.int8),
    )


def concatenate_sessions(
    sessions: list[GroupASessionData], *, anatomy_only: bool
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for session in sessions:
        current_labels = (
            session.anatomy_classes if anatomy_only else session.primary_labels
        )
        mask = current_labels >= 0
        features.append(session.features[mask])
        labels.append(current_labels[mask])
    return np.concatenate(features), np.concatenate(labels)


def fit_anatomy_model(
    features: np.ndarray, labels: np.ndarray, config: dict[str, Any]
) -> ExtraTreesClassifier:
    expected = set(range(7))
    if set(np.unique(labels)) != expected:
        raise RuntimeError("all seven Group A classes must be present during fitting")
    model = ExtraTreesClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        class_weight=str(config["class_weight"]),
        n_jobs=int(config["n_jobs"]),
        random_state=int(config["random_seed"]) + 1,
    )
    model.fit(features, labels)
    return model


def anatomy_patch_metrics(
    model: ExtraTreesClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    class_order: list[str],
) -> dict[str, Any]:
    prediction = model.predict(features).astype(np.int64)
    confusion = np.zeros((len(class_order), len(class_order)), dtype=np.int64)
    for truth, predicted in zip(labels.astype(np.int64), prediction):
        confusion[truth, predicted] += 1
    per_class: dict[str, Any] = {}
    for index, name in enumerate(class_order):
        tp = int(confusion[index, index])
        fp = int(np.sum(confusion[:, index]) - tp)
        fn = int(np.sum(confusion[index, :]) - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[name] = {
            "support": int(np.sum(confusion[index, :])),
            "precision": float(precision),
            "recall": float(recall),
        }
    return {
        "accuracy": float(np.mean(prediction == labels)),
        "macro_precision": float(
            np.mean([item["precision"] for item in per_class.values()])
        ),
        "macro_recall": float(
            np.mean([item["recall"] for item in per_class.values()])
        ),
        "per_class": per_class,
        "confusion_matrix_rows_truth_columns_prediction": confusion.tolist(),
    }


def session_prediction(
    session: GroupASessionData,
    primary_model: ExtraTreesClassifier,
    anatomy_model: ExtraTreesClassifier,
    class_order: list[str],
    reference_coordinates: dict[str, float],
    spatial_protocol: dict[str, Any],
    ordered_protocol: dict[str, Any],
    acceptance_threshold: float,
    coordinate_offsets_mm: dict[str, float],
) -> dict[str, Any]:
    features = session.features
    primary = positive_probability(primary_model, features)
    anatomy = anatomy_model.predict_proba(features)
    class_columns = {int(value): index for index, value in enumerate(anatomy_model.classes_)}
    config = SpatialMapConfig(
        grid_step_mm=float(spatial_protocol["grid_step_mm"]),
        minimum_event_separation_mm=float(
            spatial_protocol["minimum_event_separation_mm"]
        ),
        raw_peak_minimum_score=float(spatial_protocol["raw_peak_minimum_score"]),
        acceptance_score=float(acceptance_threshold),
        minimum_forward_frames=int(spatial_protocol["minimum_forward_frames"]),
        minimum_reverse_frames=int(spatial_protocol["minimum_reverse_frames"]),
    )
    class_events: list[list[Any]] = []
    independent_inferences: list[Any] = []
    for class_index, name in enumerate(class_order):
        inference, events = reconstruct_landmark(
            session.candidates,
            primary,
            anatomy[:, class_columns[class_index]],
            map_config=config,
            coordinate_offset_mm=0.0,
            minimum_top_score_margin=float(
                spatial_protocol["minimum_top_event_score_margin"]
            ),
        )
        independent_inferences.append(inference)
        class_events.append(events)

    selection_mode = str(
        ordered_protocol.get("selection_mode", "independent_top_event")
    )
    selected_events: list[Any | None]
    path_log_score = None
    path_score_margin = None
    if selection_mode == "independent_top_event":
        selected_events = []
        for inference, events in zip(independent_inferences, class_events):
            if inference.status != "SUCCESS" or inference.coordinate_mm is None:
                selected_events.append(None)
                continue
            selected_events.append(
                min(
                    (event for event in events if event.accepted),
                    key=lambda event: abs(
                        event.coordinate_mm - float(inference.coordinate_mm)
                    ),
                )
            )
    elif selection_mode == "joint_ordered_path":
        path = select_ordered_event_path(
            class_events,
            float(ordered_protocol["minimum_adjacent_separation_mm"]),
        )
        selected_events = (
            list(path.events) if path is not None else [None] * len(class_order)
        )
        if path is not None:
            path_log_score = path.log_score
            path_score_margin = path.score_margin
    else:
        raise RuntimeError(f"unsupported ordered-map selection mode: {selection_mode}")

    predictions: list[dict[str, Any]] = []
    for name, selected, events in zip(class_order, selected_events, class_events):
        offset = float(coordinate_offsets_mm[name])
        coordinate = (
            float(selected.coordinate_mm) + offset if selected is not None else None
        )
        uncertainty = (
            [
                float(selected.support_interval_mm[0]) + offset,
                float(selected.support_interval_mm[1]) + offset,
            ]
            if selected is not None
            else None
        )
        reference = float(reference_coordinates[name])
        error = (
            abs(float(coordinate) - reference)
            if coordinate is not None
            else None
        )
        predictions.append(
            {
                "anatomy": name,
                "actor_status": "SUCCESS" if selected is not None else "ABSTAIN",
                "actor_coordinate_mm": coordinate,
                "actor_uncertainty_interval_mm": uncertainty,
                "actor_reason": None if selected is not None else "NO_COMPLETE_ORDERED_PATH",
                "coordinate_offset_mm": offset,
                "reference_coordinate_mm_evaluation_only": reference,
                "localization_error_mm_evaluation_only": error,
                "accepted_event_count": sum(event.accepted for event in events),
                "selected_event_coordinate_mm": (
                    float(selected.coordinate_mm) if selected is not None else None
                ),
                "selected_event_score": (
                    float(selected.score) if selected is not None else None
                ),
                "events": [asdict(event) for event in events],
            }
        )
    successful = [
        float(item["actor_coordinate_mm"])
        for item in predictions
        if item["actor_status"] == "SUCCESS"
    ]
    complete = len(successful) == len(class_order)
    minimum_separation = float(ordered_protocol["minimum_adjacent_separation_mm"])
    ordered = complete and all(
        right - left >= minimum_separation
        for left, right in zip(successful, successful[1:])
    )
    return {
        "session_id": session.session_id,
        "split": session.split,
        "candidate_count": len(session.candidates),
        "map_actor_status": "SUCCESS" if ordered else "ABSTAIN",
        "map_actor_reason": None if ordered else "INCOMPLETE_OR_MISORDERED_MAP",
        "ordered_map": bool(ordered),
        "ordered_path_log_score": path_log_score,
        "ordered_path_score_margin": path_score_margin,
        "predictions": predictions,
    }


def calibrate_coordinate_offsets(
    results: list[dict[str, Any]],
    class_order: list[str],
    method: str,
) -> dict[str, float]:
    samples = {
        name: [
            float(prediction["reference_coordinate_mm_evaluation_only"])
            - float(prediction["actor_coordinate_mm"])
            for result in results
            for prediction in result["predictions"]
            if prediction["anatomy"] == name
            and prediction["actor_status"] == "SUCCESS"
            and prediction["actor_coordinate_mm"] is not None
        ]
        for name in class_order
    }
    if any(not values for values in samples.values()):
        return {name: 0.0 for name in class_order}
    if method == "one_global_median_offset_across_all_classes":
        offset = float(
            np.median([value for values in samples.values() for value in values])
        )
        return {name: offset for name in class_order}
    if method == "per_class_midrange_minimax_offset":
        return {
            name: 0.5 * (float(min(values)) + float(max(values)))
            for name, values in samples.items()
        }
    raise RuntimeError(f"unsupported coordinate calibration: {method}")


def evaluate_results(
    results: list[dict[str, Any]],
    class_order: list[str],
    tolerance_mm: float,
) -> dict[str, Any]:
    expected = len(results) * len(class_order)
    emitted = correct = 0
    errors: list[float] = []
    per_class: dict[str, dict[str, int]] = {
        name: {"success": 0, "wrong_stop": 0, "abstain": 0} for name in class_order
    }
    complete_maps = 0
    for result in results:
        session_correct = True
        for prediction in result["predictions"]:
            name = str(prediction["anatomy"])
            error = prediction["localization_error_mm_evaluation_only"]
            if prediction["actor_status"] == "SUCCESS":
                emitted += 1
                if error is not None and float(error) <= tolerance_mm:
                    correct += 1
                    errors.append(float(error))
                    per_class[name]["success"] += 1
                else:
                    session_correct = False
                    per_class[name]["wrong_stop"] += 1
            else:
                session_correct = False
                per_class[name]["abstain"] += 1
        if session_correct and result["ordered_map"]:
            complete_maps += 1
    wrong_stops = emitted - correct
    session_count = len(results)
    return {
        "expected_event_count": expected,
        "emitted_event_count": emitted,
        "correct_event_count": correct,
        "event_precision": correct / emitted if emitted else 0.0,
        "event_recall": correct / expected if expected else 0.0,
        "wrong_actor_stop_count": wrong_stops,
        "abstention_count": expected - emitted,
        "median_localization_error_mm": float(np.median(errors)) if errors else None,
        "p90_localization_error_mm": (
            float(np.quantile(errors, 0.90)) if errors else None
        ),
        "complete_map_session_count": complete_maps,
        "session_count": session_count,
        "complete_map_session_rate": (
            complete_maps / session_count if session_count else 0.0
        ),
        "per_class": {
            name: {
                **counts,
                "session_success_rate": (
                    counts["success"] / session_count if session_count else 0.0
                ),
            }
            for name, counts in per_class.items()
        },
    }


def gate_pass(metrics: dict[str, Any], protocol: dict[str, Any]) -> bool:
    evaluation = protocol["evaluation"]
    median = metrics["median_localization_error_mm"]
    return bool(
        metrics["event_precision"] >= float(evaluation["minimum_event_precision"])
        and metrics["event_recall"] >= float(evaluation["minimum_event_recall"])
        and metrics["complete_map_session_rate"]
        >= float(evaluation["minimum_complete_map_session_rate"])
        and median is not None
        and median <= float(evaluation["maximum_median_localization_error_mm"])
        and metrics["wrong_actor_stop_count"]
        <= int(evaluation["maximum_wrong_actor_stops"])
        and all(
            item["session_success_rate"]
            >= float(evaluation["require_every_class_session_success_rate"])
            for item in metrics["per_class"].values()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-protocol", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    protocol_hash = validate_protocol_seal(args.stage1_protocol)
    protocol = yaml.safe_load(args.stage1_protocol.read_text(encoding="utf-8"))
    if protocol["protocol_id"] not in {
        "sonogym-unity-two-stage-stage1-group-a-v2",
        "sonogym-unity-two-stage-stage1-group-a-v3",
        "sonogym-unity-two-stage-stage1-group-a-v4",
        "sonogym-unity-two-stage-stage1-group-a-v4",
    }:
        raise RuntimeError("unexpected Group A Stage 1 protocol identity")
    capture_summary = validate_capture(protocol, args.capture_root)
    reference_coordinates = validate_annotations(protocol)
    capture_protocol = yaml.safe_load(
        (args.capture_root / "context" / "protocol.yaml").read_text(encoding="utf-8")
    )
    class_order = [str(value) for value in protocol["offline_labels"]["class_order"]]
    patch_values = protocol["patch"]
    patch_spec = PatchSpec(**dataclass_kwargs(PatchSpec, patch_values))
    tracker_config = InterfaceTrackerConfig(
        **dataclass_kwargs(InterfaceTrackerConfig, protocol["tracker"])
    )
    load_kwargs = {
        "capture_root": args.capture_root,
        "patch_spec": patch_spec,
        "tracker_config": tracker_config,
        "minimum_anatomy_fraction": float(
            protocol["offline_labels"]["minimum_group_a_fraction"]
        ),
        "column_stride_pixels": int(patch_values["column_stride_pixels"]),
        "pixel_resolution_mm": float(patch_values["pixel_resolution_mm"]),
    }
    sessions = [
        load_session(specification, **load_kwargs)
        for specification in capture_protocol["sessions"]
    ]
    training = [item for item in sessions if item.split == "train_dev"]
    calibration = [item for item in sessions if item.split == "calibration_dev"]
    evaluation_sessions = [
        item for item in sessions if item.split == "untouched_audit"
    ]

    primary_x, primary_y = concatenate_sessions(training, anatomy_only=False)
    anatomy_x, anatomy_y = concatenate_sessions(training, anatomy_only=True)
    primary_model = fit_model(primary_x, primary_y, protocol["models"], 0)
    anatomy_model = fit_anatomy_model(anatomy_x, anatomy_y, protocol["models"])

    calibration_primary_x, calibration_primary_y = concatenate_sessions(
        calibration, anatomy_only=False
    )
    calibration_anatomy_x, calibration_anatomy_y = concatenate_sessions(
        calibration, anatomy_only=True
    )
    patch_metrics = {
        "primary": select_patch_threshold(
            calibration_primary_y,
            positive_probability(primary_model, calibration_primary_x),
        ),
        "anatomy": anatomy_patch_metrics(
            anatomy_model,
            calibration_anatomy_x,
            calibration_anatomy_y,
            class_order,
        ),
    }
    tolerance = float(protocol["evaluation"]["event_match_tolerance_mm"])
    zero_offsets = {name: 0.0 for name in class_order}
    threshold_rows: list[dict[str, Any]] = []
    for threshold in protocol["spatial_map"]["calibration_acceptance_thresholds"]:
        raw_results = [
            session_prediction(
                session,
                primary_model,
                anatomy_model,
                class_order,
                reference_coordinates,
                protocol["spatial_map"],
                protocol["ordered_map"],
                float(threshold),
                zero_offsets,
            )
            for session in calibration
        ]
        coordinate_offsets_mm = calibrate_coordinate_offsets(
            raw_results,
            class_order,
            str(protocol["spatial_map"]["coordinate_calibration"]),
        )
        results = [
            session_prediction(
                session,
                primary_model,
                anatomy_model,
                class_order,
                reference_coordinates,
                protocol["spatial_map"],
                protocol["ordered_map"],
                float(threshold),
                coordinate_offsets_mm,
            )
            for session in calibration
        ]
        metrics = evaluate_results(results, class_order, tolerance)
        threshold_rows.append(
            {
                "acceptance_threshold": float(threshold),
                "coordinate_offsets_mm": coordinate_offsets_mm,
                "coordinate_offset_sample_count_by_class": {
                    name: sum(
                        prediction["actor_status"] == "SUCCESS"
                        for result in raw_results
                        for prediction in result["predictions"]
                        if prediction["anatomy"] == name
                    )
                    for name in class_order
                },
                "gate_pass": gate_pass(metrics, protocol),
                "metrics": metrics,
                "results": results,
            }
        )
    passing = [item for item in threshold_rows if item["gate_pass"]]
    threshold_tie_break = str(
        protocol["spatial_map"].get("passing_threshold_tie_break", "highest")
    )
    if threshold_tie_break not in {"highest", "lowest"}:
        raise RuntimeError("passing_threshold_tie_break must be highest or lowest")
    selected = max(
        passing if passing else threshold_rows,
        key=lambda item: (
            item["metrics"]["event_precision"],
            item["metrics"]["event_recall"],
            item["metrics"]["complete_map_session_rate"],
            -(
                item["metrics"]["median_localization_error_mm"]
                if item["metrics"]["median_localization_error_mm"] is not None
                else math.inf
            ),
            (
                item["acceptance_threshold"]
                if threshold_tie_break == "highest"
                else -item["acceptance_threshold"]
            ),
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "calibration_thresholds.json", threshold_rows)
    write_json(
        args.output_dir / "development_report.json",
        {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_hash,
            "capture_job_id": int(protocol["capture"]["job_id"]),
            "capture_status": capture_summary["status"],
            "training_session_ids": [item.session_id for item in training],
            "calibration_session_ids": [item.session_id for item in calibration],
            "training_primary_examples": len(primary_y),
            "training_primary_positive": int(np.sum(primary_y)),
            "training_anatomy_examples": len(anatomy_y),
            "training_anatomy_class_counts": {
                name: int(np.count_nonzero(anatomy_y == index))
                for index, name in enumerate(class_order)
            },
            "patch_metrics": patch_metrics,
            "selected_calibration": selected,
            "development_gate": "PASS" if selected["gate_pass"] else "FAIL",
        },
    )
    if not selected["gate_pass"]:
        print(json.dumps({"group_a_status": "DEVELOPMENT_NO_GO", "selected": selected}, indent=2))
        raise SystemExit(3)

    bundle = {
        "schema": "sonogym-stage1-group-a-bundle-v2",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "capture_job_id": int(protocol["capture"]["job_id"]),
        "primary_model": primary_model,
        "anatomy_model": anatomy_model,
        "class_order": class_order,
        "patch_spec": patch_spec,
        "tracker_config": tracker_config,
        "column_stride_pixels": int(patch_values["column_stride_pixels"]),
        "pixel_resolution_mm": float(patch_values["pixel_resolution_mm"]),
        "acceptance_threshold": float(selected["acceptance_threshold"]),
        "coordinate_offsets_mm": selected["coordinate_offsets_mm"],
        "minimum_top_event_score_margin": float(
            protocol["spatial_map"]["minimum_top_event_score_margin"]
        ),
        "spatial_map": protocol["spatial_map"],
        "ordered_map": protocol["ordered_map"],
        "actor_forbidden": protocol["information_boundary"]["actor_forbidden"],
    }
    bundle_path = args.output_dir / "stage1_group_a.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    bundle_hash = file_sha256(bundle_path)
    write_json(
        args.output_dir / "bundle_manifest.json",
        {
            "bundle": str(bundle_path.resolve()),
            "sha256": bundle_hash,
            "protocol_sha256": protocol_hash,
            "evaluation_status": protocol["split_policy"]["evaluation_status"],
            "evaluation_opened": False,
        },
    )

    evaluation_primary_x, evaluation_primary_y = concatenate_sessions(
        evaluation_sessions, anatomy_only=False
    )
    evaluation_anatomy_x, evaluation_anatomy_y = concatenate_sessions(
        evaluation_sessions, anatomy_only=True
    )
    evaluation_results = [
        session_prediction(
            session,
            primary_model,
            anatomy_model,
            class_order,
            reference_coordinates,
            protocol["spatial_map"],
            protocol["ordered_map"],
            float(bundle["acceptance_threshold"]),
            {
                name: float(bundle["coordinate_offsets_mm"][name])
                for name in class_order
            },
        )
        for session in evaluation_sessions
    ]
    evaluation_metrics = evaluate_results(
        evaluation_results, class_order, tolerance
    )
    evaluation_gate = gate_pass(evaluation_metrics, protocol)
    final_report = {
        "schema": "sonogym-stage1-group-a-report-v2",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "capture_job_id": int(protocol["capture"]["job_id"]),
        "bundle_sha256_before_evaluation": bundle_hash,
        "evaluation_status": protocol["split_policy"]["evaluation_status"],
        "locked_test_claim": False,
        "acceptance_threshold": bundle["acceptance_threshold"],
        "coordinate_offsets_mm": bundle["coordinate_offsets_mm"],
        "evaluation_session_ids": [
            item.session_id for item in evaluation_sessions
        ],
        "evaluation_patch_metrics": {
            "primary": select_patch_threshold(
                evaluation_primary_y,
                positive_probability(primary_model, evaluation_primary_x),
            )["selected"],
            "anatomy": anatomy_patch_metrics(
                anatomy_model,
                evaluation_anatomy_x,
                evaluation_anatomy_y,
                class_order,
            ),
        },
        "evaluation_event_metrics": evaluation_metrics,
        "evaluation_results": evaluation_results,
        "group_a_decision": "PASS" if evaluation_gate else "NO_GO",
        "fixed_anatomy_only": True,
        "patient_generalization_established": False,
        "stage2_established": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(args.output_dir / "evaluation_report.json", final_report)
    write_json(
        args.output_dir / "bundle_manifest.json",
        {
            "bundle": str(bundle_path.resolve()),
            "sha256": bundle_hash,
            "protocol_sha256": protocol_hash,
            "evaluation_status": protocol["split_policy"]["evaluation_status"],
            "evaluation_opened": True,
            "evaluation_report": str(
                (args.output_dir / "evaluation_report.json").resolve()
            ),
        },
    )
    print(json.dumps(final_report, indent=2, sort_keys=True))
    if not evaluation_gate:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
