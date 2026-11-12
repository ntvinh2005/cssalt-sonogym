#!/usr/bin/env python3
"""Fit and evaluate the frozen lightweight SonoGym Stage 1 landmark pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.contracts import ultrasound_sha256  # noqa: E402
from two_stage_navigation.stage1_features import PatchSpec  # noqa: E402
from two_stage_navigation.stage1_interfaces import InterfaceTrackerConfig  # noqa: E402
from two_stage_navigation.stage1_map import SpatialMapConfig  # noqa: E402
from two_stage_navigation.stage1_pipeline import (  # noqa: E402
    ActorPatchCandidate,
    PatchLabelConfig,
    annotate_patch_candidate,
    binary_metrics,
    extract_actor_patch_candidates,
    reconstruct_landmark,
)


@dataclass
class SessionData:
    session_id: str
    split: str
    candidates: list[ActorPatchCandidate]
    primary_labels: np.ndarray
    verifier_labels: np.ndarray
    target_coordinate_mm: float

    @property
    def features(self) -> np.ndarray:
        if not self.candidates:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([item.feature for item in self.candidates]).astype(np.float32)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def phase_direction(phase: str) -> str | None:
    if phase == "OUTBOUND_CAUDAL":
        return "FORWARD"
    if phase == "RETURN_CRANIAL":
        return "REVERSE"
    if phase == "HOME_BASELINE":
        return None
    raise ValueError(f"unexpected acquisition phase {phase!r}")


def dataclass_kwargs(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    return {name: values[name] for name in cls.__dataclass_fields__}


def validate_protocol_seal(protocol_path: Path) -> str:
    seal_path = protocol_path.with_suffix(".sha256")
    expected = seal_path.read_text(encoding="utf-8").split()[0]
    actual = file_sha256(protocol_path)
    if actual != expected:
        raise RuntimeError("Stage 1 protocol seal mismatch")
    return actual


def validate_capture(stage1_protocol: dict[str, Any], capture_root: Path) -> dict[str, Any]:
    summary = read_json(capture_root / "results" / "capture_summary.json")
    expected = stage1_protocol["capture"]
    checks = {
        "job_id": capture_root.name,
        "protocol_id": summary["protocol_id"],
        "protocol_sha256": summary["protocol_sha256"],
        "public_manifest_sha256": summary["public_manifest_sha256"],
        "privileged_manifest_sha256": summary["privileged_manifest_sha256"],
        "frame_count": summary["frame_count"],
    }
    for key, value in checks.items():
        if str(value) != str(expected[key]):
            raise RuntimeError(f"capture provenance mismatch for {key}: {value!r}")
    if summary["status"] != "CAPTURE_PASS" or summary["g1_decision"] != "PASS":
        raise RuntimeError("capture did not pass G0/G1 integrity")
    return summary


def load_session(
    session: dict[str, Any],
    *,
    capture_root: Path,
    patch_spec: PatchSpec,
    tracker_config: InterfaceTrackerConfig,
    label_config: PatchLabelConfig,
    column_stride_pixels: int,
    pixel_resolution_mm: float,
    home_z_index: float,
    label_resolution_mm: float,
) -> SessionData:
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
    verifier_labels: list[int] = []
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
            annotation = annotate_patch_candidate(raw, candidate, patch_spec, label_config)
            candidates.append(candidate)
            primary_labels.append(annotation.primary_label)
            verifier_labels.append(
                -1 if annotation.verifier_label is None else annotation.verifier_label
            )
    goal_z_values = {round(float(row["goal_cmd_pose"][1]), 6) for row in privileged_rows}
    if len(goal_z_values) != 1:
        raise RuntimeError(f"nonconstant evaluation goal in {session_id}")
    target_coordinate_mm = (
        float(next(iter(goal_z_values))) - home_z_index
    ) * label_resolution_mm
    return SessionData(
        session_id=session_id,
        split=str(session["split"]),
        candidates=candidates,
        primary_labels=np.asarray(primary_labels, dtype=np.int8),
        verifier_labels=np.asarray(verifier_labels, dtype=np.int8),
        target_coordinate_mm=float(target_coordinate_mm),
    )


def positive_probability(model: ExtraTreesClassifier, features: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        raise RuntimeError("model has no positive class")
    return model.predict_proba(features)[:, classes.index(1)]


def concatenate_sessions(
    sessions: list[SessionData], *, verifier_only: bool
) -> tuple[np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    for session in sessions:
        features = session.features
        labels = session.verifier_labels if verifier_only else session.primary_labels
        mask = labels >= 0
        feature_rows.append(features[mask])
        label_rows.append(labels[mask])
    return np.concatenate(feature_rows), np.concatenate(label_rows)


def fit_model(
    features: np.ndarray, labels: np.ndarray, config: dict[str, Any], seed_offset: int
) -> ExtraTreesClassifier:
    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError("both model classes must be present")
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


def select_patch_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    rows = [binary_metrics(labels, probabilities, value) for value in np.linspace(0.05, 0.95, 19)]
    for row in rows:
        precision = float(row["precision"])
        recall = float(row["recall"])
        row["f1"] = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    selected = max(
        rows,
        key=lambda row: (
            min(float(row["precision"]), float(row["recall"])),
            float(row["f1"]),
            float(row["threshold"]),
        ),
    )
    return {"selected": selected, "curve": rows}


def session_prediction(
    session: SessionData,
    primary_model: ExtraTreesClassifier,
    verifier_model: ExtraTreesClassifier,
    spatial_protocol: dict[str, Any],
    acceptance_threshold: float,
) -> dict[str, Any]:
    features = session.features
    primary = positive_probability(primary_model, features)
    verifier = positive_probability(verifier_model, features)
    config = SpatialMapConfig(
        grid_step_mm=float(spatial_protocol["grid_step_mm"]),
        minimum_event_separation_mm=float(spatial_protocol["minimum_event_separation_mm"]),
        raw_peak_minimum_score=float(spatial_protocol["raw_peak_minimum_score"]),
        acceptance_score=float(acceptance_threshold),
        minimum_forward_frames=int(spatial_protocol["minimum_forward_frames"]),
        minimum_reverse_frames=int(spatial_protocol["minimum_reverse_frames"]),
    )
    inference, events = reconstruct_landmark(
        session.candidates, primary, verifier, map_config=config
    )
    accepted = [event for event in events if event.accepted]
    error = (
        abs(float(inference.coordinate_mm) - session.target_coordinate_mm)
        if inference.coordinate_mm is not None
        else None
    )
    return {
        "session_id": session.session_id,
        "split": session.split,
        "candidate_count": len(session.candidates),
        "target_coordinate_mm_evaluation_only": session.target_coordinate_mm,
        "actor_status": inference.status,
        "actor_coordinate_mm": inference.coordinate_mm,
        "actor_uncertainty_interval_mm": inference.uncertainty_interval_mm,
        "actor_reason": inference.reason,
        "accepted_event_count": len(accepted),
        "localization_error_mm_evaluation_only": error,
        "events": [asdict(event) for event in events],
    }


def evaluate_events(
    results: list[dict[str, Any]], tolerance_mm: float
) -> dict[str, Any]:
    tp = fp = fn = 0
    successful_errors: list[float] = []
    wrong_actor_stops = 0
    for result in results:
        target = float(result["target_coordinate_mm_evaluation_only"])
        accepted = [item for item in result["events"] if item["accepted"]]
        distances = [abs(float(item["coordinate_mm"]) - target) for item in accepted]
        matched = [index for index, distance in enumerate(distances) if distance <= tolerance_mm]
        if matched:
            tp += 1
            fp += len(accepted) - 1
        else:
            fn += 1
            fp += len(accepted)
        error = result["localization_error_mm_evaluation_only"]
        if result["actor_status"] == "SUCCESS":
            if error is not None and float(error) <= tolerance_mm:
                successful_errors.append(float(error))
            else:
                wrong_actor_stops += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    success_rate = len(successful_errors) / len(results) if results else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "event_precision": float(precision),
        "event_recall": float(recall),
        "session_success_count": len(successful_errors),
        "session_count": len(results),
        "session_success_rate": float(success_rate),
        "median_localization_error_mm": (
            float(np.median(successful_errors)) if successful_errors else None
        ),
        "wrong_actor_stop_count": wrong_actor_stops,
        "abstention_count": sum(item["actor_status"] == "ABSTAIN" for item in results),
    }


def gate_pass(metrics: dict[str, Any], protocol: dict[str, Any]) -> bool:
    evaluation = protocol["evaluation"]
    median = metrics["median_localization_error_mm"]
    return bool(
        metrics["event_precision"] >= float(evaluation["g2_minimum_event_precision"])
        and metrics["event_recall"] >= float(evaluation["g2_minimum_event_recall"])
        and metrics["session_success_rate"]
        >= float(evaluation["g3_minimum_session_success_rate"])
        and median is not None
        and median <= float(evaluation["g3_maximum_median_localization_error_mm"])
        and metrics["wrong_actor_stop_count"] == 0
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
    if protocol["protocol_id"] != "sonogym-unity-two-stage-stage1-v1":
        raise RuntimeError("unexpected Stage 1 protocol identity")
    capture_summary = validate_capture(protocol, args.capture_root)
    capture_protocol = yaml.safe_load(
        (args.capture_root / "context" / "protocol.yaml").read_text(encoding="utf-8")
    )
    sessions = capture_protocol["sessions"]
    patch_values = protocol["patch"]
    patch_spec = PatchSpec(
        **dataclass_kwargs(PatchSpec, patch_values)
    )
    tracker_config = InterfaceTrackerConfig(
        **dataclass_kwargs(InterfaceTrackerConfig, protocol["tracker"])
    )
    label_config = PatchLabelConfig(
        minimum_lumbar_fraction=float(protocol["offline_labels"]["minimum_lumbar_fraction"]),
        l4_label=int(protocol["offline_labels"]["l4_label"]),
    )
    load_kwargs = {
        "capture_root": args.capture_root,
        "patch_spec": patch_spec,
        "tracker_config": tracker_config,
        "label_config": label_config,
        "column_stride_pixels": int(patch_values["column_stride_pixels"]),
        "pixel_resolution_mm": float(patch_values["pixel_resolution_mm"]),
        "home_z_index": float(
            capture_protocol["geometry"]["mechanical_home_definition"]["patient_z_index"]
        ),
        "label_resolution_mm": float(
            capture_protocol["geometry"]["patient_label_resolution_mm"]
        ),
    }
    development_specs = [item for item in sessions if item["split"] != "untouched_audit"]
    development = [load_session(item, **load_kwargs) for item in development_specs]
    training = [item for item in development if item.split == "train_dev"]
    calibration = [item for item in development if item.split == "calibration_dev"]

    primary_x, primary_y = concatenate_sessions(training, verifier_only=False)
    verifier_x, verifier_y = concatenate_sessions(training, verifier_only=True)
    primary_model = fit_model(primary_x, primary_y, protocol["models"], 0)
    verifier_model = fit_model(verifier_x, verifier_y, protocol["models"], 1)

    calibration_primary_x, calibration_primary_y = concatenate_sessions(
        calibration, verifier_only=False
    )
    calibration_verifier_x, calibration_verifier_y = concatenate_sessions(
        calibration, verifier_only=True
    )
    patch_metrics = {
        "primary": select_patch_threshold(
            calibration_primary_y,
            positive_probability(primary_model, calibration_primary_x),
        ),
        "verifier": select_patch_threshold(
            calibration_verifier_y,
            positive_probability(verifier_model, calibration_verifier_x),
        ),
    }
    tolerance = float(protocol["evaluation"]["event_match_tolerance_mm"])
    threshold_rows: list[dict[str, Any]] = []
    for threshold in protocol["spatial_map"]["calibration_acceptance_thresholds"]:
        results = [
            session_prediction(
                session,
                primary_model,
                verifier_model,
                protocol["spatial_map"],
                float(threshold),
            )
            for session in calibration
        ]
        metrics = evaluate_events(results, tolerance)
        threshold_rows.append(
            {
                "acceptance_threshold": float(threshold),
                "gate_pass": gate_pass(metrics, protocol),
                "metrics": metrics,
                "results": results,
            }
        )
    passing = [item for item in threshold_rows if item["gate_pass"]]
    selected = max(
        passing if passing else threshold_rows,
        key=lambda item: (
            item["metrics"]["event_precision"],
            item["metrics"]["event_recall"],
            item["metrics"]["session_success_rate"],
            -(
                item["metrics"]["median_localization_error_mm"]
                if item["metrics"]["median_localization_error_mm"] is not None
                else math.inf
            ),
            item["acceptance_threshold"],
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
            "training_session_ids": [item.session_id for item in training],
            "calibration_session_ids": [item.session_id for item in calibration],
            "training_primary_examples": len(primary_y),
            "training_primary_positive": int(np.sum(primary_y)),
            "training_verifier_examples": len(verifier_y),
            "training_verifier_positive": int(np.sum(verifier_y)),
            "patch_metrics": patch_metrics,
            "selected_calibration": selected,
            "development_gate": "PASS" if selected["gate_pass"] else "FAIL",
        },
    )
    if not selected["gate_pass"]:
        print(json.dumps({"stage1_status": "DEVELOPMENT_NO_GO", "selected": selected}, indent=2))
        raise SystemExit(3)

    bundle = {
        "schema": "sonogym-stage1-primary-verifier-bundle-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "capture_job_id": int(protocol["capture"]["job_id"]),
        "primary_model": primary_model,
        "verifier_model": verifier_model,
        "patch_spec": patch_spec,
        "tracker_config": tracker_config,
        "column_stride_pixels": int(patch_values["column_stride_pixels"]),
        "pixel_resolution_mm": float(patch_values["pixel_resolution_mm"]),
        "acceptance_threshold": float(selected["acceptance_threshold"]),
        "spatial_map": protocol["spatial_map"],
        "actor_forbidden": protocol["information_boundary"]["actor_forbidden"],
    }
    bundle_path = args.output_dir / "stage1_primary_verifier.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    bundle_hash = file_sha256(bundle_path)
    write_json(
        args.output_dir / "bundle_manifest.json",
        {
            "bundle": str(bundle_path.resolve()),
            "sha256": bundle_hash,
            "acceptance_threshold": bundle["acceptance_threshold"],
            "protocol_sha256": protocol_hash,
            "audit_opened": False,
        },
    )

    # The locked split is first opened only after the complete bundle has a recorded hash.
    audit_specs = [item for item in sessions if item["split"] == "untouched_audit"]
    audit = [load_session(item, **load_kwargs) for item in audit_specs]
    audit_primary_x, audit_primary_y = concatenate_sessions(audit, verifier_only=False)
    audit_verifier_x, audit_verifier_y = concatenate_sessions(audit, verifier_only=True)
    audit_patch_metrics = {
        "primary": binary_metrics(
            audit_primary_y,
            positive_probability(primary_model, audit_primary_x),
            float(patch_metrics["primary"]["selected"]["threshold"]),
        ),
        "verifier": binary_metrics(
            audit_verifier_y,
            positive_probability(verifier_model, audit_verifier_x),
            float(patch_metrics["verifier"]["selected"]["threshold"]),
        ),
    }
    audit_results = [
        session_prediction(
            session,
            primary_model,
            verifier_model,
            protocol["spatial_map"],
            float(bundle["acceptance_threshold"]),
        )
        for session in audit
    ]
    audit_metrics = evaluate_events(audit_results, tolerance)
    audit_gate = gate_pass(audit_metrics, protocol)
    final_report = {
        "schema": "sonogym-stage1-landmark-reconstruction-report-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "capture_job_id": int(protocol["capture"]["job_id"]),
        "bundle_sha256_before_audit": bundle_hash,
        "acceptance_threshold": bundle["acceptance_threshold"],
        "audit_session_ids": [item.session_id for item in audit],
        "audit_patch_metrics": audit_patch_metrics,
        "audit_event_metrics": audit_metrics,
        "audit_results": audit_results,
        "g2_decision": "PASS" if audit_gate else "FAIL",
        "g3_decision": "PASS" if audit_gate else "FAIL",
        "phase1_landmark_status": "PASS" if audit_gate else "NO_GO",
        "fixed_anatomy_only": True,
        "patient_generalization_established": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(args.output_dir / "audit_report.json", final_report)
    write_json(
        args.output_dir / "bundle_manifest.json",
        {
            "bundle": str(bundle_path.resolve()),
            "sha256": bundle_hash,
            "acceptance_threshold": bundle["acceptance_threshold"],
            "protocol_sha256": protocol_hash,
            "audit_opened": True,
            "audit_report": str((args.output_dir / "audit_report.json").resolve()),
        },
    )
    print(json.dumps(final_report, indent=2, sort_keys=True))
    if not audit_gate:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
