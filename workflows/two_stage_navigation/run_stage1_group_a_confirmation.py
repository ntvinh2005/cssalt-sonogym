#!/usr/bin/env python3
"""Freeze seven-landmark actor inference, then evaluate it separately."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

from run_stage1_confirmation import (
    file_sha256,
    read_json,
    read_jsonl,
    recursive_keys,
    validate_public_session,
    validate_seal,
    write_json,
    write_seal,
)
from train_stage1 import dataclass_kwargs


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
import sys

if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.stage1_map import (  # noqa: E402
    SpatialMapConfig,
    select_ordered_event_path,
)
from two_stage_navigation.stage1_pipeline import (  # noqa: E402
    ActorPatchCandidate,
    extract_actor_patch_candidates,
    reconstruct_landmark,
)


ACTOR_PROTOCOL_ID = "sonogym-stage1-group-a-confirmation-v1"
EVALUATION_PROTOCOL_ID = "sonogym-stage1-group-a-confirmation-evaluation-v1"


def positive_probability(model: Any, features: np.ndarray) -> np.ndarray:
    columns = {int(value): index for index, value in enumerate(model.classes_)}
    if 1 not in columns:
        raise RuntimeError("primary model has no positive class")
    return model.predict_proba(features)[:, columns[1]]


def actor_session(
    session_root: Path,
    session: dict[str, Any],
    protocol: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    rows, images = validate_public_session(
        session_root,
        session,
        set(protocol["information_boundaries"]["actor_forbidden"]),
    )
    candidates: list[ActorPatchCandidate] = []
    for index in range(1, len(rows)):
        phase = str(rows[index]["phase"])
        direction = "FORWARD" if phase == "OUTBOUND_CAUDAL" else "REVERSE"
        candidates.extend(
            extract_actor_patch_candidates(
                images[index],
                probe_longitudinal_home_mm=float(
                    rows[index]["relative_position_home_mm"][1]
                ),
                relative_angle_rad=float(rows[index]["relative_angle_home_rad"]),
                declared_home_angle_rad=float(session["angle_rad"]),
                frame_sequence=index,
                direction=direction,
                patch_spec=bundle["patch_spec"],
                tracker_config=bundle["tracker_config"],
                column_stride_pixels=int(bundle["column_stride_pixels"]),
                pixel_resolution_mm=float(bundle["pixel_resolution_mm"]),
            )
        )
    if not candidates:
        raise RuntimeError(f"no actor candidates in {session['session_id']}")
    features = np.stack([candidate.feature for candidate in candidates]).astype(
        np.float32
    )
    primary = positive_probability(bundle["primary_model"], features)
    anatomy_model = bundle["anatomy_model"]
    anatomy = anatomy_model.predict_proba(features)
    anatomy_columns = {
        int(value): index for index, value in enumerate(anatomy_model.classes_)
    }
    class_order = [str(value) for value in bundle["class_order"]]
    if class_order != [str(value) for value in protocol["inference"]["class_order"]]:
        raise RuntimeError("bundle class order differs from actor protocol")
    spatial = bundle["spatial_map"]
    map_config = SpatialMapConfig(
        **dataclass_kwargs(
            SpatialMapConfig,
            {
                **spatial,
                "acceptance_score": float(bundle["acceptance_threshold"]),
            },
        )
    )
    class_events = []
    for class_index in range(len(class_order)):
        _, events = reconstruct_landmark(
            candidates,
            primary,
            anatomy[:, anatomy_columns[class_index]],
            map_config=map_config,
            coordinate_offset_mm=0.0,
            minimum_top_score_margin=float(spatial["minimum_top_event_score_margin"]),
        )
        class_events.append(events)
    ordered = bundle["ordered_map"]
    path = select_ordered_event_path(
        class_events,
        float(ordered["minimum_adjacent_separation_mm"]),
        first_event_must_be_minimum_coordinate=bool(
            ordered["first_event_must_be_minimum_coordinate"]
        ),
        last_event_must_be_maximum_coordinate=bool(
            ordered["last_event_must_be_maximum_coordinate"]
        ),
        expected_adjacent_separations_mm=tuple(
            float(value) for value in bundle["expected_adjacent_separations_mm"]
        ),
        separation_penalty_per_mm=float(ordered["separation_penalty_per_mm"]),
    )
    offsets = {
        str(key): float(value) for key, value in bundle["coordinate_offsets_mm"].items()
    }
    predictions = []
    selected_events = list(path.events) if path is not None else [None] * len(class_order)
    for name, selected, events in zip(class_order, selected_events, class_events):
        coordinate = (
            float(selected.coordinate_mm) + offsets[name]
            if selected is not None
            else None
        )
        predictions.append(
            {
                "anatomy": name,
                "actor_status": "SUCCESS" if selected is not None else "ABSTAIN",
                "actor_coordinate_mm": coordinate,
                "actor_uncertainty_interval_mm": (
                    [
                        float(selected.support_interval_mm[0]) + offsets[name],
                        float(selected.support_interval_mm[1]) + offsets[name],
                    ]
                    if selected is not None
                    else None
                ),
                "actor_reason": None if selected is not None else "NO_COMPLETE_ORDERED_PATH",
                "selected_event_coordinate_mm": (
                    float(selected.coordinate_mm) if selected is not None else None
                ),
                "selected_event_score": (
                    float(selected.score) if selected is not None else None
                ),
                "accepted_event_count": sum(event.accepted for event in events),
                "events": [asdict(event) for event in events],
            }
        )
    successful = [
        float(row["actor_coordinate_mm"])
        for row in predictions
        if row["actor_status"] == "SUCCESS"
    ]
    complete = len(successful) == len(class_order)
    minimum_separation = float(ordered["minimum_adjacent_separation_mm"])
    is_ordered = complete and all(
        right - left >= minimum_separation
        for left, right in zip(successful, successful[1:])
    )
    return {
        "session_id": str(session["session_id"]),
        "seed": int(session["seed"]),
        "declared_home": {
            "patient_x_index": float(session["patient_x_index"]),
            "angle_rad": float(session["angle_rad"]),
        },
        "frame_count": len(rows),
        "candidate_count": len(candidates),
        "map_actor_status": "SUCCESS" if is_ordered else "ABSTAIN",
        "map_actor_reason": None if is_ordered else "INCOMPLETE_OR_MISORDERED_MAP",
        "ordered_path_log_score": float(path.log_score) if path is not None else None,
        "ordered_path_score_margin": (
            float(path.score_margin)
            if path is not None and path.score_margin is not None
            else None
        ),
        "predictions": predictions,
    }


def run_actor(args: argparse.Namespace) -> int:
    started = time.monotonic()
    protocol_hash = validate_seal(args.actor_protocol)
    protocol = yaml.safe_load(args.actor_protocol.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != ACTOR_PROTOCOL_ID:
        raise RuntimeError("unexpected Group A actor protocol")
    bundle_path = Path(protocol["bundle"]["path"])
    bundle_hash = file_sha256(bundle_path)
    if bundle_hash != str(protocol["bundle"]["sha256"]):
        raise RuntimeError("Group A bundle hash mismatch")
    bundle = joblib.load(bundle_path)
    if bundle["schema"] != protocol["bundle"]["schema"]:
        raise RuntimeError("Group A bundle schema mismatch")
    if bundle["protocol_sha256"] != protocol["bundle"]["development_protocol_sha256"]:
        raise RuntimeError("Group A bundle development provenance mismatch")
    results = [
        actor_session(
            args.capture_root / "sessions" / str(session["session_id"]),
            session,
            protocol,
            bundle,
        )
        for session in protocol["sessions"]
    ]
    report = {
        "schema": "sonogym-stage1-group-a-confirmation-actor-v1",
        "actor_protocol_sha256": protocol_hash,
        "bundle_sha256": bundle_hash,
        "training_or_threshold_selection_performed": False,
        "session_count": len(results),
        "results": results,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }
    leaked = set(protocol["information_boundaries"]["actor_forbidden"]) & recursive_keys(
        report
    )
    if leaked:
        raise RuntimeError(f"forbidden fields in actor report: {sorted(leaked)}")
    write_json(args.output, report)
    digest = write_seal(args.output)
    print(json.dumps({"actor_result": str(args.output), "sha256": digest}, indent=2))
    return 0


def annotation_references(evaluation: dict[str, Any]) -> dict[str, float]:
    specification = evaluation["privileged_evaluation"]
    path = REPOSITORY_ROOT / str(specification["annotation_path"])
    if file_sha256(path) != str(specification["annotation_sha256"]):
        raise RuntimeError("annotation manifest hash mismatch")
    manifest = read_json(path)
    records = manifest["annotations"]["group_A_unity_method_reproduction"]
    return {
        str(record["anatomy_level"]): float(
            record["geometry"]["reference_centroid_home_relative_mm"]
        )
        for record in records
    }


def evaluate_actor_result(
    actor: dict[str, Any], evaluation: dict[str, Any], capture_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    class_order = [str(value) for value in evaluation["class_order"]]
    references = annotation_references(evaluation)
    if list(references) != class_order:
        raise RuntimeError("annotation class order differs from evaluator protocol")
    expected_ids = [str(value) for value in evaluation["expected_session_ids"]]
    observed_ids = [str(row["session_id"]) for row in actor["results"]]
    if observed_ids != expected_ids:
        raise RuntimeError("actor result session order mismatch")
    tolerance = float(evaluation["criteria"]["localization_tolerance_mm"])
    raw_label_by_class = {
        str(key): str(value)
        for key, value in evaluation["privileged_evaluation"][
            "raw_label_by_class"
        ].items()
    }
    sessions = []
    correct_count = 0
    emitted_count = 0
    errors = []
    per_class = {
        name: {"correct": 0, "wrong_stop": 0, "abstain": 0, "errors_mm": []}
        for name in class_order
    }
    for result in actor["results"]:
        session_id = str(result["session_id"])
        privileged_rows = read_jsonl(
            capture_root / "sessions" / session_id / "privileged" / "annotations.jsonl"
        )
        public_rows = read_jsonl(
            capture_root / "sessions" / session_id / "public" / "frames.jsonl"
        )
        phase_by_sequence = {
            int(row["frame_sequence"]): str(row["phase"]) for row in public_rows
        }
        prediction_rows = []
        for prediction in result["predictions"]:
            name = str(prediction["anatomy"])
            coordinate = prediction["actor_coordinate_mm"]
            error = (
                abs(float(coordinate) - references[name])
                if coordinate is not None
                else None
            )
            emitted = prediction["actor_status"] == "SUCCESS" and error is not None
            correct = bool(emitted and float(error) <= tolerance)
            emitted_count += int(emitted)
            correct_count += int(correct)
            if error is not None:
                errors.append(float(error))
            status = "correct" if correct else "wrong_stop" if emitted else "abstain"
            per_class[name][status] += 1
            if correct:
                per_class[name]["errors_mm"].append(float(error))
            raw_label = raw_label_by_class[name]
            visible = {
                phase: sum(
                    int(row["raw_label_pixel_counts"][raw_label] > 0)
                    for row in privileged_rows
                    if phase_by_sequence[int(row["frame_sequence"])] == phase
                )
                for phase in ("OUTBOUND_CAUDAL", "RETURN_CRANIAL")
            }
            prediction_rows.append(
                {
                    "anatomy": name,
                    "actor_status": prediction["actor_status"],
                    "actor_coordinate_mm": coordinate,
                    "reference_coordinate_mm_evaluator_only": references[name],
                    "localization_error_mm_evaluator_only": error,
                    "correct_within_tolerance": correct,
                    "visible_frames_evaluator_only": visible,
                }
            )
        sessions.append(
            {
                "session_id": session_id,
                "map_actor_status": result["map_actor_status"],
                "predictions": prediction_rows,
            }
        )
    expected_count = len(class_order) * len(expected_ids)
    complete_maps = sum(
        all(row["correct_within_tolerance"] for row in session["predictions"])
        for session in sessions
    )
    metrics = {
        "session_count": len(sessions),
        "complete_map_session_count": complete_maps,
        "expected_event_count": expected_count,
        "emitted_event_count": emitted_count,
        "correct_event_count": correct_count,
        "wrong_actor_stop_count": emitted_count - correct_count,
        "abstention_count": expected_count - emitted_count,
        "event_precision": correct_count / emitted_count if emitted_count else 0.0,
        "event_recall": correct_count / expected_count if expected_count else 0.0,
        "median_localization_error_mm": float(np.median(errors)) if errors else None,
        "p90_localization_error_mm": float(np.quantile(errors, 0.9)) if errors else None,
        "maximum_localization_error_mm": max(errors) if errors else None,
        "per_class": {
            name: {
                "correct": row["correct"],
                "wrong_stop": row["wrong_stop"],
                "abstain": row["abstain"],
                "median_correct_error_mm": (
                    float(np.median(row["errors_mm"])) if row["errors_mm"] else None
                ),
            }
            for name, row in per_class.items()
        },
    }
    return metrics, sessions


def run_evaluator(args: argparse.Namespace) -> int:
    started = time.monotonic()
    evaluation_hash = validate_seal(args.evaluation_protocol)
    evaluation = yaml.safe_load(args.evaluation_protocol.read_text(encoding="utf-8"))
    if evaluation["protocol_id"] != EVALUATION_PROTOCOL_ID:
        raise RuntimeError("unexpected Group A evaluation protocol")
    actor_protocol_hash = validate_seal(args.actor_protocol)
    if actor_protocol_hash != evaluation["actor_protocol"]["sha256"]:
        raise RuntimeError("evaluator references a different actor protocol")
    expected_actor_hash = args.actor_result.with_suffix(".sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    if file_sha256(args.actor_result) != expected_actor_hash:
        raise RuntimeError("actor result changed after freezing")
    actor = read_json(args.actor_result)
    if actor.get("schema") != "sonogym-stage1-group-a-confirmation-actor-v1":
        raise RuntimeError("unexpected Group A actor result schema")
    if actor.get("training_or_threshold_selection_performed") is not False:
        raise RuntimeError("actor result reports fitting or threshold selection")
    forbidden = {
        "raw_labels",
        "reference_coordinates",
        "reference_coordinate_mm_evaluator_only",
        "localization_error_mm_evaluator_only",
        "evaluator_score",
    }
    leaked = forbidden & recursive_keys(actor)
    if leaked:
        raise RuntimeError(f"privileged fields in frozen actor result: {sorted(leaked)}")
    metrics, sessions = evaluate_actor_result(actor, evaluation, args.capture_root)
    criteria = evaluation["criteria"]
    passed = bool(
        metrics["session_count"] == int(criteria["required_complete_capture_sessions"])
        and metrics["complete_map_session_count"]
        == int(criteria["required_complete_map_sessions"])
        and metrics["correct_event_count"] == int(criteria["required_correct_events"])
        and metrics["wrong_actor_stop_count"] <= int(criteria["maximum_wrong_stops"])
        and metrics["abstention_count"] <= int(criteria["maximum_abstentions"])
        and metrics["median_localization_error_mm"] is not None
        and metrics["median_localization_error_mm"]
        <= float(criteria["maximum_median_localization_error_mm"])
    )
    report = {
        "schema": "sonogym-stage1-group-a-confirmation-evaluation-v1",
        "decision": "PASS" if passed else "FAIL",
        "actor_result_sha256": expected_actor_hash,
        "actor_protocol_sha256": actor_protocol_hash,
        "evaluation_protocol_sha256": evaluation_hash,
        "metrics": metrics,
        "sessions": sessions,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }
    write_json(args.output, report)
    write_seal(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    actor = subparsers.add_parser("actor")
    actor.add_argument("--actor-protocol", type=Path, required=True)
    actor.add_argument("--capture-root", type=Path, required=True)
    actor.add_argument("--output", type=Path, required=True)
    evaluator = subparsers.add_parser("evaluate")
    evaluator.add_argument("--actor-protocol", type=Path, required=True)
    evaluator.add_argument("--evaluation-protocol", type=Path, required=True)
    evaluator.add_argument("--capture-root", type=Path, required=True)
    evaluator.add_argument("--actor-result", type=Path, required=True)
    evaluator.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(
        run_actor(arguments) if arguments.command == "actor" else run_evaluator(arguments)
    )
