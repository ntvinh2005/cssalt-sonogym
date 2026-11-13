#!/usr/bin/env python3
"""Freeze actor-only L4 inference, then evaluate it in a separate invocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
import sys

if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.contracts import ultrasound_sha256  # noqa: E402
from two_stage_navigation.stage1_map import SpatialMapConfig  # noqa: E402
from two_stage_navigation.stage1_pipeline import (  # noqa: E402
    ActorPatchCandidate,
    extract_actor_patch_candidates,
    reconstruct_landmark,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_seal(path: Path) -> str:
    expected = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    observed = file_sha256(path)
    if observed != expected:
        raise RuntimeError(f"protocol seal mismatch: {path}")
    return observed


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_seal(path: Path) -> str:
    digest = file_sha256(path)
    seal = path.with_suffix(".sha256")
    seal.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def recursive_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for nested in value.values():
            keys.update(recursive_keys(nested))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for nested in value:
            keys.update(recursive_keys(nested))
        return keys
    return set()


def positive_probability(model: Any, features: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        raise RuntimeError("shared model has no positive class")
    return model.predict_proba(features)[:, classes.index(1)]


def map_config(protocol: dict[str, Any]) -> SpatialMapConfig:
    values = protocol["inference"]["spatial_map"]
    return SpatialMapConfig(
        grid_step_mm=float(values["grid_step_mm"]),
        minimum_event_separation_mm=float(values["minimum_event_separation_mm"]),
        raw_peak_minimum_score=float(values["raw_peak_minimum_score"]),
        acceptance_score=float(values["acceptance_score"]),
        minimum_forward_frames=int(values["minimum_forward_frames"]),
        minimum_reverse_frames=int(values["minimum_reverse_frames"]),
    )


def directional_peak(
    candidates: list[ActorPatchCandidate],
    scores: np.ndarray,
    direction: str,
    grid_step_mm: float,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    minimum = math.floor(
        min(candidate.coordinate_mm for candidate in candidates) / grid_step_mm
    ) * grid_step_mm
    by_bin_frame: dict[tuple[int, int], float] = {}
    for candidate, score in zip(candidates, scores):
        if candidate.direction != direction:
            continue
        index = int(round((candidate.coordinate_mm - minimum) / grid_step_mm))
        key = (index, candidate.frame_sequence)
        by_bin_frame[key] = max(by_bin_frame.get(key, 0.0), float(score))
    by_bin: dict[int, list[float]] = {}
    for (index, _), score in by_bin_frame.items():
        by_bin.setdefault(index, []).append(score)
    eligible = {index: values for index, values in by_bin.items() if len(values) >= 2}
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


def infer(
    candidates: list[ActorPatchCandidate],
    shared: np.ndarray,
    anatomy: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], list[Any]]:
    spatial = protocol["inference"]["spatial_map"]
    result, events = reconstruct_landmark(
        candidates,
        shared,
        anatomy,
        map_config=map_config(protocol),
        coordinate_offset_mm=float(spatial["coordinate_offset_mm"]),
        minimum_top_score_margin=float(spatial["minimum_top_event_score_margin"]),
    )
    actor_result = asdict(result)
    coordinate_rule = str(spatial.get("coordinate_rule", "accepted_event_peak"))
    if coordinate_rule == "accepted_event_peak":
        return actor_result, events
    if coordinate_rule != "accepted_event_support_midpoint":
        raise RuntimeError(f"unsupported coordinate rule: {coordinate_rule}")
    if result.status == "SUCCESS":
        accepted = [event for event in events if event.accepted]
        selected = min(
            accepted,
            key=lambda event: abs(float(event.coordinate_mm) - float(result.coordinate_mm)),
        )
        actor_result["coordinate_mm"] = 0.5 * (
            float(selected.support_interval_mm[0])
            + float(selected.support_interval_mm[1])
        )
    return actor_result, events


def validate_public_session(
    session_root: Path,
    session: dict[str, Any],
    forbidden: set[str],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    summary = read_json(session_root / "summary.json")
    if summary.get("status") != "PASSED" or int(summary.get("frame_count", 0)) != 211:
        raise RuntimeError(f"incomplete confirmation session: {session['session_id']}")
    rows = read_jsonl(session_root / "public" / "frames.jsonl")
    images = np.load(session_root / "public" / "ultrasound.npz")["ultrasound"]
    if len(rows) != 211 or images.shape != (211, 200, 150):
        raise RuntimeError(f"invalid public arrays: {session['session_id']}")
    expected_phases = (
        ["HOME_BASELINE"]
        + ["OUTBOUND_CAUDAL"] * 105
        + ["RETURN_CRANIAL"] * 105
    )
    position = np.zeros(2, dtype=np.float64)
    angle = 0.0
    for index, (row, image) in enumerate(zip(rows, images)):
        leaked = forbidden & recursive_keys(row)
        if leaked:
            raise RuntimeError(f"forbidden actor fields: {sorted(leaked)}")
        identity = (
            str(row["session_id"]),
            int(row["frame_sequence"]),
            str(row["frame_sha256"]),
        )
        if identity != (
            str(session["session_id"]),
            index,
            ultrasound_sha256(image),
        ):
            raise RuntimeError(f"public frame identity mismatch at {identity[:2]}")
        if not np.all(np.isfinite(image)):
            raise RuntimeError(f"nonfinite image at {identity[:2]}")
        if not all(
            bool(row[field])
            for field in (
                "action_accepted",
                "action_completed",
                "contact_ok",
                "workspace_ok",
                "image_valid",
            )
        ) or bool(row["action_clamped"]):
            raise RuntimeError(f"invalid actor transition at {identity[:2]}")
        position += np.asarray(row["executed_delta_home_mm"], dtype=np.float64)
        angle = float(
            (angle + float(row["executed_delta_angle_rad"]) + math.pi)
            % (2.0 * math.pi)
            - math.pi
        )
        if not np.allclose(position, row["relative_position_home_mm"], atol=1e-9):
            raise RuntimeError(f"position reconstruction mismatch at {identity[:2]}")
        observed_angle = float(row["relative_angle_home_rad"])
        difference = (angle - observed_angle + math.pi) % (2.0 * math.pi) - math.pi
        if abs(difference) > 1e-9:
            raise RuntimeError(f"angle reconstruction mismatch at {identity[:2]}")
    if [str(row["phase"]) for row in rows] != expected_phases:
        raise RuntimeError(f"invalid scan phases: {session['session_id']}")
    return rows, images


def actor_session(
    session_root: Path,
    session: dict[str, Any],
    protocol: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    include_timeline: bool,
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
                patch_spec=checkpoint["patch_spec"],
                tracker_config=checkpoint["tracker_config"],
                column_stride_pixels=int(
                    protocol["inference"]["column_stride_pixels"]
                ),
                pixel_resolution_mm=float(
                    protocol["inference"]["pixel_resolution_mm"]
                ),
            )
        )
    if not candidates:
        raise RuntimeError(f"no actor candidates in {session['session_id']}")
    features = np.stack([candidate.feature for candidate in candidates]).astype(
        np.float32
    )
    shared = positive_probability(checkpoint["shared_image_model"], features)
    class_order = [str(value) for value in checkpoint["class_order"]]
    expected_order = [str(value) for value in protocol["inference"]["class_order"]]
    if class_order != expected_order:
        raise RuntimeError("checkpoint class order differs from actor protocol")
    l4_index = int(protocol["inference"]["l4_class_index"])
    anatomy_model = checkpoint["anatomy_image_model"]
    anatomy_column = list(anatomy_model.classes_).index(l4_index)
    anatomy = anatomy_model.predict_proba(features)[:, anatomy_column]
    scores = np.minimum(shared, anatomy)
    final, events = infer(candidates, shared, anatomy, protocol)
    grid_step = float(protocol["inference"]["spatial_map"]["grid_step_mm"])
    forward = directional_peak(candidates, scores, "FORWARD", grid_step)
    reverse = directional_peak(candidates, scores, "REVERSE", grid_step)
    direction_difference = (
        abs(float(forward["coordinate_mm"]) - float(reverse["coordinate_mm"]))
        if forward is not None and reverse is not None
        else None
    )
    frame_scores: dict[int, float] = {}
    evidence = []
    for candidate, shared_score, anatomy_score, score in zip(
        candidates, shared, anatomy, scores
    ):
        frame_scores[candidate.frame_sequence] = max(
            frame_scores.get(candidate.frame_sequence, 0.0), float(score)
        )
        evidence.append(
            {
                "frame_sequence": candidate.frame_sequence,
                "direction": candidate.direction,
                "coordinate_mm": candidate.coordinate_mm,
                "shared_probability": float(shared_score),
                "l4_probability": float(anatomy_score),
                "combined_probability": float(score),
                "track_id": candidate.track_id,
            }
        )
    timeline = []
    if include_timeline:
        timeline_frames = sorted(set(range(0, 211, 5)) | {210})
        sequences = np.asarray(
            [candidate.frame_sequence for candidate in candidates], dtype=np.int64
        )
        for frame_sequence in timeline_frames:
            mask = sequences <= frame_sequence
            selected = np.flatnonzero(mask)
            current, _ = infer(
                [candidates[index] for index in selected],
                shared[selected],
                anatomy[selected],
                protocol,
            )
            timeline.append({"frame_sequence": frame_sequence, **current})
    return {
        "session_id": str(session["session_id"]),
        "seed": int(session["seed"]),
        "declared_home": {
            "patient_x_index": float(session["patient_x_index"]),
            "angle_rad": float(session["angle_rad"]),
        },
        "frame_count": len(rows),
        "candidate_count": len(candidates),
        "actor_status": final["status"],
        "actor_coordinate_mm": final["coordinate_mm"],
        "actor_uncertainty_interval_mm": final["uncertainty_interval_mm"],
        "actor_reason": final["reason"],
        "accepted_event_count": final["accepted_event_count"],
        "events": [asdict(event) for event in events],
        "forward_peak": forward,
        "reverse_peak": reverse,
        "forward_reverse_peak_difference_mm": direction_difference,
        "maximum_combined_probability_by_frame": {
            str(index): value for index, value in sorted(frame_scores.items())
        },
        "evidence": evidence,
        "timeline": timeline,
    }


def run_actor(args: argparse.Namespace) -> int:
    started = time.monotonic()
    actor_hash = validate_seal(args.actor_protocol)
    protocol = yaml.safe_load(args.actor_protocol.read_text(encoding="utf-8"))
    if protocol["protocol_id"] not in {
        "sonogym-stage1-l4-confirmation-v1",
        "sonogym-stage1-l4-confirmation-v2",
    }:
        raise RuntimeError("unexpected actor protocol")
    checkpoint_path = Path(protocol["checkpoint"]["path"])
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != str(protocol["checkpoint"]["sha256"]):
        raise RuntimeError("checkpoint hash mismatch")
    checkpoint = joblib.load(checkpoint_path)
    if checkpoint["schema"] != protocol["checkpoint"]["schema"]:
        raise RuntimeError("checkpoint schema mismatch")
    if bool(checkpoint["deployment_authorized"]):
        raise RuntimeError("development checkpoint authorization changed unexpectedly")
    if checkpoint["protocol_sha256"] != protocol["checkpoint"]["development_protocol_sha256"]:
        raise RuntimeError("checkpoint development provenance mismatch")
    demo_session = str(protocol["demo"]["session_id"])
    results = [
        actor_session(
            args.capture_root / "sessions" / str(session["session_id"]),
            session,
            protocol,
            checkpoint,
            include_timeline=str(session["session_id"]) == demo_session,
        )
        for session in protocol["sessions"]
    ]
    report = {
        "schema": "sonogym-stage1-confirmation-actor-v1",
        "actor_protocol_sha256": actor_hash,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_deployment_authorized": False,
        "training_or_threshold_selection_performed": False,
        "session_count": len(results),
        "results": results,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }
    write_json(args.output, report)
    digest = write_seal(args.output)
    print(json.dumps({"actor_result": str(args.output), "sha256": digest}, indent=2))
    return 0


def run_evaluator(args: argparse.Namespace) -> int:
    started = time.monotonic()
    evaluation_hash = validate_seal(args.evaluation_protocol)
    evaluation = yaml.safe_load(
        args.evaluation_protocol.read_text(encoding="utf-8")
    )
    if evaluation["protocol_id"] not in {
        "sonogym-stage1-l4-confirmation-evaluation-v1",
        "sonogym-stage1-l4-confirmation-evaluation-v2",
    }:
        raise RuntimeError("unexpected evaluation protocol")
    actor_protocol_hash = validate_seal(args.actor_protocol)
    if actor_protocol_hash != evaluation["actor_protocol"]["sha256"]:
        raise RuntimeError("evaluator references a different actor protocol")
    actor_seal = args.actor_result.with_suffix(".sha256")
    expected_actor_hash = actor_seal.read_text(encoding="utf-8").split()[0]
    if file_sha256(args.actor_result) != expected_actor_hash:
        raise RuntimeError("actor result changed after freezing")
    actor = read_json(args.actor_result)
    if actor.get("schema") != "sonogym-stage1-confirmation-actor-v1":
        raise RuntimeError("unexpected actor result schema")
    if actor.get("training_or_threshold_selection_performed") is not False:
        raise RuntimeError("actor result reports fitting or threshold selection")
    forbidden_result_fields = {
        "raw_labels",
        "reference_coordinates",
        "reference_coordinate_mm_evaluator_only",
        "localization_error_mm_evaluator_only",
        "evaluator_score",
    }
    leaked_result_fields = forbidden_result_fields & recursive_keys(actor)
    if leaked_result_fields:
        raise RuntimeError(
            f"privileged fields in frozen actor result: {sorted(leaked_result_fields)}"
        )
    annotation = evaluation["privileged_evaluation"]
    annotation_path = REPOSITORY_ROOT / str(annotation["annotation_path"])
    if file_sha256(annotation_path) != str(annotation["annotation_sha256"]):
        raise RuntimeError("evaluator annotation hash mismatch")
    expected_ids = [str(value) for value in evaluation["expected_session_ids"]]
    observed_ids = [str(value["session_id"]) for value in actor["results"]]
    if observed_ids != expected_ids:
        raise RuntimeError("actor result session order mismatch")
    reference = float(
        evaluation["privileged_evaluation"][
            "reference_coordinate_home_relative_mm"
        ]
    )
    tolerance = float(evaluation["criteria"]["localization_tolerance_mm"])
    session_rows = []
    for result in actor["results"]:
        coordinate = result["actor_coordinate_mm"]
        error = abs(float(coordinate) - reference) if coordinate is not None else None
        correct = bool(
            result["actor_status"] == "SUCCESS"
            and error is not None
            and error <= tolerance
        )
        session_root = args.capture_root / "sessions" / str(result["session_id"])
        privileged_rows = read_jsonl(
            session_root / "privileged" / "annotations.jsonl"
        )
        public_rows = read_jsonl(session_root / "public" / "frames.jsonl")
        phase_by_sequence = {
            int(row["frame_sequence"]): str(row["phase"]) for row in public_rows
        }
        direction_support = {
            phase: sum(
                int(
                    row["raw_label_pixel_counts"][
                        str(evaluation["privileged_evaluation"]["raw_label"])
                    ]
                    > 0
                )
                for row in privileged_rows
                if phase_by_sequence[int(row["frame_sequence"])] == phase
            )
            for phase in ("OUTBOUND_CAUDAL", "RETURN_CRANIAL")
        }
        session_rows.append(
            {
                "session_id": result["session_id"],
                "actor_status": result["actor_status"],
                "actor_coordinate_mm": coordinate,
                "reference_coordinate_mm_evaluator_only": reference,
                "localization_error_mm_evaluator_only": error,
                "correct_within_tolerance": correct,
                "forward_reverse_peak_difference_mm": result[
                    "forward_reverse_peak_difference_mm"
                ],
                "l4_visible_frames_evaluator_only": direction_support,
            }
        )
    emitted = [row for row in session_rows if row["actor_status"] == "SUCCESS"]
    correct = [row for row in session_rows if row["correct_within_tolerance"]]
    errors = [float(row["localization_error_mm_evaluator_only"]) for row in correct]
    differences = [
        float(row["forward_reverse_peak_difference_mm"])
        for row in session_rows
        if row["forward_reverse_peak_difference_mm"] is not None
    ]
    metrics = {
        "complete_capture_session_count": len(session_rows),
        "emitted_count": len(emitted),
        "correct_count": len(correct),
        "wrong_stop_count": len(emitted) - len(correct),
        "abstention_count": len(session_rows) - len(emitted),
        "median_localization_error_mm": float(np.median(errors)) if errors else None,
        "p90_localization_error_mm": (
            float(np.quantile(errors, 0.90)) if errors else None
        ),
        "median_forward_reverse_peak_difference_mm": (
            float(np.median(differences)) if differences else None
        ),
    }
    criteria = evaluation["criteria"]
    passed = bool(
        metrics["complete_capture_session_count"]
        == int(criteria["required_complete_capture_sessions"])
        and metrics["correct_count"] == int(criteria["required_correct_sessions"])
        and metrics["wrong_stop_count"] <= int(criteria["maximum_wrong_stops"])
        and metrics["abstention_count"] <= int(criteria["maximum_abstentions"])
        and metrics["median_localization_error_mm"] is not None
        and metrics["median_localization_error_mm"]
        <= float(criteria["maximum_median_localization_error_mm"])
        and metrics["median_forward_reverse_peak_difference_mm"] is not None
        and metrics["median_forward_reverse_peak_difference_mm"]
        <= float(
            criteria["maximum_median_forward_reverse_peak_difference_mm"]
        )
    )
    report = {
        "schema": "sonogym-stage1-confirmation-evaluation-v1",
        "decision": "PASS" if passed else "FAIL",
        "ready_for_phase1_demo": passed,
        "actor_result_sha256": expected_actor_hash,
        "actor_protocol_sha256": actor_protocol_hash,
        "evaluation_protocol_sha256": evaluation_hash,
        "metrics": metrics,
        "sessions": session_rows,
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
    raise SystemExit(run_actor(arguments) if arguments.command == "actor" else run_evaluator(arguments))
