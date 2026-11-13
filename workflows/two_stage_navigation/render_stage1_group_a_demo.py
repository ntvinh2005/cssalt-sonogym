#!/usr/bin/env python3
"""Render actor-only seven-landmark Stage 1 map construction for one scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import yaml

from train_stage1 import dataclass_kwargs, phase_direction
from two_stage_navigation.stage1_map import (
    EvidenceObservation,
    SpatialEvidenceMap,
    SpatialMapConfig,
    select_ordered_event_path,
)
from two_stage_navigation.stage1_pipeline import extract_actor_patch_candidates


COLORS = (
    (80, 170, 255),
    (100, 230, 230),
    (80, 220, 120),
    (255, 210, 80),
    (255, 140, 80),
    (230, 100, 210),
    (180, 150, 255),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def positive_probability(model: Any, features: np.ndarray) -> np.ndarray:
    columns = {int(value): index for index, value in enumerate(model.classes_)}
    if 1 not in columns:
        raise RuntimeError("primary model has no positive class")
    return model.predict_proba(features)[:, columns[1]]


def map_x(value: float, minimum: float, maximum: float) -> int:
    return int(round(555 + (value - minimum) / (maximum - minimum) * 675))


def draw_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.48,
    color: tuple[int, int, int] = (225, 225, 225),
    thickness: int = 1,
) -> None:
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")

    protocol_hash = file_sha256(args.protocol)
    expected_protocol_hash = args.protocol.with_suffix(".sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    if protocol_hash != expected_protocol_hash:
        raise RuntimeError("protocol seal mismatch")
    bundle_manifest = read_json(args.bundle_manifest)
    bundle_hash = file_sha256(args.bundle)
    if bundle_hash != str(bundle_manifest["sha256"]):
        raise RuntimeError("bundle hash mismatch")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    bundle = joblib.load(args.bundle)
    evaluation = read_json(args.evaluation_report)
    if bundle["protocol_sha256"] != protocol_hash:
        raise RuntimeError("bundle and protocol do not match")
    if evaluation["protocol_sha256"] != protocol_hash:
        raise RuntimeError("evaluation report and protocol do not match")
    if evaluation["group_a_decision"] != "PASS":
        raise RuntimeError("refusing to render a non-passing Group A result")

    evaluation_ids = sorted(str(value) for value in evaluation["evaluation_session_ids"])
    session_id = args.session_id or evaluation_ids[0]
    if session_id not in evaluation_ids:
        raise RuntimeError("session is not present in the evaluation result")
    final_result = next(
        row for row in evaluation["evaluation_results"] if row["session_id"] == session_id
    )
    capture_protocol = yaml.safe_load(
        (args.capture_root / "context" / "protocol.yaml").read_text(encoding="utf-8")
    )
    session_spec = next(
        row for row in capture_protocol["sessions"] if row["session_id"] == session_id
    )
    session_root = args.capture_root / "sessions" / session_id / "public"
    images = np.load(session_root / "ultrasound.npz")["ultrasound"]
    public_rows = read_jsonl(session_root / "frames.jsonl")
    if len(images) != len(public_rows):
        raise RuntimeError("public frame metadata does not align with ultrasound images")

    candidates = []
    frame_candidate_indices: list[list[int]] = [[] for _ in public_rows]
    for frame_index, (image, row) in enumerate(zip(images, public_rows)):
        direction = phase_direction(str(row["phase"]))
        if direction is None:
            continue
        current = extract_actor_patch_candidates(
            image,
            probe_longitudinal_home_mm=float(row["relative_position_home_mm"][1]),
            relative_angle_rad=float(row["relative_angle_home_rad"]),
            declared_home_angle_rad=float(session_spec["angle_rad"]),
            frame_sequence=frame_index,
            direction=direction,
            patch_spec=bundle["patch_spec"],
            tracker_config=bundle["tracker_config"],
            column_stride_pixels=int(bundle["column_stride_pixels"]),
            pixel_resolution_mm=float(bundle["pixel_resolution_mm"]),
        )
        for candidate in current:
            frame_candidate_indices[frame_index].append(len(candidates))
            candidates.append(candidate)
    if not candidates:
        raise RuntimeError("scan produced no actor patch candidates")
    features = np.stack([candidate.feature for candidate in candidates]).astype(np.float32)
    primary = positive_probability(bundle["primary_model"], features)
    anatomy = bundle["anatomy_model"].predict_proba(features)
    anatomy_columns = {
        int(value): index for index, value in enumerate(bundle["anatomy_model"].classes_)
    }
    class_order = [str(value) for value in bundle["class_order"]]
    if len(class_order) != len(COLORS):
        raise RuntimeError("demo color table does not match the anatomy class count")
    class_probabilities = np.column_stack(
        [
            np.minimum(primary, anatomy[:, anatomy_columns[index]])
            for index in range(len(class_order))
        ]
    )

    coordinates = np.asarray([candidate.coordinate_mm for candidate in candidates])
    grid_step = float(bundle["spatial_map"]["grid_step_mm"])
    minimum = math.floor(float(np.min(coordinates)) / grid_step) * grid_step
    maximum = math.ceil(float(np.max(coordinates)) / grid_step) * grid_step
    map_config = SpatialMapConfig(
        **dataclass_kwargs(
            SpatialMapConfig,
            {
                **bundle["spatial_map"],
                "acceptance_score": float(bundle["acceptance_threshold"]),
            },
        )
    )
    maps = [SpatialEvidenceMap(minimum, maximum, map_config) for _ in class_order]
    ordered = bundle["ordered_map"]
    offsets = {str(key): float(value) for key, value in bundle["coordinate_offsets_mm"].items()}
    references = {
        str(key): float(value)
        for key, value in protocol["evaluation"]["reference_coordinates_mm"].items()
    }
    expected_separations = tuple(
        float(value) for value in bundle["expected_adjacent_separations_mm"]
    )
    final_selected = {
        row["anatomy"]: float(row["selected_event_coordinate_mm"])
        for row in final_result["predictions"]
    }

    low, high = np.quantile(images, [0.01, 0.99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise RuntimeError("invalid ultrasound display range")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "sonogym-stage1-seven-landmark-construction.mp4"
    preview_path = args.output_dir / "sonogym-stage1-seven-landmark-final.png"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (1280, 720),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the MP4 demo")

    first_complete_frame = None
    final_canvas = None
    final_path = None
    try:
        for frame_index, (image, row) in enumerate(zip(images, public_rows)):
            for candidate_index in frame_candidate_indices[frame_index]:
                candidate = candidates[candidate_index]
                for class_index, spatial_map in enumerate(maps):
                    spatial_map.add(
                        EvidenceObservation(
                            coordinate_mm=candidate.coordinate_mm,
                            probability=float(class_probabilities[candidate_index, class_index]),
                            direction=candidate.direction,
                            frame_sequence=candidate.frame_sequence,
                            track_id=candidate.track_id,
                        )
                    )
            events = [spatial_map.events() for spatial_map in maps]
            path = select_ordered_event_path(
                events,
                float(ordered["minimum_adjacent_separation_mm"]),
                first_event_must_be_minimum_coordinate=bool(
                    ordered["first_event_must_be_minimum_coordinate"]
                ),
                last_event_must_be_maximum_coordinate=bool(
                    ordered["last_event_must_be_maximum_coordinate"]
                ),
                expected_adjacent_separations_mm=expected_separations,
                separation_penalty_per_mm=float(ordered["separation_penalty_per_mm"]),
            )
            if path is not None and first_complete_frame is None:
                first_complete_frame = frame_index
            final_path = path

            normalized = np.clip((image.astype(np.float32) - low) / (high - low), 0, 1)
            ultrasound = cv2.applyColorMap(
                (normalized * 255).astype(np.uint8), cv2.COLORMAP_BONE
            )
            ultrasound = cv2.resize(ultrasound, (480, 640), interpolation=cv2.INTER_NEAREST)
            canvas = np.full((720, 1280, 3), 15, dtype=np.uint8)
            canvas[40:680, :480] = ultrasound
            cv2.rectangle(canvas, (0, 0), (479, 39), (0, 0, 0), -1)
            draw_text(canvas, "Clean SonoGym ultrasound (actor input)", (12, 27), 0.58)

            image_height, image_width = image.shape
            current_candidates = frame_candidate_indices[frame_index]
            ranked = sorted(
                current_candidates,
                key=lambda index: -float(np.max(class_probabilities[index])),
            )[:8]
            for candidate_index in ranked:
                candidate = candidates[candidate_index]
                class_index = int(np.argmax(class_probabilities[candidate_index]))
                x = int(round(candidate.column / (image_width - 1) * 479))
                y = 40 + int(round(candidate.interface_row / (image_height - 1) * 639))
                cv2.circle(canvas, (x, y), 4, COLORS[class_index], 1, cv2.LINE_AA)

            phase = str(row["phase"])
            probe_coordinate = float(row["relative_position_home_mm"][1])
            draw_text(canvas, "Stage 1: seven-landmark spatial evidence", (515, 28), 0.64)
            draw_text(
                canvas,
                f"{session_id} | frame {frame_index:03d}/{len(images)-1} | {phase} | executed z={probe_coordinate:+.1f} mm",
                (515, 54),
                0.44,
                (190, 190, 190),
            )
            draw_text(
                canvas,
                "Actor: ultrasound pixels + executed relative motion; no raw labels or target coordinates",
                (515, 78),
                0.40,
                (160, 210, 255),
            )
            status = "COMPLETE 7/7" if path is not None else "ABSTAIN: waiting for ordered bidirectional support"
            draw_text(
                canvas,
                f"Current map: {status}",
                (515, 103),
                0.48,
                (80, 230, 120) if path is not None else (100, 190, 255),
            )

            probe_x = map_x(probe_coordinate, minimum, maximum)
            cv2.line(canvas, (probe_x, 120), (probe_x, 610), (85, 85, 85), 1)
            for class_index, (name, spatial_map, class_events) in enumerate(
                zip(class_order, maps, events)
            ):
                y = 145 + class_index * 66
                color = COLORS[class_index]
                draw_text(canvas, name, (500, y + 5), 0.48, color)
                cv2.line(canvas, (555, y), (1230, y), (70, 70, 70), 1)
                for record in spatial_map.records():
                    if record.score_q75 <= 0.0:
                        continue
                    x = map_x(record.coordinate_mm, minimum, maximum)
                    height = max(1, int(round(24 * record.score_q75)))
                    intensity = float(np.clip(record.score_q75, 0.15, 1.0))
                    record_color = tuple(int(channel * intensity) for channel in color)
                    cv2.line(canvas, (x, y + 25), (x, y + 25 - height), record_color, 1)
                for event in class_events:
                    if event.accepted:
                        cv2.circle(
                            canvas,
                            (map_x(event.coordinate_mm, minimum, maximum), y),
                            4,
                            color,
                            -1,
                        )
                reference_raw = references[name] - offsets[name]
                cv2.circle(
                    canvas,
                    (map_x(reference_raw, minimum, maximum), y),
                    6,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
                if path is not None:
                    selected = path.events[class_index]
                    x = map_x(selected.coordinate_mm, minimum, maximum)
                    cv2.drawMarker(
                        canvas,
                        (x, y),
                        color,
                        cv2.MARKER_TRIANGLE_UP,
                        14,
                        2,
                        cv2.LINE_AA,
                    )
                    corrected = selected.coordinate_mm + offsets[name]
                    error = abs(corrected - references[name])
                    draw_text(
                        canvas,
                        f"{corrected:+.1f} mm  err {error:.1f}",
                        (1000, y - 7),
                        0.36,
                        color,
                    )

            draw_text(
                canvas,
                f"Home-relative projected evidence: {minimum:+.0f} to {maximum:+.0f} mm",
                (555, 637),
                0.42,
                (180, 180, 180),
            )
            draw_text(canvas, "bar = image score", (555, 662), 0.38, (165, 165, 165))
            draw_text(canvas, "dot = accepted event", (715, 662), 0.38, (165, 165, 165))
            draw_text(canvas, "triangle = actor selection", (900, 662), 0.38, (165, 165, 165))
            draw_text(canvas, "circle = evaluator-only reference", (555, 692), 0.38, (220, 220, 220))
            draw_text(
                canvas,
                f"threshold={bundle['acceptance_threshold']:.2f}; forward + reverse required",
                (855, 692),
                0.38,
                (165, 165, 165),
            )
            writer.write(canvas)
            final_canvas = canvas
    finally:
        writer.release()

    if final_canvas is None or final_path is None:
        raise RuntimeError("scan never produced a complete seven-landmark map")
    observed_final = {
        name: float(event.coordinate_mm)
        for name, event in zip(class_order, final_path.events)
    }
    if observed_final != final_selected:
        raise RuntimeError("rendered final actor path differs from the saved evaluation")
    if not cv2.imwrite(str(preview_path), final_canvas):
        raise RuntimeError("could not write final preview")
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if frame_count != len(images):
        raise RuntimeError("rendered video frame count mismatch")

    predictions = []
    for name, event in zip(class_order, final_path.events):
        actor_coordinate = float(event.coordinate_mm + offsets[name])
        predictions.append(
            {
                "anatomy": name,
                "selected_event_coordinate_mm": float(event.coordinate_mm),
                "actor_coordinate_mm": actor_coordinate,
                "evaluator_reference_mm": references[name],
                "localization_error_mm": abs(actor_coordinate - references[name]),
            }
        )
    manifest = {
        "schema": "sonogym-stage1-group-a-demo-v1",
        "session_id": session_id,
        "session_selection_rule": "lexicographically_first_evaluation_session",
        "actor_inputs": ["clean_ultrasound", "executed_home_relative_motion"],
        "actor_forbidden": bundle["actor_forbidden"],
        "evaluator_only_overlays": ["reference_coordinate", "localization_error"],
        "protocol_sha256": protocol_hash,
        "bundle_sha256": bundle_hash,
        "evaluation_report_sha256": file_sha256(args.evaluation_report),
        "frame_count": frame_count,
        "fps": args.fps,
        "duration_seconds": frame_count / args.fps,
        "first_complete_frame_sequence": first_complete_frame,
        "final_complete_landmark_count": len(final_path.events),
        "predictions": predictions,
        "video": str(video_path.resolve()),
        "video_sha256": file_sha256(video_path),
        "preview": str(preview_path.resolve()),
        "preview_sha256": file_sha256(preview_path),
        "fixed_anatomy_development_only": True,
    }
    manifest_path = args.output_dir / "demo_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
