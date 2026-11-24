#!/usr/bin/env python3
"""Render actual-execution replay of the shared incremental landmark map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflows.two_stage_navigation.build_s0010_clinician_scan_review import (
    CANVAS_SIZE,
    PANEL_HEIGHT,
    PANEL_WIDTH,
    add_title,
    put_text,
)
from workflows.two_stage_navigation.build_s0010_mbb_stage2_review import (
    evaluator_lookup,
    evaluator_panel,
    load_semantic_labels,
    map_acquisitions_to_robot_frames,
    read_json,
    read_jsonl,
    robot_panel,
    ultrasound_panel,
    validate_seal,
)
from two_stage_navigation.incremental_landmark_map import (
    IncrementalAnatomicalLandmarkMap,
    IncrementalLandmarkMapConfig,
)
from two_stage_navigation.sonoanatomy_multiview import SemanticObservation


COLORS = {
    "RIGHT_L4_TP": (255, 160, 40),
    "RIGHT_L4_SAP": (0, 220, 255),
    "RIGHT_L5_TP": (255, 80, 200),
    "RIGHT_L5_SAP": (80, 255, 120),
}


def put_text_at(
    panel: np.ndarray,
    value: str,
    x: int,
    y: int,
    *,
    color: tuple[int, int, int] = (235, 235, 235),
    scale: float = 0.40,
) -> None:
    cv2.putText(
        panel,
        value,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--map-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--historical-fallback", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_observations(
    actor: dict[str, Any],
) -> tuple[dict[int, list[tuple[str, SemanticObservation]]], list[np.ndarray]]:
    by_frame: dict[int, list[tuple[str, SemanticObservation]]] = {}
    coordinates = []
    for target in actor["targets"]:
        level = str(target["stage1_level"])
        sonoanatomy = target.get("sonoanatomy", {})
        selected = sonoanatomy.get("selected_trajectory_id")
        for trajectory in sonoanatomy.get("trajectory_decisions", []):
            if trajectory["trajectory_id"] != selected:
                continue
            for value in trajectory["semantic_observations"]:
                observation = SemanticObservation(**value)
                by_frame.setdefault(observation.frame_sequence, []).append(
                    (level, observation)
                )
                coordinates.append(np.asarray(observation.coordinate_mm, dtype=np.float64))
    return by_frame, coordinates


def map_panel(
    mapping: IncrementalAnatomicalLandmarkMap,
    frame: dict[str, Any],
    all_coordinates: list[np.ndarray],
    *,
    snapshot: dict[str, Any] | None = None,
    return_text: str = "",
) -> np.ndarray:
    if snapshot is None:
        snapshot = mapping.snapshot()
    panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 18, dtype=np.uint8)
    add_title(panel, "3 | Shared actor-owned 3D landmark map")
    plot_left, plot_top, plot_right, plot_bottom = 24, 65, 360, 330
    cv2.rectangle(panel, (plot_left, plot_top), (plot_right, plot_bottom), (90, 90, 90), 1)
    if all_coordinates:
        coordinates = np.asarray(all_coordinates)
        low = np.min(coordinates[:, :2], axis=0) - 2.0
        high = np.max(coordinates[:, :2], axis=0) + 2.0
        span = np.maximum(high - low, 1.0)

        def project(point: tuple[float, float, float]) -> tuple[int, int]:
            normalized = (np.asarray(point[:2]) - low) / span
            return (
                int(round(plot_left + normalized[0] * (plot_right - plot_left))),
                int(round(plot_bottom - normalized[1] * (plot_bottom - plot_top))),
            )

        for identity, color in COLORS.items():
            observations = mapping.observations(identity)
            for observation in observations:
                cv2.circle(panel, project(observation.coordinate_mm), 2, color, -1)
            hypothesis = snapshot["landmarks"][identity]
            if hypothesis["position_mm"] is not None:
                point = project(hypothesis["position_mm"])
                marker = 9 if hypothesis["status"] == "CONFIRMED" else 6
                cv2.drawMarker(
                    panel,
                    point,
                    color,
                    cv2.MARKER_TILTED_CROSS,
                    marker * 2,
                    2,
                    cv2.LINE_AA,
                )
    labels = {
        "RIGHT_L4_TP": "L4 TP",
        "RIGHT_L4_SAP": "L4 SAP",
        "RIGHT_L5_TP": "L5 TP",
        "RIGHT_L5_SAP": "L5 SAP",
    }
    for index, identity in enumerate(COLORS):
        row = snapshot["landmarks"][identity]
        uncertainty = row["rms_uncertainty_mm"]
        uncertainty_text = "N/A" if uncertainty is None else f"{float(uncertainty):.2f}"
        put_text_at(
            panel,
            f"{labels[identity]} {row['status']}  n={len(row['supporting_frame_sequences'])}",
            380,
            78 + index * 38,
            scale=0.40,
            color=COLORS[identity],
        )
        put_text_at(panel, f"RMS {uncertainty_text} mm", 380, 94 + index * 38, scale=0.39)
    put_text_at(
        panel,
        f"#{frame['frame_sequence']} {frame.get('scan_state', 'TP_VIEW_SEARCH')}",
        380,
        260,
        scale=0.40,
    )
    if return_text:
        put_text_at(panel, return_text, 380, 278, scale=0.39)
    put_text_at(
        panel,
        f"L4={snapshot['relations']['L4']['status']}  "
        f"L5={snapshot['relations']['L5']['status']}",
        380,
        298,
        scale=0.40,
    )
    put_text_at(
        panel,
        f"map complete={'YES' if snapshot['complete'] else 'NO'}",
        380,
        336,
        scale=0.46,
    )
    return panel


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    source_protocol = yaml.safe_load(args.source_protocol.read_text())
    map_protocol = yaml.safe_load(args.map_protocol.read_text())
    map_config = IncrementalLandmarkMapConfig(**map_protocol["landmark_map"])
    actor_path = args.session_dir / "results/actor_result.json"
    evaluator_path = args.session_dir / "results/evaluation_report.json"
    actor_sha = validate_seal(actor_path)
    evaluator_sha = validate_seal(evaluator_path)
    actor = read_json(actor_path)
    evaluator = read_json(evaluator_path)
    frames = read_jsonl(args.session_dir / "public/frames.jsonl")
    actions = read_jsonl(args.session_dir / "privileged/actions.jsonl")
    images = np.load(args.session_dir / "public/ultrasound.npz")["ultrasound"]
    raw_labels = np.load(args.session_dir / "privileged/raw_labels.npz")["raw_labels"]
    coordinates = np.load(
        args.session_dir / "privileged/pixel_to_volume_coordinates.npz"
    )["human_img_coords"]
    labels = load_semantic_labels(
        source_protocol, frames, images, raw_labels, coordinates
    )
    robot = cv2.VideoCapture(
        str(args.session_dir / "video/robot_probe_patient.mp4")
    )
    if not robot.isOpened():
        raise RuntimeError("cannot open recorded robot video")
    robot_indices = map_acquisitions_to_robot_frames(
        frames,
        actions,
        action_stride=int(source_protocol["visualization"]["action_stride"]),
        robot_frame_count=int(robot.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    observations, all_coordinates = selected_observations(actor)
    canonical = map_config.canonical_evidence
    sealed_map = actor.get("incremental_landmark_map") or {}
    map_events = {}
    return_texts = {}
    if canonical:
        for target in actor["targets"]:
            scan = target.get("sonoanatomy", {})
            before = scan.get("incremental_landmark_map_before_confirmation")
            after = scan.get("incremental_landmark_map")
            if before is None or after is None:
                continue
            attempts = scan["fresh_landmark_confirmations"]
            credited = {
                (identity, row["frame_sequence"])
                for identity, landmark in after["landmarks"].items()
                for row in landmark.get("fresh_confirmations", [])
            }
            local_attempts = [
                row["frame_sequence"] for row in attempts
                if row["confirmation_kind"] != "DEDICATED_TP_FRESH_CONFIRMATION"
            ]
            if local_attempts:
                map_events[min(local_attempts) - 1] = before
                map_events[max(local_attempts)] = after
            for row in attempts:
                used = (row["identity"], row["frame_sequence"]) in credited
                return_texts[row["frame_sequence"]] = (
                    f"return #{row['source_frame_sequence']}  credited={'YES' if used else 'NO'}"
                )
        map_events[len(frames) - 1] = sealed_map
        for identity, landmark in sealed_map["landmarks"].items():
            _, level, kind = identity.split("_")
            for row in landmark["supporting_observations"]:
                sequence = row["frame_sequence"]
                public = frames[sequence]
                values = observations.setdefault(sequence, [])
                if any(
                    obs.semantic_class == kind and obs.row == row["row"] and obs.column == row["column"]
                    for _, obs in values
                ):
                    continue
                observation = SemanticObservation(
                    semantic_class=kind,
                    frame_sequence=sequence,
                    frame_sha256=row["frame_sha256"],
                    viewpoint_id=row["viewpoint_id"],
                    row=row["row"],
                    column=row["column"],
                    coordinate_mm=tuple(row["coordinate_mm"]),
                    confidence=row["confidence"],
                    viewpoint_position_m=tuple(public["actor_base_to_ee_position_m"]),
                    viewpoint_rotation_columns=tuple(map(tuple, public["actor_base_to_ee_rotation_columns"])),
                    pixel_count=0,
                )
                values.append((level, observation))
                all_coordinates.append(np.asarray(row["coordinate_mm"]))
    target_reports = {row["target_name"]: row for row in evaluator["target_reports"]}
    inspected = evaluator_lookup(evaluator)
    full_path = args.output_dir / "s0010_incremental_landmark_map_actual_execution_full.mp4"
    blinded_path = args.output_dir / "s0010_incremental_landmark_map_actual_execution_blinded.mp4"
    writers = {
        False: cv2.VideoWriter(
            str(full_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, CANVAS_SIZE
        ),
        True: cv2.VideoWriter(
            str(blinded_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, CANVAS_SIZE
        ),
    }
    if not all(writer.isOpened() for writer in writers.values()):
        raise RuntimeError("cannot create demo video")
    mapping = IncrementalAnatomicalLandmarkMap(map_config)
    for identity, coordinate in (
        actor.get("incremental_landmark_map") or {}
    ).get("actor_anchors_mm", {}).items():
        mapping.set_actor_anchor(identity, coordinate)
    frame_count = 0
    previous_frame = None
    recorded_snapshot = mapping.snapshot() if canonical else None
    try:
        for frame, image, label, robot_index in zip(
            frames, images, labels, robot_indices
        ):
            sequence = int(frame["frame_sequence"])
            for level, observation in observations.get(sequence, []):
                mapping.add_observations(level, (observation,))
            robot.set(cv2.CAP_PROP_POS_FRAMES, robot_index)
            ok, robot_frame = robot.read()
            if not ok:
                raise RuntimeError(f"cannot read robot frame {robot_index}")
            top = np.hstack(
                (
                    robot_panel(robot_frame, frame, previous_frame),
                    ultrasound_panel(
                        image,
                        frame,
                        [
                            value.__dict__
                            for _, value in observations.get(sequence, [])
                        ],
                    ),
                )
            )
            recorded_snapshot = map_events.get(sequence, recorded_snapshot)
            map_view = map_panel(
                mapping, frame, all_coordinates,
                snapshot=recorded_snapshot,
                return_text=return_texts.get(sequence, ""),
            )
            if args.historical_fallback:
                put_text_at(map_view, "HISTORICAL REPLAY", 24, 53, scale=0.56)
            target_report = target_reports[frame["target_name"]]
            for blinded, writer in writers.items():
                bottom = np.hstack(
                    (
                        map_view,
                        evaluator_panel(
                            image,
                            label,
                            frame,
                            target_report,
                            inspected.get(sequence, {}),
                            blinded=blinded,
                        ),
                    )
                )
                writer.write(np.vstack((top, bottom)))
            frame_count += 1
            previous_frame = frame
    finally:
        robot.release()
        for writer in writers.values():
            writer.release()
    final_map = sealed_map if canonical else mapping.snapshot()
    manifest = {
        "schema": "sonogym-s0010-incremental-landmark-map-demo-v1",
        "scientific_status": (
            "HISTORICAL_REPLAY_FALLBACK" if args.historical_fallback
            else "OPENED_DEVELOPMENT_ACTUAL_EXECUTION_REPLAY"
        ),
        "independent_confirmation": False,
        "session_id": actor["session_id"],
        "source_session": str(args.session_dir.resolve()),
        "actor_result_sha256": actor_sha,
        "evaluator_result_sha256": evaluator_sha,
        "source_protocol_sha256": sha256(args.source_protocol),
        "map_protocol_sha256": sha256(args.map_protocol),
        "frame_count": frame_count,
        "fps": args.fps,
        "final_map": final_map,
        "full_video": {"path": str(full_path), "sha256": sha256(full_path)},
        "blinded_video": {
            "path": str(blinded_path),
            "sha256": sha256(blinded_path),
        },
        "information_boundary": {
            "panels_1_to_3": "sealed actor and public records only",
            "panel_4_full": "privileged evaluator labels and results",
            "panel_4_blinded": "withheld",
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "frame_count": frame_count,
                "map_complete": final_map["complete"],
                "full_video": str(full_path),
                "blinded_video": str(blinded_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
