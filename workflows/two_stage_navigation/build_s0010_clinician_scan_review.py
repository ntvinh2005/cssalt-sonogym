#!/usr/bin/env python3
"""Build synchronized full and blinded review videos for clinician-scan feasibility."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from two_stage_navigation.stage1_interfaces import track_bright_interfaces


PANEL_WIDTH = 640
PANEL_HEIGHT = 360
CANVAS_SIZE = (2 * PANEL_WIDTH, 2 * PANEL_HEIGHT)
LABEL_COLORS = {
    1: (216, 180, 0),  # TP, BGR
    2: (133, 37, 247),  # SAP
    3: (102, 209, 255),  # other bone
}
PREDICTION_COLORS = {
    "TP": LABEL_COLORS[1],
    "SAP": LABEL_COLORS[2],
    "OTHER_BONE": LABEL_COLORS[3],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--motion-evaluation", type=Path, required=True)
    parser.add_argument("--perception-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--repeats-per-observation", type=int, default=2)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def put_text(
    panel: np.ndarray,
    value: str,
    y: int,
    *,
    color: tuple[int, int, int] = (235, 235, 235),
    scale: float = 0.48,
    thickness: int = 1,
) -> None:
    cv2.putText(
        panel,
        value,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def add_title(panel: np.ndarray, value: str) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (12, 12, 12), -1)
    put_text(panel, value, 22, scale=0.56)


def normalized_bgr(image: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(
        cv2.resize(normalized, (PANEL_WIDTH, PANEL_HEIGHT)), cv2.COLOR_GRAY2BGR
    )


def expected_transition(scan_kind: str) -> str:
    if scan_kind == "LONGITUDINAL_MEDIAL_LATERAL_REVERSIBLE_SWEEP":
        return "Expected: TP loss -> reverse return -> AP-region candidate"
    if scan_kind == "TRANSVERSE_CRANIAL_CAUDAL_SWEEP":
        return "Expected: shadow/deep-window change -> reverse recovery"
    return "Expected: independent alternative-view cross-check"


def short_state(scan_kind: str) -> str:
    return {
        "LONGITUDINAL_MEDIAL_LATERAL_REVERSIBLE_SWEEP": "LONGITUDINAL M-L REVERSIBLE",
        "TRANSVERSE_CRANIAL_CAUDAL_SWEEP": "TRANSVERSE C-C REVERSIBLE",
        "ALTERNATIVE_VIEW_CONFIRMATION": "ALTERNATIVE VIEW",
    }.get(scan_kind, scan_kind)


def pair_verdict(actor: dict[str, Any], evaluator: dict[str, Any]) -> str:
    predicted = bool(actor["tp_predicted"] and actor["sap_predicted"])
    supported = bool(evaluator["both_supported"])
    if predicted and supported:
        return "PAIR TRUE ACCEPT"
    if predicted and not supported:
        return "PAIR FALSE ACCEPT"
    if not predicted and supported:
        return "PAIR MISSED / ABSTAINED"
    return "PAIR CORRECT ABSTENTION"


def robot_panel(
    robot: cv2.VideoCapture,
    frame: dict[str, Any],
    evaluation: dict[str, Any],
) -> np.ndarray:
    robot.set(cv2.CAP_PROP_POS_FRAMES, int(frame["robot_video_frame_index"]))
    ok, image = robot.read()
    if not ok:
        image = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
        put_text(image, "ROBOT FRAME UNAVAILABLE", 180, color=(0, 0, 255), scale=0.8)
    panel = cv2.resize(image, (PANEL_WIDTH, PANEL_HEIGHT))
    add_title(panel, "1 | Executed robot motion (recorded camera)")
    delta = frame.get("executed_delta_relative_mm", [0.0, 0.0])
    put_text(
        panel,
        f"actual delta patient-index: dx={float(delta[0]):+.1f} mm dz={float(delta[1]):+.1f} mm",
        316,
    )
    put_text(
        panel,
        f"orientation={float(evaluation['realized_pose_index_angle']):.3f} rad  "
        f"contact={'YES' if frame['contact_ok'] else 'NO'}  acquisition=FRESH",
        341,
    )
    return panel


def ultrasound_panel(image: np.ndarray, actor: dict[str, Any]) -> np.ndarray:
    panel = normalized_bgr(image)
    for track in track_bright_interfaces(image):
        points = np.asarray(
            [
                [
                    int(round(column * PANEL_WIDTH / image.shape[1])),
                    int(round(row * PANEL_HEIGHT / image.shape[0])),
                ]
                for column, row in zip(track.columns, track.rows)
            ],
            dtype=np.int32,
        )
        if len(points) > 1:
            cv2.polylines(panel, [points], False, (0, 255, 0), 1, cv2.LINE_AA)
    for observation in actor["semantic_observations"]:
        x = int(round(float(observation["column"]) * PANEL_WIDTH / image.shape[1]))
        y = int(round(float(observation["row"]) * PANEL_HEIGHT / image.shape[0]))
        semantic_class = str(observation["semantic_class"])
        color = PREDICTION_COLORS[semantic_class]
        cv2.circle(panel, (x, y), 7, color, 2, cv2.LINE_AA)
        put_text(panel, semantic_class, max(45, y - 8), color=color, scale=0.40)
    add_title(panel, "2 | Clean ultrasound + actor-only observations")
    put_text(
        panel,
        f"TP={actor['tp_predicted_pixels']} px  SAP={actor['sap_predicted_pixels']} px  "
        f"other={actor['other_bone_predicted_pixels']} px",
        344,
    )
    return panel


def controller_panel(
    frame: dict[str, Any],
    actor: dict[str, Any],
    evaluation: dict[str, Any],
    previous: dict[str, Any] | None,
) -> np.ndarray:
    panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 18, dtype=np.uint8)
    scan_kind = str(evaluation["scan_kind"])
    add_title(panel, "3 | Controller/evidence state (offline feasibility replay)")
    lines = [
        f"Level: RIGHT {actor['vertebral_level']}   state: {short_state(scan_kind)}",
        "Plan source: EVALUATOR-DESIGNED WAYPOINTS (NOT AUTONOMOUS)",
        f"Frame: {frame['frame_sequence']:02d}  trajectory frame: {frame['trajectory_frame_sequence']:02d}",
        expected_transition(scan_kind),
        f"Observed: interfaces={evaluation['interface_track_count']}  "
        f"shadow={float(evaluation['posterior_shadow_fraction']):.3f}",
        f"deep-window={float(evaluation['deep_window_visibility']):.3f}  "
        f"TP hypothesis={'YES' if actor['tp_predicted'] else 'NO'}  "
        f"SAP hypothesis={'YES' if actor['sap_predicted'] else 'NO'}",
        f"fresh hash={'YES' if previous is None or previous['frame_sha256'] != frame['frame_sha256'] else 'NO'}  "
        f"producer seq={frame['producer_sequence']}  contact={'YES' if frame['contact_ok'] else 'NO'}",
        "Decision: candidate evidence only; Stage 2 acceptance gate remains closed",
    ]
    for index, line in enumerate(lines):
        color = (80, 210, 255) if index == 1 else (235, 235, 235)
        put_text(panel, line, 65 + index * 35, color=color, scale=0.45)
    return panel


def evaluator_panel(
    image: np.ndarray,
    labels: np.ndarray,
    actor: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    blinded: bool,
) -> np.ndarray:
    if blinded:
        panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 18, dtype=np.uint8)
        add_title(panel, "4 | Evaluator panel WITHHELD (blinded replay)")
        put_text(panel, "Privileged anatomy is intentionally absent.", 155, scale=0.65)
        put_text(panel, "Actor panels 1-3 are unchanged.", 195, scale=0.65)
        return panel
    panel = normalized_bgr(image)
    resized_labels = cv2.resize(
        labels.astype(np.uint8), (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_NEAREST
    )
    overlay = panel.copy()
    for value, color in LABEL_COLORS.items():
        overlay[resized_labels == value] = color
    panel = cv2.addWeighted(panel, 0.62, overlay, 0.38, 0)
    add_title(panel, "4 | PRIVILEGED evaluator-only weak interface labels")
    verdict = pair_verdict(actor, evaluation)
    verdict_color = (0, 220, 0) if verdict in {"PAIR TRUE ACCEPT", "PAIR CORRECT ABSTENTION"} else (0, 80, 255)
    put_text(
        panel,
        f"weak label: TP={'YES' if evaluation['tp_supported'] else 'NO'}  "
        f"SAP={'YES' if evaluation['sap_supported'] else 'NO'}",
        316,
    )
    put_text(panel, f"{verdict} | localization error: N/A for frame evidence", 342, color=verdict_color)
    return panel


def load_frame_evaluations(path: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, int], int]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    boolean_fields = {
        "tp_supported",
        "sap_supported",
        "both_supported",
        "image_valid",
        "contact_ok",
        "action_accepted",
        "action_position_clamped",
        "action_angle_clamped",
    }
    integer_fields = {
        "frame_sequence",
        "trajectory_frame_sequence",
        "producer_sequence",
        "interface_track_count",
    }
    float_fields = {
        "deep_window_visibility",
        "posterior_shadow_fraction",
        "cortical_interface_median_row_fraction",
    }
    for row in rows:
        for field in boolean_fields:
            row[field] = row[field] == "True"
        for field in integer_fields:
            row[field] = int(row[field])
        for field in float_fields:
            row[field] = float(row[field])
        realized = json.loads(row["realized_pose_index"])
        row["realized_pose_index_angle"] = float(realized[2])
    lookup = {(str(row["session_id"]), int(row["frame_sequence"])): i for i, row in enumerate(rows)}
    return rows, lookup


def write_video(
    path: Path,
    *,
    capture_root: Path,
    sessions: list[str],
    actor_lookup: dict[tuple[str, int], dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    evaluation_lookup: dict[tuple[str, int], int],
    weak_labels: np.ndarray,
    fps: float,
    repeats: int,
    blinded: bool,
) -> int:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, CANVAS_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video: {path}")
    output_frames = 0
    try:
        for session_id in sessions:
            session_root = capture_root / "sessions" / session_id
            frames = read_jsonl(session_root / "public/frames.jsonl")
            images = np.load(session_root / "public/ultrasound.npz")["ultrasound"]
            robot = cv2.VideoCapture(str(session_root / "public/robot_execution.mp4"))
            if not robot.isOpened():
                raise RuntimeError(f"cannot open robot video for {session_id}")
            previous = None
            try:
                for index, frame in enumerate(frames):
                    key = (session_id, int(frame["frame_sequence"]))
                    actor = actor_lookup[key]
                    evaluation_index = evaluation_lookup[key]
                    evaluation = evaluation_rows[evaluation_index]
                    top = np.hstack(
                        (
                            robot_panel(robot, frame, evaluation),
                            ultrasound_panel(images[index], actor),
                        )
                    )
                    bottom = np.hstack(
                        (
                            controller_panel(frame, actor, evaluation, previous),
                            evaluator_panel(
                                images[index],
                                weak_labels[evaluation_index],
                                actor,
                                evaluation,
                                blinded=blinded,
                            ),
                        )
                    )
                    canvas = np.vstack((top, bottom))
                    for _ in range(repeats):
                        writer.write(canvas)
                        output_frames += 1
                    previous = frame
            finally:
                robot.release()
    finally:
        writer.release()
    return output_frames


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    motion_result = json.loads((args.motion_evaluation / "result.json").read_text())
    perception_result = json.loads(
        (args.perception_evaluation / "evaluator/result.json").read_text()
    )
    actor_frames = read_jsonl(args.perception_evaluation / "actor/frames.jsonl")
    actor_lookup = {
        (str(row["session_id"]), int(row["frame_sequence"])): row for row in actor_frames
    }
    evaluation_rows, evaluation_lookup = load_frame_evaluations(
        args.motion_evaluation / "frame_results.csv"
    )
    weak_labels = np.load(args.motion_evaluation / "weak_interface_labels.npz")["labels"]
    if len(evaluation_rows) != len(weak_labels):
        raise ValueError("frame-result and weak-label counts differ")
    sessions = sorted({str(row["session_id"]) for row in evaluation_rows})
    full_video = args.output_dir / "s0010_clinician_scan_actual_execution_full.mp4"
    blinded_video = args.output_dir / "s0010_clinician_scan_actual_execution_blinded.mp4"
    full_count = write_video(
        full_video,
        capture_root=args.capture_root,
        sessions=sessions,
        actor_lookup=actor_lookup,
        evaluation_rows=evaluation_rows,
        evaluation_lookup=evaluation_lookup,
        weak_labels=weak_labels,
        fps=args.fps,
        repeats=args.repeats_per_observation,
        blinded=False,
    )
    blinded_count = write_video(
        blinded_video,
        capture_root=args.capture_root,
        sessions=sessions,
        actor_lookup=actor_lookup,
        evaluation_rows=evaluation_rows,
        evaluation_lookup=evaluation_lookup,
        weak_labels=weak_labels,
        fps=args.fps,
        repeats=args.repeats_per_observation,
        blinded=True,
    )
    manifest = {
        "schema": "s0010-clinician-scan-actual-execution-review-v1",
        "mode": "offline_replay_of_evaluator_designed_feasibility_capture",
        "not_autonomous": True,
        "capture_root": str(args.capture_root.resolve()),
        "motion_evaluation": str(args.motion_evaluation.resolve()),
        "perception_evaluation": str(args.perception_evaluation.resolve()),
        "motion_feasibility_passed": motion_result["motion_feasibility_passed"],
        "independent_confirmation": False,
        "perception_result": perception_result,
        "sessions": sessions,
        "observation_count": len(evaluation_rows),
        "output_frame_count": full_count,
        "fps": args.fps,
        "full_video": {
            "path": str(full_video.resolve()),
            "sha256": sha256(full_video),
        },
        "blinded_video": {
            "path": str(blinded_video.resolve()),
            "sha256": sha256(blinded_video),
        },
        "information_boundary": {
            "panels_1_2_3": "actor-visible execution, ultrasound, and predictions",
            "panel_4_full": "privileged evaluator-only weak interface labels",
            "panel_4_blinded": "withheld",
        },
    }
    if blinded_count != full_count:
        raise RuntimeError("full and blinded videos have different frame counts")
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
