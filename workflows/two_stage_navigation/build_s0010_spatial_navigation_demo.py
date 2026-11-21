#!/usr/bin/env python3
"""Build a four-panel offline replay of verified historical robot navigation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from two_stage_navigation.stage1_interfaces import track_bright_interfaces
from two_stage_navigation.ultrasound_spatial_map import SpatialMapConfig, cortical_map_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def text(panel: np.ndarray, value: str, y: int, *, color=(240, 240, 240), scale=0.52) -> None:
    cv2.putText(panel, value, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def title(panel: np.ndarray, value: str) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 31), (12, 12, 12), -1)
    text(panel, value, 21, scale=0.58)


def image_panel(image: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    panel = cv2.cvtColor(cv2.resize(normalized, (640, 360)), cv2.COLOR_GRAY2BGR)
    for track in track_bright_interfaces(image):
        coordinates = np.asarray(
            [
                [int(round(column * 640 / 150)), int(round(row * 360 / 200))]
                for column, row in zip(track.columns, track.rows)
            ],
            dtype=np.int32,
        )
        cv2.polylines(panel, [coordinates], False, (0, 255, 0), 2, cv2.LINE_AA)
    title(panel, "1 | Clean ultrasound + actor cortical candidates")
    text(panel, f"frame={frame['frame_sequence']} phase={frame['phase'].split(':')[-1]}", 348)
    return panel


def map_panel(
    current_points: list[Any], accumulated: list[Any], bounds: tuple[float, float, float, float]
) -> np.ndarray:
    panel = np.zeros((360, 640, 3), dtype=np.uint8)
    xmin, xmax, zmin, zmax = bounds

    def xy(point: Any) -> tuple[int, int]:
        x, z, _ = point.coordinate_mm
        px = int(35 + (x - xmin) / max(xmax - xmin, 1e-6) * 570)
        py = int(330 - (z - zmin) / max(zmax - zmin, 1e-6) * 285)
        return px, py

    for point in accumulated:
        depth = np.clip(point.coordinate_mm[2] / 100.0, 0.0, 1.0)
        cv2.circle(panel, xy(point), 2, (int(180 * depth), 120, 220), -1)
    for point in current_points:
        cv2.circle(panel, xy(point), 3, (0, 255, 255), -1)
    title(panel, "2 | Accumulated actor 2.5D spatial cortical map")
    text(panel, f"prior points={len(accumulated)} current={len(current_points)}", 348)
    return panel


def label_panel(raw: np.ndarray, frame: dict[str, Any], state: str) -> np.ndarray:
    bone = np.isin(raw, np.arange(25, 44)).astype(np.uint8) * 255
    colored = cv2.applyColorMap(cv2.resize(bone, (640, 360)), cv2.COLORMAP_TURBO)
    row = int(round(float(frame["proposal_row"]) * 360 / 200))
    column = int(round(float(frame["proposal_column"]) * 640 / 150))
    cv2.drawMarker(colored, (column, row), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
    title(colored, "4 | PRIVILEGED evaluator: source-bone slice + proposal")
    text(colored, f"evaluator state={state}", 348, color=(255, 255, 255))
    return colored


def known_states(report: dict[str, Any]) -> dict[int, str]:
    result: dict[int, str] = {}
    for target in report["target_reports"]:
        if target["selected"] is not None:
            result[int(target["selected"]["frame_sequence"])] = str(target["selected"]["proxy_state"])
        for row in target["fresh_confirmations"]:
            result[int(row["frame_sequence"])] = str(row["proxy_state"])
    return result


def target_outcome(report: dict[str, Any], target_name: str) -> bool:
    row = next(item for item in report["target_reports"] if item["target_name"] == target_name)
    return bool(row["proxy_target_view_success"])


def main() -> int:
    args = parse_args()
    result = json.loads((args.analysis / "result.json").read_text())
    protocol = yaml.safe_load(args.protocol.read_text())
    hypothesis_name = result["selected_hypothesis"] or "motion_compensated_2p5d"
    hypothesis = next(row for row in protocol["hypotheses"] if row["name"] == hypothesis_name)
    shared = protocol["shared_map"]
    config = SpatialMapConfig(
        pixel_resolution_mm=float(shared["pixel_resolution_mm"]),
        label_resolution_mm=float(shared["label_resolution_mm"]),
        sample_stride_pixels=int(shared["sample_stride_pixels"]),
        association_radius_mm=float(hypothesis["association_radius_mm"]),
        minimum_view_translation_mm=float(shared["minimum_view_translation_mm"]),
        minimum_view_angle_deg=float(shared["minimum_view_angle_deg"]),
        include_depth=bool(hypothesis["include_depth"]),
    )

    reports = {}
    for session in sorted((args.run_root / "stage2").glob("mbbpoc_v3_*")):
        reports[session.name] = json.loads((session / "results/evaluation_report.json").read_text())
    success_session = next(
        session for session, report in reports.items()
        if target_outcome(report, "L5_RIGHT_TP_SURFACE_CORE")
    )
    failure = next(
        (
            (session, target)
            for session, report in reports.items()
            for target in ("L4_RIGHT_TP_SURFACE_CORE", "L5_RIGHT_TP_SURFACE_CORE")
            if not target_outcome(report, target)
        ),
        None,
    )
    scenarios = [(success_session, "L5_RIGHT_TP_SURFACE_CORE", "verified historical success")]
    if failure is not None:
        scenarios.append((failure[0], failure[1], "representative historical failure"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError("cannot create MP4")
    manifest_scenarios = []
    try:
        for session_id, target_name, label in scenarios:
            root = args.run_root / "stage2" / session_id
            frames = read_jsonl(root / "public/frames.jsonl")
            images = np.load(root / "public/ultrasound.npz")["ultrasound"]
            raw = np.load(root / "privileged/raw_labels.npz")["raw_labels"]
            report = reports[session_id]
            states = known_states(report)
            indices = [index for index, frame in enumerate(frames) if frame["target_name"] == target_name]
            all_points = []
            by_frame = {}
            for index in indices:
                frame = frames[index]
                points = list(
                    cortical_map_points(
                        images[index],
                        executed_pose_index=frame["executed_relative_pose_index"],
                        frame_sequence=int(frame["frame_sequence"]),
                        frame_sha256=str(frame["frame_sha256"]),
                        config=config,
                    )
                )
                by_frame[index] = points
                all_points.extend(points)
            xs = [point.coordinate_mm[0] for point in all_points]
            zs = [point.coordinate_mm[1] for point in all_points]
            bounds = (min(xs) - 2, max(xs) + 2, min(zs) - 2, max(zs) + 2)
            robot = cv2.VideoCapture(str(root / "video/robot_probe_patient.mp4"))
            robot_count = int(robot.get(cv2.CAP_PROP_FRAME_COUNT))
            accumulated = []
            for index in indices:
                frame = frames[index]
                current = by_frame[index]
                robot.set(cv2.CAP_PROP_POS_FRAMES, int(frame["frame_sequence"] / max(len(frames) - 1, 1) * max(robot_count - 1, 0)))
                ok, robot_frame = robot.read()
                if not ok:
                    robot_frame = np.zeros((360, 640, 3), dtype=np.uint8)
                robot_panel = cv2.resize(robot_frame, (640, 360))
                title(robot_panel, "3 | Recorded robot/probe motion (historical closed loop)")
                text(robot_panel, f"{session_id} | {target_name[:2]} | {label}", 348)
                top = np.hstack((image_panel(images[index], frame), map_panel(current, accumulated, bounds)))
                bottom = np.hstack((robot_panel, label_panel(raw[index], frame, states.get(int(frame["frame_sequence"]), "NOT_INSPECTED"))))
                canvas = np.vstack((top, bottom))
                for _ in range(2):
                    writer.write(canvas)
                accumulated.extend(current)
            robot.release()
            manifest_scenarios.append(
                {
                    "session_id": session_id,
                    "target": target_name,
                    "label": label,
                    "frame_count": len(indices),
                    "proxy_target_view_success": target_outcome(report, target_name),
                }
            )
    finally:
        writer.release()
    manifest = {
        "schema": "s0010-spatial-navigation-demo-v1",
        "video": str(args.output.resolve()),
        "mode": "offline_replay_of_historical_closed_loop_execution",
        "source_run": str(args.run_root.resolve()),
        "mapping_hypothesis": hypothesis_name,
        "mapping_phase_a_verdict": result["phase_a_verdict"],
        "scenarios": manifest_scenarios,
        "information_boundary": {
            "panels_1_2_3": "actor-visible or recorded physical execution",
            "panel_4": "privileged evaluator only",
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
