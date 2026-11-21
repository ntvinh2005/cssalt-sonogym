#!/usr/bin/env python3
"""Render the L5 spatial localizer from recorded ultrasound and executed motion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from evaluate_stage2_tp_active import CANDIDATES, locate_geometry
from two_stage_navigation.stage1_interfaces import track_bright_interfaces
from two_stage_navigation.ultrasound_spatial_map import (
    SpatialLocalizerConfig,
    SpatialMapConfig,
    SpatialViewCandidate,
    cortical_map_points,
    map_image_pixel,
    rank_spatial_target_views,
)


TARGET = "L5_RIGHT_TP_SURFACE_CORE"
PANEL_HEIGHT = 360
PANEL_WIDTH = 640


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def put_text(
    panel: np.ndarray,
    value: str,
    y: int,
    *,
    color: tuple[int, int, int] = (235, 235, 235),
    scale: float = 0.5,
) -> None:
    cv2.putText(
        panel,
        value,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def title(panel: np.ndarray, value: str) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 31), (12, 12, 12), -1)
    put_text(panel, value, 21, scale=0.58)


def map_config(protocol: dict[str, Any]) -> SpatialMapConfig:
    row = protocol["actor_geometry"]
    return SpatialMapConfig(
        pixel_resolution_mm=float(row["pixel_resolution_mm"]),
        label_resolution_mm=float(row["label_resolution_mm"]),
        sample_stride_pixels=int(row["sample_stride_pixels"]),
        association_radius_mm=float(row["association_radius_mm"]),
        minimum_view_translation_mm=float(row["minimum_view_translation_mm"]),
        minimum_view_angle_deg=float(row["minimum_view_angle_deg"]),
        include_depth=True,
    )


def localizer_config(protocol: dict[str, Any], hypothesis: str) -> SpatialLocalizerConfig:
    shared = protocol["localizer_shared"]
    row = next(item for item in protocol["hypotheses"] if item["name"] == hypothesis)
    return SpatialLocalizerConfig(
        hypothesis=hypothesis,
        support_radius_mm=float(shared["support_radius_mm"]),
        minimum_supporting_frames=int(shared["minimum_supporting_frames"]),
        maximum_support_distance_mm=float(shared["maximum_support_distance_mm"]),
        minimum_target_probability=float(shared["minimum_target_probability"]),
        target_probability_weight=float(shared["target_probability_weight"]),
        support_weight=float(shared["support_weight"]),
        support_saturation_count=int(shared["support_saturation_count"]),
        landmark_prior_scale_mm=float(row["landmark_prior_scale_mm"]),
        minimum_score=float(shared["minimum_score"]),
        minimum_margin=float(shared["minimum_margin"]),
        top_k=int(shared["top_k"]),
    )


def search_frame(frame: dict[str, Any], protocol: dict[str, Any]) -> bool:
    phase = str(frame["phase"]).split(":")[-1]
    return frame["target_name"] == TARGET and phase in protocol["observation_budget"][
        "include_phases"
    ]


def image_panel(image: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    panel = cv2.cvtColor(
        cv2.resize(normalized, (PANEL_WIDTH, PANEL_HEIGHT)), cv2.COLOR_GRAY2BGR
    )
    for track in track_bright_interfaces(image):
        coordinates = np.asarray(
            [
                [
                    int(round(column * PANEL_WIDTH / 150)),
                    int(round(row * PANEL_HEIGHT / 200)),
                ]
                for column, row in zip(track.columns, track.rows)
            ],
            dtype=np.int32,
        )
        cv2.polylines(panel, [coordinates], False, (0, 255, 0), 2, cv2.LINE_AA)
    if frame["proposal_row"] is not None:
        point = (
            int(round(float(frame["proposal_column"]) * PANEL_WIDTH / 150)),
            int(round(float(frame["proposal_row"]) * PANEL_HEIGHT / 200)),
        )
        cv2.drawMarker(panel, point, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
    title(panel, "1 | Clean ultrasound + actor cortical candidates")
    put_text(
        panel,
        f"frame={frame['frame_sequence']} phase={str(frame['phase']).split(':')[-1]}",
        348,
    )
    return panel


def actor_map_panel(
    accumulated: list[Any],
    current: list[Any],
    decision: Any,
    poses: dict[int, np.ndarray],
    candidates: dict[int, SpatialViewCandidate],
    config: SpatialMapConfig,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    xmin, xmax, zmin, zmax = bounds

    def xy(values: tuple[float, float, float] | np.ndarray) -> tuple[int, int]:
        x, z = float(values[0]), float(values[1])
        px = int(35 + (x - xmin) / max(xmax - xmin, 1e-6) * 570)
        py = int(330 - (z - zmin) / max(zmax - zmin, 1e-6) * 285)
        return px, py

    for point in accumulated:
        depth = np.clip(point.coordinate_mm[2] / 100.0, 0.0, 1.0)
        cv2.circle(panel, xy(point.coordinate_mm), 2, (int(180 * depth), 100, 210), -1)
    for point in current:
        cv2.circle(panel, xy(point.coordinate_mm), 3, (0, 255, 255), -1)
    if decision.ranking:
        top = decision.ranking[0]
        candidate = candidates[top.frame_sequence]
        coordinate = map_image_pixel(
            row=candidate.proposal_row,
            column=candidate.proposal_column,
            image_shape=(200, 150),
            executed_pose_index=poses[top.frame_sequence],
            config=config,
        )
        cv2.drawMarker(panel, xy(coordinate), (0, 180, 255), cv2.MARKER_DIAMOND, 18, 2)
    title(panel, "2 | Actor approximate 2.5D cortical map")
    put_text(panel, f"accumulated={len(accumulated)} current={len(current)}", 348)
    return panel


def ranking_panel(decision: Any, observed_frames: int) -> np.ndarray:
    panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 24, dtype=np.uint8)
    title(panel, "3 | Target-view ranking from spatial evidence")
    ranking = list(decision.ranking[:5])
    for offset, row in enumerate(ranking):
        y = 66 + offset * 49
        length = int(np.clip(row.score, 0.0, 1.0) * 360)
        color = (60, 210, 90) if row.eligible else (110, 110, 110)
        cv2.rectangle(panel, (190, y - 18), (190 + length, y + 5), color, -1)
        put_text(panel, f"#{offset + 1} frame {row.frame_sequence}", y, scale=0.48)
        put_text(
            panel,
            f"score={row.score:.3f} support={len(row.supporting_frames)}",
            y + 20,
            color=(205, 205, 205),
            scale=0.43,
        )
    selected = "none" if decision.selected is None else str(decision.selected.frame_sequence)
    put_text(
        panel,
        f"decision={decision.status} selected={selected} observations={observed_frames}",
        342,
        color=(0, 230, 255),
    )
    return panel


def evaluator_panel(
    target_xz: np.ndarray,
    proposal_xyz: np.ndarray | None,
    evaluator: dict[str, str] | None,
) -> np.ndarray:
    panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 245, dtype=np.uint8)
    xmin, zmin = np.min(target_xz, axis=0) - 12.0
    xmax, zmax = np.max(target_xz, axis=0) + 12.0

    def xy(values: np.ndarray) -> tuple[int, int]:
        px = int(45 + (values[0] - xmin) / max(xmax - xmin, 1e-6) * 540)
        py = int(315 - (values[1] - zmin) / max(zmax - zmin, 1e-6) * 255)
        return px, py

    for point in target_xz:
        cv2.circle(panel, xy(point), 2, (220, 125, 0), -1)
    if proposal_xyz is not None:
        cv2.drawMarker(panel, xy(proposal_xyz[[0, 2]]), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    title(panel, "4 | PRIVILEGED evaluator: L5 TP core and proposal")
    if evaluator is None:
        state, distance = "NO_PROPOSAL", "n/a"
    else:
        state = evaluator["proxy_state"]
        distance = f"{float(evaluator['target_distance_mm']):.2f} mm"
    put_text(panel, f"proxy_state={state} center-to-core={distance}", 348, color=(20, 20, 20))
    return panel


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = json.loads((args.analysis / "result.json").read_text())
    protocol = yaml.safe_load(args.protocol.read_text())
    hypothesis = result["selected_hypothesis"] or protocol["hypotheses"][0]["name"]
    config = map_config(protocol)
    decision_config = localizer_config(protocol, hypothesis)
    rankings = read_csv(args.analysis / "frame_rankings.csv")
    evaluator_by_session = {
        (row["session_id"], int(row["frame_sequence"])): row
        for row in rankings
        if row["hypothesis"] == hypothesis
    }
    session_results = [
        row
        for row in read_csv(args.analysis / "session_results.csv")
        if row["hypothesis"] == hypothesis
    ]
    final = [row for row in session_results if row["split"] == "final_comparison"]
    pool = final or [row for row in session_results if row["split"] == "selection"]
    correct = next((row for row in pool if row["selected_state"] == "PROXY_POSITIVE"), None)
    failure = next((row for row in pool if row["selected_state"] != "PROXY_POSITIVE"), None)
    chosen_rows = [row for row in (correct, failure) if row is not None]
    if len(chosen_rows) < 2:
        chosen_rows.extend(row for row in pool if row not in chosen_rows)
    chosen_rows = chosen_rows[:2]

    source_protocol = yaml.safe_load(
        (args.run_root / "context/s0010_mbb_single_anatomy_poc_protocol_v3.yaml").read_text()
    )
    geometry_root = Path(source_protocol["evaluator"]["geometry_root"])
    geometry_path = locate_geometry(geometry_root, CANDIDATES[TARGET])
    geometry = np.load(geometry_path)
    target = (
        geometry["vertices_mesh_index"][geometry["proposed_revised_vertex_mask"].astype(bool)]
        * float(protocol["actor_geometry"]["label_resolution_mm"])
    )
    target_xz = target[:, [0, 2]]
    output = args.output_dir / "s0010_l5_spatial_localizer_demo.mp4"
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (1280, 720)
    )
    if not writer.isOpened():
        raise RuntimeError("cannot create MP4")
    scenarios = []
    try:
        for session_row in chosen_rows:
            session_id = session_row["session_id"]
            root = args.run_root / "stage2" / session_id
            frames = read_jsonl(root / "public/frames.jsonl")
            images = np.load(root / "public/ultrasound.npz")["ultrasound"]
            native = np.load(root / "privileged/pixel_to_volume_coordinates.npz")[
                "human_img_coords"
            ]
            actor_result = json.loads((root / "results/actor_result.json").read_text())
            prior = np.asarray(
                next(
                    row for row in actor_result["targets"] if row["target_name"] == TARGET
                )["stage1_handoff_pose_index"],
                dtype=np.float64,
            )
            indices = [index for index, frame in enumerate(frames) if search_frame(frame, protocol)]
            by_frame = {}
            poses = {}
            candidates = {}
            all_points = []
            for index in indices:
                frame = frames[index]
                sequence = int(frame["frame_sequence"])
                poses[sequence] = np.asarray(frame["executed_relative_pose_index"], dtype=np.float64)
                points = list(
                    cortical_map_points(
                        images[index],
                        executed_pose_index=frame["executed_relative_pose_index"],
                        frame_sequence=sequence,
                        frame_sha256=str(frame["frame_sha256"]),
                        config=config,
                    )
                )
                by_frame[sequence] = points
                all_points.extend(points)
                if frame["proposal_row"] is not None:
                    candidates[sequence] = SpatialViewCandidate(
                        sequence,
                        str(frame["frame_sha256"]),
                        float(frame["target_probability"]),
                        float(frame["proposal_row"]),
                        float(frame["proposal_column"]),
                    )
            xs = [point.coordinate_mm[0] for point in all_points]
            zs = [point.coordinate_mm[1] for point in all_points]
            bounds = (min(xs) - 2.0, max(xs) + 2.0, min(zs) - 2.0, max(zs) + 2.0)
            accumulated = []
            observed_candidates = []
            for observed_count, index in enumerate(indices, start=1):
                frame = frames[index]
                sequence = int(frame["frame_sequence"])
                current = by_frame[sequence]
                if sequence in candidates:
                    observed_candidates.append(candidates[sequence])
                decision = rank_spatial_target_views(
                    [*accumulated, *current],
                    observed_candidates,
                    poses_by_frame=poses,
                    stage1_prior_pose_index=prior,
                    map_config=config,
                    localizer_config=decision_config,
                )
                evaluator = evaluator_by_session.get((session_id, sequence))
                proposal_xyz = None
                if frame["proposal_row"] is not None:
                    row = int(np.clip(round(float(frame["proposal_row"])), 0, 199))
                    column = int(np.clip(round(float(frame["proposal_column"])), 0, 149))
                    proposal_xyz = native[index, column * 200 + row].astype(np.float64) * 1.5
                top = np.hstack(
                    (
                        image_panel(images[index], frame),
                        actor_map_panel(
                            accumulated,
                            current,
                            decision,
                            poses,
                            candidates,
                            config,
                            bounds,
                        ),
                    )
                )
                bottom = np.hstack(
                    (
                        ranking_panel(decision, observed_count),
                        evaluator_panel(target_xz, proposal_xyz, evaluator),
                    )
                )
                canvas = np.vstack((top, bottom))
                for _ in range(2):
                    writer.write(canvas)
                accumulated.extend(current)
            scenarios.append(
                {
                    "session_id": session_id,
                    "split": session_row["split"],
                    "final_decision": session_row["decision"],
                    "selected_state": session_row["selected_state"],
                    "selected_frame_sequence": session_row["selected_frame_sequence"],
                    "search_frame_count": len(indices),
                }
            )
    finally:
        writer.release()
    manifest = {
        "schema": "s0010-l5-spatial-localizer-demo-v1",
        "mode": "offline_replay_of_historically_opened_development_sessions",
        "hypothesis": hypothesis,
        "source_run": str(args.run_root.resolve()),
        "video": str(output.resolve()),
        "scenarios": scenarios,
        "information_boundary": {
            "panels_1_to_3": "actor-visible clean ultrasound, motion, Stage 1 prior, and map",
            "panel_4": "privileged evaluator-only TP geometry and native coordinates",
        },
        "robot_video": "not included because no acquisition-metadata synchronization exists",
        "new_closed_loop_execution": False,
    }
    (args.output_dir / "demo_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
