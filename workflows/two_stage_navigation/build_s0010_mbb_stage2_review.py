#!/usr/bin/env python3
"""Render full and blinded four-panel replays of an active Stage 2 run."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import nibabel as nib
import numpy as np
import yaml

from workflows.two_stage_navigation.build_s0010_clinician_scan_review import (
    CANVAS_SIZE,
    LABEL_COLORS,
    PANEL_HEIGHT,
    PANEL_WIDTH,
    add_title,
    normalized_bgr,
    put_text,
)
from workflows.two_stage_navigation.evaluate_s0010_clinician_scan import (
    image_transition_measurements,
)
from workflows.two_stage_navigation.evaluate_stage2_tp_active import bone_source_ids
from two_stage_navigation.sonoanatomy_labels import (
    OTHER_BONE_INTERFACE,
    SAP_INTERFACE,
    TP_INTERFACE,
    right_level_masks,
    visible_interface_labels,
)
from two_stage_navigation.stage1_interfaces import track_bright_interfaces


PREDICTION_COLORS = {
    "TP": LABEL_COLORS[int(TP_INTERFACE)],
    "SAP": LABEL_COLORS[int(SAP_INTERFACE)],
    "OTHER_BONE": LABEL_COLORS[int(OTHER_BONE_INTERFACE)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--repeats-per-observation", type=int, default=2)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_seal(path: Path) -> str:
    seal = path.with_suffix(path.suffix + ".sha256")
    expected = seal.read_text(encoding="utf-8").split()[0]
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"sealed input changed: {path}")
    return observed


def map_acquisitions_to_robot_frames(
    frames: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    *,
    action_stride: int,
    robot_frame_count: int,
    pose_tolerance: float = 1e-6,
) -> list[int]:
    """Map each acquisition to the robot frame captured after its final action."""

    if action_stride <= 0 or robot_frame_count <= 0:
        raise ValueError("video stride and frame count must be positive")
    pointer = 0
    result: list[int] = []
    for frame in frames:
        phase = str(frame["phase"])
        action_count = int(frame["action_step_count"])
        if action_count < 0:
            raise ValueError("action count must not be negative")
        while pointer < len(actions) and str(actions[pointer]["phase"]) != phase:
            pointer += 1
        stop = pointer + action_count
        if stop > len(actions):
            raise ValueError("acquisition action count exceeds action history")
        if any(str(row["phase"]) != phase for row in actions[pointer:stop]):
            raise ValueError("acquisition phase does not align with action history")
        if action_count:
            realized = np.asarray(actions[stop - 1]["after_realized_pose"], dtype=np.float64)
            recorded = np.asarray(frame["executed_relative_pose_index"], dtype=np.float64)
            if realized.shape != recorded.shape or not np.allclose(
                realized, recorded, atol=pose_tolerance, rtol=0.0
            ):
                raise ValueError("action history does not reproduce acquisition pose")
        pointer = stop
        result.append(min(pointer // action_stride, robot_frame_count - 1))
    if pointer != len(actions):
        raise ValueError("action history contains unmapped trailing actions")
    return result


def semantic_observations_by_frame(actor: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for target in actor["targets"]:
        sonoanatomy = target.get("sonoanatomy") or {}
        for trajectory in sonoanatomy.get("trajectory_decisions", []):
            for observation in trajectory.get("semantic_observations", []):
                result[int(observation["frame_sequence"])].append(observation)
    return dict(result)


def expected_change(scan_state: str) -> str:
    return {
        "ACQUIRE_REFERENCE_TP_REGION": "Expected: repeated TP-region cortical evidence",
        "LONGITUDINAL_LATERAL_SURVEY": "Expected: lateral TP loss and deeper-window change",
        "REVERSE_MEDIAL_SURVEY": "Expected: TP return then AP-region morphology",
        "TRANSVERSE_TP_HANDOFF_FALLBACK": "Expected: transverse TP evidence",
        "TRANSVERSE_CRANIAL_CAUDAL_SWEEP": "Expected: reversible shadow/deep-window change",
        "ACTIVE_RESCAN": "Expected: distinguish TP/SAP from competing bone",
        "ACTIVE_COMPETING_PAIR_RESCAN": "Expected: distinguish competing TP/SAP hypotheses",
        "FRESH_ULTRASOUND_CONFIRMATION": "Expected: fresh repeatable TP/SAP evidence",
    }.get(scan_state, "Expected: actor-directed target-view refinement")


def robot_panel(
    robot_frame: np.ndarray,
    frame: dict[str, Any],
    previous: dict[str, Any] | None,
) -> np.ndarray:
    panel = cv2.resize(robot_frame, (PANEL_WIDTH, PANEL_HEIGHT))
    add_title(panel, "1 | Executed robot/probe trajectory")
    pose = np.asarray(frame["executed_relative_pose_index"], dtype=np.float64)
    delta = np.zeros(3) if previous is None else pose - np.asarray(
        previous["executed_relative_pose_index"], dtype=np.float64
    )
    put_text(
        panel,
        f"actual delta: dx={delta[0]:+.2f}  dz={delta[1]:+.2f} index  "
        f"dangle={math.degrees(delta[2]):+.2f} deg",
        316,
    )
    put_text(
        panel,
        f"orientation={pose[2]:.3f} rad  contact={'YES' if frame['contact_ok'] else 'NO'}  "
        f"fresh={'YES' if frame['producer_advanced'] else 'NO'}",
        341,
    )
    return panel


def ultrasound_panel(
    image: np.ndarray,
    frame: dict[str, Any],
    observations: list[dict[str, Any]],
) -> np.ndarray:
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
    for observation in observations:
        semantic_class = str(observation["semantic_class"])
        color = PREDICTION_COLORS[semantic_class]
        point = (
            int(round(float(observation["column"]) * PANEL_WIDTH / image.shape[1])),
            int(round(float(observation["row"]) * PANEL_HEIGHT / image.shape[0])),
        )
        cv2.circle(panel, point, 7, color, 2, cv2.LINE_AA)
    add_title(panel, "2 | Clean ultrasound + actor-only observations")
    put_text(
        panel,
        f"TP={int(frame.get('tp_predicted_pixels', 0))} px  "
        f"SAP={int(frame.get('sap_predicted_pixels', 0))} px  "
        f"other={int(frame.get('other_bone_predicted_pixels', 0))} px",
        341,
    )
    return panel


def controller_panel(
    image: np.ndarray,
    frame: dict[str, Any],
    previous_image: np.ndarray | None,
    target_actor: dict[str, Any],
) -> np.ndarray:
    panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 18, dtype=np.uint8)
    add_title(panel, "3 | Actor controller and transition evidence")
    state = str(frame.get("scan_state", frame["phase"].split(":")[-1]))
    measurements = image_transition_measurements(image)
    previous_measurements = (
        None if previous_image is None else image_transition_measurements(previous_image)
    )
    shadow = measurements["posterior_shadow_fraction"]
    shadow_text = "N/A" if shadow is None else f"{float(shadow):.3f}"
    deep_delta = 0.0
    if previous_measurements is not None:
        deep_delta = float(measurements["deep_window_visibility"]) - float(
            previous_measurements["deep_window_visibility"]
        )
    lines = [
        f"Level: RIGHT {str(frame['target_name'])[:2]}   state: {state}",
        f"Last action: {frame['phase']} ({int(frame['action_step_count'])} steps)",
        expected_change(state),
        f"Observed: interfaces={measurements['interface_track_count']}  "
        f"shadow={shadow_text}  deep delta={deep_delta:+.3f}",
        f"Actor TP={'YES' if frame.get('tp_predicted', False) else 'NO'}  "
        f"SAP={'YES' if frame.get('sap_predicted', False) else 'NO'}  "
        f"image valid={'YES' if frame['image_valid'] else 'NO'}",
        f"Target decision: {target_actor['decision']} | {target_actor['decision_reason']}",
        f"frame={frame['frame_sequence']} producer={frame['producer_sequence']} "
        f"hash={str(frame['frame_sha256'])[:10]}...",
    ]
    for index, line in enumerate(lines):
        put_text(panel, line, 66 + index * 39, scale=0.43)
    return panel


def render_actor_panels(
    robot_frame: np.ndarray,
    image: np.ndarray,
    frame: dict[str, Any],
    observations: list[dict[str, Any]],
    target_actor: dict[str, Any],
    previous_frame: dict[str, Any] | None,
    previous_image: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render panels that consume only recorded actor/public information."""

    return (
        robot_panel(robot_frame, frame, previous_frame),
        ultrasound_panel(image, frame, observations),
        controller_panel(image, frame, previous_image, target_actor),
    )


def evaluator_lookup(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for target in report["target_reports"]:
        selected = target.get("selected")
        if selected is not None:
            result[int(selected["frame_sequence"])] = selected
        for row in target.get("fresh_confirmations", []):
            result[int(row["frame_sequence"])] = row
        for row in target.get("semantic_fresh_confirmations", []):
            result.setdefault(int(row["frame_sequence"]), {}).update(row)
    return result


def evaluator_panel(
    image: np.ndarray,
    labels: np.ndarray,
    frame: dict[str, Any],
    target_report: dict[str, Any],
    inspected: dict[str, Any],
    *,
    blinded: bool,
) -> np.ndarray:
    if blinded:
        panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 18, dtype=np.uint8)
        add_title(panel, "4 | Evaluator panel WITHHELD")
        put_text(panel, "Privileged anatomy is intentionally absent.", 160, scale=0.64)
        put_text(panel, "Panels 1-3 are unchanged.", 200, scale=0.64)
        return panel
    panel = normalized_bgr(image)
    resized = cv2.resize(
        labels.astype(np.uint8),
        (PANEL_WIDTH, PANEL_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    )
    overlay = panel.copy()
    for value, color in LABEL_COLORS.items():
        overlay[resized == int(value)] = color
    panel = cv2.addWeighted(panel, 0.62, overlay, 0.38, 0.0)
    add_title(panel, "4 | PRIVILEGED evaluator-only interface labels")
    tp = bool(np.any(labels == int(TP_INTERFACE)))
    sap = bool(np.any(labels == int(SAP_INTERFACE)))
    pair = target_report.get("semantic_pair_identity") or {}
    error = inspected.get("interface_point_to_core_error_mm")
    error_text = "N/A" if error is None else f"{float(error):.2f} mm"
    put_text(panel, f"weak label: TP={'YES' if tp else 'NO'}  SAP={'YES' if sap else 'NO'}", 295)
    put_text(
        panel,
        f"selected pair correct={pair.get('pair_correct', False)}  "
        f"integrated target success={target_report['integrated_target_success']}",
        320,
    )
    put_text(
        panel,
        f"proxy={inspected.get('proxy_state', 'NOT_INSPECTED')}  localization error={error_text}",
        345,
    )
    return panel


def load_semantic_labels(
    protocol: dict[str, Any],
    frames: list[dict[str, Any]],
    images: np.ndarray,
    raw_labels: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    prediction_image = nib.load(protocol["evaluator"]["subregion_prediction_path"])
    segmentation_image = nib.load(protocol["evaluator"]["source_segmentation_path"])
    if prediction_image.shape != segmentation_image.shape or not np.allclose(
        prediction_image.affine, segmentation_image.affine
    ):
        raise ValueError("evaluator volumes use incompatible grids")
    prediction = np.asarray(prediction_image.dataobj, dtype=np.uint8)
    segmentation = np.asarray(segmentation_image.dataobj, dtype=np.uint8)
    masks = {
        level: right_level_masks(
            prediction, segmentation, prediction_image.affine, level
        )
        for level in ("L4", "L5")
    }
    bone_ids = bone_source_ids(Path(protocol["evaluator"]["label_conversion_path"]))
    return np.stack(
        [
            visible_interface_labels(
                image=np.asarray(image),
                raw_labels=np.asarray(raw),
                coordinate_map_index=np.asarray(coordinate),
                masks=masks[str(frame["target_name"])[:2]],
                bone_source_ids=bone_ids,
            ).labels
            for frame, image, raw, coordinate in zip(
                frames, images, raw_labels, coordinates
            )
        ]
    )


def write_session(
    full_writer: cv2.VideoWriter,
    blinded_writer: cv2.VideoWriter,
    *,
    session: Path,
    protocol: dict[str, Any],
    repeats: int,
) -> tuple[int, dict[str, int]]:
    actor_path = session / "results/actor_result.json"
    evaluator_path = session / "results/evaluation_report.json"
    validate_seal(actor_path)
    validate_seal(evaluator_path)
    actor = read_json(actor_path)
    evaluator = read_json(evaluator_path)
    frames = read_jsonl(session / "public/frames.jsonl")
    actions = read_jsonl(session / "privileged/actions.jsonl")
    images = np.load(session / "public/ultrasound.npz")["ultrasound"]
    raw_labels = np.load(session / "privileged/raw_labels.npz")["raw_labels"]
    coordinates = np.load(
        session / "privileged/pixel_to_volume_coordinates.npz"
    )["human_img_coords"]
    if not (
        len(frames) == len(images) == len(raw_labels) == len(coordinates)
    ):
        raise ValueError("session frame arrays differ in length")
    summary = read_json(session / "summary.json")
    robot = cv2.VideoCapture(str(session / "video/robot_probe_patient.mp4"))
    if not robot.isOpened():
        raise RuntimeError(f"cannot open robot video: {session}")
    robot_count = int(robot.get(cv2.CAP_PROP_FRAME_COUNT))
    robot_indices = map_acquisitions_to_robot_frames(
        frames,
        actions,
        action_stride=int(protocol["visualization"]["action_stride"]),
        robot_frame_count=robot_count,
    )
    recorded_count = int(summary["robot_video_frame_count"])
    if robot_count not in {recorded_count, recorded_count + 1}:
        raise ValueError("robot video frame count differs from run summary")
    labels = load_semantic_labels(protocol, frames, images, raw_labels, coordinates)
    observations = semantic_observations_by_frame(actor)
    target_actor = {row["target_name"]: row for row in actor["targets"]}
    target_evaluator = {row["target_name"]: row for row in evaluator["target_reports"]}
    inspected = evaluator_lookup(evaluator)
    state_counts: dict[str, int] = defaultdict(int)
    previous_frame = None
    previous_image = None
    output_count = 0
    try:
        for index, (frame, image, label, robot_index) in enumerate(
            zip(frames, images, labels, robot_indices)
        ):
            robot.set(cv2.CAP_PROP_POS_FRAMES, robot_index)
            ok, robot_frame = robot.read()
            if not ok:
                raise RuntimeError(f"cannot read robot frame {robot_index}: {session}")
            panels = render_actor_panels(
                robot_frame,
                image,
                frame,
                observations.get(index, []),
                target_actor[frame["target_name"]],
                previous_frame,
                previous_image,
            )
            target_report = target_evaluator[frame["target_name"]]
            top = np.hstack((panels[0], panels[1]))
            for writer, blinded in ((full_writer, False), (blinded_writer, True)):
                bottom = np.hstack(
                    (
                        panels[2],
                        evaluator_panel(
                            image,
                            label,
                            frame,
                            target_report,
                            inspected.get(index, {}),
                            blinded=blinded,
                        ),
                    )
                )
                canvas = np.vstack((top, bottom))
                for _ in range(repeats):
                    writer.write(canvas)
            state_counts[str(frame.get("scan_state", "TP_VIEW_SEARCH"))] += 1
            output_count += repeats
            previous_frame = frame
            previous_image = image
    finally:
        robot.release()
    return output_count, dict(state_counts)


def main() -> int:
    args = parse_args()
    if args.fps <= 0 or args.repeats_per_observation <= 0:
        raise ValueError("video timing must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    protocol_path = next((args.run_root / "context").glob("s0010_mbb_single_anatomy_poc_protocol_*.yaml"))
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    expected_protocol = protocol_path.with_suffix(".sha256").read_text().split()[0]
    if sha256(protocol_path) != expected_protocol:
        raise RuntimeError("protocol seal mismatch")
    end_to_end = read_json(args.run_root / "results/end_to_end_report.json")
    full_path = args.output_dir / "s0010_mbb_stage2_actual_execution_full.mp4"
    blinded_path = args.output_dir / "s0010_mbb_stage2_actual_execution_blinded.mp4"
    full_writer = cv2.VideoWriter(
        str(full_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, CANVAS_SIZE
    )
    blinded_writer = cv2.VideoWriter(
        str(blinded_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, CANVAS_SIZE
    )
    if not full_writer.isOpened() or not blinded_writer.isOpened():
        raise RuntimeError("cannot create review videos")
    frame_count = 0
    coverage: dict[str, dict[str, int]] = {}
    try:
        sessions = sorted(path for path in (args.run_root / "stage2").iterdir() if path.is_dir())
        for session in sessions:
            count, states = write_session(
                full_writer,
                blinded_writer,
                session=session,
                protocol=protocol,
                repeats=args.repeats_per_observation,
            )
            frame_count += count
            coverage[session.name] = states
    finally:
        full_writer.release()
        blinded_writer.release()
    manifest = {
        "schema": "sonogym-s0010-mbb-stage2-review-v1",
        "source_run": str(args.run_root.resolve()),
        "protocol_sha256": expected_protocol,
        "independent_confirmation": bool(
            protocol["execution"].get("independent_confirmation", False)
        ),
        "stage2_complete": bool(end_to_end["stage2"]["stage2_complete"]),
        "scientific_scope": end_to_end["scientific_scope"],
        "milestone_verdict": end_to_end["milestone_verdict"],
        "frame_count": frame_count,
        "fps": args.fps,
        "state_coverage": coverage,
        "full_video": {"path": str(full_path.resolve()), "sha256": sha256(full_path)},
        "blinded_video": {
            "path": str(blinded_path.resolve()),
            "sha256": sha256(blinded_path),
        },
        "information_boundary": {
            "panels_1_2_3": "sealed actor/public artifacts only",
            "panel_4_full": "privileged evaluator-only weak interface labels and results",
            "panel_4_blinded": "withheld",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
