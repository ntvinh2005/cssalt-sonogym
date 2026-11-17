#!/usr/bin/env python3
"""Collect one fixed-protocol Stage 2 supervised development session."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--protocol", required=True)
parser.add_argument("--session-id", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--assets-data-dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("MPLBACKEND", "Agg")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
import spinal_surgery  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from spinal_surgery.evaluation.two_stage_navigation.contracts import (  # noqa: E402
    assert_no_privileged_fields,
    ultrasound_sha256,
)
from spinal_surgery.evaluation.two_stage_navigation.acquisition_integrity import (  # noqa: E402
    AcquiredView,
    validate_new_view,
    validate_waypoint_bank,
)
from spinal_surgery.evaluation.two_stage_navigation.coordinates import (  # noqa: E402
    is_supported_stage2_dataset_protocol,
    is_stage2_trajectory_start,
    uses_coupled_stage2_dataset_controller,
    uses_segmented_stage2_dataset_capture,
    wrap_to_pi,
)
from spinal_surgery.evaluation.two_stage_navigation.stage2_models import (  # noqa: E402
    Stage2LabelConfig,
    supervised_targets,
)
from spinal_surgery.evaluation.two_stage_navigation.ultrasound_plane_geometry import (  # noqa: E402
    label_agreement,
    plane_from_full_pose,
    rotation_columns_from_quaternion_wxyz,
    sample_label_image,
)

from stage2_sim import (  # noqa: E402
    clean_ultrasound,
    current_command,
    current_realized_pose,
    current_rotation,
    move_proportional_target,
    move_to_pose,
    raw_label_slice,
    surface_proxy_ok,
    tensor_row,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def load_protocol() -> tuple[dict[str, Any], dict[str, Any], Path]:
    path = Path(args.protocol).resolve()
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not is_supported_stage2_dataset_protocol(protocol["protocol_id"]):
        raise RuntimeError("unexpected Stage 2 dataset protocol")
    seal = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if file_sha256(path) != seal:
        raise RuntimeError("Stage 2 dataset protocol seal mismatch")
    sessions = {row["session_id"]: row for row in protocol["sessions"]}
    if args.session_id not in sessions:
        raise RuntimeError("session is absent from the frozen protocol")
    return protocol, sessions[args.session_id], path


def ordered_waypoints(
    protocol: dict[str, Any], session: dict[str, Any], goal: np.ndarray
) -> list[np.ndarray]:
    if "absolute_waypoint_bank_id" in session:
        bank_id = str(session["absolute_waypoint_bank_id"])
        bank = protocol["sampling"]["absolute_waypoint_banks_index"][bank_id]
        offset = np.asarray(session.get("waypoint_offset_index", [0.0, 0.0, 0.0]))
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise RuntimeError("waypoint offset must contain three finite values")
        rows = [np.asarray(value, dtype=np.float64) + offset for value in bank]
        if not rows or any(row.shape != (3,) or not np.all(np.isfinite(row)) for row in rows):
            raise RuntimeError("absolute waypoint bank must contain finite three-value poses")
        return rows
    if "absolute_waypoints_index" in session:
        rows = [
            np.asarray(value, dtype=np.float64)
            for value in session["absolute_waypoints_index"]
        ]
        if not rows or any(row.shape != (3,) or not np.all(np.isfinite(row)) for row in rows):
            raise RuntimeError("absolute waypoint bank must contain finite three-value poses")
        return rows
    sampling = protocol["sampling"]
    x_values = [
        goal[0] + float(value) + float(session["x_jitter_index"])
        for value in sampling["goal_relative_x_offsets_index"]
    ]
    z_values = [
        goal[1] + float(value) + float(session["z_jitter_index"])
        for value in sampling["goal_relative_z_offsets_index"]
    ]
    angles = [
        float(value) + float(session["angle_jitter_rad"])
        for value in sampling["command_angles_rad"]
    ]
    rows: list[np.ndarray] = []
    for z_index, z_value in enumerate(z_values):
        current_x = x_values if z_index % 2 == 0 else list(reversed(x_values))
        for x_index, x_value in enumerate(current_x):
            current_angles = angles if (z_index + x_index) % 2 == 0 else list(reversed(angles))
            for angle in current_angles:
                rows.append(np.asarray([x_value, z_value, angle], dtype=np.float64))
    if len(rows) != int(sampling["expected_frames_per_session"]):
        raise RuntimeError("unexpected Stage 2 waypoint count")
    return rows


def main() -> int:
    protocol, session, protocol_path = load_protocol()
    output = Path(args.output_dir).resolve()
    public_dir = output / "public"
    privileged_dir = output / "privileged"
    started = time.monotonic()
    summary: dict[str, Any] = {
        "schema": "sonogym-stage2-dataset-session-summary-v1",
        "status": "FAILED",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_sha256(protocol_path),
        "session_id": session["session_id"],
        "split": session["split"],
        "seed": int(session["seed"]),
    }
    env = None
    action_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    privileged_rows: list[dict[str, Any]] = []
    images: list[np.ndarray] = []
    raw_images: list[np.ndarray] = []
    coordinate_maps: list[np.ndarray] = []
    full_pose_rows: list[dict[str, Any]] = []
    local_image_coordinates: np.ndarray | None = None
    try:
        exact_pose_replay = protocol.get("exact_pose_replay", {})
        exact_pose_replay_enabled = bool(exact_pose_replay.get("enabled", False))
        env_cfg = parse_env_cfg(
            protocol["scope"]["task"], device=args.device or "cuda:0", num_envs=1
        )
        env_cfg.seed = int(session["seed"])
        env = gym.make(protocol["scope"]["task"], cfg=env_cfg, render_mode=None)
        if exact_pose_replay_enabled:
            env.unwrapped.US_slicer.capture_snapshot_enabled = True
            env.unwrapped.US_slicer.capture_session_id = str(session["session_id"])
            env.unwrapped.US_slicer.capture_reset_id = str(
                session.get("reset_id", session["session_id"])
            )
        observation, _ = env.reset(seed=int(session["seed"]))
        unwrapped = env.unwrapped
        if tuple(observation["policy"].shape) != (1, 1, 200, 150):
            raise RuntimeError("unexpected policy observation shape")
        if unwrapped.US_slicer.sim_mode != "conv" or unwrapped.scene.num_envs != 1:
            raise RuntimeError("collector requires one conv SonoGym environment")

        raw_path = (
            Path(args.assets_data_dir)
            / "HumanModels"
            / "selected_dataset_stl"
            / "s0010"
            / "combined_label_map.nii.gz"
        )
        raw_volume = np.asarray(nib.load(raw_path).dataobj, dtype=np.uint8)
        goal = tensor_row(unwrapped.goal_cmd_pose)
        waypoints = ordered_waypoints(protocol, session, goal)
        expected_frame_count = int(
            session.get(
                "expected_frame_count",
                protocol["sampling"].get("expected_frames_per_session", len(waypoints)),
            )
        )
        if len(waypoints) != expected_frame_count:
            raise RuntimeError("session waypoint count differs from its frozen frame count")
        sampling = protocol["sampling"]
        command_range = unwrapped.US_slicer.x_z_range.detach().cpu().to(torch.float64).numpy()
        validate_waypoint_bank(
            waypoints,
            command_minimum=command_range[0],
            command_maximum=command_range[1],
            label_resolution_mm=float(unwrapped.US_slicer.label_res) * 1000.0,
            position_tolerance_mm=float(sampling["physical_tracking_position_tolerance_mm"]),
            orientation_tolerance_deg=float(
                sampling["physical_tracking_orientation_tolerance_deg"]
            ),
            separation_multiplier=float(sampling.get("waypoint_separation_multiplier", 1.0)),
        )
        trajectory_start_frames = [0]
        trajectory_seed_stride = 0
        if uses_segmented_stage2_dataset_capture(protocol["protocol_id"]):
            trajectory_start_frames = [
                int(value) for value in sampling["trajectory_start_frames"]
            ]
            trajectory_seed_stride = int(sampling["trajectory_seed_stride"])
            if trajectory_start_frames[0] != 0 or trajectory_start_frames != sorted(
                set(trajectory_start_frames)
            ):
                raise RuntimeError("invalid frozen trajectory boundaries")
        position_tolerance = float(sampling["physical_tracking_position_tolerance_mm"])
        orientation_tolerance = float(
            sampling["physical_tracking_orientation_tolerance_deg"]
        )
        label = protocol["labels"]
        label_config = Stage2LabelConfig(
            maximum_position_error_mm=float(label["validity"]["maximum_position_error_mm"]),
            maximum_orientation_error_deg=float(
                label["validity"]["maximum_orientation_error_deg"]
            ),
            minimum_l4_visible_fraction=float(
                label["validity"]["minimum_l4_visible_fraction"]
            ),
            visibility_normalization_fraction=float(
                label["quality"]["visibility_normalization_fraction"]
            ),
        )
        previous_realized = current_realized_pose(unwrapped).copy()
        previous_acquired_view: AcquiredView | None = None
        total_steps = 0
        for frame_sequence in range(expected_frame_count):
            trajectory_index = max(
                index
                for index, start in enumerate(trajectory_start_frames)
                if start <= frame_sequence
            )
            trajectory_start = trajectory_start_frames[trajectory_index]
            trajectory_id = f"{session['session_id']}_seg{trajectory_index:02d}"
            if frame_sequence == trajectory_start and frame_sequence > 0:
                trajectory_seed = int(session["seed"]) + (
                    trajectory_index * trajectory_seed_stride
                )
                observation, _ = env.reset(seed=trajectory_seed)
                if tuple(observation["policy"].shape) != (1, 1, 200, 150):
                    raise RuntimeError("unexpected policy observation shape after reset")
                goal = tensor_row(unwrapped.goal_cmd_pose)
                waypoints = ordered_waypoints(protocol, session, goal)
                previous_realized = current_realized_pose(unwrapped).copy()
            waypoint = waypoints[frame_sequence]
            maximum_steps = int(
                sampling[
                    "maximum_initial_positioning_steps"
                    if is_stage2_trajectory_start(
                        frame_sequence, trajectory_start_frames
                    )
                    else "maximum_action_steps_per_waypoint"
                ]
            )
            action_start = len(action_rows)
            controller_target = waypoint.copy()
            acceptance_target = waypoint.copy()
            movement_orientation_tolerance = orientation_tolerance
            if uses_coupled_stage2_dataset_controller(protocol["protocol_id"]):
                controller = sampling["physical_controller"]
                if waypoint[2] >= float(
                    controller["ceiling_assisted_minimum_requested_angle_rad"]
                ):
                    controller_target[2] = float(controller["controller_target_angle_rad"])
                    if bool(controller.get("accept_against_evaluator_goal_angle", False)):
                        acceptance_target[2] = goal[2]
                        movement_orientation_tolerance = float(
                            label["validity"]["maximum_orientation_error_deg"]
                        )
                trajectory = move_proportional_target(
                    env,
                    controller_target,
                    acceptance_target,
                    maximum_steps=maximum_steps,
                    maximum_position_error_mm=position_tolerance,
                    maximum_orientation_error_deg=movement_orientation_tolerance,
                    ratio=tuple(float(value) for value in controller["ratio"]),
                    allow_angle_ceiling_saturation=bool(
                        controller.get("capture_angle_ceiling_limited_samples", False)
                    ),
                    trajectory_sink=action_rows,
                )
            else:
                trajectory = move_to_pose(
                    env,
                    waypoint,
                    maximum_steps=maximum_steps,
                    maximum_position_error_mm=position_tolerance,
                    maximum_orientation_error_deg=orientation_tolerance,
                )
                action_rows.extend(trajectory)
            total_steps += len(trajectory)
            if total_steps > int(sampling["maximum_total_action_steps_per_session"]):
                raise RuntimeError("Stage 2 dataset action budget exceeded")
            for row in action_rows[action_start:]:
                row["frame_sequence"] = frame_sequence
                row["trajectory_id"] = trajectory_id

            snapshot = (
                unwrapped.US_slicer.copy_latest_acquisition_snapshot()
                if exact_pose_replay_enabled
                else None
            )
            if snapshot is not None:
                image = np.asarray(
                    snapshot["clean_ultrasound"][0, :, :, 0].detach().cpu().numpy(),
                    dtype=np.float32,
                ).copy()
                native_coordinates = np.asarray(
                    snapshot["human_img_coords"][0].detach().cpu().numpy(),
                    dtype=np.float64,
                ).copy()
                width, depth = (int(value) for value in snapshot["image_size_width_depth"])
                raw = sample_label_image(
                    volume=raw_volume,
                    coordinates_index=native_coordinates,
                    width=width,
                    depth=depth,
                ).astype(np.uint8, copy=False)
                current_pose = tensor_row(
                    unwrapped.gt_motion_generator.human_cmd_state_from_ee_pose(
                        snapshot["human_to_ee_pos"], snapshot["human_to_ee_quat"]
                    )
                )
                producer_sequence = int(snapshot["producer_sequence"])
                acquisition_sequence = int(snapshot["acquisition_sequence"])
                acquisition_id = (
                    f"{session['session_id']}:{snapshot['reset_id']}:{acquisition_sequence}"
                )
            else:
                image = clean_ultrasound(unwrapped)
                raw = raw_label_slice(unwrapped, raw_volume)
                current_pose = current_realized_pose(unwrapped)
                producer_sequence = int(unwrapped.num_step)
                acquisition_sequence = producer_sequence
                acquisition_id = f"{session['session_id']}:{trajectory_id}:{producer_sequence}"
            frame_hash = ultrasound_sha256(image)
            delta_mm = (current_pose[:2] - previous_realized[:2]) * 1.5
            delta_angle = float(wrap_to_pi(current_pose[2] - previous_realized[2]))
            transition_valid = bool(
                all(
                    bool(row["finite"])
                    and bool(row["surface_proxy_ok"])
                    and bool(row["command_mapping_valid"])
                    and not bool(row["terminated"])
                    and not bool(row["truncated"])
                    for row in trajectory
                )
                and all(not bool(row["command_position_clamped"]) for row in trajectory)
            )
            technical_valid = bool(
                transition_valid
                and surface_proxy_ok(unwrapped)
                and not bool(unwrapped.US_slicer.no_collide[0].item())
                and float(np.std(image)) > 1e-8
            )
            current_acquired_view = AcquiredView(
                acquisition_id=acquisition_id,
                session_id=str(session["session_id"]),
                reset_id=str(session.get("reset_id", session["session_id"])),
                acquisition_sequence=acquisition_sequence,
                producer_sequence=producer_sequence,
                realized_pose_index=current_pose,
                frame_sha256=frame_hash,
                action_count=len(trajectory),
                technical_image_valid=technical_valid,
            )
            viewpoint_check = None
            if previous_acquired_view is not None:
                viewpoint_check = validate_new_view(
                    previous_acquired_view,
                    current_acquired_view,
                    label_resolution_mm=1.5,
                    minimum_translation_mm=float(
                        sampling["minimum_realized_viewpoint_translation_mm"]
                    ),
                    minimum_orientation_deg=float(
                        sampling["minimum_realized_viewpoint_orientation_deg"]
                    ),
                )
                if not viewpoint_check["accepted"]:
                    raise RuntimeError(
                        f"acquired frame is stale or not a distinct viewpoint: {viewpoint_check}"
                    )
            elif len(trajectory) == 0 or not technical_valid:
                raise RuntimeError("first requested viewpoint was not physically acquired")
            position_error_mm = float(np.linalg.norm(current_pose[:2] - goal[:2]) * 1.5)
            orientation_error_deg = (
                abs(float(wrap_to_pi(current_pose[2] - goal[2]))) * 180.0 / math.pi
            )
            l4_fraction = float(np.count_nonzero(raw == 28) / raw.size)
            validity, quality = supervised_targets(
                technical_image_valid=technical_valid,
                position_error_mm=position_error_mm,
                orientation_error_deg=orientation_error_deg,
                l4_visible_fraction=l4_fraction,
                config=label_config,
            )
            public_row = {
                "schema": "sonogym-stage2-actor-frame-v1",
                "session_id": session["session_id"],
                "trajectory_id": trajectory_id,
                "trajectory_frame_sequence": frame_sequence - trajectory_start,
                "split": session["split"],
                "frame_sequence": frame_sequence,
                "producer_sequence": producer_sequence,
                "acquisition_id": acquisition_id,
                "acquisition_sequence": acquisition_sequence,
                "frame_sha256": frame_hash,
                "image_archive": "public/ultrasound.npz",
                "image_archive_index": frame_sequence,
                "executed_delta_relative_mm": delta_mm.tolist(),
                "executed_delta_angle_rad": delta_angle,
                "action_steps_since_previous_frame": len(trajectory),
                "action_accepted": transition_valid,
                "action_completed": transition_valid,
                "action_clamped": any(bool(row["command_clamped"]) for row in trajectory),
                "action_position_clamped": any(
                    bool(row["command_position_clamped"]) for row in trajectory
                ),
                "action_angle_clamped": any(
                    bool(row["command_angle_clamped"]) for row in trajectory
                ),
                "contact_ok": surface_proxy_ok(unwrapped),
                "workspace_ok": all(
                    not bool(row["command_position_clamped"]) for row in trajectory
                ),
                "image_valid": technical_valid,
                "physical_viewpoint_check": viewpoint_check,
            }
            assert_no_privileged_fields(public_row)
            public_rows.append(public_row)
            privileged_rows.append(
                {
                    "schema": "sonogym-stage2-privileged-annotation-v1",
                    "session_id": session["session_id"],
                    "trajectory_id": trajectory_id,
                    "trajectory_frame_sequence": frame_sequence - trajectory_start,
                    "trajectory_seed": int(session["seed"])
                    + trajectory_index * trajectory_seed_stride,
                    "frame_sequence": frame_sequence,
                    "frame_sha256": frame_hash,
                    "acquisition_id": acquisition_id,
                    "patient_id": "s0010",
                    "command_pose_index": current_command(unwrapped).tolist(),
                    "realized_pose_index": current_pose.tolist(),
                    "requested_waypoint_pose_index": waypoint.tolist(),
                    "controller_target_pose_index": controller_target.tolist(),
                    "goal_cmd_pose": goal.tolist(),
                    "position_error_mm": position_error_mm,
                    "orientation_error_deg": orientation_error_deg,
                    "l4_visible_fraction": l4_fraction,
                    "raw_label_pixel_counts": {
                        str(value): int(np.count_nonzero(raw == value))
                        for value in range(25, 32)
                    },
                    "technical_image_valid": technical_valid,
                    "validity_label": validity,
                    "quality_label": quality,
                    "raw_label_archive": "privileged/raw_labels.npz",
                    "raw_label_archive_index": frame_sequence,
                }
            )
            if exact_pose_replay_enabled:
                local_coordinates = np.asarray(
                    snapshot["native_local_image_coordinates_m"].detach().cpu().numpy(),
                    dtype=np.float64,
                )
                if local_image_coordinates is None:
                    local_image_coordinates = local_coordinates.copy()
                elif not np.array_equal(local_image_coordinates, local_coordinates):
                    raise RuntimeError("native local image coordinates changed during capture")
                human_to_ee_position = tensor_row(snapshot["human_to_ee_pos"])
                human_to_ee_quaternion = tensor_row(snapshot["human_to_ee_quat"])
                rotation = rotation_columns_from_quaternion_wxyz(
                    tensor_row(snapshot["human_to_ee_quat"])
                )
                full_plane = plane_from_full_pose(
                    end_effector_position_m=human_to_ee_position,
                    rotation_columns=rotation,
                    image_origin_height_m=float(snapshot["image_origin_height_m"]),
                    local_coordinates_m=local_coordinates,
                    label_resolution_m=float(snapshot["label_resolution_m"]),
                    volume_shape=tuple(int(value) for value in raw_volume.shape),
                )
                native_raw = sample_label_image(
                    volume=raw_volume,
                    coordinates_index=native_coordinates,
                    width=width,
                    depth=depth,
                ).astype(np.uint8, copy=False)
                replay_raw = sample_label_image(
                    volume=raw_volume,
                    coordinates_index=full_plane.coordinates_index,
                    width=width,
                    depth=depth,
                ).astype(np.uint8, copy=False)
                native_metric = label_agreement(raw, native_raw)
                replay_metric = label_agreement(raw, replay_raw)
                maximum_coordinate_error = float(
                    np.max(np.abs(native_coordinates - full_plane.coordinates_index))
                )
                coordinate_tolerance = float(
                    exact_pose_replay["maximum_coordinate_error_index"]
                )
                frame_passes = bool(
                    native_metric["exact_raw_label_agreement"] == 1.0
                    and replay_metric["passes_unchanged_gate"]
                    and maximum_coordinate_error <= coordinate_tolerance
                )
                coordinate_maps.append(native_coordinates.astype(np.float32))
                full_pose_rows.append(
                    {
                        "schema": "sonogym-exact-pose-replay-frame-v1",
                        "session_id": session["session_id"],
                        "frame_sequence": frame_sequence,
                        "producer_sequence": int(unwrapped.num_step),
                        "acquisition_id": acquisition_id,
                        "acquisition_sequence": acquisition_sequence,
                        "simulation_step": int(snapshot["simulation_step"]),
                        "simulation_time_s": float(snapshot["simulation_time_s"]),
                        "frame_sha256": frame_hash,
                        "human_to_ee_position_m": human_to_ee_position.tolist(),
                        "human_to_ee_quaternion_wxyz": human_to_ee_quaternion.tolist(),
                        "human_to_ee_rotation_columns": rotation.tolist(),
                        "image_origin_height_m": float(snapshot["image_origin_height_m"]),
                        "label_resolution_m": float(snapshot["label_resolution_m"]),
                        "image_resolution_m": float(snapshot["image_resolution_m"]),
                        "image_size_width_depth": [width, depth],
                        "world_to_human_position_m": tensor_row(
                            snapshot["world_to_human_pos"]
                        ).tolist(),
                        "world_to_human_quaternion_wxyz": tensor_row(
                            snapshot["world_to_human_quat"]
                        ).tolist(),
                        "world_to_ee_position_m": tensor_row(
                            snapshot["world_to_ee_pos"]
                        ).tolist(),
                        "world_to_ee_quaternion_wxyz": tensor_row(
                            snapshot["world_to_ee_quat"]
                        ).tolist(),
                        "ultrasound_array_sha256": array_sha256(image),
                        "raw_label_array_sha256": array_sha256(raw),
                        "native_coordinate_array_sha256": array_sha256(
                            native_coordinates.astype(np.float32)
                        ),
                        "coordinate_archive": "privileged/pixel_to_volume_coordinates.npz",
                        "coordinate_archive_index": frame_sequence,
                        "maximum_coordinate_error_index": maximum_coordinate_error,
                        "native_coordinate_raw_label_metrics": native_metric,
                        "full_pose_raw_label_metrics": replay_metric,
                        "frame_passes": frame_passes,
                    }
                )
                privileged_rows[-1].update(
                    {
                        "exact_pose_replay_record": "privileged/full_pose_replay.jsonl",
                        "exact_pose_replay_record_index": frame_sequence,
                    }
                )
            images.append(image.copy())
            raw_images.append(raw.copy())
            previous_realized = current_pose
            previous_acquired_view = current_acquired_view

        distinct_frame_hash_count = len({row["frame_sha256"] for row in public_rows})
        minimum_distinct_frame_hashes = int(
            protocol.get("dataset_gate", {}).get(
                "minimum_distinct_frame_hashes_per_session", len(public_rows) // 2
            )
        )
        summary["distinct_frame_hash_count"] = distinct_frame_hash_count
        summary["minimum_distinct_frame_hashes"] = minimum_distinct_frame_hashes
        distinct_frame_gate_passed = bool(
            distinct_frame_hash_count >= minimum_distinct_frame_hashes
        )
        if not distinct_frame_gate_passed and not exact_pose_replay_enabled:
            raise RuntimeError(
                "Stage 2 dataset contains insufficient distinct images: "
                f"{distinct_frame_hash_count} < {minimum_distinct_frame_hashes}"
            )
        public_dir.mkdir(parents=True, exist_ok=True)
        privileged_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(public_dir / "ultrasound.npz", ultrasound=np.stack(images))
        np.savez_compressed(privileged_dir / "raw_labels.npz", raw_labels=np.stack(raw_images))
        if exact_pose_replay_enabled:
            if len(coordinate_maps) != len(public_rows) or local_image_coordinates is None:
                raise RuntimeError("exact-pose replay records are incomplete")
            np.savez_compressed(
                privileged_dir / "pixel_to_volume_coordinates.npz",
                human_img_coords=np.stack(coordinate_maps),
                human_img_coords_integer=np.trunc(np.stack(coordinate_maps)).astype(np.int32),
                native_local_image_coordinates_m=local_image_coordinates.astype(np.float32),
            )
            write_jsonl(privileged_dir / "full_pose_replay.jsonl", full_pose_rows)
        write_jsonl(public_dir / "frames.jsonl", public_rows)
        write_jsonl(privileged_dir / "annotations.jsonl", privileged_rows)
        write_jsonl(privileged_dir / "actions.jsonl", action_rows)
        exact_pose_replay_passed_frames = sum(
            int(row["frame_passes"]) for row in full_pose_rows
        )
        exact_pose_replay_all_frames_passed = bool(
            not exact_pose_replay_enabled
            or (
                exact_pose_replay_passed_frames == len(public_rows)
                and distinct_frame_gate_passed
            )
        )
        summary.update(
            status=(
                "PASSED"
                if exact_pose_replay_all_frames_passed
                else "EXACT_POSE_REPLAY_NO_GO"
            ),
            frame_count=len(public_rows),
            trajectory_count=len(trajectory_start_frames),
            valid_frame_count=sum(int(row["validity_label"]) for row in privileged_rows),
            distinct_frame_hash_count=distinct_frame_hash_count,
            distinct_frame_gate_passed=distinct_frame_gate_passed,
            total_action_steps=total_steps,
            public_manifest="public/frames.jsonl",
            privileged_manifest="privileged/annotations.jsonl",
            ultrasound_archive="public/ultrasound.npz",
            raw_label_archive="privileged/raw_labels.npz",
            exact_pose_replay_enabled=exact_pose_replay_enabled,
            exact_pose_replay_passed_frames=exact_pose_replay_passed_frames,
            exact_pose_replay_all_frames_passed=exact_pose_replay_all_frames_passed,
        )
        return 0 if exact_pose_replay_all_frames_passed else 3
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        return 1
    finally:
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if action_rows:
            write_jsonl(privileged_dir / "actions_partial.jsonl", action_rows)
        write_json(output / "summary.json", summary)
        if env is not None:
            env.close()


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)
