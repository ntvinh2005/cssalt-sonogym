#!/usr/bin/env python3
"""Collect one frozen Stage 1 SonoGym observability session through real env steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from collections import Counter
from pathlib import Path

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
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
import spinal_surgery  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from spinal_surgery.evaluation.two_stage_navigation.contracts import (  # noqa: E402
    ActorFrame,
    FrameFreshnessGuard,
    ultrasound_sha256,
)
from spinal_surgery.evaluation.two_stage_navigation.coordinates import (  # noqa: E402
    ExecutedMotion,
    HomeRelativeOdometry,
    assess_native_command,
    command_target_from_realized_error,
    plan_native_action,
    pose_within_tolerance,
    wrap_to_pi,
)


RAW_LABEL_NAMES = {
    25: "sacrum",
    26: "S1",
    27: "L5",
    28: "L4",
    29: "L3",
    30: "L2",
    31: "L1",
}
SUPPORTED_PROTOCOL_IDS = frozenset(
    {
        "sonogym-unity-two-stage-g0-g1-v2",
        "sonogym-stage1-l4-confirmation-v1",
        "sonogym-stage1-l4-confirmation-v2",
    }
)
LUMBAR_LABELS = frozenset(RAW_LABEL_NAMES)
ADJACENT_NEGATIVE_LABELS = frozenset({26, 27, 29, 30, 31})
COMMAND_POSITION_TOLERANCE_INDEX = 0.15
COMMAND_ANGLE_TOLERANCE_RAD = 1e-4
POSITION_CORRECTION_MULTIPLIER = 2.5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def load_protocol() -> tuple[dict, dict, Path]:
    protocol_path = Path(args.protocol).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_id"] not in SUPPORTED_PROTOCOL_IDS:
        raise RuntimeError("unexpected protocol identity")
    if protocol["resource_bound_if_later_approved"]["current_protocol_authorizes_submission"]:
        raise RuntimeError("protocol must not self-authorize scheduler execution")
    sessions = {item["session_id"]: item for item in protocol["sessions"]}
    if args.session_id not in sessions:
        raise RuntimeError(f"session {args.session_id!r} is not frozen in the protocol")
    seal_path = protocol_path.with_suffix(".sha256")
    expected_hash = seal_path.read_text(encoding="utf-8").split()[0]
    actual_hash = file_sha256(protocol_path)
    if actual_hash != expected_hash:
        raise RuntimeError("protocol hash does not match protocol.sha256")
    return protocol, sessions[args.session_id], protocol_path


def tensor_row(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().to(torch.float64).reshape(value.shape[0], -1)[0].numpy()


def current_rotation(unwrapped) -> np.ndarray:
    quaternion = unwrapped.extras["human_to_ee_quat"]
    return matrix_from_quat(quaternion)[0].detach().cpu().to(torch.float64).numpy()


def current_command(unwrapped) -> np.ndarray:
    return tensor_row(unwrapped.US_slicer.current_x_z_x_angle_cmd)


def current_realized_pose(unwrapped) -> np.ndarray:
    return tensor_row(unwrapped.cur_cmd_pose)


def tracking_errors(unwrapped, target: np.ndarray) -> tuple[float, float]:
    pose = current_realized_pose(unwrapped)
    position_mm = float(np.linalg.norm(pose[:2] - target[:2]) * 1.5)
    orientation_deg = abs(float(wrap_to_pi(pose[2] - target[2]))) * 180.0 / math.pi
    return position_mm, orientation_deg


def surface_proxy_ok(unwrapped) -> bool:
    values = torch.cat(
        [
            unwrapped.US_slicer.target_position.reshape(-1),
            unwrapped.US_slicer.target_z_axis.reshape(-1),
        ]
    )
    return bool(torch.isfinite(values).all().item())


def take_step(env, target: np.ndarray) -> tuple[dict, dict[str, object]]:
    unwrapped = env.unwrapped
    before_command = current_command(unwrapped)
    before_realized = current_realized_pose(unwrapped)
    delta = target[:2] - before_command[:2]
    angle_delta = float(wrap_to_pi(target[2] - before_command[2]))
    planned = plan_native_action(
        delta,
        angle_delta,
        current_rotation(unwrapped),
        action_scale=tensor_row(unwrapped.action_scale),
        maximum_applied_action=tensor_row(unwrapped.max_action),
    )
    action = torch.as_tensor(
        planned.normalized_action.reshape(1, 3),
        dtype=torch.float32,
        device=unwrapped.device,
    )
    observation, reward, terminated, truncated, info = env.step(action)
    after_command = current_command(unwrapped)
    after_realized = current_realized_pose(unwrapped)
    command_range = (
        unwrapped.US_slicer.x_z_range.detach()
        .cpu()
        .to(torch.float64)
        .numpy()
    )
    assessment = assess_native_command(
        before_command,
        after_command,
        planned,
        target,
        command_range[0],
        command_range[1],
        position_mapping_tolerance_index=COMMAND_POSITION_TOLERANCE_INDEX,
        angle_mapping_tolerance_rad=COMMAND_ANGLE_TOLERANCE_RAD,
    )
    finite = bool(
        np.all(np.isfinite(after_command))
        and np.all(np.isfinite(after_realized))
        and bool(torch.isfinite(observation["policy"]).all().item())
        and bool(torch.isfinite(reward).all().item())
    )
    position_tracking_mm, angle_tracking_deg = tracking_errors(unwrapped, after_command)
    status: dict[str, object] = {
        "normalized_action": planned.normalized_action.tolist(),
        "before_command": before_command.tolist(),
        "after_command": after_command.tolist(),
        "predicted_command": assessment.expected_command.tolist(),
        "before_realized_pose": before_realized.tolist(),
        "after_realized_pose": after_realized.tolist(),
        "command_clamped": assessment.native_clamped,
        "command_mapping_valid": assessment.mapping_valid,
        "command_progressed": assessment.progressed_toward_target,
        "command_position_mapping_error_index": assessment.position_mapping_error_index,
        "command_angle_mapping_error_rad": assessment.angle_mapping_error_rad,
        "finite": finite,
        "surface_proxy_ok": surface_proxy_ok(unwrapped),
        "native_no_collide": bool(unwrapped.US_slicer.no_collide[0].item()),
        "position_tracking_error_mm": position_tracking_mm,
        "orientation_tracking_error_deg": angle_tracking_deg,
        "terminated": bool(terminated[0].item()),
        "truncated": bool(truncated[0].item()),
        "info_keys": sorted(info),
    }
    return observation, status


def move_to_home(
    env,
    target: np.ndarray,
    maximum_position_error_mm: float,
    maximum_orientation_error_deg: float,
    maximum_steps: int = 80,
    trajectory: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    if trajectory is None:
        trajectory = []
    command_range = (
        env.unwrapped.US_slicer.x_z_range.detach()
        .cpu()
        .to(torch.float64)
        .numpy()
    )
    for step in range(maximum_steps):
        position_tracking_mm, angle_tracking_deg = tracking_errors(env.unwrapped, target)
        if pose_within_tolerance(
            position_tracking_mm,
            angle_tracking_deg,
            maximum_position_error_mm,
            maximum_orientation_error_deg,
        ):
            return trajectory
        command_target = command_target_from_realized_error(
            current_realized_pose(env.unwrapped),
            target,
            position_multiplier=POSITION_CORRECTION_MULTIPLIER,
        )
        command_target = np.clip(command_target, command_range[0], command_range[1])
        _, status = take_step(env, command_target)
        status["home_return_step"] = step + 1
        trajectory.append(status)
        # Home is low-level feedback positioning. Command mapping remains in the
        # privileged trace; realized-pose convergence above is the acceptance test.
        if (
            not status["finite"]
            or not status["surface_proxy_ok"]
        ):
            raise RuntimeError(f"home return failed at step {step + 1}: {status}")
        if status["terminated"] or status["truncated"]:
            raise RuntimeError("environment ended during physical Home return")
    raise RuntimeError(
        f"Home return did not converge after {maximum_steps} steps: {trajectory[-1]}"
    )


def move_to_waypoint(
    env,
    target: np.ndarray,
    maximum_steps: int,
    action_rows: list[dict[str, object]],
    phase: str,
    frame_sequence: int,
) -> list[dict[str, object]]:
    trajectory: list[dict[str, object]] = []
    command_range = (
        env.unwrapped.US_slicer.x_z_range.detach()
        .cpu()
        .to(torch.float64)
        .numpy()
    )
    command_target = command_target_from_realized_error(
        current_realized_pose(env.unwrapped),
        target,
        position_multiplier=POSITION_CORRECTION_MULTIPLIER,
    )
    command_target = np.clip(command_target, command_range[0], command_range[1])
    for step in range(maximum_steps):
        _, status = take_step(env, command_target)
        status["waypoint_action_step"] = step + 1
        status["phase"] = phase
        status["frame_sequence"] = frame_sequence
        status["waypoint_target"] = target.tolist()
        status["position_correction_multiplier"] = POSITION_CORRECTION_MULTIPLIER
        waypoint_position_error_mm, waypoint_angle_error_deg = tracking_errors(
            env.unwrapped, target
        )
        status["waypoint_position_error_mm"] = waypoint_position_error_mm
        status["waypoint_orientation_error_deg"] = waypoint_angle_error_deg
        trajectory.append(status)
        action_rows.append(dict(status))
        realized = current_realized_pose(env.unwrapped)
        if (
            not status["finite"]
            or status["command_clamped"]
            or not status["command_mapping_valid"]
            or not status["command_progressed"]
        ):
            raise RuntimeError(f"waypoint command failed at action {step + 1}: {status}")
        if status["terminated"] or status["truncated"]:
            raise RuntimeError("environment ended while moving to a scan waypoint")
        if (
            waypoint_position_error_mm <= 5.0
            and waypoint_angle_error_deg <= 5.0
        ):
            return trajectory
        command_target = command_target_from_realized_error(
            realized,
            target,
            position_multiplier=POSITION_CORRECTION_MULTIPLIER,
        )
        command_target = np.clip(command_target, command_range[0], command_range[1])
    raise RuntimeError(
        f"scan waypoint did not converge after {maximum_steps} actions: {trajectory[-1]}"
    )


def raw_label_slice(unwrapped, raw_volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coordinates = (
        unwrapped.US_slicer.human_img_coords[0].detach().cpu().to(torch.int64).numpy()
    )
    raw = raw_volume[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]
    width, depth = unwrapped.US_slicer.img_size
    raw_hw = raw.reshape(width, depth, -1)[:, :, 0].T.astype(np.uint8, copy=False)
    converted_hw = (
        unwrapped.US_slicer.label_img_tensor[0, :, :, 0]
        .detach()
        .cpu()
        .numpy()
        .T.astype(np.uint8, copy=False)
    )
    return raw_hw, converted_hw


def main() -> int:
    protocol, session, protocol_path = load_protocol()
    output_dir = Path(args.output_dir).resolve()
    public_dir = output_dir / "public"
    privileged_dir = output_dir / "privileged"
    public_dir.mkdir(parents=True, exist_ok=True)
    privileged_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    started = time.monotonic()
    summary: dict[str, object] = {
        "schema": "sonogym-two-stage-capture-session-summary-v1",
        "status": "FAILED",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_sha256(protocol_path),
        "session_id": session["session_id"],
        "split": session["split"],
        "seed": int(session["seed"]),
    }
    env = None
    public_rows: list[dict[str, object]] = []
    privileged_rows: list[dict[str, object]] = []
    ultrasound_frames: list[np.ndarray] = []
    raw_label_frames: list[np.ndarray] = []
    home_trajectory: list[dict[str, object]] = []
    scan_action_rows: list[dict[str, object]] = []
    try:
        task = protocol["scope"]["task"]
        env_cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=1)
        env_cfg.seed = int(session["seed"])
        env = gym.make(task, cfg=env_cfg, render_mode=None)
        unwrapped = env.unwrapped
        if unwrapped.US_slicer.sim_mode != "conv" or unwrapped.scene.num_envs != 1:
            raise RuntimeError("collector requires one real conv SonoGym environment")
        observation, _ = env.reset(seed=int(session["seed"]))
        if tuple(observation["policy"].shape) != (1, 1, 200, 150):
            raise RuntimeError(f"unexpected policy observation shape {observation['policy'].shape}")

        raw_label_path = (
            Path(args.assets_data_dir)
            / "HumanModels"
            / "selected_dataset_stl"
            / "s0010"
            / "combined_label_map.nii.gz"
        )
        raw_volume = np.asarray(nib.load(raw_label_path).dataobj, dtype=np.uint8)
        target = np.asarray(
            [session["patient_x_index"], 195.0, session["angle_rad"]], dtype=np.float64
        )
        movement_policy = protocol["geometry"]["movement_policy"]
        maximum_position_error_mm = float(
            movement_policy["maximum_tracking_position_error_mm"]
        )
        maximum_orientation_error_deg = float(
            movement_policy["maximum_tracking_orientation_error_deg"]
        )
        reset_command = current_command(unwrapped).copy()
        reset_realized = current_realized_pose(unwrapped).copy()
        move_to_home(
            env,
            target,
            maximum_position_error_mm,
            maximum_orientation_error_deg,
            trajectory=home_trajectory,
        )
        home_pose = current_realized_pose(unwrapped).copy()
        home_command = current_command(unwrapped).copy()
        home_position_error_mm, home_orientation_error_deg = tracking_errors(
            unwrapped, target
        )
        if not pose_within_tolerance(
            home_position_error_mm,
            home_orientation_error_deg,
            maximum_position_error_mm,
            maximum_orientation_error_deg,
        ):
            raise RuntimeError("physical Home return ended outside its realized-pose tolerance")

        freshness = FrameFreshnessGuard()
        odometry = HomeRelativeOdometry()
        previous_realized = home_pose.copy()
        level_visible_frames: Counter[str] = Counter()
        direction_counts: Counter[str] = Counter()
        raw_conversion_mismatches = 0

        def capture(
            frame_sequence: int,
            phase: str,
            requested_action: np.ndarray,
            status: dict[str, object],
        ) -> None:
            nonlocal previous_realized, raw_conversion_mismatches
            current_pose = current_realized_pose(unwrapped)
            delta_home_mm = (current_pose[:2] - previous_realized[:2]) * 1.5
            delta_angle = float(wrap_to_pi(current_pose[2] - previous_realized[2]))
            transition_valid = bool(
                status["finite"]
                and not status["command_clamped"]
                and status["command_mapping_valid"]
                and status["command_progressed"]
                and status["surface_proxy_ok"]
                and not status["terminated"]
                and not status["truncated"]
            )
            odometry.apply(
                ExecutedMotion(
                    delta_home_mm=delta_home_mm,
                    delta_angle_rad=delta_angle,
                    accepted=transition_valid,
                    completed=transition_valid,
                    clamped=bool(status["command_clamped"]),
                    contact_ok=bool(status["surface_proxy_ok"]),
                    workspace_ok=not bool(status["command_clamped"]),
                )
            )
            ultrasound_hwt = (
                unwrapped.US_slicer.us_img_tensor[0].detach().cpu().to(torch.float32).numpy()
            )
            if ultrasound_hwt.shape != (200, 150, 1):
                raise RuntimeError(
                    f"unexpected clean ultrasound tensor shape {ultrasound_hwt.shape}"
                )
            ultrasound = ultrasound_hwt[:, :, 0]
            raw_labels, converted_labels = raw_label_slice(unwrapped, raw_volume)
            lumbar_mask = np.isin(raw_labels, list(LUMBAR_LABELS))
            mismatch = int(np.count_nonzero(lumbar_mask & (converted_labels != 13)))
            raw_conversion_mismatches += mismatch
            counts = {str(label): int(np.count_nonzero(raw_labels == label)) for label in RAW_LABEL_NAMES}
            for label, name in RAW_LABEL_NAMES.items():
                if counts[str(label)] > 0:
                    level_visible_frames[name] += 1
            image_valid = bool(
                np.all(np.isfinite(ultrasound))
                and float(np.std(ultrasound)) > 1e-8
                and not bool(status["native_no_collide"])
            )
            frame_hash = ultrasound_sha256(ultrasound)
            actor_frame = ActorFrame(
                session_id=str(session["session_id"]),
                frame_sequence=frame_sequence,
                producer_sequence=int(unwrapped.num_step),
                frame_sha256=frame_hash,
                ultrasound=ultrasound,
                requested_action=requested_action,
                executed_delta_home_mm=delta_home_mm,
                executed_delta_angle_rad=delta_angle,
                action_accepted=transition_valid,
                action_completed=transition_valid,
                action_clamped=bool(status["command_clamped"]),
                contact_ok=bool(status["surface_proxy_ok"]),
                workspace_ok=not bool(status["command_clamped"]),
                image_valid=image_valid,
                home_prior_id=f"workspace-grid-v1-x{int(session['patient_x_index'])}",
            )
            freshness.accept(actor_frame)
            direction_counts[phase] += 1
            public_row = actor_frame.public_metadata()
            public_row.update(
                {
                    "split": session["split"],
                    "phase": phase,
                    "image_archive": "public/ultrasound.npz",
                    "image_archive_index": frame_sequence,
                    "relative_position_home_mm": odometry.position_mm.tolist(),
                    "relative_angle_home_rad": odometry.angle_rad,
                    "native_no_collide": bool(status["native_no_collide"]),
                    "action_steps_since_previous_frame": int(
                        status.get("action_steps_since_previous_frame", 0)
                    ),
                    "requested_action_sequence": status.get(
                        "requested_action_sequence", []
                    ),
                }
            )
            goal = tensor_row(unwrapped.goal_cmd_pose)
            position_error_mm = float(np.linalg.norm(current_pose[:2] - goal[:2]) * 1.5)
            orientation_error_deg = abs(float(wrap_to_pi(current_pose[2] - goal[2]))) * 180.0 / math.pi
            adjacent_counts = {label: counts[str(label)] for label in ADJACENT_NEGATIVE_LABELS}
            adjacent_label = max(adjacent_counts, key=adjacent_counts.get)
            privileged_rows.append(
                {
                    "schema": "sonogym-two-stage-privileged-annotation-v1",
                    "session_id": session["session_id"],
                    "frame_sequence": frame_sequence,
                    "frame_sha256": frame_hash,
                    "patient_id": "s0010",
                    "anatomy_id": "vertebrae_L4",
                    "command_pose_index": current_command(unwrapped).tolist(),
                    "realized_pose_index": current_pose.tolist(),
                    "goal_cmd_pose": goal.tolist(),
                    "l4_raw_label": 28,
                    "raw_label_pixel_counts": counts,
                    "l4_visible_fraction": counts["28"] / raw_labels.size,
                    "lumbar_interface_visible": bool(sum(counts.values()) > 0),
                    "adjacent_level": RAW_LABEL_NAMES[adjacent_label]
                    if adjacent_counts[adjacent_label] > 0
                    else None,
                    "adjacent_negative_without_l4": bool(
                        adjacent_counts[adjacent_label] > 0 and counts["28"] == 0
                    ),
                    "position_error_mm": position_error_mm,
                    "orientation_error_deg": orientation_error_deg,
                    "raw_label_archive": "privileged/raw_labels.npz",
                    "raw_label_archive_index": frame_sequence,
                    "raw_to_acoustic_bone_mismatch_pixels": mismatch,
                }
            )
            public_rows.append(public_row)
            ultrasound_frames.append(ultrasound.copy())
            raw_label_frames.append(raw_labels.copy())
            previous_realized = current_pose

        baseline_status: dict[str, object] = {
            "finite": True,
            "command_clamped": False,
            "command_mapping_valid": True,
            "command_progressed": True,
            "surface_proxy_ok": surface_proxy_ok(unwrapped),
            "native_no_collide": bool(unwrapped.US_slicer.no_collide[0].item()),
            "terminated": False,
            "truncated": False,
            "action_steps_since_previous_frame": 0,
            "requested_action_sequence": [],
        }
        capture(0, "HOME_BASELINE", np.zeros(3, dtype=np.float64), baseline_status)
        frame_sequence = 1
        scan = protocol["geometry"]["longitudinal_scan"]
        z_start = int(scan["start_z_index"])
        z_end = int(scan["end_z_index"])
        paths = (
            ("OUTBOUND_CAUDAL", range(z_start - 1, z_end - 1, -1)),
            ("RETURN_CRANIAL", range(z_end + 1, z_start + 1)),
        )
        scan_action_steps = 0
        maximum_action_steps = int(scan["maximum_action_steps_per_session"])
        for phase, z_values in paths:
            for z_index in z_values:
                waypoint = np.asarray(
                    [session["patient_x_index"], float(z_index), session["angle_rad"]],
                    dtype=np.float64,
                )
                remaining_actions = maximum_action_steps - len(home_trajectory) - scan_action_steps
                if remaining_actions <= 0:
                    raise RuntimeError("frozen action-step budget exhausted before scan completion")
                waypoint_trajectory = move_to_waypoint(
                    env,
                    waypoint,
                    maximum_steps=min(8, remaining_actions),
                    action_rows=scan_action_rows,
                    phase=phase,
                    frame_sequence=frame_sequence,
                )
                scan_action_steps += len(waypoint_trajectory)
                status = dict(waypoint_trajectory[-1])
                status["action_steps_since_previous_frame"] = len(waypoint_trajectory)
                status["requested_action_sequence"] = [
                    item["normalized_action"] for item in waypoint_trajectory
                ]
                capture(
                    frame_sequence,
                    phase,
                    np.asarray(
                        waypoint_trajectory[0]["normalized_action"], dtype=np.float64
                    ),
                    status,
                )
                frame_sequence += 1

        if frame_sequence != int(scan["expected_frames_per_complete_session"]):
            raise RuntimeError(f"captured {frame_sequence} frames, expected {scan['expected_frames_per_complete_session']}")
        total_action_steps = len(home_trajectory) + scan_action_steps
        if total_action_steps > maximum_action_steps:
            raise RuntimeError(
                f"used {total_action_steps} action steps, exceeds frozen maximum "
                f"{maximum_action_steps}"
            )
        if raw_conversion_mismatches != 0:
            raise RuntimeError(f"raw/acoustic vertebral projection mismatch: {raw_conversion_mismatches}")

        np.savez_compressed(public_dir / "ultrasound.npz", ultrasound=np.stack(ultrasound_frames))
        np.savez_compressed(privileged_dir / "raw_labels.npz", raw_labels=np.stack(raw_label_frames))
        write_jsonl(public_dir / "frames.jsonl", public_rows)
        write_jsonl(privileged_dir / "annotations.jsonl", privileged_rows)
        write_json(
            privileged_dir / "home_return.json",
            {
                "reset_command": reset_command.tolist(),
                "reset_realized_pose": reset_realized.tolist(),
                "home_target_command": target.tolist(),
                "home_realized_pose": home_pose.tolist(),
                "steps": home_trajectory,
            },
        )
        summary.update(
            status="PASSED",
            frame_count=len(public_rows),
            home_return_steps=len(home_trajectory),
            scan_action_steps=scan_action_steps,
            scan_action_manifest="privileged/scan_actions.jsonl",
            total_action_steps=total_action_steps,
            direction_frame_counts=dict(direction_counts),
            level_visible_frame_counts=dict(level_visible_frames),
            l4_positive_frames=sum(
                int(row["raw_label_pixel_counts"]["28"] > 0)  # type: ignore[index]
                for row in privileged_rows
            ),
            adjacent_negative_without_l4_frames=sum(
                int(row["adjacent_negative_without_l4"]) for row in privileged_rows
            ),
            raw_to_acoustic_bone_mismatch_pixels=raw_conversion_mismatches,
            public_manifest="public/frames.jsonl",
            privileged_manifest="privileged/annotations.jsonl",
            ultrasound_archive="public/ultrasound.npz",
            raw_label_archive="privileged/raw_labels.npz",
        )
        return 0
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        return 1
    finally:
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        summary["home_return_action_count"] = len(home_trajectory)
        if home_trajectory:
            write_jsonl(privileged_dir / "home_actions.jsonl", home_trajectory)
        summary["scan_action_count"] = len(scan_action_rows)
        if scan_action_rows:
            write_jsonl(privileged_dir / "scan_actions.jsonl", scan_action_rows)
        write_json(summary_path, summary)
        if env is not None:
            env.close()


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)
