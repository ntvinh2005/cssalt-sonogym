"""Isaac-side motion and acquisition helpers for the bounded Stage 2 pilot."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from isaaclab.utils.math import matrix_from_quat

from spinal_surgery.evaluation.two_stage_navigation.coordinates import (
    assess_native_command,
    command_target_from_realized_error,
    command_with_replaced_angle,
    plan_native_action,
    plan_proportional_target_action,
    wrap_to_pi,
)


POSITION_MAPPING_TOLERANCE_INDEX = 0.15
ANGLE_MAPPING_TOLERANCE_RAD = 1e-4
POSITION_CORRECTION_MULTIPLIER = 2.5


def tensor_row(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().to(torch.float64).reshape(value.shape[0], -1)[0].numpy()


def current_rotation(unwrapped: Any) -> np.ndarray:
    quaternion = unwrapped.extras["human_to_ee_quat"]
    return matrix_from_quat(quaternion)[0].detach().cpu().to(torch.float64).numpy()


def current_command(unwrapped: Any) -> np.ndarray:
    return tensor_row(unwrapped.US_slicer.current_x_z_x_angle_cmd)


def current_realized_pose(unwrapped: Any) -> np.ndarray:
    return tensor_row(unwrapped.cur_cmd_pose)


def tracking_errors(unwrapped: Any, target: np.ndarray) -> tuple[float, float]:
    pose = current_realized_pose(unwrapped)
    position_mm = float(np.linalg.norm(pose[:2] - target[:2]) * 1.5)
    orientation_deg = abs(float(wrap_to_pi(pose[2] - target[2]))) * 180.0 / math.pi
    return position_mm, orientation_deg


def surface_proxy_ok(unwrapped: Any) -> bool:
    values = torch.cat(
        [
            unwrapped.US_slicer.target_position.reshape(-1),
            unwrapped.US_slicer.target_z_axis.reshape(-1),
        ]
    )
    return bool(torch.isfinite(values).all().item())


def take_step(env: Any, target: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    unwrapped = env.unwrapped
    before_command = current_command(unwrapped)
    before_realized = current_realized_pose(unwrapped)
    planned = plan_native_action(
        target[:2] - before_command[:2],
        float(wrap_to_pi(target[2] - before_command[2])),
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
    command_range = unwrapped.US_slicer.x_z_range.detach().cpu().to(torch.float64).numpy()
    assessment = assess_native_command(
        before_command,
        after_command,
        planned,
        target,
        command_range[0],
        command_range[1],
        position_mapping_tolerance_index=POSITION_MAPPING_TOLERANCE_INDEX,
        angle_mapping_tolerance_rad=ANGLE_MAPPING_TOLERANCE_RAD,
    )
    finite = bool(
        np.all(np.isfinite(after_command))
        and np.all(np.isfinite(after_realized))
        and bool(torch.isfinite(observation["policy"]).all().item())
        and bool(torch.isfinite(reward).all().item())
    )
    position_error_mm, orientation_error_deg = tracking_errors(unwrapped, target)
    status = {
        "normalized_action": planned.normalized_action.tolist(),
        "before_command": before_command.tolist(),
        "after_command": after_command.tolist(),
        "before_realized_pose": before_realized.tolist(),
        "after_realized_pose": after_realized.tolist(),
        "command_clamped": assessment.native_clamped,
        "command_position_clamped": assessment.position_clamped,
        "command_angle_clamped": assessment.angle_clamped,
        "command_position_mapping_error_index": assessment.position_mapping_error_index,
        "command_angle_mapping_error_rad": assessment.angle_mapping_error_rad,
        "command_mapping_valid": assessment.mapping_valid,
        "command_progressed": assessment.progressed_toward_target,
        "finite": finite,
        "surface_proxy_ok": surface_proxy_ok(unwrapped),
        "native_no_collide": bool(unwrapped.US_slicer.no_collide[0].item()),
        "target_position_error_mm": position_error_mm,
        "target_orientation_error_deg": orientation_error_deg,
        "terminated": bool(terminated[0].item()),
        "truncated": bool(truncated[0].item()),
        "info_keys": sorted(info),
    }
    return observation, status


def take_proportional_step(
    env: Any,
    target: np.ndarray,
    *,
    ratio: np.ndarray | tuple[float, float, float] = (0.02, 0.02, 0.6),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Take one coupled translation/rotation step toward a caller-supplied target."""
    unwrapped = env.unwrapped
    before_command = current_command(unwrapped)
    before_realized = current_realized_pose(unwrapped)
    planned = plan_proportional_target_action(
        before_realized,
        target,
        current_rotation(unwrapped),
        ratio=ratio,
        action_scale=tensor_row(unwrapped.action_scale),
    )
    action = torch.as_tensor(
        planned.normalized_action.reshape(1, 3),
        dtype=torch.float32,
        device=unwrapped.device,
    )
    observation, reward, terminated, truncated, info = env.step(action)
    after_command = current_command(unwrapped)
    after_realized = current_realized_pose(unwrapped)
    command_range = unwrapped.US_slicer.x_z_range.detach().cpu().to(torch.float64).numpy()
    assessment = assess_native_command(
        before_command,
        after_command,
        planned,
        target,
        command_range[0],
        command_range[1],
        position_mapping_tolerance_index=POSITION_MAPPING_TOLERANCE_INDEX,
        angle_mapping_tolerance_rad=ANGLE_MAPPING_TOLERANCE_RAD,
    )
    finite = bool(
        np.all(np.isfinite(after_command))
        and np.all(np.isfinite(after_realized))
        and bool(torch.isfinite(observation["policy"]).all().item())
        and bool(torch.isfinite(reward).all().item())
    )
    position_error_mm, orientation_error_deg = tracking_errors(unwrapped, target)
    status = {
        "normalized_action": planned.normalized_action.tolist(),
        "controller_ratio": np.asarray(ratio, dtype=np.float64).tolist(),
        "before_command": before_command.tolist(),
        "after_command": after_command.tolist(),
        "before_realized_pose": before_realized.tolist(),
        "after_realized_pose": after_realized.tolist(),
        "command_clamped": assessment.native_clamped,
        "command_position_clamped": assessment.position_clamped,
        "command_angle_clamped": assessment.angle_clamped,
        "command_position_mapping_error_index": assessment.position_mapping_error_index,
        "command_angle_mapping_error_rad": assessment.angle_mapping_error_rad,
        "command_mapping_valid": assessment.mapping_valid,
        "command_progressed": assessment.progressed_toward_target,
        "finite": finite,
        "surface_proxy_ok": surface_proxy_ok(unwrapped),
        "native_no_collide": bool(unwrapped.US_slicer.no_collide[0].item()),
        "target_position_error_mm": position_error_mm,
        "target_orientation_error_deg": orientation_error_deg,
        "terminated": bool(terminated[0].item()),
        "truncated": bool(truncated[0].item()),
        "info_keys": sorted(info),
    }
    return observation, status


def move_proportional_target(
    env: Any,
    controller_target: np.ndarray,
    acceptance_target: np.ndarray,
    *,
    maximum_steps: int,
    maximum_position_error_mm: float,
    maximum_orientation_error_deg: float,
    ratio: np.ndarray | tuple[float, float, float] = (0.02, 0.02, 0.6),
    trajectory_sink: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run bounded coupled feedback while allowing only the declared angle ceiling clamp.

    Command-space distance is diagnostic only: this controller acts from realized-pose error,
    and robot tracking lag can temporarily move the internal command away from the target while
    the realized pose converges. The bounded loop and physical validity checks remain fail-closed.
    """
    trajectory: list[dict[str, Any]] = []
    for step in range(maximum_steps):
        position_error_mm, orientation_error_deg = tracking_errors(
            env.unwrapped, acceptance_target
        )
        if (
            position_error_mm <= maximum_position_error_mm
            and orientation_error_deg <= maximum_orientation_error_deg
        ):
            return trajectory
        _, status = take_proportional_step(env, controller_target, ratio=ratio)
        status["coupled_step"] = step + 1
        status["controller_target"] = controller_target.tolist()
        status["acceptance_target"] = acceptance_target.tolist()
        trajectory.append(status)
        if trajectory_sink is not None:
            trajectory_sink.append(status)
        if (
            not status["finite"]
            or status["command_position_clamped"]
            or not status["command_mapping_valid"]
            or not status["surface_proxy_ok"]
            or status["terminated"]
            or status["truncated"]
        ):
            raise RuntimeError(f"coupled physical movement failed: {status}")
    position_error_mm, orientation_error_deg = tracking_errors(
        env.unwrapped, acceptance_target
    )
    if (
        position_error_mm <= maximum_position_error_mm
        and orientation_error_deg <= maximum_orientation_error_deg
    ):
        return trajectory
    raise TimeoutError(
        f"coupled pose did not converge after {maximum_steps} steps: "
        f"position={position_error_mm}, orientation={orientation_error_deg}"
    )


def move_to_pose(
    env: Any,
    target: np.ndarray,
    *,
    maximum_steps: int,
    maximum_position_error_mm: float,
    maximum_orientation_error_deg: float,
) -> list[dict[str, Any]]:
    command_range = env.unwrapped.US_slicer.x_z_range.detach().cpu().to(torch.float64).numpy()
    trajectory: list[dict[str, Any]] = []
    for step in range(maximum_steps):
        position_error_mm, orientation_error_deg = tracking_errors(env.unwrapped, target)
        if (
            position_error_mm <= maximum_position_error_mm
            and orientation_error_deg <= maximum_orientation_error_deg
        ):
            return trajectory
        command_target = command_target_from_realized_error(
            current_realized_pose(env.unwrapped),
            target,
            position_multiplier=POSITION_CORRECTION_MULTIPLIER,
        )
        command_target = np.clip(command_target, command_range[0], command_range[1])
        _, status = take_step(env, command_target)
        status["movement_step"] = step + 1
        status["requested_target"] = target.tolist()
        trajectory.append(status)
        if (
            not status["finite"]
            or status["command_clamped"]
            or not status["command_mapping_valid"]
            or not status["command_progressed"]
            or not status["surface_proxy_ok"]
            or status["terminated"]
            or status["truncated"]
        ):
            raise RuntimeError(f"physical movement failed: {status}")
    position_error_mm, orientation_error_deg = tracking_errors(env.unwrapped, target)
    if (
        position_error_mm <= maximum_position_error_mm
        and orientation_error_deg <= maximum_orientation_error_deg
    ):
        return trajectory
    raise TimeoutError(
        f"pose did not converge after {maximum_steps} steps: "
        f"position={position_error_mm}, orientation={orientation_error_deg}"
    )


def move_orientation_only(
    env: Any,
    target_angle_rad: float,
    *,
    maximum_steps: int,
    maximum_orientation_error_deg: float,
) -> list[dict[str, Any]]:
    trajectory: list[dict[str, Any]] = []
    for step in range(maximum_steps):
        realized = current_realized_pose(env.unwrapped)
        orientation_error_deg = (
            abs(float(wrap_to_pi(realized[2] - target_angle_rad))) * 180.0 / math.pi
        )
        if orientation_error_deg <= maximum_orientation_error_deg:
            return trajectory
        target = command_with_replaced_angle(
            current_command(env.unwrapped), target_angle_rad
        )
        _, status = take_step(env, target)
        status["orientation_step"] = step + 1
        status["requested_angle_rad"] = target_angle_rad
        trajectory.append(status)
        if (
            not status["finite"]
            or status["command_clamped"]
            or not status["command_mapping_valid"]
            or not status["command_progressed"]
            or not status["surface_proxy_ok"]
            or status["terminated"]
            or status["truncated"]
        ):
            raise RuntimeError(f"physical orientation movement failed: {status}")
    realized = current_realized_pose(env.unwrapped)
    orientation_error_deg = (
        abs(float(wrap_to_pi(realized[2] - target_angle_rad))) * 180.0 / math.pi
    )
    if orientation_error_deg <= maximum_orientation_error_deg:
        return trajectory
    raise TimeoutError(
        f"orientation did not converge after {maximum_steps} steps: "
        f"orientation={orientation_error_deg}"
    )


def realized_feedback_target(unwrapped: Any, target: np.ndarray) -> np.ndarray:
    command_range = unwrapped.US_slicer.x_z_range.detach().cpu().to(torch.float64).numpy()
    corrected = command_target_from_realized_error(
        current_realized_pose(unwrapped),
        target,
        position_multiplier=POSITION_CORRECTION_MULTIPLIER,
    )
    return np.clip(corrected, command_range[0], command_range[1])


def clean_ultrasound(unwrapped: Any) -> np.ndarray:
    image = unwrapped.US_slicer.us_img_tensor[0].detach().cpu().to(torch.float32).numpy()
    if image.shape != (200, 150, 1):
        raise RuntimeError(f"unexpected ultrasound shape {image.shape}")
    result = image[:, :, 0]
    if not np.all(np.isfinite(result)):
        raise RuntimeError("nonfinite ultrasound")
    return result


def raw_label_slice(unwrapped: Any, raw_volume: np.ndarray) -> np.ndarray:
    coordinates = unwrapped.US_slicer.human_img_coords[0].detach().cpu().to(torch.int64).numpy()
    raw = raw_volume[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]
    width, depth = unwrapped.US_slicer.img_size
    return raw.reshape(width, depth, -1)[:, :, 0].T.astype(np.uint8, copy=False)
