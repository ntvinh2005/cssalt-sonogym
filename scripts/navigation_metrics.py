#!/usr/bin/env python3
"""Pure evaluation metrics for the SonoGym navigation task.

This module intentionally has no Isaac Sim or torch dependency so coordinate,
unit, success, and episode-accounting behavior can be tested on a CPU login
node before a GPU is requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SuccessThresholds:
    position_mm: float = 10.0
    orientation_deg: float = 10.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.position_mm) or self.position_mm < 0:
            raise ValueError("position threshold must be finite and nonnegative")
        if not np.isfinite(self.orientation_deg) or self.orientation_deg < 0:
            raise ValueError("orientation threshold must be finite and nonnegative")


def _finite_array(value: Any, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"expected shape {shape}; received {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("metric inputs must be finite")
    return array


def meters_to_millimeters(value: Any) -> np.ndarray:
    """Convert finite distances from metres to millimetres."""

    return _finite_array(value) * 1000.0


def world_to_patient_points(
    world_points_m: Any,
    patient_origin_world_m: Any,
    patient_to_world_rotation: Any,
) -> np.ndarray:
    """Express world-frame point(s) in a rigid patient frame."""

    points = _finite_array(world_points_m)
    origin = _finite_array(patient_origin_world_m, shape=(3,))
    rotation = _finite_array(patient_to_world_rotation, shape=(3, 3))
    if points.shape[-1:] != (3,):
        raise ValueError("world points must end in a three-vector")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-7):
        raise ValueError("patient-to-world rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-7):
        raise ValueError("patient-to-world rotation must be right-handed")
    return (points - origin) @ rotation


def frontal_plane_position_error_mm(
    current_patient_position_m: Any,
    target_patient_position_m: Any,
    *,
    frontal_axes: tuple[int, int] = (0, 2),
) -> float:
    """Return Euclidean error after patient-frontal-plane projection."""

    current = _finite_array(current_patient_position_m, shape=(3,))
    target = _finite_array(target_patient_position_m, shape=(3,))
    if len(set(frontal_axes)) != 2 or any(
        axis not in (0, 1, 2) for axis in frontal_axes
    ):
        raise ValueError("frontal_axes must contain two distinct 3-D axes")
    error_m = np.linalg.norm((current - target)[list(frontal_axes)])
    return float(meters_to_millimeters(error_m))


def wrap_to_pi(angle_rad: Any) -> np.ndarray:
    """Wrap finite angle(s) to the half-open interval [-pi, pi)."""

    angle = _finite_array(angle_rad)
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def orientation_error_deg(current_rad: float, target_rad: float) -> float:
    """Return absolute wrapped angular error in degrees."""

    current = float(_finite_array(current_rad))
    target = float(_finite_array(target_rad))
    return float(abs(wrap_to_pi(current - target)) * 180.0 / np.pi)


def command_pose_errors(
    current_pose: Any,
    target_pose: Any,
    *,
    label_resolution_m: float,
) -> tuple[float, float]:
    """Compute errors from task-native ``[x_index, z_index, angle_rad]``."""

    current = _finite_array(current_pose, shape=(3,))
    target = _finite_array(target_pose, shape=(3,))
    resolution = float(_finite_array(label_resolution_m))
    if resolution <= 0:
        raise ValueError("label resolution must be positive")
    position_mm = float(
        np.linalg.norm(current[:2] - target[:2]) * resolution * 1000.0
    )
    angle_deg = orientation_error_deg(current[2], target[2])
    return position_mm, angle_deg


def is_success(
    position_error_mm: float,
    orientation_error_deg_value: float,
    thresholds: SuccessThresholds,
) -> bool:
    """Require position and orientation thresholds simultaneously."""

    position = float(_finite_array(position_error_mm))
    orientation = float(_finite_array(orientation_error_deg_value))
    return (
        position <= thresholds.position_mm
        and orientation <= thresholds.orientation_deg
    )


def classify_completion(
    *,
    terminated: bool,
    native_truncated: bool,
    protocol_horizon_reached: bool,
    out_of_bounds: bool,
) -> dict[str, bool | str]:
    """Keep native termination distinct from the 300-step protocol horizon."""

    if out_of_bounds and not terminated:
        raise ValueError("out_of_bounds cannot be true without termination")
    protocol_truncated = bool(
        protocol_horizon_reached and not terminated and not native_truncated
    )
    if out_of_bounds:
        reason = "out_of_bounds"
    elif terminated:
        reason = "native_termination"
    elif native_truncated:
        reason = "native_time_limit"
    elif protocol_truncated:
        reason = "protocol_horizon"
    else:
        reason = "in_progress"
    return {
        "terminated": bool(terminated),
        "native_truncated": bool(native_truncated),
        "protocol_truncated": protocol_truncated,
        "out_of_bounds": bool(out_of_bounds),
        "completion_reason": reason,
    }


class EpisodeMetricTracker:
    """Accumulate success and validity diagnostics for one evaluated episode."""

    def __init__(
        self,
        initial_pose: Any,
        target_pose: Any,
        *,
        label_resolution_m: float,
        primary: SuccessThresholds = SuccessThresholds(),
        secondary: SuccessThresholds = SuccessThresholds(5.0, 5.0),
    ) -> None:
        self.initial_pose = _finite_array(initial_pose, shape=(3,)).copy()
        self.target_pose = _finite_array(target_pose, shape=(3,)).copy()
        self.label_resolution_m = float(_finite_array(label_resolution_m))
        if self.label_resolution_m <= 0:
            raise ValueError("label resolution must be positive")
        self.primary = primary
        self.secondary = secondary
        self.initial_position_mm, self.initial_orientation_deg = command_pose_errors(
            self.initial_pose,
            self.target_pose,
            label_resolution_m=self.label_resolution_m,
        )
        self.final_pose = self.initial_pose.copy()
        self.final_position_mm = self.initial_position_mm
        self.final_orientation_deg = self.initial_orientation_deg
        self.total_reward = 0.0
        self.episode_length = 0
        self.ever_success = is_success(
            self.initial_position_mm, self.initial_orientation_deg, self.primary
        )
        self.ever_success_5mm_5deg = is_success(
            self.initial_position_mm, self.initial_orientation_deg, self.secondary
        )
        self.first_successful_step = 0 if self.ever_success else None
        self.first_successful_step_5mm_5deg = (
            0 if self.ever_success_5mm_5deg else None
        )
        self.nonfinite_observations = 0
        self.invalid_transitions = 0

    def record(
        self,
        *,
        step: int,
        pose: Any,
        reward: float,
        observation_finite: bool,
        transition_finite: bool,
    ) -> tuple[float, float, bool, bool]:
        if step != self.episode_length + 1:
            raise ValueError("episode steps must be positive and consecutive")
        try:
            pose_array = _finite_array(pose, shape=(3,))
            reward_value = float(_finite_array(reward))
            position_mm, orientation_deg = command_pose_errors(
                pose_array,
                self.target_pose,
                label_resolution_m=self.label_resolution_m,
            )
        except ValueError:
            self.invalid_transitions += 1
            raise
        if not observation_finite:
            self.nonfinite_observations += 1
        if not transition_finite:
            self.invalid_transitions += 1
        self.final_pose = pose_array.copy()
        self.final_position_mm = position_mm
        self.final_orientation_deg = orientation_deg
        self.total_reward += reward_value
        self.episode_length = step
        primary_success = is_success(position_mm, orientation_deg, self.primary)
        secondary_success = is_success(position_mm, orientation_deg, self.secondary)
        if primary_success and not self.ever_success:
            self.ever_success = True
            self.first_successful_step = step
        if secondary_success and not self.ever_success_5mm_5deg:
            self.ever_success_5mm_5deg = True
            self.first_successful_step_5mm_5deg = step
        return position_mm, orientation_deg, primary_success, secondary_success

    def finalize(
        self,
        *,
        terminated: bool,
        native_truncated: bool,
        protocol_horizon_reached: bool,
        out_of_bounds: bool,
    ) -> dict[str, Any]:
        completion = classify_completion(
            terminated=terminated,
            native_truncated=native_truncated,
            protocol_horizon_reached=protocol_horizon_reached,
            out_of_bounds=out_of_bounds,
        )
        return {
            "initial_pose": self.initial_pose.tolist(),
            "target_pose": self.target_pose.tolist(),
            "final_pose": self.final_pose.tolist(),
            "initial_position_error_mm": self.initial_position_mm,
            "initial_orientation_error_deg": self.initial_orientation_deg,
            "final_position_error_mm": self.final_position_mm,
            "final_orientation_error_deg": self.final_orientation_deg,
            "final_success": is_success(
                self.final_position_mm, self.final_orientation_deg, self.primary
            ),
            "ever_reached_success": self.ever_success,
            "first_successful_step": self.first_successful_step,
            "final_success_5mm_5deg": is_success(
                self.final_position_mm, self.final_orientation_deg, self.secondary
            ),
            "ever_reached_success_5mm_5deg": self.ever_success_5mm_5deg,
            "first_successful_step_5mm_5deg": self.first_successful_step_5mm_5deg,
            "total_reward": self.total_reward,
            "episode_length": self.episode_length,
            "nonfinite_observations": self.nonfinite_observations,
            "invalid_transitions": self.invalid_transitions,
            **completion,
        }
