"""Pure checks for synchronized ultrasound acquisition and physical viewpoint records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


PRIVILEGED_KEYS = frozenset(
    {
        "human_img_coords",
        "raw_labels",
        "segmentation",
        "world_to_human_pos",
        "world_to_ee_pos",
        "human_to_ee_pos",
        "goal_cmd_pose",
        "target_coordinates",
    }
)


def immutable_array(value: object) -> np.ndarray:
    result = np.array(value, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("snapshot arrays must be finite")
    result.setflags(write=False)
    return result


def validate_waypoint_bank(
    waypoints: Sequence[Sequence[float]],
    *,
    command_minimum: Sequence[float],
    command_maximum: Sequence[float],
    label_resolution_mm: float,
    position_tolerance_mm: float,
    orientation_tolerance_deg: float,
    separation_multiplier: float = 2.0,
) -> np.ndarray:
    requested = np.asarray(waypoints, dtype=np.float64)
    minimum = np.asarray(command_minimum, dtype=np.float64)
    maximum = np.asarray(command_maximum, dtype=np.float64)
    if requested.ndim != 2 or requested.shape[1] != 3 or not np.all(np.isfinite(requested)):
        raise ValueError("waypoints must be a finite Nx3 array")
    if minimum.shape != (3,) or maximum.shape != (3,) or np.any(minimum > maximum):
        raise ValueError("command bounds must be valid length-three vectors")
    values = (
        label_resolution_mm,
        position_tolerance_mm,
        orientation_tolerance_deg,
        separation_multiplier,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("spacing inputs must be finite and positive")
    resolved = np.clip(requested, minimum, maximum)
    if len(resolved) != len(np.unique(np.round(resolved, 9), axis=0)):
        raise ValueError("clipping produced duplicate waypoints")
    for index, (earlier, later) in enumerate(zip(resolved, resolved[1:]), start=1):
        translation_mm = float(np.linalg.norm(later[:2] - earlier[:2]) * label_resolution_mm)
        angle_deg = abs(float((later[2] - earlier[2] + np.pi) % (2 * np.pi) - np.pi))
        angle_deg *= 180.0 / np.pi
        translation_clear = translation_mm > separation_multiplier * position_tolerance_mm
        angle_clear = angle_deg > separation_multiplier * orientation_tolerance_deg
        if not (translation_clear or angle_clear):
            raise ValueError(f"waypoint {index} is swallowed by movement tolerances")
    return resolved


def validate_segment_action_budget(
    *,
    expected_frame_count: int,
    trajectory_start_frames: Sequence[int],
    maximum_initial_positioning_steps: int,
    maximum_action_steps_per_waypoint: int,
    maximum_episode_steps: int,
) -> int:
    """Return the largest declared segment budget after checking the episode horizon."""
    values = (
        expected_frame_count,
        maximum_initial_positioning_steps,
        maximum_action_steps_per_waypoint,
        maximum_episode_steps,
    )
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("frame and action budgets must be positive integers")
    starts = tuple(int(value) for value in trajectory_start_frames)
    if (
        not starts
        or starts[0] != 0
        or tuple(sorted(set(starts))) != starts
        or starts[-1] >= expected_frame_count
    ):
        raise ValueError("trajectory starts must be sorted, unique, in range, and begin at zero")
    boundaries = (*starts, expected_frame_count)
    largest_segment = max(later - earlier for earlier, later in zip(boundaries, boundaries[1:]))
    maximum_actions = maximum_initial_positioning_steps + (
        largest_segment - 1
    ) * maximum_action_steps_per_waypoint
    if maximum_actions >= maximum_episode_steps:
        raise ValueError(
            f"segment action budget {maximum_actions} reaches episode horizon {maximum_episode_steps}"
        )
    return maximum_actions


@dataclass(frozen=True)
class AcquiredView:
    acquisition_id: str
    session_id: str
    reset_id: str
    acquisition_sequence: int
    producer_sequence: int
    realized_pose_index: np.ndarray
    frame_sha256: str
    action_count: int
    technical_image_valid: bool

    def __post_init__(self) -> None:
        if not self.acquisition_id or not self.session_id or not self.reset_id:
            raise ValueError("acquisition identity fields are required")
        if self.acquisition_sequence < 0 or self.producer_sequence < 0 or self.action_count < 0:
            raise ValueError("acquisition counters must be nonnegative")
        if len(self.frame_sha256) != 64:
            raise ValueError("frame hash must be SHA-256")
        object.__setattr__(self, "realized_pose_index", immutable_array(self.realized_pose_index))
        if self.realized_pose_index.shape != (3,):
            raise ValueError("realized pose must have shape (3,)")


def validate_new_view(
    previous: AcquiredView,
    current: AcquiredView,
    *,
    label_resolution_mm: float,
    minimum_translation_mm: float,
    minimum_orientation_deg: float,
) -> dict[str, float | bool]:
    if current.session_id != previous.session_id or current.reset_id != previous.reset_id:
        raise ValueError("viewpoint history cannot cross a session or reset")
    translation_mm = float(
        np.linalg.norm(current.realized_pose_index[:2] - previous.realized_pose_index[:2])
        * label_resolution_mm
    )
    orientation_deg = abs(
        float(
            (current.realized_pose_index[2] - previous.realized_pose_index[2] + np.pi)
            % (2 * np.pi)
            - np.pi
        )
    ) * 180.0 / np.pi
    sequence_advanced = bool(
        current.acquisition_sequence > previous.acquisition_sequence
        and current.producer_sequence > previous.producer_sequence
    )
    moved = bool(
        translation_mm >= minimum_translation_mm
        or orientation_deg >= minimum_orientation_deg
    )
    accepted = bool(
        current.action_count > 0
        and sequence_advanced
        and moved
        and current.technical_image_valid
    )
    return {
        "translation_mm": translation_mm,
        "orientation_deg": orientation_deg,
        "sequence_advanced": sequence_advanced,
        "physical_viewpoint_changed": moved,
        "accepted": accepted,
    }


def validate_acquisition_join(records: Sequence[Mapping[str, object]]) -> str:
    identities = {str(record.get("acquisition_id", "")) for record in records}
    if len(identities) != 1 or "" in identities:
        raise ValueError("frame, pose, labels and coordinates must share one acquisition identity")
    return identities.pop()


def assert_detector_input_is_actor_visible(record: Mapping[str, object]) -> None:
    leaked = sorted(PRIVILEGED_KEYS.intersection(record))
    if leaked:
        raise ValueError(f"privileged detector input fields: {leaked}")
