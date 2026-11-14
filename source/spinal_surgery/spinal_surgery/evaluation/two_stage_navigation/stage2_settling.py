"""Evaluation of a bounded physical orientation-convergence check."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SettlingCriteria:
    required_frame_count: int = 10
    maximum_command_position_error_mm: float = 5.0
    maximum_command_orientation_error_deg: float = 1.0
    maximum_goal_position_error_mm: float = 10.0
    maximum_goal_orientation_error_deg: float = 10.0
    minimum_l4_visible_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.required_frame_count <= 0:
            raise ValueError("required_frame_count must be positive")
        values = (
            self.maximum_command_position_error_mm,
            self.maximum_command_orientation_error_deg,
            self.maximum_goal_position_error_mm,
            self.maximum_goal_orientation_error_deg,
            self.minimum_l4_visible_fraction,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("criteria must be finite and positive")


def evaluate_settled_frames(
    rows: Iterable[dict[str, Any]], criteria: SettlingCriteria
) -> dict[str, Any]:
    records = list(rows)
    required = criteria.required_frame_count
    producer_sequences = [int(row["producer_sequence"]) for row in records]
    fresh_sequence = all(
        later > earlier
        for earlier, later in zip(producer_sequences, producer_sequences[1:])
    )
    finite = all(
        all(
            math.isfinite(float(row[key]))
            for key in (
                "command_position_error_mm",
                "command_orientation_error_deg",
                "goal_position_error_mm",
                "goal_orientation_error_deg",
                "l4_visible_fraction",
            )
        )
        for row in records
    )
    technical_count = sum(bool(row["technical_image_valid"]) for row in records)
    command_count = sum(
        float(row["command_position_error_mm"])
        <= criteria.maximum_command_position_error_mm
        and float(row["command_orientation_error_deg"])
        <= criteria.maximum_command_orientation_error_deg
        for row in records
    )
    goal_count = sum(
        float(row["goal_position_error_mm"])
        <= criteria.maximum_goal_position_error_mm
        and float(row["goal_orientation_error_deg"])
        <= criteria.maximum_goal_orientation_error_deg
        for row in records
    )
    visible_count = sum(
        float(row["l4_visible_fraction"])
        >= criteria.minimum_l4_visible_fraction
        for row in records
    )
    joint_count = sum(
        bool(row["technical_image_valid"])
        and float(row["command_position_error_mm"])
        <= criteria.maximum_command_position_error_mm
        and float(row["command_orientation_error_deg"])
        <= criteria.maximum_command_orientation_error_deg
        and float(row["goal_position_error_mm"])
        <= criteria.maximum_goal_position_error_mm
        and float(row["goal_orientation_error_deg"])
        <= criteria.maximum_goal_orientation_error_deg
        and float(row["l4_visible_fraction"])
        >= criteria.minimum_l4_visible_fraction
        for row in records
    )
    passed = bool(
        finite
        and fresh_sequence
        and len(records) == required
        and technical_count == required
        and command_count == required
        and goal_count == required
        and visible_count == required
        and joint_count == required
    )
    return {
        "decision": "PASS" if passed else "FAIL",
        "frame_count": len(records),
        "finite": finite,
        "producer_sequence_strictly_increasing": fresh_sequence,
        "technical_image_valid_count": technical_count,
        "command_pose_pass_count": command_count,
        "goal_pose_pass_count": goal_count,
        "l4_visibility_pass_count": visible_count,
        "joint_pass_count": joint_count,
        "maximum_command_orientation_error_deg": (
            max(float(row["command_orientation_error_deg"]) for row in records)
            if records
            else None
        ),
        "maximum_goal_orientation_error_deg": (
            max(float(row["goal_orientation_error_deg"]) for row in records)
            if records
            else None
        ),
        "minimum_l4_visible_fraction": (
            min(float(row["l4_visible_fraction"]) for row in records)
            if records
            else None
        ),
    }
