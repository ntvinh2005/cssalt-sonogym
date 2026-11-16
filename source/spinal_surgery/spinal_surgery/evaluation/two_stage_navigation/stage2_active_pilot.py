"""Pure planning helpers for the approved Stage 2 active-acquisition pilot."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import assert_no_privileged_fields


ACTIVE_PILOT_PROTOCOL_IDS = frozenset(
    {
        "sonogym-stage2-active-acquisition-pilot-v1",
        "sonogym-stage2-active-acquisition-pilot-v2",
        "sonogym-stage2-active-acquisition-pilot-v3",
    }
)


def phase1_handoff_pose(
    actor_result: Mapping[str, Any],
    *,
    session_id: str,
    home_z_index: float,
    label_resolution_mm: float,
    stage2_angle_rad: float,
) -> np.ndarray:
    """Convert one successful actor-only Phase 1 result into a Stage 2 start pose."""
    assert_no_privileged_fields(actor_result)
    if actor_result.get("schema") != "sonogym-stage1-confirmation-actor-v1":
        raise ValueError("unexpected Phase 1 actor result schema")
    if actor_result.get("training_or_threshold_selection_performed") is not False:
        raise ValueError("Phase 1 handoff must come from frozen inference")
    rows = [row for row in actor_result.get("results", ()) if row.get("session_id") == session_id]
    if len(rows) != 1:
        raise ValueError("Phase 1 handoff session is missing or duplicated")
    row = rows[0]
    if row.get("actor_status") != "SUCCESS" or row.get("actor_coordinate_mm") is None:
        raise ValueError("Phase 1 handoff did not emit a target")
    home = row.get("declared_home")
    if not isinstance(home, Mapping):
        raise ValueError("Phase 1 declared Home is missing")
    values = (
        float(home["patient_x_index"]),
        float(home_z_index),
        float(row["actor_coordinate_mm"]),
        float(label_resolution_mm),
        float(stage2_angle_rad),
    )
    if not all(math.isfinite(value) for value in values) or label_resolution_mm <= 0.0:
        raise ValueError("Phase 1 handoff values must be finite")
    return np.asarray(
        [
            values[0],
            values[1] + values[2] / values[3],
            values[4],
        ],
        dtype=np.float64,
    )


def offset_poses(center: Sequence[float], offsets: Sequence[Sequence[float]]) -> tuple[np.ndarray, ...]:
    base = np.asarray(center, dtype=np.float64)
    if base.shape != (3,) or not np.all(np.isfinite(base)):
        raise ValueError("center must contain three finite values")
    result: list[np.ndarray] = []
    for offset in offsets:
        delta = np.asarray(offset, dtype=np.float64)
        if delta.shape != (3,) or not np.all(np.isfinite(delta)):
            raise ValueError("each offset must contain three finite values")
        result.append(base + delta)
    return tuple(result)


def serpentine_offsets(
    x_values: Sequence[float], z_values: Sequence[float], angle_values: Sequence[float]
) -> tuple[tuple[float, float, float], ...]:
    if not x_values or not z_values or not angle_values:
        raise ValueError("search axes must be nonempty")
    rows: list[tuple[float, float, float]] = []
    x_axis = [float(value) for value in x_values]
    z_axis = [float(value) for value in z_values]
    angle_axis = [float(value) for value in angle_values]
    if not all(math.isfinite(value) for value in (*x_axis, *z_axis, *angle_axis)):
        raise ValueError("search axes must be finite")
    for z_index, z_value in enumerate(z_axis):
        x_order = x_axis if z_index % 2 == 0 else list(reversed(x_axis))
        for x_index, x_value in enumerate(x_order):
            angles = angle_axis if (z_index + x_index) % 2 == 0 else list(reversed(angle_axis))
            rows.extend((x_value, z_value, angle) for angle in angles)
    return tuple(rows)


def relative_pose_from_start(
    realized_pose: Sequence[float], start_pose: Sequence[float], label_resolution_mm: float
) -> tuple[float, float, float]:
    realized = np.asarray(realized_pose, dtype=np.float64)
    start = np.asarray(start_pose, dtype=np.float64)
    if realized.shape != (3,) or start.shape != (3,):
        raise ValueError("poses must contain three values")
    if not np.all(np.isfinite(realized)) or not np.all(np.isfinite(start)):
        raise ValueError("poses must be finite")
    if not math.isfinite(label_resolution_mm) or label_resolution_mm <= 0.0:
        raise ValueError("label resolution must be finite and positive")
    angle = (float(realized[2] - start[2]) + math.pi) % (2.0 * math.pi) - math.pi
    return (
        float((realized[0] - start[0]) * label_resolution_mm),
        float((realized[1] - start[1]) * label_resolution_mm),
        angle,
    )


def transition_is_valid(action_rows: Sequence[Mapping[str, Any]]) -> bool:
    """Accept angle-only saturation but reject every position or safety failure."""
    if not action_rows:
        return False
    required = {
        "finite",
        "surface_proxy_ok",
        "command_mapping_valid",
        "terminated",
        "truncated",
        "command_position_clamped",
    }
    return bool(
        all(required <= set(row) for row in action_rows)
        and all(
            bool(row["finite"])
            and bool(row["surface_proxy_ok"])
            and bool(row["command_mapping_valid"])
            and not bool(row["terminated"])
            and not bool(row["truncated"])
            and not bool(row["command_position_clamped"])
            for row in action_rows
        )
    )


def validate_confirmation_motion(
    dither_offsets: Sequence[Sequence[float]],
    *,
    label_resolution_mm: float,
    position_tolerance_mm: float,
) -> tuple[float, ...]:
    """Require every confirmation dither to exceed its dedicated motion tolerance."""
    if not math.isfinite(label_resolution_mm) or label_resolution_mm <= 0.0:
        raise ValueError("label resolution must be finite and positive")
    if not math.isfinite(position_tolerance_mm) or position_tolerance_mm <= 0.0:
        raise ValueError("confirmation position tolerance must be finite and positive")
    if not dither_offsets:
        raise ValueError("at least one confirmation dither is required")
    distances: list[float] = []
    for offset in dither_offsets:
        delta = np.asarray(offset, dtype=np.float64)
        if delta.shape != (3,) or not np.all(np.isfinite(delta)):
            raise ValueError("each confirmation dither must contain three finite values")
        distance_mm = float(np.linalg.norm(delta[:2]) * label_resolution_mm)
        if distance_mm <= position_tolerance_mm:
            raise ValueError(
                "confirmation dither displacement must exceed its movement tolerance"
            )
        distances.append(distance_mm)
    return tuple(distances)
