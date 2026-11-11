"""Public actor and privileged annotation contracts for the two-stage method.

The actor contract is intentionally narrow. Simulator state needed for annotation or
independent evaluation must be serialized through :class:`PrivilegedAnnotation`, never added
to an actor payload for convenience.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


ACTOR_SCHEMA = "sonogym-two-stage-actor-frame-v1"
PRIVILEGED_SCHEMA = "sonogym-two-stage-privileged-annotation-v1"

ACTOR_FIELDS = frozenset(
    {
        "schema",
        "session_id",
        "frame_sequence",
        "producer_sequence",
        "frame_sha256",
        "ultrasound",
        "requested_action",
        "executed_delta_home_mm",
        "executed_delta_angle_rad",
        "action_accepted",
        "action_completed",
        "action_clamped",
        "contact_ok",
        "workspace_ok",
        "image_valid",
        "home_prior_id",
    }
)

# Exact forbidden names are checked recursively. Substring matching is intentionally avoided:
# a benign provenance value such as ``target_free`` must not be rejected accidentally.
FORBIDDEN_ACTOR_FIELDS = frozenset(
    {
        "patient_id",
        "anatomy_id",
        "raw_or_converted_segmentation",
        "label_image",
        "segmentation",
        "vertebra_mesh_identity",
        "vertebra_mesh_id",
        "absolute_patient_pose",
        "cur_cmd_pose",
        "command_pose_index",
        "goal_cmd_pose",
        "true_l4_target_pose",
        "target_position",
        "target_pose",
        "target_distance",
        "target_anatomy",
        "reward",
        "evaluator_score",
        "phase5a_geometric_errors",
        "l4_raw_label",
        "world_pose",
    }
)


def _finite_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def ultrasound_sha256(image: np.ndarray) -> str:
    """Hash canonical little-endian float32 pixels and their shape."""
    array = np.asarray(image, dtype="<f4")
    if array.ndim != 2:
        raise ValueError(f"ultrasound must be 2-D, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("ultrasound contains nonfinite values")
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _walk_mapping(value: Any, path: str = "payload") -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_string = str(key)
            nested_path = f"{path}.{key_string}"
            keys.append((key_string, nested_path))
            keys.extend(_walk_mapping(nested, nested_path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            keys.extend(_walk_mapping(nested, f"{path}[{index}]"))
    return keys


def assert_no_privileged_fields(payload: Mapping[str, Any]) -> None:
    """Fail closed if a privileged field is present anywhere in an actor payload."""
    violations = [path for key, path in _walk_mapping(payload) if key in FORBIDDEN_ACTOR_FIELDS]
    if violations:
        raise ValueError("privileged actor field(s): " + ", ".join(sorted(violations)))


def validate_actor_mapping(payload: Mapping[str, Any]) -> None:
    """Validate the top-level actor schema and reject undeclared fields."""
    assert_no_privileged_fields(payload)
    keys = set(payload)
    missing = ACTOR_FIELDS - keys
    extra = keys - ACTOR_FIELDS
    if missing or extra:
        raise ValueError(f"actor schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    if payload["schema"] != ACTOR_SCHEMA:
        raise ValueError(f"unexpected actor schema {payload['schema']!r}")


@dataclass(frozen=True)
class ActorFrame:
    """One actor-visible ultrasound acquisition paired with executed relative motion."""

    session_id: str
    frame_sequence: int
    producer_sequence: int
    frame_sha256: str
    ultrasound: np.ndarray
    requested_action: np.ndarray
    executed_delta_home_mm: np.ndarray
    executed_delta_angle_rad: float
    action_accepted: bool
    action_completed: bool
    action_clamped: bool
    contact_ok: bool
    workspace_ok: bool
    image_valid: bool
    home_prior_id: str
    schema: str = ACTOR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ACTOR_SCHEMA:
            raise ValueError(f"unexpected actor schema {self.schema!r}")
        if not self.session_id or not self.home_prior_id:
            raise ValueError("session_id and home_prior_id must be nonempty")
        if self.frame_sequence < 0 or self.producer_sequence < 0:
            raise ValueError("frame sequences must be nonnegative")
        image = np.asarray(self.ultrasound, dtype=np.float32)
        if image.shape != (200, 150):
            raise ValueError(f"ultrasound must have SonoGym shape (200, 150), got {image.shape}")
        if not np.all(np.isfinite(image)):
            raise ValueError("ultrasound contains nonfinite values")
        expected_hash = ultrasound_sha256(image)
        if self.frame_sha256 != expected_hash:
            raise ValueError("frame_sha256 does not match canonical ultrasound pixels")
        _finite_vector(self.requested_action, (3,), "requested_action")
        _finite_vector(self.executed_delta_home_mm, (2,), "executed_delta_home_mm")
        if not math.isfinite(float(self.executed_delta_angle_rad)):
            raise ValueError("executed_delta_angle_rad must be finite")

    @property
    def transition_valid(self) -> bool:
        return bool(
            self.action_accepted
            and self.action_completed
            and not self.action_clamped
            and self.contact_ok
            and self.workspace_ok
        )

    def public_metadata(self) -> dict[str, Any]:
        """Return JSON-safe metadata without copying the image into a manifest."""
        payload = {
            "schema": self.schema,
            "session_id": self.session_id,
            "frame_sequence": self.frame_sequence,
            "producer_sequence": self.producer_sequence,
            "frame_sha256": self.frame_sha256,
            "requested_action": np.asarray(self.requested_action, dtype=float).tolist(),
            "executed_delta_home_mm": np.asarray(
                self.executed_delta_home_mm, dtype=float
            ).tolist(),
            "executed_delta_angle_rad": float(self.executed_delta_angle_rad),
            "action_accepted": bool(self.action_accepted),
            "action_completed": bool(self.action_completed),
            "action_clamped": bool(self.action_clamped),
            "contact_ok": bool(self.contact_ok),
            "workspace_ok": bool(self.workspace_ok),
            "image_valid": bool(self.image_valid),
            "home_prior_id": self.home_prior_id,
        }
        assert_no_privileged_fields(payload)
        return payload


@dataclass(frozen=True)
class PrivilegedAnnotation:
    """Offline-only anatomy and evaluator fields keyed to an immutable actor frame."""

    session_id: str
    frame_sequence: int
    frame_sha256: str
    patient_id: str
    anatomy_id: str
    command_pose_index: np.ndarray
    l4_visible_fraction: float
    lumbar_interface_visible: bool
    adjacent_level: str | None
    position_error_mm: float
    orientation_error_deg: float
    schema: str = PRIVILEGED_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PRIVILEGED_SCHEMA:
            raise ValueError(f"unexpected privileged schema {self.schema!r}")
        if not self.session_id or not self.patient_id or not self.anatomy_id:
            raise ValueError("privileged identity fields must be nonempty")
        if self.frame_sequence < 0 or len(self.frame_sha256) != 64:
            raise ValueError("invalid privileged frame identity")
        _finite_vector(self.command_pose_index, (3,), "command_pose_index")
        if not 0.0 <= float(self.l4_visible_fraction) <= 1.0:
            raise ValueError("l4_visible_fraction must be in [0, 1]")
        for name, value in (
            ("position_error_mm", self.position_error_mm),
            ("orientation_error_deg", self.orientation_error_deg),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @property
    def join_key(self) -> tuple[str, int, str]:
        return self.session_id, self.frame_sequence, self.frame_sha256


class FrameFreshnessGuard:
    """Enforce monotonic acquisition identity without assuming pixels must change.

    A freshly rendered deterministic frame can legitimately have the same pixel hash as its
    predecessor. Freshness is therefore defined by both acquisition and producer sequences;
    the hash protects identity and data integrity, not simulator time.
    """

    def __init__(self) -> None:
        self._last_frame_sequence: int | None = None
        self._last_producer_sequence: int | None = None

    def accept(self, frame: ActorFrame) -> None:
        if self._last_frame_sequence is not None:
            if frame.frame_sequence <= self._last_frame_sequence:
                raise ValueError("stale or nonmonotonic frame_sequence")
            if frame.producer_sequence <= self._last_producer_sequence:  # type: ignore[operator]
                raise ValueError("stale or nonmonotonic producer_sequence")
        self._last_frame_sequence = frame.frame_sequence
        self._last_producer_sequence = frame.producer_sequence
