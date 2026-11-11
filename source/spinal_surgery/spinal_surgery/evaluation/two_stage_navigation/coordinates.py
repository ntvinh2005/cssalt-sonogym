"""Coordinate transforms and fail-closed Home-relative odometry.

Patient command coordinates are expressed as label-volume indices. The actor sees only
Home-relative millimetres and executed angular deltas; absolute patient coordinates remain in
the motion/annotation boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


LABEL_RESOLUTION_MM = 1.5


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def wrap_to_pi(angle_rad: float | np.ndarray) -> float | np.ndarray:
    """Wrap angles to the half-open interval [-pi, pi)."""
    angle = np.asarray(angle_rad, dtype=np.float64)
    if not np.all(np.isfinite(angle)):
        raise ValueError("angle must be finite")
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return float(wrapped) if wrapped.ndim == 0 else wrapped


def indices_to_mm(indices: object, resolution_mm: float = LABEL_RESOLUTION_MM) -> np.ndarray:
    array = np.asarray(indices, dtype=np.float64)
    if not np.all(np.isfinite(array)) or not math.isfinite(resolution_mm) or resolution_mm <= 0:
        raise ValueError("indices and resolution_mm must be finite; resolution must be positive")
    return array * resolution_mm


def mm_to_indices(mm: object, resolution_mm: float = LABEL_RESOLUTION_MM) -> np.ndarray:
    array = np.asarray(mm, dtype=np.float64)
    if not np.all(np.isfinite(array)) or not math.isfinite(resolution_mm) or resolution_mm <= 0:
        raise ValueError("millimetres and resolution_mm must be finite; resolution must be positive")
    return array / resolution_mm


@dataclass(frozen=True)
class PatientFrameConvention:
    """Declared sign convention for a fixed anatomy pilot.

    The defaults encode the source-audited `s0010` hypothesis (+x right, +z cranial). A GPU
    geometry preflight must verify these signs before capture; other patients require their own
    explicit convention.
    """

    right_sign: int = 1
    cranial_sign: int = 1
    label_resolution_mm: float = LABEL_RESOLUTION_MM

    def __post_init__(self) -> None:
        if self.right_sign not in (-1, 1) or self.cranial_sign not in (-1, 1):
            raise ValueError("axis signs must be -1 or +1")
        if not math.isfinite(self.label_resolution_mm) or self.label_resolution_mm <= 0:
            raise ValueError("label_resolution_mm must be finite and positive")

    def patient_xz_to_home_mm(self, patient_xz_index: object, home_xz_index: object) -> np.ndarray:
        patient = _finite_array(patient_xz_index, (2,), "patient_xz_index")
        home = _finite_array(home_xz_index, (2,), "home_xz_index")
        delta_mm = (patient - home) * self.label_resolution_mm
        return np.asarray(
            [self.right_sign * delta_mm[0], self.cranial_sign * delta_mm[1]],
            dtype=np.float64,
        )

    def home_mm_to_patient_xz(self, home_relative_mm: object, home_xz_index: object) -> np.ndarray:
        relative = _finite_array(home_relative_mm, (2,), "home_relative_mm")
        home = _finite_array(home_xz_index, (2,), "home_xz_index")
        signed = np.asarray(
            [self.right_sign * relative[0], self.cranial_sign * relative[1]],
            dtype=np.float64,
        )
        return home + signed / self.label_resolution_mm


def validate_planar_rotation(matrix: object, atol: float = 1e-6) -> np.ndarray:
    rotation = _finite_array(matrix, (2, 2), "probe_to_home_rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(2), atol=atol):
        raise ValueError("probe_to_home_rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=atol):
        raise ValueError("probe_to_home_rotation must be a proper rotation")
    return rotation


@dataclass(frozen=True)
class PlannedNativeAction:
    """Normalized environment action and its predicted patient-command displacement."""

    normalized_action: np.ndarray
    predicted_patient_delta_index: np.ndarray
    predicted_angle_delta_rad: float


def plan_native_action(
    patient_delta_index: object,
    angle_delta_rad: float,
    human_to_probe_rotation: object,
    *,
    action_scale: object = (2.0, 2.0, 0.1),
    maximum_applied_action: object = (4.0, 4.0, 0.2),
) -> PlannedNativeAction:
    """Invert SonoGym's probe-plane translation into one bounded native action.

    The task multiplies normalized actions by ``action_scale`` and projects the first two
    applied components through columns 0 and 1 of the human-to-probe rotation. This function
    performs the inverse mapping and scales the translational request uniformly if either
    probe-plane component would exceed the native bound.
    """
    delta = _finite_array(patient_delta_index, (2,), "patient_delta_index")
    rotation = _finite_array(human_to_probe_rotation, (3, 3), "human_to_probe_rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("human_to_probe_rotation must be orthonormal")
    scale = _finite_array(action_scale, (3,), "action_scale")
    maximum = _finite_array(maximum_applied_action, (3,), "maximum_applied_action")
    if np.any(scale <= 0.0) or np.any(maximum <= 0.0):
        raise ValueError("action scales and bounds must be positive")
    basis = rotation[np.ix_([0, 2], [0, 1])]
    if abs(float(np.linalg.det(basis))) < 1e-6:
        raise ValueError("probe translation basis is singular in patient x-z")
    probe_delta = np.linalg.solve(basis, delta)
    translation_ratio = float(np.max(np.abs(probe_delta) / maximum[:2], initial=0.0))
    if translation_ratio > 1.0:
        probe_delta = probe_delta / translation_ratio
    applied_angle = float(np.clip(angle_delta_rad, -maximum[2], maximum[2]))
    normalized = np.asarray(
        [probe_delta[0] / scale[0], probe_delta[1] / scale[1], applied_angle / scale[2]],
        dtype=np.float64,
    )
    return PlannedNativeAction(
        normalized_action=normalized,
        predicted_patient_delta_index=basis @ probe_delta,
        predicted_angle_delta_rad=applied_angle,
    )


@dataclass(frozen=True)
class PixelGeometry:
    """SonoGym ultrasound pixel geometry in the local probe plane."""

    width_pixels: int = 150
    depth_pixels: int = 200
    resolution_mm: float = 0.5

    def __post_init__(self) -> None:
        if self.width_pixels <= 1 or self.depth_pixels <= 1:
            raise ValueError("pixel dimensions must exceed one")
        if not math.isfinite(self.resolution_mm) or self.resolution_mm <= 0:
            raise ValueError("resolution_mm must be finite and positive")

    @property
    def center_column(self) -> float:
        return 0.5 * (self.width_pixels - 1)

    def pixel_to_probe_mm(self, row: float, column: float) -> np.ndarray:
        if not (math.isfinite(row) and math.isfinite(column)):
            raise ValueError("pixel coordinates must be finite")
        if not (0.0 <= row <= self.depth_pixels - 1 and 0.0 <= column <= self.width_pixels - 1):
            raise ValueError("pixel lies outside the ultrasound image")
        return np.asarray(
            [(column - self.center_column) * self.resolution_mm, row * self.resolution_mm],
            dtype=np.float64,
        )

    def probe_mm_to_pixel(self, lateral_depth_mm: object) -> np.ndarray:
        local = _finite_array(lateral_depth_mm, (2,), "lateral_depth_mm")
        column = local[0] / self.resolution_mm + self.center_column
        row = local[1] / self.resolution_mm
        pixel = np.asarray([row, column], dtype=np.float64)
        if not (0.0 <= row <= self.depth_pixels - 1 and 0.0 <= column <= self.width_pixels - 1):
            raise ValueError("probe-plane point lies outside the ultrasound image")
        return pixel

    def pixel_to_home_mm(
        self,
        row: float,
        column: float,
        probe_origin_home_mm: object,
        probe_to_home_rotation: object,
    ) -> np.ndarray:
        origin = _finite_array(probe_origin_home_mm, (2,), "probe_origin_home_mm")
        rotation = validate_planar_rotation(probe_to_home_rotation)
        return origin + rotation @ self.pixel_to_probe_mm(row, column)

    def home_mm_to_pixel(
        self,
        point_home_mm: object,
        probe_origin_home_mm: object,
        probe_to_home_rotation: object,
    ) -> np.ndarray:
        point = _finite_array(point_home_mm, (2,), "point_home_mm")
        origin = _finite_array(probe_origin_home_mm, (2,), "probe_origin_home_mm")
        rotation = validate_planar_rotation(probe_to_home_rotation)
        return self.probe_mm_to_pixel(rotation.T @ (point - origin))


@dataclass(frozen=True)
class ExecutedMotion:
    """One low-level execution result crossing into actor-relative odometry."""

    delta_home_mm: np.ndarray
    delta_angle_rad: float
    accepted: bool
    completed: bool
    clamped: bool
    contact_ok: bool
    workspace_ok: bool

    def __post_init__(self) -> None:
        _finite_array(self.delta_home_mm, (2,), "delta_home_mm")
        if not math.isfinite(float(self.delta_angle_rad)):
            raise ValueError("delta_angle_rad must be finite")

    @property
    def committable(self) -> bool:
        return bool(
            self.accepted
            and self.completed
            and not self.clamped
            and self.contact_ok
            and self.workspace_ok
        )


@dataclass
class HomeRelativeOdometry:
    """Fail-closed relative pose reconstructed only from executed motion feedback."""

    position_mm: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    angle_rad: float = 0.0

    def __post_init__(self) -> None:
        self.position_mm = _finite_array(self.position_mm, (2,), "position_mm").copy()
        if not math.isfinite(float(self.angle_rad)):
            raise ValueError("angle_rad must be finite")
        self.angle_rad = float(wrap_to_pi(self.angle_rad))

    def apply(self, motion: ExecutedMotion) -> bool:
        """Commit accepted complete motion; rejected or clamped motion does not advance."""
        if not motion.committable:
            return False
        self.position_mm += np.asarray(motion.delta_home_mm, dtype=np.float64)
        self.angle_rad = float(wrap_to_pi(self.angle_rad + motion.delta_angle_rad))
        return True

    def delta_to(self, target_position_mm: object, target_angle_rad: float) -> tuple[np.ndarray, float]:
        target = _finite_array(target_position_mm, (2,), "target_position_mm")
        if not math.isfinite(float(target_angle_rad)):
            raise ValueError("target_angle_rad must be finite")
        return target - self.position_mm, float(wrap_to_pi(target_angle_rad - self.angle_rad))
