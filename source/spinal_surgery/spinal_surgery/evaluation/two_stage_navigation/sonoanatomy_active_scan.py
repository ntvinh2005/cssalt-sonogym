"""Actor-side state machine for clinician-inspired SAP/TP probe sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class ScanState(str, Enum):
    ACQUIRE_REFERENCE_TP_REGION = "ACQUIRE_REFERENCE_TP_REGION"
    LONGITUDINAL_LATERAL_SURVEY = "LONGITUDINAL_LATERAL_SURVEY"
    REVERSE_MEDIAL_SURVEY = "REVERSE_MEDIAL_SURVEY"
    ACQUIRE_TRANSVERSE_VIEW = "ACQUIRE_TRANSVERSE_VIEW"
    TRANSVERSE_CRANIAL_CAUDAL_SWEEP = "TRANSVERSE_CRANIAL_CAUDAL_SWEEP"
    SAP_TP_IDENTITY_INFERENCE = "SAP_TP_IDENTITY_INFERENCE"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class ScanConfig:
    longitudinal_angle_rad: float = np.pi / 2.0
    transverse_angle_rad: float = 2.95
    translation_step_index: float = 3.0
    reference_offsets_index: tuple[float, ...] = (-6.0, -3.0, 0.0, 3.0, 6.0)
    maximum_lateral_steps: int = 4
    maximum_medial_steps: int = 8
    transverse_offsets_index: tuple[float, ...] = (-6.0, -3.0, 0.0, 3.0, 6.0, 3.0, 0.0, -3.0)

    def __post_init__(self) -> None:
        values = (
            self.longitudinal_angle_rad,
            self.transverse_angle_rad,
            self.translation_step_index,
            *self.reference_offsets_index,
            *self.transverse_offsets_index,
        )
        if not all(np.isfinite(values)) or self.translation_step_index <= 0.0:
            raise ValueError("scan configuration must be finite with a positive step")
        if self.maximum_lateral_steps <= 0 or self.maximum_medial_steps <= 0:
            raise ValueError("scan step budgets must be positive")


@dataclass(frozen=True)
class ScanEvidence:
    frame_sequence: int
    pose_index: tuple[float, float, float]
    tp_predicted: bool
    sap_predicted: bool
    other_bone_predicted: bool
    tp_predicted_pixels: int
    sap_predicted_pixels: int
    image_valid: bool
    contact_ok: bool
    producer_advanced: bool

    @property
    def eligible(self) -> bool:
        return self.image_valid and self.contact_ok and self.producer_advanced


def staged_orientation_targets(
    current_pose_index: Sequence[float],
    target_pose_index: Sequence[float],
    *,
    maximum_step_rad: float,
) -> tuple[np.ndarray, ...]:
    """Return rotation-only intermediate poses for a large plane change."""

    current = np.asarray(current_pose_index, dtype=np.float64)
    target = np.asarray(target_pose_index, dtype=np.float64)
    if (
        current.shape != (3,)
        or target.shape != (3,)
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(target))
        or not np.isfinite(maximum_step_rad)
        or maximum_step_rad <= 0.0
    ):
        raise ValueError("orientation staging inputs are invalid")
    difference = float((target[2] - current[2] + np.pi) % (2.0 * np.pi) - np.pi)
    step_count = max(1, int(np.ceil(abs(difference) / maximum_step_rad)))
    result = []
    for index in range(1, step_count):
        intermediate = current.copy()
        intermediate[2] = current[2] + difference * index / step_count
        result.append(intermediate)
    return tuple(result)


class ClinicianSweepController:
    """Choose scan poses from fresh ultrasound evidence and executed motion history."""

    def __init__(self, tp_anchor_pose_index: Sequence[float], config: ScanConfig) -> None:
        anchor = np.asarray(tp_anchor_pose_index, dtype=np.float64)
        if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
            raise ValueError("TP anchor pose must be a finite 3-vector")
        self.anchor = anchor
        self.config = config
        self.state = ScanState.ACQUIRE_REFERENCE_TP_REGION
        self.history: list[ScanEvidence] = []
        self._reference_cursor = 0
        self._reference: ScanEvidence | None = None
        self._lateral_steps = 0
        self._medial_steps = 0
        self._transverse_cursor = 0
        self._tp_lost = False
        self._tp_reappeared = False
        self._sap_after_reappearance = False
        self.reason = "REFERENCE_SEARCH_PENDING"

    def _reference_pose(self, offset: float) -> np.ndarray:
        return np.asarray(
            [
                self.anchor[0] + offset,
                self.anchor[1],
                self.config.longitudinal_angle_rad,
            ],
            dtype=np.float64,
        )

    def next_pose(self) -> np.ndarray | None:
        if self.state == ScanState.ACQUIRE_REFERENCE_TP_REGION:
            return self._reference_pose(
                self.config.reference_offsets_index[self._reference_cursor]
            )
        if self.state == ScanState.LONGITUDINAL_LATERAL_SURVEY:
            assert self._reference is not None
            return np.asarray(self._reference.pose_index, dtype=np.float64) + np.asarray(
                [self.config.translation_step_index * (self._lateral_steps + 1), 0.0, 0.0]
            )
        if self.state == ScanState.REVERSE_MEDIAL_SURVEY:
            assert self.history
            current = np.asarray(self.history[-1].pose_index, dtype=np.float64)
            return current + np.asarray([-self.config.translation_step_index, 0.0, 0.0])
        if self.state == ScanState.ACQUIRE_TRANSVERSE_VIEW:
            self.state = ScanState.TRANSVERSE_CRANIAL_CAUDAL_SWEEP
        if self.state == ScanState.TRANSVERSE_CRANIAL_CAUDAL_SWEEP:
            assert self._reference is not None
            offset = self.config.transverse_offsets_index[self._transverse_cursor]
            reference = np.asarray(self._reference.pose_index, dtype=np.float64)
            return np.asarray(
                [reference[0], reference[1] + offset, self.config.transverse_angle_rad],
                dtype=np.float64,
            )
        return None

    def observe(self, evidence: ScanEvidence) -> None:
        if self.state in {ScanState.SAP_TP_IDENTITY_INFERENCE, ScanState.ABSTAIN}:
            raise RuntimeError("scan controller already terminated")
        if not evidence.eligible:
            self.history.append(evidence)
            self.state = ScanState.ABSTAIN
            self.reason = "INVALID_OR_STALE_ACQUISITION"
            return
        self.history.append(evidence)
        if self.state == ScanState.ACQUIRE_REFERENCE_TP_REGION:
            self._reference_cursor += 1
            if self._reference_cursor < len(self.config.reference_offsets_index):
                return
            candidates = [
                row
                for row in self.history
                if row.tp_predicted and row.eligible
            ]
            if not candidates:
                self.state = ScanState.ABSTAIN
                self.reason = "REFERENCE_TP_NOT_FOUND"
                return
            self._reference = max(
                candidates,
                key=lambda row: (row.tp_predicted_pixels, -row.frame_sequence),
            )
            self.state = ScanState.LONGITUDINAL_LATERAL_SURVEY
            self.reason = "REFERENCE_TP_FOUND"
            return
        if self.state == ScanState.LONGITUDINAL_LATERAL_SURVEY:
            self._lateral_steps += 1
            if not evidence.tp_predicted:
                self._tp_lost = True
                self.state = ScanState.REVERSE_MEDIAL_SURVEY
                self.reason = "LATERAL_TP_LOSS_OBSERVED"
            elif self._lateral_steps >= self.config.maximum_lateral_steps:
                self.state = ScanState.REVERSE_MEDIAL_SURVEY
                self.reason = "LATERAL_BUDGET_REACHED_WITHOUT_TP_LOSS"
            return
        if self.state == ScanState.REVERSE_MEDIAL_SURVEY:
            self._medial_steps += 1
            if self._tp_lost and evidence.tp_predicted:
                self._tp_reappeared = True
            if self._tp_reappeared and evidence.sap_predicted:
                self._sap_after_reappearance = True
            if self._sap_after_reappearance or self._medial_steps >= self.config.maximum_medial_steps:
                self.state = ScanState.ACQUIRE_TRANSVERSE_VIEW
                self.reason = (
                    "TP_RETURN_AND_AP_REGION_OBSERVED"
                    if self._sap_after_reappearance
                    else "MEDIAL_BUDGET_REACHED"
                )
            return
        if self.state == ScanState.TRANSVERSE_CRANIAL_CAUDAL_SWEEP:
            self._transverse_cursor += 1
            if self._transverse_cursor >= len(self.config.transverse_offsets_index):
                self.state = ScanState.SAP_TP_IDENTITY_INFERENCE
                self.reason = "COMPLEMENTARY_SWEEPS_COMPLETE"
            return
        raise RuntimeError(f"unexpected scan state: {self.state}")

    def transition_evidence(self) -> dict[str, bool | int | str]:
        return {
            "state": self.state.value,
            "reason": self.reason,
            "tp_lost_laterally": self._tp_lost,
            "tp_reappeared_medially": self._tp_reappeared,
            "sap_after_tp_reappearance": self._sap_after_reappearance,
            "frame_count": len(self.history),
        }
