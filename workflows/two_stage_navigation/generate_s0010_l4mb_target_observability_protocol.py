#!/usr/bin/env python3
"""Freeze a bounded fresh-capture protocol from the evaluator-only pose atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROTOCOL_ID = "sonogym-s0010-l4mb-target-observability-v1"
SESSION_OFFSETS = (
    (0.0, 0.0, 0.0),
    (1.5, -1.0, 0.05),
    (-1.5, 1.0, -0.05),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pose_distance_clear(first: list[float], second: list[float]) -> bool:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    translation_mm = float(np.linalg.norm(a[:2] - b[:2]) * 1.5)
    angle_deg = abs(float(np.arctan2(np.sin(a[2] - b[2]), np.cos(a[2] - b[2]))))
    return bool(translation_mm > 3.0 or np.rad2deg(angle_deg) > 2.4)


def ordered_anchors(targets: list[dict[str, Any]], controls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not targets or not controls:
        raise ValueError("target and control anchors are required")
    ordered = list(targets)
    remaining = list(controls)
    while remaining:
        match = next(
            (
                row
                for row in remaining
                if pose_distance_clear(
                    ordered[-1]["requested_waypoint_pose_index"],
                    row["requested_waypoint_pose_index"],
                )
            ),
            None,
        )
        if match is None:
            raise ValueError("no physically distinct target-to-control transition")
        ordered.append(match)
        remaining.remove(match)
    if any(
        not pose_distance_clear(
            first["requested_waypoint_pose_index"],
            second["requested_waypoint_pose_index"],
        )
        for first, second in zip(ordered, ordered[1:])
    ):
        raise ValueError("ordered anchors contain a transition swallowed by tolerance")
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-track-alignment-pixels", type=float, default=4.0)
    parser.add_argument("--minimum-shadow-contrast", type=float, required=True)
    args = parser.parse_args()

    atlas = json.loads(args.atlas.read_text(encoding="utf-8"))
    if atlas.get("status") not in {"POSE_ATLAS_READY", "POSE_ATLAS_EXPLORATORY_READY"}:
        raise ValueError("pose atlas is not ready")
    targets = list(atlas["target_anchors"])
    controls = list(atlas["control_anchors"])
    if len(targets) != 6 or len(controls) != 6:
        raise ValueError("the frozen design requires six target and six control anchors")
    anchors = ordered_anchors(targets, controls)

    banks: dict[str, list[list[float]]] = {}
    roles: dict[str, list[dict[str, Any]]] = {}
    sessions: list[dict[str, Any]] = []
    for session_index, offset in enumerate(SESSION_OFFSETS):
        bank_id = f"bank_{session_index:02d}"
        delta = np.asarray(offset, dtype=np.float64)
        banks[bank_id] = []
        roles[bank_id] = []
        for anchor_index, row in enumerate(anchors):
            pose = np.asarray(row["requested_waypoint_pose_index"], dtype=np.float64) + delta
            banks[bank_id].append([float(value) for value in pose])
            roles[bank_id].append(
                {
                    "frame_sequence": anchor_index,
                    "declared_role": (
                        "TARGET_REGION_CANDIDATE"
                        if anchor_index < len(targets)
                        else "NEGATIVE_CONTROL"
                    ),
                    "anchor_session_id": row["session_id"],
                    "anchor_frame_sequence": int(row["frame_sequence"]),
                    "anchor_frame_sha256": row["frame_sha256"],
                    "anchor_evaluator_label": row["evaluator_label"],
                    "anchor_evaluator_reason": row["evaluator_reason"],
                    "session_pose_offset_index_rad": [float(value) for value in delta],
                }
            )
        if any(
            not pose_distance_clear(first, second)
            for first, second in zip(banks[bank_id], banks[bank_id][1:])
        ):
            raise ValueError(f"bank {bank_id} contains an invalid adjacent transition")
        session_id = f"s0010l4mbobs_{session_index:02d}"
        sessions.append(
            {
                "session_id": session_id,
                "split": "opened_development_observability",
                "seed": 2026112500 + session_index,
                "reset_id": session_id,
                "absolute_waypoint_bank_id": bank_id,
                "expected_frame_count": len(anchors),
            }
        )

    protocol = {
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_BEFORE_FRESH_CAPTURE",
        "purpose": (
            "Test whether the provisional RIGHT L5 SAP-TP endpoint relationship "
            "is observable from geometry-designed fresh ultrasound views."
        ),
        "anatomy_id": "s0010",
        "clinical_scope": {
            "medial_branch": "RIGHT_L4",
            "vertebral_level": "RIGHT_L5",
            "endpoint": "bony_surrogate_only",
            "clinical_target_validated": False,
        },
        "source": {
            "pose_atlas": str(args.atlas),
            "pose_atlas_sha256": sha256(args.atlas),
            "pose_atlas_schema": atlas["schema"],
            "pose_atlas_status": atlas["status"],
            "target_anchor_source": atlas["selection"]["target_anchor_source"],
            "source_evidence_status": "opened_development_pose_design",
        },
        "scope": {
            "task": "Isaac-robot-US-guidance-v0",
            "patient_id": "s0010",
            "ultrasound_mode": "conv",
            "development_only": True,
            "privileged_pose_design": True,
            "model_training_authorized": False,
            "stage3_authorized": False,
        },
        "information_boundaries": {
            "capture_runtime": [
                "frozen_waypoint_commands",
                "clean_ultrasound",
                "robot_motion_feedback",
            ],
            "privileged_offline_only": [
                "CT_subregion_prediction",
                "native_pixel_to_volume_coordinates",
                "raw_label_slice",
                "target_endpoint_masks",
            ],
            "future_robot_localizer_forbidden": [
                "absolute_waypoint_bank",
                "target_coordinates",
                "CT_masks",
                "native_pixel_to_volume_coordinates",
            ],
        },
        "inputs": atlas["input_hashes"],
        "sampling": {
            "expected_frames_per_session": len(anchors),
            "physical_tracking_position_tolerance_mm": 1.5,
            "physical_tracking_orientation_tolerance_deg": 1.2,
            "waypoint_separation_multiplier": 2.0,
            "minimum_realized_viewpoint_translation_mm": 2.0,
            "minimum_realized_viewpoint_orientation_deg": 1.0,
            "maximum_initial_positioning_steps": 350,
            "maximum_action_steps_per_waypoint": 100,
            "maximum_total_action_steps_per_session": 4200,
            "trajectory_start_frames": [0, 2, 4, 6, 8, 10],
            "trajectory_seed_stride": 1000,
            "enforce_episode_action_budget": True,
            "physical_controller": {
                "name": "coupled_proportional_target",
                "ratio": [0.02, 0.02, 0.6],
                "ceiling_assisted_minimum_requested_angle_rad": 2.99,
                "controller_target_angle_rad": 3.14,
                "accept_against_evaluator_goal_angle": False,
                "capture_angle_ceiling_limited_samples": False,
            },
            "absolute_waypoint_banks_index": banks,
            "planned_waypoint_evidence_by_bank": roles,
        },
        "sessions": sessions,
        "labels": {
            "technical_image_valid": "finite_nonconstant_contact_workspace_and_transition_valid",
            "validity": {
                "maximum_position_error_mm": 50.0,
                "maximum_orientation_error_deg": 30.0,
                "minimum_l4_visible_fraction": 0.0,
            },
            "quality": {"visibility_normalization_fraction": 0.20},
        },
        "dataset_gate": {
            "minimum_distinct_frame_hashes_per_session": len(anchors),
            "require_all_exact_pose_replay": True,
        },
        "exact_pose_replay": {
            "enabled": True,
            "image_width_pixels": 150,
            "image_depth_pixels": 200,
            "label_resolution_m": 0.0015,
            "image_origin_height_m": 0.133,
            "maximum_coordinate_error_index": 0.001,
            "minimum_exact_raw_label_agreement": 0.985,
            "minimum_bone_mask_iou": 0.90,
            "require_direct_coordinate_exact_agreement": True,
            "require_all_frames_to_pass": True,
        },
        "observability_gate": {
            "minimum_sap_core_first_visible_pixels": 1,
            "minimum_tp_core_first_visible_pixels": 1,
            "minimum_target_image_margin_pixels": 4,
            "maximum_interface_track_distance_pixels": float(
                args.minimum_track_alignment_pixels
            ),
            "minimum_posterior_shadow_contrast": float(args.minimum_shadow_contrast),
            "minimum_positive_target_frames_per_session": 2,
            "minimum_positive_sessions": 2,
            "maximum_positive_control_frames": 0,
            "ambiguous_is_not_negative": True,
        },
        "resource_bound": {
            "account": "anest-dept",
            "gpu_count": 1,
            "gpu_model": "NVIDIA_L4",
            "cpu_cores": 4,
            "memory_gb": 30,
            "walltime": "00:30:00",
            "maximum_new_gpu_jobs": 1,
            "maximum_storage_growth_gb": 1.0,
        },
        "change_policy": [
            "Do not change waypoints, labels or observability gates after capture.",
            "Do not expose atlas geometry or absolute waypoints to a future robot localizer.",
            "Do not treat a bony surrogate as a nerve or intervention endpoint.",
        ],
    }
    args.output.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    seal = args.output.with_suffix(".sha256")
    seal.write_text(f"{sha256(args.output)}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({"protocol": str(args.output), "sha256": sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
