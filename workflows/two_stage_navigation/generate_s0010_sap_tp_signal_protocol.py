#!/usr/bin/env python3
"""Freeze a fresh simulator capture from the broad SAP/TP surface search."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROTOCOL_ID = "sonogym-s0010-l4mb-sap-tp-signal-v1"
TARGET_COUNT = 8
CONTROL_COUNT = 4
SESSION_COUNT = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def separated(first: list[float], second: list[float]) -> bool:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    translation_mm = float(np.linalg.norm(a[:2] - b[:2]) * 1.5)
    angle_deg = float(np.rad2deg(abs(np.arctan2(np.sin(a[2] - b[2]), np.cos(a[2] - b[2])))))
    return bool(translation_mm >= 3.0 or angle_deg >= 2.5)


def ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("candidate rows are empty")
    remaining = list(rows[1:])
    ordered = [rows[0]]
    while remaining:
        match = next(
            (
                row
                for row in remaining
                if separated(ordered[-1]["pose_index_rad"], row["pose_index_rad"])
            ),
            None,
        )
        if match is None:
            raise ValueError("cannot order candidates into physically distinct transitions")
        ordered.append(match)
        remaining.remove(match)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-result", type=Path, required=True)
    parser.add_argument("--control-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args()

    search = json.loads(args.search_result.read_text(encoding="utf-8"))
    if search["status"] != "SAP_TP_SURFACE_CAPTURE_CANDIDATES_FOUND":
        raise ValueError("broad search did not produce SAP/TP surface candidates")
    candidates = list(search["selected_capture_candidates"][:TARGET_COUNT])
    if len(candidates) != TARGET_COUNT:
        raise ValueError("eight target candidates are required")
    for row in candidates:
        if not (
            int(row["sap_first_visible_pixels"]) > 0
            and int(row["tp_first_visible_pixels"]) > 0
            and int(row["target_margin_pixels"]) >= 4
        ):
            raise ValueError("target candidate violates the frozen geometry screen")

    historical = yaml.safe_load(args.control_protocol.read_text(encoding="utf-8"))
    historical_bank = historical["sampling"]["absolute_waypoint_banks_index"]["bank_00"]
    historical_roles = historical["sampling"]["planned_waypoint_evidence_by_bank"]["bank_00"]
    controls = [
        {
            "pose_index_rad": [float(value) for value in pose],
            "source_role": role["declared_role"],
            "source_frame_sequence": int(role["frame_sequence"]),
        }
        for pose, role in zip(historical_bank, historical_roles)
        if role["declared_role"] == "NEGATIVE_CONTROL"
    ][:CONTROL_COUNT]
    if len(controls) != CONTROL_COUNT:
        raise ValueError("four historical negative controls are required")

    rows = [
        {**row, "declared_role": "SAP_TP_SURFACE_CANDIDATE"} for row in candidates
    ] + [{**row, "declared_role": "NEGATIVE_CONTROL"} for row in controls]
    rows = ordered_rows(rows)
    waypoints = [row["pose_index_rad"] for row in rows]
    sessions = [
        {
            "session_id": f"s0010saptpsig_{index:02d}",
            "split": "opened_development_observability",
            "seed": 2026112600 + index,
            "reset_id": f"s0010saptpsig_{index:02d}",
            "absolute_waypoint_bank_id": f"bank_{index:02d}",
            "expected_frame_count": len(waypoints),
        }
        for index in range(SESSION_COUNT)
    ]
    roles = [
        {
            "frame_sequence": index,
            "declared_role": row["declared_role"],
            "search_pose_index_rad": row["pose_index_rad"],
            "predicted_sap_first_visible_pixels": row.get("sap_first_visible_pixels"),
            "predicted_tp_first_visible_pixels": row.get("tp_first_visible_pixels"),
            "predicted_target_margin_pixels": row.get("target_margin_pixels"),
        }
        for index, row in enumerate(rows)
    ]
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_BEFORE_FRESH_CAPTURE",
        "purpose": "Test whether separate RIGHT L5 SAP and TP first-hit surfaces have clean ultrasound evidence.",
        "anatomy_id": "s0010",
        "clinical_scope": {
            "medial_branch": "RIGHT_L4",
            "vertebral_level": "RIGHT_L5",
            "endpoint": "bony_surrogate_only",
            "clinical_target_validated": False,
        },
        "source": {
            "implementation_commit": args.implementation_commit,
            "search_result": str(args.search_result),
            "search_result_sha256": sha256(args.search_result),
            "search_status": search["status"],
            "control_protocol": str(args.control_protocol),
            "control_protocol_sha256": sha256(args.control_protocol),
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
            "capture_runtime": ["frozen_waypoint_commands", "clean_ultrasound", "robot_motion_feedback"],
            "privileged_offline_only": [
                "CT_subregion_prediction",
                "native_pixel_to_volume_coordinates",
                "raw_label_slice",
                "SAP_and_TP_masks",
            ],
            "future_robot_localizer_forbidden": [
                "absolute_waypoint_bank",
                "target_coordinates",
                "CT_masks",
                "native_pixel_to_volume_coordinates",
            ],
        },
        "inputs": search["input_hashes"],
        "sampling": {
            "expected_frames_per_session": len(waypoints),
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
            "absolute_waypoint_banks_index": {
                f"bank_{index:02d}": waypoints for index in range(SESSION_COUNT)
            },
            "planned_waypoint_evidence_by_bank": {
                f"bank_{index:02d}": roles for index in range(SESSION_COUNT)
            },
        },
        "sessions": sessions,
        "labels": {
            "technical_image_valid": "finite_nonconstant_contact_workspace_and_transition_valid",
            "validity": {
                "maximum_position_error_mm": 50.0,
                "maximum_orientation_error_deg": 30.0,
                "minimum_l4_visible_fraction": 0.000001,
            },
            "quality": {"visibility_normalization_fraction": 0.20},
        },
        "dataset_gate": {
            "minimum_distinct_frame_hashes_per_session": len(waypoints),
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
            "minimum_sap_first_visible_pixels": 3,
            "minimum_tp_first_visible_pixels": 3,
            "minimum_target_image_margin_pixels": 4,
            "maximum_interface_track_distance_pixels": 4.0,
            "minimum_posterior_shadow_contrast": 0.05,
            "minimum_supported_target_frames_per_session": 2,
            "minimum_supported_sessions": 2,
            "maximum_supported_control_frames": 0,
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
            "Do not change waypoints or gates after capture.",
            "Do not expose privileged geometry or absolute waypoints to a future robot localizer.",
            "Do not treat visible SAP and TP surfaces as an MBB endpoint.",
        ],
    }
    args.output.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    args.output.with_suffix(".sha256").write_text(
        f"{sha256(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"protocol": str(args.output), "sha256": sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
