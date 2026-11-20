#!/usr/bin/env python3
"""Generate sealed development and holdout SAP validation capture protocols."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml


LEVEL_WAYPOINTS_PER_SESSION = 20
FRAMES_PER_SESSION = 2 * LEVEL_WAYPOINTS_PER_SESSION


def waypoint_pool(level: str) -> list[list[float]]:
    xs = np.arange(145.0, 163.1, 3.0)
    zs = (
        np.arange(137.5, 149.6, 3.0)
        if level == "L4"
        else np.arange(120.5, 132.6, 3.0)
    )
    angles = np.asarray([2.77, 2.83, 2.89, 2.95])
    center = np.asarray([157.0, 141.0 if level == "L4" else 127.0, 2.88])
    rows = [
        [float(x), float(z), float(angle)]
        for x in xs
        for z in zs
        for angle in angles
    ]
    return sorted(
        rows,
        key=lambda row: (
            float(np.linalg.norm((np.asarray(row[:2]) - center[:2]) / np.asarray([6.0, 6.0]))),
            abs(row[2] - center[2]),
            row,
        ),
    )


def allocate_banks() -> dict[str, list[list[float]]]:
    banks = {f"bank_{index}": [] for index in range(6)}
    for level in ("L4", "L5"):
        pool = waypoint_pool(level)
        selected = pool[: 6 * LEVEL_WAYPOINTS_PER_SESSION]
        for index in range(6):
            banks[f"bank_{index}"].extend(
                selected[index::6][:LEVEL_WAYPOINTS_PER_SESSION]
            )
    return banks


def protocol(
    *,
    protocol_id: str,
    implementation_commit: str,
    sessions: list[dict[str, Any]],
    banks: dict[str, list[list[float]]],
) -> dict[str, Any]:
    return {
        "protocol_id": protocol_id,
        "status": "PRE_GPU_FROZEN",
        "purpose": (
            "Acquire session-disjoint RIGHT L4/L5 SAP development or holdout views "
            "after freezing image-only interface and simulator-proxy label criteria."
        ),
        "source": {
            "implementation_commit": implementation_commit,
            "prediction_job": 52078067,
            "prediction_sha256": "a78daeae9159f0df28ba6fffcb8e8d7ef6c2e87f490c9598acfcb82efad1677f",
            "planning_rule": "fixed_disjoint_pose_grid_before_capture",
        },
        "scope": {
            "task": "Isaac-robot-US-guidance-v0",
            "patient_id": "s0010",
            "ultrasound_mode": "conv",
            "privileged_dataset_collection": True,
            "runtime_classifier_geometry_access": False,
            "stage3_authorized": False,
        },
        "sampling": {
            "expected_frames_per_session": FRAMES_PER_SESSION,
            "waypoint_level_order": [
                {"level": "L4", "start_index": 0, "stop_index_exclusive": 20},
                {"level": "L5", "start_index": 20, "stop_index_exclusive": 40},
            ],
            "physical_tracking_position_tolerance_mm": 1.5,
            "physical_tracking_orientation_tolerance_deg": 1.2,
            "waypoint_separation_multiplier": 2.0,
            "minimum_realized_viewpoint_translation_mm": 2.0,
            "minimum_realized_viewpoint_orientation_deg": 1.0,
            "maximum_initial_positioning_steps": 350,
            "maximum_action_steps_per_waypoint": 150,
            "maximum_total_action_steps_per_session": 8000,
            "trajectory_start_frames": list(range(0, FRAMES_PER_SESSION, 2)),
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
        },
        "sessions": sessions,
        "labels": {
            "technical_image_valid": "finite_nonconstant_contact_workspace_and_transition_valid",
            "validity": {
                "maximum_position_error_mm": 10.0,
                "maximum_orientation_error_deg": 10.0,
                "minimum_l4_visible_fraction": 0.05,
            },
            "quality": {"visibility_normalization_fraction": 0.20},
        },
        "dataset_gate": {"minimum_distinct_frame_hashes_per_session": FRAMES_PER_SESSION},
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
        "label_gate": {
            "alignment_tolerance_pixels": 4.0,
            "minimum_surface_columns": 5,
            "minimum_aligned_fraction": 0.40,
            "minimum_aligned_run_columns": 5,
            "maximum_median_distance_pixels": 4.0,
            "minimum_local_gradient": 0.10,
            "minimum_local_contrast": 0.02,
            "minimum_posterior_shadow": 0.02,
            "ambiguous_excluded_from_binary_targets": True,
            "minimum_matched_acceptance_rate": 0.50,
            "maximum_mismatched_false_alignment_rate": 0.10,
            "minimum_correct_projection_ranking_rate": 0.70,
        },
        "change_policy": [
            "Do not change waypoint banks or label criteria after capture.",
            "Do not inspect holdout images before model and threshold are frozen.",
            "Do not expose privileged geometry to classifier features.",
        ],
    }


def write_protocol(path: Path, document: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    banks = allocate_banks()
    development_sessions = [
        {
            "session_id": f"s0010sapval_train_{index:02d}",
            "split": "train",
            "seed": 2026112101 + index,
            "reset_id": f"s0010sapval_train_{index:02d}",
            "absolute_waypoint_bank_id": f"bank_{index}",
            "expected_frame_count": FRAMES_PER_SESSION,
        }
        for index in range(3)
    ] + [
        {
            "session_id": "s0010sapval_calibration_00",
            "split": "calibration",
            "seed": 2026112104,
            "reset_id": "s0010sapval_calibration_00",
            "absolute_waypoint_bank_id": "bank_3",
            "expected_frame_count": FRAMES_PER_SESSION,
        }
    ]
    holdout_sessions = [
        {
            "session_id": f"s0010sapval_evaluation_{index:02d}",
            "split": "evaluation",
            "seed": 2026112105 + index,
            "reset_id": f"s0010sapval_evaluation_{index:02d}",
            "absolute_waypoint_bank_id": f"bank_{4 + index}",
            "expected_frame_count": FRAMES_PER_SESSION,
        }
        for index in range(2)
    ]
    write_protocol(
        args.output_dir / "s0010_sap_validation_development_protocol_v1.yaml",
        protocol(
            protocol_id="sonogym-s0010-sap-validation-development-v1",
            implementation_commit=args.implementation_commit,
            sessions=development_sessions,
            banks={key: value for key, value in banks.items() if key in {"bank_0", "bank_1", "bank_2", "bank_3"}},
        ),
    )
    write_protocol(
        args.output_dir / "s0010_sap_validation_holdout_protocol_v1.yaml",
        protocol(
            protocol_id="sonogym-s0010-sap-validation-holdout-v1",
            implementation_commit=args.implementation_commit,
            sessions=holdout_sessions,
            banks={key: value for key, value in banks.items() if key in {"bank_4", "bank_5"}},
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
