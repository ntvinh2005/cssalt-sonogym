#!/usr/bin/env python3
"""Generate the sealed fresh-capture protocol from a frozen pose plan and model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


PROTOCOL_ID = "sonogym-s0010-sap-tp-pair-confirmation-v1r3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--carry-forward-protocol", type=Path)
    parser.add_argument("--carry-forward-session-count", type=int, default=0)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan["schema"] != "s0010-sap-tp-pair-confirmation-plan-v1":
        raise ValueError("unexpected pose-plan schema")
    sessions = []
    banks = {}
    evidence = {}
    for session_index, planned in enumerate(plan["sessions"]):
        bank_id = f"bank_{session_index:02d}"
        rows = planned["rows"]
        if len(rows) != 14:
            raise ValueError("each confirmation session requires fourteen frames")
        if sum(row["declared_role"] == "SAP_TP_SURFACE_CANDIDATE" for row in rows) != 8:
            raise ValueError("each session requires eight target candidates")
        if sum(row["declared_role"] == "CLEAR_NEGATIVE" for row in rows) != 6:
            raise ValueError("each session requires six controls")
        banks[bank_id] = [row["pose_index_rad"] for row in rows]
        evidence[bank_id] = [
            {
                "frame_sequence": index,
                "declared_role": row["declared_role"],
                "search_pose_index_rad": row["pose_index_rad"],
                "predicted_evaluator_label": row["evaluator_label"],
                "predicted_evaluator_reason": row["evaluator_reason"],
                "predicted_sap_first_visible_pixels": int(
                    row["sap_first_visible_pixels"]
                ),
                "predicted_tp_first_visible_pixels": int(
                    row["tp_first_visible_pixels"]
                ),
                "predicted_target_margin_pixels": int(row["target_margin_pixels"]),
            }
            for index, row in enumerate(rows)
        ]
        sessions.append(
            {
                "session_id": planned["session_id"],
                "split": "unopened_pair_confirmation",
                "seed": int(planned["seed"]),
                "reset_id": planned["session_id"],
                "absolute_waypoint_bank_id": bank_id,
                "expected_frame_count": 14,
            }
        )
    carry_forward: dict[str, object] | None = None
    if args.carry_forward_session_count:
        if args.carry_forward_protocol is None:
            raise ValueError("carry-forward session count requires a source protocol")
        if not 0 < args.carry_forward_session_count < len(sessions):
            raise ValueError("carry-forward session count must leave a fresh session")
        source_protocol = yaml.safe_load(
            args.carry_forward_protocol.read_text(encoding="utf-8")
        )
        if source_protocol["source"]["checkpoint_sha256"] != plan["checkpoint_sha256"]:
            raise ValueError("carry-forward protocol used another checkpoint")
        for session_index in range(args.carry_forward_session_count):
            bank_id = f"bank_{session_index:02d}"
            source_session = source_protocol["sessions"][session_index]
            if source_session["session_id"] != sessions[session_index]["session_id"]:
                raise ValueError("carry-forward session identity changed")
            banks[bank_id] = source_protocol["sampling"][
                "absolute_waypoint_banks_index"
            ][bank_id]
            evidence[bank_id] = source_protocol["sampling"][
                "planned_waypoint_evidence_by_bank"
            ][bank_id]
        carry_forward = {
            "protocol": str(args.carry_forward_protocol),
            "protocol_sha256": sha256(args.carry_forward_protocol),
            "session_ids": [
                row["session_id"]
                for row in sessions[: args.carry_forward_session_count]
            ],
        }
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_BEFORE_FRESH_CAPTURE",
        "purpose": "Evaluate the frozen image-only SAP/TP interval selector on new pose groups.",
        "anatomy_id": "s0010",
        "clinical_scope": {
            "medial_branch": "RIGHT_L4",
            "vertebral_level": "RIGHT_L5",
            "endpoint": "bony_surrogate_only",
            "clinical_target_validated": False,
        },
        "source": {
            "implementation_commit": args.implementation_commit,
            "pose_plan": str(args.plan),
            "pose_plan_sha256": sha256(args.plan),
            "checkpoint": plan["checkpoint"],
            "checkpoint_sha256": plan["checkpoint_sha256"],
            "training_protocol_sha256": plan["training_protocol_sha256"],
            "source_evidence_status": "opened_development_training_then_unopened_pose_capture",
            "carry_forward": carry_forward,
        },
        "scope": {
            "task": "Isaac-robot-US-guidance-v0",
            "patient_id": "s0010",
            "ultrasound_mode": "conv",
            "privileged_pose_design": True,
            "model_refit_after_capture": False,
            "stage3_authorized": False,
        },
        "information_boundaries": {
            "model_inputs": [
                "clean_ultrasound",
                "prior_clean_ultrasound",
                "executed_relative_motion",
            ],
            "privileged_offline_only": [
                "CT_subregion_prediction",
                "native_pixel_to_volume_coordinates",
                "raw_label_slice",
                "SAP_and_TP_masks",
                "absolute_waypoint_bank",
            ],
        },
        "sampling": {
            "expected_frames_per_session": 14,
            "physical_tracking_position_tolerance_mm": 1.5,
            "physical_tracking_orientation_tolerance_deg": 1.2,
            "waypoint_separation_multiplier": 2.0,
            "minimum_realized_viewpoint_translation_mm": 2.0,
            "minimum_realized_viewpoint_orientation_deg": 1.0,
            "maximum_initial_positioning_steps": 350,
            "maximum_action_steps_per_waypoint": 100,
            "maximum_total_action_steps_per_session": 5000,
            "trajectory_start_frames": [0, 2, 4, 6, 8, 10, 12],
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
            "planned_waypoint_evidence_by_bank": evidence,
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
            "minimum_distinct_frame_hashes_per_session": 14,
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
        "pair_label_gate": {
            "maximum_segment_surface_distance_pixels": 4.0,
            "minimum_aligned_segment_points": 3,
            "minimum_sap_first_visible_pixels": 3,
            "minimum_tp_first_visible_pixels": 3,
            "minimum_target_image_margin_pixels": 4,
        },
        "track_window_config": {
            "window_columns": 15,
            "stride_columns": 5,
            "minimum_center_separation_columns": 8,
            "maximum_tracks": 2,
        },
        "decision_gate": {
            "minimum_pair_probability": 0.50,
            "minimum_prior_view_support": 1,
            "minimum_support_translation_mm": 2.0,
            "minimum_support_angle_deg": 1.0,
            "maximum_support_translation_mm": 15.0,
            "maximum_support_angle_deg": 15.0,
            "maximum_scaled_pair_distance": 1.0,
            "support_score_increment": 0.10,
            "maximum_counted_support": 2,
        },
        "final_gate": {
            "minimum_top1_correct_sessions": 3,
            "minimum_top3_correct_sessions": 3,
            "minimum_pair_precision": 0.75,
            "maximum_clear_negative_false_accept_rate": 0.10,
            "minimum_correct_accepted_targets_per_session": 2,
            "minimum_actual_targets_per_session": 4,
            "minimum_actual_clear_negatives_per_session": 4,
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
            "Do not refit the model or change thresholds after capture.",
            "Do not expose privileged geometry or absolute waypoints to the selector.",
            "Do not treat visible SAP and TP surfaces as an MBB endpoint.",
        ],
    }
    args.output.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    digest = sha256(args.output)
    args.output.with_suffix(".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"protocol": str(args.output), "sha256": digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
