#!/usr/bin/env python3
"""Freeze a four-session public-6DoF SAP/TP confirmation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import correct_s0010_sap_observability_v1 as sap_correction
import prepare_s0010_sap_tp_pair_confirmation as pair_prepare
import run_s0010_l4_medial_branch_target as target_workflow
import search_s0010_l4mb_target_viewpoints as search
from two_stage_navigation.acquisition_integrity import validate_waypoint_bank


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_poses(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"pose_index_rad": waypoint}
        for bank in protocol["sampling"]["absolute_waypoint_banks_index"].values()
        for waypoint in bank
    ]


def evidence_row(row: dict[str, Any], role: str, frame_sequence: int) -> dict[str, Any]:
    return {
        "frame_sequence": frame_sequence,
        "declared_role": role,
        "search_pose_index_rad": row["pose_index_rad"],
        "predicted_evaluator_label": row["evaluator_label"],
        "predicted_evaluator_reason": row["evaluator_reason"],
        "predicted_sap_first_visible_pixels": int(row["sap_first_visible_pixels"]),
        "predicted_tp_first_visible_pixels": int(row["tp_first_visible_pixels"]),
        "predicted_target_margin_pixels": int(row["target_margin_pixels"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--negative-protocol", type=Path, required=True)
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--exact-candidates", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--patient-surface-dir", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base = yaml.safe_load(args.base_protocol.read_text(encoding="utf-8"))
    training = yaml.safe_load(args.training_protocol.read_text(encoding="utf-8"))
    negative_protocol = yaml.safe_load(
        args.negative_protocol.read_text(encoding="utf-8")
    )
    development = json.loads(args.development_result.read_text(encoding="utf-8"))
    selected = development["selected_configuration"]
    expected = {
        "surface_radius_mm": 2.0,
        "relative_vector_tolerance_mm": 4.0,
        "minimum_support": 1,
    }
    if any(selected[key] != value for key, value in expected.items()):
        raise ValueError("opened-development configuration differs from frozen values")
    checkpoint = Path(base["source"]["checkpoint"])
    if sha256(checkpoint) != base["source"]["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint hash changed")

    _, sap_mask, tp_mask, contract = target_workflow.load_target_geometry(
        args.prediction, args.segmentation
    )
    bone_ids = sap_correction.bone_source_ids(args.label_conversion)
    segmentation = np.asanyarray(
        target_workflow.nib.load(str(args.segmentation)).dataobj
    ).astype(np.uint8)
    surface_y = search.safe_torch(args.patient_surface_dir / "body_lowest_y_array.pt")
    surface_normals = search.safe_torch(
        args.patient_surface_dir / "body_surface_normal_array.pt"
    )

    excluded = protocol_poses(training) + protocol_poses(base) + protocol_poses(
        negative_protocol
    )
    target_pool = [
        row
        for row in pair_prepare.read_exact_candidates(args.exact_candidates)
        if row["sap_first_visible_pixels"] >= 3
        and row["tp_first_visible_pixels"] >= 3
        and row["target_margin_pixels"] >= 4
    ]
    targets = pair_prepare.select_pairs(target_pool, 16, excluded)
    negative_pool = pair_prepare.negative_candidates(
        negative_protocol,
        surface_y=surface_y,
        surface_normals=surface_normals,
        segmentation=segmentation,
        bone_ids=bone_ids,
        sap_mask=sap_mask,
        tp_mask=tp_mask,
        contract=contract,
    )
    tp_only = [
        row
        for row in negative_pool
        if row["evaluator_reason"] == "TP_ONLY_WITHOUT_SAP_IN_PLANE"
    ]
    other = [
        row
        for row in negative_pool
        if row["evaluator_reason"] == "OTHER_BONE_OR_BACKGROUND"
    ]
    tp_controls = pair_prepare.select_pairs(tp_only, 4, excluded + targets)
    other_controls = pair_prepare.select_pairs(
        other, 8, excluded + targets + tp_controls
    )

    protocol = deepcopy(base)
    protocol["protocol_id"] = "sonogym-s0010-sap-tp-public-6dof-confirmation-v1"
    protocol["status"] = "FROZEN_BEFORE_FRESH_CAPTURE"
    protocol["purpose"] = (
        "Evaluate frozen ultrasound pair evidence with synchronized public robot 6DoF "
        "geometry on four new pose groups."
    )
    protocol["source"] = {
        "implementation_commit": args.implementation_commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "base_protocol": str(args.base_protocol),
        "base_protocol_sha256": sha256(args.base_protocol),
        "training_protocol_sha256": sha256(args.training_protocol),
        "opened_development_result": str(args.development_result),
        "opened_development_result_sha256": sha256(args.development_result),
        "exact_candidates_sha256": sha256(args.exact_candidates),
        "source_evidence_status": "opened_development_then_unopened_public_6dof_capture",
    }
    protocol["information_boundaries"] = {
        "actor_inputs": [
            "clean_ultrasound",
            "synchronized_base_to_end_effector_pose_from_robot_proprioception",
            "public_probe_image_calibration",
            "prior_actor_observations",
        ],
        "privileged_offline_only": [
            "CT_subregion_prediction",
            "native_pixel_to_volume_coordinates",
            "raw_label_slice",
            "SAP_and_TP_masks",
            "absolute_waypoint_bank",
            "world_to_robot_base_transform",
        ],
    }
    protocol["public_probe_calibration"] = {
        "image_width_pixels": 150,
        "image_depth_pixels": 200,
        "pixel_resolution_m": 0.0005,
        "image_origin_height_m": 0.133,
    }
    protocol["sampling"]["absolute_waypoint_banks_index"] = {}
    protocol["sampling"]["planned_waypoint_evidence_by_bank"] = {}
    protocol["sampling"]["trajectory_start_frames"] = list(range(0, 14, 2))
    sessions: list[dict[str, Any]] = []
    plan_sessions: list[dict[str, Any]] = []
    for session_index in range(4):
        rows = (
            targets[session_index * 8 : (session_index + 1) * 8]
            + tp_controls[session_index * 2 : (session_index + 1) * 2]
            + other_controls[session_index * 4 : (session_index + 1) * 4]
        )
        roles = ["SAP_TP_SURFACE_CANDIDATE"] * 8 + ["CLEAR_NEGATIVE"] * 6
        bank_id = f"bank_{session_index:02d}"
        session_id = f"s0010saptp6dof_{session_index:02d}"
        waypoints = [row["pose_index_rad"] for row in rows]
        validate_waypoint_bank(
            waypoints,
            command_minimum=[-1.0e9, -1.0e9, -1.0e9],
            command_maximum=[1.0e9, 1.0e9, 1.0e9],
            label_resolution_mm=pair_prepare.LABEL_RESOLUTION_MM,
            position_tolerance_mm=pair_prepare.POSITION_TOLERANCE_MM,
            orientation_tolerance_deg=pair_prepare.ORIENTATION_TOLERANCE_DEG,
            separation_multiplier=pair_prepare.SEPARATION_MULTIPLIER,
        )
        protocol["sampling"]["absolute_waypoint_banks_index"][bank_id] = waypoints
        protocol["sampling"]["planned_waypoint_evidence_by_bank"][bank_id] = [
            evidence_row(row, role, index)
            for index, (row, role) in enumerate(zip(rows, roles))
        ]
        session = {
            "session_id": session_id,
            "split": "unopened_public_6dof_confirmation",
            "seed": 2026120100 + session_index,
            "reset_id": session_id,
            "absolute_waypoint_bank_id": bank_id,
            "expected_frame_count": 14,
        }
        sessions.append(session)
        plan_sessions.append({**session, "rows": rows})
    protocol["sessions"] = sessions
    protocol["decision_gate"] = {
        "minimum_pair_probability": 0.5,
        "surface_radius_mm": expected["surface_radius_mm"],
        "relative_pair_vector_tolerance_mm": expected[
            "relative_vector_tolerance_mm"
        ],
        "minimum_independent_view_support": expected["minimum_support"],
        "minimum_support_translation_mm": 2.0,
        "minimum_support_angle_deg": 1.0,
        "support_score_increment": 0.1,
        "maximum_counted_support": 2,
    }
    protocol["final_gate"].update(
        minimum_top1_correct_sessions=4,
        minimum_top3_correct_sessions=4,
    )
    protocol["actor_6dof_gate"] = {
        "maximum_median_coordinate_error_mm": 0.001,
        "maximum_p90_coordinate_error_mm": 0.001,
        "maximum_coordinate_error_mm": 0.01,
        "require_acquisition_id_match": True,
        "require_frame_hash_match": True,
        "require_same_step_snapshot": True,
    }
    protocol["change_policy"] = [
        "Do not refit the image model or change geometric thresholds after capture.",
        "Do not expose evaluator geometry or absolute waypoints to the selector.",
        "Do not use confirmation sessions for post-hoc repair.",
        "Do not treat visible SAP and TP surfaces as a validated nerve endpoint.",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = args.output_dir / "s0010_sap_tp_public_6dof_confirmation_v1.yaml"
    protocol_path.write_text(
        yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8"
    )
    seal_path = protocol_path.with_suffix(".sha256")
    seal_path.write_text(
        f"{sha256(protocol_path)}  {protocol_path.name}\n", encoding="utf-8"
    )
    plan_path = args.output_dir / "pose_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": "s0010-sap-tp-public-6dof-confirmation-plan-v1",
                "protocol_sha256": sha256(protocol_path),
                "checkpoint_sha256": sha256(checkpoint),
                "sessions": plan_sessions,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "pose_plan_sha256": sha256(plan_path),
        "checkpoint_sha256": sha256(checkpoint),
        "session_count": 4,
        "target_pose_count": len(targets),
        "clear_negative_pose_count": len(tp_controls) + len(other_controls),
        "target_pool_count": len(target_pool),
        "negative_pool_count": len(negative_pool),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
