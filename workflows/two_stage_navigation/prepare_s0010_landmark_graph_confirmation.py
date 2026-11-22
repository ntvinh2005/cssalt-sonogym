#!/usr/bin/env python3
"""Freeze fresh L5 SAP/TP capture groups for the landmark-graph selector."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

import correct_s0010_sap_observability_v1 as sap_correction
import prepare_s0010_sap_tp_6dof_confirmation as previous
import prepare_s0010_sap_tp_pair_confirmation as pair_prepare
import run_s0010_l4_medial_branch_target as target_workflow
import search_s0010_l4mb_target_viewpoints as search
from two_stage_navigation.acquisition_integrity import validate_waypoint_bank


PROTOCOL_ID = "sonogym-s0010-l5-landmark-graph-confirmation-v1r5"


def protocol_poses(path: Path) -> list[dict[str, Any]]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    return previous.protocol_poses(protocol)


def select_available_pairs(
    rows: list[dict[str, Any]],
    maximum_pair_count: int,
    excluded: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select every robust distinct pair available, up to a frozen maximum."""

    chosen: list[dict[str, Any]] = []
    ordered = sorted(
        rows,
        key=lambda row: (
            -min(
                int(row.get("sap_first_visible_pixels", 0)),
                int(row.get("tp_first_visible_pixels", 0)),
            ),
            -int(row.get("target_margin_pixels", -1)),
            tuple(pair_prepare.pose(row)),
        ),
    )
    for first in ordered:
        if not pair_prepare.new_against(first, excluded + chosen):
            continue
        second = next(
            (
                candidate
                for candidate in ordered
                if candidate is not first
                and pair_prepare.new_against(candidate, excluded + chosen)
                and pair_prepare.distinct(first, candidate)
                and pair_prepare.capture_pair_is_robust(first, candidate)
            ),
            None,
        )
        if second is None:
            continue
        chosen.extend((first, second))
        if len(chosen) == maximum_pair_count * 2:
            break
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--exclude-protocol", type=Path, action="append", default=[])
    parser.add_argument("--negative-protocol", type=Path, required=True)
    parser.add_argument("--exact-candidates", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--patient-surface-dir", type=Path, required=True)
    parser.add_argument("--graph-checkpoint", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base = yaml.safe_load(args.base_protocol.read_text(encoding="utf-8"))
    negative_protocol = yaml.safe_load(
        args.negative_protocol.read_text(encoding="utf-8")
    )
    bundle = joblib.load(args.graph_checkpoint)
    if bundle.get("schema") != "s0010-l5-multiview-landmark-graph-v1":
        raise ValueError("unexpected landmark-graph checkpoint schema")
    required = {
        "top_k_per_landmark": 5,
        "cluster_radius_mm": 3.0,
        "cluster_candidate_limit": 250,
        "minimum_independent_support": 2,
        "minimum_consensus_score": 0.75,
    }
    if any(bundle.get(key) != value for key, value in required.items()):
        raise ValueError("checkpoint configuration differs from frozen development")

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

    excluded = previous.protocol_poses(base)
    for path in args.exclude_protocol:
        excluded.extend(protocol_poses(path))
    target_pool = [
        row
        for row in pair_prepare.read_exact_candidates(args.exact_candidates)
        if row["sap_first_visible_pixels"] >= 3
        and row["tp_first_visible_pixels"] >= 3
        and row["target_margin_pixels"] >= 4
    ]
    target_candidates = pair_prepare.select_pairs(target_pool, 17, excluded)
    targets = target_candidates[:32]
    # The original session-03 initial command [158, 114, 2.05] plateaued at
    # roughly three degrees orientation error in three technical attempts.
    # Replace it before image acquisition with the next identically eligible
    # reserve, without inspecting any model or evaluator outcome at that pose.
    unreachable_target = targets[24]
    targets[24] = target_candidates[32]
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
    tp_controls = pair_prepare.select_pairs(tp_only, 2, excluded + targets)
    other_controls = select_available_pairs(
        other, 24, excluded + targets + tp_controls
    )
    if len(other_controls) < 32:
        raise ValueError(
            "fewer than 32 physically distinct other-bone controls remain: "
            f"{len(other_controls)}"
        )

    protocol = deepcopy(base)
    protocol["protocol_id"] = PROTOCOL_ID
    protocol["status"] = "FROZEN_BEFORE_FRESH_CAPTURE"
    protocol["purpose"] = (
        "Independently confirm the frozen L5 TP/SAP landmark graph with fresh "
        "ultrasound and synchronized actor-visible robot 6DoF poses."
    )
    protocol["source"] = {
        "implementation_commit": args.implementation_commit,
        "graph_checkpoint": str(args.graph_checkpoint.resolve()),
        "graph_checkpoint_sha256": previous.sha256(args.graph_checkpoint),
        "tp_checkpoint_sha256": bundle["tp_checkpoint_sha256"],
        "sap_checkpoint_sha256": bundle["sap_checkpoint_sha256"],
        "base_protocol_sha256": previous.sha256(args.base_protocol),
        "exact_candidates_sha256": previous.sha256(args.exact_candidates),
        "opened_development_sessions": bundle["opened_development_session_ids"],
        "source_evidence_status": "frozen_after_opened_development",
        "technical_waypoint_replacement": {
            "excluded_pose_index_rad": unreachable_target["pose_index_rad"],
            "replacement_pose_index_rad": targets[24]["pose_index_rad"],
            "reason": "controller_orientation_plateau_before_image_acquisition",
        },
        "technical_seed_replacement": {
            "session_id": "s0010l5graph_03",
            "excluded_seed": 2026120203,
            "replacement_seed": 2026120213,
            "reason": "reset_specific_controller_plateau_before_image_acquisition",
        },
    }
    protocol["information_boundaries"]["actor_inputs"] = [
        "clean_ultrasound",
        "synchronized_base_to_end_effector_pose_from_robot_proprioception",
        "public_probe_image_calibration",
        "prior_actor_observations",
    ]
    protocol["sampling"]["absolute_waypoint_banks_index"] = {}
    protocol["sampling"]["planned_waypoint_evidence_by_bank"] = {}
    protocol["sampling"]["trajectory_start_frames"] = list(range(0, 14, 2))
    protocol["sampling"]["maximum_initial_positioning_steps"] = 700
    protocol["sampling"]["episode_length_s_override"] = 500.0
    sessions: list[dict[str, Any]] = []
    plan_sessions: list[dict[str, Any]] = []
    for session_index in range(4):
        rows = (
            targets[session_index * 8 : (session_index + 1) * 8]
            + tp_controls[session_index : session_index + 1]
            + other_controls[session_index * 5 : (session_index + 1) * 5]
        )
        roles = ["SAP_TP_SURFACE_CANDIDATE"] * 8 + ["CLEAR_NEGATIVE"] * 6
        bank_id = f"bank_{session_index:02d}"
        session_id = f"s0010l5graph_{session_index:02d}"
        waypoints = [row["pose_index_rad"] for row in rows]
        validate_waypoint_bank(
            waypoints,
            command_minimum=[-1.0e9] * 3,
            command_maximum=[1.0e9] * 3,
            label_resolution_mm=pair_prepare.LABEL_RESOLUTION_MM,
            position_tolerance_mm=pair_prepare.POSITION_TOLERANCE_MM,
            orientation_tolerance_deg=pair_prepare.ORIENTATION_TOLERANCE_DEG,
            separation_multiplier=pair_prepare.SEPARATION_MULTIPLIER,
        )
        protocol["sampling"]["absolute_waypoint_banks_index"][bank_id] = waypoints
        protocol["sampling"]["planned_waypoint_evidence_by_bank"][bank_id] = [
            previous.evidence_row(row, role, index)
            for index, (row, role) in enumerate(zip(rows, roles))
        ]
        session = {
            "session_id": session_id,
            "split": "unopened_l5_landmark_graph_target_confirmation",
            "seed": 2026120213 if session_index == 3 else 2026120200 + session_index,
            "reset_id": session_id,
            "absolute_waypoint_bank_id": bank_id,
            "expected_frame_count": 14,
        }
        sessions.append(session)
        plan_sessions.append({**session, "rows": rows})
    negative_controls = other_controls[20:]
    negative_split = len(negative_controls) // 2
    negative_session_rows = [
        negative_controls[:negative_split], negative_controls[negative_split:]
    ]
    for negative_index, rows in enumerate(negative_session_rows):
        session_index = 4 + negative_index
        roles = ["CLEAR_NEGATIVE"] * len(rows)
        bank_id = f"bank_{session_index:02d}"
        session_id = f"s0010l5graph_{session_index:02d}"
        waypoints = [row["pose_index_rad"] for row in rows]
        validate_waypoint_bank(
            waypoints,
            command_minimum=[-1.0e9] * 3,
            command_maximum=[1.0e9] * 3,
            label_resolution_mm=pair_prepare.LABEL_RESOLUTION_MM,
            position_tolerance_mm=pair_prepare.POSITION_TOLERANCE_MM,
            orientation_tolerance_deg=pair_prepare.ORIENTATION_TOLERANCE_DEG,
            separation_multiplier=pair_prepare.SEPARATION_MULTIPLIER,
        )
        protocol["sampling"]["absolute_waypoint_banks_index"][bank_id] = waypoints
        protocol["sampling"]["planned_waypoint_evidence_by_bank"][bank_id] = [
            previous.evidence_row(row, role, index)
            for index, (row, role) in enumerate(zip(rows, roles))
        ]
        session = {
            "session_id": session_id,
            "split": "unopened_l5_landmark_graph_negative_confirmation",
            "seed": 2026120200 + session_index,
            "reset_id": session_id,
            "absolute_waypoint_bank_id": bank_id,
            "expected_frame_count": len(rows),
        }
        sessions.append(session)
        plan_sessions.append({**session, "rows": rows})
    protocol["sessions"] = sessions
    protocol["landmark_graph"] = required
    protocol["landmark_graph"]["minimum_view_translation_mm"] = 2.0
    protocol["landmark_graph"]["minimum_view_angle_deg"] = 1.0
    protocol["final_gate"] = {
        "minimum_top1_correct_sessions": 4,
        "minimum_top3_correct_sessions": 4,
        "minimum_accepted_correct_sessions": 4,
        "maximum_wrong_accepted_sessions": 0,
        "minimum_pair_precision": 0.75,
        "maximum_clear_negative_false_accept_rate": 0.10,
        "minimum_actual_targets_per_target_session": 4,
        "minimum_actual_clear_negatives_per_target_session": 4,
        "minimum_actual_clear_negatives_per_negative_session": max(
            4, min(len(rows) for rows in negative_session_rows) - 2
        ),
    }
    protocol["change_policy"] = [
        "Do not refit any checkpoint or change graph thresholds after capture.",
        "Do not expose evaluator geometry or absolute waypoints to the selector.",
        "Do not use confirmation sessions for post-hoc repair and still call them confirmation.",
        "Do not treat SAP/TP bone evidence as a validated nerve endpoint.",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = args.output_dir / "s0010_l5_landmark_graph_confirmation_v1r5.yaml"
    protocol_path.write_text(
        yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8"
    )
    seal_path = protocol_path.with_suffix(".sha256")
    seal_path.write_text(
        f"{previous.sha256(protocol_path)}  {protocol_path.name}\n", encoding="utf-8"
    )
    plan_path = args.output_dir / "pose_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": "s0010-l5-landmark-graph-confirmation-plan-v1",
                "protocol_sha256": previous.sha256(protocol_path),
                "graph_checkpoint_sha256": previous.sha256(args.graph_checkpoint),
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
        "protocol_sha256": previous.sha256(protocol_path),
        "pose_plan_sha256": previous.sha256(plan_path),
        "graph_checkpoint_sha256": previous.sha256(args.graph_checkpoint),
        "session_count": 6,
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
