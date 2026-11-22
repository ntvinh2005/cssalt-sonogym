#!/usr/bin/env python3
"""Freeze public-6DoF L4 SAP/TP development capture groups."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import yaml

import correct_s0010_sap_observability_v1 as sap_correction
import prepare_s0010_landmark_graph_confirmation as graph_prepare
import prepare_s0010_sap_tp_6dof_confirmation as previous
import prepare_s0010_sap_tp_pair_confirmation as pair_prepare
import run_s0010_l4_medial_branch_target as target_workflow
import search_s0010_l4mb_target_viewpoints as search
from two_stage_navigation.acquisition_integrity import validate_waypoint_bank


PROTOCOL_ID = "sonogym-s0010-l4-landmark-graph-development-v1"


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
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base = yaml.safe_load(args.base_protocol.read_text(encoding="utf-8"))
    negative_protocol = yaml.safe_load(
        args.negative_protocol.read_text(encoding="utf-8")
    )
    _, sap_mask, tp_mask, contract = target_workflow.load_target_geometry(
        args.prediction, args.segmentation, level_label=28, level_name="L4"
    )
    bone_ids = sap_correction.bone_source_ids(args.label_conversion)
    segmentation = np.asanyarray(
        target_workflow.nib.load(str(args.segmentation)).dataobj
    ).astype(np.uint8)
    surface_y = search.safe_torch(args.patient_surface_dir / "body_lowest_y_array.pt")
    surface_normals = search.safe_torch(
        args.patient_surface_dir / "body_surface_normal_array.pt"
    )
    excluded = []
    for path in args.exclude_protocol:
        excluded.extend(graph_prepare.protocol_poses(path))
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
    tp_controls = graph_prepare.select_available_pairs(
        tp_only, 2, excluded + targets
    )
    if len(tp_controls) < 4:
        raise ValueError(f"fewer than four independent L4 TP-only controls: {len(tp_controls)}")
    other_controls = graph_prepare.select_available_pairs(
        other, 10, excluded + targets + tp_controls
    )
    if len(other_controls) < 20:
        raise ValueError(
            f"fewer than twenty independent L4 other-bone controls: {len(other_controls)}"
        )

    protocol = deepcopy(base)
    protocol["protocol_id"] = PROTOCOL_ID
    protocol["status"] = "FROZEN_OPENED_DEVELOPMENT_CAPTURE"
    protocol["purpose"] = (
        "Collect public-6DoF L4 SAP/TP development observations for a "
        "session-separated landmark-graph experiment."
    )
    protocol["source"] = {
        "implementation_commit": args.implementation_commit,
        "exact_candidates": str(args.exact_candidates.resolve()),
        "exact_candidates_sha256": previous.sha256(args.exact_candidates),
        "base_protocol_sha256": previous.sha256(args.base_protocol),
        "evidence_status": "privileged_pose_design_for_opened_development",
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
    sessions = []
    for session_index in range(4):
        rows = (
            targets[session_index * 8 : (session_index + 1) * 8]
            + tp_controls[session_index : session_index + 1]
            + other_controls[session_index * 5 : (session_index + 1) * 5]
        )
        roles = ["L4_SAP_TP_SURFACE_CANDIDATE"] * 8 + ["CLEAR_NEGATIVE"] * 6
        bank_id = f"bank_{session_index:02d}"
        session_id = f"s0010l4graphdev_{session_index:02d}"
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
        sessions.append(
            {
                "session_id": session_id,
                "split": "opened_l4_landmark_graph_development",
                "seed": 2026120300 + session_index,
                "reset_id": session_id,
                "absolute_waypoint_bank_id": bank_id,
                "expected_frame_count": 14,
            }
        )
    protocol["sessions"] = sessions
    protocol["change_policy"] = [
        "These sessions are opened development evidence, never independent confirmation.",
        "Absolute waypoints and evaluator masks must not enter actor features.",
        "The L5 SAP image checkpoint may be tested only as an explicit transfer baseline.",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = args.output_dir / "s0010_l4_landmark_graph_development_v1.yaml"
    protocol_path.write_text(
        yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8"
    )
    seal_path = protocol_path.with_suffix(".sha256")
    seal_path.write_text(
        f"{previous.sha256(protocol_path)}  {protocol_path.name}\n", encoding="utf-8"
    )
    manifest = {
        "protocol": str(protocol_path),
        "protocol_sha256": previous.sha256(protocol_path),
        "session_count": 4,
        "target_pose_count": len(targets),
        "tp_only_pose_count": len(tp_controls),
        "other_bone_pose_count": len(other_controls),
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
