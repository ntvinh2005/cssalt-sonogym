#!/usr/bin/env python3
"""Freeze targeted SAP repair captures plus unopened evaluation sessions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LEVELS = ("L4", "L5")
SESSION_DESIGN = (
    ("s0010sapval_train_03", "train", 30),
    ("s0010sapval_train_04", "train", 30),
    ("s0010sapval_train_05", "train", 30),
    ("s0010sapval_calibration_01", "calibration", 30),
    ("s0010sapval_evaluation_00", "evaluation", 20),
    ("s0010sapval_evaluation_01", "evaluation", 20),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def accepted_base_poses(
    manifest: Path, source_protocol: dict[str, Any]
) -> dict[str, list[np.ndarray]]:
    sessions = {item["session_id"]: item for item in source_protocol["sessions"]}
    result: dict[str, list[np.ndarray]] = {level: [] for level in LEVELS}
    seen: dict[str, set[tuple[float, float, float]]] = {level: set() for level in LEVELS}
    for row in read_csv(manifest):
        if row["selected_for_dataset"].lower() != "true":
            continue
        if row["provisional_label"] != "SAP_RELATED_CORTICAL_INTERFACE":
            continue
        level = row["conditioned_level"]
        session = sessions[row["session_id"]]
        bank = source_protocol["sampling"]["absolute_waypoint_banks_index"][
            session["absolute_waypoint_bank_id"]
        ]
        pose = tuple(float(value) for value in bank[int(row["frame_sequence"])])
        if pose not in seen[level]:
            seen[level].add(pose)
            result[level].append(np.asarray(pose, dtype=np.float64))
    if any(not values for values in result.values()):
        raise ValueError("each level requires at least one accepted development pose")
    return result


def targeted_pool(level: str, bases: list[np.ndarray]) -> list[list[float]]:
    xs = np.arange(143.0, 165.6, 2.5)
    zs = np.arange(135.0, 152.6, 2.5) if level == "L4" else np.arange(118.0, 135.6, 2.5)
    angles = np.asarray([2.74, 2.79, 2.84, 2.89, 2.94, 2.99])
    candidates = [
        np.asarray([x, z, angle], dtype=np.float64)
        for x in xs
        for z in zs
        for angle in angles
    ]
    def distance(row: np.ndarray) -> float:
        return min(
            float(
                np.linalg.norm(
                    (row - base) / np.asarray([4.0, 4.0, 0.08], dtype=np.float64)
                )
            )
            for base in bases
        )
    ordered = sorted(candidates, key=lambda row: (distance(row), tuple(row)))
    required = sum(item[2] for item in SESSION_DESIGN)
    if len(ordered) < required:
        raise AssertionError("targeted grid is smaller than the frozen requirement")
    return [[float(value) for value in row] for row in ordered[:required]]


def allocate_banks(pools: dict[str, list[list[float]]]) -> dict[str, list[list[float]]]:
    banks = {session_id: [] for session_id, _, _ in SESSION_DESIGN}
    for level in LEVELS:
        cursor = 0
        for session_id, _, per_level in SESSION_DESIGN:
            banks[session_id].extend(pools[level][cursor : cursor + per_level])
            cursor += per_level
        if cursor != len(pools[level]):
            raise AssertionError("targeted pool allocation mismatch")
    return banks


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-capture-job", type=int, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_protocol = yaml.safe_load(args.source_protocol.read_text(encoding="utf-8"))
    bases = accepted_base_poses(args.source_manifest, source_protocol)
    pools = {level: targeted_pool(level, bases[level]) for level in LEVELS}
    banks = allocate_banks(pools)
    sessions = []
    for index, (session_id, split, per_level) in enumerate(SESSION_DESIGN):
        sessions.append(
            {
                "session_id": session_id,
                "split": split,
                "seed": 2026112201 + index,
                "reset_id": session_id,
                "absolute_waypoint_bank_id": session_id,
                "expected_frame_count": 2 * per_level,
            }
        )
    document = {
        "protocol_id": "sonogym-s0010-sap-validation-followup-v1",
        "status": "PRE_GPU_FROZEN",
        "purpose": "Target matched development poses and preserve two unopened evaluation sessions.",
        "source": {
            "implementation_commit": args.implementation_commit,
            "development_capture_job": args.source_capture_job,
            "development_manifest_sha256": file_sha256(args.source_manifest),
            "target_selection": "distance_to_development_matched_positive_poses",
            "prediction_job": 52078067,
            "prediction_sha256": "a78daeae9159f0df28ba6fffcb8e8d7ef6c2e87f490c9598acfcb82efad1677f",
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
            "physical_tracking_position_tolerance_mm": 1.5,
            "physical_tracking_orientation_tolerance_deg": 1.2,
            "waypoint_separation_multiplier": 2.0,
            "minimum_realized_viewpoint_translation_mm": 2.0,
            "minimum_realized_viewpoint_orientation_deg": 1.0,
            "maximum_initial_positioning_steps": 350,
            "maximum_action_steps_per_waypoint": 150,
            "maximum_total_action_steps_per_session": 12000,
            "trajectory_seed_stride": 1000,
            "enforce_episode_action_budget": True,
            "waypoint_level_order_by_session": {
                session_id: [
                    {"level": "L4", "start_index": 0, "stop_index_exclusive": per_level},
                    {"level": "L5", "start_index": per_level, "stop_index_exclusive": 2 * per_level},
                ]
                for session_id, _, per_level in SESSION_DESIGN
            },
            "trajectory_start_frames_by_session": {
                session_id: list(range(0, 2 * per_level, 2))
                for session_id, _, per_level in SESSION_DESIGN
            },
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
        "dataset_gate": {
            "require_all_expected_frames": True,
            "minimum_distinct_frame_hashes_per_session": 40,
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
            "Do not inspect evaluation images before model and threshold are frozen.",
            "Do not expose privileged geometry to classifier features.",
            "Do not reuse development failures as independent confirmation.",
        ],
    }
    args.output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    digest = file_sha256(args.output)
    args.output.with_suffix(".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    summary = {
        "accepted_base_pose_counts": {level: len(values) for level, values in bases.items()},
        "captured_frames": sum(len(values) for values in banks.values()),
        "evaluation_frames_sealed": sum(
            2 * count for _, split, count in SESSION_DESIGN if split == "evaluation"
        ),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
