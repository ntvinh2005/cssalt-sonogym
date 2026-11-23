#!/usr/bin/env python3
"""Freeze evaluator-designed L4/L5 scans that reproduce published probe motions."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import correct_s0010_sap_observability_v1 as sap_correction
import run_s0010_l4_medial_branch_target as target_workflow
import search_s0010_l4mb_target_viewpoints as search


PROTOCOL_ID = "sonogym-s0010-clinician-scan-feasibility-v1"
LONG_AXIS_ANGLE_RAD = math.pi / 2.0
TRANSVERSE_ANGLE_RAD = 2.95
LABEL_RESOLUTION_MM = 1.5
POSE_MINIMUM = np.asarray([100.0, 70.0, 1.0], dtype=np.float64)
POSE_MAXIMUM = np.asarray([180.0, 200.0, 3.0], dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_candidates(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    **row,
                    "pose": np.asarray(json.loads(row["pose_index_rad"]), dtype=np.float64),
                    "sap": int(row["sap_first_visible_pixels"]),
                    "tp": int(row["tp_first_visible_pixels"]),
                    "margin": int(row["target_margin_pixels"]),
                }
            )
    if not rows:
        raise ValueError(f"candidate file is empty: {path}")
    return rows


def pose_is_valid(pose: np.ndarray) -> bool:
    return bool(
        pose.shape == (3,)
        and np.all(np.isfinite(pose))
        and np.all(pose >= POSE_MINIMUM)
        and np.all(pose <= POSE_MAXIMUM)
        and pose[2] < 2.99
    )


def distinct_anchor_rows(
    rows: list[dict[str, Any]], *, angle_rad: float, count: int
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if pose_is_valid(row["pose"])
        and abs(float(row["pose"][2]) - angle_rad) <= 0.10
        and row["sap"] > 0
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (
            -min(row["sap"], max(row["tp"], 1)),
            -row["sap"],
            -row["tp"],
            -row["margin"],
            tuple(row["pose"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    for row in ranked:
        if not selected or all(
            np.linalg.norm(row["pose"][:2] - item["pose"][:2]) * LABEL_RESOLUTION_MM
            >= 6.0
            for item in selected
        ):
            selected.append(row)
        if len(selected) == count:
            return selected
    raise ValueError(f"fewer than {count} spatially distinct anchors near angle {angle_rad}")


def longitudinal_sweep(anchor: np.ndarray) -> list[np.ndarray]:
    """Move lateral-to-medial while the image plane follows the spine axis."""

    result = []
    for offset in (-8, -6, -4, -2, 0, 2, 4, 6):
        pose = np.asarray([anchor[0] + offset, anchor[1], LONG_AXIS_ANGLE_RAD])
        if not pose_is_valid(pose):
            raise ValueError("longitudinal sweep leaves the native workspace")
        result.append(pose)
    return result


def cranial_caudal_transition(
    anchor: np.ndarray,
    evaluator: Any,
) -> list[np.ndarray]:
    """Select a forward scan with the largest TP change, then reverse it."""

    line = []
    for offset in range(-10, 12, 2):
        pose = np.asarray([anchor[0], anchor[1] + offset, TRANSVERSE_ANGLE_RAD])
        if pose_is_valid(pose):
            line.append((pose, evaluator(pose)))
    if len(line) < 7:
        raise ValueError("cranial-caudal line is too short")
    differences = [
        abs(int(line[index + 1][1]["tp_first_visible_pixels"]) - int(line[index][1]["tp_first_visible_pixels"]))
        for index in range(len(line) - 1)
    ]
    boundary = int(np.argmax(differences))
    start = min(max(boundary - 2, 0), len(line) - 5)
    forward = [line[index][0] for index in range(start, start + 5)]
    return forward + [forward[3], forward[2], forward[1]]


def alternative_view_sweep(anchor: np.ndarray) -> list[np.ndarray]:
    """Rotate through physically acquired views without changing the target position."""

    angles = (LONG_AXIS_ANGLE_RAD, 1.90, 2.20, 2.50, 2.75, 2.95, 2.50, 1.90)
    result = [np.asarray([anchor[0], anchor[1], angle]) for angle in angles]
    if not all(pose_is_valid(pose) for pose in result):
        raise ValueError("alternative-view sweep leaves the native workspace")
    return result


def validate_sequence(poses: list[np.ndarray]) -> None:
    if len(poses) != 8 or not all(pose_is_valid(pose) for pose in poses):
        raise ValueError("each scan sequence must contain eight valid poses")
    for before, after in zip(poses, poses[1:]):
        translation_mm = float(np.linalg.norm(after[:2] - before[:2]) * LABEL_RESOLUTION_MM)
        rotation_deg = math.degrees(abs(float(after[2] - before[2])))
        if translation_mm < 2.0 and rotation_deg < 1.0:
            raise ValueError("consecutive scan views are not physically distinct")


def evidence_row(
    pose: np.ndarray,
    *,
    level: str,
    scan_kind: str,
    evaluator: Any,
) -> dict[str, Any]:
    evidence = evaluator(pose)
    return {
        "declared_role": scan_kind,
        "vertebral_level": level,
        "pose_index_rad": [float(value) for value in pose],
        "evaluator_label": evidence["evaluator_label"],
        "evaluator_reason": evidence["evaluator_reason"],
        "planned_sap_first_visible_pixels": int(evidence["sap_first_visible_pixels"]),
        "planned_tp_first_visible_pixels": int(evidence["tp_first_visible_pixels"]),
        "planned_target_margin_pixels": int(evidence["target_margin_pixels"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--l4-candidates", type=Path, required=True)
    parser.add_argument("--l5-candidates", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--patient-surface-dir", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base = yaml.safe_load(args.base_protocol.read_text(encoding="utf-8"))
    segmentation_image = target_workflow.nib.load(str(args.segmentation))
    segmentation = np.asanyarray(segmentation_image.dataobj).astype(np.uint8)
    surface_y = search.safe_torch(args.patient_surface_dir / "body_lowest_y_array.pt")
    surface_normals = search.safe_torch(
        args.patient_surface_dir / "body_surface_normal_array.pt"
    )
    bone_ids = sap_correction.bone_source_ids(args.label_conversion)

    candidate_paths = {"L4": args.l4_candidates, "L5": args.l5_candidates}
    level_labels = {"L4": 28, "L5": 27}
    sessions = []
    banks: dict[str, list[list[float]]] = {}
    evidence_banks: dict[str, list[dict[str, Any]]] = {}
    design_summary: dict[str, Any] = {}

    for level in ("L4", "L5"):
        _, sap_mask, tp_mask, contract = target_workflow.load_target_geometry(
            args.prediction,
            args.segmentation,
            level_label=level_labels[level],
            level_name=level,
        )

        def evaluator(pose: np.ndarray) -> dict[str, Any]:
            return search.exact_pose_evidence(
                pose,
                surface_y=surface_y,
                surface_normals=surface_normals,
                segmentation=segmentation,
                bone_ids=bone_ids,
                sap_mask=sap_mask,
                tp_mask=tp_mask,
                contract=contract,
            )

        candidates = read_candidates(candidate_paths[level])
        longitudinal = distinct_anchor_rows(
            candidates, angle_rad=LONG_AXIS_ANGLE_RAD, count=2
        )
        transverse = distinct_anchor_rows(
            candidates, angle_rad=TRANSVERSE_ANGLE_RAD, count=2
        )
        level_plans = []
        for variant in range(2):
            long_poses = longitudinal_sweep(longitudinal[variant]["pose"])
            transition_poses = cranial_caudal_transition(
                transverse[variant]["pose"], evaluator
            )
            alternative_poses = alternative_view_sweep(transverse[variant]["pose"])
            for poses in (long_poses, transition_poses, alternative_poses):
                validate_sequence(poses)
            all_poses = long_poses + transition_poses + alternative_poses
            bank_id = f"{level.lower()}_bank_{variant:02d}"
            session_id = f"s0010clinical{level.lower()}_{variant:02d}"
            banks[bank_id] = [pose.tolist() for pose in all_poses]
            evidence_banks[bank_id] = []
            for scan_kind, poses in (
                ("LONGITUDINAL_LATERAL_SWEEP", long_poses),
                ("TRANSVERSE_CRANIAL_CAUDAL_SWEEP", transition_poses),
                ("ALTERNATIVE_VIEW_CONFIRMATION", alternative_poses),
            ):
                evidence_banks[bank_id].extend(
                    evidence_row(
                        pose,
                        level=level,
                        scan_kind=scan_kind,
                        evaluator=evaluator,
                    )
                    for pose in poses
                )
            sessions.append(
                {
                    "session_id": session_id,
                    "split": "opened_clinician_scan_feasibility",
                    "seed": 2026112300 + len(sessions),
                    "reset_id": session_id,
                    "vertebral_level": level,
                    "absolute_waypoint_bank_id": bank_id,
                    "expected_frame_count": 24,
                }
            )
            level_plans.append(
                {
                    "session_id": session_id,
                    "longitudinal_anchor": longitudinal[variant]["pose"].tolist(),
                    "transverse_anchor": transverse[variant]["pose"].tolist(),
                }
            )
        design_summary[level] = level_plans

    protocol = deepcopy(base)
    protocol["protocol_id"] = PROTOCOL_ID
    protocol["status"] = "FROZEN_OPENED_CLINICIAN_SCAN_FEASIBILITY"
    protocol["purpose"] = (
        "Test whether real longitudinal, transverse, cranial-caudal and "
        "alternative-view probe motions reveal reproducible L4/L5 TP/SAP transitions."
    )
    protocol["clinical_scope"] = {
        "structures": ["RIGHT_L4_TP", "RIGHT_L4_SAP", "RIGHT_L5_TP", "RIGHT_L5_SAP"],
        "needle_or_injection": False,
        "clinical_target_validated": False,
    }
    protocol["source"] = {
        "implementation_commit": args.implementation_commit,
        "base_protocol_sha256": sha256(args.base_protocol),
        "l4_candidates": str(args.l4_candidates.resolve()),
        "l4_candidates_sha256": sha256(args.l4_candidates),
        "l5_candidates": str(args.l5_candidates.resolve()),
        "l5_candidates_sha256": sha256(args.l5_candidates),
        "reference": "ASRA lumbar medial branch ultrasound scanning workflow",
        "evidence_status": "privileged_opened_feasibility_capture_design",
    }
    sampling = protocol["sampling"]
    sampling["expected_frames_per_session"] = 24
    sampling["trajectory_start_frames"] = [0, 8, 16]
    sampling["absolute_waypoint_banks_index"] = banks
    sampling["planned_waypoint_evidence_by_bank"] = evidence_banks
    sampling["maximum_initial_positioning_steps"] = 700
    sampling["maximum_action_steps_per_waypoint"] = 250
    sampling["maximum_total_action_steps_per_session"] = 7000
    sampling["episode_length_s_override"] = 700.0
    sampling["physical_controller"]["ceiling_assisted_minimum_requested_angle_rad"] = 2.99
    sampling["physical_controller"]["capture_angle_ceiling_limited_samples"] = False
    protocol["sessions"] = sessions
    protocol["change_policy"] = [
        "This capture is opened feasibility evidence, never independent confirmation.",
        "Absolute waypoints and anatomical masks are experiment-design and evaluator-only inputs.",
        "No pose at or above 2.99 rad is allowed because the current controller cannot track it fail-closed.",
        "Repeated return poses demonstrate reappearance but do not count as independent views.",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = args.output_dir / "s0010_clinician_scan_feasibility_v1.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    protocol_path.with_suffix(".sha256").write_text(
        f"{sha256(protocol_path)}  {protocol_path.name}\n", encoding="utf-8"
    )
    result = {
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "session_count": len(sessions),
        "frame_count": sum(item["expected_frame_count"] for item in sessions),
        "trajectory_count": len(sessions) * 3,
        "design": design_summary,
        "long_axis_angle_rad": LONG_AXIS_ANGLE_RAD,
        "transverse_angle_rad": TRANSVERSE_ANGLE_RAD,
        "transverse_offset_from_orthogonal_deg": math.degrees(math.pi - TRANSVERSE_ANGLE_RAD),
        "anatomical_labels": "SIMULATOR_WEAK_ANATOMICAL_LABEL",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
