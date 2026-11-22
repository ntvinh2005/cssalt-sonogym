#!/usr/bin/env python3
"""Freeze the image-only pair model and select new evaluator-designed poses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

import correct_s0010_sap_observability_v1 as sap_correction
import run_s0010_l4_medial_branch_target as target_workflow
import run_s0010_sap_tp_pair_development as development
import search_s0010_l4mb_target_viewpoints as search
from two_stage_navigation.ultrasound_sap_tp_configuration import (
    track_window_pair_candidates,
)
from two_stage_navigation.acquisition_integrity import validate_waypoint_bank


LABEL_RESOLUTION_MM = 1.5
POSITION_TOLERANCE_MM = 1.5
ORIENTATION_TOLERANCE_DEG = 1.2
SEPARATION_MULTIPLIER = 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pose(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(row["pose_index_rad"], dtype=np.float64)


def pose_distances(first: dict[str, Any], second: dict[str, Any]) -> tuple[float, float]:
    delta = pose(first) - pose(second)
    translation = float(np.linalg.norm(delta[:2]) * 1.5)
    angle = float(np.degrees(abs(np.arctan2(np.sin(delta[2]), np.cos(delta[2])))))
    return translation, angle


def distinct(first: dict[str, Any], second: dict[str, Any]) -> bool:
    translation, angle = pose_distances(first, second)
    return bool(
        translation > SEPARATION_MULTIPLIER * POSITION_TOLERANCE_MM
        or angle > SEPARATION_MULTIPLIER * ORIENTATION_TOLERANCE_DEG
    )


def nearby(first: dict[str, Any], second: dict[str, Any]) -> bool:
    translation, angle = pose_distances(first, second)
    return bool(translation <= 15.0 and angle <= 15.0)


def new_against(row: dict[str, Any], excluded: list[dict[str, Any]]) -> bool:
    return all(distinct(row, previous) for previous in excluded)


def select_pairs(
    rows: list[dict[str, Any]],
    pair_count: int,
    excluded: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    ordered = sorted(
        rows,
        key=lambda row: (
            -min(
                int(row.get("sap_first_visible_pixels", 0)),
                int(row.get("tp_first_visible_pixels", 0)),
            ),
            -int(row.get("target_margin_pixels", -1)),
            tuple(pose(row)),
        ),
    )
    for first in ordered:
        if not new_against(first, excluded + chosen):
            continue
        second = next(
            (
                candidate
                for candidate in ordered
                if candidate is not first
                and new_against(candidate, excluded + chosen)
                and distinct(first, candidate)
                and nearby(first, candidate)
            ),
            None,
        )
        if second is None:
            continue
        chosen.extend((first, second))
        if len(chosen) == pair_count * 2:
            return chosen
    raise ValueError(f"could not select {pair_count} physically distinct pose pairs")


def old_training_rows(
    protocol: dict[str, Any],
    run_root: Path,
    sap_mask: np.ndarray,
    tp_mask: np.ndarray,
    contract: Any,
    bone_ids: set[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session in protocol["sessions"]:
        session_dir = run_root / "sessions" / session["session_id"]
        images = np.load(session_dir / "public/ultrasound.npz")["ultrasound"]
        coordinates = np.load(
            session_dir / "privileged/pixel_to_volume_coordinates.npz"
        )["human_img_coords"]
        raw_labels = np.load(session_dir / "privileged/raw_labels.npz")["raw_labels"]
        for index, (image, coordinate_map, raw) in enumerate(
            zip(images, coordinates, raw_labels)
        ):
            projected = target_workflow.projected_masks(
                coordinate_map, contract, sap_mask, tp_mask
            )
            first_bone = target_workflow.first_bone_mask(raw, bone_ids)
            sap_visible = projected["sap"] & first_bone
            tp_visible = projected["tp"] & first_bone
            pairs = [
                development._pair_record(
                    first,
                    second,
                    feature,
                    tp_visible,
                    sap_visible,
                    {
                        "maximum_segment_surface_distance_pixels": 4.0,
                        "minimum_aligned_segment_points": 3,
                    },
                )
                for first, second, feature in track_window_pair_candidates(
                    image,
                    window_columns=15,
                    stride_columns=5,
                    minimum_center_separation_columns=8,
                    maximum_tracks=2,
                )
            ]
            rows.append({"frame_sequence": index, "pairs": pairs})
    return rows


def read_exact_candidates(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = dict(row)
            parsed["pose_index_rad"] = json.loads(row["pose_index_rad"])
            for key in (
                "sap_first_visible_pixels",
                "tp_first_visible_pixels",
                "target_margin_pixels",
            ):
                parsed[key] = int(row[key])
            rows.append(parsed)
    return rows


def negative_candidates(
    protocol: dict[str, Any],
    *,
    surface_y: np.ndarray,
    surface_normals: np.ndarray,
    segmentation: np.ndarray,
    bone_ids: set[int],
    sap_mask: np.ndarray,
    tp_mask: np.ndarray,
    contract: Any,
) -> list[dict[str, Any]]:
    bases: list[np.ndarray] = []
    for bank_id, bank in protocol["sampling"]["absolute_waypoint_banks_index"].items():
        roles = protocol["sampling"]["planned_waypoint_evidence_by_bank"][bank_id]
        for waypoint, role in zip(bank, roles):
            if role["declared_role"] == "NEGATIVE_CONTROL":
                bases.append(np.asarray(waypoint, dtype=np.float64))
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    offsets = [
        np.asarray([dx, dz, da], dtype=np.float64)
        for dx in (-2.0, 0.0, 2.0)
        for dz in (-2.0, 0.0, 2.0)
        for da in (-0.05, 0.0, 0.05)
        if (dx, dz, da) != (0.0, 0.0, 0.0)
    ]
    for base in bases:
        for offset in offsets:
            candidate_pose = base + offset
            key = tuple(float(round(value, 6)) for value in candidate_pose)
            if key in seen or not (1.0 <= candidate_pose[2] <= 3.0):
                continue
            seen.add(key)
            evidence = search.exact_pose_evidence(
                candidate_pose,
                surface_y=surface_y,
                surface_normals=surface_normals,
                segmentation=segmentation,
                bone_ids=bone_ids,
                sap_mask=sap_mask,
                tp_mask=tp_mask,
                contract=contract,
            )
            if evidence["evaluator_label"] != "PROXY_NEGATIVE":
                continue
            if evidence["evaluator_reason"] not in {
                "TP_ONLY_WITHOUT_SAP_IN_PLANE",
                "OTHER_BONE_OR_BACKGROUND",
            }:
                continue
            candidates.append(evidence)
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--exact-candidates", type=Path, required=True)
    parser.add_argument("--negative-protocol", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--patient-surface-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    training_protocol = yaml.safe_load(args.training_protocol.read_text(encoding="utf-8"))
    negative_protocol = yaml.safe_load(args.negative_protocol.read_text(encoding="utf-8"))
    _, sap_mask, tp_mask, contract = target_workflow.load_target_geometry(
        args.prediction, args.segmentation
    )
    bone_ids = sap_correction.bone_source_ids(args.label_conversion)
    rows = old_training_rows(
        training_protocol,
        args.training_run,
        sap_mask,
        tp_mask,
        contract,
        bone_ids,
    )
    model, feature_mean, feature_scale = development._fit_model(
        rows,
        set(range(12)),
        seed=20261122,
        trees=240,
        maximum_negative_pairs_per_frame=20,
    )
    checkpoint = args.output_dir / "sap_tp_pair_model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
            "threshold": 0.50,
            "training_protocol_sha256": sha256(args.training_protocol),
        },
        checkpoint,
    )

    old_poses = [
        {"pose_index_rad": waypoint}
        for bank in training_protocol["sampling"]["absolute_waypoint_banks_index"].values()
        for waypoint in bank
    ]
    exact = read_exact_candidates(args.exact_candidates)
    target_pool = [
        row
        for row in exact
        if row["sap_first_visible_pixels"] >= 3
        and row["tp_first_visible_pixels"] >= 3
        and row["target_margin_pixels"] >= 4
    ]
    targets = select_pairs(target_pool, 12, old_poses)

    segmentation = np.asanyarray(
        target_workflow.nib.load(str(args.segmentation)).dataobj
    ).astype(np.uint8)
    surface_y = search.safe_torch(args.patient_surface_dir / "body_lowest_y_array.pt")
    surface_normals = search.safe_torch(
        args.patient_surface_dir / "body_surface_normal_array.pt"
    )
    negative_pool = negative_candidates(
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
    tp_controls = select_pairs(tp_only, 5, old_poses + targets)[:9]
    other_controls = select_pairs(other, 5, old_poses + targets + tp_controls)[:9]

    sessions: list[dict[str, Any]] = []
    for session_index in range(3):
        session_targets = targets[session_index * 8 : (session_index + 1) * 8]
        session_tp = tp_controls[session_index * 3 : (session_index + 1) * 3]
        session_other = other_controls[session_index * 3 : (session_index + 1) * 3]
        rows_for_session = [
            {**row, "declared_role": "SAP_TP_SURFACE_CANDIDATE"}
            for row in session_targets
        ] + [
            {**row, "declared_role": "CLEAR_NEGATIVE"}
            for row in session_tp + session_other
        ]
        validate_waypoint_bank(
            [row["pose_index_rad"] for row in rows_for_session],
            command_minimum=[-1.0e9, -1.0e9, -1.0e9],
            command_maximum=[1.0e9, 1.0e9, 1.0e9],
            label_resolution_mm=LABEL_RESOLUTION_MM,
            position_tolerance_mm=POSITION_TOLERANCE_MM,
            orientation_tolerance_deg=ORIENTATION_TOLERANCE_DEG,
            separation_multiplier=SEPARATION_MULTIPLIER,
        )
        sessions.append(
            {
                "session_id": f"s0010saptppair_{session_index:02d}",
                "seed": 2026112700 + session_index,
                "rows": rows_for_session,
            }
        )
    plan = {
        "schema": "s0010-sap-tp-pair-confirmation-plan-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "training_frame_count": len(rows),
        "training_positive_pair_count": sum(
            pair["positive"] for row in rows for pair in row["pairs"]
        ),
        "training_protocol_sha256": sha256(args.training_protocol),
        "exact_candidates_sha256": sha256(args.exact_candidates),
        "negative_protocol_sha256": sha256(args.negative_protocol),
        "target_pool_count": len(target_pool),
        "negative_pool_count": len(negative_pool),
        "sessions": sessions,
    }
    plan_path = args.output_dir / "pose_plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": sha256(checkpoint),
                "pose_plan_sha256": sha256(plan_path),
                "training_frame_count": len(rows),
                "target_pose_count": len(targets),
                "clear_negative_pose_count": len(tp_controls) + len(other_controls),
                "tp_only_pose_count": len(tp_controls),
                "other_bone_pose_count": len(other_controls),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
