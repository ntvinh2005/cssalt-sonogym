#!/usr/bin/env python3
"""Search physically valid probe commands for RIGHT L4/L5 SAP/TP visibility.

This is an evaluator-only design tool.  It uses the provisional CT subregion
masks to select commands for a later simulator capture.  Its output is not an
actor input and does not establish ultrasound visibility by itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import correct_s0010_sap_observability_v1 as sap_correction
import run_s0010_l4_medial_branch_target as target_workflow
from two_stage_navigation.ultrasound_plane_geometry import (
    plane_from_reduced_pose,
    project_points_to_pixels,
)


IMAGE_WIDTH = 150
IMAGE_DEPTH = 200
IMAGE_RESOLUTION_M = 0.0005
LABEL_RESOLUTION_M = 0.0015
PROBE_HEIGHT_M = 0.13
IMAGE_ORIGIN_HEIGHT_M = 0.133
PLANE_TOLERANCE_MM = 1.5
LEVEL_LABEL = {"L4": 28, "L5": 27}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_torch(path: Path) -> np.ndarray:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def native_flat_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Convert the CPU plane array to the slicer's width-major flat order."""

    array = np.asarray(coordinates, dtype=np.float64)
    if array.shape != (IMAGE_DEPTH, IMAGE_WIDTH, 3):
        raise ValueError(f"unexpected plane shape: {array.shape}")
    return array.transpose(1, 0, 2).reshape(-1, 3)


def target_margin(projected: dict[str, np.ndarray]) -> int:
    pixels = np.argwhere(projected["sap_core"] | projected["tp_core"])
    if not len(pixels):
        return -1
    return int(
        min(
            pixels[:, 0].min(),
            IMAGE_DEPTH - 1 - pixels[:, 0].max(),
            pixels[:, 1].min(),
            IMAGE_WIDTH - 1 - pixels[:, 1].max(),
        )
    )


def exact_pose_evidence(
    pose: np.ndarray,
    *,
    surface_y: np.ndarray,
    surface_normals: np.ndarray,
    segmentation: np.ndarray,
    bone_ids: set[int],
    sap_mask: np.ndarray,
    tp_mask: np.ndarray,
    contract: Any,
) -> dict[str, Any]:
    plane = plane_from_reduced_pose(
        realized_pose_index=pose,
        surface_y_index=surface_y,
        surface_normals=surface_normals,
        width=IMAGE_WIDTH,
        depth=IMAGE_DEPTH,
        image_resolution_m=IMAGE_RESOLUTION_M,
        label_resolution_m=LABEL_RESOLUTION_M,
        probe_height_m=PROBE_HEIGHT_M,
        image_origin_height_m=IMAGE_ORIGIN_HEIGHT_M,
        volume_shape=segmentation.shape,
    )
    coordinates = native_flat_coordinates(plane.coordinates_index)
    projected = target_workflow.projected_masks(coordinates, contract, sap_mask, tp_mask)
    indices = np.trunc(coordinates).astype(np.int64)
    raw = segmentation[indices[:, 0], indices[:, 1], indices[:, 2]].reshape(
        IMAGE_WIDTH, IMAGE_DEPTH
    ).T
    first_bone = target_workflow.first_bone_mask(raw, bone_ids)
    label, reason = target_workflow.target_evidence_label(projected, first_bone)
    return {
        "pose_index_rad": [float(value) for value in pose],
        "evaluator_label": label,
        "evaluator_reason": reason,
        "sap_core_projected_pixels": int(np.count_nonzero(projected["sap_core"])),
        "tp_core_projected_pixels": int(np.count_nonzero(projected["tp_core"])),
        "sap_core_first_visible_pixels": int(
            np.count_nonzero(projected["sap_core"] & first_bone)
        ),
        "tp_core_first_visible_pixels": int(
            np.count_nonzero(projected["tp_core"] & first_bone)
        ),
        "sap_first_visible_pixels": int(np.count_nonzero(projected["sap"] & first_bone)),
        "tp_first_visible_pixels": int(np.count_nonzero(projected["tp"] & first_bone)),
        "sap_projected_pixels": int(np.count_nonzero(projected["sap"])),
        "tp_projected_pixels": int(np.count_nonzero(projected["tp"])),
        "target_margin_pixels": target_margin(projected),
        "clipped_sample_count": int(plane.clipped_sample_count),
    }


def coarse_candidates(
    *,
    surface_y: np.ndarray,
    surface_normals: np.ndarray,
    volume_shape: tuple[int, int, int],
    contract: Any,
    x_values: Iterable[float],
    z_values: Iterable[float],
    angle_values: Iterable[float],
) -> list[dict[str, Any]]:
    sap_points = np.argwhere(contract.sap_core)
    tp_points = np.argwhere(contract.tp_core)
    result: list[dict[str, Any]] = []
    for x_value in x_values:
        for z_value in z_values:
            for angle_value in angle_values:
                pose = np.asarray([x_value, z_value, angle_value], dtype=np.float64)
                plane = plane_from_reduced_pose(
                    realized_pose_index=pose,
                    surface_y_index=surface_y,
                    surface_normals=surface_normals,
                    # Only the origin and rotation are needed in this pass.
                    # A one-pixel plane avoids constructing 30,000 unused
                    # coordinates for every command in the broad search.
                    width=1,
                    depth=1,
                    image_resolution_m=IMAGE_RESOLUTION_M,
                    label_resolution_m=LABEL_RESOLUTION_M,
                    probe_height_m=PROBE_HEIGHT_M,
                    image_origin_height_m=IMAGE_ORIGIN_HEIGHT_M,
                    volume_shape=None,
                )
                sap = project_points_to_pixels(
                    points_index=sap_points,
                    image_origin_index=plane.origin_index,
                    rotation_columns=plane.rotation_columns,
                    width=IMAGE_WIDTH,
                    depth=IMAGE_DEPTH,
                    image_resolution_m=IMAGE_RESOLUTION_M,
                    label_resolution_m=LABEL_RESOLUTION_M,
                    plane_tolerance_mm=PLANE_TOLERANCE_MM,
                )
                tp = project_points_to_pixels(
                    points_index=tp_points,
                    image_origin_index=plane.origin_index,
                    rotation_columns=plane.rotation_columns,
                    width=IMAGE_WIDTH,
                    depth=IMAGE_DEPTH,
                    image_resolution_m=IMAGE_RESOLUTION_M,
                    label_resolution_m=LABEL_RESOLUTION_M,
                    plane_tolerance_mm=PLANE_TOLERANCE_MM,
                )
                sap_inside = int(np.count_nonzero(sap["inside"]))
                tp_inside = int(np.count_nonzero(tp["inside"]))
                if not sap_inside or not tp_inside:
                    continue
                rows = np.concatenate((sap["rows"][sap["inside"]], tp["rows"][tp["inside"]]))
                columns = np.concatenate(
                    (sap["columns"][sap["inside"]], tp["columns"][tp["inside"]])
                )
                margin = int(
                    min(
                        rows.min(),
                        IMAGE_DEPTH - 1 - rows.max(),
                        columns.min(),
                        IMAGE_WIDTH - 1 - columns.max(),
                    )
                )
                result.append(
                    {
                        "pose_index_rad": [float(value) for value in pose],
                        "coarse_sap_core_points": sap_inside,
                        "coarse_tp_core_points": tp_inside,
                        "coarse_margin_pixels": margin,
                    }
                )
    return result


def physically_distinct(first: dict[str, Any], second: dict[str, Any]) -> bool:
    a = np.asarray(first["pose_index_rad"], dtype=np.float64)
    b = np.asarray(second["pose_index_rad"], dtype=np.float64)
    translation_mm = float(np.linalg.norm(a[:2] - b[:2]) * 1.5)
    angle_deg = float(np.rad2deg(abs(np.arctan2(np.sin(a[2] - b[2]), np.cos(a[2] - b[2])))))
    return bool(translation_mm >= 3.0 or angle_deg >= 2.5)


def select_diverse(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    priority = {"PROXY_POSITIVE": 0, "PROXY_AMBIGUOUS": 1, "PROXY_NEGATIVE": 2}
    ordered = sorted(
        rows,
        key=lambda row: (
            priority[row["evaluator_label"]],
            -min(
                int(row["sap_core_first_visible_pixels"]),
                int(row["tp_core_first_visible_pixels"]),
            ),
            -min(
                int(row["sap_first_visible_pixels"]),
                int(row["tp_first_visible_pixels"]),
            ),
            -int(row["target_margin_pixels"]),
            tuple(row["pose_index_rad"]),
        ),
    )
    chosen: list[dict[str, Any]] = []
    for row in ordered:
        if all(physically_distinct(row, previous) for previous in chosen):
            chosen.append(row)
        if len(chosen) == count:
            break
    return chosen


def calibration_summary(capture_root: Path, surface_y: np.ndarray, surface_normals: np.ndarray) -> dict[str, Any]:
    errors: list[np.ndarray] = []
    frame_medians: list[float] = []
    for session in sorted((capture_root / "sessions").iterdir()):
        annotation_path = session / "privileged/annotations.jsonl"
        coordinate_path = session / "privileged/pixel_to_volume_coordinates.npz"
        if not annotation_path.is_file() or not coordinate_path.is_file():
            continue
        annotations = [
            json.loads(line) for line in annotation_path.read_text(encoding="utf-8").splitlines()
        ]
        native = np.load(coordinate_path)["human_img_coords"]
        for index, annotation in enumerate(annotations):
            plane = plane_from_reduced_pose(
                realized_pose_index=np.asarray(annotation["realized_pose_index"]),
                surface_y_index=surface_y,
                surface_normals=surface_normals,
                width=IMAGE_WIDTH,
                depth=IMAGE_DEPTH,
                image_resolution_m=IMAGE_RESOLUTION_M,
                label_resolution_m=LABEL_RESOLUTION_M,
                probe_height_m=PROBE_HEIGHT_M,
                image_origin_height_m=IMAGE_ORIGIN_HEIGHT_M,
            )
            difference = np.linalg.norm(
                native_flat_coordinates(plane.coordinates_index) - native[index], axis=1
            ) * (LABEL_RESOLUTION_M * 1000.0)
            errors.append(difference)
            frame_medians.append(float(np.median(difference)))
    if not errors:
        raise ValueError("capture root contains no calibration frames")
    combined = np.concatenate(errors)
    return {
        "frame_count": len(frame_medians),
        "point_count": int(len(combined)),
        "median_error_mm": float(np.median(combined)),
        "p90_error_mm": float(np.quantile(combined, 0.90)),
        "p95_error_mm": float(np.quantile(combined, 0.95)),
        "maximum_error_mm": float(np.max(combined)),
        "frames_with_median_error_at_most_1_5_mm": int(
            np.count_nonzero(np.asarray(frame_medians) <= 1.5)
        ),
        "status": "SCREENING_ONLY_REQUIRES_FULL_POSE_CAPTURE_CONFIRMATION",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--patient-surface-dir", type=Path, required=True)
    parser.add_argument("--calibration-capture-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--x-step", type=float, default=2.0)
    parser.add_argument("--z-step", type=float, default=2.0)
    parser.add_argument("--angle-step", type=float, default=0.025)
    parser.add_argument("--level", choices=sorted(LEVEL_LABEL), default="L5")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    _, sap_mask, tp_mask, contract = target_workflow.load_target_geometry(
        args.prediction,
        args.segmentation,
        level_label=LEVEL_LABEL[args.level],
        level_name=args.level,
    )
    segmentation = np.asanyarray(target_workflow.nib.load(str(args.segmentation)).dataobj).astype(
        np.uint8
    )
    surface_y = safe_torch(args.patient_surface_dir / "body_lowest_y_array.pt")
    surface_normals = safe_torch(args.patient_surface_dir / "body_surface_normal_array.pt")
    bone_ids = sap_correction.bone_source_ids(args.label_conversion)
    calibration = calibration_summary(
        args.calibration_capture_root, surface_y, surface_normals
    )

    coarse = coarse_candidates(
        surface_y=surface_y,
        surface_normals=surface_normals,
        volume_shape=segmentation.shape,
        contract=contract,
        x_values=np.arange(100.0, 180.0 + 1e-9, args.x_step),
        z_values=np.arange(70.0, 200.0 + 1e-9, args.z_step),
        angle_values=np.arange(1.0, 3.0 + 1e-9, args.angle_step),
    )
    coarse.sort(
        key=lambda row: (
            -min(row["coarse_sap_core_points"], row["coarse_tp_core_points"]),
            -row["coarse_margin_pixels"],
        )
    )
    exact = [
        {
            **row,
            **exact_pose_evidence(
                np.asarray(row["pose_index_rad"]),
                surface_y=surface_y,
                surface_normals=surface_normals,
                segmentation=segmentation,
                bone_ids=bone_ids,
                sap_mask=sap_mask,
                tp_mask=tp_mask,
                contract=contract,
            ),
        }
        for row in coarse
    ]
    positives = [row for row in exact if row["evaluator_label"] == "PROXY_POSITIVE"]
    surface_pair = [
        row
        for row in exact
        if int(row["sap_first_visible_pixels"]) > 0
        and int(row["tp_first_visible_pixels"]) > 0
        and int(row["target_margin_pixels"]) >= 4
    ]
    selected = select_diverse(positives if positives else surface_pair, 12)
    if positives:
        status = "TARGET_CORE_CAPTURE_CANDIDATES_FOUND"
    elif surface_pair:
        status = "SAP_TP_SURFACE_CAPTURE_CANDIDATES_FOUND"
    else:
        status = "NO_FIRST_VISIBLE_SAP_TP_CANDIDATE"
    result = {
        "schema": "s0010-sap-tp-broad-probe-search-v2",
        "vertebral_level": args.level,
        "status": status,
        "information_boundary": "evaluator_only_waypoint_design",
        "search_domain": {
            "x_index": [100.0, 180.0, args.x_step],
            "z_index": [70.0, 200.0, args.z_step],
            "angle_rad": [1.0, 3.0, args.angle_step],
            "command_domain_source": "robotic_US_guidance.yaml",
        },
        "input_hashes": {
            "prediction": sha256(args.prediction),
            "segmentation": sha256(args.segmentation),
            "label_conversion": sha256(args.label_conversion),
        },
        "reduced_pose_calibration": calibration,
        "candidate_counts": {
            "coarse_core_covisible": len(coarse),
            "PROXY_POSITIVE": len(positives),
            "PROXY_AMBIGUOUS": sum(
                row["evaluator_label"] == "PROXY_AMBIGUOUS" for row in exact
            ),
            "PROXY_NEGATIVE": sum(
                row["evaluator_label"] == "PROXY_NEGATIVE" for row in exact
            ),
            "SAP_TP_SURFACE_FIRST_VISIBLE": len(surface_pair),
        },
        "selected_capture_candidates": selected,
        "limitations": [
            "The reduced pose is only a screening approximation.",
            "A simulator capture with synchronized full pose is required.",
            "Geometry visibility does not establish a clean-image cortical signal.",
        ],
    }
    (args.output_dir / "search_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "coarse_candidates.csv", coarse)
    write_csv(args.output_dir / "exact_candidates.csv", exact)
    write_csv(args.output_dir / "selected_capture_candidates.csv", selected)
    print(json.dumps({"status": status, **result["candidate_counts"]}, sort_keys=True))
    return 0 if status.endswith("CAPTURE_CANDIDATES_FOUND") else 2


if __name__ == "__main__":
    raise SystemExit(main())
