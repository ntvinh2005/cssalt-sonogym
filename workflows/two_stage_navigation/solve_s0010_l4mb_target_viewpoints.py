#!/usr/bin/env python3
"""Solve new RIGHT L5 SAP/TP viewpoints from a calibrated scan-plane model.

The solver is evaluator-only. It learns the public pose-to-plane calibration
from synchronized historical records, validates it by held-out session, and
then searches for bounded poses whose predicted first bone contains both
target-region endpoint cores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

import correct_s0010_sap_observability_v1 as sap_correction
import plan_s0010_l4mb_target_observability as atlas_planner
import run_s0010_l4_medial_branch_target as target_workflow


IMAGE_DEPTH = 200
IMAGE_WIDTH = 150
MAXIMUM_P95_CALIBRATION_ERROR_INDEX = 0.75
MAXIMUM_OUT_OF_PLANE_ERROR_INDEX = 0.75


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def features(poses: np.ndarray) -> np.ndarray:
    array = np.asarray(poses, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError("poses must be a finite Nx3 array")
    x = array[:, 0]
    z = array[:, 1]
    angle = array[:, 2]
    sine = np.sin(angle)
    cosine = np.cos(angle)
    return np.column_stack(
        (
            np.ones(len(array)),
            x,
            z,
            sine,
            cosine,
            x * sine,
            x * cosine,
            z * sine,
            z * cosine,
            x * z,
            x * x,
            z * z,
        )
    )


def plane_parameters(coordinates: np.ndarray) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.shape != (IMAGE_DEPTH * IMAGE_WIDTH, 3):
        raise ValueError("native coordinate map has an unexpected shape")
    image = coords.reshape(IMAGE_WIDTH, IMAGE_DEPTH, 3)
    origin = image[0, 0]
    row_step = np.median(np.diff(image, axis=1).reshape(-1, 3), axis=0)
    column_step = np.median(np.diff(image, axis=0).reshape(-1, 3), axis=0)
    return np.concatenate((origin, row_step, column_step))


def fit_model(poses: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    design = features(poses)
    targets = np.asarray(parameters, dtype=np.float64)
    if targets.shape != (len(design), 9):
        raise ValueError("plane parameters must have shape Nx9")
    coefficients, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
    return coefficients


def predict_parameters(poses: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return features(poses) @ np.asarray(coefficients, dtype=np.float64)


def coordinates_from_parameters(parameters: np.ndarray) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (9,):
        raise ValueError("one plane requires nine parameters")
    origin, row_step, column_step = values[:3], values[3:6], values[6:9]
    columns, rows = np.meshgrid(
        np.arange(IMAGE_WIDTH, dtype=np.float64),
        np.arange(IMAGE_DEPTH, dtype=np.float64),
        indexing="ij",
    )
    return (
        origin[None, None, :]
        + rows[:, :, None] * row_step[None, None, :]
        + columns[:, :, None] * column_step[None, None, :]
    ).reshape(-1, 3)


def point_projection(points: np.ndarray, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(parameters, dtype=np.float64)
    origin, row_step, column_step = values[:3], values[3:6], values[6:9]
    basis = np.column_stack((row_step, column_step))
    coefficients = np.linalg.pinv(basis) @ (np.asarray(points, dtype=np.float64) - origin).T
    reconstructed = origin[:, None] + basis @ coefficients
    residual = np.linalg.norm(reconstructed.T - np.asarray(points, dtype=np.float64), axis=1)
    return coefficients.T, residual


def sampled_raw_labels(coordinates: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
    indices = np.trunc(np.asarray(coordinates, dtype=np.float64)).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < np.asarray(segmentation.shape)), axis=1)
    values = np.zeros(len(indices), dtype=np.uint8)
    source = indices[valid]
    values[valid] = segmentation[source[:, 0], source[:, 1], source[:, 2]]
    return values.reshape(IMAGE_WIDTH, IMAGE_DEPTH).T


def calibration_errors(
    poses: np.ndarray, parameters: np.ndarray, sessions: np.ndarray
) -> np.ndarray:
    errors: list[np.ndarray] = []
    sample_rc = np.asarray(
        ((0, 0), (0, 149), (199, 0), (199, 149), (100, 75)), dtype=np.float64
    )
    for held_out in sorted(set(str(value) for value in sessions)):
        test = sessions == held_out
        train = ~test
        coefficients = fit_model(poses[train], parameters[train])
        predictions = predict_parameters(poses[test], coefficients)
        for predicted, actual in zip(predictions, parameters[test]):
            predicted_coordinates = (
                predicted[:3][None, :]
                + sample_rc[:, :1] * predicted[3:6][None, :]
                + sample_rc[:, 1:] * predicted[6:9][None, :]
            )
            actual_coordinates = (
                actual[:3][None, :]
                + sample_rc[:, :1] * actual[3:6][None, :]
                + sample_rc[:, 1:] * actual[6:9][None, :]
            )
            errors.append(np.linalg.norm(predicted_coordinates - actual_coordinates, axis=1))
    return np.concatenate(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, action="append", required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--tp-revision", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prediction_image, sap_mask, tp_mask, contract = target_workflow.load_target_geometry(
        args.prediction, args.segmentation
    )
    segmentation = np.asarray(nib.load(args.segmentation).dataobj, dtype=np.uint8)
    bone_ids = sap_correction.bone_source_ids(args.label_conversion)
    poses: list[list[float]] = []
    parameters: list[np.ndarray] = []
    session_ids: list[str] = []
    source_frames = 0
    for session_dir in atlas_planner.session_directories(args.reference_root):
        annotations = atlas_planner.read_jsonl(session_dir / "privileged/annotations.jsonl")
        maps = np.load(session_dir / "privileged/pixel_to_volume_coordinates.npz")[
            "human_img_coords"
        ]
        if len(annotations) != len(maps):
            raise ValueError(f"annotation/coordinate mismatch: {session_dir}")
        for annotation, coordinate_map in zip(annotations, maps):
            pose = annotation.get("realized_pose_index")
            if pose is None:
                raise ValueError(f"realized pose missing: {session_dir}")
            poses.append([float(value) for value in pose])
            parameters.append(plane_parameters(coordinate_map))
            session_ids.append(session_dir.name)
            source_frames += 1
    pose_array = np.asarray(poses, dtype=np.float64)
    parameter_array = np.asarray(parameters, dtype=np.float64)
    session_array = np.asarray(session_ids)
    errors = calibration_errors(pose_array, parameter_array, session_array)
    error_summary = {
        "median_index": float(np.median(errors)),
        "p90_index": float(np.quantile(errors, 0.90)),
        "p95_index": float(np.quantile(errors, 0.95)),
        "maximum_index": float(np.max(errors)),
        "p95_mm": float(np.quantile(errors, 0.95) * 1.5),
    }
    if error_summary["p95_index"] > MAXIMUM_P95_CALIBRATION_ERROR_INDEX:
        status = "CALIBRATION_MODEL_UNRELIABLE"
        result = {
            "schema": "s0010-l4mb-target-viewpoint-solver-v1",
            "status": status,
            "source_frame_count": source_frames,
            "source_session_count": len(set(session_ids)),
            "calibration_error": error_summary,
            "candidate_counts": {},
            "target_anchors": [],
            "control_anchors": [],
        }
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": status, "p95_index": error_summary["p95_index"]}))
        return 2

    coefficients = fit_model(pose_array, parameter_array)
    lower = np.quantile(pose_array, 0.01, axis=0)
    upper = np.quantile(pose_array, 0.99, axis=0)
    x_values = np.arange(lower[0], upper[0] + 1e-9, 0.75)
    z_values = np.arange(lower[1], upper[1] + 1e-9, 0.75)
    angle_values = np.arange(lower[2], upper[2] + 1e-9, 0.015)
    sap_points = np.argwhere(contract.sap_core)
    tp_points = np.argwhere(contract.tp_core)
    candidates: list[dict[str, Any]] = []
    plane_intersections = 0
    for angle in angle_values:
        for x in x_values:
            batch = np.column_stack(
                (
                    np.full(len(z_values), x),
                    z_values,
                    np.full(len(z_values), angle),
                )
            )
            predicted_batch = predict_parameters(batch, coefficients)
            for pose, predicted in zip(batch, predicted_batch):
                sap_rc, sap_residual = point_projection(sap_points, predicted)
                tp_rc, tp_residual = point_projection(tp_points, predicted)
                sap_inside = (
                    (sap_residual <= MAXIMUM_OUT_OF_PLANE_ERROR_INDEX)
                    & (sap_rc[:, 0] >= 4)
                    & (sap_rc[:, 0] < IMAGE_DEPTH - 4)
                    & (sap_rc[:, 1] >= 4)
                    & (sap_rc[:, 1] < IMAGE_WIDTH - 4)
                )
                tp_inside = (
                    (tp_residual <= MAXIMUM_OUT_OF_PLANE_ERROR_INDEX)
                    & (tp_rc[:, 0] >= 4)
                    & (tp_rc[:, 0] < IMAGE_DEPTH - 4)
                    & (tp_rc[:, 1] >= 4)
                    & (tp_rc[:, 1] < IMAGE_WIDTH - 4)
                )
                if not np.any(sap_inside) or not np.any(tp_inside):
                    continue
                plane_intersections += 1
                coordinate_map = coordinates_from_parameters(predicted)
                projected = target_workflow.projected_masks(
                    coordinate_map, contract, sap_mask, tp_mask
                )
                raw = sampled_raw_labels(coordinate_map, segmentation)
                first_bone = target_workflow.first_bone_mask(raw, bone_ids)
                label, reason = target_workflow.target_evidence_label(projected, first_bone)
                sap_visible = int(np.count_nonzero(projected["sap_core"] & first_bone))
                tp_visible = int(np.count_nonzero(projected["tp_core"] & first_bone))
                target_pixels = np.argwhere(projected["sap_core"] | projected["tp_core"])
                margin = (
                    int(
                        min(
                            target_pixels[:, 0].min(),
                            IMAGE_DEPTH - 1 - target_pixels[:, 0].max(),
                            target_pixels[:, 1].min(),
                            IMAGE_WIDTH - 1 - target_pixels[:, 1].max(),
                        )
                    )
                    if len(target_pixels)
                    else -1
                )
                candidates.append(
                    {
                        "session_id": "geometry_solver",
                        "frame_sequence": len(candidates),
                        "frame_sha256": hashlib.sha256(
                            np.asarray(pose, dtype="<f8").tobytes()
                        ).hexdigest(),
                        "requested_waypoint_pose_index": [float(value) for value in pose],
                        "evaluator_label": label,
                        "evaluator_reason": reason,
                        "sap_core_first_visible_pixels": sap_visible,
                        "tp_core_first_visible_pixels": tp_visible,
                        "combined_core_first_visible_pixels": sap_visible + tp_visible,
                        "target_core_margin_pixels": margin,
                        "surface_gradient_median": 0.0,
                        "sap_plane_residual_index": float(np.min(sap_residual)),
                        "tp_plane_residual_index": float(np.min(tp_residual)),
                    }
                )

    positives = [row for row in candidates if row["evaluator_label"] == "PROXY_POSITIVE"]
    controls = [row for row in candidates if row["evaluator_label"] == "PROXY_NEGATIVE"]
    selected_targets = atlas_planner.select_diverse(
        positives,
        atlas_planner.TARGET_ANCHOR_COUNT,
        ranking_fields=(
            "combined_core_first_visible_pixels",
            "target_core_margin_pixels",
        ),
    )
    selected_controls = atlas_planner.select_diverse(
        controls,
        atlas_planner.CONTROL_ANCHOR_COUNT,
        ranking_fields=("target_core_margin_pixels",),
    )
    status = (
        "GEOMETRY_SOLVED_VIEWPOINTS_READY"
        if len(selected_targets) == atlas_planner.TARGET_ANCHOR_COUNT
        and len(selected_controls) == atlas_planner.CONTROL_ANCHOR_COUNT
        else "NO_SUPPORTED_TARGET_VIEWPOINTS"
    )
    result = {
        "schema": "s0010-l4mb-target-observability-pose-atlas-v1",
        "status": "POSE_ATLAS_READY" if status == "GEOMETRY_SOLVED_VIEWPOINTS_READY" else status,
        "solver_status": status,
        "information_boundary": "evaluator_only_pose_design",
        "source_frame_count": source_frames,
        "source_session_count": len(set(session_ids)),
        "calibration_error": error_summary,
        "search": {
            "pose_bounds_1_99_percent": [lower.tolist(), upper.tolist()],
            "grid_steps": [0.75, 0.75, 0.015],
            "plane_intersection_count": plane_intersections,
            "evaluated_candidate_count": len(candidates),
            "maximum_out_of_plane_error_index": MAXIMUM_OUT_OF_PLANE_ERROR_INDEX,
        },
        "candidate_counts": {
            label: sum(row["evaluator_label"] == label for row in candidates)
            for label in (
                "PROXY_POSITIVE",
                "PROXY_AMBIGUOUS",
                "PROXY_NEGATIVE",
                "TECHNICALLY_INVALID",
            )
        },
        "selection": {
            "target_anchor_count": atlas_planner.TARGET_ANCHOR_COUNT,
            "control_anchor_count": atlas_planner.CONTROL_ANCHOR_COUNT,
            "minimum_translation_separation_index": atlas_planner.MINIMUM_TRANSLATION_SEPARATION_INDEX,
            "minimum_angle_separation_rad": atlas_planner.MINIMUM_ANGLE_SEPARATION_RAD,
        },
        "input_hashes": {
            "prediction": sha256(args.prediction),
            "segmentation": sha256(args.segmentation),
            "tp_revision": sha256(args.tp_revision),
        },
        "target_anchors": selected_targets,
        "control_anchors": selected_controls,
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "p95_calibration_error_index": error_summary["p95_index"],
                **result["candidate_counts"],
            },
            sort_keys=True,
        )
    )
    return 0 if status == "GEOMETRY_SOLVED_VIEWPOINTS_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
