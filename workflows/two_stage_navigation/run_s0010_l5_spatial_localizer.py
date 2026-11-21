#!/usr/bin/env python3
"""Validate spatial geometry and rank RIGHT L5 TP views from recorded search data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import yaml

from two_stage_navigation.tp_surface_core_dataset import PatchProposal
from two_stage_navigation.ultrasound_plane_geometry import (
    plane_from_full_pose,
    project_points_to_pixels,
)
from two_stage_navigation.ultrasound_spatial_map import (
    CorticalMapPoint,
    SpatialLocalizerConfig,
    SpatialMapConfig,
    SpatialViewCandidate,
    associate_spatial_points,
    calibrated_image_coordinates_mm,
    cortical_map_points,
    rank_spatial_target_views,
)

from evaluate_stage2_tp_active import CANDIDATES, locate_geometry, proxy_state


TARGET = "L5_RIGHT_TP_SURFACE_CORE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def map_config(protocol: dict[str, Any]) -> SpatialMapConfig:
    row = protocol["actor_geometry"]
    return SpatialMapConfig(
        pixel_resolution_mm=float(row["pixel_resolution_mm"]),
        label_resolution_mm=float(row["label_resolution_mm"]),
        sample_stride_pixels=int(row["sample_stride_pixels"]),
        association_radius_mm=float(row["association_radius_mm"]),
        minimum_view_translation_mm=float(row["minimum_view_translation_mm"]),
        minimum_view_angle_deg=float(row["minimum_view_angle_deg"]),
        include_depth=True,
    )


def localizer_config(protocol: dict[str, Any], hypothesis: dict[str, Any]) -> SpatialLocalizerConfig:
    row = protocol["localizer_shared"]
    return SpatialLocalizerConfig(
        hypothesis=str(hypothesis["name"]),
        support_radius_mm=float(row["support_radius_mm"]),
        minimum_supporting_frames=int(row["minimum_supporting_frames"]),
        maximum_support_distance_mm=float(row["maximum_support_distance_mm"]),
        minimum_target_probability=float(row["minimum_target_probability"]),
        target_probability_weight=float(row["target_probability_weight"]),
        support_weight=float(row["support_weight"]),
        support_saturation_count=int(row["support_saturation_count"]),
        landmark_prior_scale_mm=float(hypothesis["landmark_prior_scale_mm"]),
        minimum_score=float(row["minimum_score"]),
        minimum_margin=float(row["minimum_margin"]),
        top_k=int(row["top_k"]),
    )


def search_frame(frame: dict[str, Any], protocol: dict[str, Any]) -> bool:
    if frame["target_name"] != TARGET:
        return False
    phase = str(frame["phase"]).split(":")[-1]
    return phase in set(protocol["observation_budget"]["include_phases"])


def load_geometries(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for name, candidate in CANDIDATES.items():
        path = locate_geometry(root, candidate)
        archive = np.load(path)
        result[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "vertices": archive["vertices_mesh_index"].astype(np.float64),
            "core": archive["proposed_revised_vertex_mask"].astype(bool),
            "uncertain": archive["boundary_uncertainty_vertex_mask"].astype(bool),
        }
    return result


def geometry_for_frame(
    frame: dict[str, Any],
    pose: dict[str, Any],
    coordinates: np.ndarray,
    local: np.ndarray,
    geometries: dict[str, dict[str, Any]],
    slab_tolerance_mm: float,
    bone_tree: cKDTree,
    target_tree: cKDTree,
) -> dict[str, Any]:
    if frame["proposal_row"] is None or frame["proposal_column"] is None:
        return {
            "proxy_state": "NO_PROPOSAL",
            "target_distance_mm": None,
            "bone_surface_distance_mm": None,
        }
    plane = plane_from_full_pose(
        end_effector_position_m=np.asarray(pose["human_to_ee_position_m"]),
        rotation_columns=np.asarray(pose["human_to_ee_rotation_columns"]),
        image_origin_height_m=float(pose["image_origin_height_m"]),
        local_coordinates_m=local,
        label_resolution_m=float(pose["label_resolution_m"]),
    )
    projections: dict[str, dict[str, np.ndarray]] = {}
    for name, geometry in geometries.items():
        core = project_points_to_pixels(
            points_index=geometry["vertices"][geometry["core"]],
            image_origin_index=plane.origin_index,
            rotation_columns=plane.rotation_columns,
            width=150,
            depth=200,
            image_resolution_m=float(pose["image_resolution_m"]),
            label_resolution_m=float(pose["label_resolution_m"]),
            plane_tolerance_mm=slab_tolerance_mm,
        )
        uncertain = project_points_to_pixels(
            points_index=geometry["vertices"][geometry["uncertain"]],
            image_origin_index=plane.origin_index,
            rotation_columns=plane.rotation_columns,
            width=150,
            depth=200,
            image_resolution_m=float(pose["image_resolution_m"]),
            label_resolution_m=float(pose["label_resolution_m"]),
            plane_tolerance_mm=slab_tolerance_mm,
        )
        projections[name] = {
            "core_rows": core["rows"][core["inside"]],
            "core_columns": core["columns"][core["inside"]],
            "uncertain_rows": uncertain["rows"][uncertain["inside"]],
            "uncertain_columns": uncertain["columns"][uncertain["inside"]],
        }
    proposal = PatchProposal(
        str(frame["proposal_track_id"]),
        float(frame["proposal_row"]),
        float(frame["proposal_column"]),
        np.empty(0),
    )
    state, _ = proxy_state(TARGET, proposal, projections)
    row = int(np.clip(round(float(frame["proposal_row"])), 0, 199))
    column = int(np.clip(round(float(frame["proposal_column"])), 0, 149))
    point_mm = coordinates[column * 200 + row].astype(np.float64) * 1.5
    return {
        "proxy_state": state,
        "target_distance_mm": float(target_tree.query(point_mm)[0]),
        "bone_surface_distance_mm": float(bone_tree.query(point_mm)[0]),
    }


def calibration_metrics(
    frames: list[dict[str, Any]],
    poses: list[dict[str, Any]],
    native: np.ndarray,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    geometry = protocol["actor_geometry"]
    errors: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    depth_rows: list[np.ndarray] = []
    angles: list[np.ndarray] = []
    for frame, pose, coordinates in zip(frames, poses, native):
        calibrated = calibrated_image_coordinates_mm(
            end_effector_position_m=pose["human_to_ee_position_m"],
            rotation_columns=np.asarray(pose["human_to_ee_rotation_columns"]),
            width=int(geometry["width_pixels"]),
            depth=int(geometry["depth_pixels"]),
            pixel_resolution_mm=float(geometry["pixel_resolution_mm"]),
            image_origin_height_mm=float(geometry["image_origin_height_mm"]),
        )
        expected = (
            coordinates.reshape(150, 200, 3).transpose(1, 0, 2).astype(np.float64)
            * float(geometry["label_resolution_mm"])
        )
        delta = calibrated - expected
        error = np.linalg.norm(delta, axis=2)
        errors.append(error.ravel())
        deltas.append(delta.reshape(-1, 3))
        depth_rows.append(np.repeat(np.arange(200), 150))
        angles.append(np.full(200 * 150, float(frame["executed_relative_pose_index"][2])))
    error = np.concatenate(errors)
    delta = np.concatenate(deltas)
    rows = np.concatenate(depth_rows)
    angle = np.concatenate(angles)
    depth_summary = []
    bins = protocol["calibration_evaluator"]["depth_bins_rows"]
    for lower, upper in zip(bins[:-1], bins[1:]):
        values = error[(rows >= lower) & (rows < upper)]
        depth_summary.append(
            {
                "row_start": int(lower),
                "row_stop": int(upper),
                "median_error_mm": float(np.median(values)),
                "p90_error_mm": float(np.quantile(values, 0.90)),
                "maximum_error_mm": float(np.max(values)),
            }
        )
    orientation_summary = []
    bins = protocol["calibration_evaluator"]["orientation_bins_rad"]
    for lower, upper in zip(bins[:-1], bins[1:]):
        values = error[(angle >= lower) & (angle < upper)]
        if values.size:
            orientation_summary.append(
                {
                    "angle_start_rad": float(lower),
                    "angle_stop_rad": float(upper),
                    "pixel_count": int(values.size),
                    "median_error_mm": float(np.median(values)),
                    "p90_error_mm": float(np.quantile(values, 0.90)),
                    "maximum_error_mm": float(np.max(values)),
                }
            )
    result = {
        "pixel_count": int(error.size),
        "median_error_mm": float(np.median(error)),
        "p90_error_mm": float(np.quantile(error, 0.90)),
        "maximum_error_mm": float(np.max(error)),
        "mean_bias_xyz_mm": np.mean(delta, axis=0).tolist(),
        "depth_summary": depth_summary,
        "orientation_summary": orientation_summary,
    }
    gate = protocol["calibration_evaluator"]
    result["passes_coordinate_gate"] = bool(
        result["median_error_mm"] <= gate["maximum_median_coordinate_error_mm"]
        and result["p90_error_mm"] <= gate["maximum_p90_coordinate_error_mm"]
        and result["maximum_error_mm"] <= gate["maximum_coordinate_error_mm"]
    )
    return result


def map_geometry_metrics(
    points: list[CorticalMapPoint],
    poses_by_frame: dict[int, np.ndarray],
    native: np.ndarray,
    bone_tree: cKDTree,
    target_tree: cKDTree,
    config: SpatialMapConfig,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    native_mm = np.asarray(
        [native[p.frame_sequence, p.column * 200 + p.row] * config.label_resolution_mm for p in points],
        dtype=np.float64,
    )
    actor = np.asarray([p.coordinate_mm for p in points], dtype=np.float64)
    horizontal_error = np.linalg.norm(actor[:, :2] - native_mm[:, [0, 2]], axis=1)
    bone_distance = bone_tree.query(native_mm)[0]
    target_distance = target_tree.query(native_mm)[0]
    edges = associate_spatial_points(points, poses_by_frame=poses_by_frame, config=config)
    repeated_distance = np.asarray(
        [np.linalg.norm(native_mm[e.left_index] - native_mm[e.right_index]) for e in edges]
    )
    tolerance = float(protocol["surface_evaluator"]["target_surface_tolerance_mm"])
    target_status = target_distance <= tolerance
    crossing = np.asarray(
        [target_status[e.left_index] != target_status[e.right_index] for e in edges],
        dtype=bool,
    )
    bone_tolerance = float(protocol["surface_evaluator"]["bone_surface_tolerance_mm"])
    return {
        "point_count": len(points),
        "association_count": len(edges),
        "reduced_2p5d_horizontal_error_mm": {
            "median": float(np.median(horizontal_error)),
            "p90": float(np.quantile(horizontal_error, 0.90)),
            "maximum": float(np.max(horizontal_error)),
            "mean_bias_xz_mm": np.mean(actor[:, :2] - native_mm[:, [0, 2]], axis=0).tolist(),
        },
        "nearest_bone_surface_distance_mm": {
            "median": float(np.median(bone_distance)),
            "p90": float(np.quantile(bone_distance, 0.90)),
            "maximum": float(np.max(bone_distance)),
            "inlier_fraction": float(np.mean(bone_distance <= bone_tolerance)),
        },
        "nearest_l5_tp_core_distance_mm": {
            "median": float(np.median(target_distance)),
            "p90": float(np.quantile(target_distance, 0.90)),
            "inlier_fraction": float(np.mean(target_status)),
        },
        "repeated_view_native_separation_mm": {
            "median": None if not repeated_distance.size else float(np.median(repeated_distance)),
            "p90": None if not repeated_distance.size else float(np.quantile(repeated_distance, 0.90)),
            "maximum": None if not repeated_distance.size else float(np.max(repeated_distance)),
        },
        "target_nontarget_association_fraction": (
            None if not crossing.size else float(np.mean(crossing))
        ),
    }


def session_data(
    session_id: str,
    *,
    run_root: Path,
    protocol: dict[str, Any],
    source_protocol: dict[str, Any],
    geometries: dict[str, dict[str, Any]],
    bone_tree: cKDTree,
    target_tree: cKDTree,
) -> dict[str, Any]:
    root = run_root / "stage2" / session_id
    frames = read_jsonl(root / "public/frames.jsonl")
    poses = read_jsonl(root / "privileged/full_pose_replay.jsonl")
    images = np.load(root / "public/ultrasound.npz")["ultrasound"]
    native_archive = np.load(root / "privileged/pixel_to_volume_coordinates.npz")
    native = native_archive["human_img_coords"]
    local = native_archive["native_local_image_coordinates_m"].astype(np.float64)
    actor_result = json.loads((root / "results/actor_result.json").read_text())
    if not len(frames) == len(poses) == len(images) == len(native):
        raise RuntimeError(f"incomplete frame join: {session_id}")
    indices = [index for index, frame in enumerate(frames) if search_frame(frame, protocol)]
    config = map_config(protocol)
    points: list[CorticalMapPoint] = []
    poses_by_frame = {}
    candidates = []
    evaluation = {}
    for index in indices:
        frame = frames[index]
        sequence = int(frame["frame_sequence"])
        poses_by_frame[sequence] = np.asarray(frame["executed_relative_pose_index"], dtype=np.float64)
        points.extend(
            cortical_map_points(
                images[index],
                executed_pose_index=frame["executed_relative_pose_index"],
                frame_sequence=sequence,
                frame_sha256=str(frame["frame_sha256"]),
                config=config,
            )
        )
        evaluation[sequence] = geometry_for_frame(
            frame,
            poses[index],
            native[index],
            local,
            geometries,
            float(source_protocol["evaluator"]["surface_slab_tolerance_mm"]),
            bone_tree,
            target_tree,
        )
        if frame["proposal_row"] is not None and frame["proposal_column"] is not None:
            candidates.append(
                SpatialViewCandidate(
                    frame_sequence=sequence,
                    frame_sha256=str(frame["frame_sha256"]),
                    target_probability=float(frame["target_probability"]),
                    proposal_row=float(frame["proposal_row"]),
                    proposal_column=float(frame["proposal_column"]),
                )
            )
    ordered_sequences = [int(frames[index]["frame_sequence"]) for index in indices]
    rng = np.random.default_rng(int(protocol["control_settings"]["shuffled_motion_seed"]))
    shuffled_order = np.asarray(ordered_sequences, dtype=np.int64)
    rng.shuffle(shuffled_order)
    if len(shuffled_order) > 1 and any(
        int(left) == int(right) for left, right in zip(ordered_sequences, shuffled_order)
    ):
        shuffled_order = np.roll(np.asarray(ordered_sequences, dtype=np.int64), 1)
    source_pose_by_sequence = {
        int(frames[index]["frame_sequence"]): np.asarray(
            frames[index]["executed_relative_pose_index"], dtype=np.float64
        )
        for index in indices
    }
    shuffled_poses_by_frame = {
        sequence: source_pose_by_sequence[int(shuffled)]
        for sequence, shuffled in zip(ordered_sequences, shuffled_order)
    }
    shuffled_points: list[CorticalMapPoint] = []
    for index in indices:
        frame = frames[index]
        sequence = int(frame["frame_sequence"])
        shuffled_points.extend(
            cortical_map_points(
                images[index],
                executed_pose_index=shuffled_poses_by_frame[sequence],
                frame_sequence=sequence,
                frame_sha256=str(frame["frame_sha256"]),
                config=config,
            )
        )
    target_actor = next(row for row in actor_result["targets"] if row["target_name"] == TARGET)
    return {
        "root": root,
        "frames": frames,
        "images": images,
        "native": native,
        "poses": poses,
        "indices": indices,
        "points": points,
        "poses_by_frame": poses_by_frame,
        "shuffled_points": shuffled_points,
        "shuffled_poses_by_frame": shuffled_poses_by_frame,
        "candidates": candidates,
        "evaluation": evaluation,
        "prior": np.asarray(target_actor["stage1_handoff_pose_index"], dtype=np.float64),
        "historical_selected_sequence": int(target_actor["selected_frame_sequence"]),
        "calibration": calibration_metrics(frames, poses, native, protocol),
        "map_geometry": map_geometry_metrics(
            points, poses_by_frame, native, bone_tree, target_tree, config, protocol
        ),
    }


def evaluate_hypothesis(
    session_id: str,
    data: dict[str, Any],
    hypothesis: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decision = rank_spatial_target_views(
        data["points"],
        data["candidates"],
        poses_by_frame=data["poses_by_frame"],
        stage1_prior_pose_index=data["prior"],
        map_config=map_config(protocol),
        localizer_config=localizer_config(protocol, hypothesis),
    )
    shuffled_decision = rank_spatial_target_views(
        data["shuffled_points"],
        data["candidates"],
        poses_by_frame=data["shuffled_poses_by_frame"],
        stage1_prior_pose_index=data["prior"],
        map_config=map_config(protocol),
        localizer_config=localizer_config(protocol, hypothesis),
    )
    frame_rows = []
    for rank, view in enumerate(decision.ranking, start=1):
        evaluator = data["evaluation"][view.frame_sequence]
        frame_rows.append(
            {
                "session_id": session_id,
                "hypothesis": hypothesis["name"],
                "rank": rank,
                "frame_sequence": view.frame_sequence,
                "frame_sha256": view.frame_sha256,
                "score": view.score,
                "target_probability": view.target_probability,
                "supporting_frame_count": len(view.supporting_frames),
                "supporting_frames": ";".join(map(str, view.supporting_frames)),
                "median_support_distance_mm": view.median_support_distance_mm,
                "stage1_prior_distance_mm": view.landmark_prior_distance_mm,
                "eligible": view.eligible,
                "selected": bool(decision.selected and decision.selected.frame_sequence == view.frame_sequence),
                **evaluator,
            }
        )
    selected_sequence = None if decision.selected is None else decision.selected.frame_sequence
    selected_eval = None if selected_sequence is None else data["evaluation"][selected_sequence]
    top3 = frame_rows[:3]
    ordered_sequences = [int(data["frames"][index]["frame_sequence"]) for index in data["indices"]]
    observations_required = None
    if decision.selected is not None:
        required = set(decision.selected.supporting_frames) | {decision.selected.frame_sequence}
        observations_required = max(ordered_sequences.index(value) for value in required) + 1
    single = max(data["candidates"], key=lambda row: (row.target_probability, -row.frame_sequence))
    historical = data["historical_selected_sequence"]
    shuffled_sequence = (
        None if shuffled_decision.selected is None else shuffled_decision.selected.frame_sequence
    )
    shuffled_eval = (
        None if shuffled_sequence is None else data["evaluation"][shuffled_sequence]
    )
    record = {
        "session_id": session_id,
        "hypothesis": hypothesis["name"],
        "candidate_count": len(data["candidates"]),
        "positive_candidates": sum(
            data["evaluation"][row.frame_sequence]["proxy_state"] == "PROXY_POSITIVE"
            for row in data["candidates"]
        ),
        "ambiguous_candidates": sum(
            data["evaluation"][row.frame_sequence]["proxy_state"] == "PROXY_AMBIGUOUS"
            for row in data["candidates"]
        ),
        "decision": decision.status,
        "reason": decision.reason,
        "selected_frame_sequence": selected_sequence,
        "selected_state": "ABSTAIN" if selected_eval is None else selected_eval["proxy_state"],
        "selected_localization_error_mm": None if selected_eval is None else selected_eval["target_distance_mm"],
        "selected_supporting_frames": 0 if decision.selected is None else len(decision.selected.supporting_frames),
        "selected_uncertainty_mm": None if decision.selected is None else decision.selected.median_support_distance_mm,
        "observations_required": observations_required,
        "top3_contains_positive": any(row["proxy_state"] == "PROXY_POSITIVE" for row in top3),
        "top3_states": [row["proxy_state"] for row in top3],
        "ambiguous_top3_count": sum(row["proxy_state"] == "PROXY_AMBIGUOUS" for row in top3),
        "single_view_sequence": single.frame_sequence,
        "single_view_state": data["evaluation"][single.frame_sequence]["proxy_state"],
        "single_view_localization_error_mm": data["evaluation"][single.frame_sequence]["target_distance_mm"],
        "historical_sequence": historical,
        "historical_state": data["evaluation"][historical]["proxy_state"],
        "historical_localization_error_mm": data["evaluation"][historical]["target_distance_mm"],
        "shuffled_motion_decision": shuffled_decision.status,
        "shuffled_motion_sequence": shuffled_sequence,
        "shuffled_motion_state": (
            "ABSTAIN" if shuffled_eval is None else shuffled_eval["proxy_state"]
        ),
        "shuffled_motion_localization_error_mm": (
            None if shuffled_eval is None else shuffled_eval["target_distance_mm"]
        ),
    }
    return record, frame_rows


def summarize(rows: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    correct = [row for row in rows if row["selected_state"] == "PROXY_POSITIVE"]
    wrong = [
        row for row in rows
        if row["decision"] == "ACCEPT" and row["selected_state"] != "PROXY_POSITIVE"
    ]
    errors = [row["selected_localization_error_mm"] for row in correct]
    support = [row["selected_supporting_frames"] for row in correct]
    result = {
        "session_count": len(rows),
        "correct_accepted_sessions": len(correct),
        "wrong_accepted_sessions": len(wrong),
        "abstained_sessions": sum(row["decision"] == "ABSTAIN" for row in rows),
        "top3_positive_sessions": sum(row["top3_contains_positive"] for row in rows),
        "mean_localization_error_mm": None if not errors else float(np.mean(errors)),
        "minimum_selected_supporting_frames": 0 if not support else int(min(support)),
        "mean_observations_required": float(
            np.mean([row["observations_required"] for row in correct])
        ) if correct else None,
    }
    result["gate_pass"] = bool(
        result["correct_accepted_sessions"] >= gate["minimum_correct_accepted_sessions"]
        and result["wrong_accepted_sessions"] <= gate["maximum_wrong_accepted_sessions"]
        and result["top3_positive_sessions"] >= gate["minimum_top3_positive_sessions"]
        and result["mean_localization_error_mm"] is not None
        and result["mean_localization_error_mm"] <= gate["maximum_mean_localization_error_mm"]
        and result["minimum_selected_supporting_frames"] >= gate["minimum_selected_supporting_frames"]
    )
    return result


def main() -> int:
    started = time.monotonic()
    args = parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    expected = args.protocol.with_suffix(".sha256").read_text().split()[0]
    if sha256(args.protocol) != expected:
        raise RuntimeError("protocol seal mismatch")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_protocol = yaml.safe_load(
        (args.run_root / "context/s0010_mbb_single_anatomy_poc_protocol_v3.yaml").read_text()
    )
    geometries = load_geometries(Path(source_protocol["evaluator"]["geometry_root"]))
    volume_path = Path(protocol["surface_evaluator"]["combined_label_map"])
    volume = np.asanyarray(nib.load(volume_path).dataobj).astype(np.int16)
    labels = np.asarray(protocol["surface_evaluator"]["vertebral_bone_labels"])
    bone = np.isin(volume, labels)
    surface = bone & ~ndimage.binary_erosion(bone)
    bone_tree = cKDTree(np.argwhere(surface).astype(np.float64) * 1.5)
    target_core = geometries[TARGET]["vertices"][geometries[TARGET]["core"]] * 1.5
    target_tree = cKDTree(target_core)

    selection_data = {
        session_id: session_data(
            session_id,
            run_root=args.run_root,
            protocol=protocol,
            source_protocol=source_protocol,
            geometries=geometries,
            bone_tree=bone_tree,
            target_tree=target_tree,
        )
        for session_id in protocol["splits"]["method_selection_sessions"]
    }
    session_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    selection_summaries = {}
    for hypothesis in protocol["hypotheses"]:
        rows = []
        for session_id, data in selection_data.items():
            record, frames = evaluate_hypothesis(session_id, data, hypothesis, protocol)
            record["split"] = "selection"
            for row in frames:
                row["split"] = "selection"
            rows.append(record)
            session_rows.append(record)
            frame_rows.extend(frames)
        selection_summaries[hypothesis["name"]] = summarize(
            rows, protocol["method_selection_gate"]
        )
    passing = []
    for order, hypothesis in enumerate(protocol["hypotheses"]):
        summary = selection_summaries[hypothesis["name"]]
        if summary["gate_pass"]:
            passing.append(
                (
                    summary["mean_localization_error_mm"],
                    -summary["minimum_selected_supporting_frames"],
                    order,
                    hypothesis["name"],
                )
            )
    chosen = min(passing)[-1] if passing else None
    final_summary = None
    final_data: dict[str, dict[str, Any]] = {}
    if chosen is not None:
        hypothesis = next(row for row in protocol["hypotheses"] if row["name"] == chosen)
        final_rows = []
        for session_id in protocol["splits"]["final_comparison_sessions"]:
            data = session_data(
                session_id,
                run_root=args.run_root,
                protocol=protocol,
                source_protocol=source_protocol,
                geometries=geometries,
                bone_tree=bone_tree,
                target_tree=target_tree,
            )
            final_data[session_id] = data
            record, frames = evaluate_hypothesis(session_id, data, hypothesis, protocol)
            record["split"] = "final_comparison"
            for row in frames:
                row["split"] = "final_comparison"
            final_rows.append(record)
            session_rows.append(record)
            frame_rows.extend(frames)
        final_summary = summarize(final_rows, protocol["final_readiness_gate"])

    all_data = selection_data | final_data
    calibration = {session_id: data["calibration"] for session_id, data in all_data.items()}
    map_geometry = {session_id: data["map_geometry"] for session_id, data in all_data.items()}
    calibration_pass = all(row["passes_coordinate_gate"] for row in calibration.values())
    surface_gate = protocol["surface_evaluator"]
    surface_pass = all(
        row["nearest_bone_surface_distance_mm"]["inlier_fraction"]
        >= surface_gate["minimum_bone_surface_inlier_fraction"]
        and row["target_nontarget_association_fraction"] is not None
        and row["target_nontarget_association_fraction"]
        <= surface_gate["maximum_target_nontarget_association_fraction"]
        for row in map_geometry.values()
    )
    full_pose_public = bool(
        protocol["actor_geometry"]["runtime_pose_audit"][
            "synchronized_6dof_in_historical_public_records"
        ]
    )
    spatial_verdict = (
        "SPATIAL_GEOMETRY_VALIDATED"
        if calibration_pass and surface_pass and full_pose_public
        else "SPATIAL_GEOMETRY_PARTIAL"
        if calibration_pass or surface_pass
        else "SPATIAL_GEOMETRY_UNSUPPORTED"
    )
    localizer_ready = bool(final_summary and final_summary["gate_pass"])
    localizer_verdict = (
        "L5_SPATIAL_LOCALIZER_READY"
        if localizer_ready
        else "L5_SPATIAL_LOCALIZER_PROMISING"
        if chosen is not None
        else "L5_SPATIAL_LOCALIZER_FAILED"
    )
    integration_verdict = (
        "STAGE2_INTEGRATION_INTERFACE_READY"
        if localizer_ready
        else "STAGE2_INTEGRATION_BLOCKED"
    )
    result = {
        "schema": "s0010-l5-spatial-localizer-result-v1",
        "source_commit": args.source_commit,
        "protocol_sha256": expected,
        "source_run": str(args.run_root.resolve()),
        "evidence_status": "historically_opened_development",
        "actor_geometry_audit": protocol["actor_geometry"]["runtime_pose_audit"],
        "calibration": calibration,
        "map_geometry": map_geometry,
        "selection_summaries": selection_summaries,
        "selected_hypothesis": chosen,
        "final_summary": final_summary,
        "session_results": session_rows,
        "spatial_reconstruction_verdict": spatial_verdict,
        "l5_target_localization_verdict": localizer_verdict,
        "integration_verdict": integration_verdict,
        "new_closed_loop_execution": False,
        "full_mbb_intervention": "BLOCKED",
        "runtime_seconds": time.monotonic() - started,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "geometry_validation.json").write_text(
        json.dumps({"calibration": calibration, "map_geometry": map_geometry}, indent=2) + "\n"
    )
    write_csv(output / "session_results.csv", session_rows)
    write_csv(output / "frame_rankings.csv", frame_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    names = list(selection_summaries)
    axes[0].bar(
        np.arange(len(names)) - 0.17,
        [selection_summaries[name]["correct_accepted_sessions"] for name in names],
        0.34,
        label="correct accepted",
    )
    axes[0].bar(
        np.arange(len(names)) + 0.17,
        [selection_summaries[name]["wrong_accepted_sessions"] for name in names],
        0.34,
        label="wrong accepted",
    )
    axes[0].set_xticks(range(len(names)), names, rotation=15)
    axes[0].set_title("Method-selection sessions")
    axes[0].legend()
    for session_id, row in map_geometry.items():
        axes[1].scatter(
            row["nearest_bone_surface_distance_mm"]["inlier_fraction"],
            row["nearest_l5_tp_core_distance_mm"]["inlier_fraction"],
            label=session_id,
        )
    axes[1].set_xlabel("cortical candidates within 3 mm of vertebral surface")
    axes[1].set_ylabel("cortical candidates within 5 mm of L5 TP core")
    axes[1].set_title("Evaluator-only surface consistency")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "localizer_summary.png")
    plt.close(fig)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
