#!/usr/bin/env python3
"""Test calibrated multi-view pair consistency on opened s0010 captures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

import run_s0010_sap_tp_pair_development as development
from two_stage_navigation.ultrasound_sap_tp_configuration import (
    CorticalSegment,
    track_window_pair_candidates,
)
from two_stage_navigation.ultrasound_spatial_map import (
    SpatialMapConfig,
    calibrated_image_coordinates_mm,
    calibrated_pose_is_distinct,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def segment_points(segment: CorticalSegment, coordinates: np.ndarray) -> np.ndarray:
    rows = np.clip(np.rint(segment.rows).astype(int), 0, coordinates.shape[0] - 1)
    columns = np.clip(
        np.rint(segment.columns).astype(int), 0, coordinates.shape[1] - 1
    )
    return coordinates[rows, columns]


def curve_distance(left: np.ndarray, right: np.ndarray) -> float:
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return float(
        max(np.median(np.min(distances, axis=1)), np.median(np.min(distances, axis=0)))
    )


def selected_pair(
    image: np.ndarray,
    pose: dict[str, Any],
    model: Any,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    candidates = track_window_pair_candidates(image, **protocol["track_window_config"])
    if not candidates:
        return {
            "base_score": 0.0,
            "first_segment": None,
            "second_segment": None,
            "first": None,
            "second": None,
            "first_center": None,
            "second_center": None,
        }
    features = np.asarray([feature for _, _, feature in candidates], dtype=np.float64)
    probabilities = model.predict_proba(features)[:, 1]
    selected = int(np.argmax(probabilities))
    first, second, _ = candidates[selected]
    geometry = pair_geometry(first, second, pose)
    return {"base_score": float(probabilities[selected]), **geometry}


def pair_geometry(
    first: CorticalSegment,
    second: CorticalSegment,
    pose: dict[str, Any],
) -> dict[str, Any]:
    if "actor_base_to_ee_position_m" in pose:
        position = pose["actor_base_to_ee_position_m"]
        rotation = pose["actor_base_to_ee_rotation_columns"]
    else:
        position = pose["human_to_ee_position_m"]
        rotation = pose["human_to_ee_rotation_columns"]
    coordinates = calibrated_image_coordinates_mm(
        end_effector_position_m=position,
        rotation_columns=np.asarray(rotation),
        width=150,
        depth=200,
        pixel_resolution_mm=float(pose["image_resolution_m"]) * 1000.0,
        image_origin_height_mm=float(pose["image_origin_height_m"]) * 1000.0,
    )
    first_points = segment_points(first, coordinates)
    second_points = segment_points(second, coordinates)
    return {
        "first_segment": first,
        "second_segment": second,
        "first": first_points,
        "second": second_points,
        "first_center": np.mean(first_points, axis=0),
        "second_center": np.mean(second_points, axis=0),
    }


def add_support(
    rows: list[dict[str, Any]],
    *,
    surface_radius_mm: float,
    relative_vector_tolerance_mm: float,
    minimum_support: int,
) -> None:
    distinct_config = SpatialMapConfig(
        minimum_view_translation_mm=2.0,
        minimum_view_angle_deg=1.0,
    )
    for session_id in sorted({row["session_id"] for row in rows}):
        session = sorted(
            [row for row in rows if row["session_id"] == session_id],
            key=lambda row: row["frame_sequence"],
        )
        for index, row in enumerate(session):
            supports: list[str] = []
            if row["first"] is not None:
                current_vector = row["second_center"] - row["first_center"]
                for previous_index, previous in enumerate(session):
                    if previous_index == index:
                        continue
                    if previous["first"] is None:
                        continue
                    if not calibrated_pose_is_distinct(
                        row["pose_position_m"],
                        row["pose_rotation"],
                        previous["pose_position_m"],
                        previous["pose_rotation"],
                        distinct_config,
                    ):
                        continue
                    first_distance = curve_distance(row["first"], previous["first"])
                    second_distance = curve_distance(row["second"], previous["second"])
                    previous_vector = previous["second_center"] - previous["first_center"]
                    vector_error = float(np.linalg.norm(current_vector - previous_vector))
                    if (
                        max(first_distance, second_distance) <= surface_radius_mm
                        and vector_error <= relative_vector_tolerance_mm
                    ):
                        supports.append(previous["sample_id"])
            row["support_sample_ids"] = supports
            row["support_count"] = len(supports)
            row["accepted"] = bool(
                row["base_score"] >= 0.5 and len(supports) >= minimum_support
            )
            row["accumulated_score"] = min(
                1.0, row["base_score"] + 0.1 * min(len(supports), 2)
            )


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nonambiguous_accepted = [
        row for row in rows if row["accepted"] and row["actual_state"] != "AMBIGUOUS"
    ]
    correct = [row for row in nonambiguous_accepted if row["selected_pair_correct"]]
    negatives = [row for row in rows if row["actual_state"] == "CLEAR_NEGATIVE"]
    accepted_negatives = [row for row in negatives if row["accepted"]]
    sessions = []
    for session_id in sorted({row["session_id"] for row in rows}):
        session_rows = [row for row in rows if row["session_id"] == session_id]
        accepted = sorted(
            [row for row in session_rows if row["accepted"]],
            key=lambda row: row["accumulated_score"],
            reverse=True,
        )
        sessions.append(
            {
                "session_id": session_id,
                "accepted_count": len(accepted),
                "correct_accepted_target_count": sum(
                    row["selected_pair_correct"] and row["actual_state"] == "TARGET"
                    for row in accepted
                ),
                "accepted_clear_negative_count": sum(
                    row["actual_state"] == "CLEAR_NEGATIVE" for row in accepted
                ),
                "top1_correct": bool(accepted and accepted[0]["selected_pair_correct"]),
                "top3_contains_correct": bool(
                    any(row["selected_pair_correct"] for row in accepted[:3])
                ),
            }
        )
    return {
        "accepted_count": len([row for row in rows if row["accepted"]]),
        "correct_accepted_pair_count": len(correct),
        "pair_precision": len(correct) / max(len(nonambiguous_accepted), 1),
        "accepted_clear_negative_count": len(accepted_negatives),
        "clear_negative_false_accept_rate": len(accepted_negatives)
        / max(len(negatives), 1),
        "top1_correct_sessions": sum(row["top1_correct"] for row in sessions),
        "top3_correct_sessions": sum(row["top3_contains_correct"] for row in sessions),
        "session_results": sessions,
    }


def copy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    checkpoint_path = Path(protocol["source"]["checkpoint"])
    if sha256(checkpoint_path) != protocol["source"]["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint hash changed")
    model = joblib.load(checkpoint_path)["model"]
    evaluation = {
        row["sample_id"]: row
        for row in csv.DictReader(
            (args.run_root / "results/evaluation/frame_results.csv").open(
                encoding="utf-8"
            )
        )
    }
    rows: list[dict[str, Any]] = []
    for session in protocol["sessions"]:
        session_id = session["session_id"]
        root = args.run_root / "sessions" / session_id
        frames = read_jsonl(root / "public/frames.jsonl")
        poses = read_jsonl(root / "privileged/full_pose_replay.jsonl")
        images = np.load(root / "public/ultrasound.npz")["ultrasound"]
        for index, (frame, pose, image) in enumerate(zip(frames, poses, images)):
            sample_id = f"{session_id}:{index:02d}"
            frozen = evaluation[sample_id]
            geometry = selected_pair(image, pose, model, protocol)
            if not math.isclose(
                geometry["base_score"], float(frozen["base_score"]), abs_tol=1e-12
            ):
                raise RuntimeError(f"frozen score mismatch: {sample_id}")
            rows.append(
                {
                    "sample_id": sample_id,
                    "session_id": session_id,
                    "frame_sequence": index,
                    "frame_sha256": frame["frame_sha256"],
                    "actual_state": frozen["actual_state"],
                    "selected_pair_correct": frozen["selected_pair_correct"] == "True",
                    "pose_position_m": np.asarray(pose["human_to_ee_position_m"]),
                    "pose_rotation": np.asarray(pose["human_to_ee_rotation_columns"]),
                    "pose_record": pose,
                    "image": image,
                    **geometry,
                }
            )

    configurations = []
    for surface_radius in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        for vector_tolerance in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
            for minimum_support in (1, 2):
                candidate = copy_rows(rows)
                add_support(
                    candidate,
                    surface_radius_mm=surface_radius,
                    relative_vector_tolerance_mm=vector_tolerance,
                    minimum_support=minimum_support,
                )
                result = metrics(candidate)
                configurations.append(
                    {
                        "surface_radius_mm": surface_radius,
                        "relative_vector_tolerance_mm": vector_tolerance,
                        "minimum_support": minimum_support,
                        **{key: value for key, value in result.items() if key != "session_results"},
                        "minimum_correct_per_session": min(
                            row["correct_accepted_target_count"]
                            for row in result["session_results"]
                        ),
                    }
                )
    viable = [
        row
        for row in configurations
        if row["pair_precision"] >= 0.75
        and row["clear_negative_false_accept_rate"] <= 0.10
        and row["minimum_correct_per_session"] >= 2
        and row["top1_correct_sessions"] == 3
        and row["top3_correct_sessions"] == 3
    ]
    selected = (
        sorted(
            viable,
            key=lambda row: (
                -row["correct_accepted_pair_count"],
                row["clear_negative_false_accept_rate"],
                row["surface_radius_mm"],
                row["relative_vector_tolerance_mm"],
                row["minimum_support"],
            ),
        )[0]
        if viable
        else sorted(
            configurations,
            key=lambda row: (
                -row["pair_precision"],
                row["clear_negative_false_accept_rate"],
                -row["correct_accepted_pair_count"],
            ),
        )[0]
    )
    final_rows = copy_rows(rows)
    add_support(
        final_rows,
        surface_radius_mm=selected["surface_radius_mm"],
        relative_vector_tolerance_mm=selected["relative_vector_tolerance_mm"],
        minimum_support=selected["minimum_support"],
    )
    final = metrics(final_rows)
    rng = np.random.default_rng(20261122)
    shuffled_rows = copy_rows(rows)
    for session_id in sorted({row["session_id"] for row in shuffled_rows}):
        indices = [
            index for index, row in enumerate(shuffled_rows) if row["session_id"] == session_id
        ]
        positions = [shuffled_rows[index]["pose_position_m"] for index in indices]
        rotations = [shuffled_rows[index]["pose_rotation"] for index in indices]
        order = rng.permutation(len(indices))
        for destination, source in enumerate(order):
            shuffled_rows[indices[destination]]["pose_position_m"] = positions[source]
            shuffled_rows[indices[destination]]["pose_rotation"] = rotations[source]
            pose = {
                "human_to_ee_position_m": positions[source],
                "human_to_ee_rotation_columns": rotations[source],
                "image_resolution_m": 0.0005,
                "image_origin_height_m": 0.133,
            }
            if shuffled_rows[indices[destination]]["first_segment"] is not None:
                geometry = pair_geometry(
                    shuffled_rows[indices[destination]]["first_segment"],
                    shuffled_rows[indices[destination]]["second_segment"],
                    pose,
                )
                for key in ("first", "second", "first_center", "second_center"):
                    shuffled_rows[indices[destination]][key] = geometry[key]
    add_support(
        shuffled_rows,
        surface_radius_mm=selected["surface_radius_mm"],
        relative_vector_tolerance_mm=selected["relative_vector_tolerance_mm"],
        minimum_support=selected["minimum_support"],
    )
    shuffled = metrics(shuffled_rows)
    image_shuffled_rows = copy_rows(rows)
    for session_id in sorted({row["session_id"] for row in image_shuffled_rows}):
        indices = [
            index
            for index, row in enumerate(image_shuffled_rows)
            if row["session_id"] == session_id
        ]
        order = np.roll(np.arange(len(indices)), 5)
        for destination, source in enumerate(order):
            row = image_shuffled_rows[indices[destination]]
            image = image_shuffled_rows[indices[source]]["image"]
            geometry = selected_pair(image, row["pose_record"], model, protocol)
            row.update(geometry)
            row["selected_pair_correct"] = False
    add_support(
        image_shuffled_rows,
        surface_radius_mm=selected["surface_radius_mm"],
        relative_vector_tolerance_mm=selected["relative_vector_tolerance_mm"],
        minimum_support=selected["minimum_support"],
    )
    image_shuffled = metrics(image_shuffled_rows)

    pose_only_results = []
    for radius_mm in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0):
        for minimum_support in (1, 2):
            pose_rows = copy_rows(rows)
            for session_id in sorted({row["session_id"] for row in pose_rows}):
                session_rows = sorted(
                    [row for row in pose_rows if row["session_id"] == session_id],
                    key=lambda row: row["frame_sequence"],
                )
                for index, row in enumerate(session_rows):
                    support = [
                        previous["sample_id"]
                        for previous_index, previous in enumerate(session_rows)
                        if previous_index != index
                        if np.linalg.norm(
                            row["pose_position_m"] - previous["pose_position_m"]
                        )
                        * 1000.0
                        <= radius_mm
                    ]
                    row["support_sample_ids"] = support
                    row["support_count"] = len(support)
                    row["accepted"] = len(support) >= minimum_support
                    row["accumulated_score"] = min(len(support) / 3.0, 1.0)
                    row["selected_pair_correct"] = row["actual_state"] == "TARGET"
            pose_result = metrics(pose_rows)
            pose_only_results.append(
                {
                    "position_radius_mm": radius_mm,
                    "minimum_support": minimum_support,
                    **{
                        key: value
                        for key, value in pose_result.items()
                        if key != "session_results"
                    },
                    "minimum_correct_per_session": min(
                        row["correct_accepted_target_count"]
                        for row in pose_result["session_results"]
                    ),
                }
            )
    best_pose_only = sorted(
        pose_only_results,
        key=lambda row: (
            -(row["pair_precision"] >= 0.75),
            -(row["clear_negative_false_accept_rate"] <= 0.10),
            -row["minimum_correct_per_session"],
            -row["correct_accepted_pair_count"],
            row["clear_negative_false_accept_rate"],
            row["position_radius_mm"],
        ),
    )[0]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    serializable = []
    for row in final_rows:
        serializable.append(
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "first",
                    "second",
                    "first_segment",
                    "second_segment",
                    "first_center",
                    "second_center",
                    "pose_position_m",
                    "pose_rotation",
                    "pose_record",
                    "image",
                }
            }
            | {"support_sample_ids": ";".join(row["support_sample_ids"])}
        )
    write_csv(args.output_dir / "frame_results.csv", serializable)
    write_csv(args.output_dir / "configuration_results.csv", configurations)
    result = {
        "opened_development_only": True,
        "historical_pose_source_is_evaluator_only": True,
        "actor_runtime_promotion_forbidden": True,
        "selected_configuration": selected,
        "viable_configuration_count": len(viable),
        "calibrated_geometry": final,
        "shuffled_pose_control": shuffled,
        "shuffled_image_control": image_shuffled,
        "best_pose_only_control": best_pose_only,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
