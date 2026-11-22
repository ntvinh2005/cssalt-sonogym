#!/usr/bin/env python3
"""Validate synchronized public robot pose against evaluator-only coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from two_stage_navigation.acquisition_integrity import validate_actor_6dof_record
from two_stage_navigation.ultrasound_plane_geometry import (
    rotation_columns_from_quaternion_wxyz,
)
from two_stage_navigation.ultrasound_spatial_map import (
    calibrated_image_coordinates_mm,
    coordinates_in_target_frame_mm,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def actor_coordinates_in_human_mm(
    actor: dict[str, Any], evaluator: dict[str, Any]
) -> np.ndarray:
    actor_rotation = np.asarray(actor["actor_base_to_ee_rotation_columns"])
    base_coordinates = calibrated_image_coordinates_mm(
        end_effector_position_m=actor["actor_base_to_ee_position_m"],
        rotation_columns=actor_rotation,
        width=int(evaluator["image_size_width_depth"][0]),
        depth=int(evaluator["image_size_width_depth"][1]),
        pixel_resolution_mm=float(evaluator["image_resolution_m"]) * 1000.0,
        image_origin_height_mm=float(evaluator["image_origin_height_m"]) * 1000.0,
        home_up_axis=actor_rotation[:, 2],
    )
    world_to_base_rotation = rotation_columns_from_quaternion_wxyz(
        np.asarray(evaluator["world_to_base_quaternion_wxyz"])
    )
    world_to_human_rotation = rotation_columns_from_quaternion_wxyz(
        np.asarray(evaluator["world_to_human_quaternion_wxyz"])
    )
    return coordinates_in_target_frame_mm(
        base_coordinates,
        world_to_source_position_m=evaluator["world_to_base_position_m"],
        world_to_source_rotation=world_to_base_rotation,
        world_to_target_position_m=evaluator["world_to_human_position_m"],
        world_to_target_rotation=world_to_human_rotation,
    )


def validate_session(session_dir: Path) -> dict[str, Any]:
    actors = read_jsonl(session_dir / "public/frames.jsonl")
    evaluators = read_jsonl(session_dir / "privileged/full_pose_replay.jsonl")
    native = np.load(session_dir / "privileged/pixel_to_volume_coordinates.npz")[
        "human_img_coords"
    ]
    if not len(actors) == len(evaluators) == len(native):
        raise ValueError("actor, evaluator and coordinate frame counts differ")
    errors = []
    acquisition_ids = []
    for actor, evaluator, coordinates in zip(actors, evaluators, native):
        validate_actor_6dof_record(actor)
        acquisition_ids.append(
            actor["acquisition_id"] == evaluator["acquisition_id"]
            and actor["frame_sha256"] == evaluator["frame_sha256"]
            and actor["simulation_step"] == evaluator["simulation_step"]
            and np.isclose(actor["simulation_time_s"], evaluator["simulation_time_s"])
        )
        reconstructed = actor_coordinates_in_human_mm(actor, evaluator)
        expected = (
            coordinates.reshape(150, 200, 3).transpose(1, 0, 2).astype(np.float64)
            * float(evaluator["label_resolution_m"])
            * 1000.0
        )
        errors.append(np.linalg.norm(reconstructed - expected, axis=2).ravel())
    error = np.concatenate(errors)
    return {
        "session_id": actors[0]["session_id"],
        "frame_count": len(actors),
        "acquisition_identity_all_match": bool(all(acquisition_ids)),
        "median_coordinate_error_mm": float(np.median(error)),
        "p90_coordinate_error_mm": float(np.quantile(error, 0.90)),
        "maximum_coordinate_error_mm": float(np.max(error)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--session-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    gate = protocol["actor_6dof_gate"]
    sessions = [
        validate_session(args.run_root / "sessions" / session_id)
        for session_id in args.session_id
    ]
    passed = all(
        row["acquisition_identity_all_match"]
        and row["median_coordinate_error_mm"]
        <= float(gate["maximum_median_coordinate_error_mm"])
        and row["p90_coordinate_error_mm"]
        <= float(gate["maximum_p90_coordinate_error_mm"])
        and row["maximum_coordinate_error_mm"]
        <= float(gate["maximum_coordinate_error_mm"])
        for row in sessions
    )
    result = {
        "passed": passed,
        "protocol_sha256": sha256(args.protocol),
        "actor_pose_source": "robot_base_to_end_effector_proprioception",
        "evaluator_coordinates_used_by_actor": False,
        "sessions": sessions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
