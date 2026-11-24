#!/usr/bin/env python3
"""Acquire a fixed longitudinal/transverse ultrasound sweep and map its observations."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from isaaclab.app import AppLauncher


def checked_sha256(path, expected):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"checksum mismatch: {path}")
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--stage1-actor", required=True, type=Path)
    parser.add_argument("--stage1-session", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evaluator-segmentation", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    launcher = AppLauncher(args)
    app = launcher.app

    import gymnasium as gym
    import nibabel as nib
    import numpy as np
    import yaml
    import isaaclab_tasks  # noqa: F401
    import spinal_surgery  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from stage2_sim import (
        current_realized_pose, move_proportional_target, surface_proxy_ok, tensor_row,
    )
    from spinal_surgery.evaluation.two_stage_navigation.acquisition_integrity import (
        validate_actor_6dof_record, validate_waypoint_bank,
    )
    from spinal_surgery.evaluation.two_stage_navigation.contracts import ultrasound_sha256
    from spinal_surgery.evaluation.two_stage_navigation.incremental_landmark_map import (
        IncrementalLandmarkMapConfig,
    )
    from spinal_surgery.evaluation.two_stage_navigation.simple_landmark_mapping import (
        LEVELS, extract_observations, project_image, simple_map, stage1_region, systematic_sweep,
    )
    from spinal_surgery.evaluation.two_stage_navigation.sonoanatomy_active_scan import (
        staged_orientation_targets,
    )
    from spinal_surgery.evaluation.two_stage_navigation.sonoanatomy_segmentation import (
        load_segmentation_checkpoint, segmentation_probabilities,
    )
    from spinal_surgery.evaluation.two_stage_navigation.stage2_active_pilot import (
        transition_is_valid, validate_episode_horizon,
    )
    from spinal_surgery.evaluation.two_stage_navigation.ultrasound_plane_geometry import (
        rotation_columns_from_quaternion_wxyz, sample_label_image,
    )

    started = time.monotonic()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    public = out / "public"
    public.mkdir()
    protocol_hash, stage1_hash = (
        checked_sha256(path, path.with_suffix(".sha256").read_text().split()[0])
        for path in (args.protocol, args.stage1_actor)
    )
    protocol = yaml.safe_load(args.protocol.read_text())
    stage1 = json.loads(args.stage1_actor.read_text())
    region = stage1_region(stage1, args.stage1_session, protocol)
    models, model_seals = {}, {}
    for level in LEVELS:
        spec = protocol["models"]["sonoanatomy"][level]
        path = Path(spec["path"])
        model_seals[level] = checked_sha256(path, spec["sha256"])
        model, bundle = load_segmentation_checkpoint(path, expected_level=level)
        models[level] = model, bundle["thresholds"]

    volume = None if args.evaluator_segmentation is None else np.asarray(
        nib.load(args.evaluator_segmentation).dataobj, dtype=np.uint8,
    )
    evaluator_poses, native_coordinates, raw_labels = [], [], []
    frames, images, failed_moves = [], [], []
    observations, ambiguities, masks = defaultdict(list), defaultdict(list), defaultdict(list)
    actions = []
    scan = protocol["sonoanatomy_scan"]
    motion = protocol["motion"]
    validate_episode_horizon(motion["maximum_total_action_steps"], motion["episode_length_s"])
    env = None
    completed = False
    try:
        cfg = parse_env_cfg(protocol["scope"]["task"], device=args.device or "cuda:0", num_envs=1)
        cfg.seed, cfg.episode_length_s = args.seed, motion["episode_length_s"]
        env = gym.make(protocol["scope"]["task"], cfg=cfg)
        simulator = env.unwrapped
        sensor = simulator.US_slicer
        sensor.capture_snapshot_enabled = True
        session = out.name
        sensor.capture_session_id = sensor.capture_reset_id = session
        env.reset(seed=args.seed)
        if sensor.sim_mode != "conv":
            raise RuntimeError("systematic sweep requires conv SonoGym ultrasound")
        limits = sensor.x_z_range.detach().cpu().numpy()
        waypoints = systematic_sweep(region, scan, limits[0], limits[1])
        position_tolerance = motion["coarse_position_tolerance_mm"]
        orientation_tolerance = motion["coarse_orientation_tolerance_deg"]
        validate_waypoint_bank(
            [pose for _, pose in waypoints], command_minimum=limits[0], command_maximum=limits[1],
            label_resolution_mm=protocol["geometry"]["label_resolution_mm"],
            position_tolerance_mm=position_tolerance,
            orientation_tolerance_deg=orientation_tolerance, separation_multiplier=2.0,
        )
        (out / "waypoints.json").write_text(json.dumps(waypoints, indent=2) + "\n")
        prior_producer = int(simulator.num_step)
        prior_plane = None
        for waypoint_index, (plane, pose) in enumerate(waypoints):
            target = np.asarray(pose)
            start_action = len(actions)
            intermediates = ()
            if plane != prior_plane:
                intermediates = staged_orientation_targets(
                    current_realized_pose(simulator), target,
                    maximum_step_rad=scan["maximum_orientation_step_rad"],
                )
            try:
                for step_target in (*intermediates, target):
                    remaining = motion["maximum_total_action_steps"] - len(actions)
                    if remaining <= 0:
                        raise RuntimeError("systematic sweep action budget exhausted")
                    move_proportional_target(
                        env, step_target, step_target,
                        maximum_steps=min(remaining, motion["maximum_steps_per_move"]),
                        maximum_position_error_mm=position_tolerance,
                        maximum_orientation_error_deg=orientation_tolerance,
                        ratio=tuple(motion["controller_ratio"]), trajectory_sink=actions,
                    )
            except TimeoutError as failure:
                failed_moves.append({"waypoint": waypoint_index, "reason": str(failure)})
                continue
            prior_plane = plane
            snapshot = sensor.copy_latest_acquisition_snapshot()
            image = np.asarray(snapshot["clean_ultrasound"][0, :, :, 0].detach().cpu(), dtype=np.float32)
            quaternion = tensor_row(snapshot["actor_base_to_ee_quat"])
            producer = int(snapshot["producer_sequence"])
            realized = current_realized_pose(simulator)
            contact_ok = surface_proxy_ok(simulator) and not bool(sensor.no_collide[0].item())
            transition_valid = transition_is_valid(actions[start_action:])
            producer_advanced = producer > prior_producer
            prior_producer = max(producer, prior_producer)
            image_valid = contact_ok and producer_advanced and transition_valid and float(image.std()) > 1e-8
            levels = region.levels_at(realized[1])
            frame = {
                "session_id": session, "stage1_session_id": args.stage1_session,
                "waypoint_index": waypoint_index, "phase": plane,
                "frame_sequence": len(frames), "frame_sha256": ultrasound_sha256(image),
                "producer_sequence": producer,
                "acquisition_id": f"{session}:{snapshot['reset_id']}:{snapshot['acquisition_sequence']}",
                "simulation_step": int(snapshot["simulation_step"]),
                "simulation_time_s": float(snapshot["simulation_time_s"]),
                "actor_base_to_ee_position_m": tensor_row(snapshot["actor_base_to_ee_pos"]).tolist(),
                "actor_base_to_ee_quaternion_wxyz": quaternion.tolist(),
                "actor_base_to_ee_rotation_columns": rotation_columns_from_quaternion_wxyz(quaternion).tolist(),
                "actor_joint_position_rad": tensor_row(snapshot["actor_joint_pos"]).tolist(),
                "requested_relative_pose_index": list(pose),
                "executed_relative_pose_index": realized.tolist(),
                "level_hypotheses": levels,
                "contact_ok": contact_ok, "producer_advanced": producer_advanced,
                "transition_valid": transition_valid, "image_valid": image_valid,
            }
            if volume is not None:
                private_coordinates = np.asarray(
                    snapshot["human_img_coords"][0].detach().cpu(), dtype=np.float32,
                ).copy()
                raw_labels.append(sample_label_image(
                    volume=volume, coordinates_index=private_coordinates,
                    width=image.shape[1], depth=image.shape[0],
                ).astype(np.uint8))
                native_coordinates.append(private_coordinates)
                evaluator_poses.append({
                    key: frame[key] for key in ("frame_sequence", "frame_sha256",
                                                "producer_sequence", "acquisition_id", "session_id")
                } | {
                    "human_to_ee_position_m": tensor_row(snapshot["human_to_ee_pos"]).tolist(),
                    "human_to_ee_rotation_columns": rotation_columns_from_quaternion_wxyz(
                        tensor_row(snapshot["human_to_ee_quat"]),
                    ).tolist(),
                    "label_resolution_m": float(snapshot["label_resolution_m"]),
                    "image_resolution_m": float(snapshot["image_resolution_m"]),
                    "image_origin_height_m": float(snapshot["image_origin_height_m"]),
                })
            frames.append(frame)
            images.append(image.copy())
            if image_valid:
                validate_actor_6dof_record(frame)
                coordinates = project_image(frame, image.shape, protocol["geometry"]["ultrasound_pixel_resolution_mm"])
            for level in LEVELS:
                mask = np.full(image.shape, 255, dtype=np.uint8)
                if image_valid and level in levels:
                    model, thresholds = models[level]
                    probability = segmentation_probabilities(model, image[None])[0]
                    rows, mask = extract_observations(probability.argmax(axis=0), probability,
                                                      thresholds, frame, coordinates)
                    destination = observations if len(levels) == 1 else ambiguities
                    for row in rows:
                        destination[f"RIGHT_{level}_{row.semantic_class}"].append(row)
                masks[level].append(mask)
        completed = True
    finally:
        (public / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in frames))
        np.savez_compressed(public / "ultrasound.npz", ultrasound=np.asarray(images, dtype=np.float32))
        np.savez_compressed(public / "unet_masks.npz", **{level: np.asarray(masks[level]) for level in LEVELS})
        if volume is not None:
            private = out / "privileged"
            private.mkdir()
            (private / "full_pose_replay.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in evaluator_poses)
            )
            np.savez_compressed(private / "raw_labels.npz", raw_labels=np.asarray(raw_labels))
            np.savez_compressed(private / "pixel_to_volume_coordinates.npz",
                                human_img_coords=np.asarray(native_coordinates))
        result = {
            "label": "FRESH SYSTEMATIC SWEEP — DEVELOPMENT",
            "seed": args.seed, "anatomy": protocol["scope"]["patient_id"],
            "completed_trajectory": completed, "failed_moves": failed_moves,
            "action_steps": len(actions), "frame_count": len(frames),
            "valid_frame_count": sum(frame["image_valid"] for frame in frames),
            "elapsed_seconds": time.monotonic() - started,
            "stage1_region": asdict(region), "stage1_sha256": stage1_hash,
            "protocol_sha256": protocol_hash, "checkpoint_sha256": model_seals,
            "observations": {key: [asdict(row) for row in rows] for key, rows in observations.items()},
            "ambiguous_observations": {key: [asdict(row) for row in rows] for key, rows in ambiguities.items()},
            "landmarks": simple_map(observations, ambiguities, IncrementalLandmarkMapConfig(**protocol["landmark_map"]))
            if completed else None,
        }
        (out / "actor_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
