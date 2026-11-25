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
    parser.add_argument("--revision", type=Path)
    parser.add_argument("--robustness-arm", choices=("E1", "E1-M", "E2"))
    parser.add_argument("--robustness-profile", type=Path)
    parser.add_argument("--record-robot", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    launcher = AppLauncher(args)
    app = launcher.app

    import cv2
    import gymnasium as gym
    import nibabel as nib
    import numpy as np
    import yaml
    import isaaclab_tasks  # noqa: F401
    import spinal_surgery  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from stage2_sim import (
        current_realized_pose, move_proportional_target, surface_proxy_ok, tensor_row, tracking_errors,
    )
    from spinal_surgery.evaluation.two_stage_navigation.acquisition_integrity import (
        validate_actor_6dof_record, validate_waypoint_bank,
    )
    from spinal_surgery.evaluation.two_stage_navigation.contracts import ultrasound_sha256
    from spinal_surgery.evaluation.two_stage_navigation.incremental_landmark_map import (
        IncrementalLandmarkMapConfig,
    )
    from spinal_surgery.evaluation.two_stage_navigation.simple_landmark_mapping import (
        LEVELS, add_outer_transverse_samples, extract_observations, project_image, sap_level_at,
        simple_map, stage1_region, systematic_sweep, transverse_cranial_coordinate,
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

    from robust_sweep import append_transverse_lanes, hold_probe, nominal_probe_transform, transform_error

    if bool(args.robustness_profile) != bool(args.robustness_arm) or (args.robustness_arm and not args.revision):
        parser.error("robustness requires a profile, arm and sealed SAP revision")
    robustness = yaml.safe_load(args.robustness_profile.read_text()) if args.robustness_profile else None
    if robustness and args.seed not in robustness.get("capture_seeds", robustness["development_seeds"]):
        parser.error("seed is outside the predeclared development comparison")

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
    revision_hash = None
    tp_from_added_waypoints = True
    if args.revision:
        revision_hash = checked_sha256(args.revision, args.revision.with_suffix(".sha256").read_text().split()[0])
        revision = yaml.safe_load(args.revision.read_text())
        if (revision["source_protocol_sha256"] != protocol_hash
                or revision["scope"] != protocol["scope"]["patient_id"]
                or args.stage1_session != revision["stage1_session"] or args.seed not in revision["seeds"]
                or revision["acquisition"] != "outer_transverse_midpoints_near_l4"
                or revision["sap_level_assignment"] != "projected_position_with_stage1_reference_bands"
                or args.record_robot != (args.seed == revision["demo_seed"])):
            raise ValueError("execution does not match the sealed SAP revision")
        tp_from_added_waypoints = revision.get("tp_evidence_on_added_sap_waypoints", True)
        if not isinstance(tp_from_added_waypoints, bool):
            raise ValueError("TP evidence selection must be Boolean")
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
    sap_proposals, sap_routing = defaultdict(list), []
    calibration, camera_error, initial_pose = None, None, None
    camera_dir = public / "robot"
    if args.record_robot:
        camera_dir.mkdir()

    def capture_camera(name):
        nonlocal camera_error
        if not args.record_robot or camera_error:
            return
        try:
            view = np.asarray(env.render())
            rgb = cv2.cvtColor(view[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(camera_dir / name), rgb):
                raise RuntimeError("could not write robot camera frame")
        except (AttributeError, IndexError, TypeError, ValueError, RuntimeError, cv2.error) as error:
            camera_error = f"{type(error).__name__}: {error}"

    scan = protocol["sonoanatomy_scan"]
    motion = protocol["motion"]
    validate_episode_horizon(motion["maximum_total_action_steps"], motion["episode_length_s"])
    env = None
    completed = False
    try:
        cfg = parse_env_cfg(protocol["scope"]["task"], device=args.device or "cuda:0", num_envs=1)
        cfg.seed, cfg.episode_length_s = args.seed, motion["episode_length_s"]
        env = gym.make(protocol["scope"]["task"], cfg=cfg,
                       render_mode="rgb_array" if args.record_robot else None)
        simulator = env.unwrapped
        sensor = simulator.US_slicer
        sensor.capture_snapshot_enabled = True
        session = out.name
        sensor.capture_session_id = sensor.capture_reset_id = session
        env.reset(seed=args.seed)
        initial_pose = current_realized_pose(simulator).tolist()
        if args.record_robot:
            visual = protocol["visualization"]
            simulator.sim.set_camera_view(eye=tuple(visual["camera_eye_world_m"]),
                                          target=tuple(visual["camera_target_world_m"]))
            capture_camera("reset.jpg")
        if sensor.sim_mode != "conv":
            raise RuntimeError("systematic sweep requires conv SonoGym ultrasound")
        limits = sensor.x_z_range.detach().cpu().numpy()
        waypoints = systematic_sweep(region, scan, limits[0], limits[1])
        original_poses = {pose for _, pose in waypoints}
        position_tolerance = motion["coarse_position_tolerance_mm"]
        orientation_tolerance = motion["coarse_orientation_tolerance_deg"]
        if args.revision:
            waypoints = add_outer_transverse_samples(
                waypoints, region, scan["reference_offsets_index"],
                minimum_separation_index=2 * position_tolerance / protocol["geometry"]["label_resolution_mm"],
            )
        if args.robustness_arm in ("E1", "E1-M"):
            density = robustness["E1"]
            offsets = density["medium_lateral_offsets_index"]
            length = density["medium"]
            if args.robustness_arm == "E1":
                offsets = offsets + density["dense_lateral_offsets_index"]
                length = density["dense"]
            waypoints = append_transverse_lanes(waypoints, region, offsets)
            if len(waypoints) != length:
                raise ValueError("waypoint count differs from the profile")
        stabilization = robustness["E2"] if args.robustness_arm == "E2" else None
        if stabilization:
            position_tolerance = stabilization["position_tolerance_mm"]
            orientation_tolerance = stabilization["orientation_tolerance_deg"]
        validate_waypoint_bank(
            [pose for _, pose in waypoints], command_minimum=limits[0], command_maximum=limits[1],
            label_resolution_mm=protocol["geometry"]["label_resolution_mm"],
            position_tolerance_mm=position_tolerance,
            orientation_tolerance_deg=orientation_tolerance, separation_multiplier=2.0,
        )
        (out / "waypoints.json").write_text(json.dumps(waypoints, indent=2) + "\n")
        prior_producer = int(simulator.num_step)
        prior_plane = None
        sweep_started = time.monotonic()
        for waypoint_index, (plane, pose) in enumerate(waypoints):
            target = np.asarray(pose)
            waypoint_started = time.monotonic()
            probes_samples = []
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
                if stabilization:
                    steps = stabilization["hold_action_steps"]
                    for attempt in range(stabilization["maximum_retries"] + 1):
                        if len(actions) + steps >= motion["maximum_total_action_steps"]:
                            raise TimeoutError("action budget exceeded during stabilization")
                        before = sensor.copy_latest_acquisition_snapshot()
                        hold_probe(env, sensor, actions, steps)
                        after = sensor.copy_latest_acquisition_snapshot()
                        probes_samples.append(transform_error(
                            tensor_row(before["human_to_ee_pos"]), tensor_row(before["human_to_ee_quat"]),
                            tensor_row(after["human_to_ee_pos"]), tensor_row(after["human_to_ee_quat"]),
                        ))
                        pos_error, angle_error = tracking_errors(simulator, target)
                        if pos_error <= position_tolerance and angle_error <= orientation_tolerance:
                            break
                        if attempt == stabilization["maximum_retries"]:
                            raise TimeoutError("did not retain pose tolerance after bounded hold")
                        move_proportional_target(
                            env, target, target, maximum_steps=min(
                                motion["maximum_steps_per_move"], motion["maximum_total_action_steps"] - len(actions),
                            ), maximum_position_error_mm=position_tolerance,
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
            tp_eligible = tp_from_added_waypoints or pose in original_poses
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
                "original_waypoint": pose in original_poses,
                "tp_eligible_for_mapping": tp_eligible,
                "contact_ok": contact_ok, "producer_advanced": producer_advanced,
                "transition_valid": transition_valid, "image_valid": image_valid,
            }
            if robustness:
                nominal_position, nominal_quaternion = nominal_probe_transform(sensor, pose)
                pos_error, rot_error = transform_error(
                    tensor_row(nominal_position), tensor_row(nominal_quaternion),
                    tensor_row(snapshot["human_to_ee_pos"]), tensor_row(snapshot["human_to_ee_quat"]),
                )
                frame.update(
                    robustness_arm=args.robustness_arm,
                    commanded_nominal_position_m=tensor_row(nominal_position).tolist(),
                    commanded_nominal_rotation_columns=rotation_columns_from_quaternion_wxyz(
                        tensor_row(nominal_quaternion),
                    ).tolist(),
                    full_nominal_position_error_mm=pos_error, full_nominal_rotation_error_deg=rot_error,
                    reduced_tracking_error=tracking_errors(simulator, target),
                    stabilization_deltas_mm_deg=probes_samples,
                    sweep_elapsed_seconds=time.monotonic() - sweep_started,
                    waypoint_elapsed_seconds=time.monotonic() - waypoint_started,
                    action_steps_at_capture=len(actions),
                )
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
            capture_camera(f"{frame['frame_sequence']:03d}.jpg")
            if image_valid:
                validate_actor_6dof_record(frame)
                coordinates = project_image(frame, image.shape, protocol["geometry"]["ultrasound_pixel_resolution_mm"])
            for level in LEVELS:
                mask = np.full(image.shape, 255, dtype=np.uint8)
                if image_valid and (level in levels or args.revision):
                    model, thresholds = models[level]
                    probability = segmentation_probabilities(model, image[None])[0]
                    rows, mask = extract_observations(probability.argmax(axis=0), probability,
                                                      thresholds, frame, coordinates)
                    destination = observations if len(levels) == 1 else ambiguities
                    for row in rows:
                        if args.revision and row.semantic_class == "SAP":
                            sap_proposals[level].append(row)
                        elif level in levels and (row.semantic_class != "TP" or tp_eligible):
                            destination[f"RIGHT_{level}_{row.semantic_class}"].append(row)
                masks[level].append(mask)
        if args.revision:
            calibration = transverse_cranial_coordinate(
                frames, protocol["geometry"]["label_resolution_mm"], scan["translation_step_index"] / 2,
            )
            for level, rows in sap_proposals.items():
                identity = f"RIGHT_{level}_SAP"
                for row in rows:
                    index, candidates = sap_level_at(row.coordinate_mm, region, calibration)
                    status = "LEVEL_MISMATCH"
                    if candidates == (level,):
                        observations[identity].append(row)
                        status = "ELIGIBLE"
                    elif len(candidates) > 1:
                        ambiguities[identity].append(row)
                        status = "AMBIGUOUS_LEVEL"
                    elif not candidates:
                        status = "OUTSIDE_L4_L5"
                    sap_routing.append({
                        "checkpoint_level": level, "projected_craniocaudal_index": index,
                        "candidate_levels": candidates, "status": status, **asdict(row),
                    })
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
        started_mapping = time.monotonic()
        mapping = (
            simple_map(observations, ambiguities, IncrementalLandmarkMapConfig(**protocol["landmark_map"]))
            if completed else None
        )
        mapping_seconds = time.monotonic() - started_mapping
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
            "initial_executed_pose_index": initial_pose,
            "revision_sha256": revision_hash, "robot_camera_error": camera_error,
            "tp_evidence_on_added_sap_waypoints": tp_from_added_waypoints,
            "cranial_coordinate": asdict(calibration) if calibration else None,
            "sap_routing": sap_routing,
            "mapping_seconds": mapping_seconds,
            **({"robustness_arm": args.robustness_arm,
                "robustness_profile_sha256": hashlib.sha256(args.robustness_profile.read_bytes()).hexdigest()}
               if robustness else {}),
            "landmarks": mapping,
        }
        (out / "actor_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
