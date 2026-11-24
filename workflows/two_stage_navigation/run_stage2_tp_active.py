#!/usr/bin/env python3
"""Run actor-only L4/L5 TP active acquisition with offline replay snapshots."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--protocol", required=True)
parser.add_argument("--phase1-actor-result", required=True)
parser.add_argument("--stage1-session-id", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--assets-data-dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("MPLBACKEND", "Agg")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import joblib  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
import spinal_surgery  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from spinal_surgery.evaluation.two_stage_navigation.contracts import (  # noqa: E402
    assert_no_privileged_fields,
    ultrasound_sha256,
)
from spinal_surgery.evaluation.two_stage_navigation.acquisition_integrity import (  # noqa: E402
    validate_actor_6dof_record,
)
from spinal_surgery.evaluation.two_stage_navigation.coordinates import wrap_to_pi  # noqa: E402
from spinal_surgery.evaluation.two_stage_navigation.stage1_features import (  # noqa: E402
    PatchSpec,
)
from spinal_surgery.evaluation.two_stage_navigation.stage2_active_pilot import (  # noqa: E402
    offset_poses,
    serpentine_offsets,
    transition_is_valid,
    validate_confirmation_motion,
    validate_episode_horizon,
    validate_physical_grid,
)
from spinal_surgery.evaluation.two_stage_navigation.stage2_tp_active import (  # noqa: E402
    POC_PROTOCOL_IDS,
    TargetView,
    actor_confirmation_requirement_met,
    group_a_target_handoff_pose,
    score_target_image,
    perception_mode,
    select_tp_reacquisition_confirmations,
    select_tp_refinement_anchor,
    select_target_view,
    tp_reacquisition_offsets,
    tp_refinement_is_required,
)
from spinal_surgery.evaluation.two_stage_navigation.sonoanatomy_active_scan import (  # noqa: E402
    ClinicianSweepController,
    ScanConfig,
    ScanEvidence,
    ScanState,
    pair_confirmation_requirement_met,
    staged_orientation_targets,
    tp_handoff_crosscheck_requirement_met,
)
from spinal_surgery.evaluation.two_stage_navigation.sonoanatomy_multiview import (  # noqa: E402
    MultiviewConfig,
    SemanticObservation,
    cluster_observations,
    decide_tp_sap_pair,
    semantic_observations,
)
from spinal_surgery.evaluation.two_stage_navigation.incremental_landmark_map import (  # noqa: E402
    IncrementalAnatomicalLandmarkMap,
    IncrementalLandmarkMapConfig,
    select_landmark_trajectory,
)
from spinal_surgery.evaluation.two_stage_navigation.sonoanatomy_segmentation import (  # noqa: E402
    load_segmentation_checkpoint,
    predicted_presence,
    segmentation_probabilities,
)
from spinal_surgery.evaluation.two_stage_navigation.ultrasound_plane_geometry import (  # noqa: E402
    plane_from_full_pose,
    rotation_columns_from_quaternion_wxyz,
    sample_label_image,
)
from spinal_surgery.evaluation.two_stage_navigation.ultrasound_spatial_map import (  # noqa: E402
    calibrated_image_coordinates_mm,
)

from stage2_sim import (  # noqa: E402
    current_realized_pose,
    move_proportional_target,
    surface_proxy_ok,
    tensor_row,
)


FEATURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/stage1_features.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def validate_seal(path: Path) -> str:
    expected = path.with_suffix(".sha256").read_text().split()[0]
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"seal mismatch for {path}")
    return observed


def load_feature_pickle_support() -> None:
    """Register the historical module name embedded in the frozen L5 pickle."""

    if "tp_proxy_features" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location("tp_proxy_features", FEATURE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot register frozen TP feature module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def patch_spec(value: Any) -> PatchSpec:
    if isinstance(value, PatchSpec):
        return value
    if hasattr(value, "__dict__"):
        value = value.__dict__
    return PatchSpec(**{key: value[key] for key in PatchSpec.__dataclass_fields__})


class RobotVideo:
    def __init__(self, env: Any, path: Path, *, fps: int, stride: int) -> None:
        self.env = env
        self.path = path
        self.fps = fps
        self.stride = stride
        self.actions = 0
        self.frames = 0
        self.phase = "initial"
        self.writer: cv2.VideoWriter | None = None

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def capture(self, _status: dict[str, Any] | None = None, *, force: bool = False) -> None:
        if not force:
            self.actions += 1
            if self.actions % self.stride:
                return
        frame = np.asarray(self.env.render())[:, :, :3].astype(np.uint8)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 42), (10, 10, 10), -1)
        cv2.putText(
            bgr,
            f"s0010 MBB POC | {self.phase} | action {self.actions}",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                (bgr.shape[1], bgr.shape[0]),
            )
            if not self.writer.isOpened():
                raise RuntimeError("cannot create robot video")
        self.writer.write(bgr)
        self.frames += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def main() -> int:
    started = time.monotonic()
    protocol_path = Path(args.protocol).resolve()
    protocol_hash = validate_seal(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol.get("protocol_id") not in POC_PROTOCOL_IDS:
        raise RuntimeError("unexpected protocol")
    session_spec = next(
        (row for row in protocol["execution"]["sessions"] if row["stage1_session_id"] == args.stage1_session_id),
        None,
    )
    if session_spec is None:
        raise RuntimeError("stage1 session is absent from protocol")

    output = Path(args.output_dir).resolve()
    public_dir = output / "public"
    privileged_dir = output / "privileged"
    result_dir = output / "results"
    summary: dict[str, Any] = {
        "schema": "sonogym-s0010-mbb-poc-run-summary-v1",
        "status": "RUNTIME_FAIL",
        "protocol_sha256": protocol_hash,
        "stage1_session_id": args.stage1_session_id,
    }
    env = None
    video: RobotVideo | None = None
    images: list[np.ndarray] = []
    raw_labels: list[np.ndarray] = []
    coordinate_maps: list[np.ndarray] = []
    public_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    target_results: list[dict[str, Any]] = []
    local_coordinates: np.ndarray | None = None
    try:
        actor_path = Path(args.phase1_actor_result).resolve()
        actor_hash = validate_seal(actor_path)
        actor = json.loads(actor_path.read_text())

        model_specs = protocol["models"]
        active_perception_mode = perception_mode(protocol)
        unet_only = active_perception_mode == "unet_only"
        sonoanatomy_models: dict[str, dict[str, Any]] = {}
        for level, specification in model_specs.get("sonoanatomy", {}).items():
            path = Path(specification["path"])
            if sha256(path) != specification["sha256"]:
                raise RuntimeError(f"{level} sonoanatomy model checksum mismatch")
            model, bundle = load_segmentation_checkpoint(path, expected_level=level)
            sonoanatomy_models[level] = {
                "model": model,
                "thresholds": bundle["thresholds"],
                "path": path,
                "sha256": specification["sha256"],
            }
        if sonoanatomy_models and set(sonoanatomy_models) != {"L4", "L5"}:
            raise RuntimeError("sonoanatomy scanning requires both L4 and L5 checkpoints")
        if unet_only:
            if set(model_specs) != {"sonoanatomy"}:
                raise RuntimeError("U-Net-only mode forbids legacy TP model specifications")
            models = {
                f"{level}_RIGHT_TP_SURFACE_CORE": {
                    "threshold": float(
                        sonoanatomy_models[level]["thresholds"]["TP"]["threshold"]
                    )
                }
                for level in ("L4", "L5")
            }
        else:
            load_feature_pickle_support()
            l4_path = Path(model_specs["l4"]["path"])
            l5_path = Path(model_specs["l5"]["path"])
            if sha256(l4_path) != model_specs["l4"]["sha256"]:
                raise RuntimeError("L4 model checksum mismatch")
            if sha256(l5_path) != model_specs["l5"]["sha256"]:
                raise RuntimeError("L5 model checksum mismatch")
            l4_bundle = joblib.load(l4_path)
            l5_bundle = joblib.load(l5_path)
            models = {
                "L4_RIGHT_TP_SURFACE_CORE": {
                    "estimator": l4_bundle["estimator"],
                    "class_id": 1,
                    "threshold": float(l4_bundle["threshold"]),
                    "patch_spec": patch_spec(l4_bundle["patch_spec"]),
                },
                "L5_RIGHT_TP_SURFACE_CORE": {
                    "estimator": l5_bundle["estimator"],
                    "class_id": 2,
                    "threshold": float(l5_bundle["thresholds"][2]),
                    "patch_spec": patch_spec(l5_bundle["patch_spec"]),
                },
            }
            if (
                models["L4_RIGHT_TP_SURFACE_CORE"]["patch_spec"]
                != models["L5_RIGHT_TP_SURFACE_CORE"]["patch_spec"]
            ):
                raise RuntimeError("L4 and L5 preprocessing contracts differ")
        landmark_map_config = None
        shared_landmark_map = None
        if "landmark_map" in protocol:
            landmark_map_config = IncrementalLandmarkMapConfig(
                **protocol["landmark_map"]
            )
            shared_landmark_map = IncrementalAnatomicalLandmarkMap(
                landmark_map_config
            )

        env_cfg = parse_env_cfg(protocol["scope"]["task"], device=args.device or "cuda:0", num_envs=1)
        env_cfg.seed = int(session_spec["seed"])
        motion = protocol["motion"]
        validate_episode_horizon(int(motion["maximum_total_action_steps"]), float(motion["episode_length_s"]))
        env_cfg.episode_length_s = float(motion["episode_length_s"])
        visual = protocol["visualization"]
        env = gym.make(
            protocol["scope"]["task"], cfg=env_cfg,
            render_mode="rgb_array" if bool(visual["record_robot_video"]) else None,
        )
        env.unwrapped.US_slicer.capture_snapshot_enabled = True
        env.unwrapped.US_slicer.capture_session_id = str(session_spec["session_id"])
        env.unwrapped.US_slicer.capture_reset_id = str(session_spec["session_id"])
        observation, _ = env.reset(seed=int(session_spec["seed"]))
        unwrapped = env.unwrapped
        if tuple(observation["policy"].shape) != (1, 1, 200, 150):
            raise RuntimeError("unexpected ultrasound observation shape")
        if bool(visual["record_robot_video"]):
            unwrapped.sim.set_camera_view(
                eye=tuple(visual["camera_eye_world_m"]),
                target=tuple(visual["camera_target_world_m"]),
            )
            video = RobotVideo(
                env, output / "video/robot_probe_patient.mp4",
                fps=int(visual["fps"]), stride=int(visual["action_stride"]),
            )
            video.capture(force=True)

        raw_volume = np.asarray(
            nib.load(
                Path(args.assets_data_dir)
                / "HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
            ).dataobj,
            dtype=np.uint8,
        )
        label_resolution = float(protocol["geometry"]["label_resolution_mm"])
        total_actions = 0
        previous_producer = int(unwrapped.num_step)

        def move(target: np.ndarray, phase: str, position_tolerance: float, angle_tolerance: float) -> list[dict[str, Any]]:
            nonlocal total_actions
            if video is not None:
                video.set_phase(phase)
            trajectory: list[dict[str, Any]] = []
            try:
                try:
                    move_proportional_target(
                        env, target, target,
                        maximum_steps=int(motion["maximum_steps_per_move"]),
                        maximum_position_error_mm=position_tolerance,
                        maximum_orientation_error_deg=angle_tolerance,
                        ratio=tuple(motion["controller_ratio"]),
                        allow_angle_ceiling_saturation=False,
                        trajectory_sink=trajectory,
                        step_callback=None if video is None else video.capture,
                    )
                finally:
                    total_actions += len(trajectory)
                    for row in trajectory:
                        row["phase"] = phase
                    action_rows.extend(trajectory)
            except Exception:
                if total_actions > int(motion["maximum_total_action_steps"]):
                    raise RuntimeError("action budget exceeded")
                raise
            if total_actions > int(motion["maximum_total_action_steps"]):
                raise RuntimeError("action budget exceeded")
            return trajectory

        def stage_orientation_change(target: np.ndarray, phase: str) -> None:
            """Execute large plane changes without coupling them to translation."""

            current = current_realized_pose(unwrapped).copy()
            maximum_step = float(
                protocol.get("sonoanatomy_scan", {}).get(
                    "maximum_orientation_step_rad", 0.35
                )
            )
            intermediates = staged_orientation_targets(
                current,
                target,
                maximum_step_rad=maximum_step,
            )
            maximum_attempts = int(
                protocol.get("sonoanatomy_scan", {}).get(
                    "maximum_orientation_stage_attempts", 1
                )
            )
            if maximum_attempts < 1:
                raise ValueError("orientation stage attempts must be positive")
            for index, intermediate in enumerate(intermediates, start=1):
                for attempt in range(1, maximum_attempts + 1):
                    try:
                        move(
                            intermediate,
                            f"{phase}:orientation_stage_{index}_of_{len(intermediates)}"
                            f":attempt_{attempt}_of_{maximum_attempts}",
                            float(motion["coarse_position_tolerance_mm"]),
                            float(motion["coarse_orientation_tolerance_deg"]),
                        )
                        break
                    except TimeoutError:
                        if attempt == maximum_attempts:
                            raise

        def acquire(target_pose: np.ndarray, target_name: str, phase: str, position_tolerance: float, angle_tolerance: float) -> TargetView:
            nonlocal previous_producer, local_coordinates
            trajectory = move(target_pose, phase, position_tolerance, angle_tolerance)
            snapshot = unwrapped.US_slicer.copy_latest_acquisition_snapshot()
            image = np.asarray(
                snapshot["clean_ultrasound"][0, :, :, 0].detach().cpu().numpy(),
                dtype=np.float32,
            ).copy()
            native_coordinates = np.asarray(
                snapshot["human_img_coords"][0].detach().cpu().numpy(), dtype=np.float64
            ).copy()
            width, depth = (int(value) for value in snapshot["image_size_width_depth"])
            raw = sample_label_image(
                volume=raw_volume, coordinates_index=native_coordinates,
                width=width, depth=depth,
            ).astype(np.uint8, copy=False)
            producer = int(snapshot["producer_sequence"])
            producer_advanced = producer > previous_producer
            previous_producer = producer
            transition_valid = transition_is_valid(trajectory)
            contact_ok = bool(
                surface_proxy_ok(unwrapped)
                and not bool(unwrapped.US_slicer.no_collide[0].item())
            )
            image_valid = bool(
                transition_valid and producer_advanced and contact_ok
                and float(np.std(image)) > 1e-8
            )
            realized = current_realized_pose(unwrapped).copy()
            sequence = len(public_rows)
            frame_hash = ultrasound_sha256(image)
            actor_position = tensor_row(snapshot["actor_base_to_ee_pos"])
            actor_quaternion = tensor_row(snapshot["actor_base_to_ee_quat"])
            actor_rotation = rotation_columns_from_quaternion_wxyz(actor_quaternion)
            model = models[target_name]
            if unet_only:
                level = target_name.split("_", 1)[0]
                semantic_probability = segmentation_probabilities(
                    sonoanatomy_models[level]["model"], np.asarray([image])
                )[0]
                tp_observations = [
                    row
                    for row in semantic_observations(
                        semantic_probability,
                        thresholds=sonoanatomy_models[level]["thresholds"],
                        end_effector_position_m=actor_position,
                        rotation_columns=actor_rotation,
                        frame_sequence=sequence,
                        frame_sha256=frame_hash,
                    )
                    if row.semantic_class == "TP"
                ]
                proposal = (
                    max(
                        tp_observations,
                        key=lambda row: (row.pixel_count, row.confidence),
                    )
                    if tp_observations
                    else None
                )
                probability = 0.0 if proposal is None else float(proposal.confidence)
                proposal_count = len(tp_observations)
            else:
                probability, proposal, proposal_count = score_target_image(
                    image,
                    target=target_name,
                    estimator=model["estimator"],
                    class_id=int(model["class_id"]),
                    patch_spec=model["patch_spec"],
                    column_stride_pixels=int(
                        protocol["perception"]["proposal_column_stride_pixels"]
                    ),
                )
            view = TargetView(
                target=target_name,
                frame_sequence=sequence,
                producer_sequence=producer,
                frame_sha256=frame_hash,
                pose_index=tuple(float(value) for value in realized),
                probability=probability,
                proposal_row=None if proposal is None else float(proposal.row),
                proposal_column=None if proposal is None else float(proposal.column),
                proposal_track_id=(
                    None
                    if proposal is None
                    else getattr(proposal, "track_id", "unet_tp_component")
                ),
                transition_valid=transition_valid,
                image_valid=image_valid,
                producer_advanced=producer_advanced,
            )
            public = {
                "schema": "sonogym-stage2-tp-active-actor-frame-v1",
                "session_id": str(session_spec["session_id"]),
                "stage1_session_id": args.stage1_session_id,
                "frame_sequence": sequence,
                "producer_sequence": producer,
                "frame_sha256": frame_hash,
                "target_name": target_name,
                "phase": phase,
                "requested_relative_pose_index": [float(value) for value in target_pose],
                "executed_relative_pose_index": [float(value) for value in realized],
                "target_probability": probability,
                "target_threshold": float(model["threshold"]),
                "perception_mode": active_perception_mode,
                "proposal_row": view.proposal_row,
                "proposal_column": view.proposal_column,
                "proposal_track_id": view.proposal_track_id,
                "proposal_count": proposal_count,
                "transition_valid": transition_valid,
                "image_valid": image_valid,
                "contact_ok": contact_ok,
                "producer_advanced": producer_advanced,
                "action_step_count": len(trajectory),
                "actor_base_to_ee_position_m": actor_position.tolist(),
                "actor_base_to_ee_quaternion_wxyz": actor_quaternion.tolist(),
                "actor_base_to_ee_rotation_columns": actor_rotation.tolist(),
                "actor_joint_position_rad": tensor_row(snapshot["actor_joint_pos"]).tolist(),
                "simulation_step": int(snapshot["simulation_step"]),
                "simulation_time_s": float(snapshot["simulation_time_s"]),
            }
            assert_no_privileged_fields(public)
            validate_actor_6dof_record(public)
            local = np.asarray(
                snapshot["native_local_image_coordinates_m"].detach().cpu().numpy(), dtype=np.float64
            )
            if local_coordinates is None:
                local_coordinates = local.copy()
            elif not np.array_equal(local_coordinates, local):
                raise RuntimeError("native image coordinate grid changed")
            human_position = tensor_row(snapshot["human_to_ee_pos"])
            human_quaternion = tensor_row(snapshot["human_to_ee_quat"])
            rotation = rotation_columns_from_quaternion_wxyz(human_quaternion)
            replay = plane_from_full_pose(
                end_effector_position_m=human_position,
                rotation_columns=rotation,
                image_origin_height_m=float(snapshot["image_origin_height_m"]),
                local_coordinates_m=local,
                label_resolution_m=float(snapshot["label_resolution_m"]),
                volume_shape=tuple(raw_volume.shape),
            )
            maximum_coordinate_error = float(np.max(np.abs(native_coordinates - replay.coordinates_index)))
            if maximum_coordinate_error > float(protocol["evaluator"]["maximum_coordinate_error_index"]):
                raise RuntimeError("full-pose replay coordinate mismatch")
            acquisition_id = f"{session_spec['session_id']}:{snapshot['reset_id']}:{snapshot['acquisition_sequence']}"
            pose_rows.append(
                {
                    "schema": "sonogym-exact-pose-replay-frame-v1",
                    "session_id": str(session_spec["session_id"]),
                    "frame_sequence": sequence,
                    "producer_sequence": producer,
                    "acquisition_id": acquisition_id,
                    "frame_sha256": frame_hash,
                    "human_to_ee_position_m": human_position.tolist(),
                    "human_to_ee_quaternion_wxyz": human_quaternion.tolist(),
                    "human_to_ee_rotation_columns": rotation.tolist(),
                    "world_to_base_position_m": tensor_row(
                        snapshot["world_to_base_pos"]
                    ).tolist(),
                    "world_to_base_quaternion_wxyz": tensor_row(
                        snapshot["world_to_base_quat"]
                    ).tolist(),
                    "image_origin_height_m": float(snapshot["image_origin_height_m"]),
                    "label_resolution_m": float(snapshot["label_resolution_m"]),
                    "image_resolution_m": float(snapshot["image_resolution_m"]),
                    "image_size_width_depth": [width, depth],
                    "maximum_coordinate_error_index": maximum_coordinate_error,
                    "native_coordinate_array_sha256": array_sha256(native_coordinates.astype(np.float32)),
                    "raw_label_array_sha256": array_sha256(raw),
                }
            )
            public["acquisition_id"] = acquisition_id
            assert_no_privileged_fields(public)
            public_rows.append(public)
            images.append(image)
            raw_labels.append(raw)
            coordinate_maps.append(native_coordinates.astype(np.float32))
            return view

        search = protocol["search"]
        grid = validate_physical_grid(
            search["coarse_x_offsets_index"], search["coarse_z_offsets_index"],
            search["coarse_angle_offsets_rad"], label_resolution_mm=label_resolution,
            position_tolerance_mm=float(motion["coarse_position_tolerance_mm"]),
            orientation_tolerance_deg=float(motion["coarse_orientation_tolerance_deg"]),
        )
        validate_confirmation_motion(
            search["confirmation_dither_offsets"], label_resolution_mm=label_resolution,
            position_tolerance_mm=float(motion["confirmation_position_tolerance_mm"]),
        )
        if sonoanatomy_models:
            scan_spec = protocol["sonoanatomy_scan"]
            freshness_retries = int(scan_spec.get("maximum_freshness_retries", 0))
            if freshness_retries not in {0, 1}:
                raise ValueError("semantic freshness retry count must be zero or one")
            semantic_move_attempts = int(
                scan_spec.get("maximum_semantic_move_attempts", 1)
            )
            if semantic_move_attempts not in {1, 2}:
                raise ValueError("semantic move attempts must be one or two")
            if freshness_retries:
                validate_confirmation_motion(
                    [scan_spec["freshness_retry_dither_offset_index"]],
                    label_resolution_mm=label_resolution,
                    position_tolerance_mm=float(motion["fine_position_tolerance_mm"]),
                )
            validate_confirmation_motion(
                scan_spec["confirmation_dither_offsets"],
                label_resolution_mm=label_resolution,
                position_tolerance_mm=float(
                    motion["confirmation_position_tolerance_mm"]
                ),
            )
            competing_pair_offsets = scan_spec.get(
                "competing_pair_rescan_offsets_index", []
            )
            maximum_competing_pair_anchors = int(
                scan_spec.get("maximum_competing_pair_rescan_anchors", 0)
            )
            if float(scan_spec["minimum_score_margin"]) > 0.0:
                if not competing_pair_offsets or maximum_competing_pair_anchors < 1:
                    raise ValueError(
                        "competing-pair rescanning requires offsets and an anchor budget"
                    )
                for offset in competing_pair_offsets:
                    delta = np.asarray(offset, dtype=np.float64)
                    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
                        raise ValueError(
                            "competing-pair rescan offsets must be finite 3-vectors"
                        )
                    position_mm = float(
                        np.linalg.norm(delta[:2]) * label_resolution
                    )
                    orientation_deg = math.degrees(abs(float(delta[2])))
                    if (
                        position_mm
                        <= float(motion["fine_position_tolerance_mm"])
                        and orientation_deg
                        <= float(motion["fine_orientation_tolerance_deg"])
                    ):
                        raise ValueError(
                            "competing-pair rescan offset must exceed a fine-motion tolerance"
                        )
        target_to_level = {
            "L4_RIGHT_TP_SURFACE_CORE": "L4",
            "L5_RIGHT_TP_SURFACE_CORE": "L5",
        }
        for target_index, target_name in enumerate(protocol["targets"]):
            handoff = group_a_target_handoff_pose(
                actor, session_id=args.stage1_session_id,
                target_level=target_to_level[target_name],
                home_z_index=float(protocol["geometry"]["home_z_index"]),
                label_resolution_mm=label_resolution,
                stage2_angle_rad=float(protocol["geometry"]["stage2_angle_rad"]),
            )
            if target_index > 0 and bool(
                protocol.get("level_transition_handoff_settle", False)
            ):
                move(
                    handoff,
                    f"{target_name}:level_transition_handoff_settle",
                    float(motion["coarse_position_tolerance_mm"]),
                    float(motion["coarse_orientation_tolerance_deg"]),
                )
            views: list[TargetView] = []
            for pose in offset_poses(
                handoff,
                serpentine_offsets(
                    search["coarse_x_offsets_index"], search["coarse_z_offsets_index"],
                    search["coarse_angle_offsets_rad"],
                ),
            ):
                views.append(acquire(
                    pose, target_name, f"{target_name}:coarse",
                    float(motion["coarse_position_tolerance_mm"]),
                    float(motion["coarse_orientation_tolerance_deg"]),
                ))
            coarse_views = list(views)
            if any(
                not row.transition_valid
                or not row.image_valid
                or not row.producer_advanced
                for row in coarse_views
            ):
                raise RuntimeError("coarse search produced invalid or stale evidence")
            if len({row.frame_sha256 for row in coarse_views}) != len(coarse_views):
                raise RuntimeError("coarse search produced duplicate images")
            best = max(views, key=lambda row: (row.probability, -row.frame_sequence))
            best_pose = np.asarray(best.pose_index, dtype=np.float64)
            for pose in offset_poses(best_pose, search["fine_offsets"]):
                views.append(acquire(
                    pose, target_name, f"{target_name}:fine",
                    float(motion["fine_position_tolerance_mm"]),
                    float(motion["fine_orientation_tolerance_deg"]),
                ))
            best = max(views, key=lambda row: (row.probability, -row.frame_sequence))
            best_pose = np.asarray(best.pose_index, dtype=np.float64)
            for pose in offset_poses(best_pose, search["micro_offsets"]):
                views.append(acquire(
                    pose, target_name, f"{target_name}:micro",
                    float(motion["micro_position_tolerance_mm"]),
                    float(motion["micro_orientation_tolerance_deg"]),
                ))
            model = models[target_name]
            rule = protocol["decision"]
            decision = select_target_view(
                views, target=target_name,
                probability_threshold=float(model["threshold"]),
                required_support=int(rule["required_supporting_viewpoints"]),
                neighborhood_translation_mm=float(rule["maximum_neighbor_translation_mm"]),
                neighborhood_angle_deg=float(rule["maximum_neighbor_angle_deg"]),
                minimum_view_translation_mm=float(rule["minimum_view_translation_mm"]),
                minimum_view_angle_deg=float(rule["minimum_view_angle_deg"]),
                label_resolution_mm=label_resolution,
            )
            confirmations: list[TargetView] = []
            return_errors: list[dict[str, float]] = []
            if decision.status == "ACCEPT" and decision.selected is not None:
                selected_pose = np.asarray(decision.selected.pose_index, dtype=np.float64)
                for index, offset in enumerate(search["confirmation_dither_offsets"]):
                    dither = selected_pose + np.asarray(offset, dtype=np.float64)
                    move(
                        dither, f"{target_name}:confirmation_dither_{index + 1}",
                        float(motion["confirmation_position_tolerance_mm"]),
                        float(motion["confirmation_orientation_tolerance_deg"]),
                    )
                    confirmed = acquire(
                        selected_pose, target_name, f"{target_name}:fresh_confirmation_{index + 1}",
                        float(motion["confirmation_position_tolerance_mm"]),
                        float(motion["confirmation_orientation_tolerance_deg"]),
                    )
                    confirmations.append(confirmed)
                    realized = np.asarray(confirmed.pose_index)
                    return_errors.append(
                        {
                            "position_mm": float(np.linalg.norm(realized[:2] - selected_pose[:2]) * label_resolution),
                            "orientation_deg": math.degrees(abs(float(wrap_to_pi(realized[2] - selected_pose[2])))),
                        }
                    )
            required_fresh_count = int(search["required_fresh_confirmation_count"])
            minimum_actor_positive_count = int(
                search.get(
                    "minimum_actor_positive_fresh_confirmations",
                    required_fresh_count,
                )
            )
            fresh_pass = bool(
                actor_confirmation_requirement_met(
                    confirmations,
                    probability_threshold=float(model["threshold"]),
                    required_count=required_fresh_count,
                    minimum_positive_count=minimum_actor_positive_count,
                )
                and decision.selected is not None
                and all(row.frame_sha256 != decision.selected.frame_sha256 for row in confirmations)
            )
            semantic_tp_confirmations: list[dict[str, Any]] = []
            semantic_tp_crosscheck_pass = False
            if sonoanatomy_models and decision.selected is not None:
                level = target_to_level[target_name]
                handoff_semantic_model = sonoanatomy_models[level]
                confirmation_probabilities = segmentation_probabilities(
                    handoff_semantic_model["model"],
                    np.asarray([images[row.frame_sequence] for row in confirmations]),
                )
                for row, probability in zip(
                    confirmations, confirmation_probabilities
                ):
                    presence = predicted_presence(
                        probability, handoff_semantic_model["thresholds"]
                    )
                    semantic_tp_confirmations.append(
                        {
                            "frame_sequence": row.frame_sequence,
                            "frame_sha256": row.frame_sha256,
                            "tp_predicted": bool(presence["tp_predicted"]),
                            "tp_predicted_pixels": int(
                                presence["tp_predicted_pixels"]
                            ),
                        }
                    )
                semantic_tp_crosscheck_pass = tp_handoff_crosscheck_requirement_met(
                    [
                        {
                            "frame_sequence": row.frame_sequence,
                            "frame_sha256": row.frame_sha256,
                            "probability": row.probability,
                            "image_valid": row.image_valid,
                            "producer_advanced": row.producer_advanced,
                        }
                        for row in confirmations
                    ],
                    semantic_tp_confirmations,
                    required_count=required_fresh_count,
                    legacy_probability_threshold=float(model["threshold"]),
                    selected_frame_sha256=decision.selected.frame_sha256,
                )
            tp_handoff_recovery_executed = False
            if (
                sonoanatomy_models
                and decision.selected is not None
                and not (fresh_pass or semantic_tp_crosscheck_pass)
                and "landmark_map_confirmation" in protocol
            ):
                tp_handoff_recovery_executed = True
                level = target_to_level[target_name]
                handoff_semantic_model = sonoanatomy_models[level]
                selected_pose = np.asarray(
                    decision.selected.pose_index, dtype=np.float64
                )
                recovery_rows = []
                for index, offset in enumerate(
                    protocol["landmark_map_confirmation"].get(
                        "tp_local_search_offsets_index", []
                    )
                ):
                    requested_pose = selected_pose + np.asarray(
                        offset, dtype=np.float64
                    )
                    recovered = acquire(
                        requested_pose,
                        target_name,
                        f"{target_name}:tp_handoff_recovery_{index + 1}",
                        float(motion["fine_position_tolerance_mm"]),
                        float(motion["fine_orientation_tolerance_deg"]),
                    )
                    probability = segmentation_probabilities(
                        handoff_semantic_model["model"],
                        np.asarray([images[recovered.frame_sequence]]),
                    )[0]
                    presence = predicted_presence(
                        probability, handoff_semantic_model["thresholds"]
                    )
                    public_rows[recovered.frame_sequence].update(
                        {
                            "scan_state": "TP_HANDOFF_RECOVERY",
                            "trajectory_id": (
                                f"{target_name}:tp_handoff_recovery"
                            ),
                            "tp_predicted": bool(presence["tp_predicted"]),
                            "sap_predicted": bool(presence["sap_predicted"]),
                            "other_bone_predicted": bool(
                                presence["other_bone_predicted"]
                            ),
                            "tp_predicted_pixels": int(
                                presence["tp_predicted_pixels"]
                            ),
                            "sap_predicted_pixels": int(
                                presence["sap_predicted_pixels"]
                            ),
                            "other_bone_predicted_pixels": int(
                                presence["other_bone_predicted_pixels"]
                            ),
                        }
                    )
                    assert_no_privileged_fields(
                        public_rows[recovered.frame_sequence]
                    )
                    realized = np.asarray(recovered.pose_index, dtype=np.float64)
                    recovery_rows.append(
                        {
                            "view": recovered,
                            "semantic": {
                                "frame_sequence": recovered.frame_sequence,
                                "frame_sha256": recovered.frame_sha256,
                                "tp_predicted": bool(
                                    presence["tp_predicted"]
                                ),
                                "tp_predicted_pixels": int(
                                    presence["tp_predicted_pixels"]
                                ),
                            },
                            "return_error": {
                                "position_mm": float(
                                    np.linalg.norm(
                                        realized[:2] - requested_pose[:2]
                                    )
                                    * label_resolution
                                ),
                                "orientation_deg": math.degrees(
                                    abs(
                                        float(
                                            wrap_to_pi(
                                                realized[2]
                                                - requested_pose[2]
                                            )
                                        )
                                    )
                                ),
                            },
                        }
                    )
                semantic_positive_recovery = [
                    row
                    for row in recovery_rows
                    if row["semantic"]["tp_predicted"]
                    and row["view"].image_valid
                    and row["view"].producer_advanced
                    and row["view"].frame_sha256
                    != decision.selected.frame_sha256
                ]
                selected_recovery = sorted(
                    semantic_positive_recovery,
                    key=lambda row: (
                        -row["view"].probability,
                        -row["semantic"]["tp_predicted_pixels"],
                        row["view"].frame_sequence,
                    ),
                )[:required_fresh_count]
                if len(selected_recovery) == required_fresh_count:
                    confirmations = [row["view"] for row in selected_recovery]
                    semantic_tp_confirmations = [
                        row["semantic"] for row in selected_recovery
                    ]
                    return_errors = [
                        row["return_error"] for row in selected_recovery
                    ]
                    fresh_pass = actor_confirmation_requirement_met(
                        confirmations,
                        probability_threshold=float(model["threshold"]),
                        required_count=required_fresh_count,
                        minimum_positive_count=minimum_actor_positive_count,
                    )
                    semantic_tp_crosscheck_pass = (
                        tp_handoff_crosscheck_requirement_met(
                            [
                                {
                                    "frame_sequence": row.frame_sequence,
                                    "frame_sha256": row.frame_sha256,
                                    "probability": row.probability,
                                    "image_valid": row.image_valid,
                                    "producer_advanced": row.producer_advanced,
                                }
                                for row in confirmations
                            ],
                            semantic_tp_confirmations,
                            required_count=required_fresh_count,
                            legacy_probability_threshold=float(
                                model["threshold"]
                            ),
                            selected_frame_sha256=(
                                decision.selected.frame_sha256
                            ),
                        )
                    )
            semantic_scan_handoff_pass = bool(
                fresh_pass or semantic_tp_crosscheck_pass
            )
            target_result = {
                "target_name": target_name,
                "stage1_level": target_to_level[target_name],
                "stage1_handoff_pose_index": handoff.tolist(),
                "search_frame_count": len(views),
                "search_distinct_hash_count": len({row.frame_sha256 for row in views}),
                "decision": decision.status if fresh_pass else "ABSTAIN",
                "decision_reason": decision.reason if fresh_pass else "FRESH_CONFIRMATION_FAILED",
                "selected_frame_sequence": None if decision.selected is None else decision.selected.frame_sequence,
                "selected_probability": None if decision.selected is None else decision.selected.probability,
                "supporting_frame_sequences": list(decision.supporting_frame_sequences),
                "supporting_viewpoint_count": decision.distinct_viewpoint_count,
                "fresh_confirmation_frame_sequences": [row.frame_sequence for row in confirmations],
                "fresh_confirmation_probabilities": [row.probability for row in confirmations],
                "minimum_actor_positive_fresh_confirmations": minimum_actor_positive_count,
                "fresh_confirmation_passed": fresh_pass,
                "semantic_tp_handoff_crosscheck": {
                    "confirmations": semantic_tp_confirmations,
                    "required_semantic_positive_count": required_fresh_count,
                    "passed": semantic_tp_crosscheck_pass,
                    "unlocks_scanning_only": True,
                },
                "semantic_scan_handoff_passed": semantic_scan_handoff_pass,
                "tp_handoff_recovery_executed": (
                    tp_handoff_recovery_executed
                ),
                "return_errors": return_errors,
            }
            if (
                sonoanatomy_models
                and semantic_scan_handoff_pass
                and decision.selected is not None
            ):
                level = target_to_level[target_name]
                semantic_model = sonoanatomy_models[level]
                level_actor_anchors = {}
                dedicated_tp_confirmation_rows: list[dict[str, Any]] = []
                if shared_landmark_map is not None:
                    selected_public = public_rows[
                        int(decision.selected.frame_sequence)
                    ]
                    selected_rotation = np.asarray(
                        selected_public[
                            "actor_base_to_ee_rotation_columns"
                        ],
                        dtype=np.float64,
                    )
                    actor_coordinates = calibrated_image_coordinates_mm(
                        end_effector_position_m=selected_public[
                            "actor_base_to_ee_position_m"
                        ],
                        rotation_columns=selected_rotation,
                        width=images[decision.selected.frame_sequence].shape[1],
                        depth=images[decision.selected.frame_sequence].shape[0],
                        pixel_resolution_mm=float(
                            protocol["geometry"][
                                "ultrasound_pixel_resolution_mm"
                            ]
                        ),
                        image_origin_height_mm=133.0,
                        home_up_axis=selected_rotation[:, 2],
                    )
                    anchor_row = int(
                        np.clip(
                            round(float(decision.selected.proposal_row)),
                            0,
                            actor_coordinates.shape[0] - 1,
                        )
                    )
                    anchor_column = int(
                        np.clip(
                            round(float(decision.selected.proposal_column)),
                            0,
                            actor_coordinates.shape[1] - 1,
                        )
                    )
                    tp_identity = f"RIGHT_{level}_TP"
                    tp_anchor = actor_coordinates[anchor_row, anchor_column]
                    if not unet_only:
                        level_actor_anchors[tp_identity] = tp_anchor
                        shared_landmark_map.set_actor_anchor(
                            tp_identity, tp_anchor
                        )
                    dedicated_views = [decision.selected, *confirmations]
                    dedicated_probabilities = segmentation_probabilities(
                        semantic_model["model"],
                        np.asarray(
                            [images[row.frame_sequence] for row in dedicated_views]
                        ),
                    )
                    dedicated_observations: list[
                        tuple[TargetView, SemanticObservation]
                    ] = []
                    for dedicated_view, probability in zip(
                        dedicated_views, dedicated_probabilities
                    ):
                        if (
                            not dedicated_view.image_valid
                            or not dedicated_view.producer_advanced
                            or dedicated_view.probability
                            < float(model["threshold"])
                        ):
                            continue
                        public = public_rows[dedicated_view.frame_sequence]
                        presence = predicted_presence(
                            probability, semantic_model["thresholds"]
                        )
                        public.update(
                            {
                                "scan_state": "DEDICATED_TP_EVIDENCE",
                                "trajectory_id": (
                                    f"{target_name}:dedicated_tp_evidence"
                                ),
                                "tp_predicted": bool(
                                    presence["tp_predicted"]
                                ),
                                "sap_predicted": bool(
                                    presence["sap_predicted"]
                                ),
                                "other_bone_predicted": bool(
                                    presence["other_bone_predicted"]
                                ),
                                "tp_predicted_pixels": int(
                                    presence["tp_predicted_pixels"]
                                ),
                                "sap_predicted_pixels": int(
                                    presence["sap_predicted_pixels"]
                                ),
                                "other_bone_predicted_pixels": int(
                                    presence["other_bone_predicted_pixels"]
                                ),
                            }
                        )
                        assert_no_privileged_fields(public)
                        candidates = [
                            row
                            for row in semantic_observations(
                                probability,
                                thresholds=semantic_model["thresholds"],
                                end_effector_position_m=public[
                                    "actor_base_to_ee_position_m"
                                ],
                                rotation_columns=np.asarray(
                                    public["actor_base_to_ee_rotation_columns"],
                                    dtype=np.float64,
                                ),
                                frame_sequence=dedicated_view.frame_sequence,
                                frame_sha256=dedicated_view.frame_sha256,
                            )
                            if row.semantic_class == "TP"
                        ]
                        if not candidates:
                            continue
                        proposal_row = float(public["proposal_row"])
                        proposal_column = float(public["proposal_column"])
                        selected_observation = min(
                            candidates,
                            key=lambda row: (
                                math.hypot(
                                    row.row - proposal_row,
                                    row.column - proposal_column,
                                ),
                                -row.pixel_count,
                                -row.confidence,
                            ),
                        )
                        dedicated_observations.append(
                            (dedicated_view, selected_observation)
                        )
                    source = next(
                        (
                            row
                            for row in dedicated_observations
                            if row[0].frame_sequence
                            == decision.selected.frame_sequence
                        ),
                        None,
                    )
                    if source is not None:
                        shared_landmark_map.add_observations(
                            level, [source[1]]
                        )
                        fresh_dedicated = [
                            row
                            for row in dedicated_observations
                            if row[0].frame_sequence
                            != decision.selected.frame_sequence
                        ]
                        if fresh_dedicated:
                            shared_landmark_map.add_reacquisition_observations(
                                level,
                                "TP",
                                reference_position_mm=source[1].coordinate_mm,
                                reference_frame_hashes=[
                                    source[1].frame_sha256
                                ],
                                observations=[row[1] for row in fresh_dedicated],
                                association_radius_mm=(
                                    landmark_map_config.association_radius_mm
                                    if unet_only
                                    else landmark_map_config.maximum_tp_anchor_distance_mm
                                ),
                            )
                        selected_pose = np.asarray(
                            decision.selected.pose_index, dtype=np.float64
                        )
                        return_error_by_sequence = {
                            row.frame_sequence: error
                            for row, error in zip(confirmations, return_errors)
                        }
                        for fresh_view, observation in fresh_dedicated:
                            error = return_error_by_sequence[
                                fresh_view.frame_sequence
                            ]
                            dedicated_tp_confirmation_rows.append(
                                {
                                    "identity": tp_identity,
                                    "confirmation_kind": (
                                        "DEDICATED_TP_FRESH_CONFIRMATION"
                                    ),
                                    "source_frame_sequence": (
                                        decision.selected.frame_sequence
                                    ),
                                    "source_frame_sha256": (
                                        decision.selected.frame_sha256
                                    ),
                                    "source_executed_pose_index": (
                                        selected_pose.tolist()
                                    ),
                                    "requested_pose_index": (
                                        selected_pose.tolist()
                                    ),
                                    "frame_sequence": fresh_view.frame_sequence,
                                    "frame_sha256": fresh_view.frame_sha256,
                                    "image_valid": fresh_view.image_valid,
                                    "producer_advanced": (
                                        fresh_view.producer_advanced
                                    ),
                                    "executed_pose_index": list(
                                        fresh_view.pose_index
                                    ),
                                    "return_position_error_mm": float(
                                        error["position_mm"]
                                    ),
                                    "return_orientation_error_deg": float(
                                        error["orientation_deg"]
                                    ),
                                    "predicted": True,
                                    "semantic_observations": [
                                        asdict(observation)
                                    ],
                                    "fresh_relative_to_source": (
                                        fresh_view.frame_sha256
                                        != decision.selected.frame_sha256
                                    ),
                                }
                            )
                scan_spec = protocol["sonoanatomy_scan"]
                scan_config = ScanConfig(
                    longitudinal_angle_rad=float(scan_spec["longitudinal_angle_rad"]),
                    transverse_angle_rad=float(scan_spec["transverse_angle_rad"]),
                    translation_step_index=float(scan_spec["translation_step_index"]),
                    reference_offsets_index=tuple(
                        float(value) for value in scan_spec["reference_offsets_index"]
                    ),
                    maximum_lateral_steps=int(scan_spec["maximum_lateral_steps"]),
                    maximum_medial_steps=int(scan_spec["maximum_medial_steps"]),
                    transverse_offsets_index=tuple(
                        float(value) for value in scan_spec["transverse_offsets_index"]
                    ),
                )
                controller = ClinicianSweepController(
                    decision.selected.pose_index,
                    scan_config,
                )
                semantic_frames: list[dict[str, Any]] = []
                observations_by_trajectory: dict[str, list[SemanticObservation]] = {}

                def acquire_semantic(
                    scan_pose: np.ndarray,
                    *,
                    state_name: str,
                    trajectory_id: str,
                ) -> tuple[TargetView, dict[str, Any] | None]:
                    phase = f"{target_name}:sonoanatomy:{state_name}"
                    stage_orientation_change(scan_pose, phase)
                    maximum_retries = int(
                        scan_spec.get("maximum_freshness_retries", 0)
                    )
                    retry_offset = np.asarray(
                        scan_spec.get(
                            "freshness_retry_dither_offset_index", [0.0, 0.0, 0.0]
                        ),
                        dtype=np.float64,
                    )
                    retry_count = 0

                    def acquire_scan_pose() -> TargetView:
                        maximum_attempts = int(
                            scan_spec.get("maximum_semantic_move_attempts", 1)
                        )
                        for attempt in range(1, maximum_attempts + 1):
                            try:
                                return acquire(
                                    scan_pose,
                                    target_name,
                                    f"{phase}:move_attempt_{attempt}_of_{maximum_attempts}",
                                    float(motion["fine_position_tolerance_mm"]),
                                    float(motion["fine_orientation_tolerance_deg"]),
                                )
                            except TimeoutError:
                                if attempt == maximum_attempts:
                                    raise
                        raise RuntimeError("semantic movement attempt loop is empty")

                    while True:
                        scan_view = acquire_scan_pose()
                        public = public_rows[-1]
                        public.update(
                            {
                                "scan_state": state_name,
                                "trajectory_id": trajectory_id,
                                "freshness_retry_count": retry_count,
                            }
                        )
                        eligible = bool(
                            scan_view.image_valid
                            and scan_view.producer_advanced
                            and public["contact_ok"]
                        )
                        if eligible:
                            break
                        assert_no_privileged_fields(public)
                        retryable_stale_frame = bool(
                            not scan_view.producer_advanced and public["contact_ok"]
                        )
                        if not retryable_stale_frame or retry_count >= maximum_retries:
                            return scan_view, None
                        retry_count += 1
                        dither_pose = scan_pose + retry_offset
                        retry_motion = move(
                            dither_pose,
                            f"{phase}:freshness_retry_dither_{retry_count}",
                            float(motion["fine_position_tolerance_mm"]),
                            float(motion["fine_orientation_tolerance_deg"]),
                        )
                        if not retry_motion:
                            return scan_view, None
                    probability = segmentation_probabilities(
                        semantic_model["model"], np.asarray([images[-1]])
                    )[0]
                    presence = predicted_presence(
                        probability, semantic_model["thresholds"]
                    )
                    public.update(
                        {
                            "tp_predicted": bool(presence["tp_predicted"]),
                            "sap_predicted": bool(presence["sap_predicted"]),
                            "other_bone_predicted": bool(
                                presence["other_bone_predicted"]
                            ),
                            "tp_predicted_pixels": int(
                                presence["tp_predicted_pixels"]
                            ),
                            "sap_predicted_pixels": int(
                                presence["sap_predicted_pixels"]
                            ),
                            "other_bone_predicted_pixels": int(
                                presence["other_bone_predicted_pixels"]
                            ),
                        }
                    )
                    assert_no_privileged_fields(public)
                    frame_observations = semantic_observations(
                        probability,
                        thresholds=semantic_model["thresholds"],
                        end_effector_position_m=public[
                            "actor_base_to_ee_position_m"
                        ],
                        rotation_columns=np.asarray(
                            public["actor_base_to_ee_rotation_columns"],
                            dtype=np.float64,
                        ),
                        frame_sequence=scan_view.frame_sequence,
                        frame_sha256=scan_view.frame_sha256,
                    )
                    observations_by_trajectory.setdefault(trajectory_id, []).extend(
                        frame_observations
                    )
                    semantic_frame = {
                        "frame_sequence": scan_view.frame_sequence,
                        "frame_sha256": scan_view.frame_sha256,
                        "pose_index": list(scan_view.pose_index),
                        "trajectory_id": trajectory_id,
                        "scan_state": state_name,
                        **presence,
                    }
                    semantic_frames.append(semantic_frame)
                    return scan_view, presence

                while controller.state not in {
                    ScanState.SAP_TP_IDENTITY_INFERENCE,
                    ScanState.ABSTAIN,
                }:
                    scan_pose = controller.next_pose()
                    if scan_pose is None:
                        break
                    current_state = controller.state
                    trajectory_id = (
                        f"{target_name}:longitudinal"
                        if current_state
                        in {
                            ScanState.ACQUIRE_REFERENCE_TP_REGION,
                            ScanState.LONGITUDINAL_LATERAL_SURVEY,
                            ScanState.REVERSE_MEDIAL_SURVEY,
                        }
                        else f"{target_name}:transverse"
                    )
                    scan_view, presence = acquire_semantic(
                        scan_pose,
                        state_name=current_state.value,
                        trajectory_id=trajectory_id,
                    )
                    if presence is None:
                        presence = {
                            "tp_predicted": False,
                            "sap_predicted": False,
                            "other_bone_predicted": False,
                            "tp_predicted_pixels": 0,
                            "sap_predicted_pixels": 0,
                        }
                    controller.observe(
                        ScanEvidence(
                            frame_sequence=scan_view.frame_sequence,
                            pose_index=scan_view.pose_index,
                            tp_predicted=bool(presence["tp_predicted"]),
                            sap_predicted=bool(presence["sap_predicted"]),
                            other_bone_predicted=bool(
                                presence["other_bone_predicted"]
                            ),
                            tp_predicted_pixels=int(
                                presence["tp_predicted_pixels"]
                            ),
                            sap_predicted_pixels=int(
                                presence["sap_predicted_pixels"]
                            ),
                            image_valid=scan_view.image_valid,
                            contact_ok=bool(public_rows[-1]["contact_ok"]),
                            producer_advanced=scan_view.producer_advanced,
                        )
                    )

                fallback_transverse_executed = False
                if (
                    controller.state == ScanState.ABSTAIN
                    and controller.reason == "REFERENCE_TP_NOT_FOUND"
                ):
                    fallback_transverse_executed = True
                    transverse_trajectory = f"{target_name}:transverse_fallback"
                    tp_anchor = np.asarray(decision.selected.pose_index, dtype=np.float64)
                    for offset in scan_config.transverse_offsets_index:
                        pose = np.asarray(
                            [
                                tp_anchor[0],
                                tp_anchor[1] + offset,
                                scan_config.transverse_angle_rad,
                            ],
                            dtype=np.float64,
                        )
                        acquire_semantic(
                            pose,
                            state_name="TRANSVERSE_TP_HANDOFF_FALLBACK",
                            trajectory_id=transverse_trajectory,
                        )

                tp_candidates = [
                    row for row in semantic_frames if row["tp_predicted"]
                ]
                sap_candidates = [
                    row for row in semantic_frames if row["sap_predicted"]
                ]
                active_trajectory = f"{target_name}:active_rescan"
                seen_rescan_poses: set[tuple[float, ...]] = set()
                if tp_candidates or sap_candidates:
                    strongest = []
                    if tp_candidates:
                        strongest.append(
                            max(
                                tp_candidates,
                                key=lambda row: (
                                    int(row["tp_predicted_pixels"]),
                                    -int(row["frame_sequence"]),
                                ),
                            )
                        )
                    if sap_candidates:
                        strongest.append(
                            max(
                                sap_candidates,
                                key=lambda row: (
                                    int(row["sap_predicted_pixels"]),
                                    -int(row["frame_sequence"]),
                                ),
                            )
                        )
                    unique_anchors = []
                    seen_anchor_poses = set()
                    for row in strongest:
                        key = tuple(
                            round(float(value), 3) for value in row["pose_index"]
                        )
                        if key not in seen_anchor_poses:
                            unique_anchors.append(np.asarray(row["pose_index"], dtype=np.float64))
                            seen_anchor_poses.add(key)
                    for anchor in unique_anchors:
                        for offset in scan_spec["active_rescan_offsets_index"]:
                            pose = anchor + np.asarray(offset, dtype=np.float64)
                            key = tuple(round(float(value), 3) for value in pose)
                            if key in seen_rescan_poses:
                                continue
                            seen_rescan_poses.add(key)
                            acquire_semantic(
                                pose,
                                state_name="ACTIVE_RESCAN",
                                trajectory_id=active_trajectory,
                            )

                multiview_config = MultiviewConfig(
                    association_radius_mm=float(scan_spec["association_radius_mm"]),
                    minimum_supporting_frames_per_structure=int(
                        scan_spec["minimum_supporting_frames_per_structure"]
                    ),
                    minimum_score_margin=float(scan_spec["minimum_score_margin"]),
                )

                def evaluate_trajectories() -> tuple[
                    list[dict[str, Any]],
                    list[tuple[float, str, dict[str, Any]]],
                    dict[str, tuple[Any, Any]],
                ]:
                    decisions = []
                    accepted = []
                    states = {}
                    for trajectory_id, semantic_rows in observations_by_trajectory.items():
                        clusters = cluster_observations(
                            semantic_rows,
                            association_radius_mm=multiview_config.association_radius_mm,
                        )
                        pair = decide_tp_sap_pair(
                            clusters, config=multiview_config
                        )
                        states[trajectory_id] = (clusters, pair)
                        pair_payload = {
                            "status": pair.status,
                            "reason": pair.reason,
                            "selected": (
                                None
                                if pair.selected is None
                                else asdict(pair.selected)
                            ),
                            "hypothesis_count": len(pair.hypotheses),
                        }
                        decisions.append(
                            {
                                "trajectory_id": trajectory_id,
                                "pair_decision": pair_payload,
                                "semantic_observations": [
                                    asdict(row) for row in semantic_rows
                                ],
                                "semantic_clusters": [
                                    asdict(row) for row in clusters
                                ],
                            }
                        )
                        if pair.status == "ACCEPT" and pair.selected is not None:
                            accepted.append(
                                (pair.selected.score, trajectory_id, pair_payload)
                            )
                    return decisions, accepted, states

                (
                    trajectory_decisions,
                    accepted_trajectories,
                    trajectory_states,
                ) = evaluate_trajectories()
                competing_pair_rescan_executed = False
                active_state = trajectory_states.get(active_trajectory)
                if active_state is not None:
                    active_clusters, active_pair = active_state
                    if (
                        active_pair.reason == "COMPETING_PAIR_HYPOTHESES"
                        and len(active_pair.hypotheses) >= 2
                    ):
                        competing_pair_rescan_executed = True
                        by_sequence = {
                            int(row["frame_sequence"]): row
                            for row in semantic_frames
                            if row["trajectory_id"] == active_trajectory
                        }
                        competing_pair_anchors = []
                        for hypothesis in active_pair.hypotheses[:2]:
                            for semantic_class, cluster_index in (
                                ("TP", hypothesis.tp_cluster_index),
                                ("SAP", hypothesis.sap_cluster_index),
                            ):
                                cluster = active_clusters[cluster_index]
                                candidates = [
                                    by_sequence[frame]
                                    for frame in cluster.supporting_frames
                                    if frame in by_sequence
                                ]
                                if not candidates:
                                    continue
                                pixel_key = (
                                    "tp_predicted_pixels"
                                    if semantic_class == "TP"
                                    else "sap_predicted_pixels"
                                )
                                competing_pair_anchors.append(
                                    max(
                                        candidates,
                                        key=lambda row: (
                                            int(row[pixel_key]),
                                            -int(row["frame_sequence"]),
                                        ),
                                    )
                                )
                        unique_competing_pair_anchors = []
                        seen_competing_pair_poses = set()
                        for row in competing_pair_anchors:
                            key = tuple(
                                round(float(value), 3)
                                for value in row["pose_index"]
                            )
                            if key in seen_competing_pair_poses:
                                continue
                            seen_competing_pair_poses.add(key)
                            unique_competing_pair_anchors.append(
                                np.asarray(row["pose_index"], dtype=np.float64)
                            )
                        maximum_anchors = int(
                            scan_spec.get("maximum_competing_pair_rescan_anchors", 4)
                        )
                        for anchor in unique_competing_pair_anchors[:maximum_anchors]:
                            for offset in scan_spec.get(
                                "competing_pair_rescan_offsets_index", []
                            ):
                                pose = anchor + np.asarray(offset, dtype=np.float64)
                                key = tuple(
                                    round(float(value), 3) for value in pose
                                )
                                if key in seen_rescan_poses:
                                    continue
                                seen_rescan_poses.add(key)
                                acquire_semantic(
                                    pose,
                                    state_name="ACTIVE_COMPETING_PAIR_RESCAN",
                                    trajectory_id=active_trajectory,
                                )
                        (
                            trajectory_decisions,
                            accepted_trajectories,
                            trajectory_states,
                        ) = evaluate_trajectories()
                selected_semantic_trajectory = None
                selected_semantic_pair = None
                landmark_trajectory_decision = None
                if (
                    shared_landmark_map is not None
                    and landmark_map_config is not None
                ):
                    landmark_trajectory_decision = select_landmark_trajectory(
                        level,
                        observations_by_trajectory,
                        landmark_map_config,
                        anchors=level_actor_anchors,
                    )
                    selected_semantic_trajectory = (
                        landmark_trajectory_decision.trajectory_id
                    )
                    if selected_semantic_trajectory is not None:
                        selected_state = trajectory_states.get(
                            selected_semantic_trajectory
                        )
                        if selected_state is not None:
                            selected_pair = selected_state[1]
                            selected_semantic_pair = {
                                "status": selected_pair.status,
                                "reason": selected_pair.reason,
                                "selected": (
                                    None
                                    if selected_pair.selected is None
                                    else asdict(selected_pair.selected)
                                ),
                                "hypothesis_count": len(
                                    selected_pair.hypotheses
                                ),
                            }
                elif accepted_trajectories:
                    _, selected_semantic_trajectory, selected_semantic_pair = max(
                        accepted_trajectories,
                        key=lambda row: (row[0], row[1]),
                    )
                pair_view_candidates = [
                    row
                    for row in semantic_frames
                    if shared_landmark_map is None
                    and row["trajectory_id"] == selected_semantic_trajectory
                    and row["tp_predicted"]
                    and row["sap_predicted"]
                ]
                selected_pair_view = (
                    max(
                        pair_view_candidates,
                        key=lambda row: (
                            min(
                                int(row["tp_predicted_pixels"]),
                                int(row["sap_predicted_pixels"]),
                            ),
                            -int(row["frame_sequence"]),
                        ),
                    )
                    if pair_view_candidates
                    else None
                )
                semantic_confirmations = []
                if selected_pair_view is not None:
                    selected_pair_pose = np.asarray(
                        selected_pair_view["pose_index"], dtype=np.float64
                    )
                    for index, offset in enumerate(
                        scan_spec["confirmation_dither_offsets"]
                    ):
                        dither = selected_pair_pose + np.asarray(offset, dtype=np.float64)
                        move(
                            dither,
                            f"{target_name}:sonoanatomy_confirmation_dither_{index + 1}",
                            float(motion["confirmation_position_tolerance_mm"]),
                            float(motion["confirmation_orientation_tolerance_deg"]),
                        )
                        confirmed = acquire(
                            selected_pair_pose,
                            target_name,
                            f"{target_name}:sonoanatomy_fresh_confirmation_{index + 1}",
                            float(motion["confirmation_position_tolerance_mm"]),
                            float(motion["confirmation_orientation_tolerance_deg"]),
                        )
                        probability = segmentation_probabilities(
                            semantic_model["model"], np.asarray([images[-1]])
                        )[0]
                        presence = predicted_presence(
                            probability, semantic_model["thresholds"]
                        )
                        public_rows[-1].update(
                            {
                                "scan_state": "FRESH_ULTRASOUND_CONFIRMATION",
                                "trajectory_id": f"{target_name}:confirmation",
                                "tp_predicted": bool(presence["tp_predicted"]),
                                "sap_predicted": bool(presence["sap_predicted"]),
                                "other_bone_predicted": bool(
                                    presence["other_bone_predicted"]
                                ),
                                "tp_predicted_pixels": int(
                                    presence["tp_predicted_pixels"]
                                ),
                                "sap_predicted_pixels": int(
                                    presence["sap_predicted_pixels"]
                                ),
                                "other_bone_predicted_pixels": int(
                                    presence["other_bone_predicted_pixels"]
                                ),
                            }
                        )
                        assert_no_privileged_fields(public_rows[-1])
                        realized = np.asarray(confirmed.pose_index, dtype=np.float64)
                        semantic_confirmations.append(
                            {
                                "frame_sequence": confirmed.frame_sequence,
                                "frame_sha256": confirmed.frame_sha256,
                                "image_valid": confirmed.image_valid,
                                "producer_advanced": confirmed.producer_advanced,
                                "return_position_error_mm": float(
                                    np.linalg.norm(realized[:2] - selected_pair_pose[:2])
                                    * label_resolution
                                ),
                                "return_orientation_error_deg": math.degrees(
                                    abs(
                                        float(
                                            wrap_to_pi(
                                                realized[2] - selected_pair_pose[2]
                                            )
                                        )
                                    )
                                ),
                                **presence,
                            }
                        )
                semantic_fresh_pass = bool(
                    pair_confirmation_requirement_met(
                        semantic_confirmations,
                        required_count=len(scan_spec["confirmation_dither_offsets"]),
                        minimum_positive_count=int(
                            scan_spec[
                                "minimum_actor_positive_fresh_confirmations"
                            ]
                        ),
                    )
                    and all(
                        row["frame_sha256"] != selected_pair_view["frame_sha256"]
                        for row in semantic_confirmations
                    )
                    and selected_semantic_pair is not None
                )
                target_result["sonoanatomy"] = {
                    "checkpoint_sha256": semantic_model["sha256"],
                    "controller_transition_evidence": controller.transition_evidence(),
                    "fallback_transverse_executed": fallback_transverse_executed,
                    "competing_pair_rescan_executed": (
                        competing_pair_rescan_executed
                    ),
                    "trajectory_decisions": trajectory_decisions,
                    "selected_trajectory_id": selected_semantic_trajectory,
                    "selected_pair_decision": selected_semantic_pair,
                    "selected_pair_frame_sequence": (
                        None
                        if selected_pair_view is None
                        else selected_pair_view["frame_sequence"]
                    ),
                    "scan_frame_count": len(semantic_frames),
                    "fresh_confirmations": semantic_confirmations,
                    "minimum_actor_positive_fresh_confirmations": int(
                        scan_spec["minimum_actor_positive_fresh_confirmations"]
                    ),
                    "fresh_confirmation_passed": semantic_fresh_pass,
                }
                if not semantic_fresh_pass:
                    target_result["decision"] = "ABSTAIN"
                    target_result["decision_reason"] = (
                        "SAP_TP_FRESH_CONFIRMATION_FAILED"
                    )
                elif semantic_tp_crosscheck_pass:
                    target_result["decision"] = "ACCEPT"
                    target_result["decision_reason"] = (
                        "SUPPORTED_SAP_TP_TARGET_VIEW_AFTER_TP_CROSSCHECK"
                    )
                if (
                    shared_landmark_map is not None
                    and landmark_map_config is not None
                ):
                    selected_rows = observations_by_trajectory.get(
                        selected_semantic_trajectory or "", []
                    )
                    if selected_rows:
                        shared_landmark_map.add_observations(
                            level,
                            [
                                row
                                for row in selected_rows
                                if row.semantic_class == "SAP"
                            ],
                        )
                    map_before_confirmation = shared_landmark_map.snapshot()
                    confirmation_spec = protocol["landmark_map_confirmation"]
                    maximum_poses = int(
                        confirmation_spec["poses_per_landmark"]
                    )
                    minimum_positive = int(
                        confirmation_spec[
                            "minimum_positive_fresh_returns_per_landmark"
                        ]
                    )
                    if not 1 <= minimum_positive <= maximum_poses:
                        raise ValueError(
                            "invalid independent landmark confirmation count"
                        )
                    semantic_frames_by_sequence = {
                        int(row["frame_sequence"]): row
                        for row in semantic_frames
                    }
                    landmark_confirmations = list(
                        dedicated_tp_confirmation_rows
                    )
                    tp_refinement_records: list[dict[str, Any]] = []
                    pending_reacquisition_observations = {}
                    for semantic_class in ("TP", "SAP"):
                        identity = f"RIGHT_{level}_{semantic_class}"
                        hypothesis = map_before_confirmation["landmarks"][
                            identity
                        ]
                        hypothesis_position = np.asarray(
                            hypothesis["position_mm"], dtype=np.float64
                        )
                        support_rows = list(
                            hypothesis["supporting_observations"]
                        )[: (1 if semantic_class == "SAP" else maximum_poses)]
                        for support_index, support in enumerate(support_rows):
                            source_sequence = int(support["frame_sequence"])
                            source_frame = semantic_frames_by_sequence.get(
                                source_sequence
                            )
                            if source_frame is None:
                                if (
                                    semantic_class != "TP"
                                    or source_sequence
                                    != decision.selected.frame_sequence
                                ):
                                    continue
                                anchor_pose = np.asarray(
                                    decision.selected.pose_index,
                                    dtype=np.float64,
                                )
                            else:
                                anchor_pose = np.asarray(
                                    source_frame["pose_index"], dtype=np.float64
                                )
                            confirmation_trajectory = (
                                f"{target_name}:landmark_confirmation"
                            )
                            confirmation_views = []
                            if semantic_class == "SAP":
                                pose_plan = [
                                    (
                                        f"LOCAL_SEARCH_{index + 1}",
                                        anchor_pose
                                        + np.asarray(offset, dtype=np.float64),
                                    )
                                    for index, offset in enumerate(
                                        confirmation_spec.get(
                                            "sap_local_search_offsets_index",
                                            [[0.0, 0.0, 0.0]],
                                        )
                                    )
                                ]
                            else:
                                tp_offsets = tp_reacquisition_offsets(
                                    confirmation_spec,
                                    level=level,
                                    supporting_view_count=len(
                                        hypothesis["supporting_viewpoint_ids"]
                                    ),
                                    minimum_supporting_views=(
                                        landmark_map_config.minimum_supporting_views
                                    ),
                                )
                                if tp_offsets:
                                    pose_plan = [
                                        (
                                            f"LOCAL_SEARCH_{index + 1}",
                                            anchor_pose
                                            + np.asarray(offset, dtype=np.float64),
                                        )
                                        for index, offset in enumerate(tp_offsets)
                                    ]
                                else:
                                    pose_plan = [("PHYSICAL_RETURN", anchor_pose)]
                            def acquire_landmark_confirmation(
                                confirmation_kind: str,
                                confirmation_pose: np.ndarray,
                                *,
                                refinement_center: dict[str, Any] | None = None,
                            ) -> tuple[dict[str, Any], list[SemanticObservation]]:
                                confirmed, presence = acquire_semantic(
                                    confirmation_pose,
                                    state_name=(
                                        f"FRESH_{semantic_class}_{confirmation_kind}"
                                    ),
                                    trajectory_id=confirmation_trajectory,
                                )
                                frame_observations = [
                                    row
                                    for row in observations_by_trajectory.get(
                                        confirmation_trajectory, []
                                    )
                                    if row.frame_sequence
                                    == confirmed.frame_sequence
                                    and row.semantic_class == semantic_class
                                    and np.linalg.norm(
                                        np.asarray(
                                            row.coordinate_mm,
                                            dtype=np.float64,
                                        )
                                        - hypothesis_position
                                    )
                                    <= (
                                        landmark_map_config.association_radius_mm
                                        if unet_only
                                        else (
                                            landmark_map_config.maximum_tp_anchor_distance_mm
                                            if semantic_class == "TP"
                                            else landmark_map_config.association_radius_mm
                                        )
                                    )
                                ]
                                if semantic_class == "TP":
                                    public = public_rows[
                                        confirmed.frame_sequence
                                    ]
                                    proposal_row = public.get("proposal_row")
                                    proposal_column = public.get(
                                        "proposal_column"
                                    )
                                    if (
                                        presence is None
                                        or not presence["tp_predicted"]
                                        or float(public["target_probability"])
                                        < float(model["threshold"])
                                        or proposal_row is None
                                        or proposal_column is None
                                        or not frame_observations
                                    ):
                                        typed_observations = []
                                    else:
                                        typed_observations = [
                                            min(
                                                frame_observations,
                                                key=lambda row: (
                                                    math.hypot(
                                                        row.row
                                                        - float(proposal_row),
                                                        row.column
                                                        - float(proposal_column),
                                                    ),
                                                    -row.pixel_count,
                                                    -row.confidence,
                                                ),
                                            )
                                        ]
                                else:
                                    typed_observations = frame_observations
                                predicted = bool(
                                    presence is not None
                                    and presence[
                                        f"{semantic_class.lower()}_predicted"
                                    ]
                                    and typed_observations
                                )
                                result = {
                                    "identity": identity,
                                    "confirmation_kind": confirmation_kind,
                                    "source_frame_sequence": source_sequence,
                                    "source_frame_sha256": support[
                                        "frame_sha256"
                                    ],
                                    "source_executed_pose_index": (
                                        anchor_pose.tolist()
                                    ),
                                    "requested_pose_index": (
                                        confirmation_pose.tolist()
                                    ),
                                    "frame_sequence": confirmed.frame_sequence,
                                    "frame_sha256": confirmed.frame_sha256,
                                    "image_valid": confirmed.image_valid,
                                    "producer_advanced": (
                                        confirmed.producer_advanced
                                    ),
                                    "executed_pose_index": list(
                                        confirmed.pose_index
                                    ),
                                    "return_position_error_mm": float(
                                        np.linalg.norm(
                                            np.asarray(
                                                confirmed.pose_index[:2]
                                            )
                                            - confirmation_pose[:2]
                                        )
                                        * label_resolution
                                    ),
                                    "return_orientation_error_deg": (
                                        math.degrees(
                                            abs(
                                                float(
                                                    wrap_to_pi(
                                                        confirmed.pose_index[2]
                                                        - confirmation_pose[2]
                                                    )
                                                )
                                            )
                                        )
                                    ),
                                    "predicted": predicted,
                                    "semantic_observations": [
                                        asdict(row) for row in typed_observations
                                    ],
                                    "fresh_relative_to_source": (
                                        confirmed.frame_sha256
                                        != support["frame_sha256"]
                                    ),
                                }
                                if refinement_center is not None:
                                    result.update(
                                        {
                                            "refinement_center_frame_sequence": int(
                                                refinement_center["frame_sequence"]
                                            ),
                                            "refinement_center_executed_pose_index": list(
                                                refinement_center[
                                                    "executed_pose_index"
                                                ]
                                            ),
                                            "refinement_recursive": False,
                                        }
                                    )
                                return result, typed_observations

                            for confirmation_kind, confirmation_pose in pose_plan:
                                confirmation, typed_observations = (
                                    acquire_landmark_confirmation(
                                        confirmation_kind,
                                        confirmation_pose,
                                    )
                                )
                                if confirmation["predicted"]:
                                    pending_reacquisition_observations.setdefault(
                                        identity, []
                                    ).extend(typed_observations)
                                confirmation_views.append(confirmation)
                            if semantic_class == "TP":
                                refinement_offsets = confirmation_spec.get(
                                    "tp_refinement_offsets_index", []
                                )
                                refinement_required = (
                                    False
                                    if unet_only
                                    else tp_refinement_is_required(
                                    confirmation_views,
                                    minimum_predicted_count=int(
                                        confirmation_spec.get(
                                            "tp_refinement_minimum_predicted_count",
                                            required_fresh_count,
                                        )
                                    ),
                                    large_interface_pixel_count=int(
                                        confirmation_spec.get(
                                            "tp_refinement_large_interface_pixels",
                                            2**31 - 1,
                                        )
                                    ),
                                    )
                                )
                                refinement_center = select_tp_refinement_anchor(
                                    confirmation_views,
                                    public_rows,
                                )
                                tp_refinement_records.append(
                                    {
                                        "source_frame_sequence": source_sequence,
                                        "required": refinement_required,
                                        "initial_predicted_count": sum(
                                            bool(row["predicted"])
                                            for row in confirmation_views
                                        ),
                                        "initial_maximum_interface_pixels": max(
                                            (
                                                int(observation["pixel_count"])
                                                for row in confirmation_views
                                                for observation in row[
                                                    "semantic_observations"
                                                ]
                                            ),
                                            default=0,
                                        ),
                                        "center_frame_sequence": (
                                            None
                                            if refinement_center is None
                                            else int(
                                                refinement_center[
                                                    "frame_sequence"
                                                ]
                                            )
                                        ),
                                        "executed": bool(
                                            refinement_required
                                            and refinement_offsets
                                            and refinement_center is not None
                                        ),
                                    }
                                )
                                if (
                                    refinement_required
                                    and refinement_offsets
                                    and refinement_center is not None
                                ):
                                    refinement_pose = np.asarray(
                                        refinement_center["executed_pose_index"],
                                        dtype=np.float64,
                                    )
                                    for index, offset in enumerate(
                                        refinement_offsets
                                    ):
                                        offset_array = np.asarray(
                                            offset, dtype=np.float64
                                        )
                                        confirmation, typed_observations = (
                                            acquire_landmark_confirmation(
                                                f"REFINEMENT_{index + 1}",
                                                refinement_pose
                                                + offset_array,
                                                refinement_center=refinement_center,
                                            )
                                        )
                                        map_update_eligible = bool(
                                            np.any(np.abs(offset_array) > 1e-12)
                                            or bool(
                                                confirmation_spec.get(
                                                    "tp_refinement_zero_offset_updates_map",
                                                    False,
                                                )
                                            )
                                        )
                                        confirmation["map_update_eligible"] = (
                                            map_update_eligible
                                        )
                                        if (
                                            confirmation["predicted"]
                                            and map_update_eligible
                                        ):
                                            pending_reacquisition_observations.setdefault(
                                                identity, []
                                            ).extend(typed_observations)
                                        confirmation_views.append(confirmation)
                            landmark_confirmations.extend(confirmation_views)
                        pending = pending_reacquisition_observations.get(
                            identity, []
                        )
                        if pending:
                            shared_landmark_map.add_reacquisition_observations(
                                level,
                                semantic_class,
                                reference_position_mm=hypothesis["position_mm"],
                                reference_frame_hashes=hypothesis[
                                    "supporting_frame_hashes"
                                ],
                                observations=pending,
                                association_radius_mm=(
                                    landmark_map_config.association_radius_mm
                                    if unet_only
                                    else (
                                        landmark_map_config.maximum_tp_anchor_distance_mm
                                        if semantic_class == "TP"
                                        else landmark_map_config.association_radius_mm
                                    )
                                ),
                                preserve_frozen_surface_extent=unet_only,
                            )
                    selected_fresh_frames_by_identity = {
                        identity: set(
                            shared_landmark_map.selected_reacquisition_frame_sequences(
                                identity
                            )
                        )
                        for identity in (
                            f"RIGHT_{level}_TP",
                            f"RIGHT_{level}_SAP",
                        )
                    }
                    positive_by_identity = {
                        identity: len(
                            {
                                int(row["frame_sequence"])
                                for row in landmark_confirmations
                                if row["identity"] == identity
                                and int(row["frame_sequence"])
                                in selected_fresh_frames_by_identity[identity]
                                and bool(row["predicted"])
                                and bool(row["image_valid"])
                                and bool(row["producer_advanced"])
                                and bool(row["fresh_relative_to_source"])
                            }
                        )
                        for identity in (
                            f"RIGHT_{level}_TP",
                            f"RIGHT_{level}_SAP",
                        )
                    }
                    map_after_confirmation = shared_landmark_map.snapshot()
                    independent_fresh_pass = bool(
                        shared_landmark_map.level_confirmed(level)
                        and all(
                            count >= minimum_positive
                            for count in positive_by_identity.values()
                        )
                    )
                    final_tp_reacquisition = (
                        select_tp_reacquisition_confirmations(
                            [
                                row
                                for row in landmark_confirmations
                                if row["identity"] == f"RIGHT_{level}_TP"
                            ],
                            public_rows,
                            required_count=required_fresh_count,
                            probability_threshold=float(model["threshold"]),
                        )
                    )
                    if len(final_tp_reacquisition) == required_fresh_count:
                        target_result["fresh_confirmation_frame_sequences"] = [
                            int(row["frame_sequence"])
                            for row in final_tp_reacquisition
                        ]
                        target_result["fresh_confirmation_probabilities"] = [
                            float(
                                public_rows[int(row["frame_sequence"])][
                                    "target_probability"
                                ]
                            )
                            for row in final_tp_reacquisition
                        ]
                        target_result["fresh_confirmation_passed"] = True
                        target_result["return_errors"] = [
                            {
                                "position_mm": float(
                                    row["return_position_error_mm"]
                                ),
                                "orientation_deg": float(
                                    row["return_orientation_error_deg"]
                                ),
                            }
                            for row in final_tp_reacquisition
                        ]
                    target_result["sonoanatomy"].update(
                        {
                            "confirmation_mode": (
                                "INDEPENDENT_LANDMARK_FRESH_RETURN"
                            ),
                            "landmark_trajectory_decision": (
                                None
                                if landmark_trajectory_decision is None
                                else asdict(landmark_trajectory_decision)
                            ),
                            "incremental_landmark_map_before_confirmation": (
                                map_before_confirmation
                            ),
                            "incremental_landmark_map": map_after_confirmation,
                            "fresh_landmark_confirmations": (
                                landmark_confirmations
                            ),
                            "positive_fresh_returns_by_identity": (
                                positive_by_identity
                            ),
                            "minimum_positive_fresh_returns_per_landmark": (
                                minimum_positive
                            ),
                            "fresh_confirmation_passed": (
                                independent_fresh_pass
                            ),
                            "final_tp_confirmation_from_landmark_reacquisition": (
                                len(final_tp_reacquisition)
                                == required_fresh_count
                            ),
                            "tp_refinement_records": tp_refinement_records,
                        }
                    )
                    if independent_fresh_pass:
                        target_result["decision"] = "ACCEPT"
                        target_result["decision_reason"] = (
                            "INDEPENDENT_LANDMARKS_CONFIRMED_IN_SHARED_MAP"
                        )
                    else:
                        target_result["decision"] = "ABSTAIN"
                        target_result["decision_reason"] = (
                            "INCREMENTAL_LANDMARK_MAP_CONFIRMATION_FAILED"
                        )
            elif sonoanatomy_models:
                target_result["sonoanatomy"] = {
                    "scan_frame_count": 0,
                    "fresh_confirmation_passed": False,
                    "reason": "TP_HANDOFF_NOT_CONFIRMED",
                }
                target_result["decision"] = "ABSTAIN"
                target_result["decision_reason"] = "TP_HANDOFF_NOT_CONFIRMED"
            assert_no_privileged_fields(target_result)
            target_results.append(target_result)

        public_dir.mkdir(parents=True, exist_ok=True)
        privileged_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(public_dir / "ultrasound.npz", ultrasound=np.stack(images))
        np.savez_compressed(privileged_dir / "raw_labels.npz", raw_labels=np.stack(raw_labels))
        np.savez_compressed(
            privileged_dir / "pixel_to_volume_coordinates.npz",
            human_img_coords=np.stack(coordinate_maps),
            native_local_image_coordinates_m=np.asarray(local_coordinates, dtype=np.float32),
        )
        write_jsonl(public_dir / "frames.jsonl", public_rows)
        write_jsonl(privileged_dir / "full_pose_replay.jsonl", pose_rows)
        write_jsonl(privileged_dir / "actions.jsonl", action_rows)
        actor_result = {
            "schema": "sonogym-s0010-mbb-poc-actor-result-v1",
            "protocol_sha256": protocol_hash,
            "phase1_actor_result_sha256": actor_hash,
            "session_id": str(session_spec["session_id"]),
            "stage1_session_id": args.stage1_session_id,
            "seed": int(session_spec["seed"]),
            "outcome": "TARGET_VIEWS_CONFIRMED" if all(row["decision"] == "ACCEPT" for row in target_results) else "ABSTAIN",
            "targets": target_results,
            "incremental_landmark_map": (
                None
                if shared_landmark_map is None
                else shared_landmark_map.snapshot()
            ),
            "frame_count": len(public_rows),
            "distinct_frame_hash_count": len({row["frame_sha256"] for row in public_rows}),
            "action_step_count": total_actions,
            "model_training_performed": False,
            "geometry_used_by_actor": False,
            "perception_mode": active_perception_mode,
            "extra_trees_loaded": not unet_only,
            "tp_anchor_dependency": not unet_only,
        }
        assert_no_privileged_fields(actor_result)
        write_json(result_dir / "actor_result.json", actor_result)
        actor_seal = sha256(result_dir / "actor_result.json")
        (result_dir / "actor_result.json.sha256").write_text(f"{actor_seal}  actor_result.json\n")
        summary.update(
            status="COMPLETED",
            actor_outcome=actor_result["outcome"],
            actor_result_sha256=actor_seal,
            frame_count=len(public_rows),
            action_step_count=total_actions,
            physical_grid=grid,
            robot_video_frame_count=0 if video is None else video.frames,
        )
        return 0
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        return 1
    finally:
        if video is not None:
            video.capture(force=True)
            video.close()
        if env is not None:
            env.close()
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json(output / "summary.json", summary)
        # Preserve a failed run's causal evidence for diagnosis.  These files are
        # explicitly partial and can never be mistaken for a sealed actor result.
        if public_rows and not (public_dir / "frames.jsonl").exists():
            public_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(public_dir / "frames_partial.jsonl", public_rows)
            np.savez_compressed(public_dir / "ultrasound_partial.npz", ultrasound=np.stack(images))
        if pose_rows and not (privileged_dir / "full_pose_replay.jsonl").exists():
            privileged_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(privileged_dir / "full_pose_replay_partial.jsonl", pose_rows)
            np.savez_compressed(
                privileged_dir / "pixel_to_volume_coordinates_partial.npz",
                human_img_coords=np.stack(coordinate_maps),
                native_local_image_coordinates_m=np.asarray(local_coordinates, dtype=np.float32),
            )


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)

