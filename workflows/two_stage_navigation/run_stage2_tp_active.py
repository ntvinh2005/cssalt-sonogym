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
    select_target_view,
)
from spinal_surgery.evaluation.two_stage_navigation.sonoanatomy_active_scan import (  # noqa: E402
    ClinicianSweepController,
    ScanConfig,
    ScanEvidence,
    ScanState,
    staged_orientation_targets,
)
from spinal_surgery.evaluation.two_stage_navigation.sonoanatomy_multiview import (  # noqa: E402
    MultiviewConfig,
    SemanticObservation,
    cluster_observations,
    decide_tp_sap_pair,
    semantic_observations,
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

        load_feature_pickle_support()
        model_specs = protocol["models"]
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
        if models["L4_RIGHT_TP_SURFACE_CORE"]["patch_spec"] != models["L5_RIGHT_TP_SURFACE_CORE"]["patch_spec"]:
            raise RuntimeError("L4 and L5 preprocessing contracts differ")
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
            trajectory = move_proportional_target(
                env, target, target,
                maximum_steps=int(motion["maximum_steps_per_move"]),
                maximum_position_error_mm=position_tolerance,
                maximum_orientation_error_deg=angle_tolerance,
                ratio=tuple(motion["controller_ratio"]),
                allow_angle_ceiling_saturation=False,
                step_callback=None if video is None else video.capture,
            )
            total_actions += len(trajectory)
            if total_actions > int(motion["maximum_total_action_steps"]):
                raise RuntimeError("action budget exceeded")
            for row in trajectory:
                row["phase"] = phase
            action_rows.extend(trajectory)
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
            for index, intermediate in enumerate(intermediates, start=1):
                move(
                    intermediate,
                    f"{phase}:orientation_stage_{index}_of_{len(intermediates)}",
                    float(motion["coarse_position_tolerance_mm"]),
                    float(motion["coarse_orientation_tolerance_deg"]),
                )

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
            model = models[target_name]
            probability, proposal, proposal_count = score_target_image(
                image, target=target_name, estimator=model["estimator"],
                class_id=int(model["class_id"]), patch_spec=model["patch_spec"],
                column_stride_pixels=int(protocol["perception"]["proposal_column_stride_pixels"]),
            )
            realized = current_realized_pose(unwrapped).copy()
            sequence = len(public_rows)
            frame_hash = ultrasound_sha256(image)
            actor_position = tensor_row(snapshot["actor_base_to_ee_pos"])
            actor_quaternion = tensor_row(snapshot["actor_base_to_ee_quat"])
            actor_rotation = rotation_columns_from_quaternion_wxyz(actor_quaternion)
            view = TargetView(
                target=target_name,
                frame_sequence=sequence,
                producer_sequence=producer,
                frame_sha256=frame_hash,
                pose_index=tuple(float(value) for value in realized),
                probability=probability,
                proposal_row=None if proposal is None else float(proposal.row),
                proposal_column=None if proposal is None else float(proposal.column),
                proposal_track_id=None if proposal is None else proposal.track_id,
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
        target_to_level = {
            "L4_RIGHT_TP_SURFACE_CORE": "L4",
            "L5_RIGHT_TP_SURFACE_CORE": "L5",
        }
        for target_name in protocol["targets"]:
            handoff = group_a_target_handoff_pose(
                actor, session_id=args.stage1_session_id,
                target_level=target_to_level[target_name],
                home_z_index=float(protocol["geometry"]["home_z_index"]),
                label_resolution_mm=label_resolution,
                stage2_angle_rad=float(protocol["geometry"]["stage2_angle_rad"]),
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
                    outbound = move(
                        dither, f"{target_name}:confirmation_dither_{index + 1}",
                        float(motion["confirmation_position_tolerance_mm"]),
                        float(motion["confirmation_orientation_tolerance_deg"]),
                    )
                    if not outbound:
                        raise RuntimeError("confirmation dither produced no action")
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
                "return_errors": return_errors,
            }
            if sonoanatomy_models and fresh_pass and decision.selected is not None:
                level = target_to_level[target_name]
                semantic_model = sonoanatomy_models[level]
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
                ) -> tuple[TargetView, dict[str, Any]]:
                    phase = f"{target_name}:sonoanatomy:{state_name}"
                    stage_orientation_change(scan_pose, phase)
                    scan_view = acquire(
                        scan_pose,
                        target_name,
                        phase,
                        float(motion["fine_position_tolerance_mm"]),
                        float(motion["fine_orientation_tolerance_deg"]),
                    )
                    probability = segmentation_probabilities(
                        semantic_model["model"], np.asarray([images[-1]])
                    )[0]
                    presence = predicted_presence(
                        probability, semantic_model["thresholds"]
                    )
                    public = public_rows[-1]
                    public.update(
                        {
                            "scan_state": state_name,
                            "trajectory_id": trajectory_id,
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

                tp_candidates = [
                    row for row in semantic_frames if row["tp_predicted"]
                ]
                sap_candidates = [
                    row for row in semantic_frames if row["sap_predicted"]
                ]
                if tp_candidates and sap_candidates:
                    strongest = [
                        max(
                            tp_candidates,
                            key=lambda row: (
                                int(row["tp_predicted_pixels"]),
                                -int(row["frame_sequence"]),
                            ),
                        ),
                        max(
                            sap_candidates,
                            key=lambda row: (
                                int(row["sap_predicted_pixels"]),
                                -int(row["frame_sequence"]),
                            ),
                        ),
                    ]
                    unique_anchors = []
                    seen_anchor_poses = set()
                    for row in strongest:
                        key = tuple(
                            round(float(value), 3) for value in row["pose_index"]
                        )
                        if key not in seen_anchor_poses:
                            unique_anchors.append(np.asarray(row["pose_index"], dtype=np.float64))
                            seen_anchor_poses.add(key)
                    active_trajectory = f"{target_name}:active_rescan"
                    seen_rescan_poses = set()
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

                trajectory_decisions = []
                accepted_trajectories = []
                multiview_config = MultiviewConfig(
                    association_radius_mm=float(scan_spec["association_radius_mm"]),
                    minimum_supporting_frames_per_structure=int(
                        scan_spec["minimum_supporting_frames_per_structure"]
                    ),
                    minimum_score_margin=float(scan_spec["minimum_score_margin"]),
                )
                for trajectory_id, semantic_rows in observations_by_trajectory.items():
                    clusters = cluster_observations(
                        semantic_rows,
                        association_radius_mm=multiview_config.association_radius_mm,
                    )
                    pair = decide_tp_sap_pair(clusters, config=multiview_config)
                    pair_payload = {
                        "status": pair.status,
                        "reason": pair.reason,
                        "selected": (
                            None if pair.selected is None else asdict(pair.selected)
                        ),
                        "hypothesis_count": len(pair.hypotheses),
                    }
                    trajectory_decisions.append(
                        {
                            "trajectory_id": trajectory_id,
                            "pair_decision": pair_payload,
                            "semantic_observations": [
                                asdict(row) for row in semantic_rows
                            ],
                            "semantic_clusters": [asdict(row) for row in clusters],
                        }
                    )
                    if pair.status == "ACCEPT" and pair.selected is not None:
                        accepted_trajectories.append(
                            (pair.selected.score, trajectory_id, pair_payload)
                        )
                selected_semantic_trajectory = None
                selected_semantic_pair = None
                if accepted_trajectories:
                    _, selected_semantic_trajectory, selected_semantic_pair = max(
                        accepted_trajectories,
                        key=lambda row: (row[0], row[1]),
                    )
                pair_view_candidates = [
                    row
                    for row in semantic_frames
                    if row["trajectory_id"] == selected_semantic_trajectory
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
                    len(semantic_confirmations)
                    == len(scan_spec["confirmation_dither_offsets"])
                    and len(
                        {row["frame_sha256"] for row in semantic_confirmations}
                    )
                    == len(semantic_confirmations)
                    and all(
                        row["frame_sha256"] != selected_pair_view["frame_sha256"]
                        for row in semantic_confirmations
                    )
                    and all(
                        row["image_valid"]
                        and row["producer_advanced"]
                        and row["tp_predicted"]
                        and row["sap_predicted"]
                        for row in semantic_confirmations
                    )
                    and selected_semantic_pair is not None
                )
                target_result["sonoanatomy"] = {
                    "checkpoint_sha256": semantic_model["sha256"],
                    "controller_transition_evidence": controller.transition_evidence(),
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
                    "fresh_confirmation_passed": semantic_fresh_pass,
                }
                if not semantic_fresh_pass:
                    target_result["decision"] = "ABSTAIN"
                    target_result["decision_reason"] = (
                        "SAP_TP_FRESH_CONFIRMATION_FAILED"
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
            "frame_count": len(public_rows),
            "distinct_frame_hash_count": len({row["frame_sha256"] for row in public_rows}),
            "action_step_count": total_actions,
            "model_training_performed": False,
            "geometry_used_by_actor": False,
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

