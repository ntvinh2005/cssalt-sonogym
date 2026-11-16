#!/usr/bin/env python3
"""Run one sealed continuous Stage 2 active-acquisition pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--protocol", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--assets-data-dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("MPLBACKEND", "Agg")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import cv2  # noqa: E402
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
from spinal_surgery.evaluation.two_stage_navigation.coordinates import (  # noqa: E402
    wrap_to_pi,
)
from spinal_surgery.evaluation.two_stage_navigation.stage2_active_pilot import (  # noqa: E402
    ACTIVE_PILOT_PROTOCOL_IDS,
    offset_poses,
    phase1_handoff_pose,
    relative_pose_from_start,
    serpentine_offsets,
    transition_is_valid,
    validate_confirmation_motion,
    validate_episode_horizon,
    validate_physical_grid,
)
from spinal_surgery.evaluation.two_stage_navigation.stage2_models import (  # noqa: E402
    ScoredView,
    score_ultrasound,
    select_best_view,
)
from spinal_surgery.evaluation.two_stage_navigation.stage2_temporal_decision import (  # noqa: E402
    DecisionState,
    ReplayObservation,
    TemporalDecisionConfig,
    fresh_confirmation_passes_distinct,
    replay_decision,
)

from stage2_sim import (  # noqa: E402
    clean_ultrasound,
    current_realized_pose,
    move_proportional_target,
    raw_label_slice,
    surface_proxy_ok,
    tensor_row,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def seal(path: Path) -> str:
    digest = file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def load_sealed_yaml(path: Path) -> tuple[dict[str, Any], str]:
    expected = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    observed = file_sha256(path)
    if observed != expected:
        raise RuntimeError("protocol seal mismatch")
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") not in ACTIVE_PILOT_PROTOCOL_IDS:
        raise RuntimeError("unexpected protocol")
    return protocol, observed


class RobotVideoRecorder:
    """Stream the Isaac rendered camera to a compact, explicitly non-actor MP4."""

    def __init__(self, env: Any, output: Path, config: dict[str, Any]) -> None:
        self.env = env
        self.output = output
        self.fps = int(config["fps"])
        self.action_stride = int(config["action_stride"])
        self.writer: cv2.VideoWriter | None = None
        self.action_count = 0
        self.frame_count = 0
        self.phase = "initial"
        if self.fps <= 0 or self.action_stride <= 0:
            raise ValueError("robot-video fps and action stride must be positive")

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def capture(self, _status: dict[str, Any] | None = None, *, force: bool = False) -> None:
        if not force:
            self.action_count += 1
            if self.action_count % self.action_stride:
                return
        frame = np.asarray(self.env.render())
        if frame.ndim != 3 or frame.shape[2] not in {3, 4}:
            raise RuntimeError(f"unexpected Isaac render shape: {frame.shape}")
        rgb = frame[:, :, :3].astype(np.uint8, copy=False)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 42), (10, 10, 10), -1)
        cv2.putText(
            bgr,
            f"Isaac Sim rendered view | {self.phase} | physical action {self.action_count}",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        if self.writer is None:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(
                str(self.output),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self.fps),
                (bgr.shape[1], bgr.shape[0]),
            )
            if not self.writer.isOpened():
                raise RuntimeError("OpenCV could not create the Isaac camera MP4")
        self.writer.write(bgr)
        self.frame_count += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def main() -> int:
    protocol_path = Path(args.protocol).resolve()
    protocol, protocol_hash = load_sealed_yaml(protocol_path)
    output = Path(args.output_dir).resolve()
    public_dir = output / "public"
    privileged_dir = output / "privileged"
    result_dir = output / "results"
    started = time.monotonic()
    summary: dict[str, Any] = {
        "schema": "sonogym-stage2-active-pilot-summary-v1",
        "status": "RUNTIME_FAIL",
        "protocol_sha256": protocol_hash,
        "seed": int(protocol["execution"]["seed"]),
    }
    env = None
    robot_video: RobotVideoRecorder | None = None
    action_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    privileged_rows: list[dict[str, Any]] = []
    images: list[np.ndarray] = []
    observations: list[ReplayObservation] = []
    scored_views: list[ScoredView] = []
    absolute_pose_by_frame: dict[int, np.ndarray] = {}
    try:
        handoff_spec = protocol["phase1_handoff"]
        handoff_path = Path(handoff_spec["actor_result"])
        if file_sha256(handoff_path) != handoff_spec["actor_result_sha256"]:
            raise RuntimeError("Phase 1 actor result hash mismatch")
        phase1_actor = json.loads(handoff_path.read_text(encoding="utf-8"))
        handoff = phase1_handoff_pose(
            phase1_actor,
            session_id=str(handoff_spec["session_id"]),
            home_z_index=float(handoff_spec["home_z_index"]),
            label_resolution_mm=float(protocol["geometry"]["label_resolution_mm"]),
            stage2_angle_rad=float(protocol["geometry"]["stage2_handoff_angle_rad"]),
        )

        model_spec = protocol["model"]
        model_path = Path(model_spec["bundle"])
        if file_sha256(model_path) != model_spec["bundle_sha256"]:
            raise RuntimeError("model bundle hash mismatch")
        bundle = joblib.load(model_path)
        classifier = bundle["classifier"]
        regressor = bundle["regressor"]
        threshold = float(model_spec["validity_probability_threshold"])

        rule = protocol["decision"]
        decision_config = TemporalDecisionConfig(
            validity_probability_threshold=threshold,
            maximum_neighbor_translation_mm=float(rule["maximum_neighbor_translation_mm"]),
            maximum_neighbor_angle_deg=float(rule["maximum_neighbor_angle_deg"]),
            maximum_quality_drop=float(rule["maximum_quality_drop"]),
            required_distinct_hash_observations=int(rule["required_distinct_hash_observations"]),
            minimum_distinct_view_translation_mm=float(rule["minimum_distinct_view_translation_mm"]),
            minimum_distinct_view_angle_deg=float(rule["minimum_distinct_view_angle_deg"]),
        )

        visualization = protocol.get("visualization", {})
        record_robot_video = bool(visualization.get("record_robot_video", False))
        env_cfg = parse_env_cfg(
            protocol["scope"]["task"], device=args.device or "cuda:0", num_envs=1
        )
        env_cfg.seed = int(protocol["execution"]["seed"])
        motion = protocol["motion"]
        if "episode_length_s" in motion:
            declared_episode_length_s = float(motion["episode_length_s"])
            validate_episode_horizon(
                int(motion["maximum_total_action_steps"]),
                declared_episode_length_s,
            )
            env_cfg.episode_length_s = declared_episode_length_s
        env = gym.make(
            protocol["scope"]["task"],
            cfg=env_cfg,
            render_mode="rgb_array" if record_robot_video else None,
        )
        observation, _ = env.reset(seed=int(protocol["execution"]["seed"]))
        unwrapped = env.unwrapped
        if tuple(observation["policy"].shape) != (1, 1, 200, 150):
            raise RuntimeError("unexpected policy observation shape")
        if unwrapped.US_slicer.sim_mode != "conv" or unwrapped.scene.num_envs != 1:
            raise RuntimeError("pilot requires one conv SonoGym environment")
        if record_robot_video:
            unwrapped.sim.set_camera_view(
                eye=tuple(float(value) for value in visualization["camera_eye_world_m"]),
                target=tuple(float(value) for value in visualization["camera_target_world_m"]),
            )
            robot_video = RobotVideoRecorder(
                env,
                output / "video" / "sonogym-robot-probe-patient.mp4",
                visualization,
            )
            robot_video.capture(force=True)

        raw_path = (
            Path(args.assets_data_dir)
            / "HumanModels"
            / "selected_dataset_stl"
            / "s0010"
            / "combined_label_map.nii.gz"
        )
        raw_volume = np.asarray(nib.load(raw_path).dataobj, dtype=np.uint8)
        evaluator_goal = tensor_row(unwrapped.goal_cmd_pose).copy()
        search = protocol["search"]
        label_resolution = float(protocol["geometry"]["label_resolution_mm"])
        grid_spec = validate_physical_grid(
            search["coarse_x_offsets_index"],
            search["coarse_z_offsets_index"],
            search["coarse_angle_offsets_rad"],
            label_resolution_mm=label_resolution,
            position_tolerance_mm=float(motion["coarse_position_tolerance_mm"]),
            orientation_tolerance_deg=float(motion["coarse_orientation_tolerance_deg"]),
        )
        confirmation_position_tolerance = float(
            motion.get("confirmation_position_tolerance_mm", motion["position_tolerance_mm"])
        )
        strict_confirmation_motion = protocol["protocol_id"] in {
            "sonogym-stage2-active-acquisition-pilot-v2",
            "sonogym-stage2-active-acquisition-pilot-v3",
            "sonogym-stage2-active-acquisition-grid-demo-v4",
            "sonogym-stage2-active-acquisition-grid-demo-v4r1",
            "sonogym-stage2-active-acquisition-grid-demo-v5",
        }
        if strict_confirmation_motion:
            confirmation_displacements = validate_confirmation_motion(
                search["confirmation_dither_offsets"],
                label_resolution_mm=label_resolution,
                position_tolerance_mm=confirmation_position_tolerance,
            )
        else:
            confirmation_displacements = tuple(
                float(np.linalg.norm(np.asarray(offset, dtype=np.float64)[:2]) * label_resolution)
                for offset in search["confirmation_dither_offsets"]
            )
        maximum_total_steps = int(motion["maximum_total_action_steps"])
        total_action_steps = 0
        previous_producer = int(unwrapped.num_step)

        def move_only(
            target: np.ndarray,
            phase: str,
            *,
            position_tolerance_mm: float | None = None,
            orientation_tolerance_deg: float | None = None,
        ) -> list[dict[str, Any]]:
            nonlocal total_action_steps
            if robot_video is not None:
                robot_video.set_phase(phase)
            controller_target = target.copy()
            if target[2] >= float(motion["ceiling_assisted_minimum_angle_rad"]):
                controller_target[2] = float(motion["controller_target_angle_rad"])
            trajectory = move_proportional_target(
                env,
                controller_target,
                target,
                maximum_steps=int(
                    motion[
                        "maximum_initial_positioning_steps"
                        if phase == "phase1_handoff"
                        else "maximum_steps_per_move"
                    ]
                ),
                maximum_position_error_mm=float(
                    motion["position_tolerance_mm"]
                    if position_tolerance_mm is None
                    else position_tolerance_mm
                ),
                maximum_orientation_error_deg=float(
                    motion["orientation_tolerance_deg"]
                    if orientation_tolerance_deg is None
                    else orientation_tolerance_deg
                ),
                ratio=tuple(float(value) for value in motion["controller_ratio"]),
                allow_angle_ceiling_saturation=bool(motion["allow_angle_ceiling_saturation"]),
                step_callback=None if robot_video is None else robot_video.capture,
            )
            total_action_steps += len(trajectory)
            if total_action_steps > maximum_total_steps:
                raise RuntimeError("active pilot action budget exceeded")
            for row in trajectory:
                row["phase"] = phase
            action_rows.extend(trajectory)
            return trajectory

        def acquire(
            target: np.ndarray,
            phase: str,
            *,
            position_tolerance_mm: float | None = None,
            orientation_tolerance_deg: float | None = None,
        ) -> tuple[ReplayObservation, ScoredView]:
            nonlocal previous_producer
            trajectory = move_only(
                target,
                phase,
                position_tolerance_mm=position_tolerance_mm,
                orientation_tolerance_deg=orientation_tolerance_deg,
            )
            transition_valid = transition_is_valid(trajectory)
            image = clean_ultrasound(unwrapped)
            frame_hash = ultrasound_sha256(image)
            realized = current_realized_pose(unwrapped).copy()
            producer = int(unwrapped.num_step)
            producer_advanced = producer > previous_producer
            previous_producer = producer
            image_valid = bool(
                transition_valid
                and producer_advanced
                and surface_proxy_ok(unwrapped)
                and not bool(unwrapped.US_slicer.no_collide[0].item())
                and float(np.std(image)) > 1e-8
            )
            probability, quality = score_ultrasound(image, classifier, regressor)
            frame_sequence = len(public_rows)
            relative_pose = relative_pose_from_start(realized, handoff, label_resolution)
            requested_relative_pose = relative_pose_from_start(target, handoff, label_resolution)
            replay_row = ReplayObservation(
                session_id=str(protocol["execution"]["session_id"]),
                trajectory_id=str(protocol["execution"]["session_id"]),
                frame_sequence=frame_sequence,
                trajectory_frame_sequence=frame_sequence,
                producer_sequence=producer,
                frame_sha256=frame_hash,
                relative_pose=relative_pose,
                validity_probability=probability,
                quality=quality,
                transition_valid=transition_valid,
                image_valid=image_valid,
                producer_advanced=producer_advanced,
            )
            scored = ScoredView(
                candidate_id=f"frame-{frame_sequence}",
                relative_pose=relative_pose,
                validity_probability=probability,
                quality=quality,
                transition_valid=transition_valid,
                image_valid=image_valid,
                frame_sha256=frame_hash,
            )
            public = {
                "schema": "sonogym-stage2-active-pilot-actor-frame-v1",
                "session_id": replay_row.session_id,
                "frame_sequence": frame_sequence,
                "producer_sequence": producer,
                "frame_sha256": frame_hash,
                "phase": phase,
                "relative_pose": list(relative_pose),
                "requested_relative_pose": list(requested_relative_pose),
                "validity_probability": probability,
                "quality_prediction": quality,
                "transition_valid": transition_valid,
                "image_valid": image_valid,
                "producer_advanced": producer_advanced,
                "action_step_count": len(trajectory),
            }
            assert_no_privileged_fields(public)
            raw = raw_label_slice(unwrapped, raw_volume)
            position_error = float(np.linalg.norm(realized[:2] - evaluator_goal[:2]) * label_resolution)
            orientation_error = abs(float(wrap_to_pi(realized[2] - evaluator_goal[2]))) * 180.0 / math.pi
            privileged_rows.append(
                {
                    "frame_sequence": frame_sequence,
                    "frame_sha256": frame_hash,
                    "phase": phase,
                    "realized_pose_index": realized.tolist(),
                    "goal_cmd_pose": evaluator_goal.tolist(),
                    "position_error_mm": position_error,
                    "orientation_error_deg": orientation_error,
                    "l4_visible_fraction": float(np.count_nonzero(raw == 28) / raw.size),
                }
            )
            public_rows.append(public)
            images.append(image.copy())
            observations.append(replay_row)
            scored_views.append(scored)
            absolute_pose_by_frame[frame_sequence] = realized
            return replay_row, scored

        move_only(handoff, "phase1_handoff")
        coarse = offset_poses(
            handoff,
            serpentine_offsets(
                search["coarse_x_offsets_index"],
                search["coarse_z_offsets_index"],
                search["coarse_angle_offsets_rad"],
            ),
        )
        for target in coarse:
            acquire(
                target,
                "coarse",
                position_tolerance_mm=float(motion["coarse_position_tolerance_mm"]),
                orientation_tolerance_deg=float(motion["coarse_orientation_tolerance_deg"]),
            )
        coarse_rows = [row for row in public_rows if row["phase"] == "coarse"]
        if len(coarse_rows) != int(search["expected_coarse_observations"]):
            raise RuntimeError("coarse grid observation count mismatch")
        if any(
            not row["producer_advanced"]
            or not row["image_valid"]
            or int(row["action_step_count"]) <= 0
            for row in coarse_rows
        ):
            raise RuntimeError("every coarse grid pose must execute and produce a fresh valid frame")
        if len({row["frame_sha256"] for row in coarse_rows}) != len(coarse_rows):
            raise RuntimeError("coarse grid did not produce one distinct image per pose")

        best = select_best_view(scored_views, threshold)
        if best is not None:
            best_sequence = int(best.candidate_id.split("-")[-1])
            fine_center = absolute_pose_by_frame[best_sequence]
            for target in offset_poses(fine_center, search["fine_offsets"]):
                acquire(
                    target,
                    "fine",
                    position_tolerance_mm=float(motion["fine_position_tolerance_mm"]),
                    orientation_tolerance_deg=float(motion["fine_orientation_tolerance_deg"]),
                )
            best = select_best_view(scored_views, threshold)
        if best is not None:
            best_sequence = int(best.candidate_id.split("-")[-1])
            micro_center = absolute_pose_by_frame[best_sequence]
            for target in offset_poses(micro_center, search["micro_offsets"]):
                acquire(
                    target,
                    "micro",
                    position_tolerance_mm=float(motion["micro_position_tolerance_mm"]),
                    orientation_tolerance_deg=float(motion["micro_orientation_tolerance_deg"]),
                )

        decision = replay_decision(
            observations,
            policy="candidate_d_multi_view",
            config=decision_config,
        )
        confirmations: list[ReplayObservation] = []
        return_error_mm = None
        return_error_deg = None
        fresh_pass = False
        confirmation_moves: list[dict[str, Any]] = []
        if decision.state is DecisionState.ACCEPT and decision.selected is not None:
            selected_pose = absolute_pose_by_frame[decision.selected.frame_sequence]
            for index, dither in enumerate(search["confirmation_dither_offsets"]):
                dither_pose = selected_pose + np.asarray(dither, dtype=np.float64)
                outbound = move_only(
                    dither_pose,
                    f"confirmation_dither_{index + 1}",
                    position_tolerance_mm=confirmation_position_tolerance,
                )
                if strict_confirmation_motion and not outbound:
                    raise RuntimeError("confirmation dither produced no physical action")
                confirmed, _ = acquire(
                    selected_pose,
                    f"fresh_confirmation_{index + 1}",
                    position_tolerance_mm=confirmation_position_tolerance,
                )
                if strict_confirmation_motion and (
                    not confirmed.transition_valid or not confirmed.producer_advanced
                ):
                    raise RuntimeError("confirmation return did not produce a fresh valid frame")
                confirmations.append(confirmed)
                confirmation_moves.append(
                    {
                        "confirmation_index": index + 1,
                        "declared_dither_displacement_mm": confirmation_displacements[index],
                        "outbound_action_steps": len(outbound),
                        "return_action_steps": int(public_rows[-1]["action_step_count"]),
                        "confirmation_frame_sequence": confirmed.frame_sequence,
                        "confirmation_producer_sequence": confirmed.producer_sequence,
                    }
                )
            returned = current_realized_pose(unwrapped)
            return_error_mm = float(np.linalg.norm(returned[:2] - selected_pose[:2]) * label_resolution)
            return_error_deg = abs(float(wrap_to_pi(returned[2] - selected_pose[2]))) * 180.0 / math.pi
            fresh_pass = fresh_confirmation_passes_distinct(
                decision.selected,
                confirmations,
                config=decision_config,
                required_count=int(search["required_fresh_confirmation_count"]),
            )

        actor_outcome = (
            "ACCEPT"
            if decision.state is DecisionState.ACCEPT and fresh_pass
            else "ABSTAIN"
        )
        selected = decision.selected
        actor_result = {
            "schema": "sonogym-stage2-active-pilot-actor-result-v1",
            "protocol_sha256": protocol_hash,
            "phase1_actor_result_sha256": handoff_spec["actor_result_sha256"],
            "session_id": protocol["execution"]["session_id"],
            "outcome": actor_outcome,
            "search_frame_count": len(observations) - len(confirmations),
            "total_frame_count": len(observations),
            "total_action_steps": total_action_steps,
            "distinct_frame_hash_count": len({row.frame_sha256 for row in observations}),
            "selected_frame_sequence": None if selected is None else selected.frame_sequence,
            "selected_frame_sha256": None if selected is None else selected.frame_sha256,
            "selected_relative_pose": None if selected is None else list(selected.relative_pose),
            "selected_validity_probability": None if selected is None else selected.validity_probability,
            "selected_quality_prediction": None if selected is None else selected.quality,
            "supporting_hash_count": 0 if decision.evidence is None else decision.evidence.distinct_hash_count,
            "supporting_viewpoint_count": 0 if decision.evidence is None else decision.evidence.distinct_viewpoint_count,
            "fresh_confirmation_frame_sequences": [row.frame_sequence for row in confirmations],
            "confirmation_moves": confirmation_moves,
            "fresh_confirmation_passed": fresh_pass,
            "return_position_error_mm": return_error_mm,
            "return_orientation_error_deg": return_error_deg,
            "model_retrained": False,
            "decision_parameters_changed": False,
            "physical_grid": grid_spec,
        }
        assert_no_privileged_fields(actor_result)
        public_dir.mkdir(parents=True, exist_ok=True)
        privileged_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(public_dir / "ultrasound.npz", ultrasound=np.stack(images))
        write_jsonl(public_dir / "frames.jsonl", public_rows)
        write_jsonl(privileged_dir / "annotations.jsonl", privileged_rows)
        write_jsonl(privileged_dir / "actions.jsonl", action_rows)
        write_json(result_dir / "actor_result.json", actor_result)
        actor_hash = seal(result_dir / "actor_result.json")
        summary.update(
            status="COMPLETED",
            actor_outcome=actor_outcome,
            actor_result_sha256=actor_hash,
            frame_count=len(public_rows),
            action_step_count=total_action_steps,
            robot_video_frame_count=0 if robot_video is None else robot_video.frame_count,
        )
        return 0
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        return 1
    finally:
        if env is not None:
            if robot_video is not None:
                robot_video.capture(force=True)
                robot_video.close()
            env.close()
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if robot_video is not None:
            summary["robot_video_frame_count"] = robot_video.frame_count
            summary["robot_video"] = str(robot_video.output)
        write_json(output / "summary.json", summary)
        if action_rows and not (privileged_dir / "actions.jsonl").exists():
            write_jsonl(privileged_dir / "actions_partial.jsonl", action_rows)
        if public_rows and not (public_dir / "frames.jsonl").exists():
            write_jsonl(public_dir / "frames_partial.jsonl", public_rows)
        if images and not (public_dir / "ultrasound.npz").exists():
            public_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                public_dir / "ultrasound_partial.npz",
                ultrasound=np.stack(images),
            )


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)
