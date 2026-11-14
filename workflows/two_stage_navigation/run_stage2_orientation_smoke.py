#!/usr/bin/env python3
"""Run one bounded physical orientation-convergence check in SonoGym."""

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
from spinal_surgery.evaluation.two_stage_navigation.coordinates import wrap_to_pi  # noqa: E402
from spinal_surgery.evaluation.two_stage_navigation.stage2_settling import (  # noqa: E402
    SettlingCriteria,
    evaluate_settled_frames,
)

from stage2_sim import (  # noqa: E402
    clean_ultrasound,
    current_realized_pose,
    move_proportional_target,
    move_orientation_only,
    move_to_pose,
    raw_label_slice,
    realized_feedback_target,
    surface_proxy_ok,
    take_proportional_step,
    take_step,
    tensor_row,
)


PROTOCOL_IDS = {
    "sonogym-stage2-orientation-smoke-v1",
    "sonogym-stage2-orientation-smoke-v2",
    "sonogym-stage2-orientation-smoke-v3",
    "sonogym-stage2-coupled-smoke-v4",
}


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


def main() -> int:
    protocol_path = Path(args.protocol).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_id"] not in PROTOCOL_IDS:
        raise RuntimeError("unexpected orientation protocol")
    protocol_hash = file_sha256(protocol_path)
    expected_hash = protocol_path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if protocol_hash != expected_hash:
        raise RuntimeError("orientation protocol seal mismatch")
    output = Path(args.output_dir).resolve()
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema": "sonogym-stage2-orientation-smoke-result-v1",
        "decision": "ERROR",
        "protocol_sha256": protocol_hash,
    }
    env = None
    action_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    privileged_rows: list[dict[str, Any]] = []
    images: list[np.ndarray] = []
    raw_images: list[np.ndarray] = []
    try:
        env_cfg = parse_env_cfg(
            protocol["scope"]["task"], device=args.device or "cuda:0", num_envs=1
        )
        env_cfg.seed = int(protocol["session"]["seed"])
        env = gym.make(protocol["scope"]["task"], cfg=env_cfg, render_mode=None)
        observation, _ = env.reset(seed=int(protocol["session"]["seed"]))
        unwrapped = env.unwrapped
        if tuple(observation["policy"].shape) != (1, 1, 200, 150):
            raise RuntimeError("unexpected policy observation shape")
        goal = tensor_row(unwrapped.goal_cmd_pose)
        requested = np.asarray(
            [goal[0], goal[1], float(protocol["motion"]["requested_angle_rad"])],
            dtype=np.float64,
        )
        controller_target = requested.copy()
        if protocol["protocol_id"] == "sonogym-stage2-coupled-smoke-v4":
            controller_target[2] = float(protocol["motion"]["controller_target_angle_rad"])
        orientation_trajectory: list[dict[str, Any]] = []
        coupled_trajectory: list[dict[str, Any]] = []
        if protocol["protocol_id"] == "sonogym-stage2-coupled-smoke-v4":
            coupled_trajectory = move_proportional_target(
                env,
                controller_target,
                requested,
                maximum_steps=int(protocol["motion"]["maximum_coupled_steps"]),
                maximum_position_error_mm=float(
                    protocol["criteria"]["maximum_command_position_error_mm"]
                ),
                maximum_orientation_error_deg=float(
                    protocol["criteria"]["maximum_command_orientation_error_deg"]
                ),
                ratio=tuple(float(value) for value in protocol["motion"]["controller_ratio"]),
            )
            action_rows.extend(coupled_trajectory)
        elif protocol["protocol_id"] in {
            "sonogym-stage2-orientation-smoke-v2",
            "sonogym-stage2-orientation-smoke-v3",
        }:
            orientation_trajectory = move_orientation_only(
                env,
                requested[2],
                maximum_steps=int(protocol["motion"]["maximum_orientation_steps"]),
                maximum_orientation_error_deg=float(
                    protocol["criteria"]["maximum_command_orientation_error_deg"]
                ),
            )
            action_rows.extend(orientation_trajectory)
        trajectory: list[dict[str, Any]] = []
        if protocol["protocol_id"] != "sonogym-stage2-coupled-smoke-v4":
            trajectory = move_to_pose(
                env,
                requested,
                maximum_steps=int(
                    protocol["motion"].get(
                        "maximum_positioning_steps",
                        protocol["motion"].get("maximum_settling_steps"),
                    )
                ),
                maximum_position_error_mm=float(
                    protocol["criteria"]["maximum_command_position_error_mm"]
                ),
                maximum_orientation_error_deg=float(
                    protocol["criteria"]["maximum_command_orientation_error_deg"]
                ),
            )
            action_rows.extend(trajectory)
        raw_path = (
            Path(args.assets_data_dir)
            / "HumanModels"
            / "selected_dataset_stl"
            / "s0010"
            / "combined_label_map.nii.gz"
        )
        raw_volume = np.asarray(nib.load(raw_path).dataobj, dtype=np.uint8)
        criteria = SettlingCriteria(**protocol["criteria"])
        previous_pose = current_realized_pose(unwrapped).copy()
        for frame_sequence in range(criteria.required_frame_count):
            if protocol["protocol_id"] == "sonogym-stage2-coupled-smoke-v4":
                _, status = take_proportional_step(
                    env,
                    controller_target,
                    ratio=tuple(
                        float(value) for value in protocol["motion"]["controller_ratio"]
                    ),
                )
            else:
                feedback_target = realized_feedback_target(unwrapped, requested)
                _, status = take_step(env, feedback_target)
            status["frame_sequence"] = frame_sequence
            action_rows.append(status)
            pose = current_realized_pose(unwrapped)
            image = clean_ultrasound(unwrapped)
            raw = raw_label_slice(unwrapped, raw_volume)
            frame_hash = ultrasound_sha256(image)
            technical = bool(
                status["finite"]
                and not status["command_position_clamped"]
                and status["command_mapping_valid"]
                and status["surface_proxy_ok"]
                and not status["terminated"]
                and not status["truncated"]
                and not bool(unwrapped.US_slicer.no_collide[0].item())
                and float(np.std(image)) > 1e-8
            )
            command_position_error = float(np.linalg.norm(pose[:2] - requested[:2]) * 1.5)
            command_orientation_error = (
                abs(float(wrap_to_pi(pose[2] - requested[2]))) * 180.0 / math.pi
            )
            goal_position_error = float(np.linalg.norm(pose[:2] - goal[:2]) * 1.5)
            goal_orientation_error = (
                abs(float(wrap_to_pi(pose[2] - goal[2]))) * 180.0 / math.pi
            )
            l4_fraction = float(np.count_nonzero(raw == 28) / raw.size)
            public = {
                "schema": "sonogym-stage2-orientation-actor-frame-v1",
                "frame_sequence": frame_sequence,
                "producer_sequence": int(unwrapped.num_step),
                "frame_sha256": frame_hash,
                "executed_delta_relative_mm": ((pose[:2] - previous_pose[:2]) * 1.5).tolist(),
                "executed_delta_angle_rad": float(wrap_to_pi(pose[2] - previous_pose[2])),
                "action_accepted": technical,
                "action_completed": technical,
                "action_clamped": bool(status["command_clamped"]),
                "action_position_clamped": bool(status["command_position_clamped"]),
                "action_angle_clamped": bool(status["command_angle_clamped"]),
                "contact_ok": surface_proxy_ok(unwrapped),
                "workspace_ok": not bool(status["command_position_clamped"]),
                "image_valid": technical,
                "image_archive_index": frame_sequence,
            }
            assert_no_privileged_fields(public)
            public_rows.append(public)
            privileged_rows.append(
                {
                    "frame_sequence": frame_sequence,
                    "producer_sequence": int(unwrapped.num_step),
                    "frame_sha256": frame_hash,
                    "technical_image_valid": technical,
                    "command_position_error_mm": command_position_error,
                    "command_orientation_error_deg": command_orientation_error,
                    "goal_position_error_mm": goal_position_error,
                    "goal_orientation_error_deg": goal_orientation_error,
                    "l4_visible_fraction": l4_fraction,
                    "realized_pose_index": pose.tolist(),
                    "requested_command_pose": requested.tolist(),
                    "goal_cmd_pose": goal.tolist(),
                }
            )
            images.append(image.copy())
            raw_images.append(raw.copy())
            previous_pose = pose
        metrics = evaluate_settled_frames(privileged_rows, criteria)
        result.update(
            decision=metrics["decision"],
            orientation_action_steps=len(orientation_trajectory),
            positioning_action_steps=len(trajectory),
            coupled_action_steps=len(coupled_trajectory),
            settling_action_steps=(
                len(orientation_trajectory) + len(trajectory) + len(coupled_trajectory)
            ),
            requested_command_pose=requested.tolist(),
            controller_target_pose=controller_target.tolist(),
            goal_cmd_pose_evaluator_only=goal.tolist(),
            metrics=metrics,
        )
        public_dir = output / "public"
        privileged_dir = output / "privileged"
        public_dir.mkdir(parents=True, exist_ok=True)
        privileged_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(public_dir / "ultrasound.npz", ultrasound=np.stack(images))
        np.savez_compressed(privileged_dir / "raw_labels.npz", raw_labels=np.stack(raw_images))
        write_jsonl(public_dir / "frames.jsonl", public_rows)
        write_jsonl(privileged_dir / "annotations.jsonl", privileged_rows)
        return 0 if metrics["decision"] == "PASS" else 3
    except TimeoutError as error:
        result["decision"] = "FAIL"
        result["engineering_status"] = "PHYSICAL_CONVERGENCE_FAIL"
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        return 3
    except Exception as error:
        result["decision"] = "ERROR"
        result["engineering_status"] = "TECHNICAL_ERROR"
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        return 2
    finally:
        result["elapsed_seconds"] = round(time.monotonic() - started, 6)
        result["action_count"] = len(action_rows)
        if action_rows:
            write_jsonl(output / "privileged" / "actions.jsonl", action_rows)
        write_json(output / "result.json", result)
        if env is not None:
            env.close()


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)
