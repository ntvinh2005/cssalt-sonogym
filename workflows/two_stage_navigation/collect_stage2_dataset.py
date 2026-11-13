#!/usr/bin/env python3
"""Collect one fixed-protocol Stage 2 supervised development session."""

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
parser.add_argument("--session-id", required=True)
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
from spinal_surgery.evaluation.two_stage_navigation.stage2_models import (  # noqa: E402
    Stage2LabelConfig,
    supervised_targets,
)

from stage2_sim import (  # noqa: E402
    clean_ultrasound,
    current_command,
    current_realized_pose,
    move_to_pose,
    raw_label_slice,
    surface_proxy_ok,
    tensor_row,
)


PROTOCOL_ID = "sonogym-stage2-dataset-v1"


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


def load_protocol() -> tuple[dict[str, Any], dict[str, Any], Path]:
    path = Path(args.protocol).resolve()
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("unexpected Stage 2 dataset protocol")
    seal = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if file_sha256(path) != seal:
        raise RuntimeError("Stage 2 dataset protocol seal mismatch")
    sessions = {row["session_id"]: row for row in protocol["sessions"]}
    if args.session_id not in sessions:
        raise RuntimeError("session is absent from the frozen protocol")
    return protocol, sessions[args.session_id], path


def ordered_waypoints(
    protocol: dict[str, Any], session: dict[str, Any], goal: np.ndarray
) -> list[np.ndarray]:
    sampling = protocol["sampling"]
    x_values = [
        goal[0] + float(value) + float(session["x_jitter_index"])
        for value in sampling["goal_relative_x_offsets_index"]
    ]
    z_values = [
        goal[1] + float(value) + float(session["z_jitter_index"])
        for value in sampling["goal_relative_z_offsets_index"]
    ]
    angles = [
        float(value) + float(session["angle_jitter_rad"])
        for value in sampling["command_angles_rad"]
    ]
    rows: list[np.ndarray] = []
    for z_index, z_value in enumerate(z_values):
        current_x = x_values if z_index % 2 == 0 else list(reversed(x_values))
        for x_index, x_value in enumerate(current_x):
            current_angles = angles if (z_index + x_index) % 2 == 0 else list(reversed(angles))
            for angle in current_angles:
                rows.append(np.asarray([x_value, z_value, angle], dtype=np.float64))
    if len(rows) != int(sampling["expected_frames_per_session"]):
        raise RuntimeError("unexpected Stage 2 waypoint count")
    return rows


def main() -> int:
    protocol, session, protocol_path = load_protocol()
    output = Path(args.output_dir).resolve()
    public_dir = output / "public"
    privileged_dir = output / "privileged"
    started = time.monotonic()
    summary: dict[str, Any] = {
        "schema": "sonogym-stage2-dataset-session-summary-v1",
        "status": "FAILED",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": file_sha256(protocol_path),
        "session_id": session["session_id"],
        "split": session["split"],
        "seed": int(session["seed"]),
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
        env_cfg.seed = int(session["seed"])
        env = gym.make(protocol["scope"]["task"], cfg=env_cfg, render_mode=None)
        observation, _ = env.reset(seed=int(session["seed"]))
        unwrapped = env.unwrapped
        if tuple(observation["policy"].shape) != (1, 1, 200, 150):
            raise RuntimeError("unexpected policy observation shape")
        if unwrapped.US_slicer.sim_mode != "conv" or unwrapped.scene.num_envs != 1:
            raise RuntimeError("collector requires one conv SonoGym environment")

        raw_path = (
            Path(args.assets_data_dir)
            / "HumanModels"
            / "selected_dataset_stl"
            / "s0010"
            / "combined_label_map.nii.gz"
        )
        raw_volume = np.asarray(nib.load(raw_path).dataobj, dtype=np.uint8)
        goal = tensor_row(unwrapped.goal_cmd_pose)
        waypoints = ordered_waypoints(protocol, session, goal)
        sampling = protocol["sampling"]
        position_tolerance = float(sampling["physical_tracking_position_tolerance_mm"])
        orientation_tolerance = float(
            sampling["physical_tracking_orientation_tolerance_deg"]
        )
        label = protocol["labels"]
        label_config = Stage2LabelConfig(
            maximum_position_error_mm=float(label["validity"]["maximum_position_error_mm"]),
            maximum_orientation_error_deg=float(
                label["validity"]["maximum_orientation_error_deg"]
            ),
            minimum_l4_visible_fraction=float(
                label["validity"]["minimum_l4_visible_fraction"]
            ),
            visibility_normalization_fraction=float(
                label["quality"]["visibility_normalization_fraction"]
            ),
        )
        previous_realized = current_realized_pose(unwrapped).copy()
        total_steps = 0
        for frame_sequence, waypoint in enumerate(waypoints):
            maximum_steps = int(
                sampling[
                    "maximum_initial_positioning_steps"
                    if frame_sequence == 0
                    else "maximum_action_steps_per_waypoint"
                ]
            )
            trajectory = move_to_pose(
                env,
                waypoint,
                maximum_steps=maximum_steps,
                maximum_position_error_mm=position_tolerance,
                maximum_orientation_error_deg=orientation_tolerance,
            )
            total_steps += len(trajectory)
            if total_steps > int(sampling["maximum_total_action_steps_per_session"]):
                raise RuntimeError("Stage 2 dataset action budget exceeded")
            for row in trajectory:
                row["frame_sequence"] = frame_sequence
                action_rows.append(row)

            image = clean_ultrasound(unwrapped)
            raw = raw_label_slice(unwrapped, raw_volume)
            frame_hash = ultrasound_sha256(image)
            current_pose = current_realized_pose(unwrapped)
            delta_mm = (current_pose[:2] - previous_realized[:2]) * 1.5
            delta_angle = float(wrap_to_pi(current_pose[2] - previous_realized[2]))
            transition_valid = bool(
                all(bool(row["finite"]) and bool(row["surface_proxy_ok"]) for row in trajectory)
                and all(not bool(row["command_clamped"]) for row in trajectory)
            )
            technical_valid = bool(
                transition_valid
                and surface_proxy_ok(unwrapped)
                and not bool(unwrapped.US_slicer.no_collide[0].item())
                and float(np.std(image)) > 1e-8
            )
            position_error_mm = float(np.linalg.norm(current_pose[:2] - goal[:2]) * 1.5)
            orientation_error_deg = (
                abs(float(wrap_to_pi(current_pose[2] - goal[2]))) * 180.0 / math.pi
            )
            l4_fraction = float(np.count_nonzero(raw == 28) / raw.size)
            validity, quality = supervised_targets(
                technical_image_valid=technical_valid,
                position_error_mm=position_error_mm,
                orientation_error_deg=orientation_error_deg,
                l4_visible_fraction=l4_fraction,
                config=label_config,
            )
            public_row = {
                "schema": "sonogym-stage2-actor-frame-v1",
                "session_id": session["session_id"],
                "split": session["split"],
                "frame_sequence": frame_sequence,
                "producer_sequence": int(unwrapped.num_step),
                "frame_sha256": frame_hash,
                "image_archive": "public/ultrasound.npz",
                "image_archive_index": frame_sequence,
                "executed_delta_relative_mm": delta_mm.tolist(),
                "executed_delta_angle_rad": delta_angle,
                "action_steps_since_previous_frame": len(trajectory),
                "action_accepted": transition_valid,
                "action_completed": transition_valid,
                "action_clamped": any(bool(row["command_clamped"]) for row in trajectory),
                "contact_ok": surface_proxy_ok(unwrapped),
                "workspace_ok": all(not bool(row["command_clamped"]) for row in trajectory),
                "image_valid": technical_valid,
            }
            assert_no_privileged_fields(public_row)
            public_rows.append(public_row)
            privileged_rows.append(
                {
                    "schema": "sonogym-stage2-privileged-annotation-v1",
                    "session_id": session["session_id"],
                    "frame_sequence": frame_sequence,
                    "frame_sha256": frame_hash,
                    "patient_id": "s0010",
                    "command_pose_index": current_command(unwrapped).tolist(),
                    "realized_pose_index": current_pose.tolist(),
                    "goal_cmd_pose": goal.tolist(),
                    "position_error_mm": position_error_mm,
                    "orientation_error_deg": orientation_error_deg,
                    "l4_visible_fraction": l4_fraction,
                    "raw_label_pixel_counts": {
                        str(value): int(np.count_nonzero(raw == value))
                        for value in range(25, 32)
                    },
                    "technical_image_valid": technical_valid,
                    "validity_label": validity,
                    "quality_label": quality,
                    "raw_label_archive": "privileged/raw_labels.npz",
                    "raw_label_archive_index": frame_sequence,
                }
            )
            images.append(image.copy())
            raw_images.append(raw.copy())
            previous_realized = current_pose

        if len({row["frame_sha256"] for row in public_rows}) < len(public_rows) // 2:
            raise RuntimeError("Stage 2 dataset contains insufficient distinct images")
        public_dir.mkdir(parents=True, exist_ok=True)
        privileged_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(public_dir / "ultrasound.npz", ultrasound=np.stack(images))
        np.savez_compressed(privileged_dir / "raw_labels.npz", raw_labels=np.stack(raw_images))
        write_jsonl(public_dir / "frames.jsonl", public_rows)
        write_jsonl(privileged_dir / "annotations.jsonl", privileged_rows)
        write_jsonl(privileged_dir / "actions.jsonl", action_rows)
        summary.update(
            status="PASSED",
            frame_count=len(public_rows),
            valid_frame_count=sum(int(row["validity_label"]) for row in privileged_rows),
            distinct_frame_hash_count=len({row["frame_sha256"] for row in public_rows}),
            total_action_steps=total_steps,
            public_manifest="public/frames.jsonl",
            privileged_manifest="privileged/annotations.jsonl",
            ultrasound_archive="public/ultrasound.npz",
            raw_label_archive="privileged/raw_labels.npz",
        )
        return 0
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        return 1
    finally:
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json(output / "summary.json", summary)
        if env is not None:
            env.close()


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)
