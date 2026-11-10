#!/usr/bin/env python3
"""Run the frozen Phase 5A four-controller SonoGym evaluation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--protocol", required=True)
parser.add_argument("--manifest", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument(
    "--controller",
    choices=("oracle", "zero_action", "bounded_random", "ppo_pilot"),
    help="Evaluate one controller in a fresh process (protocol-v2 mode).",
)
parser.add_argument(
    "--batch-index",
    type=int,
    help="Evaluate one four-start batch in a fresh process (protocol-v2 mode).",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from ruamel.yaml import YAML

import isaaclab_tasks  # noqa: F401
import spinal_surgery  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from skrl.utils.runner.torch import Runner
from spinal_surgery.lab.controllers.expert_guidance import ExpertGuidance

from navigation_metrics import EpisodeMetricTracker, SuccessThresholds


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_rows(value: torch.Tensor) -> list[list[float]]:
    return value.detach().cpu().to(torch.float64).numpy().tolist()


def load_inputs() -> tuple[dict, dict, Path, Path, Path]:
    protocol_path = Path(args.protocol).resolve()
    manifest_path = Path(args.manifest).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    for required in (protocol_path, manifest_path, checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    protocol = YAML(typ="safe").load(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_hash = protocol["initial_conditions"]["sha256"]
    if sha256_file(manifest_path) != expected_manifest_hash:
        raise RuntimeError("initial-condition manifest hash does not match protocol")
    expected_checkpoint_hash = protocol["ppo_checkpoint"]["sha256"]
    if sha256_file(checkpoint_path) != expected_checkpoint_hash:
        raise RuntimeError("PPO checkpoint hash does not match protocol")
    if manifest["count"] != 32 or len(manifest["starts"]) != 32:
        raise RuntimeError("protocol requires exactly 32 starts")
    if protocol["execution"]["num_parallel_envs"] != 4:
        raise RuntimeError("protocol requires four parallel environments")
    if protocol["execution"]["max_action_steps"] != 300:
        raise RuntimeError("protocol requires a 300-action-step horizon")
    output_dir.mkdir(parents=True, exist_ok=True)
    return protocol, manifest, checkpoint_path, output_dir, manifest_path


def deterministic_policy_action(runner: Runner, observation: torch.Tensor) -> torch.Tensor:
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    outputs = runner.agent.act(observation, timestep=0, timesteps=0)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    return outputs[-1].get("mean_actions", outputs[0])


def main() -> int:
    protocol, manifest, checkpoint_path, output_dir, manifest_path = load_inputs()
    task = protocol["task"]["gym_id"]
    num_envs = int(protocol["execution"]["num_parallel_envs"])
    horizon = int(protocol["execution"]["max_action_steps"])
    primary = SuccessThresholds(**protocol["metrics"]["primary_success"])
    secondary = SuccessThresholds(**protocol["metrics"]["secondary_success"])
    simulator_seed = int(protocol["seeds"]["simulator"])
    if (args.controller is None) != (args.batch_index is None):
        raise RuntimeError("--controller and --batch-index must be supplied together")
    if args.batch_index is not None and not 0 <= args.batch_index < 8:
        raise RuntimeError("--batch-index must be in [0, 7]")

    env_cfg = parse_env_cfg(
        task, device=args.device, num_envs=num_envs, use_fabric=True
    )
    env_cfg.seed = simulator_seed
    raw_env = gym.make(task, cfg=env_cfg, render_mode=None)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)
    unwrapped = raw_env.unwrapped
    if not hasattr(unwrapped, "set_evaluation_initial_commands"):
        raise RuntimeError("evaluation replay hook is unavailable")
    env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")

    runner = None
    if args.controller is None or args.controller == "ppo_pilot":
        agent_cfg = load_cfg_from_registry(task, "skrl_cfg_entry_point")
        agent_cfg["seed"] = int(protocol["seeds"]["ppo"])
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        agent_cfg["agent"]["experiment"]["directory"] = str(output_dir / "runner")
        agent_cfg["agent"]["experiment"]["experiment_name"] = "phase5a_eval"
        runner = Runner(env, agent_cfg)
        runner.agent.load(str(checkpoint_path))
        runner.agent.set_running_mode("eval")

    oracle_cfg = protocol["controllers"]["oracle"]
    oracle = None
    if args.controller is None or args.controller == "oracle":
        oracle = ExpertGuidance(
            unwrapped.max_action,
            unwrapped.action_scale,
            oracle_cfg["ratio"],
            unwrapped.sim.device,
        )
    random_seed = int(protocol["seeds"]["bounded_random"])
    random_rng = np.random.Generator(np.random.PCG64(random_seed))
    action_low = np.asarray(unwrapped.single_action_space.low, dtype=np.float32)
    action_high = np.asarray(unwrapped.single_action_space.high, dtype=np.float32)
    random_actions = random_rng.uniform(
        action_low,
        action_high,
        size=(32, horizon, action_low.size),
    ).astype(np.float32)

    starts = manifest["starts"]
    requested = np.asarray(
        [item["requested_command_x_index_z_index_angle_rad"] for item in starts],
        dtype=np.float32,
    )
    expected_effective = np.asarray(
        [item["expected_effective_command_after_native_clamp"] for item in starts],
        dtype=np.float64,
    )
    controller_names = (
        [args.controller] if args.controller else protocol["controllers"]["order"]
    )
    batch_starts = (
        [args.batch_index * num_envs]
        if args.batch_index is not None
        else list(range(0, 32, num_envs))
    )
    reference_initials: dict[str, dict[str, np.ndarray]] = {}
    initial_states: dict[str, dict[str, list[float]]] = {}
    controller_reports: dict[str, dict] = {}
    pairing_deltas: list[dict] = []
    trajectory_path = output_dir / "per_step.jsonl.gz"
    started = time.monotonic()
    reset_count = 0

    with gzip.open(trajectory_path, "wt", encoding="utf-8") as step_stream:
        for controller_name in controller_names:
            torch.manual_seed(int(protocol["seeds"][controller_name]))
            torch.cuda.manual_seed_all(int(protocol["seeds"][controller_name]))
            episodes: list[dict] = []
            controller_started = time.monotonic()
            for batch_start in batch_starts:
                batch_number = batch_start // num_envs
                batch_commands = requested[batch_start : batch_start + num_envs]
                print(
                    f"PHASE5A_BATCH_START controller={controller_name} "
                    f"batch={batch_number} starts={batch_start}:{batch_start + num_envs}",
                    flush=True,
                )
                unwrapped.set_evaluation_initial_commands(batch_commands)
                if reset_count == 0:
                    observation, _ = env.reset()
                    reset_method = "explicit_initial_reset"
                else:
                    # A direct wrapper reset after a protocol-truncated episode hung
                    # in failed job 50641608. Trigger the native time-limit auto-reset
                    # path already validated by Phase 4 smoke testing, and discard this
                    # inter-episode transition from all Phase 5A measurements.
                    unwrapped.episode_length_buf[:] = unwrapped.max_episode_length - 2
                    reset_actions = torch.zeros(
                        (num_envs, action_low.size),
                        dtype=torch.float32,
                        device=unwrapped.sim.device,
                    )
                    observation, _, reset_terminated, reset_truncated, _ = env.step(
                        reset_actions
                    )
                    if bool(reset_terminated.any()) or not bool(reset_truncated.all()):
                        raise RuntimeError("native time-limit auto-reset did not fire synchronously")
                    reset_method = "native_time_limit_auto_reset"
                reset_count += 1
                print(
                    f"PHASE5A_RESET_DONE controller={controller_name} "
                    f"batch={batch_number} method={reset_method}",
                    flush=True,
                )
                if not bool(torch.isfinite(observation).all()):
                    raise RuntimeError("nonfinite initial ultrasound observation")

                effective_commands = np.asarray(
                    tensor_rows(unwrapped.US_slicer.current_x_z_x_angle_cmd)
                )
                if not np.allclose(
                    effective_commands,
                    expected_effective[batch_start : batch_start + num_envs],
                    atol=1.0e-5,
                    rtol=0.0,
                ):
                    raise RuntimeError("native reset command did not match frozen manifest")
                initial_poses = np.asarray(tensor_rows(unwrapped.cur_cmd_pose))
                goal_poses = np.asarray(tensor_rows(unwrapped.goal_cmd_pose))
                trackers = [
                    EpisodeMetricTracker(
                        initial_poses[index],
                        goal_poses[index],
                        label_resolution_m=float(unwrapped.US_slicer.label_res),
                        primary=primary,
                        secondary=secondary,
                    )
                    for index in range(num_envs)
                ]
                constraint_events = [0] * num_envs
                command_clamp_steps = [0] * num_envs
                action_saturation_dims = [0] * num_envs
                action_dims = [0] * num_envs
                final_terminated = [False] * num_envs
                final_native_truncated = [False] * num_envs
                completed = [False] * num_envs

                for local_index in range(num_envs):
                    start_id = starts[batch_start + local_index]["start_id"]
                    current = {
                        "realized_pose": initial_poses[local_index],
                        "goal_pose": goal_poses[local_index],
                    }
                    initial_states[start_id] = {
                        "realized_pose": current["realized_pose"].astype(float).tolist(),
                        "goal_pose": current["goal_pose"].astype(float).tolist(),
                    }
                    if controller_name == controller_names[0]:
                        reference_initials[start_id] = current
                    else:
                        reference = reference_initials[start_id]
                        position_delta_mm = float(
                            np.linalg.norm(
                                current["realized_pose"][:2]
                                - reference["realized_pose"][:2]
                            )
                            * float(unwrapped.US_slicer.label_res)
                            * 1000.0
                        )
                        orientation_delta_deg = float(
                            abs(
                                (current["realized_pose"][2] - reference["realized_pose"][2] + np.pi)
                                % (2 * np.pi)
                                - np.pi
                            )
                            * 180.0
                            / np.pi
                        )
                        goal_delta = float(
                            np.max(np.abs(current["goal_pose"] - reference["goal_pose"]))
                        )
                        pairing_deltas.append(
                            {
                                "controller": controller_name,
                                "start_id": start_id,
                                "realized_position_delta_mm_from_oracle_reset": position_delta_mm,
                                "realized_orientation_delta_deg_from_oracle_reset": orientation_delta_deg,
                                "goal_pose_max_abs_delta": goal_delta,
                            }
                        )

                for step in range(1, horizon + 1):
                    with torch.inference_mode():
                        if controller_name == "oracle":
                            assert oracle is not None
                            actions = oracle.get_action(unwrapped.extras)
                        elif controller_name == "zero_action":
                            actions = torch.zeros(
                                (num_envs, action_low.size),
                                dtype=observation.dtype,
                                device=observation.device,
                            )
                        elif controller_name == "bounded_random":
                            actions = torch.as_tensor(
                                random_actions[
                                    batch_start : batch_start + num_envs, step - 1, :
                                ],
                                dtype=observation.dtype,
                                device=observation.device,
                            )
                        elif controller_name == "ppo_pilot":
                            assert runner is not None
                            actions = deterministic_policy_action(runner, observation)
                        else:
                            raise RuntimeError(f"unknown controller {controller_name}")
                        if not bool(torch.isfinite(actions).all()):
                            raise RuntimeError("nonfinite controller action")
                        next_observation, rewards, terminated, truncated, _ = env.step(
                            actions
                        )

                    observation_finite = torch.isfinite(next_observation).reshape(
                        num_envs, -1
                    ).all(dim=1)
                    reward_finite = torch.isfinite(rewards).reshape(num_envs, -1).all(
                        dim=1
                    )
                    pose_tensor = unwrapped.cur_cmd_pose.detach()
                    pose_finite = torch.isfinite(pose_tensor).all(dim=1)
                    transition_finite = reward_finite & pose_finite
                    applied_actions = unwrapped.actions.detach()
                    saturated = applied_actions.abs() >= (
                        unwrapped.max_action - 1.0e-6
                    )
                    current_commands = unwrapped.US_slicer.current_x_z_x_angle_cmd.detach()
                    at_command_bound = torch.isclose(
                        current_commands, unwrapped.US_slicer.x_z_range[0], atol=1.0e-5
                    ) | torch.isclose(
                        current_commands, unwrapped.US_slicer.x_z_range[1], atol=1.0e-5
                    )
                    no_collide_sensor = unwrapped.US_slicer.no_collide.detach()

                    for local_index in range(num_envs):
                        if completed[local_index]:
                            continue
                        global_index = batch_start + local_index
                        pose = tensor_rows(pose_tensor[local_index : local_index + 1])[0]
                        reward = float(rewards[local_index].detach().cpu().item())
                        pos_mm, angle_deg, success, success_5 = trackers[
                            local_index
                        ].record(
                            step=step,
                            pose=pose,
                            reward=reward,
                            observation_finite=bool(observation_finite[local_index].item()),
                            transition_finite=bool(transition_finite[local_index].item()),
                        )
                        constraint_events[local_index] += int(
                            bool(no_collide_sensor[local_index].detach().cpu().item())
                        )
                        command_clamp_steps[local_index] += int(
                            bool(at_command_bound[local_index].any().detach().cpu().item())
                        )
                        action_saturation_dims[local_index] += int(
                            saturated[local_index].sum().detach().cpu().item()
                        )
                        action_dims[local_index] += int(
                            saturated[local_index].numel()
                        )
                        was_terminated = bool(
                            terminated[local_index].detach().cpu().item()
                        )
                        was_truncated = bool(
                            truncated[local_index].detach().cpu().item()
                        )
                        step_stream.write(
                            json.dumps(
                                {
                                    "controller": controller_name,
                                    "start_id": starts[global_index]["start_id"],
                                    "step": step,
                                    "pose": pose,
                                    "action": tensor_rows(
                                        actions[local_index : local_index + 1]
                                    )[0],
                                    "applied_action": tensor_rows(
                                        applied_actions[
                                            local_index : local_index + 1
                                        ]
                                    )[0],
                                    "reward": reward,
                                    "position_error_mm": pos_mm,
                                    "orientation_error_deg": angle_deg,
                                    "success_10mm_10deg": success,
                                    "success_5mm_5deg": success_5,
                                    "native_terminated": was_terminated,
                                    "native_truncated": was_truncated,
                                    "native_no_collide_sensor": bool(
                                        no_collide_sensor[local_index]
                                        .detach()
                                        .cpu()
                                        .item()
                                    ),
                                    "command_at_native_bound": bool(
                                        at_command_bound[local_index]
                                        .any()
                                        .detach()
                                        .cpu()
                                        .item()
                                    ),
                                },
                                allow_nan=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        if was_terminated or was_truncated:
                            final_terminated[local_index] = was_terminated
                            final_native_truncated[local_index] = was_truncated
                            completed[local_index] = True
                    observation = next_observation
                    if any(completed) and not all(completed):
                        raise RuntimeError(
                            "asynchronous native completion occurred before protocol horizon"
                        )
                    if all(completed):
                        break

                for local_index in range(num_envs):
                    global_index = batch_start + local_index
                    episode = trackers[local_index].finalize(
                        terminated=final_terminated[local_index],
                        native_truncated=final_native_truncated[local_index],
                        protocol_horizon_reached=not completed[local_index]
                        and trackers[local_index].episode_length == horizon,
                        out_of_bounds=False,
                    )
                    episode.update(
                        {
                            "controller": controller_name,
                            "start_id": starts[global_index]["start_id"],
                            "start_index": global_index,
                            "requested_initial_command": requested[
                                global_index
                            ].astype(float).tolist(),
                            "effective_initial_command": effective_commands[
                                local_index
                            ].tolist(),
                            "native_no_collide_sensor_steps": constraint_events[
                                local_index
                            ],
                            "command_at_native_bound_steps": command_clamp_steps[
                                local_index
                            ],
                            "action_saturation_rate": (
                                action_saturation_dims[local_index]
                                / action_dims[local_index]
                                if action_dims[local_index]
                                else 0.0
                            ),
                        }
                    )
                    episodes.append(episode)
                step_stream.flush()
                print(
                    f"PHASE5A_BATCH_DONE controller={controller_name} "
                    f"batch={batch_number} episodes={len(episodes)}",
                    flush=True,
                )

            controller_reports[controller_name] = {
                "seed": int(protocol["seeds"][controller_name]),
                "elapsed_seconds": time.monotonic() - controller_started,
                "episodes": episodes,
            }
            print(
                f"PHASE5A_CONTROLLER_DONE controller={controller_name} "
                f"episodes={len(episodes)}",
                flush=True,
            )

    replay_tolerance = protocol["initial_conditions"]["realized_pairing_tolerance"]
    max_position_delta = max(
        (item["realized_position_delta_mm_from_oracle_reset"] for item in pairing_deltas),
        default=0.0,
    )
    max_angle_delta = max(
        (item["realized_orientation_delta_deg_from_oracle_reset"] for item in pairing_deltas),
        default=0.0,
    )
    max_goal_delta = max(
        (item["goal_pose_max_abs_delta"] for item in pairing_deltas), default=0.0
    )
    replay_verified = (
        max_position_delta <= replay_tolerance["position_mm"]
        and max_angle_delta <= replay_tolerance["orientation_deg"]
        and max_goal_delta <= replay_tolerance["goal_pose_abs"]
    )
    all_episodes = [
        episode
        for controller in controller_reports.values()
        for episode in controller["episodes"]
    ]
    finite_and_valid = all(
        episode["nonfinite_observations"] == 0
        and episode["invalid_transitions"] == 0
        for episode in all_episodes
    )
    process_isolated_batch = args.controller is not None
    report = {
        "status": "PASSED"
        if finite_and_valid and (process_isolated_batch or replay_verified)
        else "FAILED",
        "protocol_id": protocol["protocol_id"],
        "task": task,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "environment": {
            "label_resolution_m": float(unwrapped.US_slicer.label_res),
            "action_space_low": action_low.tolist(),
            "action_space_high": action_high.tolist(),
            "native_max_episode_length": int(unwrapped.max_episode_length),
            "physics_dt_seconds": float(unwrapped.physics_dt),
            "action_dt_seconds": float(unwrapped.step_dt),
            "native_out_of_bounds_termination_enabled": False,
        },
        "execution": {
            "num_starts": len(all_episodes),
            "num_parallel_envs": num_envs,
            "max_action_steps": horizon,
            "controllers": controller_names,
            "process_isolated_batch": process_isolated_batch,
            "batch_index": args.batch_index,
            "initial_reset_method": "explicit_initial_reset",
            "inter_batch_reset_method": (
                "none_fresh_process_per_batch"
                if process_isolated_batch
                else "discarded_native_time_limit_auto_reset"
            ),
            "transitions": sum(
                episode["episode_length"] for episode in all_episodes
            ),
            "elapsed_seconds": time.monotonic() - started,
        },
        "paired_replay": {
            "verified": replay_verified if not process_isolated_batch else None,
            "verification_scope": (
                "complete_in_process"
                if not process_isolated_batch
                else "deferred_to_protocol_v2_aggregator"
            ),
            "max_realized_position_delta_mm": max_position_delta,
            "max_realized_orientation_delta_deg": max_angle_delta,
            "max_goal_pose_abs_delta": max_goal_delta,
            "per_start_deltas": pairing_deltas,
        },
        "all_observations_rewards_poses_finite": finite_and_valid,
        "initial_states": initial_states,
        "controllers": controller_reports,
        "per_step_path": str(trajectory_path),
    }
    report_path = output_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "transitions": report["execution"]["transitions"],
                "elapsed_seconds": report["execution"]["elapsed_seconds"],
                "paired_replay": report["paired_replay"],
                "output": str(report_path),
            },
            indent=2,
            default=str,
        )
    )
    unwrapped.set_evaluation_initial_commands(None)
    env.close()
    return 0 if report["status"] == "PASSED" else 1


try:
    raise SystemExit(main())
finally:
    simulation_app.close()
