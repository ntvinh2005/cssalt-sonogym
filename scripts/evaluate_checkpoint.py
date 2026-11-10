#!/usr/bin/env python3
"""Evaluate a SonoGym checkpoint or zero-action control without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-robot-US-guidance-v0")
parser.add_argument("--mode", choices=["checkpoint", "zero_action"], required=True)
parser.add_argument("--checkpoint")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--episodes-per-env", type=int, default=5)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.mode == "checkpoint" and not args.checkpoint:
    parser.error("--checkpoint is required in checkpoint mode")
if args.num_envs != 4 or args.episodes_per_env != 5:
    parser.error("the preregistered pilot requires 4 environments and 5 episodes per environment")

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import spinal_surgery  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_row(tensor: torch.Tensor, index: int) -> list[float]:
    return [float(value) for value in tensor[index].detach().cpu().flatten().tolist()]


def start_state(unwrapped, index: int) -> dict[str, object]:
    return {
        "requested_initial_command": tensor_row(
            unwrapped.US_slicer.current_x_z_x_angle_cmd, index
        ),
        "realized_initial_pose": tensor_row(unwrapped.cur_cmd_pose, index),
        "goal_pose": tensor_row(unwrapped.goal_cmd_pose, index),
        "initial_composite_distance": float(
            unwrapped.distance_to_goal[index].detach().cpu().item()
        ),
    }


def main() -> int:
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    env_cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=args.num_envs, use_fabric=True
    )
    env_cfg.seed = args.seed
    raw_env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)
    unwrapped = raw_env.unwrapped
    unwrapped.seed(args.seed)
    env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")

    runner = None
    checkpoint_record = None
    if args.mode == "checkpoint":
        agent_cfg = load_cfg_from_registry(args.task, "skrl_cfg_entry_point")
        agent_cfg["seed"] = args.seed
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        agent_cfg["agent"]["experiment"]["directory"] = str(
            output_path.parent / "runner"
        )
        agent_cfg["agent"]["experiment"]["experiment_name"] = "evaluation"
        runner = Runner(env, agent_cfg)
        runner.agent.load(str(checkpoint_path))
        runner.agent.set_running_mode("eval")
        checkpoint_record = {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": file_sha256(checkpoint_path),
        }

    # Runner construction consumes PyTorch RNG state. Re-seed immediately before the
    # first environment reset so checkpoint and zero-action arms receive the same
    # requested initial-command sequence.
    unwrapped.seed(args.seed)
    obs, _ = env.reset()
    if not bool(torch.isfinite(obs).all()):
        raise RuntimeError("nonfinite initial observation")

    episode_counts = [0] * args.num_envs
    returns = [0.0] * args.num_envs
    lengths = [0] * args.num_envs
    saturation_dims = [0] * args.num_envs
    action_dims = [0] * args.num_envs
    raw_action_abs_max = [0.0] * args.num_envs
    applied_action_abs_max = [0.0] * args.num_envs
    starts = [start_state(unwrapped, index) for index in range(args.num_envs)]
    episodes: list[dict[str, object]] = []
    vector_steps = 0
    loop_start = time.monotonic()

    while min(episode_counts) < args.episodes_per_env:
        with torch.inference_mode():
            if runner is not None:
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
            else:
                actions = torch.zeros(
                    (args.num_envs, unwrapped.cfg.action_space),
                    dtype=obs.dtype,
                    device=obs.device,
                )
            if not bool(torch.isfinite(actions).all()):
                raise RuntimeError("nonfinite policy action")
            next_obs, rewards, terminated, truncated, _ = env.step(actions)

        if not bool(torch.isfinite(next_obs).all()):
            raise RuntimeError("nonfinite observation")
        if not bool(torch.isfinite(rewards).all()):
            raise RuntimeError("nonfinite reward")

        applied_actions = unwrapped.actions.detach()
        saturated = applied_actions.abs() >= (unwrapped.max_action - 1.0e-6)
        vector_steps += 1
        for index in range(args.num_envs):
            if episode_counts[index] >= args.episodes_per_env:
                continue
            reward = float(rewards[index].detach().cpu().item())
            returns[index] += reward
            lengths[index] += 1
            saturation_dims[index] += int(saturated[index].sum().detach().cpu().item())
            action_dims[index] += int(applied_actions[index].numel())
            raw_action_abs_max[index] = max(
                raw_action_abs_max[index],
                float(actions[index].abs().max().detach().cpu().item()),
            )
            applied_action_abs_max[index] = max(
                applied_action_abs_max[index],
                float(applied_actions[index].abs().max().detach().cpu().item()),
            )
            was_terminated = bool(terminated[index].detach().cpu().item())
            was_truncated = bool(truncated[index].detach().cpu().item())
            if not (was_terminated or was_truncated):
                continue

            initial_distance = float(starts[index]["initial_composite_distance"])
            episode_return = returns[index]
            episodes.append(
                {
                    "mode": args.mode,
                    "environment_index": index,
                    "episode_index": episode_counts[index],
                    **starts[index],
                    "episode_return": episode_return,
                    "final_composite_distance_from_reward_telescoping": (
                        initial_distance - episode_return
                    ),
                    "composite_distance_improved": episode_return > 0.0,
                    "length": lengths[index],
                    "terminated": was_terminated,
                    "truncated": was_truncated,
                    "action_saturation_rate": (
                        saturation_dims[index] / action_dims[index]
                        if action_dims[index]
                        else 0.0
                    ),
                    "raw_action_abs_max": raw_action_abs_max[index],
                    "applied_action_abs_max": applied_action_abs_max[index],
                }
            )
            episode_counts[index] += 1
            returns[index] = 0.0
            lengths[index] = 0
            saturation_dims[index] = 0
            action_dims[index] = 0
            raw_action_abs_max[index] = 0.0
            applied_action_abs_max[index] = 0.0
            if episode_counts[index] < args.episodes_per_env:
                starts[index] = start_state(unwrapped, index)

        obs = next_obs

    loop_elapsed = time.monotonic() - loop_start
    report = {
        "status": "PASSED",
        "task": args.task,
        "mode": args.mode,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "episodes_per_env": args.episodes_per_env,
        "episodes_completed": len(episodes),
        "expected_episode_length": int(unwrapped.max_episode_length),
        "vector_steps": vector_steps,
        "transitions": vector_steps * args.num_envs,
        "evaluation_loop_seconds": loop_elapsed,
        "transitions_per_second": vector_steps * args.num_envs / loop_elapsed,
        "all_observations_actions_rewards_finite": True,
        "checkpoint": checkpoint_record,
        "episodes": sorted(
            episodes,
            key=lambda item: (item["environment_index"], item["episode_index"]),
        ),
    }
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))
    env.close()
    return 0


try:
    raise SystemExit(main())
finally:
    simulation_app.close()
