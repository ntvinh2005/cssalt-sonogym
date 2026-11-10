#!/usr/bin/env python3
"""GPU smoke test for the real SonoGym guidance environment."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-robot-US-guidance-v0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--output-dir", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs != 1:
        parser.error("The initial smoke test is intentionally fixed to one environment")
    if not 20 <= args.steps <= 50:
        parser.error("The bounded smoke test requires between 20 and 50 action steps")
    return args


args_cli = parse_args()
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("MPLBACKEND", "Agg")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
import spinal_surgery  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def numeric_arrays(value):
    if isinstance(value, dict):
        for key, child in value.items():
            for name, array in numeric_arrays(child):
                yield f"{key}.{name}" if name else key, array
    elif torch.is_tensor(value):
        yield "", value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        yield "", value


def validate_numeric(name: str, value) -> dict[str, dict[str, object]]:
    diagnostics: dict[str, dict[str, object]] = {}
    arrays = list(numeric_arrays(value))
    if not arrays:
        raise AssertionError(f"{name} did not contain a tensor or ndarray")
    for child_name, array in arrays:
        full_name = f"{name}.{child_name}".rstrip(".")
        if not np.issubdtype(array.dtype, np.number):
            raise AssertionError(f"{full_name} has non-numeric dtype {array.dtype}")
        if not np.isfinite(array).all():
            raise AssertionError(f"{full_name} contains NaN or infinity")
        diagnostics[full_name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite": True,
            "minimum": float(array.min()),
            "maximum": float(array.max()),
            "mean": float(array.mean()),
            "nonzero_fraction": float(np.count_nonzero(array) / array.size),
        }
    return diagnostics


def tensor_row(value: torch.Tensor) -> list[float]:
    return value.detach().cpu().reshape(value.shape[0], -1)[0].tolist()


def main() -> None:
    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    env = None
    started = time.monotonic()
    summary: dict[str, object] = {
        "status": "FAILED",
        "task": args_cli.task,
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "steps_requested": args_cli.steps,
        "gpu_test": True,
    }
    samples: dict[str, np.ndarray] = {}
    trajectory: list[dict[str, object]] = []
    try:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device or "cuda:0",
            num_envs=args_cli.num_envs,
        )
        env_cfg.seed = args_cli.seed
        env = gym.make(args_cli.task, cfg=env_cfg)

        observation, _ = env.reset(seed=args_cli.seed)
        summary["observation_keys"] = sorted(observation.keys())
        summary["initial_observation"] = validate_numeric("observation", observation)
        first_policy = dict(numeric_arrays(observation))
        if "policy" not in first_policy:
            raise AssertionError("policy ultrasound observation is missing")
        if tuple(first_policy["policy"].shape) != (1, 1, 150, 200):
            raise AssertionError(
                f"unexpected policy ultrasound shape {first_policy['policy'].shape}; "
                "expected (1, 1, 150, 200)"
            )
        samples["reset_0"] = first_policy["policy"][0]

        unwrapped = env.unwrapped
        initial_joint_pos = unwrapped.robot.data.joint_pos.detach().clone()
        initial_ee_pose = unwrapped.US_ee_pose_w.detach().clone()
        initial_sim_counter = int(unwrapped._sim_step_counter)
        physics_dt = float(unwrapped.physics_dt)

        action_shape = env.unwrapped.single_action_space.shape
        low = torch.as_tensor(env.unwrapped.single_action_space.low, device=env.unwrapped.device)
        high = torch.as_tensor(env.unwrapped.single_action_space.high, device=env.unwrapped.device)
        generator = torch.Generator(device=env.unwrapped.device).manual_seed(args_cli.seed)

        termination_seen = False
        for step in range(args_cli.steps):
            unit = torch.rand(
                (args_cli.num_envs, *action_shape),
                generator=generator,
                device=env.unwrapped.device,
            )
            action = low + unit * (high - low)
            observation, reward, terminated, truncated, _ = env.step(action)
            observation_diagnostics = validate_numeric("observation", observation)
            reward_diagnostics = validate_numeric("reward", reward)
            if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
                raise AssertionError("termination outputs must be boolean tensors")
            if terminated.shape != truncated.shape:
                raise AssertionError("terminated and truncated shapes differ")
            termination_seen = termination_seen or bool(torch.any(terminated | truncated).item())
            trajectory.append(
                {
                    "step": step + 1,
                    "sim_counter": int(unwrapped._sim_step_counter),
                    "action": tensor_row(action),
                    "reward": tensor_row(reward),
                    "terminated": bool(terminated[0].item()),
                    "truncated": bool(truncated[0].item()),
                    "joint_position": tensor_row(unwrapped.robot.data.joint_pos),
                    "end_effector_pose": tensor_row(unwrapped.US_ee_pose_w),
                }
            )
            if step in (0, args_cli.steps - 1):
                policy = dict(numeric_arrays(observation)).get("policy")
                if policy is not None:
                    samples[f"step_{step + 1}"] = policy[0]
                    summary[f"step_{step + 1}_observation"] = observation_diagnostics
                    summary[f"step_{step + 1}_reward"] = reward_diagnostics

        final_joint_pos = unwrapped.robot.data.joint_pos.detach().clone()
        final_ee_pose = unwrapped.US_ee_pose_w.detach().clone()
        final_sim_counter = int(unwrapped._sim_step_counter)
        joint_delta = float(torch.max(torch.abs(final_joint_pos - initial_joint_pos)).item())
        ee_delta = float(torch.max(torch.abs(final_ee_pose - initial_ee_pose)).item())
        if final_sim_counter <= initial_sim_counter:
            raise AssertionError("simulator step counter did not advance")
        if max(joint_delta, ee_delta) <= 1.0e-8:
            raise AssertionError("robot joint and end-effector states did not advance")
        summary["simulation_advance"] = {
            "initial_counter": initial_sim_counter,
            "final_counter": final_sim_counter,
            "physics_dt_seconds": physics_dt,
            "elapsed_simulation_seconds": (final_sim_counter - initial_sim_counter) * physics_dt,
            "max_joint_position_delta": joint_delta,
            "max_end_effector_pose_delta": ee_delta,
        }

        # Exercise the environment's real time-limit path without waiting hundreds of steps.
        # This modifies only the episode progress buffer used by the smoke test; observations,
        # actions, rewards, simulator dynamics, and task configuration remain unchanged.
        unwrapped.episode_length_buf[:] = unwrapped.max_episode_length - 2
        zero_action = torch.zeros(
            (args_cli.num_envs, *action_shape), device=unwrapped.device
        )
        _, timeout_reward, timeout_terminated, timeout_truncated, _ = env.step(zero_action)
        validate_numeric("timeout_reward", timeout_reward)
        if not bool(torch.all(timeout_truncated).item()):
            raise AssertionError("time-limit probe did not produce truncation")
        summary["termination_probe"] = {
            "method": "episode_length_buffer_advanced_to_time_limit",
            "terminated": bool(timeout_terminated[0].item()),
            "truncated": bool(timeout_truncated[0].item()),
        }

        second_observation, _ = env.reset(seed=args_cli.seed + 1)
        summary["second_reset_observation"] = validate_numeric(
            "observation", second_observation
        )
        policy = dict(numeric_arrays(second_observation)).get("policy")
        if policy is not None:
            samples["reset_1"] = policy[0]
        if not samples:
            raise AssertionError("no policy observations were available to save")

        np.savez_compressed(output_dir / "ultrasound_original_observations.npz", **samples)
        (output_dir / "trajectory.json").write_text(
            json.dumps(trajectory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary.update(
            status="PASSED",
            steps_completed=args_cli.steps,
            termination_seen=termination_seen,
            sample_file="ultrasound_original_observations.npz",
            sample_semantics="raw simulated policy observation; not a rendered visualization",
            trajectory_file="trajectory.json",
        )
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        raise
    finally:
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
