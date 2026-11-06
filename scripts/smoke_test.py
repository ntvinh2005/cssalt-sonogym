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
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output-dir", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs != 1:
        parser.error("The initial smoke test is intentionally fixed to one environment")
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


def validate_numeric(name: str, value) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    arrays = list(numeric_arrays(value))
    if not arrays:
        raise AssertionError(f"{name} did not contain a tensor or ndarray")
    for child_name, array in arrays:
        full_name = f"{name}.{child_name}".rstrip(".")
        shapes[full_name] = list(array.shape)
        if not np.issubdtype(array.dtype, np.number):
            raise AssertionError(f"{full_name} has non-numeric dtype {array.dtype}")
        if not np.isfinite(array).all():
            raise AssertionError(f"{full_name} contains NaN or infinity")
    return shapes


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
    try:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device or "cuda:0",
            num_envs=args_cli.num_envs,
        )
        env_cfg.seed = args_cli.seed
        env = gym.make(args_cli.task, cfg=env_cfg)

        observation, _ = env.reset(seed=args_cli.seed)
        summary["initial_observation_shapes"] = validate_numeric("observation", observation)
        first_policy = dict(numeric_arrays(observation))
        if "policy" in first_policy:
            samples["reset_0"] = first_policy["policy"][0]

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
            validate_numeric("observation", observation)
            validate_numeric("reward", reward)
            if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
                raise AssertionError("termination outputs must be boolean tensors")
            if terminated.shape != truncated.shape:
                raise AssertionError("terminated and truncated shapes differ")
            termination_seen = termination_seen or bool(torch.any(terminated | truncated).item())
            if step in (0, args_cli.steps - 1):
                policy = dict(numeric_arrays(observation)).get("policy")
                if policy is not None:
                    samples[f"step_{step + 1}"] = policy[0]

        second_observation, _ = env.reset(seed=args_cli.seed + 1)
        summary["second_reset_observation_shapes"] = validate_numeric(
            "observation", second_observation
        )
        policy = dict(numeric_arrays(second_observation)).get("policy")
        if policy is not None:
            samples["reset_1"] = policy[0]
        if not samples:
            raise AssertionError("no policy observations were available to save")

        np.savez_compressed(output_dir / "ultrasound_samples.npz", **samples)
        summary.update(
            status="PASSED",
            steps_completed=args_cli.steps,
            termination_seen=termination_seen,
            sample_file="ultrasound_samples.npz",
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
