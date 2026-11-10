#!/usr/bin/env python3
"""Record a short deterministic rendered SonoGym checkpoint playback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-robot-US-guidance-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--frames", type=int, default=180)
parser.add_argument("--output-dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 60 <= args.frames <= 300:
    parser.error("the bounded demo requires 60 to 300 frames")
args.enable_cameras = True

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import spinal_surgery  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    env_cfg.seed = args.seed
    raw_env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    raw_env = gym.wrappers.RecordVideo(
        raw_env,
        video_folder=str(output_dir),
        step_trigger=lambda step: step == 0,
        video_length=args.frames,
        disable_logger=True,
        name_prefix="sonogym-checkpoint-demo",
    )
    unwrapped = raw_env.unwrapped
    env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")

    agent_cfg = load_cfg_from_registry(args.task, "skrl_cfg_entry_point")
    agent_cfg["seed"] = args.seed
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    agent_cfg["agent"]["experiment"]["directory"] = str(output_dir / "runner")
    agent_cfg["agent"]["experiment"]["experiment_name"] = "demo"
    runner = Runner(env, agent_cfg)
    runner.agent.load(str(checkpoint))
    runner.agent.set_running_mode("eval")

    unwrapped.seed(args.seed)
    obs, _ = env.reset()
    rewards = []
    terminated_seen = False
    truncated_seen = False
    for _ in range(args.frames):
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            if not bool(torch.isfinite(actions).all()):
                raise RuntimeError("nonfinite demo action")
            obs, reward, terminated, truncated, _ = env.step(actions)
        if not bool(torch.isfinite(obs).all()) or not bool(torch.isfinite(reward).all()):
            raise RuntimeError("nonfinite demo transition")
        rewards.append(float(reward[0].detach().cpu().item()))
        terminated_seen = terminated_seen or bool(terminated[0].detach().cpu().item())
        truncated_seen = truncated_seen or bool(truncated[0].detach().cpu().item())
    env.close()

    videos = sorted(output_dir.glob("*.mp4"))
    if len(videos) != 1 or videos[0].stat().st_size == 0:
        raise RuntimeError(f"expected one nonempty MP4, found {videos}")
    video = videos[0]
    manifest = {
        "status": "PASSED",
        "task": args.task,
        "seed": args.seed,
        "frames_requested": args.frames,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "video": str(video),
        "video_bytes": video.stat().st_size,
        "video_sha256": sha256(video),
        "reward_sum_over_recording": sum(rewards),
        "reward_values_finite": True,
        "terminated_seen": terminated_seen,
        "truncated_seen": truncated_seen,
        "semantics": "rendered simulator visualization; not the raw ultrasound policy observation",
        "scientific_scope": "demonstration artifact only; not a policy-quality evaluation",
    }
    manifest_path = output_dir / "demo-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


try:
    raise SystemExit(main())
finally:
    simulation_app.close()
