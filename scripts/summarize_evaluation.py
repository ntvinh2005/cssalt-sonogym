#!/usr/bin/env python3
"""Compare paired SonoGym checkpoint and zero-action evaluation reports."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def describe(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def arm_summary(report: dict) -> dict[str, object]:
    episodes = report["episodes"]
    returns = [float(item["episode_return"]) for item in episodes]
    final_distances = [
        float(item["final_composite_distance_from_reward_telescoping"])
        for item in episodes
    ]
    saturation_rates = [float(item["action_saturation_rate"]) for item in episodes]
    return {
        "episodes": len(episodes),
        "episode_return": describe(returns),
        "final_composite_distance": describe(final_distances),
        "composite_distance_improvement_rate": statistics.fmean(
            float(item["composite_distance_improved"]) for item in episodes
        ),
        "native_termination_rate": statistics.fmean(
            float(item["terminated"]) for item in episodes
        ),
        "time_limit_truncation_rate": statistics.fmean(
            float(item["truncated"]) for item in episodes
        ),
        "action_saturation_rate": describe(saturation_rates),
        "vector_steps": report["vector_steps"],
        "transitions": report["transitions"],
        "evaluation_loop_seconds": report["evaluation_loop_seconds"],
        "transitions_per_second": report["transitions_per_second"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-report", required=True)
    parser.add_argument("--zero-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = json.loads(Path(args.checkpoint_report).read_text(encoding="utf-8"))
    zero = json.loads(Path(args.zero_report).read_text(encoding="utf-8"))
    for report, expected_mode in ((checkpoint, "checkpoint"), (zero, "zero_action")):
        if report.get("status") != "PASSED" or report.get("mode") != expected_mode:
            raise RuntimeError(f"invalid {expected_mode} report")
        if report.get("episodes_completed") != 20:
            raise RuntimeError(f"{expected_mode} did not complete 20 episodes")
        if not report.get("all_observations_actions_rewards_finite"):
            raise RuntimeError(f"{expected_mode} contains nonfinite values")
    if (
        checkpoint["task"] != zero["task"]
        or checkpoint["seed"] != zero["seed"]
        or checkpoint["num_envs"] != zero["num_envs"]
        or checkpoint["episodes_per_env"] != zero["episodes_per_env"]
    ):
        raise RuntimeError("evaluation arms do not share task, seed, or episode layout")

    checkpoint_episodes = checkpoint["episodes"]
    zero_episodes = zero["episodes"]
    max_initial_command_difference = 0.0
    paired_return_differences = []
    for checkpoint_episode, zero_episode in zip(checkpoint_episodes, zero_episodes, strict=True):
        checkpoint_key = (
            checkpoint_episode["environment_index"],
            checkpoint_episode["episode_index"],
        )
        zero_key = (zero_episode["environment_index"], zero_episode["episode_index"])
        if checkpoint_key != zero_key:
            raise RuntimeError("paired episode ordering differs")
        differences = [
            abs(float(left) - float(right))
            for left, right in zip(
                checkpoint_episode["requested_initial_command"],
                zero_episode["requested_initial_command"],
                strict=True,
            )
        ]
        max_initial_command_difference = max(max_initial_command_difference, *differences)
        paired_return_differences.append(
            float(checkpoint_episode["episode_return"])
            - float(zero_episode["episode_return"])
        )
    paired = max_initial_command_difference <= 1.0e-6
    finite = all(math.isfinite(value) for value in paired_return_differences)
    if not paired:
        raise RuntimeError(
            "requested initial commands are not paired: "
            f"max difference {max_initial_command_difference}"
        )
    if not finite:
        raise RuntimeError("paired return differences contain nonfinite values")

    mean_difference = statistics.fmean(paired_return_differences)
    if mean_difference > 0:
        exploratory_result = "CHECKPOINT_HIGHER_MEAN_RETURN_THAN_ZERO_ACTION"
    elif mean_difference < 0:
        exploratory_result = "CHECKPOINT_LOWER_MEAN_RETURN_THAN_ZERO_ACTION"
    else:
        exploratory_result = "EQUAL_MEAN_RETURN"
    summary = {
        "engineering_status": "PASSED",
        "task": checkpoint["task"],
        "evaluation_seed": checkpoint["seed"],
        "paired_episodes": len(paired_return_differences),
        "paired_initial_commands": paired,
        "paired_initial_command_max_abs_difference": max_initial_command_difference,
        "checkpoint": arm_summary(checkpoint),
        "zero_action": arm_summary(zero),
        "paired_checkpoint_minus_zero_return": {
            **describe(paired_return_differences),
            "fraction_checkpoint_higher": statistics.fmean(
                float(value > 0) for value in paired_return_differences
            ),
        },
        "exploratory_relative_result": exploratory_result,
        "native_success_criterion_available": False,
        "native_out_of_bounds_termination_enabled": False,
        "policy_usefulness": "NOT_ESTABLISHED_NO_NATIVE_SUCCESS_CRITERION",
        "clinical_safety": "NOT_EVALUATED",
        "interpretation": (
            "Return and composite-distance comparisons are paired engineering evidence. "
            "They are not a task-success, generalization, or clinical-safety result."
        ),
    }
    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
