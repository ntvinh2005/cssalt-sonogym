#!/usr/bin/env python3
"""Summarize Phase 5A results, apply the frozen gate, and create plots."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [center - half, center + half]


def bootstrap_median_interval(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(values), size=(5000, len(values)))
    medians = np.median(values[indices], axis=1)
    return np.percentile(medians, [2.5, 97.5]).astype(float).tolist()


def controller_summary(episodes: list[dict], seed: int) -> dict:
    position = np.asarray([item["final_position_error_mm"] for item in episodes])
    orientation = np.asarray([item["final_orientation_error_deg"] for item in episodes])
    returns = np.asarray([item["total_reward"] for item in episodes])
    lengths = np.asarray([item["episode_length"] for item in episodes])
    success_count = sum(bool(item["final_success"]) for item in episodes)
    secondary_count = sum(bool(item["final_success_5mm_5deg"]) for item in episodes)
    ever_count = sum(bool(item["ever_reached_success"]) for item in episodes)
    completion = Counter(item["completion_reason"] for item in episodes)
    return {
        "episodes": len(episodes),
        "final_success_count": success_count,
        "final_success_rate": success_count / len(episodes),
        "final_success_wilson_95": wilson_interval(success_count, len(episodes)),
        "final_success_5mm_5deg_count": secondary_count,
        "final_success_5mm_5deg_rate": secondary_count / len(episodes),
        "ever_reached_success_count": ever_count,
        "ever_reached_success_rate": ever_count / len(episodes),
        "median_position_error_mm": float(np.median(position)),
        "median_position_error_bootstrap_95": bootstrap_median_interval(position, seed),
        "p90_position_error_mm": float(np.percentile(position, 90)),
        "median_orientation_error_deg": float(np.median(orientation)),
        "median_orientation_error_bootstrap_95": bootstrap_median_interval(orientation, seed + 1),
        "p90_orientation_error_deg": float(np.percentile(orientation, 90)),
        "mean_episode_length": float(np.mean(lengths)),
        "median_episode_length": float(np.median(lengths)),
        "completion_distribution": dict(sorted(completion.items())),
        "native_no_collide_sensor_episode_count": sum(
            item["native_no_collide_sensor_steps"] > 0 for item in episodes
        ),
        "native_no_collide_sensor_total_steps": sum(
            item["native_no_collide_sensor_steps"] for item in episodes
        ),
        "command_bound_episode_count": sum(
            item["command_at_native_bound_steps"] > 0 for item in episodes
        ),
        "command_bound_total_steps": sum(
            item["command_at_native_bound_steps"] for item in episodes
        ),
        "mean_action_saturation_rate": float(
            np.mean([item["action_saturation_rate"] for item in episodes])
        ),
        "mean_episode_return": float(np.mean(returns)),
        "median_episode_return": float(np.median(returns)),
        "nonfinite_observations": sum(item["nonfinite_observations"] for item in episodes),
        "invalid_transitions": sum(item["invalid_transitions"] for item in episodes),
    }


def indexed(episodes: list[dict]) -> dict[str, dict]:
    return {item["start_id"]: item for item in episodes}


def paired_summary(oracle: list[dict], control: list[dict]) -> dict:
    oracle_by_start = indexed(oracle)
    control_by_start = indexed(control)
    starts = sorted(oracle_by_start)
    if starts != sorted(control_by_start):
        raise RuntimeError("controller start sets are not paired")
    position_advantage = np.asarray(
        [
            control_by_start[start]["final_position_error_mm"]
            - oracle_by_start[start]["final_position_error_mm"]
            for start in starts
        ]
    )
    orientation_advantage = np.asarray(
        [
            control_by_start[start]["final_orientation_error_deg"]
            - oracle_by_start[start]["final_orientation_error_deg"]
            for start in starts
        ]
    )
    success_advantage = np.asarray(
        [
            int(oracle_by_start[start]["final_success"])
            - int(control_by_start[start]["final_success"])
            for start in starts
        ]
    )
    return {
        "oracle_minus_control_success_count": int(np.sum(success_advantage)),
        "oracle_minus_control_success_rate_percentage_points": float(
            np.mean(success_advantage) * 100.0
        ),
        "median_paired_position_error_reduction_mm": float(
            np.median(position_advantage)
        ),
        "median_paired_orientation_error_reduction_deg": float(
            np.median(orientation_advantage)
        ),
        "starts_with_lower_oracle_position_error": int(np.sum(position_advantage > 0)),
        "starts_with_lower_oracle_orientation_error": int(
            np.sum(orientation_advantage > 0)
        ),
        "per_start": [
            {
                "start_id": start,
                "oracle_minus_control_success": int(success_advantage[index]),
                "control_minus_oracle_position_error_mm": float(position_advantage[index]),
                "control_minus_oracle_orientation_error_deg": float(
                    orientation_advantage[index]
                ),
            }
            for index, start in enumerate(starts)
        ],
    }


def write_episode_csv(path: Path, controllers: dict[str, list[dict]]) -> None:
    rows = [episode for episodes in controllers.values() for episode in episodes]
    scalar_keys = sorted(
        key
        for key, value in rows[0].items()
        if not isinstance(value, (list, dict))
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in scalar_keys})


def make_plots(output_dir: Path, controllers: dict[str, list[dict]]) -> None:
    colors = {
        "oracle": "#1b9e77",
        "zero_action": "#7570b3",
        "bounded_random": "#d95f02",
        "ppo_pilot": "#e7298a",
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, episodes in controllers.items():
        ax.scatter(
            [item["final_position_error_mm"] for item in episodes],
            [item["final_orientation_error_deg"] for item in episodes],
            label=name,
            alpha=0.72,
            s=34,
            color=colors[name],
        )
    ax.axvline(10.0, color="black", linestyle="--", linewidth=1)
    ax.axhline(10.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(5.0, color="grey", linestyle=":", linewidth=1)
    ax.axhline(5.0, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Final frontal-plane position error (mm)")
    ax.set_ylabel("Final wrapped orientation error (deg)")
    ax.set_title("SonoGym navigation final errors (32 frozen starts)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "final_error_scatter.png", dpi=180)
    plt.close(fig)

    oracle = indexed(controllers["oracle"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for column, control_name in enumerate(("zero_action", "bounded_random")):
        control = indexed(controllers[control_name])
        starts = sorted(oracle)
        x = np.arange(len(starts))
        axes[0, column].plot(
            x, [oracle[start]["final_position_error_mm"] for start in starts], label="oracle"
        )
        axes[0, column].plot(
            x,
            [control[start]["final_position_error_mm"] for start in starts],
            label=control_name,
        )
        axes[1, column].plot(
            x, [oracle[start]["final_orientation_error_deg"] for start in starts]
        )
        axes[1, column].plot(
            x, [control[start]["final_orientation_error_deg"] for start in starts]
        )
        axes[0, column].set_title(f"Oracle vs {control_name}")
        axes[0, column].set_ylabel("Position error (mm)")
        axes[1, column].set_ylabel("Orientation error (deg)")
        axes[1, column].set_xlabel("Frozen start index")
        axes[0, column].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "paired_oracle_controls.png", dpi=180)
    plt.close(fig)


def representative_trajectories(
    output_dir: Path, per_step_path: Path, controllers: dict[str, list[dict]]
) -> dict:
    oracle = controllers["oracle"]
    by_start = indexed(oracle)
    highest_position = max(
        oracle, key=lambda item: item["initial_position_error_mm"]
    )["start_id"]
    highest_orientation = max(
        oracle, key=lambda item: item["initial_orientation_error_deg"]
    )["start_id"]
    lowest_normalized = min(
        oracle,
        key=lambda item: item["initial_position_error_mm"] / 10.0
        + item["initial_orientation_error_deg"] / 10.0,
    )["start_id"]
    selected = list(dict.fromkeys(["start_000", highest_position, highest_orientation, lowest_normalized]))
    records = []
    with gzip.open(per_step_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item["start_id"] in selected:
                records.append(item)
    payload = {
        "selection_rule": [
            "start_000",
            "largest oracle initial position error",
            "largest oracle initial orientation error",
            "smallest oracle initial (position/10 + orientation/10)",
        ],
        "selected_start_ids": selected,
        "selected_final_outcomes": {
            name: {
                start: {
                    "final_success": indexed(episodes)[start]["final_success"],
                    "final_position_error_mm": indexed(episodes)[start]["final_position_error_mm"],
                    "final_orientation_error_deg": indexed(episodes)[start]["final_orientation_error_deg"],
                }
                for start in selected
            }
            for name, episodes in controllers.items()
        },
        "steps": records,
    }
    (output_dir / "representative_trajectories.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {key: value for key, value in payload.items() if key != "steps"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    evaluation_path = Path(args.evaluation).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    controllers = {
        name: payload["episodes"]
        for name, payload in evaluation["controllers"].items()
    }
    expected_names = {"oracle", "zero_action", "bounded_random", "ppo_pilot"}
    if set(controllers) != expected_names:
        raise RuntimeError("evaluation is missing a frozen controller arm")

    table = {
        name: controller_summary(episodes, 20261110 + index * 10)
        for index, (name, episodes) in enumerate(controllers.items())
    }
    paired = {
        control: paired_summary(controllers["oracle"], controllers[control])
        for control in ("zero_action", "bounded_random", "ppo_pilot")
    }
    metric_correctness = (
        evaluation["status"] == "PASSED"
        and evaluation["paired_replay"]["verified"]
        and evaluation["all_observations_rewards_poses_finite"]
        and all(item["episodes"] == 32 for item in table.values())
        and all(item["invalid_transitions"] == 0 for item in table.values())
    )
    task_feasibility = table["oracle"]["final_success_count"] >= 26
    oracle = table["oracle"]
    discrimination_checks = {}
    for control_name in ("zero_action", "bounded_random"):
        control = table[control_name]
        discrimination_checks[control_name] = {
            "success_advantage_at_least_8": (
                oracle["final_success_count"] - control["final_success_count"] >= 8
            ),
            "lower_median_position_error": (
                oracle["median_position_error_mm"]
                < control["median_position_error_mm"]
            ),
            "lower_median_orientation_error": (
                oracle["median_orientation_error_deg"]
                < control["median_orientation_error_deg"]
            ),
        }
    metric_discrimination = all(
        all(checks.values()) for checks in discrimination_checks.values()
    )
    decision = (
        "METRIC_GO"
        if metric_correctness and task_feasibility and metric_discrimination
        else "METRIC_NO_GO"
    )
    write_episode_csv(output_dir / "episodes.csv", controllers)
    make_plots(output_dir, controllers)
    representatives = representative_trajectories(
        output_dir, Path(evaluation["per_step_path"]), controllers
    )
    summary = {
        "decision": decision,
        "gate": {
            "metric_correctness": metric_correctness,
            "task_feasibility_oracle_at_least_26_of_32": task_feasibility,
            "metric_discrimination": metric_discrimination,
            "discrimination_checks": discrimination_checks,
        },
        "controller_table": table,
        "paired_comparisons": paired,
        "representative_trajectories": representatives,
        "ppo_is_diagnostic_only": True,
        "native_boundary_termination_disabled": True,
        "uncertainty": {
            "success": "Wilson 95% interval",
            "medians": "5000-resample percentile bootstrap 95% interval",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
