#!/usr/bin/env python3
"""Summarize a bounded skrl PPO pilot and structurally inspect its checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_elapsed_seconds(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"Elapsed \(wall clock\) time .*: ([0-9:.]+)", text)
    if not match:
        raise ValueError(f"elapsed wall time is missing from {path}")
    parts = [float(part) for part in match.group(1).split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"unexpected elapsed format: {match.group(1)}")


def scalar_metrics(log_root: Path) -> tuple[dict[str, dict[str, float | int]], list[str]]:
    metrics: dict[str, list[float]] = {}
    event_files = sorted(log_root.rglob("events.out.tfevents.*"))
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            metrics.setdefault(tag, []).extend(event.value for event in accumulator.Scalars(tag))
    summary = {
        tag: {
            "count": len(values),
            "first": values[0],
            "last": values[-1],
            "minimum": min(values),
            "maximum": max(values),
        }
        for tag, values in sorted(metrics.items())
        if values
    }
    return summary, [str(path) for path in event_files]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--vector-steps", type=int, required=True)
    parser.add_argument("--transitions", type=int, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    checkpoints = sorted(
        path
        for pattern in ("*.pt", "*.pth", "*.ckpt")
        for path in (run_dir / "logs").rglob(pattern)
    )
    if not checkpoints:
        raise RuntimeError("PPO pilot finished without a checkpoint")
    checkpoint = checkpoints[-1]
    checkpoint_object = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_keys = sorted(checkpoint_object) if isinstance(checkpoint_object, dict) else []

    scalars, event_files = scalar_metrics(run_dir / "logs")
    finite = all(
        math.isfinite(float(value))
        for metric in scalars.values()
        for key, value in metric.items()
        if key != "count"
    )
    loss_tags = [tag for tag in scalars if "loss" in tag.lower()]
    reward_tags = [tag for tag in scalars if "reward" in tag.lower()]
    elapsed_seconds = parse_elapsed_seconds(run_dir / "metrics" / "time.txt")
    report = {
        "status": "PASSED" if finite and loss_tags else "FAILED",
        "configured_ppo_updates": args.updates,
        "completed_vector_steps": args.vector_steps,
        "collected_transitions": args.transitions,
        "elapsed_seconds": elapsed_seconds,
        "transitions_per_second": args.transitions / elapsed_seconds,
        "nonfinite_scalar_values": not finite,
        "event_files": event_files,
        "loss_tags": loss_tags,
        "reward_tags": reward_tags,
        "scalar_metrics": scalars,
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "loaded_on_cpu": True,
            "object_type": type(checkpoint_object).__name__,
            "top_level_keys": checkpoint_keys,
        },
    }
    (run_dir / "pilot-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASSED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
