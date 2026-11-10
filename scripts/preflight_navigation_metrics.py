#!/usr/bin/env python3
"""CPU-only fail-closed preflight for Phase 5A navigation validation."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path

from ruamel.yaml import YAML


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="experiments/navigation_metric_validation/protocol.yaml",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--json-output")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    protocol_path = (project / args.protocol).resolve()
    protocol = YAML(typ="safe").load(protocol_path.read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: object) -> None:
        checks.append(
            {
                "name": name,
                "status": "PASSED" if condition else "FAILED",
                "detail": detail,
            }
        )

    manifest_path = project / protocol["initial_conditions"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check(
        "manifest_sha256",
        sha256_file(manifest_path) == protocol["initial_conditions"]["sha256"],
        sha256_file(manifest_path),
    )
    check(
        "exactly_32_starts",
        manifest.get("count") == 32 and len(manifest.get("starts", [])) == 32,
        manifest.get("count"),
    )
    start_ids = [item["start_id"] for item in manifest["starts"]]
    check("unique_ordered_start_ids", start_ids == [f"start_{i:03d}" for i in range(32)], start_ids)
    checkpoint_path = Path(args.checkpoint or protocol["ppo_checkpoint"]["path"])
    check("checkpoint_exists", checkpoint_path.is_file(), str(checkpoint_path))
    if checkpoint_path.is_file():
        check(
            "checkpoint_sha256",
            sha256_file(checkpoint_path) == protocol["ppo_checkpoint"]["sha256"],
            sha256_file(checkpoint_path),
        )
    guidance_path = project / protocol["task"]["guidance_config"]
    check(
        "guidance_config_sha256",
        sha256_file(guidance_path) == protocol["task"]["guidance_config_sha256"],
        sha256_file(guidance_path),
    )
    source = (
        project
        / "source/spinal_surgery/spinal_surgery/tasks/robot_US_guidance/robotic_US_guidance.py"
    ).read_text(encoding="utf-8")
    check(
        "replay_hook_present",
        "def set_evaluation_initial_commands" in source
        and "cmd_target_poses = self._evaluation_initial_commands.clone()" in source,
        "evaluation-only reset injection before native USSlicer clamp/IK",
    )
    check(
        "frozen_horizon",
        protocol["execution"]["max_action_steps"] == 300,
        protocol["execution"]["max_action_steps"],
    )
    resources = protocol["resources"]
    check(
        "resource_budget",
        resources["account"] == "anest-dept"
        and resources["gpu_count"] == 1
        and resources["gpu_model"] == "NVIDIA L4"
        and resources["cpu_cores"] == 4
        and resources["memory"] == "30gb"
        and resources["walltime"] == "02:00:00",
        resources,
    )

    loader = unittest.TestLoader()
    suite = loader.discover(str(project / "tests"), pattern="test_navigation_metrics.py")
    test_output = io.StringIO()
    result = unittest.TextTestRunner(stream=test_output, verbosity=2).run(suite)
    check(
        "metric_unit_tests",
        result.wasSuccessful() and result.testsRun == 11,
        {
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "output": test_output.getvalue(),
        },
    )
    report = {
        "status": "PASSED" if all(item["status"] == "PASSED" for item in checks) else "FAILED",
        "gpu_used": False,
        "protocol": str(protocol_path),
        "checks": checks,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    sys.stdout.write(rendered)
    if args.json_output:
        destination = Path(args.json_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
