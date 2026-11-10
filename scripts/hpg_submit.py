#!/usr/bin/env python3
"""Preview or submit the serial CSSALT SonoGym jobs."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_KEYS = {
    "SONOGYM_PROJECT_DIR",
    "SONOGYM_RUNTIME_DIR",
    "SONOGYM_ENV_PREFIX",
    "SONOGYM_OUTPUT_DIR",
    "SLURM_ACCOUNT",
    "SLURM_PARTITION",
    "SLURM_CPUS",
    "SLURM_GPUS",
    "SLURM_MEMORY",
}


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    missing = sorted(REQUIRED_KEYS - values.keys())
    if missing:
        raise ValueError(f"missing keys in {path}: {', '.join(missing)}")
    return values


def validate_budget(cfg: dict[str, str]) -> None:
    if cfg["SLURM_ACCOUNT"] != "anest-dept":
        raise ValueError("refusing to charge a Slurm account other than anest-dept")
    if int(cfg["SLURM_CPUS"]) != 4 or int(cfg["SLURM_GPUS"]) != 1:
        raise ValueError("initial jobs must request exactly 4 CPUs and 1 GPU")
    if cfg["SLURM_PARTITION"] != "hpg-turin":
        raise ValueError("initial pinned stack is restricted to the L4 hpg-turin partition")


def join_configured_path(base: str, *parts: str) -> str:
    """Join a configured path without reinterpreting POSIX Blue paths on Windows."""
    separator = "/" if base.startswith("/") else os.sep
    return separator.join([base.rstrip("/\\"), *parts])


def check_project_queue(allowed_predecessor: int | None = None) -> None:
    if not shutil.which("squeue"):
        raise RuntimeError("squeue is unavailable; submission must run on HiPerGator")
    result = subprocess.run(
        ["squeue", "--me", "--noheader", "--account", "anest-dept", "--format", "%i|%T|%j"],
        check=True,
        capture_output=True,
        text=True,
    )
    project_jobs = [
        line for line in result.stdout.splitlines() if line.split("|")[-1].startswith("cssalt-sonogym-")
    ]
    unexpected_jobs = [
        line
        for line in project_jobs
        if allowed_predecessor is None or line.split("|", 1)[0] != str(allowed_predecessor)
    ]
    if unexpected_jobs:
        raise RuntimeError(
            "another CSSALT SonoGym job is pending/running; the initial workflow is serial:\n"
            + "\n".join(unexpected_jobs)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "job",
        choices=[
            "smoke",
            "ppo",
            "evaluation",
            "demo",
            "navigation_metrics",
            "navigation_metrics_v2",
        ],
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--submit", action="store_true")
    parser.add_argument("--config", default=".sonogym.env")
    parser.add_argument(
        "--afterany",
        type=int,
        help="Submit after the specified Slurm job finishes, preserving serial GPU use.",
    )
    args = parser.parse_args()
    if args.afterany is not None and args.afterany <= 0:
        parser.error("--afterany must be a positive Slurm job ID")

    config_path = Path(args.config).resolve()
    cfg = load_env(config_path)
    validate_budget(cfg)

    walltime_keys = {
        "smoke": ("SMOKE_WALLTIME", "00:30:00"),
        "ppo": ("PPO_WALLTIME", "01:30:00"),
        "evaluation": ("EVAL_WALLTIME", "00:30:00"),
        "demo": ("DEMO_WALLTIME", "00:15:00"),
        "navigation_metrics": ("NAVIGATION_METRICS_WALLTIME", "02:00:00"),
        "navigation_metrics_v2": ("NAVIGATION_METRICS_V2_WALLTIME", "02:00:00"),
    }
    walltime_key, default_walltime = walltime_keys[args.job]
    walltime = cfg.get(walltime_key, default_walltime)
    batch_script = join_configured_path(
        cfg["SONOGYM_PROJECT_DIR"], "hpg", "jobs", f"{args.job}.sbatch"
    )
    log_dir = join_configured_path(cfg["SONOGYM_OUTPUT_DIR"], "slurm")
    job_name = f"cssalt-sonogym-{args.job}"

    command = [
        "sbatch",
        "--parsable",
        f"--account={cfg['SLURM_ACCOUNT']}",
        f"--partition={cfg['SLURM_PARTITION']}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cfg['SLURM_CPUS']}",
        f"--gpus={cfg['SLURM_GPUS']}",
        f"--mem={cfg['SLURM_MEMORY']}",
        f"--time={walltime}",
        *([f"--dependency=afterany:{args.afterany}"] if args.afterany else []),
        f"--job-name={job_name}",
        f"--output={join_configured_path(log_dir, '%x-%j.out')}",
        f"--export=ALL,SONOGYM_CONFIG_FILE={config_path}",
        batch_script,
    ]

    print("Resolved resource request:")
    print(
        f"  account={cfg['SLURM_ACCOUNT']} partition={cfg['SLURM_PARTITION']} "
        f"nodes=1 tasks=1 cpus={cfg['SLURM_CPUS']} gpus={cfg['SLURM_GPUS']} "
        f"memory={cfg['SLURM_MEMORY']} walltime={walltime}"
    )
    print("Resolved command:")
    print("  " + shlex.join(command))
    if args.dry_run:
        print("DRY_RUN: no scheduler command was executed")
        return 0

    if os.name == "nt":
        raise RuntimeError("submission must run on HiPerGator, not Windows")
    check_project_queue(args.afterany)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print(f"Submitted job {result.stdout.strip()}")
    print("This helper only serializes jobs named cssalt-sonogym-*; it cannot police unrelated jobs.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
