#!/usr/bin/env python3
"""CPU-safe SonoGym installation, configuration, and asset checks."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


EXPECTED_PACKAGES = {
    "isaacsim": "4.5.0.0",
    "torch": "2.5.1",
    "torchvision": "0.20.1",
    "skrl": "1.4.2",
    "numpy": "1.26.4",
}


def load_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip().strip("'\"")
    return result


def load_yaml(path: Path):
    try:
        from ruamel.yaml import YAML

        with path.open("r", encoding="utf-8") as stream:
            return YAML(typ="safe").load(stream)
    except ImportError:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=".sonogym.env")
    parser.add_argument("--static", action="store_true", help="Only validate repository files and baseline YAML")
    parser.add_argument("--json-output")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    checks: list[dict[str, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "status": "PASSED" if ok else "FAILED", "detail": detail})

    baseline = project / "config" / "guidance_baseline.yaml"
    try:
        cfg = load_yaml(baseline)
        check("baseline_yaml", True, str(baseline))
        check("model_based_ultrasound", cfg["sim"]["us"] == "conv", str(cfg["sim"]["us"]))
        check(
            "visualization_disabled",
            cfg["sim"]["vis_us"] is False and cfg["sim"]["vis_seg_map"] is False,
            f"vis_us={cfg['sim']['vis_us']} vis_seg_map={cfg['sim']['vis_seg_map']}",
        )
        check("trajectory_recording_disabled", cfg["if_record_traj"] is False, str(cfg["if_record_traj"]))
    except Exception as exc:
        check("baseline_yaml", False, f"{type(exc).__name__}: {exc}")

    check("smoke_script", (project / "scripts" / "smoke_test.py").is_file(), "real environment smoke script")
    check("ppo_job", (project / "hpg" / "jobs" / "ppo.sbatch").is_file(), "bounded PPO pilot")
    check("upstream_license", (project / "LICENCE").is_file(), "LICENCE retained")

    if not args.static:
        env_file = Path(args.config).resolve()
        values = load_env(env_file)
        for key in (
            "SONOGYM_PROJECT_DIR",
            "SONOGYM_ASSETS_DATA_DIR",
            "SONOGYM_MODELS_DIR",
            "SONOGYM_OUTPUT_DIR",
        ):
            check(f"config_{key.lower()}", bool(values.get(key)), values.get(key, "missing"))
        check("slurm_account", values.get("SLURM_ACCOUNT") == "anest-dept", values.get("SLURM_ACCOUNT", "missing"))
        check("slurm_partition", values.get("SLURM_PARTITION") == "hpg-turin", values.get("SLURM_PARTITION", "missing"))
        check("cpu_budget", int(values.get("SLURM_CPUS", "999")) == 4, values.get("SLURM_CPUS", "missing"))
        check("gpu_budget", int(values.get("SLURM_GPUS", "999")) == 1, values.get("SLURM_GPUS", "missing"))

        assets = Path(values.get("SONOGYM_ASSETS_DATA_DIR", "/missing"))
        required_assets = [
            "Robots/Kuka/med14/kuka_US.usd",
            "HumanModels/selected_dataset_body_from_urdf/s0010/combined_wrapwrap/combined_wrapwrap.usd",
            "HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz",
            "HumanModels/selected_dataset_stl/s0010/vertebrae_L4.stl",
            "HumanModels/selected_dataset_stl/s0010/standard_right_traj_L4.stl",
            "HumanModels/selected_dataset/s0010/ct.nii.gz",
            "MedicalBed/usd_no_contact/hospital_bed.usd",
        ]
        for relative in required_assets:
            check(f"asset:{relative}", (assets / relative).is_file(), str(assets / relative))
        models = Path(values.get("SONOGYM_MODELS_DIR", "/missing"))
        check("published_model", (models / "pix2pix_rand_down_up.pth").is_file(), str(models))
        check("training_samples", (models / "training_samples").is_dir(), str(models / "training_samples"))

        for distribution, expected in EXPECTED_PACKAGES.items():
            try:
                actual = importlib.metadata.version(distribution)
                ok = actual == expected or (distribution == "isaacsim" and actual.startswith("4.5.0"))
                check(f"package:{distribution}", ok, f"expected={expected} actual={actual}")
            except importlib.metadata.PackageNotFoundError:
                check(f"package:{distribution}", False, "not installed")
        pip_check = subprocess.run(
            [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
        )
        check("pip_check", pip_check.returncode == 0, (pip_check.stdout + pip_check.stderr).strip())

    report = {
        "host": platform.node(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "glibc": " ".join(platform.libc_ver()),
        "gpu_checks": "NOT_RUN_NO_GPU",
        "checks": checks,
        "status": "PASSED" if all(item["status"] == "PASSED" for item in checks) else "FAILED",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        Path(args.json_output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
