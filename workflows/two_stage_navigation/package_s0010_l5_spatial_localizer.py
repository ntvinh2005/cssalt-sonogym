#!/usr/bin/env python3
"""Package the L5 spatial-localizer result, replay demo, and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-run", type=Path, required=True)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    args = parse_args()
    result = json.loads((args.analysis_run / "analysis/result.json").read_text())
    root = args.output_dir / "s0010_l5_spatial_localizer_review"
    if root.exists():
        raise RuntimeError(f"refusing existing package directory: {root}")
    (root / "analysis").mkdir(parents=True)
    (root / "demo").mkdir()
    (root / "method").mkdir()

    for name in (
        "result.json",
        "geometry_validation.json",
        "session_results.csv",
        "frame_rankings.csv",
        "localizer_summary.png",
    ):
        copy(args.analysis_run / "analysis" / name, root / "analysis" / name)
    copy(args.analysis_run / "tests.log", root / "analysis/tests.log")
    copy(
        args.demo_dir / "s0010_l5_spatial_localizer_demo.mp4",
        root / "demo/s0010_l5_spatial_localizer_demo.mp4",
    )
    copy(args.demo_dir / "demo_manifest.json", root / "demo/demo_manifest.json")
    copy(
        args.project_dir
        / "experiments/unity_two_stage_transfer/s0010_l5_spatial_localizer_protocol_v1.yaml",
        root / "method/s0010_l5_spatial_localizer_protocol_v1.yaml",
    )
    copy(
        args.project_dir
        / "experiments/unity_two_stage_transfer/s0010_l5_spatial_localizer_protocol_v1.sha256",
        root / "method/s0010_l5_spatial_localizer_protocol_v1.sha256",
    )
    copy(
        args.project_dir / "docs/research/sonogym_s0010_spatial_navigation_poc.md",
        root / "REPORT.md",
    )

    start = f"""# START HERE

This package shows an **offline replay** of the RIGHT L5 TP surface-core spatial localizer on historically opened fixed-`s0010` development sessions.

## What to open

1. `demo/s0010_l5_spatial_localizer_demo.mp4`
2. `analysis/localizer_summary.png`
3. `REPORT.md`, section 3

## Result

- spatial reconstruction: `{result['spatial_reconstruction_verdict']}`
- L5 localization: `{result['l5_target_localization_verdict']}`
- Stage 2 integration: `{result['integration_verdict']}`
- selected hypothesis: `{result['selected_hypothesis']}`
- new closed-loop execution: **no**

The actor panels use clean ultrasound, executed reduced motion, the Stage 1 prior, and public calibration. The fourth video panel is explicitly privileged evaluator evidence. The reserved final-comparison sessions were not opened because neither hypothesis passed the method-selection gate.

This package does not establish SAP, MBB-target, needle, injection, clinical, or patient-generalization validity.
"""
    (root / "START_HERE.md").write_text(start)
    commit = subprocess.check_output(
        ["git", "-C", str(args.project_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    provenance = {
        "schema": "s0010-l5-spatial-localizer-review-v1",
        "commit": commit,
        "analysis_job": args.analysis_run.name,
        "analysis_run": str(args.analysis_run.resolve()),
        "source_run": result["source_run"],
        "protocol_sha256": result["protocol_sha256"],
        "evidence_status": result["evidence_status"],
        "selected_hypothesis": result["selected_hypothesis"],
        "new_closed_loop_execution": False,
        "final_comparison_opened": result["final_summary"] is not None,
    }
    (root / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")

    members = sorted(path for path in root.rglob("*") if path.is_file())
    checksum_lines = [f"{sha256(path)}  {path.relative_to(root)}" for path in members]
    (root / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")
    archive = args.output_dir / "s0010_l5_spatial_localizer_review.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                handle.write(path, Path(root.name) / path.relative_to(root))
    summary = {
        "package_directory": str(root.resolve()),
        "zip": str(archive.resolve()),
        "zip_size_bytes": archive.stat().st_size,
        "zip_sha256": sha256(archive),
        "member_count": sum(1 for path in root.rglob("*") if path.is_file()),
    }
    (args.output_dir / "package_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
