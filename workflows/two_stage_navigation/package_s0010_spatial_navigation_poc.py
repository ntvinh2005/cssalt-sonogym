#!/usr/bin/env python3
"""Package the s0010 spatial-mapping result and historical navigation replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--historical-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_video_frame(video: Path, path: Path, fraction: float) -> None:
    capture = cv2.VideoCapture(str(video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(fraction * max(count - 1, 0)))
    ok, frame = capture.read()
    capture.release()
    if not ok or not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"cannot extract video frame: {fraction}")


def main() -> int:
    args = parse_args()
    result = json.loads((args.analysis / "result.json").read_text())
    demo_manifest = json.loads(args.demo.with_suffix(".json").read_text())
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    files = {
        "REPORT.md": args.report,
        "result.json": args.analysis / "result.json",
        "mapping_metrics.csv": args.analysis / "mapping_metrics.csv",
        "spatial_mapping_summary.png": args.analysis / "spatial_mapping_summary.png",
        "protocol.yaml": args.protocol,
        "protocol.sha256": args.protocol.with_suffix(".sha256"),
        "historical_poc_result.json": args.historical_result,
        "s0010_spatial_navigation_replay.mp4": args.demo,
        "s0010_spatial_navigation_replay.json": args.demo.with_suffix(".json"),
    }
    for name, source in files.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output / name)
    save_video_frame(args.demo, output / "L5_SUCCESS_REPLAY.png", 0.25)
    save_video_frame(args.demo, output / "L4_FAILURE_REPLAY.png", 0.75)

    start = f"""# START HERE

## What this package demonstrates

- The historical closed-loop run completed Stage 1 and RIGHT L5 TP target-view acquisition, physical return, and fresh confirmation in 4/4 sessions.
- `s0010_spatial_navigation_replay.mp4` replays one verified historical L5 success and one representative historical L4 failure.
- Panel 1 contains clean ultrasound plus actor cortical candidates.
- Panel 2 contains Motion-compensated spatial cortical mapping from actor-visible pose feedback.
- Panel 3 contains actual recorded robot/probe motion.
- Panel 4 contains privileged evaluator evidence and is not available to the actor.

## What failed

- The new two-target mapping gate returned `{result['phase_a_verdict']}`.
- L5 passed both selection-session records; L4 failed its p90 or false-association gate in both.
- Phase B was not run, the final-comparison sessions remained unopened by the corrected runner, and no GPU job was submitted.

## Evidence status

- Video mode: `{demo_manifest['mode']}`.
- New map evidence: historically opened development replay, not independent confirmation.
- Highest verified milestone: `NAVIGATION_L5_ONLY_PASS`, from job `52039283`.
- `FULL_MBB_INTERVENTION = BLOCKED` because no validated MBB target-region or needle/injection contract exists.

Read `REPORT.md` for gates, commands, metrics, provenance, and limitations.
"""
    (output / "START_HERE.md").write_text(start)
    provenance = {
        "schema": "s0010-spatial-navigation-poc-package-v1",
        "source_commit": args.source_commit,
        "analysis_job": 52246037,
        "demo_job": 52246624,
        "historical_closed_loop_job": 52039283,
        "phase_a_verdict": result["phase_a_verdict"],
        "phase_b_verdict": result["phase_b_verdict"],
        "gpu_submitted": False,
        "highest_verified_milestone": "NAVIGATION_L5_ONLY_PASS",
        "evidence_basis": "previously_recorded_closed_loop_replay",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    members = sorted(path for path in output.iterdir() if path.is_file())
    checksum_lines = [f"{sha256(path)}  {path.name}" for path in members]
    (output / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")
    zip_path = output.parent / "s0010_spatial_navigation_poc.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            archive.write(path, arcname=f"s0010_spatial_navigation_poc/{path.name}")
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
    summary = {
        "source_directory": str(output),
        "zip_path": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256(zip_path),
        "member_count": len(list(output.iterdir())),
    }
    (output.parent / "package_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
