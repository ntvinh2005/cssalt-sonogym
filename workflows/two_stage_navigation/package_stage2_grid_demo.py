#!/usr/bin/env python3
"""Package the two-video physical-grid demo with source evidence and checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import cv2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_metadata(path: Path) -> dict[str, float | int | str]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if frame_count <= 0 or fps <= 0.0 or width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video metadata: {path}")
    return {
        "file": path.name,
        "sha256": sha256(path),
        "frame_count": frame_count,
        "fps": fps,
        "duration_seconds": frame_count / fps,
        "width": width,
        "height": height,
    }


def extract_preview(video: Path, output: Path, frame_index: int) -> None:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None or not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"could not extract preview frame {frame_index} from {video}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rendered-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()

    actor = json.loads((args.run_dir / "results" / "actor_result.json").read_text())
    evaluator = json.loads((args.run_dir / "results" / "evaluation_report.json").read_text())
    if actor["outcome"] != "ACCEPT" or evaluator["decision"] not in {
        "DEVELOPMENT_PASS",
        "DEVELOPMENT_NO_GO",
    }:
        raise RuntimeError("unexpected source result; package requires a completed actor ACCEPT")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    map_video = args.output_dir / "01-ultrasound-spatial-grid-demo.mp4"
    robot_video = args.output_dir / "02-robot-probe-patient-camera.mp4"
    shutil.copy2(
        args.rendered_dir / "sonogym-stage2-active-acquisition-demo.mp4",
        map_video,
    )
    shutil.copy2(args.run_dir / "video" / "sonogym-robot-probe-patient.mp4", robot_video)
    shutil.copy2(
        args.rendered_dir / "sonogym-stage2-active-acquisition-final.png",
        args.output_dir / "01-grid-final.png",
    )
    robot_meta = video_metadata(robot_video)
    extract_preview(
        robot_video,
        args.output_dir / "02-robot-camera-midpoint.png",
        int(robot_meta["frame_count"]) // 2,
    )
    evidence_dir = args.output_dir / "evidence"
    evidence_dir.mkdir()
    protocol_paths = sorted(args.run_dir.glob("context/*protocol*.yaml"))
    if len(protocol_paths) != 1:
        raise RuntimeError("run context must contain exactly one protocol YAML")
    for source, destination in (
        (args.run_dir / "results" / "actor_result.json", "actor_result.json"),
        (args.run_dir / "results" / "evaluation_report.json", "evaluation_report.json"),
        (args.run_dir / "summary.json", "run_summary.json"),
        (protocol_paths[0], "protocol.yaml"),
        (args.run_dir / "context" / "slurm-job.txt", "slurm-job.txt"),
        (args.run_dir / "context" / "application-exit-codes.env", "exit-codes.env"),
        (args.run_dir / "metrics" / "slurm.csv", "resource-usage.csv"),
    ):
        shutil.copy2(source, evidence_dir / destination)

    videos = [video_metadata(map_video), robot_meta]
    manifest = {
        "schema": "sonogym-stage2-physical-grid-demo-package-v1",
        "source_job_id": int(args.run_dir.name),
        "actor_outcome": actor["outcome"],
        "evaluator_decision": evaluator["decision"],
        "fixed_anatomy": "s0010",
        "position_grid": "3x3 positions, 6 mm x 6 mm spacing",
        "angles_per_position": 3,
        "angle_spacing_deg": actor["physical_grid"]["minimum_angle_spacing_deg"],
        "coarse_pose_count": actor["physical_grid"]["pose_cell_count"],
        "final_position_error_mm": evaluator["final_position_error_mm"],
        "final_orientation_error_deg": evaluator["final_orientation_error_deg"],
        "videos": videos,
        "scientific_limit": "Development visualization on fixed s0010; not independent confirmation.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "# SonoGym Stage 2 physical-grid demo\n\n"
        "- `01-ultrasound-spatial-grid-demo.mp4`: simulated ultrasound, commanded 3x3 grid, "
        "realized trajectory, coarse/fine/micro selection and fresh confirmation.\n"
        "- `02-robot-probe-patient-camera.mp4`: Isaac Sim rendered visualization of the robot "
        "moving the probe on the complete patient asset available in this task. The asset is a "
        "cropped torso, not a full-body anatomical model.\n\n"
        "The actor returned ACCEPT and all 27 coarse poses produced fresh, distinct images. "
        f"The independent evaluator returned {evaluator['decision']}: position error was "
        f"{evaluator['final_position_error_mm']:.3f} mm, while orientation error was "
        f"{evaluator['final_orientation_error_deg']:.3f} degrees (gate: <=10 degrees). "
        "Do not interpret this package as scientific Stage 2 PASS or clinical evidence.\n",
        encoding="utf-8",
    )

    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, Path(args.output_dir.name) / path.relative_to(args.output_dir))
    print(json.dumps({"zip": str(args.zip), "sha256": sha256(args.zip), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
