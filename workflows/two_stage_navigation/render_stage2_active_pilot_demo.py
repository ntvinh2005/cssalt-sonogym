#!/usr/bin/env python3
"""Render a short actor-first Stage 2 active-acquisition demo from sealed artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PHASE_COLORS = {
    "coarse": (255, 170, 70),
    "fine": (90, 220, 230),
    "micro": (100, 220, 110),
    "fresh_confirmation_1": (220, 120, 240),
    "fresh_confirmation_2": (220, 120, 240),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_seal(path: Path) -> str:
    seal = path.with_suffix(path.suffix + ".sha256")
    expected = seal.read_text(encoding="utf-8").split()[0]
    observed = file_sha256(path)
    if observed != expected:
        raise RuntimeError(f"sealed input changed: {path}")
    return observed


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def text(
    canvas: np.ndarray,
    value: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.48,
    color: tuple[int, int, int] = (225, 225, 225),
    thickness: int = 1,
) -> None:
    cv2.putText(
        canvas,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def map_point(
    pose: list[float],
    x_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
) -> tuple[int, int]:
    x = 560 + (float(pose[0]) - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) * 570
    y = 520 - (float(pose[1]) - z_bounds[0]) / (z_bounds[1] - z_bounds[0]) * 390
    return int(round(x)), int(round(y))


def phase_title(phase: str) -> str:
    return {
        "coarse": "COARSE SEARCH: sample a broad pose grid",
        "fine": "FINE SEARCH: refine around best valid region",
        "micro": "MICRO SEARCH: small local refinements",
        "fresh_confirmation_1": "RETURN + FRESH CONFIRMATION 1",
        "fresh_confirmation_2": "RETURN + FRESH CONFIRMATION 2",
    }[phase]


def render_frame(
    image: np.ndarray,
    rows: list[dict[str, Any]],
    index: int,
    actor: dict[str, Any],
    evaluation: dict[str, Any],
    low: float,
    high: float,
) -> np.ndarray:
    row = rows[index]
    normalized = np.clip((image.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    ultrasound = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_BONE)
    ultrasound = cv2.resize(ultrasound, (500, 650), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((720, 1280, 3), 15, dtype=np.uint8)
    canvas[45:695, :500] = ultrasound
    cv2.rectangle(canvas, (0, 0), (499, 44), (0, 0, 0), -1)
    text(canvas, "SonoGym simulated ultrasound (actor input)", (12, 29), scale=0.58)

    phase = str(row["phase"])
    color = PHASE_COLORS[phase]
    text(canvas, "Stage 2: active best-valid-view acquisition", (535, 31), scale=0.68)
    text(canvas, phase_title(phase), (535, 62), scale=0.52, color=color)
    text(
        canvas,
        f"frame {index:02d}/{len(rows)-1}  producer {row['producer_sequence']}  "
        f"executed actions {row['action_step_count']}",
        (535, 88),
        scale=0.42,
        color=(185, 185, 185),
    )
    text(
        canvas,
        "Actor inputs: ultrasound + executed relative motion; no L4 coordinates or labels",
        (535, 111),
        scale=0.38,
        color=(160, 210, 255),
    )

    poses = [list(map(float, item["relative_pose"])) for item in rows[: index + 1]]
    all_poses = [list(map(float, item["relative_pose"])) for item in rows]
    xs = [pose[0] for pose in all_poses]
    zs = [pose[1] for pose in all_poses]
    x_bounds = (min(xs) - 1.5, max(xs) + 1.5)
    z_bounds = (min(zs) - 1.5, max(zs) + 1.5)
    cv2.rectangle(canvas, (560, 130), (1130, 520), (85, 85, 85), 1)
    text(canvas, "Home-relative spatial search map (mm)", (690, 548), scale=0.46)
    text(canvas, "x translation", (900, 575), scale=0.38, color=(170, 170, 170))
    text(canvas, "z", (540, 145), scale=0.38, color=(170, 170, 170))
    for previous, current in zip(poses, poses[1:]):
        cv2.line(
            canvas,
            map_point(previous, x_bounds, z_bounds),
            map_point(current, x_bounds, z_bounds),
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
    for frame_index, (pose, item) in enumerate(zip(poses, rows[: index + 1])):
        point = map_point(pose, x_bounds, z_bounds)
        point_color = PHASE_COLORS[str(item["phase"])]
        radius = 5 if frame_index == index else 3
        cv2.circle(canvas, point, radius, point_color, -1, cv2.LINE_AA)
        if bool(item["image_valid"]):
            quality_radius = max(2, int(round(3 + 5 * float(item["quality_prediction"]))))
            cv2.circle(canvas, point, quality_radius, point_color, 1, cv2.LINE_AA)

    selected_sequence = int(actor["selected_frame_sequence"])
    if index >= 39:
        selected_point = map_point(
            list(map(float, rows[selected_sequence]["relative_pose"])), x_bounds, z_bounds
        )
        cv2.drawMarker(
            canvas,
            selected_point,
            (255, 255, 255),
            cv2.MARKER_STAR,
            22,
            2,
            cv2.LINE_AA,
        )
        text(canvas, "star = frozen selected candidate", (535, 603), scale=0.38)

    valid = float(row["validity_probability"])
    quality = float(row["quality_prediction"])
    status_color = (90, 230, 120) if bool(row["image_valid"]) else (100, 175, 255)
    text(
        canvas,
        f"current view: validity p={valid:.3f}  predicted quality={quality:.3f}",
        (535, 632),
        scale=0.46,
        color=status_color,
    )
    text(
        canvas,
        f"fresh={str(bool(row['producer_advanced'])).lower()}  "
        f"transition_valid={str(bool(row['transition_valid'])).lower()}  "
        f"hash={str(row['frame_sha256'])[:10]}...",
        (535, 657),
        scale=0.39,
        color=(185, 185, 185),
    )

    if index == len(rows) - 1:
        text(canvas, "ACTOR: ACCEPT", (535, 687), scale=0.55, color=(80, 235, 110), thickness=2)
        text(
            canvas,
            f"Evaluator-only after actor freeze: PASS  {evaluation['final_position_error_mm']:.2f} mm  "
            f"{evaluation['final_orientation_error_deg']:.2f} deg",
            (735, 687),
            scale=0.43,
            color=(90, 210, 255),
        )
    else:
        text(
            canvas,
            "Decision remains causal; evaluator values are hidden until final actor output",
            (535, 687),
            scale=0.38,
            color=(150, 150, 150),
        )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--final-hold-frames", type=int, default=20)
    args = parser.parse_args()
    if args.fps <= 0 or args.final_hold_frames < 0:
        raise ValueError("invalid video timing")

    actor_path = args.run_dir / "results" / "actor_result.json"
    evaluation_path = args.run_dir / "results" / "evaluation_report.json"
    actor_hash = validate_seal(actor_path)
    evaluation_hash = validate_seal(evaluation_path)
    actor = read_json(actor_path)
    evaluation = read_json(evaluation_path)
    if actor["outcome"] != "ACCEPT" or evaluation["decision"] != "DEVELOPMENT_PASS":
        raise RuntimeError("refusing to render a non-passing development result")

    rows = read_jsonl(args.run_dir / "public" / "frames.jsonl")
    images = np.load(args.run_dir / "public" / "ultrasound.npz")["ultrasound"]
    if len(rows) != len(images) or len(rows) != int(actor["total_frame_count"]):
        raise RuntimeError("frame artifacts do not align")
    if [int(row["frame_sequence"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError("frame sequence is not contiguous")
    low, high = np.quantile(images, [0.01, 0.99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise RuntimeError("invalid ultrasound display range")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video = args.output_dir / "sonogym-stage2-active-acquisition-demo.mp4"
    preview = args.output_dir / "sonogym-stage2-active-acquisition-final.png"
    selected_preview = args.output_dir / "sonogym-stage2-selected-candidate.png"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (1280, 720)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the MP4 demo")
    canvases: dict[int, np.ndarray] = {}
    try:
        for index, image in enumerate(images):
            canvas = render_frame(image, rows, index, actor, evaluation, float(low), float(high))
            writer.write(canvas)
            if index in {int(actor["selected_frame_sequence"]), len(rows) - 1}:
                canvases[index] = canvas.copy()
        for _ in range(args.final_hold_frames):
            writer.write(canvases[len(rows) - 1])
    finally:
        writer.release()
    if not cv2.imwrite(str(preview), canvases[len(rows) - 1]):
        raise RuntimeError("could not write final preview")
    if not cv2.imwrite(str(selected_preview), canvases[int(actor["selected_frame_sequence"])]):
        raise RuntimeError("could not write selected-candidate preview")

    capture = cv2.VideoCapture(str(video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    expected_count = len(rows) + args.final_hold_frames
    if frame_count != expected_count:
        raise RuntimeError(f"video frame count mismatch: {frame_count} != {expected_count}")
    manifest = {
        "schema": "sonogym-stage2-active-pilot-demo-v1",
        "source_run": str(args.run_dir.resolve()),
        "actor_result_sha256": actor_hash,
        "evaluation_report_sha256": evaluation_hash,
        "actor_outcome": actor["outcome"],
        "development_decision": evaluation["decision"],
        "frame_count": frame_count,
        "source_observation_count": len(rows),
        "fps": args.fps,
        "duration_seconds": frame_count / args.fps,
        "video": str(video.resolve()),
        "video_sha256": file_sha256(video),
        "preview": str(preview.resolve()),
        "preview_sha256": file_sha256(preview),
        "selected_preview": str(selected_preview.resolve()),
        "selected_preview_sha256": file_sha256(selected_preview),
        "fixed_anatomy_development_only": True,
        "independent_confirmation": False,
    }
    manifest_path = args.output_dir / "demo_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
