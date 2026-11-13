#!/usr/bin/env python3
"""Render one frozen confirmation scan without altering actor or evaluator output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_seal(path: Path) -> str:
    expected = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    observed = file_sha256(path)
    if observed != expected:
        raise RuntimeError(f"sealed input changed: {path}")
    return observed


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def chart_x(value: float, minimum: float, maximum: float) -> int:
    if maximum <= minimum:
        return 530
    return int(round(515 + (value - minimum) / (maximum - minimum) * 500))


def chart_y(value: float) -> int:
    return int(round(505 - np.clip(value, 0.0, 1.0) * 350))


def canvas_frame(
    image: np.ndarray,
    public_row: dict[str, Any],
    evidence: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    final_actor: dict[str, Any],
    evaluator: dict[str, Any],
    low: float,
    high: float,
) -> np.ndarray:
    normalized = np.clip((image.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    ultrasound = (normalized * 255.0).astype(np.uint8)
    ultrasound = cv2.applyColorMap(ultrasound, cv2.COLORMAP_BONE)
    ultrasound = cv2.resize(ultrasound, (450, 600), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((600, 1050, 3), dtype=np.uint8)
    canvas[:, :450] = ultrasound
    cv2.rectangle(canvas, (0, 0), (449, 42), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        "Clean SonoGym ultrasound (actor input)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    sequence = int(public_row["frame_sequence"])
    phase = str(public_row["phase"])
    coordinate = float(public_row["relative_position_home_mm"][1])
    current_evidence = [row for row in evidence if int(row["frame_sequence"]) <= sequence]
    coordinates = [float(row["coordinate_mm"]) for row in evidence]
    minimum = min(coordinates) - 5.0
    maximum = max(coordinates) + 5.0
    cv2.putText(canvas, f"Phase 1 scan: {phase}", (480, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Frame {sequence:03d}/210  executed z={coordinate:+.1f} mm", (480, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (515, 155), (1015, 505), (120, 120, 120), 1)
    threshold_y = chart_y(0.5)
    cv2.line(canvas, (515, threshold_y), (1015, threshold_y), (100, 100, 100), 1)
    cv2.putText(canvas, "0.50 frozen threshold", (820, threshold_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
    for row in current_evidence:
        color = (255, 150, 40) if row["direction"] == "FORWARD" else (40, 190, 255)
        cv2.circle(
            canvas,
            (
                chart_x(float(row["coordinate_mm"]), minimum, maximum),
                chart_y(float(row["combined_probability"])),
            ),
            1,
            color,
            -1,
        )
    current = max(
        (row for row in timeline if int(row["frame_sequence"]) <= sequence),
        key=lambda row: int(row["frame_sequence"]),
        default={"status": "ABSTAIN", "coordinate_mm": None, "reason": "NO_EVIDENCE"},
    )
    if current["status"] == "SUCCESS":
        actor_x = chart_x(float(current["coordinate_mm"]), minimum, maximum)
        cv2.line(canvas, (actor_x, 145), (actor_x, 515), (60, 230, 60), 2)
    reference = float(evaluator["reference_coordinate_mm_evaluator_only"])
    evaluator_x = chart_x(reference, minimum, maximum)
    cv2.line(canvas, (evaluator_x, 145), (evaluator_x, 515), (60, 60, 240), 1, cv2.LINE_AA)
    status_text = (
        f"Actor: SUCCESS {float(current['coordinate_mm']):+.1f} mm"
        if current["status"] == "SUCCESS"
        else f"Actor: ABSTAIN ({current.get('reason') or 'insufficient evidence'})"
    )
    cv2.putText(canvas, status_text, (480, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (60, 230, 60) if current["status"] == "SUCCESS" else (70, 190, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Evaluator-only L4 reference: {reference:+.2f} mm", (480, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 100, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Combined image evidence", (515, 545), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "cranial/caudal Home-relative coordinate (mm)", (650, 576), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)
    final_status = str(final_actor["actor_status"])
    cv2.putText(canvas, f"Frozen final actor output: {final_status}", (480, 594), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-protocol", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--actor-result", type=Path, required=True)
    parser.add_argument("--evaluation-result", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-preview", type=Path, required=True)
    args = parser.parse_args()
    validate_seal(args.actor_protocol)
    validate_seal(args.actor_result)
    validate_seal(args.evaluation_result)
    protocol = yaml.safe_load(args.actor_protocol.read_text(encoding="utf-8"))
    actor = read_json(args.actor_result)
    evaluation = read_json(args.evaluation_result)
    session_id = str(protocol["demo"]["session_id"])
    actor_session = next(row for row in actor["results"] if row["session_id"] == session_id)
    evaluator_session = next(row for row in evaluation["sessions"] if row["session_id"] == session_id)
    session_root = args.capture_root / "sessions" / session_id
    images = np.load(session_root / "public" / "ultrasound.npz")["ultrasound"]
    public_rows = read_jsonl(session_root / "public" / "frames.jsonl")
    low, high = np.quantile(images, [0.01, 0.99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise RuntimeError("invalid ultrasound display range")
    fps = int(protocol["demo"]["frames_per_second"])
    duration = len(images) / fps
    if duration > float(protocol["demo"]["maximum_duration_seconds"]):
        raise RuntimeError("demo duration exceeds frozen bound")
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (1050, 600),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the MP4 demo")
    final_canvas = None
    try:
        for image, public_row in zip(images, public_rows):
            final_canvas = canvas_frame(
                image,
                public_row,
                actor_session["evidence"],
                actor_session["timeline"],
                actor_session,
                evaluator_session,
                float(low),
                float(high),
            )
            writer.write(final_canvas)
    finally:
        writer.release()
    if final_canvas is None or not args.output_video.is_file() or args.output_video.stat().st_size == 0:
        raise RuntimeError("empty demo output")
    if not cv2.imwrite(str(args.output_preview), final_canvas):
        raise RuntimeError("could not write demo preview")
    check = cv2.VideoCapture(str(args.output_video))
    frame_count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()
    if frame_count != len(images):
        raise RuntimeError(f"video frame count mismatch: {frame_count} != {len(images)}")
    print(
        json.dumps(
            {
                "session_id": session_id,
                "video": str(args.output_video),
                "video_sha256": file_sha256(args.output_video),
                "frame_count": frame_count,
                "fps": fps,
                "duration_seconds": duration,
                "preview": str(args.output_preview),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
