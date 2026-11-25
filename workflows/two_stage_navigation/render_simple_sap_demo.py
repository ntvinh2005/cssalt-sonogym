#!/usr/bin/env python3
"""Render a development sweep from synchronized robot and ultrasound acquisitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np
import yaml

from render_simple_landmark_mapping import COLORS, actor_figure, draw_map, load, overlay, text
from run_simple_landmark_mapping import load_public, write_json
from two_stage_navigation.incremental_landmark_map import IncrementalLandmarkMapConfig
from two_stage_navigation.simple_landmark_mapping import simple_map
from two_stage_navigation.sonoanatomy_multiview import SemanticObservation


WIDTH, HEIGHT = 1600, 900
FPS = 6
TITLE = "FRESH SYSTEMATIC SWEEP - DEVELOPMENT"


def blank():
    return np.full((HEIGHT, WIDTH, 3), (26, 28, 33), dtype=np.uint8)


def image_tile(panel, image, x, y, width, height):
    if image is None:
        text(panel, "Robot image unavailable", x + 15, y + height // 2, .7)
        return False
    ratio = min(width / image.shape[1], height / image.shape[0])
    size = (int(image.shape[1] * ratio), int(image.shape[0] * ratio))
    scaled = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    left, top = x + (width - size[0]) // 2, y + (height - size[1]) // 2
    panel[top:top + size[1], left:left + size[0]] = scaled
    return True


def robot_image(path):
    return cv2.imread(str(path)) if path.is_file() else None


def prefix_map(actor, sequence, config, final):
    included = []
    for key in ("observations", "ambiguous_observations"):
        included.append({
            name: [row for row in values if row["frame_sequence"] <= sequence]
            for name, values in actor[key].items()
        })
    if final:
        return actor["landmarks"], *included
    observations, ambiguities = (
        {name: [SemanticObservation(**row) for row in values] for name, values in group.items()}
        for group in included
    )
    return simple_map(observations, ambiguities, config), *included


def map_image(figure, sequence, actor, config, bounds, final):
    figure.clear()
    axis = figure.add_subplot(projection="3d")
    draw_map(axis, *prefix_map(actor, sequence, config, final))
    for axis_name, (low, high) in zip("xyz", bounds):
        getattr(axis, f"set_{axis_name}lim")(float(low), float(high))
    axis.set_box_aspect(bounds[:, 1] - bounds[:, 0])
    axis.set_title("Final actor map" if final else "Accumulating actor observations", pad=3)
    figure.canvas.draw()
    return cv2.cvtColor(np.asarray(figure.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)


def render_video(session, output, actor, frames, images, valid, masks, evaluation, protocol):
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("could not open sweep MP4 writer")
    figure = plt.figure(figsize=(7.4, 6.3), dpi=120, layout="constrained")
    FigureCanvasAgg(figure)
    points = np.asarray([
        row["coordinate_mm"] for group in ("observations", "ambiguous_observations")
        for rows in actor[group].values() for row in rows
    ]).reshape(-1, 3)
    bounds = np.column_stack((points.min(0) - 3, points.max(0) + 3)) if len(points) else np.array(
        [[0., 1.], [0., 1.], [0., 1.]]
    )
    config = IncrementalLandmarkMapConfig(**protocol["landmark_map"])
    camera = session / "public/robot"
    written = 0
    camera_count = 0

    def write(panel, repeat):
        nonlocal written
        for _ in range(repeat):
            writer.write(panel)
        written += repeat

    try:
        intro = blank()
        text(intro, TITLE, 42, 64, 1.1)
        text(intro, f"s0010  |  seed {actor['seed']}  |  Stage 1 reference {frames[valid[0]]['stage1_session_id']}",
             42, 116, .82)
        image_tile(intro, robot_image(camera / "reset.jpg"), 45, 150, 1010, 675)
        region = actor["stage1_region"]
        text(intro, "Stage 1 plan", 1100, 300, .88)
        text(intro, f"L4 reference index  {region['l4_z']:.2f}", 1100, 360, .72)
        text(intro, f"L5 reference index  {region['l5_z']:.2f}", 1100, 406, .72)
        first = frames[valid[0]]["requested_relative_pose_index"]
        text(intro, f"First waypoint  x={first[0]:.1f}, z={first[1]:.1f}", 1100, 480, .68)
        text(intro, "Robot at reset", 52, 859, .68)
        write(intro, 3 * FPS)

        for sequence in valid:
            frame = frames[sequence]
            panel = blank()
            text(panel, TITLE, 32, 53, 1.03)
            x, z, angle = frame["executed_relative_pose_index"]
            text(panel, (f"Seed {actor['seed']}  |  Frame {sequence:02d}  |  {frame['phase'].title()}  |  "
                         f"Executed x={x:.2f}  z={z:.2f}  angle={angle:.3f} rad"), 32, 97, .72)
            image = images[sequence]
            for col, (label, mask) in enumerate((
                ("Ultrasound", np.zeros(image.shape, dtype=np.uint8)),
                ("L4 U-Net", masks["L4"][sequence]), ("L5 U-Net", masks["L5"][sequence]),
            )):
                left = 26 + col * 224
                text(panel, label, left + 9, 145, .68)
                rgb = cv2.cvtColor(overlay(image, mask), cv2.COLOR_RGB2BGR)
                image_tile(panel, rgb, left, 155, 215, 278)
            text(panel, "TP cyan  |  SAP magenta  |  Other bone green", 38, 464, .62)
            text(panel, "Synchronized robot camera", 38, 507, .69)
            camera_count += image_tile(panel, robot_image(camera / f"{sequence:03d}.jpg"), 28, 527, 670, 317)
            mapped = map_image(figure, sequence, actor, config, bounds, sequence == valid[-1])
            image_tile(panel, mapped, 716, 132, 861, 686)
            if frame.get("tp_eligible_for_mapping") is False:
                text(panel, "Supplemental SAP acquisition", 37, 873, .62)
            text(panel, "Actor evidence is shown in acquisition order", 750, 850, .67)
            write(panel, 2)

        end = blank()
        text(end, "OFFLINE EVALUATION OF THE FINAL ACTOR MAP", 44, 66, .98)
        image_tile(end, map_image(figure, valid[-1], actor, config, bounds, True), 42, 120, 845, 712)
        text(end, "Provisional simulator identity", 900, 187, .78)
        for i, name in enumerate(COLORS):
            row = evaluation["score"]["landmarks"][name]
            text(end, name.replace("_", " "), 910, 260 + i * 100, .77)
            text(end, f"{row['outcome']}  |  {row['supporting_viewpoint_count']} distinct views",
                 910, 298 + i * 100, .70)
        text(end, f"{len(valid)} valid ultrasound frames", 910, 736, .70)
        write(end, 4 * FPS)
    finally:
        writer.release()
        plt.close(figure)

    reader = cv2.VideoCapture(str(output))
    frames_decoded, fps = 0, float(reader.get(cv2.CAP_PROP_FPS))
    sample = None
    while True:
        ok, image = reader.read()
        if not ok:
            break
        frames_decoded += 1
        if frames_decoded == 3 * FPS + 6:
            sample = image.copy()
    reader.release()
    if frames_decoded != written or not np.isfinite(fps) or fps <= 0 or sample is None:
        raise RuntimeError(f"MP4 verification failed: {frames_decoded}/{written}, fps={fps}")
    cv2.imwrite(str(output.with_name("video_preview.png")), sample)
    return {
        "seed": actor["seed"], "decoded_frames": frames_decoded, "fps": fps,
        "duration_seconds": frames_decoded / fps, "dimensions": [WIDTH, HEIGHT],
        "verified_acquisition_frames": len(valid), "robot_camera_frames": camera_count,
        "robot_camera_error": actor.get("robot_camera_error"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    actor = load(args.session / "actor_result.json")
    evaluation = load(args.session / "evaluation.json")
    protocol = yaml.safe_load(args.protocol.read_text())
    frames, images, valid, _, _, _ = load_public(
        args.session, protocol["geometry"]["ultrasound_pixel_resolution_mm"],
    )
    with np.load(args.session / "public/unet_masks.npz") as archive:
        masks = dict(archive)
    actor["level_hypotheses_by_frame"] = {str(k): frames[k]["level_hypotheses"] for k in valid}
    actor_figure(args.output / "final_actor_map.png", args.session.name, images, frames,
                 valid, masks, actor, actor["landmarks"], title=TITLE)
    if not args.static_only:
        manifest = render_video(args.session, args.output / "revised_systematic_sweep.mp4", actor,
                                frames, images, valid, masks, evaluation, protocol)
        write_json(args.output / "video_manifest.json", manifest)


if __name__ == "__main__":
    main()
