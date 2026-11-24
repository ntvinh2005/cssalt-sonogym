#!/usr/bin/env python3
"""Render recorded ultrasound and actor/oracle landmark maps separately."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OFFLINE = "OFFLINE RECORDED-DATA MAPPING EXPERIMENT"
ORACLE = "ORACLE_PERCEPTION_DIAGNOSTIC — NOT AUTONOMOUS"
COLORS = {
    "RIGHT_L4_TP": "#2b83ba", "RIGHT_L4_SAP": "#6a3d9a",
    "RIGHT_L5_TP": "#e08628", "RIGHT_L5_SAP": "#15876d",
}
MASK_COLORS = np.array([[0, 0, 0], [30, 215, 240], [220, 105, 230], [80, 170, 120]])
MARKERS = {"MAPPED": "*", "AMBIGUOUS": "X", "CANDIDATE": "D"}


def load(path):
    return json.loads(path.read_text())


def overlay(image, mask):
    low, high = np.quantile(image, [0.05, 0.95])
    gray = np.clip((image - low) / max(high - low, 1e-6), 0, 1) * 255
    panel = np.repeat(gray[:, :, None], 3, axis=2)
    for class_id, color in enumerate(MASK_COLORS[1:], start=1):
        pixels = mask == class_id
        panel[pixels] = 0.30 * panel[pixels] + 0.70 * color
    return panel.astype(np.uint8)


def draw_map(axis, mapping, observations, ambiguities):
    for identity, color in COLORS.items():
        available = observations.get(identity, [])
        if available:
            points = np.asarray([row["coordinate_mm"] for row in available])
            axis.scatter(*points.T, s=8, color=color, alpha=0.25, depthshade=False)
        uncertain = ambiguities.get(identity, [])
        if uncertain:
            points = np.asarray([row["coordinate_mm"] for row in uncertain])
            axis.scatter(*points.T, s=9, marker="x", color=color, alpha=0.25, depthshade=False)
        landmark = mapping[identity]
        for cluster in landmark["competing_clusters"]:
            if cluster["distinct_viewpoints"] >= 2:
                axis.scatter(*cluster["position_mm"], s=30, marker="o",
                             facecolors="none", edgecolors=color, depthshade=False)
        selection = landmark["selected"]
        if selection is not None:
            axis.scatter(*selection["position_mm"], label=f"{identity.replace('_', ' ')}  {landmark['status']}",
                         marker=MARKERS[landmark["status"]], s=120, color=color,
                         edgecolors="#24272c", linewidths=0.55, depthshade=False)
        else:
            axis.scatter([], [], [], label=f"{identity.replace('_', ' ')}  {landmark['status']}", color=color)
    axis.set(xlabel="Base X (mm)", ylabel="Base Y (mm)", zlabel="Base Z (mm)")
    bounds = np.asarray([axis.get_xlim3d(), axis.get_ylim3d(), axis.get_zlim3d()])
    axis.set_box_aspect(np.maximum(bounds[:, 1] - bounds[:, 0], 1))
    axis.view_init(elev=29, azim=-67)
    axis.legend(loc="upper left", fontsize=8, framealpha=0.8)


def draw_trajectory(axis, frames, valid, choices, region):
    positions = np.asarray([frames[seq]["executed_relative_pose_index"] for seq in valid])
    axis.plot(*positions[:, :2].T, color="#6e7987", linewidth=0.7, alpha=0.7)
    locations = axis.scatter(*positions[:, :2].T, c=valid, cmap="viridis", s=12)
    axis.axhspan(*region["ambiguous_z"], color="#e09d3b", alpha=0.17, label="L4/L5 overlap")
    for sequence in choices:
        x, z, _ = frames[sequence]["executed_relative_pose_index"]
        axis.annotate(str(sequence), (x, z), color="#11151c", fontsize=8)
    axis.set(xlabel="Executed lateral command index", ylabel="Executed cranial–caudal index",
             title="Recorded probe path")
    axis.legend(loc="best", fontsize=8)
    axis.figure.colorbar(locations, ax=axis, shrink=.8, label="Frame sequence")


def choose_frames(masks, route, frames, valid):
    chosen = []
    for longitudinal in (True, False):
        for level in ("L4", "L5"):
            eligible = [seq for seq in valid if route[str(seq)] == [level]
                        and (frames[seq]["executed_relative_pose_index"][2] <= 1.8 if longitudinal
                             else frames[seq]["executed_relative_pose_index"][2] >= 2.7)]
            if eligible:
                sequence = max(eligible, key=lambda seq: np.isin(masks[level][seq], (1, 2)).sum())
                chosen.append((sequence, level, "longitudinal" if longitudinal else "transverse"))
    return chosen


def actor_figure(output, session, images, frames, valid, masks, data, mapping):
    route = data["level_hypotheses_by_frame"]
    selected = choose_frames(masks, route, frames, valid)
    figure = plt.figure(figsize=(15, 8), facecolor="#ffffff", layout="constrained")
    grid = figure.add_gridspec(2, 4, height_ratios=(2.4, 1.0))
    axes = figure.add_subplot(grid[0, :2], projection="3d")
    draw_map(axes, mapping, data["observations"], data["ambiguous_observations"])
    axes.set_title("U-Net + simple 3D map")
    draw_trajectory(figure.add_subplot(grid[0, 2:]), frames, valid, [row[0] for row in selected],
                    data["stage1_region"])
    for column, (sequence, level, plane) in enumerate(selected):
        axis = figure.add_subplot(grid[1, column])
        axis.imshow(overlay(images[sequence], masks[level][sequence]))
        axis.set_title(f"{level} · {plane} · frame {sequence}", fontsize=10)
        axis.set_axis_off()
    figure.suptitle(f"{OFFLINE}\n{session} · TP yellow  |  SAP magenta  |  Other bone green", fontsize=14)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def oracle_figure(output, session, source):
    evidence = load(source / "observations.json")
    result = load(source / "simple_map.json")
    figure = plt.figure(figsize=(9, 7), layout="constrained")
    axes = figure.add_subplot(projection="3d")
    draw_map(axes, result["landmarks"], evidence["observations"], {})
    figure.suptitle(f"{ORACLE}\n{session} · provisional visible interfaces", fontsize=13)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def text(panel, value, x, y, size=0.54, color=(245, 245, 245)):
    cv2.putText(panel, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def replay(output, overview, images, frames, valid, masks, route):
    fps = 8
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open MP4 video writer: {output}")
    overview_image = cv2.imread(str(overview))
    if overview_image is None:
        writer.release()
        raise RuntimeError(f"cannot read the replay overview: {overview}")
    inset = cv2.resize(overview_image, (768, 410), interpolation=cv2.INTER_AREA)
    try:
        for sequence in valid:
            frame = frames[sequence]
            image = images[sequence]
            levels = route[str(sequence)]
            panel = np.full((720, 1280, 3), (20, 23, 30), dtype=np.uint8)
            text(panel, OFFLINE, 32, 48, 0.93)
            text(panel, f"{frame['session_id']}  |  Frame {sequence:03d}  |  {frame['phase'].split(':')[-1]}",
                 32, 90, 0.7)
            left = cv2.cvtColor(overlay(image, np.zeros_like(image)), cv2.COLOR_RGB2BGR)
            panel[160:440, 32:242] = cv2.resize(left, (210, 280))
            text(panel, "Genuine ultrasound", 32, 142)
            for index, level in enumerate(levels):
                right = cv2.cvtColor(overlay(image, masks[level][sequence]), cv2.COLOR_RGB2BGR)
                x = 278 + index * 230
                panel[160:440, x:x+210] = cv2.resize(right, (210, 280))
                text(panel, f"{level} U-Net", x, 142)
            if len(levels) == 1:
                panel[130:540, 512:1280] = inset
            else:
                panel[130:396, 760:1258] = cv2.resize(inset, (498, 266), interpolation=cv2.INTER_AREA)
            x, z, angle = frame["executed_relative_pose_index"]
            text(panel, f"Executed x={x:.1f}  z={z:.1f}  angle={angle:.2f} rad", 32, 488)
            text(panel, "TP yellow  |  SAP magenta  |  Other bone green", 32, 550)
            if len(levels) == 2:
                text(panel, "Both vertebral levels remain possible for this view", 32, 595)
            text(panel, "Map and trajectory at right summarize the complete recorded session", 32, 664)
            writer.write(panel)
    finally:
        writer.release()
    decoded, frame_count = cv2.VideoCapture(str(output)), 0
    while decoded.read()[0]:
        frame_count += 1
    decoded.release()
    if frame_count != len(valid):
        raise RuntimeError(f"replay decoded {frame_count}/{len(valid)} frames")
    return {"frames": frame_count, "fps": fps, "seconds": frame_count / fps, "size": [1280, 720]}



def comparison_figure(output, result):
    arms = ("A: Oracle + simple", "B: U-Net + simple", "C: U-Net + existing")
    outcomes = (("CORRECT", "#238a77"), ("WRONG", "#c75547"), ("UNRESOLVED", "#afb8c5"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True, layout="constrained")
    total = len(result["sessions"])
    for axis, key, name in zip(axes, ("A", "B", "C"), arms):
        counts = result["comparison"][key]["per_landmark"]
        start = np.zeros(len(COLORS))
        for outcome, color in outcomes:
            values = [counts[identity].get(outcome, 0) for identity in COLORS]
            axis.barh(list(COLORS), values, left=start, color=color, label=outcome.title())
            start += values
        axis.set(xlim=(0, total), xticks=range(total + 1), xlabel="Recorded sessions", title=name)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#e9edf0")
    axes[0].set_yticks(range(len(COLORS)), [name.replace("_", " ") for name in COLORS])
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower center", bbox_to_anchor=(.5, -.34), ncol=3, frameon=False)
    fig.suptitle(f"{OFFLINE}\nIdentity against provisional simulator labels", fontsize=13)
    fig.savefig(output, dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = load(args.results / "comparison.json")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for index, session in enumerate(result["sessions"]):
        name = session["session"]
        source = Path(session["source"])
        actor = args.results / name / "actor"
        oracle = args.results / name / "oracle"
        frames = [json.loads(line) for line in (source / "public/frames.jsonl").read_text().splitlines()]
        with np.load(source / "public/ultrasound.npz") as archive:
            images = archive["ultrasound"]
        with np.load(actor / "unet_masks.npz") as archive:
            masks = dict(archive)
        valid = session["valid_frames"]
        data = load(actor / "observations.json")
        mapping = load(actor / "simple_map.json")
        actor_path = args.output / f"{name}_actor_map.png"
        oracle_path = args.output / f"{name}_oracle_map.png"
        actor_figure(actor_path, name, images, frames, valid, masks, data, mapping)
        oracle_figure(oracle_path, name, oracle)
        manifest[name] = {"actor_map": actor_path.name, "oracle_map": oracle_path.name}
        if index == 0:
            video_path = args.output / f"{name}_recorded_replay.mp4"
            manifest[name]["replay"] = {
                "path": video_path.name,
                **replay(video_path, actor_path, images, frames, valid, masks, data["level_hypotheses_by_frame"]),
            }
    summary = args.output / "comparison.png"
    comparison_figure(summary, result)
    manifest["comparison"] = summary.name
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
