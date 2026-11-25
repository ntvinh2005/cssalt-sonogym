"""Render matched views of the observed anatomical surfaces."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml

from dense import (
    COLORS, IDENTITIES, cloud_from_arrays, distinct_pose_matrix, load_cache, nearest,
    public_context, voxel_fuse,
)
from run_simple_landmark_mapping import sha256, write_json
from two_stage_navigation.simple_landmark_mapping import LEVELS, extract_observations


RGB = np.asarray(COLORS) / 255
LABELS = ("L4 TP", "L4 SAP", "L5 TP", "L5 SAP")
SIZE = (1400, 780)
FPS = 8


def load_cloud(path):
    with np.load(path) as artifact:
        ids = [IDENTITIES.index(identity) for identity in artifact["identity"]]
        return cloud_from_arrays(artifact["xyz_mm"], ids, artifact["frame_sequence"], artifact["confidence"])


def prefix(cloud, sequence):
    include = cloud.frame_sequence <= sequence
    return cloud_from_arrays(*[a[include] for a in
                               (cloud.xyz_mm, cloud.code, cloud.frame_sequence, cloud.confidence)])


def bounds_for(*arrays):
    points = np.concatenate(arrays)
    if not len(points):
        return np.tile((-1., 1.), (3, 1))
    low = points.min(0) - 1.
    high = points.max(0) + 1.
    center = (low + high) / 2
    radii = np.maximum(high - low, 5.) / 2
    return np.column_stack((center - radii, center + radii))


def set_view(axis, bounds, ticks=True):
    axis.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2])
    axis.set_box_aspect(np.diff(bounds, axis=1).ravel())
    axis.view_init(elev=23, azim=-58)
    if ticks:
        axis.set(xlabel="Base X (mm)", ylabel="Base Y (mm)", zlabel="Base Z (mm)")
        axis.tick_params(labelsize=7)
    else:
        axis.set_axis_off()


def scatter(axis, cloud, size=9, alpha=.9):
    for i in range(len(IDENTITIES)):
        points = cloud.xyz_mm[cloud.code == i]
        if len(points):
            axis.scatter(*points.T, s=size, c=[RGB[i]], alpha=alpha, depthshade=False)


def legend(figure, bottom=.018):
    figure.legend([Line2D([], [], color=color, marker="o", linestyle="none") for color in RGB],
                  LABELS, loc="lower center", bbox_to_anchor=(.5, bottom), ncol=4, frameon=False)


def plot_progression(output, seed, valid, p0, p1, distinct, voxel, bounds):
    fig, axes = plt.subplots(4, 4, figsize=(14, 12), subplot_kw={"projection": "3d"})
    for col, fraction in enumerate((.25, .5, .75, 1.)):
        n = int(np.ceil(fraction * len(valid)))
        p0_now, p1_now = (prefix(c, valid[n - 1]) for c in (p0, p1))
        fused0, fused1 = (voxel_fuse(c, distinct, voxel)[0] for c in (p0_now, p1_now))
        titles = ("Oracle raw", "Learned raw", "Oracle fused", "Learned fused")
        for axis, cloud, title in zip(axes[:, col], (p0_now, p1_now, fused0, fused1), titles):
            scatter(axis, cloud, size=10)
            set_view(axis, bounds, ticks=False)
            axis.set_title(f"{title}\n{n}/{len(valid)} waypoints", fontsize=10)
    fig.suptitle(f"Observed surface reconstruction on s0010 | Seed {seed}", fontsize=17)
    fig.subplots_adjust(top=.91, bottom=.07, wspace=.03, hspace=.20)
    legend(fig)
    fig.savefig(output / "progression.png", dpi=130)
    plt.close(fig)


def plot_coverage(output, oracle, raw, fused, gate):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), subplot_kw={"projection": "3d"})
    for col, label in enumerate(LABELS):
        all_points = [prefix_cloud.xyz_mm[prefix_cloud.code == col] for prefix_cloud in (oracle, raw, fused)]
        bounds = bounds_for(*all_points)
        ref, *learned = all_points
        for axis, mapped, method in zip(axes[:, col], learned, ("Raw", "Fused")):
            covered = nearest(ref, mapped) <= gate
            axis.scatter(*ref[covered].T, s=12, c="#8f989e", label="Covered Oracle sample",
                         depthshade=False)
            axis.scatter(*ref[~covered].T, s=13, c="#770c36", label="Uncovered Oracle sample",
                         depthshade=False)
            axis.scatter(*mapped.T, s=7, c=[RGB[col]], alpha=.65, label="Learned sample",
                         depthshade=False)
            set_view(axis, bounds, ticks=False)
            axis.set_title(f"{label} | {method}\n{covered.sum()}/{len(covered)} covered", fontsize=11)
    handles = [Line2D([], [], marker="o", color=c, linestyle="none") for c in ("#8f989e", "#770c36", "#1a1a1a")]
    fig.legend(handles, ["Oracle within 1.5 mm", "Oracle gap", "Learned sample in class color"],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Learned coverage of the eligible Oracle-visible surface", fontsize=16)
    fig.subplots_adjust(top=.88, bottom=.09)
    fig.savefig(output / "coverage_by_structure.png", dpi=145)
    plt.close(fig)


def plot_uncertainty(output, directory, bounds, voxel):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), subplot_kw={"projection": "3d"},
                            layout="constrained")
    for axis, method, title in zip(axes, ("B2", "B3"), ("Oracle fusion", "Learned fusion")):
        with np.load(directory / method / "cloud.npz") as a:
            one = a["unique_viewpoints"] == 1
            axis.scatter(*a["xyz_mm"][one].T, color="#b4bac1", s=8, depthshade=False)
            legend_points = axis.scatter(*a["xyz_mm"][~one].T, c=a["view_rms_mm"][~one], s=20,
                                         cmap="viridis", vmin=0, vmax=np.sqrt(3) * voxel / 2,
                                         depthshade=False)
            conflict = a["conflicting_semantics"]
            axis.scatter(*a["xyz_mm"][conflict].T, c="#ac2554", marker="x", s=35, depthshade=False)
            axis.set_title(f"{title}\n{int((~one).sum())} multi-view samples")
        set_view(axis, bounds)
    fig.colorbar(legend_points, ax=axes.tolist(), shrink=.63, pad=.045, label="Across-view RMS (mm)")
    fig.suptitle("Gray: one distinct view · crosses: competing semantics")
    fig.savefig(output / "fusion_uncertainty.png", dpi=145)
    plt.close(fig)


def render_video(output, source, public, raw, distinct, voxel, bounds, probabilities, rules, seed):
    path = output / "incremental_dense.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"failed opening video writer: {path}")
    fig = plt.figure(figsize=(SIZE[0] / 100, SIZE[1] / 100), dpi=100)
    n = len(public["valid"])
    camera_count = 0
    try:
        for i, seq in enumerate(public["valid"]):
            fig.clear()
            frame, images = public["frames"][seq], public["images"]
            x, y, z = 1000 * np.asarray(frame["actor_base_to_ee_position_m"])
            angle = frame["executed_relative_pose_index"][2]
            fig.text(.04, .937, f"Recorded s0010 sweep | Seed {seed}", fontsize=18, weight="bold")
            fig.text(.04, .894, f"Waypoint {i + 1}/{n} · {frame['phase'].title()} · "
                     f"Probe XYZ ({x:.1f}, {y:.1f}, {z:.1f}) mm · Executed angle {angle:.2f} rad",
                     fontsize=10)
            us = fig.add_axes([.035, .385, .23, .45])
            us.imshow(images[seq], cmap="gray", vmin=0, vmax=max(.01, np.quantile(images[seq], .99)))
            cloud = prefix(raw, seq)
            for j, level in enumerate(LEVELS):
                prob = probabilities[level][i]
                _, accepted = extract_observations(
                    prob.argmax(0), prob, rules[level], frame, public["coords"][seq]
                )
                for value in (1, 2):
                    if np.any(accepted == value):
                        us.contour(accepted == value, colors=[RGB[2 * j + value - 1]],
                                   levels=[.5], linewidths=.75, alpha=.9)
            us.set_title("Ultrasound and learned masks", fontsize=11)
            us.set_axis_off()
            cam = fig.add_axes([.035, .075, .23, .245])
            camera_path = source / "public/robot" / f"{seq:03d}.jpg"
            photo = cv2.imread(str(camera_path)) if camera_path.is_file() else None
            if photo is not None:
                cam.imshow(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
                camera_count += 1
            cam.set_title("Synchronized robot camera", fontsize=11)
            cam.set_axis_off()
            fused, _ = voxel_fuse(cloud, distinct, voxel)
            for pos, item, title in ((.27, cloud, "Raw learned interfaces"),
                                     (.63, fused, "Learned voxel fusion")):
                axis = fig.add_axes([pos, .115, .34, .70], projection="3d")
                scatter(axis, item, size=10)
                set_view(axis, bounds)
                axis.set_title(f"{title} · {len(item.code)} samples", fontsize=11)
            legend(fig, .032)
            fig.canvas.draw()
            data = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
            writer.write(data)
            if i == n - 1:
                for _ in range(2 * FPS):
                    writer.write(data)
    finally:
        writer.release()
        plt.close(fig)
    reader = cv2.VideoCapture(str(path))
    decoded, fps = 0, float(reader.get(cv2.CAP_PROP_FPS))
    preview = None
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        decoded += 1
        if decoded == n:
            preview = frame
    reader.release()
    if decoded != n + 2 * FPS or fps <= 0 or preview is None or camera_count != n:
        raise ValueError(f"video verification failed: decoded={decoded} fps={fps} camera={camera_count}")
    cv2.imwrite(str(output / "video_preview.png"), preview)
    write_json(output / "video_verification.json", {
        "seed": seed, "frames": decoded, "fps": fps, "duration_seconds": decoded / fps,
        "verified_acquisition_frames": n, "robot_camera_frames": camera_count,
        "dimensions": SIZE, "sha256": sha256(path),
        "scope": "Offline learned reconstruction. Each fused frame uses only points available by that waypoint.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "dense", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    cv2.setNumThreads(1)
    manifest = json.loads(args.manifest.read_text())
    seed = manifest["output_schema"]["comparison_video_seed"]
    source = Path(next(s["path"] for s in manifest["sessions"] if s["seed"] == seed))
    protocol = yaml.safe_load(Path(manifest["protocol"]["path"]).read_text())
    public = public_context(source, protocol, manifest)
    probabilities, rules, _ = load_cache(
        Path(manifest["shared_probability_cache"]["path"]), sha256(args.manifest), manifest["models"],
        seed, public["valid"], public["images"].shape[1:],
    )
    directory = args.dense / str(seed)
    oracle, raw, fused = [load_cloud(directory / method / "cloud.npz") for method in ("B0", "B1", "B3")]
    voxel = manifest["projection"]["simulator_label_spacing_mm"]
    gate = manifest["evaluation"]["spatial_distance_gate_mm"]
    distinct = distinct_pose_matrix(public["frames"], protocol["landmark_map"])
    bounds = bounds_for(oracle.xyz_mm, raw.xyz_mm, fused.xyz_mm)
    plot_progression(args.output, seed, public["valid"], oracle, raw, distinct, voxel, bounds)
    plot_coverage(args.output, oracle, raw, fused, gate)
    plot_uncertainty(args.output, directory, bounds, voxel)
    render_video(args.output, source, public, raw, distinct, voxel, bounds, probabilities, rules, seed)


if __name__ == "__main__":
    main()
