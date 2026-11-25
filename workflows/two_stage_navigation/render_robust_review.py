"""Render geometry, observed surfaces and speckle comparisons from robustness artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def tracking_and_trajectories(root, output):
    colors = ["#3267ad", "#198969", "#c44144"]
    sessions = []
    for seed in (2026121700, 2026121702):
        dense = root / f"E1_{seed}"
        stable = root / f"E2_{seed}"
        if not (dense / "public/frames.jsonl").exists():
            continue
        frames = rows(dense / "public/frames.jsonl")
        poses = rows(dense / "privileged/full_pose_replay.jsonl")
        assert all(x["acquisition_id"] == y["acquisition_id"] for x, y in zip(frames, poses))
        sessions.append((seed, frames, poses, rows(stable / "public/frames.jsonl")
                         if (stable / "public/frames.jsonl").exists() else []))
    if not sessions:
        return
    fig = plt.figure(figsize=(13, 6), layout="constrained")
    for index, (seed, frames, poses, _) in enumerate(sessions, start=1):
        axis = fig.add_subplot(1, len(sessions), index, projection="3d")
        for group, (lo, hi) in enumerate(((0, 54), (54, 102), (102, 150))):
            observed = [(f, p) for f, p in zip(frames, poses) if lo <= f["waypoint_index"] < hi]
            if not observed:
                continue
            command = 1000 * np.array([f["commanded_nominal_position_m"] for f, _ in observed])
            executed = 1000 * np.array([p["human_to_ee_position_m"] for _, p in observed])
            axis.plot(*command.T, color=colors[group], ls=":", lw=1, alpha=.65)
            axis.plot(*executed.T, color=colors[group], marker=".", ms=3, lw=.8,
                      label=["54 original targets", "48 medium additions", "48 dense additions"][group])
        axis.set(xlabel="Anatomy x (mm)", ylabel="Anatomy y (mm)", zlabel="Anatomy z (mm)",
                 title=f"Seed {seed}")
        axis.legend(loc="upper left", fontsize=8)
        axis.view_init(29, -54)
    fig.suptitle("Commanded nominal positions (dotted) and executed ultrasound positions (solid)")
    fig.savefig(output / "sweep_trajectories.png", dpi=170)
    plt.close(fig)
    keys = ("full_nominal_position_error_mm", "full_nominal_rotation_error_deg")
    fig, axes = plt.subplots(len(sessions), 2, figsize=(12.5, 3 * len(sessions)), squeeze=False,
                             layout="constrained")
    for i, (seed, frames, _, stable) in enumerate(sessions):
        for j, (column, axis) in enumerate(zip(keys, axes[i])):
            for acquisition, tag in ((frames, "Original controller"), (stable, "Stabilized")):
                selected = [f for f in acquisition if f["image_valid"] and f["waypoint_index"] < 54]
                axis.plot([f["waypoint_index"] for f in selected], [f[column] for f in selected],
                          ".-", ms=3, lw=1, label=tag)
            axis.axhline((1., .5)[j], ls=":", c="#9a4444", lw=1, label="Proposed tolerance")
            axis.set(xlabel="Original waypoint", ylabel=("Position error (mm)", "Rotation error (degrees)")[j],
                     title=f"Seed {seed}")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    fig.suptitle("Full probe errors relative to the nominal surface-planner target")
    fig.savefig(output / "matched_probe_pose_errors.png", dpi=170)
    plt.close(fig)


def sap_surfaces(mapping, output):
    selected = [row for row in mapping["sessions"] if row["arm"] == "E0" and "surface" in row]
    identity = "RIGHT_L4_SAP"
    colors = ["#2c72ba", "#d25c42"]
    if not selected:
        return
    fig = plt.figure(figsize=(14, 4 * len(selected)), layout="constrained")
    columns = (("oracle", "point"), ("oracle", "surface"), ("UNet", "point"), ("UNet", "surface"))
    for i, row in enumerate(selected):
        clouds = [c["observed_surface_mm"] for kind in ("oracle", "UNet")
                  for c in clusters(row["surface"][kind]["mapping"][identity])]
        flat = np.concatenate(clouds)
        low, high = flat.min(0) - 1, flat.max(0) + 1
        for j, (perception, mapper) in enumerate(columns):
            axis = fig.add_subplot(len(selected), len(columns), i * len(columns) + j + 1, projection="3d")
            entry = row[mapper][perception]
            hypothesis = entry["mapping"][identity]
            for k, cluster in enumerate(clusters(hypothesis)):
                support = cluster["supporting_observations"]
                if "observed_surface_mm" in cluster:
                    points = np.asarray(cluster["observed_surface_mm"])
                    axis.scatter(*points.T, s=4, color=colors[k], alpha=.45)
                axis.scatter(*np.asarray([r["coordinate_mm"] for r in support]).T,
                             s=20, color=colors[k])
                axis.scatter(*cluster["position_mm"], marker="x", s=55, color=colors[k])
            outcome = entry["score"]["landmarks"][identity]["outcome"]
            axis.set_title(f"{row['seed']} · {perception} · {mapper}\n{hypothesis['status']} / {outcome}",
                           fontsize=9)
            for name, lo, hi in zip("xyz", low, high):
                getattr(axis, f"set_{name}lim")(lo, hi)
            axis.set(xlabel="Base x (mm)", ylabel="Base y (mm)", zlabel="Base z (mm)")
            axis.tick_params(labelsize=7)
            axis.view_init(27, -66)
    fig.suptitle("RIGHT L4 SAP | two highest-ranked hypotheses | blue selected · red competing")
    fig.savefig(output / "L4_SAP_point_vs_surface.png", dpi=155)
    plt.close(fig)


def clusters(hypothesis):
    return ([hypothesis["selected"]] + hypothesis["competing_clusters"][:1]
            if hypothesis["selected"] else [])


def image_panel(axis, image, mask, title, upper):
    axis.imshow(image, cmap="gray", vmin=0, vmax=upper)
    for kind, color in ((1, "#35eeff"), (2, "#ff5f9c")):
        pixels = mask == kind
        if pixels.any():
            axis.contour(pixels, levels=[.5], colors=[color], linewidths=.8)
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def speckle_and_models(review, output):
    data = review / "E3-dataset"
    if (data / "factorial.npz").exists():
        a = np.load(data / "factorial.npz")
        figures = ("A0", "A1", "B1", "C1", "D0")
        details = json.loads((data / "factorial.json").read_text())
        upper = np.quantile([a[k + "_image"] for k in figures], .995)
        fig, axes = plt.subplots(1, len(figures), figsize=(13, 3.6), layout="constrained")
        titles = ("Reference", "Same pose / new speckle", "New pose / same speckle",
                  "New pose and speckle", "Same pose and speckle")
        for ax, key, title in zip(axes, figures, titles):
            datum = details[key]
            image_panel(ax, a[key + "_image"], a[key + "_labels"],
                        f"{title}\nMAE {datum['image_mae_from_A0']:.3f} · TP {datum['visible_l5_tp_pixels']} px",
                        upper)
        fig.suptitle("Controlled simulator rendering at L5 TP waypoint 5  |  Oracle contours")
        fig.savefig(output / "controlled_speckle.png", dpi=165)
        plt.close(fig)
    predictions = review / "E3-training/validation_predictions.npz"
    if not predictions.exists():
        return
    a = np.load(predictions)
    images, truth, before, after = (a[k] for k in ("ultrasound", "labels", "baseline", "fine_tuned"))
    conditions = [(truth == 1).any(axis=(1, 2)), (truth == 2).any(axis=(1, 2)),
                  ~(truth == 1).any(axis=(1, 2)) & (before == 1).any(axis=(1, 2))]
    examples = [int(np.argmax(cond)) for cond in conditions if np.any(cond)]
    upper = np.quantile(images[examples], .995)
    fig, axes = plt.subplots(len(examples), 3, figsize=(8.8, 3 * len(examples)), squeeze=False,
                             layout="constrained")
    for index, row in zip(examples, axes):
        for axis, mask, title in zip(row, (truth, before, after), ("Oracle", "Original L5 U-Net", "Fine-tuned L5 U-Net")):
            image_panel(axis, images[index], mask[index], f"Validation frame {index} · {title}", upper)
    fig.suptitle("Held-out pose neighborhoods and speckle fields | cyan TP · magenta SAP")
    fig.savefig(output / "L5_original_vs_fine_tuned.png", dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("acquisition", "review", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tracking_and_trajectories(args.acquisition, args.output)
    mapping = args.review / "mapping/results.json"
    if mapping.exists():
        sap_surfaces(json.loads(mapping.read_text()), args.output)
    speckle_and_models(args.review, args.output)


if __name__ == "__main__":
    main()
