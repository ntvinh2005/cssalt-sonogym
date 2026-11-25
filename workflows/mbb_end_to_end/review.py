"""Display original ultrasound, reconstructed surfaces and sealed localization outcomes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from workflows.mbb_end_to_end.evaluate import ARMS, DISPLAY, array_file, checked, nearest


def cloud_panel(ax, cloud, center, title, result=None, legacy=None, references=None, trace=None, prefix=None):
    xyz = cloud["xyz_mm"]
    keep = np.linalg.norm(xyz - center, axis=1) <= 32
    if prefix is not None:
        keep &= cloud["frame_sequence"] <= prefix
    kinds = cloud["identity"]
    for name, color in DISPLAY.items():
        points = xyz[keep & (kinds == name)]
        ax.scatter(*points.T, c=color, s=4 if name == "OTHER_BONE" else 7, depthshade=False, alpha=.65)
    if legacy is not None:
        ax.scatter(*legacy.T, c="#bd2e20", marker="x", s=21, depthshade=False)
    if trace is not None:
        ax.plot(*trace.T, color="#572d70", linewidth=1., linestyle="--", marker=".", markersize=4)
    if references:
        for shape in references.values():
            ax.scatter(*shape.T, c="#161717", marker="o", s=17, depthshade=False)
    if result and result.get("candidate"):
        region = result["candidate"]
        for key, marker, color in (("observed_region_mm", "o", "#1976d2"),
                                   ("inferred_region_mm", "^", "#141414")):
            points = np.asarray(region.get(key, [])).reshape(-1, 3)
            if len(points):
                ax.scatter(*points.T, c=color, marker=marker, s=28, depthshade=False)
    limits = [tuple(c + np.array([-24, 24])) for c in center]
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel="X (mm)", ylabel="Y (mm)",
           zlabel="Z (mm)", title=title)
    ax.set_box_aspect((1., 1., 1.))
    ax.view_init(elev=28, azim=-66)


def legend(fig, add_reference=False, candidate=False):
    labels = dict(DISPLAY)
    if candidate:
        labels.update({"Observed candidate": "#1976d2", "Inferred candidate": "#141414"})
    if add_reference:
        labels["Historical midpoint · evaluator only"] = "#bd2e20"
        labels["CT cortical trace · evaluator only"] = "#572d70"
    names = {"RIGHT_L5_TP":"Right L5 TP", "RIGHT_L5_SAP":"Right L5 SAP", "RIGHT_L4_TP":"Right L4 TP",
             "RIGHT_L4_SAP":"Right L4 SAP", "OTHER_BONE":"Other bone", "UNCERTAIN_BONE":"Uncertain bone"}
    handles = [Line2D([], [], marker="o", color=color, linestyle="none", label=names.get(k, k))
               for k, color in labels.items()]
    fig.legend(handles=handles, loc="outside lower center", ncols=5, frameon=False, fontsize=9)


def annotate_image(ax, image, cloud, sequence, title):
    vmax = max(float(np.quantile(image, .99)), .01)
    ax.imshow(image, cmap="gray", vmin=0., vmax=vmax)
    for name, color in DISPLAY.items():
        mask = (cloud["frame_sequence"] == sequence) & (cloud["identity"] == name)
        samples = cloud["pixel_row_col"][mask]
        ax.scatter(samples[:, 1], samples[:, 0], s=4, c=color, alpha=.8)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def semantic_examples(output, images, cloud, seed):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), layout="constrained")
    for ax, label, title in zip(axes, ("RIGHT_L5_TP", "RIGHT_L5_SAP", "OTHER_BONE"),
                                ("Right L5 TP", "Right L5 SAP", "Other bone")):
        mask = (cloud["identity"] == label) & (cloud["level_hint"] == "L5")
        if np.any(mask):
            seq = int(np.bincount(cloud["frame_sequence"][mask]).argmax())
            annotate_image(ax, images[seq], cloud, seq, f"{title} · frame {seq}")
        else:
            ax.set_title(f"{title} · no eligible frame")
            ax.axis("off")
    legend(fig)
    fig.suptitle(f"Recorded seed {seed} · image-supported learned interfaces")
    fig.savefig(output / "recorded_semantic_examples.png", dpi=155)
    plt.close(fig)


def recorded(args):
    protocol = json.loads(args.protocol.read_text())
    source = json.loads(checked(protocol["source_manifest"]).read_text())
    metrics = json.loads((args.replay / "results.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    fig = plt.figure(figsize=(15, 15), layout="constrained")
    image_fig, panels = plt.subplots(len(metrics["sessions"]), 3, figsize=(9, 13), layout="constrained")
    has_candidate = False
    for i, s in enumerate(metrics["sessions"]):
        seed = s["seed"]
        geometry = s["evaluator_geometry_for_review"]
        legacy = np.asarray(geometry["historical_midpoint_mm"])
        trace = np.asarray(geometry["diagnostic_cortical_trace_mm"])
        center = legacy.mean(0)
        references = {k: np.asarray(v) for k, v in geometry["anatomical_regions_mm"].items()}
        learned = array_file(args.reconstruction / seed / "learned.npz")
        oracle = array_file(args.reconstruction / seed / "oracle.npz")
        old = array_file(checked(source["clouds"][seed]["B1"]["cloud.npz"]))
        for j, (entry, cloud, label) in enumerate((
            ("E0", old, "Historical TP and SAP"),
            ("E1", oracle, "Complete Oracle-visible bone"),
            ("E2", learned, "Complete learned bone"),
        )):
            panel = fig.add_subplot(len(metrics["sessions"]), 3, i * 3 + j + 1, projection="3d")
            result = s["arms"][entry]["prediction"] if entry in ARMS else None
            has_candidate |= bool(result and result.get("candidate"))
            status = result["status"] if result else s["E0"]["U-Net"]["status"]
            cloud_panel(panel, cloud, center, f"{seed}  {label}\n{status}",
                        result=result, legacy=legacy, references=references, trace=trace)
        session = next(item for item in source["sessions"] if str(item["seed"]) == seed)
        parent = Path(session["path"])
        with np.load(parent / "public/ultrasound.npz", allow_pickle=False) as frames:
            images = frames["ultrasound"]
        if i == 0:
            semantic_examples(args.output, images, learned, seed)
        extra = np.isin(oracle["identity"], ("OTHER_BONE", "UNCERTAIN_BONE"))
        if np.any(extra):
            distance = np.where(extra, nearest(oracle["xyz_mm"], trace), np.inf)
            sequence = int(oracle["frame_sequence"][int(distance.argmin())])
        else:
            sequence = 0
        panels[i, 0].imshow(images[sequence], cmap="gray", vmin=0.,
                            vmax=max(float(np.quantile(images[sequence], .99)), .01))
        panels[i, 0].set_title(f"{seed} · frame {sequence}\nRecorded ultrasound", fontsize=10)
        panels[i, 0].axis("off")
        annotate_image(panels[i, 1], images[sequence], oracle, sequence, "Oracle-visible")
        annotate_image(panels[i, 2], images[sequence], learned, sequence, "Learned projection")
    legend(fig, add_reference=True, candidate=has_candidate)
    legend(image_fig)
    fig.suptitle("Matched observed surfaces in robot-base coordinates")
    image_fig.suptitle("Recorded ultrasound and projected interface samples")
    fig.savefig(args.output / "recorded_surfaces.png", dpi=155)
    image_fig.savefig(args.output / "recorded_ultrasound.png", dpi=155)
    plt.close(fig)
    plt.close(image_fig)


def fresh_video(args):
    import cv2

    cloud = array_file(args.cloud)
    output = json.loads(args.prediction.read_text())
    frames = [json.loads(line) for line in (args.session / "public/frames.jsonl").read_text().splitlines()]
    with np.load(args.session / "public/ultrasound.npz", allow_pickle=False) as file:
        images = file["ultrasound"]
    valid = [f for f in frames if f["image_valid"]]
    center = np.median(cloud["xyz_mm"][np.isin(cloud["identity"], ["RIGHT_L5_TP", "RIGHT_L5_SAP"])], axis=0)
    if not np.isfinite(center).all():
        center = np.median(cloud["xyz_mm"], axis=0)
    args.output.mkdir(parents=True, exist_ok=False)
    width, height, fps = 1200, 720, 8
    path = args.output / "fresh_oblique_sweep.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("could not open MP4 writer")
    fig = plt.figure(figsize=(12, 7.2), dpi=100, layout="constrained")
    try:
        for i, frame in enumerate(valid):
            seq = frame["frame_sequence"]
            fig.clear()
            left, right = fig.subfigures(1, 2, width_ratios=(.65, 1.))
            us, camera = left.subplots(2, 1)
            photo = cv2.imread(str(args.session / "public/robot" / f"{seq:03d}.jpg"))
            if photo is None:
                raise ValueError(f"missing synchronized robot image {seq}")
            annotate_image(us, images[seq], cloud, seq, "Ultrasound and learned cortical samples")
            camera.imshow(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
            camera.axis("off")
            camera.set_title("Executed probe at acquisition")
            ax = right.subplots(subplot_kw={"projection":"3d"})
            final = i == len(valid) - 1
            status = output["status"] if final else "acquiring"
            cloud_panel(ax, cloud, center, f"Observed cortical surfaces · {status}",
                        result=output if final else None, prefix=seq)
            xyz = 1000 * np.asarray(frame["actor_base_to_ee_position_m"])
            fig.suptitle(f"Fresh matched development sweep · waypoint {frame['waypoint_index'] + 1}"
                         f" · {frame['phase']} · pose ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) mm")
            fig.canvas.draw()
            image = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
            writer.write(image)
            if final:
                for _ in range(fps * 2):
                    writer.write(image)
    finally:
        writer.release()
        plt.close(fig)
    reader = cv2.VideoCapture(str(path))
    decoded = 0
    while reader.read()[0]:
        decoded += 1
    video_fps = reader.get(cv2.CAP_PROP_FPS)
    reader.release()
    if decoded != len(valid) + 2 * fps or video_fps != fps:
        raise ValueError("MP4 failed full decode verification")
    (args.output / "video_verification.json").write_text(json.dumps({
        "decoded_frames": decoded, "fresh_acquisitions": len(valid), "camera_frames_used": len(valid),
        "fps": fps, "duration_seconds": decoded / fps, "final_actor_status": output["status"],
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode, keys, fn in (
        ("recorded", ("protocol", "replay", "reconstruction", "output"), recorded),
        ("fresh-video", ("session", "cloud", "prediction", "output"), fresh_video),
    ):
        command = modes.add_parser(mode)
        for key in keys:
            command.add_argument("--" + key, type=Path, required=True)
        command.set_defaults(run=fn)
    opts = parser.parse_args()
    opts.run(opts)


if __name__ == "__main__":
    main()
