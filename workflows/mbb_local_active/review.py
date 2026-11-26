"""Render actual synchronized robot and ultrasound frames from the sealed study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from workflows.mbb_direct_sonographic.prepare import read_json, verify, write_json


def overlay(image, decision):
    lo, hi = np.percentile(image, (1, 99.5))
    gray = np.uint8(np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1) * 255)
    scale = 3
    gray = cv2.resize(gray, (gray.shape[1] * scale, gray.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    ink = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    semantic = decision.get("semantic_pixels", {})
    for name, color in (("RIGHT_L5_TP", (105, 228, 68)), ("RIGHT_L5_SAP", (255, 172, 71))):
        for row, col in semantic.get(name, []):
            cv2.circle(ink, (col * scale, row * scale), 3, color, -1)
    for path in decision.get("echo_paths", []):
        if path["status"] == "JOINED":
            for col, row in zip(path["columns"], path["rows"]):
                if row is not None:
                    cv2.circle(ink, (col * scale, row * scale), 3, (98, 228, 248), -1)
    for row, col in decision.get("candidate_pixels", []):
        cv2.circle(ink, (col * scale, row * scale), 4, (30, 125, 255), 1)
    return ink


def montage(frames, images, actor, evaluation, sequence, out):
    fig, axes = plt.subplots(len(sequence), 3, figsize=(8.8, 2.75 * len(sequence)), squeeze=False,
                             layout="constrained")
    for index, k in enumerate(sequence):
        camera = cv2.imread(str(out.parent / "public" / "robot" / f"{k:03d}.jpg"))
        if camera is not None:
            axes[index, 0].imshow(cv2.cvtColor(camera, cv2.COLOR_BGR2RGB))
        else:
            axes[index, 0].imshow(images[k], cmap="gray")
        axes[index, 0].set_title(f"Acquisition {k:03d}", fontsize=9)
        for col, (name, row) in enumerate((
            ("U-Net", actor["frames"][k]), ("Oracle-visible", evaluation["oracle_frames"][k]),
        ), 1):
            img = overlay(images[k], row)
            axes[index, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), extent=(0, 75, 100, 0))
            status = row["status"].replace("_", " ").lower()
            axes[index, col].set_title(f"{name}\n{status}", fontsize=7)
            axes[index, col].set_ylabel("Depth (mm)", fontsize=7)
        for ax in axes[index]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(out / f"ultrasound_{sequence[0]:03d}_{sequence[-1]:03d}.png", dpi=150)
    plt.close(fig)


def video(session, out, images, rows, indices):
    preview = out / "acquisition_unverified.mp4"
    writer = cv2.VideoWriter(str(preview), cv2.VideoWriter_fourcc(*"mp4v"), 3., (1024, 640))
    if not writer.isOpened():
        return None
    count = 0
    for i in indices:
        image = cv2.imread(str(session / "public/robot" / f"{i:03d}.jpg"))
        if image is None:
            continue
        picture = np.zeros((640, 1024, 3), dtype=np.uint8)
        picture[54:586, :574] = cv2.resize(image, (574, 532))
        picture[38:638, 574:1024] = overlay(images[i], rows[i])
        cv2.putText(picture, f"Executed waypoint acquisition {i:03d}", (13, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, .70, (245, 245, 245), 1, cv2.LINE_AA)
        status = rows[i]["status"].replace("_", " ").lower()
        cv2.putText(picture, status, (14, 619),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (245, 245, 245), 1, cv2.LINE_AA)
        writer.write(picture)
        count += 1
    writer.release()
    capture = cv2.VideoCapture(str(preview))
    decoded = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        decoded += 1
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if decoded != count or count == 0 or fps <= 0:
        return None
    final = out / "actual_acquisition.mp4"
    preview.rename(final)
    return {"path": str(final), "frames": decoded, "duration_seconds": decoded / fps,
            "sample_rate_description": "one image per recorded waypoint, shown at fixed 3 fps"}


def review(session, evidence_dir):
    for line in (session / "local_actor.sha256").read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        verify(session / name, checksum)
    actor = read_json(session / "local_actor.json")
    evidence = read_json(evidence_dir / "evaluation.json")
    frames = [json.loads(line) for line in (session / "public/frames.jsonl").read_text().splitlines()]
    with np.load(session / "public/ultrasound.npz", allow_pickle=False) as data:
        images = data["ultrasound"]
    out = session / "review"
    out.mkdir(exist_ok=False)
    n = actor["initial_frame_count"]
    initial = actor["source_sequence"]
    local = list(range(n, len(frames)))
    if initial is None:
        indices = list(range(0, n, 16))
    else:
        indices = [initial, *local]
    for start in range(0, len(indices), 6):
        montage(frames, images, actor, evidence, indices[start:start + 6], out)
    fig = plt.figure(figsize=(11, 5), layout="constrained")
    for panel, mode in enumerate(("unet", "oracle_visible"), 1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        executed = 1000 * np.asarray([p["actor_base_to_ee_position_m"] for p in frames])
        ax.plot(*executed[:n].T, color="#888888", lw=.7, label="Matched initial sweep")
        if local:
            ax.plot(*executed[n:].T, color="#3d98bc", marker=".", label="Additional executed poses")
        for r in evidence["candidate_projection_evidence"]:
            if r["perception"] == mode and r["scope"] == "after":
                xyz = np.array(r["candidate_xyz_mm"])
                ax.scatter(*xyz.T, s=20, label=f"Candidate {r['frame']}")
        ax.legend(fontsize=6)
        ax.set(xlabel="Robot x (mm)", ylabel="Robot y (mm)", zlabel="Robot z (mm)", title=mode)
    fig.savefig(out / "actual_trajectory_and_regions.png", dpi=170)
    plt.close(fig)
    footage = video(session, out, images, actor["frames"], indices)
    write_json(out / "video_verification.json", footage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    review(*vars(parser.parse_args()).values())
