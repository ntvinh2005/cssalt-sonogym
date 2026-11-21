#!/usr/bin/env python3
"""Render recorded model behavior and package the representation comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from two_stage_navigation.stage1_interfaces import track_bright_interfaces
from two_stage_navigation.ultrasound_surface_features import topology_feature

import correct_s0010_sap_observability_v1 as correction
import run_s0010_sap_observability as observability


POSITIVE = "SAP_RELATED_CORTICAL_INTERFACE"
NEGATIVE = "OTHER_OR_BACKGROUND"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def public_frame_by_sequence(
    rows: list[dict[str, Any]], frame_sequence: int
) -> dict[str, Any]:
    matches = [row for row in rows if int(row["frame_sequence"]) == frame_sequence]
    if len(matches) != 1:
        raise ValueError(
            f"expected one public frame for sequence {frame_sequence}, found {len(matches)}"
        )
    return matches[0]


def render_frame(
    *,
    image: np.ndarray,
    public: dict[str, Any],
    score: float,
    threshold: float,
    status: str,
    pseudo_projection: np.ndarray,
    first_hit_surface: np.ndarray,
    title: str,
) -> plt.Figure:
    tracks = track_bright_interfaces(image)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), dpi=100)
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Clean ultrasound")
    axes[0].set_axis_off()

    axes[1].imshow(image, cmap="gray")
    colors = ("#00e5ff", "#ffca28", "#ab47bc")
    for track, color in zip(tracks, colors):
        axes[1].plot(track.columns, track.rows, color=color, linewidth=1.5)
    axes[1].set_title(
        "Actor-visible interfaces\n"
        f"p={score:.3f}; cut={threshold:.2f}; "
        f"{'positive' if score >= threshold else 'negative'}",
        fontsize=9,
    )
    axes[1].set_axis_off()

    axes[2].imshow(image, cmap="gray")
    rows, columns = np.where(pseudo_projection)
    axes[2].scatter(columns, rows, s=5, c="#00bcd4", alpha=0.45, label="pseudo-SAP")
    rows, columns = np.where(first_hit_surface)
    axes[2].scatter(columns, rows, s=12, c="#ffd54f", label="first-hit surface")
    readable_status = status.lower().replace("_", " ")
    if readable_status == "sap related cortical interface":
        readable_status = "SAP-related cortical interface"
    axes[2].set_title(
        f"Privileged evaluation only\n{readable_status}", fontsize=9
    )
    axes[2].legend(loc="upper left", fontsize=6)
    axes[2].set_axis_off()

    delta = public.get("executed_delta_relative_mm", [0.0, 0.0])
    angle = float(public.get("executed_delta_angle_rad", 0.0)) * 180.0 / np.pi
    fig.suptitle(
        f"{title}; move=({float(delta[0]):.1f}, {float(delta[1]):.1f}) mm, "
        f"rotation={angle:.1f} deg",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return fig


def figure_array(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def plot_comparison(result: dict[str, Any], diagnostic: dict[str, Any], path: Path) -> None:
    baseline = result["frozen_l5_baseline"]
    selected = result["evaluation"]["L5"]
    diagnostics = {row["variant"]: row for row in diagnostic["results"]["L5"]}
    rows = [
        ("patch baseline", baseline, 8),
        ("surface only", diagnostics["topology"]["evaluation"], diagnostics["topology"]["all_frame_coverage"]["ambiguous_activations"]),
        ("image context", diagnostics["context"]["evaluation"], diagnostics["context"]["all_frame_coverage"]["ambiguous_activations"]),
        ("selected full image", selected["metrics"], selected["all_frame_coverage"]["ambiguous_activations"]),
        ("with prior view", diagnostics["causal"]["evaluation"], diagnostics["causal"]["all_frame_coverage"]["ambiguous_activations"]),
    ]
    labels = [row[0] for row in rows]
    recall = [row[1]["recall"] for row in rows]
    f1 = [row[1]["f1"] for row in rows]
    ambiguous = [row[2] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    axes[0].bar(x - 0.18, recall, width=0.36, label="recall")
    axes[0].bar(x + 0.18, f1, width=0.36, label="F1")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_title("L5 clear-label images (opened evaluation)")
    axes[0].legend()
    axes[1].bar(x, ambiguous, color="#ef6c00")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_ylabel("activated ambiguous images / 36")
    axes[1].set_title("Evidence unresolved by the label contract")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dataset", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    package = output / "s0010_ultrasound_representation_review"
    if package.exists():
        shutil.rmtree(package)
    figures = package / "figures"
    examples = figures / "examples"
    figures.mkdir(parents=True)
    examples.mkdir()

    result = json.loads((args.experiment_dir / "result.json").read_text())
    diagnostic = json.loads(
        (args.experiment_dir / "post_selection_diagnostic.json").read_text()
    )
    checkpoint = Path(diagnostic["checkpoints"]["L5_topology"]["path"])
    bundle = joblib.load(checkpoint)
    estimator = bundle["estimator"]
    threshold = float(bundle["threshold"])
    positive_column = list(estimator.classes_).index(1)
    clear_rows = read_csv(args.evaluation_dataset / "sample_manifest.csv")
    clear = {
        (row["session_id"], int(row["frame_sequence"])): row["provisional_label"]
        for row in clear_rows
        if row["conditioned_level"] == "L5"
    }
    prediction_image = nib.load(args.prediction)
    segmentation_image = nib.load(args.segmentation)
    masks = observability.pseudo_sap_masks(
        np.asarray(prediction_image.dataobj),
        np.asarray(segmentation_image.dataobj),
        prediction_image.affine,
    )
    bone_ids = correction.bone_source_ids(args.label_conversion)
    rendered: list[dict[str, Any]] = []
    writer = imageio.get_writer(
        package / "s0010_l5_surface_representation_demo.mp4",
        fps=4,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    )
    try:
        for session_id in ("s0010sapval_evaluation_04", "s0010sapval_evaluation_05"):
            session = args.capture_root / "sessions" / session_id
            public_rows = read_jsonl(session / "public/frames.jsonl")
            ultrasound = np.load(session / "public/ultrasound.npz")["ultrasound"]
            coordinates = np.load(
                session / "privileged/pixel_to_volume_coordinates.npz"
            )["human_img_coords"]
            raw_labels = np.load(session / "privileged/raw_labels.npz")["raw_labels"]
            for public in public_rows:
                index = int(public["image_archive_index"])
                image = np.asarray(ultrasound[index])
                score = float(
                    estimator.predict_proba(topology_feature(image)[None, :])[
                        0, positive_column
                    ]
                )
                key = (session_id, int(public["frame_sequence"]))
                status = clear.get(key, "AMBIGUOUS")
                projected = observability.projection_diagnostics(
                    coordinates[index], masks["L5"]
                )["projected"]
                first_hit = correction.first_bone_surface(
                    raw_labels=raw_labels[index],
                    corrected_projection=projected,
                    bone_ids=bone_ids,
                )["visible_pseudo_surface"]
                fig = render_frame(
                    image=image,
                    public=public,
                    score=score,
                    threshold=threshold,
                    status=status,
                    pseudo_projection=projected["pseudo"],
                    first_hit_surface=first_hit,
                    title=f"{session_id} frame {int(public['frame_sequence']):02d}",
                )
                writer.append_data(figure_array(fig))
                rendered.append(
                    {
                        "session_id": session_id,
                        "frame_sequence": int(public["frame_sequence"]),
                        "status": status,
                        "score": score,
                    }
                )
                plt.close(fig)
    finally:
        writer.close()

    selected_examples: list[dict[str, Any]] = []
    for status in (POSITIVE, NEGATIVE, "AMBIGUOUS"):
        candidates = [row for row in rendered if row["status"] == status]
        if not candidates:
            continue
        chosen = max(candidates, key=lambda row: row["score"])
        session = args.capture_root / "sessions" / chosen["session_id"]
        public = public_frame_by_sequence(
            read_jsonl(session / "public/frames.jsonl"),
            int(chosen["frame_sequence"]),
        )
        index = int(public["image_archive_index"])
        image = np.load(session / "public/ultrasound.npz")["ultrasound"][index]
        coordinates = np.load(session / "privileged/pixel_to_volume_coordinates.npz")[
            "human_img_coords"
        ][index]
        raw_labels = np.load(session / "privileged/raw_labels.npz")["raw_labels"][index]
        projected = observability.projection_diagnostics(coordinates, masks["L5"])[
            "projected"
        ]
        first_hit = correction.first_bone_surface(
            raw_labels=raw_labels,
            corrected_projection=projected,
            bone_ids=bone_ids,
        )["visible_pseudo_surface"]
        fig = render_frame(
            image=image,
            public=public,
            score=chosen["score"],
            threshold=threshold,
            status=status,
            pseudo_projection=projected["pseudo"],
            first_hit_surface=first_hit,
            title=f"{chosen['session_id']} frame {chosen['frame_sequence']:02d}",
        )
        path = examples / f"L5_highest_{status.lower()}.png"
        fig.savefig(path)
        plt.close(fig)
        selected_examples.append(
            {
                "status": status,
                "session_id": chosen["session_id"],
                "frame_sequence": chosen["frame_sequence"],
                "score": chosen["score"],
                "path": str(path.relative_to(package)),
            }
        )

    plot_comparison(result, diagnostic, figures / "l5_comparison.png")
    (package / "example_index.json").write_text(
        json.dumps(selected_examples, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    copies = {
        args.experiment_dir / "result.json": package / "result.json",
        args.experiment_dir / "frozen_selection.json": package / "frozen_selection.json",
        args.experiment_dir / "post_selection_diagnostic.json": package / "post_selection_diagnostic.json",
        args.experiment_dir / "evaluation_clear_image_decisions.csv": package / "evaluation_clear_image_decisions.csv",
        args.experiment_dir / "evaluation_all_frame_scores.csv": package / "evaluation_all_frame_scores.csv",
        args.protocol: package / args.protocol.name,
        args.report: package / "REPORT.md",
        checkpoint: package / checkpoint.name,
    }
    for source, destination in copies.items():
        shutil.copy2(source, destination)
    provenance = {
        "packaging_job_id": os.environ.get("SLURM_JOB_ID", "not_running_under_slurm"),
        "source_commit": args.source_commit,
        "result_sha256": sha256(package / "result.json"),
        "diagnostic_sha256": sha256(package / "post_selection_diagnostic.json"),
        "protocol_sha256": sha256(package / args.protocol.name),
        "checkpoint_sha256": sha256(package / checkpoint.name),
        "demo_sha256": sha256(package / "s0010_l5_surface_representation_demo.mp4"),
        "actor_feature_privileged_fields": [],
        "evaluation_panels_are_privileged": True,
    }
    (package / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(package)}"
        for path in sorted(package.rglob("*"))
        if path.is_file()
    ]
    (package / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    zip_path = output / "s0010_ultrasound_representation_review.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(Path(package.name) / path.relative_to(package)))
    print(
        json.dumps(
            {
                "zip": str(zip_path),
                "zip_sha256": sha256(zip_path),
                "demo": str(package / "s0010_l5_surface_representation_demo.mp4"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
