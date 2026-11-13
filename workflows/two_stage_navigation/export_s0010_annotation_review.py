#!/usr/bin/env python3
"""Export detailed evaluator-only s0010 annotation review images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import trimesh
from matplotlib.patches import Rectangle


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LEVEL_COLORS = {
    "sacrum": "#6a3d9a",
    "S1": "#cab2d6",
    "L5": "#1f78b4",
    "L4": "#e31a1c",
    "L3": "#33a02c",
    "L2": "#ff7f00",
    "L1": "#b15928",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def label_values(record: dict[str, Any]) -> list[int]:
    value = record["raw_source_label"]
    return [int(item) for item in value] if isinstance(value, list) else [int(value)]


def geometry_summary(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    geometry = record.get("geometry")
    if geometry is None:
        return None
    if "frames" in geometry:
        summary = geometry["frames"]["mesh_index"]
    elif "candidate_surface_points" in geometry:
        summary = geometry["candidate_surface_points"]["mesh_index"]
    else:
        return None
    return (
        np.asarray(summary["centroid"], dtype=np.float64),
        np.asarray(summary["bounds"][0], dtype=np.float64),
        np.asarray(summary["bounds"][1], dtype=np.float64),
    )


def plot_mpr(
    *,
    ct: np.ndarray,
    segmentation: np.ndarray,
    record: dict[str, Any],
    centroid: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    output: Path,
) -> None:
    x, y, z = np.clip(
        np.rint(centroid).astype(int), 0, np.asarray(ct.shape) - 1
    )
    labels = label_values(record)
    mask = np.isin(segmentation, labels)
    views = [
        (
            "Axial",
            ct[:, :, z].T,
            mask[:, :, z].T,
            (centroid[0], centroid[1]),
            (lower[0], lower[1], upper[0] - lower[0], upper[1] - lower[1]),
            "Right (+x)",
            "Anterior (+y)",
        ),
        (
            "Sagittal",
            ct[x, :, :].T,
            mask[x, :, :].T,
            (centroid[1], centroid[2]),
            (lower[1], lower[2], upper[1] - lower[1], upper[2] - lower[2]),
            "Anterior (+y)",
            "Superior (+z)",
        ),
        (
            "Coronal",
            ct[:, y, :].T,
            mask[:, y, :].T,
            (centroid[0], centroid[2]),
            (lower[0], lower[2], upper[0] - lower[0], upper[2] - lower[2]),
            "Right (+x)",
            "Superior (+z)",
        ),
    ]
    status = str(record["status"])
    color = "#00c8ff" if status == "PROVISIONAL" else "#ff3faf"
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.4), constrained_layout=True)
    for axis, (title, image, label_mask, marker, bounds, xlabel, ylabel) in zip(
        axes, views
    ):
        axis.imshow(np.clip(image, -300, 1200), cmap="gray", origin="lower")
        if np.any(label_mask):
            axis.contour(label_mask, levels=[0.5], colors=["#ffe600"], linewidths=1.0)
        axis.scatter([marker[0]], [marker[1]], c=color, marker="+", s=100, linewidths=2)
        axis.add_patch(
            Rectangle(
                (bounds[0], bounds[1]),
                max(bounds[2], 1.0),
                max(bounds[3], 1.0),
                fill=False,
                edgecolor=color,
                linewidth=1.2,
                linestyle="--",
            )
        )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
    uncertainty = record.get("positional_uncertainty_mm")
    figure.suptitle(
        "EVALUATOR-ONLY "
        + status
        + " | "
        + str(record["anatomy_level"])
        + " "
        + str(record["landmark_name"])
        + " "
        + str(record["laterality"])
        + " | confidence="
        + str(record["confidence"])
        + " | uncertainty_mm="
        + ("not assigned" if uncertainty is None else f"{float(uncertainty):.3f}")
        + "\nYellow contour=source raw label; colored cross/box=review candidate, not actor input"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def plot_geometry_overview(
    group_a: list[dict[str, Any]],
    group_b: list[dict[str, Any]],
    output: Path,
) -> None:
    figure = plt.figure(figsize=(15, 11))
    axis = figure.add_subplot(111, projection="3d")
    rng = np.random.default_rng(20261113)
    for record in group_a:
        mesh_path = Path(record["source_files"][1]["path"])
        mesh = trimesh.load_mesh(mesh_path, process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        selected = rng.choice(len(vertices), size=min(2500, len(vertices)), replace=False)
        points = vertices[selected]
        level = str(record["anatomy_level"])
        axis.scatter(
            points[:, 0],
            points[:, 2],
            points[:, 1],
            s=0.7,
            alpha=0.20,
            color=LEVEL_COLORS[level],
            label=level,
        )
    for record in group_b:
        summary = geometry_summary(record)
        if summary is None:
            continue
        centroid = summary[0]
        provisional = record["status"] == "PROVISIONAL"
        color = "#00c8ff" if provisional else "#ff3faf"
        marker = "o" if provisional else "x"
        axis.scatter(
            [centroid[0]],
            [centroid[2]],
            [centroid[1]],
            s=45,
            marker=marker,
            color=color,
        )
        axis.text(
            centroid[0],
            centroid[2],
            centroid[1],
            safe_name(
                "{}_{}_{}".format(
                    record["anatomy_level"],
                    record["landmark_name"],
                    record["laterality"],
                )
            ),
            fontsize=5,
        )
    axis.set_xlabel("Right (+x), mesh index")
    axis.set_ylabel("Superior (+z), mesh index")
    axis.set_zlabel("Anterior (+y), mesh index")
    axis.set_title(
        "Evaluator-only 3D review | cyan=PROVISIONAL | magenta=UNRESOLVED candidate"
    )
    axis.view_init(elev=15, azim=-90)
    axis.legend(loc="upper left", fontsize=7, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_unresolved_summary(records: list[dict[str, Any]], output: Path) -> None:
    rows = [
        [
            str(record["anatomy_level"]),
            str(record["landmark_name"]),
            str(record["laterality"]),
            "geometry candidate" if record.get("geometry") is not None else "no geometry",
            "; ".join(record["limitations"]),
        ]
        for record in records
        if record["status"] == "UNRESOLVED"
    ]
    figure, axis = plt.subplots(figsize=(18, 0.48 * len(rows) + 2.0))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["level", "landmark", "laterality", "record", "why unresolved"],
        loc="center",
        cellLoc="left",
        colWidths=[0.08, 0.16, 0.08, 0.10, 0.58],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.35)
    axis.set_title(
        "UNRESOLVED records: no precise landmark should be inferred from these candidates",
        pad=18,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    ct = np.asarray(
        nib.load(manifest["provenance"]["ct"]["path"]).dataobj, dtype=np.float32
    )
    segmentation = np.asarray(
        nib.load(manifest["provenance"]["segmentation"]["path"]).dataobj,
        dtype=np.uint8,
    )
    group_a = manifest["annotations"]["group_A_unity_method_reproduction"]
    group_b = manifest["annotations"]["group_B_future_MBB_bony_geometry"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for record in group_a:
        summary = record["geometry"]["segmentation"]["mesh_index"]
        output = args.output_dir / "group_a" / (
            safe_name(str(record["anatomy_level"])) + "_major_structure_mpr.png"
        )
        plot_mpr(
            ct=ct,
            segmentation=segmentation,
            record=record,
            centroid=np.asarray(summary["centroid"], dtype=np.float64),
            lower=np.asarray(summary["bounds"][0], dtype=np.float64),
            upper=np.asarray(summary["bounds"][1], dtype=np.float64),
            output=output,
        )
        generated.append(output)

    unresolved_without_geometry: list[dict[str, Any]] = []
    for index, record in enumerate(group_b):
        summary = geometry_summary(record)
        if summary is None:
            unresolved_without_geometry.append(record)
            continue
        output = args.output_dir / "group_b" / (
            f"{index:02d}_"
            + safe_name(
                "{}_{}_{}_{}".format(
                    record["anatomy_level"],
                    record["landmark_name"],
                    record["laterality"],
                    record["status"],
                )
            )
            + ".png"
        )
        plot_mpr(
            ct=ct,
            segmentation=segmentation,
            record=record,
            centroid=summary[0],
            lower=summary[1],
            upper=summary[2],
            output=output,
        )
        generated.append(output)

    overview = args.output_dir / "three_dimensional_overview.png"
    plot_geometry_overview(group_a, group_b, overview)
    generated.append(overview)
    unresolved = args.output_dir / "unresolved_records.png"
    plot_unresolved_summary(group_b, unresolved)
    generated.append(unresolved)

    with (args.output_dir / "unresolved_records.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["anatomy_level", "landmark_name", "laterality", "limitations"]
        )
        for record in group_b:
            if record["status"] == "UNRESOLVED":
                writer.writerow(
                    [
                        record["anatomy_level"],
                        record["landmark_name"],
                        record["laterality"],
                        "; ".join(record["limitations"]),
                    ]
                )

    copied = {
        "annotations.json": args.manifest,
        "annotations.sha256": args.manifest.with_suffix(".sha256"),
        "readiness_report.md": REPOSITORY_ROOT
        / "docs/research/sonogym_s0010_anatomical_ground_truth_readiness.md",
        "notation.md": REPOSITORY_ROOT / "docs/notation.md",
        "ultrasound_correspondence.png": REPOSITORY_ROOT
        / "docs/research/figures/s0010_anatomical_ground_truth/ultrasound_correspondence.png",
        "original_three_dimensional_overview.png": REPOSITORY_ROOT
        / "docs/research/figures/s0010_anatomical_ground_truth/lumbar_mesh_regions.png",
    }
    for name, source in copied.items():
        shutil.copy2(source, args.output_dir / name)
        generated.append(args.output_dir / name)

    file_records = [
        {
            "path": str(path.relative_to(args.output_dir)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(generated)
    ]
    index = {
        "schema": "sonogym-s0010-annotation-review-v1",
        "patient_id": "s0010",
        "source_manifest_sha256": sha256(args.manifest),
        "group_a_image_count": len(group_a),
        "group_b_geometry_image_count": sum(
            geometry_summary(record) is not None for record in group_b
        ),
        "group_b_unresolved_without_geometry_count": len(
            unresolved_without_geometry
        ),
        "interpretation": {
            "yellow_contour": "privileged raw source label",
            "cyan": "PROVISIONAL review candidate",
            "magenta": "UNRESOLVED geometry candidate",
            "actor_input": False,
            "clinical_validation": False,
        },
        "files": file_records,
    }
    (args.output_dir / "review_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    readme = """# s0010 anatomical annotation review

All overlays are evaluator-only and must not be used as actor inference input.

- group_a/: seven major sacral/lumbar raw-label annotations.
- group_b/: every provisional or unresolved record with available candidate geometry.
- three_dimensional_overview.png: all major meshes and candidate centroids.
- unresolved_records.png/csv: structures for which no precise annotation was supported.
- ultrasound_correspondence.png: clean ultrasound beside privileged raw-label overlays.
- annotations.json: complete coordinates, provenance, uncertainty, status, and limitations.

Color key:
- yellow contour: source raw segmentation label;
- cyan marker/box: PROVISIONAL candidate;
- magenta marker/box: UNRESOLVED candidate geometry.

These are simulator engineering annotations for reviewer inspection, not clinician-validated anatomy.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
