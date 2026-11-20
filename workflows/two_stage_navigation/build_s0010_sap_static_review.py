#!/usr/bin/env python3
"""Build the static RIGHT L4/L5 SAP review package for s0010.

The package is privileged review evidence. It preserves the current heuristic
candidate and uncertainty masks and never promotes them to semantic anatomy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
import numpy as np


PACKAGE_NAME = "s0010_right_l4l5_sap_review_v1"
LEVELS = ("L4", "L5")
STRUCTURE_IDS = {level: f"RIGHT_{level}_SAP" for level in LEVELS}
SAP_IDS = {level: f"{level}_RIGHT_superior_articular_process" for level in LEVELS}
TP_IDS = {level: f"{level}_RIGHT_transverse_process" for level in LEVELS}
COLORS = {
    "bone": "#b8b8b8",
    "candidate": "#2f78b7",
    "uncertainty": "#f28e2b",
    "tp": "#2ca02c",
}
VIEW_SPECS = (
    ("posterior", 10, -90, False, "Posterior view"),
    ("posterior_oblique", 20, -50, False, "Posterior-oblique view"),
    ("lateral", 10, 0, False, "Right lateral view"),
    ("medial_oblique", 20, 145, False, "Medial-oblique facet context"),
    ("superior", 88, -90, False, "Superior / axial-like view"),
    ("candidate_boundary_closeup", 22, -50, True, "Candidate boundary close-up"),
    ("uncertainty_closeup", 35, -70, True, "Uncertainty-zone close-up"),
    ("relative_to_tp", 18, -45, False, "SAP candidate relative to TP core"),
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[2]
AUDIT = _load_module(
    "s0010_sap_audit_for_static_review",
    ROOT / "workflows/two_stage_navigation/audit_s0010_sap_geometry.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--surface-slab-tolerance-mm", type=float, default=1.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def candidate_face_count(faces: np.ndarray, mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.all(mask[np.asarray(faces, dtype=np.int64)], axis=1)))


def load_geometries(geometry_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        level: {
            "sap": AUDIT.load_geometry(AUDIT.locate_geometry(geometry_root, SAP_IDS[level])),
            "tp": AUDIT.load_geometry(AUDIT.locate_geometry(geometry_root, TP_IDS[level])),
        }
        for level in LEVELS
    }


def _projection(
    points: np.ndarray,
    plane: Any,
    pose: dict[str, Any],
    tolerance_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    projected = AUDIT.project_points_to_pixels(
        points_index=points,
        image_origin_index=plane.origin_index,
        rotation_columns=plane.rotation_columns,
        width=150,
        depth=200,
        image_resolution_m=float(pose["image_resolution_m"]),
        label_resolution_m=float(pose["label_resolution_m"]),
        plane_tolerance_mm=tolerance_mm,
    )
    inside = projected["inside"]
    return projected["rows"][inside], projected["columns"][inside]


def collect_correspondence(
    *,
    v3_root: Path,
    geometries: dict[str, dict[str, dict[str, Any]]],
    tolerance_mm: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray], float, float]:
    records = {level: [] for level in LEVELS}
    images: dict[str, np.ndarray] = {}
    maximum_coordinate_error = 0.0
    display_percentiles: list[float] = []
    for session_dir in sorted((v3_root / "stage2").glob("mbbpoc_v3_*")):
        frames = read_jsonl(session_dir / "public/frames.jsonl")
        poses = read_jsonl(session_dir / "privileged/full_pose_replay.jsonl")
        ultrasound = np.load(session_dir / "public/ultrasound.npz")["ultrasound"]
        coordinate_archive = np.load(
            session_dir / "privileged/pixel_to_volume_coordinates.npz"
        )
        native = coordinate_archive["human_img_coords"].astype(np.float64)
        local = coordinate_archive["native_local_image_coordinates_m"].astype(np.float64)
        if not (len(frames) == len(poses) == len(ultrasound) == len(native)):
            raise RuntimeError(f"misaligned records in {session_dir}")
        for index, (frame, pose, image, native_map) in enumerate(
            zip(frames, poses, ultrasound, native)
        ):
            if frame["frame_sequence"] != index or pose["frame_sequence"] != index:
                raise RuntimeError("frame sequence mismatch")
            if frame["frame_sha256"] != pose["frame_sha256"]:
                raise RuntimeError("frame identity mismatch")
            plane = AUDIT.plane_from_full_pose(
                end_effector_position_m=np.asarray(pose["human_to_ee_position_m"]),
                rotation_columns=np.asarray(pose["human_to_ee_rotation_columns"]),
                image_origin_height_m=float(pose["image_origin_height_m"]),
                local_coordinates_m=local,
                label_resolution_m=float(pose["label_resolution_m"]),
            )
            error = float(np.max(np.abs(plane.coordinates_index - native_map)))
            maximum_coordinate_error = max(maximum_coordinate_error, error)
            display_percentiles.append(float(np.percentile(image, 99.5)))
            image_key = f"{session_dir.name}:{index}"
            images[image_key] = np.asarray(image, dtype=np.float32)
            for level in LEVELS:
                sap = geometries[level]["sap"]
                tp = geometries[level]["tp"]
                sap_rows, sap_columns = _projection(
                    sap["vertices"][sap["original"]], plane, pose, tolerance_mm
                )
                uncertain_rows, uncertain_columns = _projection(
                    sap["vertices"][sap["uncertain"]], plane, pose, tolerance_mm
                )
                tp_rows, tp_columns = _projection(
                    tp["vertices"][tp["proposed"]], plane, pose, tolerance_mm
                )
                unique_pixels = AUDIT.unique_pixel_count(sap_rows, sap_columns)
                boundary_touch = bool(
                    len(sap_rows)
                    and (
                        np.any(sap_rows <= 3)
                        or np.any(sap_rows >= 196)
                        or np.any(sap_columns <= 3)
                        or np.any(sap_columns >= 146)
                    )
                )
                records[level].append(
                    {
                        "session_id": session_dir.name,
                        "frame_sequence": index,
                        "frame_sha256": frame["frame_sha256"],
                        "acquisition_id": frame["acquisition_id"],
                        "producer_sequence": frame["producer_sequence"],
                        "target_name": frame["target_name"],
                        "phase": frame["phase"],
                        "target_probability": float(frame["target_probability"]),
                        "image_key": image_key,
                        "maximum_coordinate_error_index": error,
                        "unique_projected_pixels": unique_pixels,
                        "projected_vertex_entries": int(len(sap_rows)),
                        "boundary_touching": boundary_touch,
                        "sap_rows": sap_rows,
                        "sap_columns": sap_columns,
                        "uncertain_rows": uncertain_rows,
                        "uncertain_columns": uncertain_columns,
                        "tp_rows": tp_rows,
                        "tp_columns": tp_columns,
                    }
                )
    return records, images, maximum_coordinate_error, float(np.median(display_percentiles))


def select_curated_examples(
    records: list[dict[str, Any]], preferred_target_name: str | None = None
) -> list[dict[str, Any]]:
    """Select six deterministic, distinct review examples from one structure."""

    used_hashes: set[str] = set()
    selected: list[dict[str, Any]] = []

    def add(category: str, candidates: list[dict[str, Any]], reason: str) -> None:
        for row in candidates:
            if row["frame_sha256"] not in used_hashes:
                selected.append({**row, "review_category": category, "selection_reason": reason})
                used_hashes.add(row["frame_sha256"])
                return

    present = [row for row in records if row["unique_projected_pixels"] > 0]
    absent = [row for row in records if row["unique_projected_pixels"] == 0]
    strongest = sorted(
        present,
        key=lambda row: (-row["unique_projected_pixels"], row["session_id"], row["frame_sequence"]),
    )
    add(
        "best_geometric_intersection",
        strongest,
        "largest unique projected-pixel support; not a semantic positive label",
    )
    first_session = selected[0]["session_id"] if selected else None
    add(
        "different_session_intersection",
        [row for row in strongest if row["session_id"] != first_session],
        "strong intersection from a different reset session",
    )
    if present:
        median_support = float(np.median([row["unique_projected_pixels"] for row in present]))
        middle = sorted(
            present,
            key=lambda row: (
                abs(row["unique_projected_pixels"] - median_support),
                row["session_id"],
                row["frame_sequence"],
            ),
        )
        add(
            "ambiguous_unreviewed_identity",
            middle,
            "median geometric support with acoustic visibility and SAP identity unreviewed",
        )
    touching = sorted(
        [row for row in present if row["boundary_touching"]],
        key=lambda row: (-row["unique_projected_pixels"], row["session_id"], row["frame_sequence"]),
    )
    if touching:
        add(
            "boundary_touching",
            touching,
            "projected candidate touches the ultrasound image boundary",
        )
    else:
        add(
            "lowest_support_intersection",
            sorted(
                present,
                key=lambda row: (
                    row["unique_projected_pixels"], row["session_id"], row["frame_sequence"]
                ),
            ),
            "no boundary-touching example was available; lowest nonzero support used",
        )
    add(
        "negative_geometric_contrast",
        sorted(
            absent,
            key=lambda row: (
                preferred_target_name is not None
                and row["target_name"] != preferred_target_name,
                -row["target_probability"],
                row["session_id"],
                row["frame_sequence"],
            ),
        ),
        "verified plane with no projected candidate pixel; not a semantic SAP negative",
    )
    add(
        "projection_alignment_check",
        sorted(
            present,
            key=lambda row: (
                row["maximum_coordinate_error_index"],
                -row["unique_projected_pixels"],
                row["session_id"],
                row["frame_sequence"],
            ),
        ),
        "low reconstruction error with nonzero projected support",
    )
    if len(selected) < 6:
        add(
            "additional_context",
            strongest + sorted(absent, key=lambda row: (row["session_id"], row["frame_sequence"])),
            "deterministic nonduplicate context used because a requested category was unavailable",
        )
    if len(selected) != 6:
        raise RuntimeError(f"expected six curated examples, got {len(selected)}")
    return selected


def _equal_axes(ax: Any, bounds: np.ndarray) -> None:
    center = np.mean(bounds, axis=0)
    radius = float(np.max(np.ptp(bounds, axis=0)) / 2)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def render_3d_view(
    *,
    level: str,
    sap: dict[str, Any],
    tp: dict[str, Any],
    output: Path,
    elevation: float,
    azimuth: float,
    closeup: bool,
    title: str,
) -> None:
    vertices = sap["vertices"]
    faces = sap["faces"]
    candidate = vertices[sap["original"]]
    uncertain = vertices[sap["uncertain"]]
    tp_core = tp["vertices"][tp["proposed"]]
    figure = plt.figure(figsize=(8, 7), dpi=170)
    axis = figure.add_subplot(111, projection="3d")
    surface = Poly3DCollection(
        vertices[faces[::2]], facecolor=COLORS["bone"], edgecolor="none", alpha=0.12
    )
    axis.add_collection3d(surface)
    axis.scatter(
        *candidate.T,
        s=12,
        facecolors="none",
        edgecolors=COLORS["candidate"],
        linewidths=0.6,
        alpha=0.75,
    )
    axis.scatter(*uncertain.T, s=2.5, c=COLORS["uncertainty"], alpha=0.70)
    axis.scatter(*tp_core.T, s=3, c=COLORS["tp"], alpha=0.65)
    bounds = candidate if closeup else vertices
    if closeup:
        margin = np.asarray([8.0, 8.0, 8.0])
        bounds = np.vstack([np.min(candidate, axis=0) - margin, np.max(candidate, axis=0) + margin])
    _equal_axes(axis, bounds)
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_xlabel("Right (+x), index")
    axis.set_ylabel("Anterior (+y), index")
    axis.set_zlabel("Superior (+z), index")
    axis.set_title(f"RIGHT {level} SAP — {title}\nCandidate and uncertainty coincide; semantic subset is empty")
    legend = [
        Line2D([0], [0], marker="o", color="none", markeredgecolor=COLORS["candidate"],
               markerfacecolor="none", label="original heuristic candidate", markersize=7),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["uncertainty"],
               markeredgecolor="none", label="unresolved uncertainty", markersize=6),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["tp"],
               markeredgecolor="none", label="proposed TP surface core (context)", markersize=6),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["bone"],
               markeredgecolor="none", label="whole vertebral surface", markersize=7),
    ]
    axis.legend(handles=legend, loc="upper left", fontsize=7)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def make_contact_sheet(paths: list[Path], output: Path, title: str) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=150)
    for axis, path in zip(axes.flat, paths):
        axis.imshow(mpimg.imread(path))
        axis.set_axis_off()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def render_ct_context(
    *, asset_root: Path, level: str, sap: dict[str, Any], output: Path
) -> None:
    ct = np.asanyarray(nib.load(asset_root / "selected_dataset/s0010/ct.nii.gz").dataobj)
    labels = np.asanyarray(
        nib.load(asset_root / "selected_dataset_stl/s0010/combined_label_map.nii.gz").dataobj
    )
    points = sap["vertices"][sap["original"]]
    center = np.rint(np.median(points, axis=0)).astype(int)
    label_value = 28 if level == "L4" else 27
    figure, axes = plt.subplots(3, 5, figsize=(15, 9), dpi=150)
    planes = (
        ("Sagittal", 0, 1, 2),
        ("Coronal", 1, 0, 2),
        ("Axial", 2, 0, 1),
    )
    for row, (name, fixed, horizontal, vertical) in enumerate(planes):
        for column, offset in enumerate((-4, -2, 0, 2, 4)):
            coordinate = int(center[fixed] + offset)
            if fixed == 0:
                image = ct[coordinate, :, :].T
                segmentation = labels[coordinate, :, :].T
            elif fixed == 1:
                image = ct[:, coordinate, :].T
                segmentation = labels[:, coordinate, :].T
            else:
                image = ct[:, :, coordinate].T
                segmentation = labels[:, :, coordinate].T
            in_slab = np.abs(points[:, fixed] - coordinate) <= 0.75
            axis = axes[row, column]
            axis.imshow(np.clip(image, -300, 1200), cmap="gray", origin="lower")
            axis.contour(segmentation == label_value, levels=[0.5], colors=["lime"], linewidths=0.7)
            axis.scatter(
                points[in_slab, horizontal], points[in_slab, vertical],
                s=8, c=COLORS["uncertainty"], alpha=0.75,
            )
            axis.set_title(f"{name} {offset * 1.5:+.1f} mm", fontsize=8)
            axis.set_axis_off()
    figure.suptitle(
        f"RIGHT {level} SAP CT context — green=whole-{level} label; orange=1.5-index candidate slab"
    )
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def render_ultrasound_example(
    *,
    structure_id: str,
    row: dict[str, Any],
    image: np.ndarray,
    directory: Path,
    ordinal: int,
    display_vmax: float,
) -> dict[str, Any]:
    category = row["review_category"]
    stem = f"{structure_id}_{ordinal:02d}_{category}"
    raw_path = directory / f"{stem}_raw.png"
    overlay_path = directory / f"{stem}_overlay.png"
    comparison_path = directory / f"{stem}_comparison.png"
    plt.imsave(raw_path, image, cmap="gray", vmin=0.0, vmax=display_vmax)

    figure, axis = plt.subplots(figsize=(5, 6), dpi=150)
    axis.imshow(image, cmap="gray", vmin=0.0, vmax=display_vmax)
    axis.scatter(
        row["sap_columns"], row["sap_rows"], s=24, facecolors="none",
        edgecolors=COLORS["candidate"], linewidths=0.8, label="heuristic SAP candidate",
    )
    axis.scatter(
        row["uncertain_columns"], row["uncertain_rows"], s=5,
        c=COLORS["uncertainty"], alpha=0.75, label="uncertainty (same region)",
    )
    axis.scatter(
        row["tp_columns"], row["tp_rows"], s=10,
        c=COLORS["tp"], marker="x", alpha=0.65, label="TP core context",
    )
    axis.set_title(
        f"{structure_id} evaluator-only overlay\n{row['session_id']} frame {row['frame_sequence']}"
    )
    axis.legend(loc="lower left", fontsize=6)
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(overlay_path)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=150)
    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=display_vmax)
    axes[0].set_title("Clean actor-visible ultrasound")
    axes[1].imshow(image, cmap="gray", vmin=0.0, vmax=display_vmax)
    axes[1].scatter(
        row["sap_columns"], row["sap_rows"], s=22, facecolors="none",
        edgecolors=COLORS["candidate"], linewidths=0.8,
    )
    axes[1].scatter(
        row["uncertain_columns"], row["uncertain_rows"], s=5,
        c=COLORS["uncertainty"], alpha=0.75,
    )
    axes[1].scatter(row["tp_columns"], row["tp_rows"], s=9, c=COLORS["tp"], marker="x")
    axes[1].set_title("Privileged geometry overlay")
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle(
        f"{structure_id}: {category}\nProjection is verified; visibility and SAP identity are not reviewed"
    )
    figure.tight_layout()
    figure.savefig(comparison_path)
    plt.close(figure)
    return {
        "structure_id": structure_id,
        "review_category": category,
        "selection_reason": row["selection_reason"],
        "session_id": row["session_id"],
        "frame_sequence": row["frame_sequence"],
        "producer_sequence": row["producer_sequence"],
        "acquisition_id": row["acquisition_id"],
        "frame_sha256": row["frame_sha256"],
        "phase": row["phase"],
        "target_name": row["target_name"],
        "unique_projected_pixels": row["unique_projected_pixels"],
        "projected_vertex_entries": row["projected_vertex_entries"],
        "maximum_coordinate_error_index": row["maximum_coordinate_error_index"],
        "boundary_touching": row["boundary_touching"],
        "projection_status": "VERIFIED",
        "acoustic_visibility": "NOT_REVIEWED",
        "semantic_identity": "NOT_REVIEWED",
        "raw_image": str(raw_path.relative_to(directory.parent.parent)),
        "overlay_image": str(overlay_path.relative_to(directory.parent.parent)),
        "comparison_image": str(comparison_path.relative_to(directory.parent.parent)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_geometry(
    *, level: str, sap: dict[str, Any], asset_root: Path, directory: Path
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / f"RIGHT_{level}_SAP_candidate_and_uncertainty.npz"
    np.savez_compressed(
        npz_path,
        vertices_mesh_index=sap["vertices"].astype(np.float32),
        faces=sap["faces"].astype(np.int32),
        original_candidate_vertex_mask=sap["original"],
        proposed_semantic_vertex_mask=sap["proposed"],
        uncertainty_vertex_mask=sap["uncertain"],
        source_mesh_sha256=np.asarray(sap["source_mesh_sha256"]),
        coordinate_frame=np.asarray("segmentation_mesh_index_xyz"),
        spacing_mm=np.asarray(1.5),
    )
    csv_path = directory / f"RIGHT_{level}_SAP_candidate_vertices.csv"
    selected = np.flatnonzero(sap["original"])
    write_csv(
        csv_path,
        [
            {
                "source_vertex_index": int(index),
                "x_index": float(sap["vertices"][index, 0]),
                "y_index": float(sap["vertices"][index, 1]),
                "z_index": float(sap["vertices"][index, 2]),
                "in_uncertainty_zone": bool(sap["uncertain"][index]),
                "in_proposed_semantic_subset": bool(sap["proposed"][index]),
            }
            for index in selected
        ],
    )
    mesh_source = asset_root / f"selected_dataset_stl/s0010/vertebrae_{level}.stl"
    mesh_copy = directory / mesh_source.name
    shutil.copy2(mesh_source, mesh_copy)
    return {
        "candidate_npz": str(npz_path.relative_to(directory.parent)),
        "candidate_npz_sha256": sha256(npz_path),
        "candidate_csv": str(csv_path.relative_to(directory.parent)),
        "candidate_csv_sha256": sha256(csv_path),
        "packaged_source_mesh": str(mesh_copy.relative_to(directory.parent)),
        "packaged_source_mesh_sha256": sha256(mesh_copy),
    }


def _markdown_ultrasound_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| structure | category | session/frame | projected pixels | raw | overlay | comparison |",
        "|---|---|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['structure_id']} | {row['review_category']} | "
            f"{row['session_id']} / {row['frame_sequence']} | {row['unique_projected_pixels']} | "
            f"[{Path(row['raw_image']).name}]({Path(row['raw_image']).name}) | "
            f"[{Path(row['overlay_image']).name}]({Path(row['overlay_image']).name}) | "
            f"[{Path(row['comparison_image']).name}]({Path(row['comparison_image']).name}) |"
        )
    return "\n".join(lines)


def write_package_documents(
    *,
    package: Path,
    source_revision: str,
    audit: dict[str, Any],
    manifest: dict[str, Any],
    ultrasound_rows: list[dict[str, Any]],
) -> None:
    verdict_reason = (
        "The original heuristic candidate is retained unchanged, its proposed semantic subset is "
        "empty, and every candidate vertex remains in the uncertainty zone."
    )
    (package / "00_EXECUTIVE_SUMMARY.md").write_text(
        f"""# s0010 RIGHT L4/L5 SAP static review

Goal: allow independent review of the existing provisional RIGHT L4/L5 SAP boundaries on simulator anatomy `s0010` without running code.

- Source commit: `{source_revision}`
- Audit source: job `52053307`
- Annotation basis: existing candidates from correction job `51779234`; no geometry refinement was applied.
- Method: versioned surface masks are shown in whole-vertebra, CT and exact-pose ultrasound context. Geometry overlays are privileged reviewer evidence.
- `RIGHT_L4_SAP`: **`UNRESOLVED`** — {verdict_reason}
- `RIGHT_L5_SAP`: **`UNRESOLVED`** — {verdict_reason}

This package does not provide a nerve coordinate, MBB target region, needle endpoint, clinical validation or actor-visible geometry input.
""",
        encoding="utf-8",
    )
    (package / "01_REVIEW_GUIDE.md").write_text(
        """# Review guide

SAP means superior articular process. The candidate is a heuristic surface selection on the complete vertebral STL; it is not an independently accepted semantic segmentation. The uncertainty zone contains vertices whose membership cannot be resolved from the current source semantics. In this package the candidate and uncertainty masks coincide exactly, and the proposed semantic subset is empty.

## Colors

- Blue open markers: original heuristic SAP candidate.
- Orange filled markers: unresolved uncertainty; these occupy the same vertices as the blue candidate.
- Green markers: proposed TP surface core, shown only as neighboring context.
- Gray surface: complete L4 or L5 vertebral mesh.
- Lime CT contour: complete vertebral-level segmentation, not an SAP mask.

## Figure types

- `figures/3d/`: fixed, labelled surface views and overview sheets.
- `figures/geometry_context/`: CT neighborhood, SAP/TP relationship and L4-versus-L5 comparison.
- `figures/ultrasound/`: clean images, separate privileged overlays and matched comparisons.

## Review focus

Judge whether the selected surface isolates the RIGHT SAP from lamina, pars, pedicle and adjacent facet/posterior elements; whether the base and superior/articular extent are plausible; and whether L4/L5 definitions are internally consistent. A technically accurate ultrasound projection proves only plane correspondence. Bright cortical signal, acoustic visibility and semantic SAP identity require separate review.
""",
        encoding="utf-8",
    )
    (package / "04_MAPPING_VALIDATION.md").write_text(
        f"""# Mapping validation

The V3 acquisitions store the full probe transform, patient-relative slicing inputs and native pixel-to-volume coordinate map under one acquisition identity. The review builder reconstructs the plane from the recorded full pose and compares every coordinate with the saved native map before projecting candidate vertices.

- Sessions: {audit['coordinate_transform_audit']['session_count']}
- Frames: {audit['coordinate_transform_audit']['frame_count']}
- Maximum coordinate disagreement: `{audit['coordinate_transform_audit']['maximum_coordinate_error_index']}` volume index
- Gate: `<= {audit['coordinate_transform_audit']['required_maximum_coordinate_error_index']}` volume index
- Result: **PASS**

This validates technical geometry-to-image correspondence. It does not validate the candidate as SAP, prove that the interface is acoustically visible, or establish that SAP is distinguishable from surrounding bone.
""",
        encoding="utf-8",
    )
    rows_by_structure = {
        structure: [row for row in ultrasound_rows if row["structure_id"] == structure]
        for structure in STRUCTURE_IDS.values()
    }
    for number, level in ((5, "L4"), (6, "L5")):
        structure = STRUCTURE_IDS[level]
        examples = rows_by_structure[structure]
        key_examples = "\n".join(
            f"- `{row['review_category']}`: [{Path(row['comparison_image']).name}]({row['comparison_image']})"
            for row in examples
        )
        (package / f"0{number}_{structure}_DECISION_SHEET.md").write_text(
            f"""# {structure} decision sheet

## Candidate summary

- Current recommendation: **`UNRESOLVED`**.
- Candidate vertices: {manifest['structures'][structure]['candidate_vertex_count']}.
- Proposed semantic vertices: 0.
- Uncertainty vertices: {manifest['structures'][structure]['uncertainty_vertex_count']}.
- Geometry refinement in this package: none.

## Strongest evidence in favor

- The candidate is reproducible from a versioned mesh and mask.
- Exact-pose projection is technically verified.
- Multiple acquired planes intersect the candidate.

## Strongest reasons for doubt

- No source object or label semantically isolates SAP.
- Candidate and uncertainty masks are identical.
- Plane intersection does not prove a visible or distinguishable SAP interface.

## Key 3D evidence

- [Overview sheet](figures/3d/{structure}_overview_sheet.png)
- [Boundary close-up](figures/3d/{structure}_candidate_boundary_closeup.png)
- [Uncertainty close-up](figures/3d/{structure}_uncertainty_closeup.png)
- [CT context](figures/geometry_context/{structure}_CT_context.png)

## Key ultrasound evidence

{key_examples}

## Open questions

1. Does the candidate isolate the RIGHT {level} SAP without adjacent posterior-element contamination?
2. Which vertices should be included, excluded or retained as uncertain at the base and articular surface?
3. In which matched ultrasound examples is the candidate-specific cortical interface actually visible?
4. Is the visible interface distinguishable from TP or another adjacent bony surface?
""",
            encoding="utf-8",
        )
    (package / "07_REVIEW_RESPONSE_TEMPLATE.md").write_text(
        """# Reviewer response template

Reviewer name and qualification:

Review date:

## Reviewed structure

- Structure: `RIGHT_L4_SAP` / `RIGHT_L5_SAP`
- Exact geometry SHA-256:
- Decision: `ACCEPT_PROVISIONAL` / `REVISE` / `UNRESOLVED`
- Rationale:
- Requested inclusion:
- Requested exclusion:
- Region that must remain uncertain:
- Ultrasound visibility decision and cited frame(s):
- Semantic-identity decision and cited frame(s):
- Confidence and basis:
- Corrected geometry file, if any:
- Follow-up action:

Repeat the section for the second structure. A decision applies only to the exact geometry checksum recorded in `02_ANNOTATION_MANIFEST.json`.
""",
        encoding="utf-8",
    )
    (package / "09_SUGGESTED_REVISION_HINTS.md").write_text(
        """# Suggested revision hints

These are questions for review, not proposed final anatomy.

- Both candidates were created by coordinate-based surface selection; check straight coordinate-cut edges for inclusion of lamina, pars, pedicle or adjacent posterior surface.
- Decide whether a conservative, semantically defensible SAP surface core exists. Do not shrink the region only to obtain a cleaner image.
- Mark the SAP base separately when CT resolution or fused surface geometry does not support a sharp boundary.
- Use the paired clean/overlay ultrasound images to distinguish geometric intersection from a visible cortical interface.
- Return corrected source-vertex indices or an edited mask against the packaged STL; screenshots alone are insufficient for geometry promotion.
""",
        encoding="utf-8",
    )
    (package / "figures/ultrasound/ULTRASOUND_INDEX.md").write_text(
        "# Curated ultrasound index\n\n"
        "All overlays are evaluator-only. No row is a semantic positive or negative label.\n\n"
        + _markdown_ultrasound_table(ultrasound_rows)
        + "\n",
        encoding="utf-8",
    )


def validate_package(package: Path, manifest: dict[str, Any], targets: dict[str, Any]) -> dict[str, Any]:
    references: set[str] = set()
    for structure in manifest["structures"].values():
        references.update(structure["figure_files"])
        references.update(structure["ultrasound_files"])
        references.update(
            value
            for key, value in structure["geometry_files"].items()
            if not key.endswith("_sha256")
        )
    for target in targets["targets"]:
        references.update(target["files"])
    missing = sorted(path for path in references if not (package / path).is_file())
    markdown = sorted(package.glob("*.md")) + sorted(package.rglob("*.md"))
    unreadable = []
    for path in dict.fromkeys(markdown):
        try:
            if not path.read_text(encoding="utf-8").strip():
                unreadable.append(str(path.relative_to(package)))
        except UnicodeDecodeError:
            unreadable.append(str(path.relative_to(package)))
    return {
        "referenced_file_count": len(references),
        "missing_referenced_files": missing,
        "markdown_file_count": len(dict.fromkeys(markdown)),
        "unreadable_or_empty_markdown": unreadable,
        "package_relative_references_only": all(not Path(path).is_absolute() for path in references),
        "pass": not missing and not unreadable and all(not Path(path).is_absolute() for path in references),
    }


def write_checksums(package: Path) -> Path:
    checksum_path = package / "checksums.sha256"
    files = sorted(path for path in package.rglob("*") if path.is_file() and path != checksum_path)
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(package)}\n" for path in files),
        encoding="utf-8",
    )
    return checksum_path


def create_zip(package: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, f"{package.name}/{path.relative_to(package)}")
    with zipfile.ZipFile(zip_path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"corrupt ZIP member: {corrupt}")


def main() -> None:
    args = parse_args()
    source_revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    package = output_root / PACKAGE_NAME
    if package.exists() and any(package.iterdir()):
        raise RuntimeError(f"refusing nonempty output package: {package}")
    package.mkdir(parents=True, exist_ok=True)
    for relative in ("figures/3d", "figures/geometry_context", "figures/ultrasound", "geometry"):
        (package / relative).mkdir(parents=True, exist_ok=True)

    asset_root = args.asset_root.resolve()
    geometry_root = args.geometry_root.resolve()
    audit_root = args.audit_root.resolve()
    audit = json.loads((audit_root / "sap_audit_result.json").read_text())
    geometries = load_geometries(geometry_root)
    records, images, maximum_error, display_vmax = collect_correspondence(
        v3_root=args.v3_root.resolve(),
        geometries=geometries,
        tolerance_mm=args.surface_slab_tolerance_mm,
    )
    if maximum_error > 0.001:
        raise RuntimeError("exact-pose mapping gate failed")
    if abs(maximum_error - audit["coordinate_transform_audit"]["maximum_coordinate_error_index"]) > 1e-12:
        raise RuntimeError("mapping result differs from the frozen audit")

    structure_records: dict[str, dict[str, Any]] = {}
    ultrasound_rows: list[dict[str, Any]] = []
    for level in LEVELS:
        structure = STRUCTURE_IDS[level]
        sap = geometries[level]["sap"]
        tp = geometries[level]["tp"]
        view_paths = []
        for name, elevation, azimuth, closeup, title in VIEW_SPECS:
            path = package / f"figures/3d/{structure}_{name}.png"
            render_3d_view(
                level=level, sap=sap, tp=tp, output=path,
                elevation=elevation, azimuth=azimuth, closeup=closeup, title=title,
            )
            view_paths.append(path)
        overview = package / f"figures/3d/{structure}_overview_sheet.png"
        make_contact_sheet(view_paths, overview, f"{structure} static 3D review overview")
        ct_context = package / f"figures/geometry_context/{structure}_CT_context.png"
        render_ct_context(asset_root=asset_root, level=level, sap=sap, output=ct_context)
        relation_context = package / f"figures/geometry_context/{structure}_TP_context.png"
        shutil.copy2(package / f"figures/3d/{structure}_relative_to_tp.png", relation_context)
        geometry_files = export_geometry(
            level=level, sap=sap, asset_root=asset_root, directory=package / "geometry"
        )
        chosen = select_curated_examples(
            records[level], preferred_target_name=f"{level}_RIGHT_TP_SURFACE_CORE"
        )
        selected_rows = []
        for ordinal, row in enumerate(chosen, start=1):
            selected_rows.append(
                render_ultrasound_example(
                    structure_id=structure,
                    row=row,
                    image=images[row["image_key"]],
                    directory=package / "figures/ultrasound",
                    ordinal=ordinal,
                    display_vmax=display_vmax,
                )
            )
        ultrasound_rows.extend(selected_rows)
        figure_files = [str(path.relative_to(package)) for path in view_paths + [overview, ct_context, relation_context]]
        ultrasound_files = [
            row[key]
            for row in selected_rows
            for key in ("raw_image", "overlay_image", "comparison_image")
        ]
        structure_records[structure] = {
            "anatomy_id": "s0010",
            "structure_name": f"RIGHT {level} superior articular process",
            "candidate_status": "UNRESOLVED",
            "source_candidate_id": SAP_IDS[level],
            "source_mesh_path": audit["asset_inventory"]["level_meshes"][level]["path"],
            "source_mesh_sha256": sap["source_mesh_sha256"],
            "candidate_vertex_count": int(np.count_nonzero(sap["original"])),
            "candidate_face_count": candidate_face_count(sap["faces"], sap["original"]),
            "proposed_semantic_vertex_count": int(np.count_nonzero(sap["proposed"])),
            "uncertainty_vertex_count": int(np.count_nonzero(sap["uncertain"])),
            "candidate_equals_uncertainty": bool(np.array_equal(sap["original"], sap["uncertain"])),
            "segmentation_method_summary": "existing coordinate-selected surface candidate; no refinement applied",
            "geometry_version_sha256": sap["sha256"],
            "projection_pipeline": "EXACT_POSE_REPLAY via ultrasound_plane_geometry.py",
            "projection_tolerance_mm": args.surface_slab_tolerance_mm,
            "review_figure_count": len(figure_files),
            "ultrasound_example_count": len(selected_rows),
            "overlay_example_count": len(selected_rows),
            "figure_files": figure_files,
            "ultrasound_files": ultrasound_files,
            "geometry_files": geometry_files,
        }

    comparison = package / "figures/geometry_context/RIGHT_L4_vs_RIGHT_L5_SAP_comparison.png"
    make_contact_sheet(
        [
            package / f"figures/3d/{STRUCTURE_IDS[level]}_{name}.png"
            for level in LEVELS
            for name in ("posterior", "candidate_boundary_closeup", "uncertainty_closeup", "relative_to_tp")
        ],
        comparison,
        "RIGHT L4 versus L5 SAP candidate comparison",
    )
    shutil.copy2(audit_root / "sap_ct_mpr.png", package / "figures/geometry_context/L4_L5_full_CT_context.png")
    for structure in structure_records.values():
        structure["figure_files"].extend(
            [
                "figures/geometry_context/RIGHT_L4_vs_RIGHT_L5_SAP_comparison.png",
                "figures/geometry_context/L4_L5_full_CT_context.png",
            ]
        )
        structure["review_figure_count"] = len(structure["figure_files"])

    write_csv(package / "figures/ultrasound/ULTRASOUND_INDEX.csv", ultrasound_rows)
    manifest = {
        "schema": "sonogym-s0010-right-l4l5-sap-static-review-v1",
        "package_name": PACKAGE_NAME,
        "source_revision": source_revision,
        "anatomy_id": "s0010",
        "laterality": "RIGHT",
        "annotation_basis": "existing candidates from correction job 51779234",
        "annotation_refinement_applied": False,
        "review_decision_recorded": False,
        "semantic_training_authorized": False,
        "actor_usable": False,
        "mbb_target_region_defined": False,
        "source_audit_job": "52053307",
        "source_audit_result_sha256": sha256(audit_root / "sap_audit_result.json"),
        "source_v3_job": "52039283",
        "frame_pool": {
            "sessions": 4,
            "frames": 336,
            "candidate_frame_records": 672,
        },
        "mapping_validation": {
            "mode": "EXACT_POSE_REPLAY",
            "maximum_coordinate_error_index": maximum_error,
            "gate_index": 0.001,
            "pass": True,
        },
        "ultrasound_display": {
            "vmin": 0.0,
            "vmax": display_vmax,
            "vmax_rule": "median per-frame 99.5th percentile across the 336-frame pool",
        },
        "structures": structure_records,
    }
    (package / "02_ANNOTATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_rows = [
        {
            "structure_id": structure,
            "anatomy_id": record["anatomy_id"],
            "candidate_status": record["candidate_status"],
            "candidate_vertex_count": record["candidate_vertex_count"],
            "candidate_face_count": record["candidate_face_count"],
            "uncertainty_vertex_count": record["uncertainty_vertex_count"],
            "number_of_review_figures": record["review_figure_count"],
            "number_of_ultrasound_examples": record["ultrasound_example_count"],
            "number_of_overlay_examples": record["overlay_example_count"],
            "notes": "candidate equals uncertainty; semantic subset empty; no refinement",
        }
        for structure, record in structure_records.items()
    ]
    write_csv(package / "03_STRUCTURE_SUMMARY.csv", summary_rows)
    targets = {
        "schema": "sonogym-s0010-right-l4l5-sap-review-targets-v1",
        "allowed_decisions": ["ACCEPT_PROVISIONAL", "REVISE", "UNRESOLVED"],
        "targets": [],
    }
    for level in LEVELS:
        structure = STRUCTURE_IDS[level]
        targets["targets"].extend(
            [
                {
                    "target_id": f"{structure}_main_candidate",
                    "description": "Judge the exact original heuristic surface candidate.",
                    "files": [
                        f"figures/3d/{structure}_overview_sheet.png",
                        f"figures/3d/{structure}_candidate_boundary_closeup.png",
                        structure_records[structure]["geometry_files"]["candidate_npz"],
                    ],
                },
                {
                    "target_id": f"{structure}_uncertainty_zone",
                    "description": "Judge which vertices must remain uncertain or be revised.",
                    "files": [
                        f"figures/3d/{structure}_uncertainty_closeup.png",
                        f"figures/geometry_context/{structure}_CT_context.png",
                    ],
                },
                {
                    "target_id": f"{structure}_ultrasound_correspondence",
                    "description": "Judge overlay consistency, acoustic visibility and semantic identity separately.",
                    "files": [
                        row["comparison_image"]
                        for row in ultrasound_rows
                        if row["structure_id"] == structure
                    ],
                },
            ]
        )
    (package / "08_REVIEW_TARGETS.json").write_text(
        json.dumps(targets, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_package_documents(
        package=package,
        source_revision=source_revision,
        audit=audit,
        manifest=manifest,
        ultrasound_rows=ultrasound_rows,
    )
    validation = validate_package(package, manifest, targets)
    if not validation["pass"]:
        raise RuntimeError(f"package validation failed: {validation}")
    (package / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_path = write_checksums(package)
    zip_path = output_root / f"{PACKAGE_NAME}.zip"
    create_zip(package, zip_path)
    result = {
        "schema": "sonogym-s0010-right-l4l5-sap-static-review-build-v1",
        "source_revision": source_revision,
        "package_directory": str(package),
        "zip_path": str(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256(zip_path),
        "checksum_entries": len(checksum_path.read_text().splitlines()),
        "ultrasound_examples": len(ultrasound_rows),
        "verdicts": {structure: "UNRESOLVED" for structure in STRUCTURE_IDS.values()},
        "annotation_refinement_applied": False,
        "validation": validation,
    }
    (output_root / "build_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
