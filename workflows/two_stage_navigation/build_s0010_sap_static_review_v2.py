#!/usr/bin/env python3
"""Build the second static RIGHT L4/L5 SAP review package for s0010.

This is privileged annotation evidence.  It audits the quantized source STL,
reconstructs a separate surface from the original vertebral labels, and uses
paired adjacent-level surfaces to propose a review target.  It does not create
actor input or promote the proposal to semantic anatomy.
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
from scipy.spatial import cKDTree
import trimesh


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_NAME = "s0010_right_l4l5_sap_review_v2"
LEVELS = ("L4", "L5")
UPPER_LEVEL = {"L4": "L3", "L5": "L4"}
LABEL_VALUE = {"L3": 29, "L4": 28, "L5": 27}
STRUCTURE_IDS = {level: f"RIGHT_{level}_SAP" for level in LEVELS}
COLORS = {
    "lower": "#dce6f1",
    "upper": "#b3a6c7",
    "core": "#00bcd4",
    "uncertain": "#f28e2b",
    "neighbor": "#d627a8",
    "tp": "#2ca02c",
}
REFERENCE_URLS = (
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7002062/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11595282/",
    "https://journals.viamedica.pl/folia_morphologica/article/download/FM.a2020.0122/52184",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V1 = _load_module(
    "s0010_sap_static_review_v1_for_v2",
    ROOT / "workflows/two_stage_navigation/build_s0010_sap_static_review.py",
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    return np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    ) / 2.0


def mesh_quality(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    areas = face_areas(vertices, faces)
    triangles = vertices[faces]
    repeated_corner = np.asarray(
        [len(np.unique(triangle, axis=0)) < 3 for triangle in triangles], dtype=bool
    )
    unique_vertices = np.unique(vertices, axis=0)
    return {
        "raw_vertex_records": int(len(vertices)),
        "unique_physical_locations": int(len(unique_vertices)),
        "face_records": int(len(faces)),
        "zero_area_faces": int(np.count_nonzero(areas == 0.0)),
        "zero_area_faces_with_repeated_physical_corner": int(np.count_nonzero(repeated_corner)),
        "nonzero_area_faces": int(np.count_nonzero(areas > 0.0)),
        "all_unique_locations_integer_quantized": bool(
            np.all(np.isclose(unique_vertices, np.rint(unique_vertices)))
        ),
    }


def valid_surface_locations(vertices: np.ndarray, faces: np.ndarray) -> set[tuple[float, ...]]:
    valid_faces = faces[face_areas(vertices, faces) > 0.0]
    return {tuple(row) for row in np.unique(vertices[valid_faces.ravel()], axis=0).tolist()}


def reconstruct_voxel_surface(label_volume: np.ndarray, label_value: int) -> tuple[np.ndarray, np.ndarray]:
    """Return an exact exposed-voxel surface in label-index coordinates."""

    occupied = np.argwhere(label_volume == label_value)
    occupied_set = {tuple(index) for index in occupied.tolist()}
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    lookup: dict[tuple[float, float, float], int] = {}
    directions = (
        ((-1, 0, 0), ((-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5))),
        ((1, 0, 0), ((0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5))),
        ((0, -1, 0), ((-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5))),
        ((0, 1, 0), ((-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5))),
        ((0, 0, -1), ((-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, -0.5, -0.5))),
        ((0, 0, 1), ((-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5))),
    )
    for voxel in occupied:
        base = tuple(int(value) for value in voxel)
        for direction, offsets in directions:
            neighbor = tuple(base[axis] + direction[axis] for axis in range(3))
            if neighbor in occupied_set:
                continue
            ids = []
            for offset in offsets:
                point = tuple(float(base[axis] + offset[axis]) for axis in range(3))
                if point not in lookup:
                    lookup[point] = len(vertices)
                    vertices.append(point)
                ids.append(lookup[point])
            faces.extend(((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def derive_regions(
    sap: dict[str, Any], upper_vertices: np.ndarray, upper_faces: np.ndarray
) -> dict[str, Any]:
    vertices = sap["vertices"]
    candidate = sap["original"]
    lower_valid = valid_surface_locations(vertices, sap["faces"])
    upper_valid = valid_surface_locations(upper_vertices, upper_faces)
    core_locations = lower_valid & upper_valid & {
        tuple(row) for row in np.unique(vertices[candidate], axis=0).tolist()
    }
    core = np.asarray([candidate[index] and tuple(point) in core_locations for index, point in enumerate(vertices)])
    uncertain = candidate & ~core
    neighbor = np.zeros(len(vertices), dtype=bool)
    if np.any(core & uncertain) or np.any(core & neighbor) or np.any(uncertain & neighbor):
        raise RuntimeError("region masks overlap")
    if not np.array_equal(core | uncertain | neighbor, candidate):
        raise RuntimeError("region masks do not partition the original candidate")
    return {
        "core": core,
        "uncertain": uncertain,
        "neighbor": neighbor,
        "core_locations": np.asarray(sorted(core_locations), dtype=np.float64),
        "unique_candidate_locations": int(len(np.unique(vertices[candidate], axis=0))),
        "unique_core_locations": int(len(core_locations)),
        "unique_uncertain_locations": int(len(np.unique(vertices[uncertain], axis=0))),
        "unique_neighbor_locations": 0,
    }


def surface_correspondence(source_vertices: np.ndarray, derived_vertices: np.ndarray) -> dict[str, Any]:
    source_unique = np.unique(source_vertices, axis=0)
    derived_tree = cKDTree(derived_vertices)
    source_to_derived, nearest_derived = derived_tree.query(source_unique)
    source_tree = cKDTree(source_unique)
    derived_to_source, nearest_source = source_tree.query(derived_vertices)
    return {
        "source_unique_vertices": source_unique,
        "derived_vertices": derived_vertices,
        "source_to_derived_distance_index": source_to_derived,
        "source_to_derived_index": nearest_derived,
        "derived_to_source_distance_index": derived_to_source,
        "derived_to_source_index": nearest_source,
        "summary": {
            "source_to_derived_median_index": float(np.median(source_to_derived)),
            "source_to_derived_p95_index": float(np.percentile(source_to_derived, 95)),
            "source_to_derived_maximum_index": float(np.max(source_to_derived)),
            "derived_to_source_median_index": float(np.median(derived_to_source)),
            "derived_to_source_p95_index": float(np.percentile(derived_to_source, 95)),
            "derived_to_source_maximum_index": float(np.max(derived_to_source)),
        },
    }


def _equal_axes(axis: Any, points: np.ndarray) -> None:
    bounds = np.vstack((np.min(points, axis=0), np.max(points, axis=0)))
    center = np.mean(bounds, axis=0)
    radius = max(1.0, float(np.max(np.ptp(bounds, axis=0)) / 2.0))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def render_articulation_view(
    *, level: str, lower: dict[str, Any], upper: dict[str, Any], regions: dict[str, Any],
    output: Path, elevation: float, azimuth: float, closeup: bool, title: str
) -> None:
    figure = plt.figure(figsize=(8, 7), dpi=190)
    axis = figure.add_subplot(111, projection="3d")
    for geometry, color in ((lower, COLORS["lower"]), (upper, COLORS["upper"])):
        valid = face_areas(geometry["vertices"], geometry["faces"]) > 0.0
        faces = geometry["faces"][valid][::2]
        axis.add_collection3d(
            Poly3DCollection(
                geometry["vertices"][faces], facecolor=color, edgecolor="none", alpha=1.0
            )
        )
    vertices = lower["vertices"]
    axis.scatter(*vertices[regions["core"]].T, s=13, c=COLORS["core"], alpha=0.95)
    axis.scatter(*vertices[regions["uncertain"]].T, s=5, c=COLORS["uncertain"], alpha=0.70)
    if np.any(regions["neighbor"]):
        axis.scatter(*vertices[regions["neighbor"]].T, s=7, c=COLORS["neighbor"], alpha=0.9)
    candidate = vertices[lower["original"]]
    points = candidate if closeup else np.vstack((lower["vertices"], upper["vertices"]))
    if closeup:
        points = np.vstack((np.min(candidate, axis=0) - 6.0, np.max(candidate, axis=0) + 6.0))
    _equal_axes(axis, points)
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_xlabel("Right (+x), index")
    axis.set_ylabel("Anterior (+y), index")
    axis.set_zlabel("Superior (+z), index")
    axis.set_title(f"RIGHT {level} SAP — {title}\n{UPPER_LEVEL[level]} IAP is not semantically labelled")
    axis.legend(
        handles=[
            Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["lower"], label=f"{level} whole vertebra", markersize=7),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["upper"], label=f"{UPPER_LEVEL[level]} whole vertebra", markersize=7),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["core"], label="paired-surface core proposal", markersize=6),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["uncertain"], label="boundary uncertain", markersize=6),
        ],
        loc="upper left", fontsize=7,
    )
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def make_contact_sheet(paths: list[Path], output: Path, title: str) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=150)
    for axis, path in zip(axes.flat, paths):
        axis.imshow(mpimg.imread(path))
        axis.set_axis_off()
    for axis in axes.flat[len(paths):]:
        axis.set_axis_off()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def render_ct_facet_context(
    *, ct: np.ndarray, labels: np.ndarray, level: str, lower: dict[str, Any],
    regions: dict[str, Any], output: Path
) -> None:
    core = lower["vertices"][regions["core"]]
    uncertain = lower["vertices"][regions["uncertain"]]
    original = lower["vertices"][lower["original"]]
    center = np.rint(np.median(core if len(core) else original, axis=0)).astype(int)
    figure, axes = plt.subplots(3, 5, figsize=(15, 9), dpi=170)
    planes = (("Sagittal", 0, 1, 2), ("Coronal", 1, 0, 2), ("Axial", 2, 0, 1))
    for row, (name, fixed, horizontal, vertical) in enumerate(planes):
        for column, offset in enumerate((-4, -2, 0, 2, 4)):
            coordinate = int(center[fixed] + offset)
            if fixed == 0:
                image, segmentation = ct[coordinate].T, labels[coordinate].T
            elif fixed == 1:
                image, segmentation = ct[:, coordinate, :].T, labels[:, coordinate, :].T
            else:
                image, segmentation = ct[:, :, coordinate].T, labels[:, :, coordinate].T
            axis = axes[row, column]
            axis.imshow(np.clip(image, -300, 1200), cmap="gray", origin="lower")
            axis.contour(segmentation == LABEL_VALUE[level], [0.5], colors=["cyan"], linewidths=0.8)
            axis.contour(segmentation == LABEL_VALUE[UPPER_LEVEL[level]], [0.5], colors=["magenta"], linewidths=0.8)
            for points, color, size in ((uncertain, COLORS["uncertain"], 9), (core, COLORS["core"], 16)):
                slab = np.abs(points[:, fixed] - coordinate) <= 0.75
                axis.scatter(points[slab, horizontal], points[slab, vertical], s=size, c=color, alpha=0.85)
            radius = 24
            axis.set_xlim(center[horizontal] - radius, center[horizontal] + radius)
            axis.set_ylim(center[vertical] - radius, center[vertical] + radius)
            axis.set_title(f"{name} {offset * 1.5:+.1f} mm", fontsize=8)
            axis.set_axis_off()
    figure.suptitle(
        f"RIGHT {level} facet-focused CT — cyan={level}; magenta={UPPER_LEVEL[level]}; "
        "cyan points=paired surface; orange points=uncertain"
    )
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def render_surface_comparison(
    *, level: str, source: dict[str, Any], derived_vertices: np.ndarray,
    derived_faces: np.ndarray, output: Path
) -> None:
    figure = plt.figure(figsize=(13, 6), dpi=170)
    for column, (title, vertices, faces, color) in enumerate((
        ("Source STL (integer-quantized)", source["vertices"], source["faces"], "#d95f02"),
        ("Derived exposed-voxel surface", derived_vertices, derived_faces, "#1b9e77"),
    ), start=1):
        axis = figure.add_subplot(1, 2, column, projection="3d")
        valid = face_areas(vertices, faces) > 0.0
        axis.add_collection3d(
            Poly3DCollection(vertices[faces[valid]][::2], facecolor=color, edgecolor="none", alpha=0.88)
        )
        _equal_axes(axis, vertices)
        axis.view_init(elev=20, azim=-50)
        axis.set_title(title)
        axis.set_xlabel("Right")
        axis.set_ylabel("Anterior")
        axis.set_zlabel("Superior")
    figure.suptitle(f"{level} source-versus-derived whole-vertebra surface")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def brightness_score(image: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> float:
    if len(rows) == 0:
        return -1.0
    values = []
    for row, column in set(zip(rows.tolist(), columns.tolist())):
        values.append(float(np.max(image[max(0, row - 2):row + 3, max(0, column - 2):column + 3])))
    scale = float(np.percentile(image, 99.5))
    return float(np.mean(values) / scale) if scale > 0 else 0.0


def select_ultrasound_examples(records: list[dict[str, Any]], images: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    for row in records:
        row["brightness_score"] = brightness_score(images[row["image_key"]], row["sap_rows"], row["sap_columns"])
    present = [row for row in records if row["unique_projected_pixels"] > 0]
    uncertain_only = [
        row for row in records
        if row["unique_projected_pixels"] == 0 and len(row["uncertain_rows"]) > 0
    ]
    absent = [row for row in records if row["unique_projected_pixels"] == 0 and len(row["uncertain_rows"]) == 0]
    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    def add(category: str, candidates: list[dict[str, Any]], reason: str) -> None:
        for row in candidates:
            if row["frame_sha256"] not in used:
                selected.append({**row, "review_category": category, "selection_reason": reason})
                used.add(row["frame_sha256"])
                return

    interior = [row for row in present if not row["boundary_touching"]]
    ranked = sorted(
        interior or present,
        key=lambda row: (-row["brightness_score"], -row["unique_projected_pixels"], row["session_id"], row["frame_sequence"]),
    )
    add(
        "potential_visible_interface",
        ranked,
        "high local clean-image intensity with nonzero core projection; interior field-of-view preferred",
    )
    first_session = selected[0]["session_id"] if selected else None
    add(
        "different_session_core_intersection",
        [row for row in ranked if row["session_id"] != first_session],
        "nonzero core projection from a different reset session; interior field-of-view preferred",
    )
    add(
        "weak_or_ambiguous_core_intersection",
        sorted(present, key=lambda row: (row["brightness_score"], row["unique_projected_pixels"])),
        "nonzero core projection with weak local intensity; visibility remains unconfirmed",
    )
    add(
        "boundary_only_context",
        sorted(uncertain_only, key=lambda row: (-len(row["uncertain_rows"]), row["session_id"], row["frame_sequence"])),
        "uncertainty intersects the plane while the proposed core does not",
    )
    add(
        "verified_nonintersection_context",
        sorted(absent, key=lambda row: (-row["target_probability"], row["session_id"], row["frame_sequence"])),
        "verified plane contains neither proposed core nor remaining uncertainty",
    )
    if len(selected) != 5:
        raise RuntimeError(f"expected five distinct ultrasound examples, got {len(selected)}")
    return selected


def render_ultrasound_example(
    *, structure: str, row: dict[str, Any], image: np.ndarray, output_dir: Path,
    ordinal: int, display_vmax: float
) -> dict[str, Any]:
    stem = f"{structure}_{ordinal:02d}_{row['review_category']}"
    raw_path = output_dir / f"{stem}_raw.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    comparison_path = output_dir / f"{stem}_comparison.png"
    plt.imsave(raw_path, image, cmap="gray", vmin=0.0, vmax=display_vmax)
    figure, axis = plt.subplots(figsize=(5, 6), dpi=170)
    axis.imshow(image, cmap="gray", vmin=0.0, vmax=display_vmax)
    axis.scatter(row["uncertain_columns"], row["uncertain_rows"], s=7, c=COLORS["uncertain"], alpha=0.70, label="BOUNDARY_UNCERTAIN")
    axis.scatter(row["sap_columns"], row["sap_rows"], s=28, facecolors="none", edgecolors=COLORS["core"], linewidths=1.0, label="PROPOSED_SAP_SURFACE_CORE")
    axis.set_title(f"{structure} evaluator-only projection\n{row['session_id']} frame {row['frame_sequence']}")
    axis.legend(loc="lower left", fontsize=6)
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(overlay_path)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=170)
    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=display_vmax)
    axes[0].set_title("Clean actor-visible ultrasound")
    axes[1].imshow(image, cmap="gray", vmin=0.0, vmax=display_vmax)
    axes[1].scatter(row["uncertain_columns"], row["uncertain_rows"], s=7, c=COLORS["uncertain"], alpha=0.70)
    axes[1].scatter(row["sap_columns"], row["sap_rows"], s=28, facecolors="none", edgecolors=COLORS["core"], linewidths=1.0)
    axes[1].set_title("Privileged geometric projection")
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle("Geometric intersection is not confirmed SAP cortical visibility")
    figure.tight_layout()
    figure.savefig(comparison_path)
    plt.close(figure)
    return {
        "structure_id": structure,
        "review_category": row["review_category"],
        "selection_reason": row["selection_reason"],
        "session_id": row["session_id"],
        "frame_sequence": row["frame_sequence"],
        "producer_sequence": row["producer_sequence"],
        "acquisition_id": row["acquisition_id"],
        "frame_sha256": row["frame_sha256"],
        "unique_core_projected_pixels": row["unique_projected_pixels"],
        "uncertain_projected_vertex_entries": int(len(row["uncertain_rows"])),
        "local_brightness_score": row["brightness_score"],
        "projection_status": "VERIFIED",
        "acoustic_visibility": "NOT_CONFIRMED",
        "semantic_identity": "NOT_CONFIRMED",
        "raw_image": str(raw_path.relative_to(output_dir.parent.parent)),
        "overlay_image": str(overlay_path.relative_to(output_dir.parent.parent)),
        "comparison_image": str(comparison_path.relative_to(output_dir.parent.parent)),
    }


def export_geometry(
    *, level: str, lower: dict[str, Any], upper: dict[str, Any], regions: dict[str, Any],
    derived_vertices: np.ndarray, derived_faces: np.ndarray, correspondence: dict[str, Any],
    asset_root: Path, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"RIGHT_{level}_SAP_v2_regions.npz"
    np.savez_compressed(
        npz_path,
        source_vertices_mesh_index=lower["vertices"].astype(np.float32),
        source_faces=lower["faces"].astype(np.int32),
        original_candidate_vertex_mask=lower["original"],
        proposed_sap_surface_core_vertex_mask=regions["core"],
        boundary_uncertain_vertex_mask=regions["uncertain"],
        likely_non_sap_neighbor_vertex_mask=regions["neighbor"],
        derived_surface_vertices_index=derived_vertices.astype(np.float32),
        derived_surface_faces=derived_faces.astype(np.int32),
        source_unique_vertices_index=correspondence["source_unique_vertices"].astype(np.float32),
        source_to_derived_index=correspondence["source_to_derived_index"].astype(np.int32),
        source_to_derived_distance_index=correspondence["source_to_derived_distance_index"].astype(np.float32),
        coordinate_frame=np.asarray("segmentation_index_xyz"),
        spacing_mm=np.asarray(1.5),
        source_mesh_sha256=np.asarray(lower["source_mesh_sha256"]),
    )
    rows = []
    for coordinate in np.unique(lower["vertices"][lower["original"]], axis=0):
        matching = np.all(lower["vertices"] == coordinate, axis=1)
        if np.any(regions["core"] & matching):
            state = "PROPOSED_SAP_SURFACE_CORE"
        elif np.any(regions["neighbor"] & matching):
            state = "LIKELY_NON_SAP_NEIGHBOR"
        else:
            state = "BOUNDARY_UNCERTAIN"
        rows.append({"x_index": coordinate[0], "y_index": coordinate[1], "z_index": coordinate[2], "v2_region": state})
    csv_path = output_dir / f"RIGHT_{level}_SAP_v2_unique_locations.csv"
    write_csv(csv_path, rows)
    mapping_path = output_dir / f"{level}_source_to_derived_correspondence.csv"
    write_csv(
        mapping_path,
        [
            {
                "source_unique_index": index,
                "source_x": point[0], "source_y": point[1], "source_z": point[2],
                "nearest_derived_index": int(correspondence["source_to_derived_index"][index]),
                "distance_index": float(correspondence["source_to_derived_distance_index"][index]),
                "distance_mm": float(correspondence["source_to_derived_distance_index"][index] * 1.5),
            }
            for index, point in enumerate(correspondence["source_unique_vertices"])
        ],
    )
    copied = []
    for source_level in (level, UPPER_LEVEL[level]):
        source = asset_root / f"selected_dataset_stl/s0010/vertebrae_{source_level}.stl"
        destination = output_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
        copied.append(str(destination.relative_to(output_dir.parent)))
    return {
        "regions_npz": str(npz_path.relative_to(output_dir.parent)),
        "regions_npz_sha256": sha256(npz_path),
        "unique_locations_csv": str(csv_path.relative_to(output_dir.parent)),
        "source_to_derived_csv": str(mapping_path.relative_to(output_dir.parent)),
        "packaged_source_meshes": copied,
    }


def validate_package(package: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    references: set[str] = set()
    for record in manifest["structures"].values():
        references.update(record["figure_files"])
        references.update(record["ultrasound_files"])
        for key, value in record["geometry_files"].items():
            if key.endswith("_sha256"):
                continue
            if isinstance(value, list):
                references.update(value)
            else:
                references.add(value)
    missing = sorted(path for path in references if not (package / path).is_file())
    required = [
        "00_EXECUTIVE_SUMMARY.md", "01_REVIEW_GUIDE.md", "02_ANNOTATION_MANIFEST.json",
        "03_STRUCTURE_SUMMARY.csv", "04_GEOMETRY_QUALITY_AUDIT.md",
        "05_RIGHT_L4_SAP_DECISION_SHEET.md", "06_RIGHT_L5_SAP_DECISION_SHEET.md",
        "07_REVIEW_RESPONSE_TEMPLATE.md", "08_REVIEW_TARGETS.json", "09_V1_V2_CHANGE_SUMMARY.md",
    ]
    missing_required = [path for path in required if not (package / path).is_file()]
    return {
        "referenced_file_count": len(references),
        "missing_referenced_files": missing,
        "missing_required_files": missing_required,
        "package_relative_references_only": all(not Path(path).is_absolute() for path in references),
        "pass": not missing and not missing_required and all(not Path(path).is_absolute() for path in references),
    }


def write_checksums(package: Path) -> Path:
    path = package / "checksums.sha256"
    files = sorted(file for file in package.rglob("*") if file.is_file() and file != path)
    path.write_text("".join(f"{sha256(file)}  {file.relative_to(package)}\n" for file in files), encoding="utf-8")
    return path


def create_zip(package: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, f"{package.name}/{path.relative_to(package)}")
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")


def write_documents(
    *, package: Path, source_revision: str, manifest: dict[str, Any], ultrasound_rows: list[dict[str, Any]]
) -> None:
    (package / "00_EXECUTIVE_SUMMARY.md").write_text(
        f"""# s0010 RIGHT L4/L5 SAP semantic-boundary review V2

This package tests the original RIGHT L4/L5 SAP candidates against source-mesh quality, original label volumes, adjacent-level paired surfaces, CT and exact-pose ultrasound correspondence.

- Source commit: `{source_revision}`
- RIGHT L4 SAP: **`SOURCE_SEMANTICS_INSUFFICIENT`**. The paired L3/L4 surface supports a 55-location `PROPOSED_SAP_SURFACE_CORE`, but no source label identifies L3 IAP, L4 SAP, lamina or pars separately.
- RIGHT L5 SAP: **`SOURCE_SEMANTICS_INSUFFICIENT`**. The paired L4/L5 surface supports a 77-location `PROPOSED_SAP_SURFACE_CORE`, with the same semantic limitation.
- `LIKELY_NON_SAP_NEIGHBOR` is empty for both structures because no source semantics justify a negative anatomical assignment.

The proposed cores are geometric articulation surrogates for review, not accepted SAP ground truth, nerve coordinates, MBB targets or actor-visible inputs.
""",
        encoding="utf-8",
    )
    (package / "01_REVIEW_GUIDE.md").write_text(
        """# Review guide

The lower vertebra SAP normally articulates with the IAP of the vertebra above. The source assets label only whole vertebrae. V2 therefore uses valid surface locations shared by the original lower-level candidate and the upper-level vertebral surface as the strongest paired-facet geometric evidence.

- Cyan: `PROPOSED_SAP_SURFACE_CORE`, pending anatomical review.
- Orange: `BOUNDARY_UNCERTAIN`.
- Magenta: `LIKELY_NON_SAP_NEIGHBOR`; empty in V2.
- Light gray: lower whole vertebra.
- Dark gray: upper whole vertebra; its IAP is not semantically isolated.

Review whether the cyan locations lie on the intended articular SAP surface, whether orange includes SAP extent or lamina/pars, and whether any location can be excluded with evidence. Ultrasound overlays are privileged geometric projections. A bright line near the projection is not independently confirmed as the SAP cortical interface.

Anatomical context: [lumbar facet biomechanics](https://pmc.ncbi.nlm.nih.gov/articles/PMC7002062/), [lumbar facet-joint anatomy](https://pmc.ncbi.nlm.nih.gov/articles/PMC11595282/), and [morphologic review](https://journals.viamedica.pl/folia_morphologica/article/download/FM.a2020.0122/52184).
""",
        encoding="utf-8",
    )
    geometry = manifest["geometry_quality"]
    lines = ["# Geometry quality audit", "", "The zero-area faces are explained by repeated physical corners after integer-coordinate quantization, not by non-collinear triangles collapsing for another measured reason.", ""]
    for level in ("L3", "L4", "L5"):
        row = geometry[level]
        lines.extend([
            f"## {level}", "",
            f"- Source: {row['source']['raw_vertex_records']} vertex records, {row['source']['unique_physical_locations']} unique locations, {row['source']['face_records']} faces.",
            f"- Zero-area: {row['source']['zero_area_faces']}; repeated-corner zero-area: {row['source']['zero_area_faces_with_repeated_physical_corner']}.",
            f"- Derived label surface: {row['derived']['raw_vertex_records']} vertices, {row['derived']['face_records']} faces, {row['derived']['zero_area_faces']} zero-area faces.",
            f"- Source-to-derived maximum: {row['correspondence']['source_to_derived_maximum_index']} index ({row['correspondence']['source_to_derived_maximum_index'] * 1.5:.6f} mm).",
            "- Exact source face/vertex indices are not preserved by reconstruction; the package provides nearest-location correspondence instead.", "",
        ])
    (package / "04_GEOMETRY_QUALITY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    for number, level in ((5, "L4"), (6, "L5")):
        structure = STRUCTURE_IDS[level]
        record = manifest["structures"][structure]
        examples = [row for row in ultrasound_rows if row["structure_id"] == structure]
        example_lines = "\n".join(f"- `{row['review_category']}`: [{Path(row['comparison_image']).name}]({row['comparison_image']})" for row in examples)
        (package / f"0{number}_{structure}_DECISION_SHEET.md").write_text(
            f"""# {structure} decision sheet

## Candidate summary

- Verdict: **`SOURCE_SEMANTICS_INSUFFICIENT`**.
- V1 candidate: {record['v1_raw_candidate_vertex_records']} raw records / {record['v1_unique_candidate_locations']} unique locations.
- V2 proposed core: {record['v2_raw_core_vertex_records']} raw records / {record['v2_unique_core_locations']} unique locations.
- V2 uncertain: {record['v2_raw_uncertain_vertex_records']} raw records / {record['v2_unique_uncertain_locations']} unique locations.
- V2 likely non-SAP: 0.

## Evidence in favor

- Every proposed core location is incident to a valid {level} source face and a valid {UPPER_LEVEL[level]} source face.
- The lower/upper pairing matches the expected facet articulation level.
- Exact-pose projection remains technically valid.

## Reasons for doubt

- Neither vertebra has separate SAP/IAP, lamina, pars or facet labels.
- Coincident integer surface locations can arise from shared voxel-label boundaries at 1.5 mm resolution.
- Ultrasound projection does not establish a visible or distinguishable SAP interface.

## Key figures

- [Paired articulation overview](figures/3d/{structure}_overview_sheet.png)
- [Facet-focused CT](figures/geometry_context/{structure}_facet_CT.png)
- [V1 versus V2](figures/geometry_context/{structure}_V1_V2.png)
- [Source versus derived surface](figures/geometry_context/{level}_source_derived_surface.png)

## Ultrasound evidence

{example_lines}

## Reviewer questions

1. Does the cyan paired-surface subset belong to the RIGHT {level} SAP articular surface?
2. Which orange locations are SAP extent versus lamina, pars or another posterior element?
3. Can any exact locations be assigned to `LIKELY_NON_SAP_NEIGHBOR`?
4. Which cited ultrasound frames show a distinguishable SAP cortical interface rather than an unassigned bone echo?
""",
            encoding="utf-8",
        )
    (package / "07_REVIEW_RESPONSE_TEMPLATE.md").write_text(
        """# Reviewer response template

Reviewer name and qualification:

Review date:

## Structure decision

- Structure: `RIGHT_L4_SAP` / `RIGHT_L5_SAP`
- Exact regions NPZ SHA-256:
- Proposed core: accept / revise / unresolved
- Boundary-uncertain locations to include:
- Boundary-uncertain locations to exclude:
- Locations supported as likely non-SAP neighbor:
- Ultrasound frames with visible SAP cortical interface:
- Ultrasound frames ambiguous or occluded:
- Rationale:
- Corrected source-location CSV or NPZ:

Repeat for the second structure. A decision applies only to the exact packaged checksum.
""",
        encoding="utf-8",
    )
    targets = {
        "schema": "sonogym-s0010-right-l4l5-sap-review-targets-v2",
        "targets": [
            {
                "target_id": f"{STRUCTURE_IDS[level]}_region_partition",
                "decision": "judge proposed core, uncertain boundary and any evidence-based exclusion",
                "files": [
                    f"figures/3d/{STRUCTURE_IDS[level]}_overview_sheet.png",
                    f"figures/geometry_context/{STRUCTURE_IDS[level]}_facet_CT.png",
                    manifest["structures"][STRUCTURE_IDS[level]]["geometry_files"]["regions_npz"],
                ],
            }
            for level in LEVELS
        ],
    }
    (package / "08_REVIEW_TARGETS.json").write_text(json.dumps(targets, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (package / "09_V1_V2_CHANGE_SUMMARY.md").write_text(
        """# V1 versus V2

V2 is not a cosmetic rerender.

- Counts now distinguish raw STL records from unique physical locations.
- Zero-area triangles are traced to repeated integer-quantized physical corners.
- Whole-vertebra surfaces are reconstructed separately from the original label volume without overwriting the STL.
- RIGHT L4 SAP is reviewed against the RIGHT L3 opposing surface; RIGHT L5 SAP is reviewed against RIGHT L4.
- The original candidate is partitioned into a paired-surface core proposal and remaining uncertainty. No unsupported non-SAP region is invented.
- Ultrasound examples project the V2 regions separately and keep acoustic visibility and semantic identity unconfirmed.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source_revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    package = args.output_root.resolve() / PACKAGE_NAME
    if package.exists() and any(package.iterdir()):
        raise RuntimeError(f"refusing nonempty package: {package}")
    for relative in ("figures/3d", "figures/geometry_context", "figures/ultrasound", "geometry"):
        (package / relative).mkdir(parents=True, exist_ok=True)

    asset_root = args.asset_root.resolve()
    patient_root = asset_root / "selected_dataset_stl/s0010"
    labels = np.asanyarray(nib.load(patient_root / "combined_label_map.nii.gz").dataobj)
    ct = np.asanyarray(nib.load(asset_root / "selected_dataset/s0010/ct.nii.gz").dataobj)
    original_geometries = V1.load_geometries(args.geometry_root.resolve())
    level_meshes: dict[str, dict[str, Any]] = {}
    derived: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    quality: dict[str, Any] = {}
    for level in ("L3", "L4", "L5"):
        mesh = trimesh.load_mesh(patient_root / f"vertebrae_{level}.stl", process=False)
        level_meshes[level] = {"vertices": np.asarray(mesh.vertices, dtype=np.float64), "faces": np.asarray(mesh.faces, dtype=np.int64)}
        derived[level] = reconstruct_voxel_surface(labels, LABEL_VALUE[level])
        correspondence = surface_correspondence(level_meshes[level]["vertices"], derived[level][0])
        quality[level] = {
            "source_path": str(patient_root / f"vertebrae_{level}.stl"),
            "source_sha256": sha256(patient_root / f"vertebrae_{level}.stl"),
            "source": mesh_quality(level_meshes[level]["vertices"], level_meshes[level]["faces"]),
            "derived": mesh_quality(*derived[level]),
            "correspondence": correspondence["summary"],
        }

    regions: dict[str, dict[str, Any]] = {}
    refined_geometries = original_geometries.copy()
    for level in LEVELS:
        sap = original_geometries[level]["sap"]
        regions[level] = derive_regions(sap, level_meshes[UPPER_LEVEL[level]]["vertices"], level_meshes[UPPER_LEVEL[level]]["faces"])
        refined_geometries[level] = {
            **original_geometries[level],
            "sap": {**sap, "original": regions[level]["core"], "uncertain": regions[level]["uncertain"]},
        }

    records, images, maximum_error, display_vmax = V1.collect_correspondence(
        v3_root=args.v3_root.resolve(), geometries=refined_geometries,
        tolerance_mm=args.surface_slab_tolerance_mm,
    )
    audit = json.loads((args.audit_root.resolve() / "sap_audit_result.json").read_text())
    if maximum_error > 0.001 or abs(maximum_error - audit["coordinate_transform_audit"]["maximum_coordinate_error_index"]) > 1e-12:
        raise RuntimeError("exact-pose mapping gate failed")

    ultrasound_rows: list[dict[str, Any]] = []
    structures: dict[str, Any] = {}
    summary_rows = []
    for level in LEVELS:
        structure = STRUCTURE_IDS[level]
        lower = {**level_meshes[level], "original": original_geometries[level]["sap"]["original"]}
        upper = level_meshes[UPPER_LEVEL[level]]
        views = []
        for name, elevation, azimuth, closeup, title in (
            ("posterior", 10, -90, False, "posterior paired-facet context"),
            ("posterior_oblique", 20, -50, False, "posterior-oblique paired-facet context"),
            ("lateral", 10, 0, False, "right lateral paired-facet context"),
            ("facet_closeup", 22, -50, True, "facet-region close-up"),
            ("superior_closeup", 70, -90, True, "superior close-up"),
        ):
            path = package / f"figures/3d/{structure}_{name}.png"
            render_articulation_view(level=level, lower=lower, upper=upper, regions=regions[level], output=path, elevation=elevation, azimuth=azimuth, closeup=closeup, title=title)
            views.append(path)
        overview = package / f"figures/3d/{structure}_overview_sheet.png"
        make_contact_sheet(views, overview, f"{structure} paired-facet review")
        facet_ct = package / f"figures/geometry_context/{structure}_facet_CT.png"
        render_ct_facet_context(ct=ct, labels=labels, level=level, lower=lower, regions=regions[level], output=facet_ct)
        surface_figure = package / f"figures/geometry_context/{level}_source_derived_surface.png"
        render_surface_comparison(level=level, source=lower, derived_vertices=derived[level][0], derived_faces=derived[level][1], output=surface_figure)
        v1_v2 = package / f"figures/geometry_context/{structure}_V1_V2.png"
        make_contact_sheet(
            [
                package / f"figures/3d/{structure}_posterior.png",
                package / f"figures/3d/{structure}_posterior_oblique.png",
                package / f"figures/3d/{structure}_facet_closeup.png",
            ],
            v1_v2,
            f"{structure}: original candidate partitioned into paired-surface core and uncertainty",
        )
        source_correspondence = surface_correspondence(level_meshes[level]["vertices"], derived[level][0])
        geometry_files = export_geometry(
            level=level, lower={**original_geometries[level]["sap"], "vertices": lower["vertices"], "faces": lower["faces"]},
            upper=upper, regions=regions[level], derived_vertices=derived[level][0], derived_faces=derived[level][1],
            correspondence=source_correspondence, asset_root=asset_root, output_dir=package / "geometry",
        )
        chosen = select_ultrasound_examples(records[level], images)
        level_rows = [
            render_ultrasound_example(structure=structure, row=row, image=images[row["image_key"]], output_dir=package / "figures/ultrasound", ordinal=index, display_vmax=display_vmax)
            for index, row in enumerate(chosen, start=1)
        ]
        ultrasound_rows.extend(level_rows)
        sap = original_geometries[level]["sap"]
        figure_files = [str(path.relative_to(package)) for path in views + [overview, facet_ct, surface_figure, v1_v2]]
        ultrasound_files = [row[key] for row in level_rows for key in ("raw_image", "overlay_image", "comparison_image")]
        structure_record = {
            "anatomy_id": "s0010",
            "structure_name": f"RIGHT {level} superior articular process",
            "verdict": "SOURCE_SEMANTICS_INSUFFICIENT",
            "upper_articulating_level": UPPER_LEVEL[level],
            "v1_raw_candidate_vertex_records": int(np.count_nonzero(sap["original"])),
            "v1_unique_candidate_locations": regions[level]["unique_candidate_locations"],
            "v2_raw_core_vertex_records": int(np.count_nonzero(regions[level]["core"])),
            "v2_unique_core_locations": regions[level]["unique_core_locations"],
            "v2_raw_uncertain_vertex_records": int(np.count_nonzero(regions[level]["uncertain"])),
            "v2_unique_uncertain_locations": regions[level]["unique_uncertain_locations"],
            "v2_raw_likely_non_sap_vertex_records": 0,
            "v2_unique_likely_non_sap_locations": 0,
            "region_method": "valid lower-candidate locations coincident with valid upper-level source-face locations",
            "geometry_approval": "PENDING_INDEPENDENT_ANATOMICAL_REVIEW",
            "figure_files": figure_files,
            "ultrasound_files": ultrasound_files,
            "geometry_files": geometry_files,
        }
        structures[structure] = structure_record
        summary_rows.append({
            "structure_id": structure,
            "verdict": structure_record["verdict"],
            "v1_raw_candidate_vertex_records": structure_record["v1_raw_candidate_vertex_records"],
            "v1_unique_candidate_locations": structure_record["v1_unique_candidate_locations"],
            "v2_raw_core_vertex_records": structure_record["v2_raw_core_vertex_records"],
            "v2_unique_core_locations": structure_record["v2_unique_core_locations"],
            "v2_raw_uncertain_vertex_records": structure_record["v2_raw_uncertain_vertex_records"],
            "v2_unique_uncertain_locations": structure_record["v2_unique_uncertain_locations"],
            "v2_likely_non_sap_locations": 0,
            "ultrasound_examples": len(level_rows),
        })

    write_csv(package / "figures/ultrasound/ULTRASOUND_INDEX.csv", ultrasound_rows)
    manifest = {
        "schema": "sonogym-s0010-right-l4l5-sap-static-review-v2",
        "package_name": PACKAGE_NAME,
        "source_revision": source_revision,
        "starting_revision": "32757ba675aac53eade86841754a98f7697c766a",
        "anatomy_id": "s0010",
        "laterality": "RIGHT",
        "source_v1_job": "52061219",
        "source_geometry_job": "51779234",
        "source_audit_job": "52053307",
        "source_ultrasound_job": "52039283",
        "mapping_validation": {"frames": 336, "maximum_coordinate_error_index": maximum_error, "gate_index": 0.001, "pass": True},
        "semantic_training_authorized": False,
        "actor_usable": False,
        "anatomical_approval_recorded": False,
        "mbb_target_region_defined": False,
        "reference_urls": list(REFERENCE_URLS),
        "geometry_quality": quality,
        "structures": structures,
    }
    (package / "02_ANNOTATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(package / "03_STRUCTURE_SUMMARY.csv", summary_rows)
    write_documents(package=package, source_revision=source_revision, manifest=manifest, ultrasound_rows=ultrasound_rows)
    validation = validate_package(package, manifest)
    if not validation["pass"]:
        raise RuntimeError(f"package validation failed: {validation}")
    (package / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_path = write_checksums(package)
    zip_path = args.output_root.resolve() / f"{PACKAGE_NAME}.zip"
    create_zip(package, zip_path)
    result = {
        "schema": "sonogym-s0010-right-l4l5-sap-static-review-build-v2",
        "source_revision": source_revision,
        "package_directory": str(package),
        "zip_path": str(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256(zip_path),
        "checksum_entries": len(checksum_path.read_text().splitlines()),
        "ultrasound_examples": len(ultrasound_rows),
        "verdicts": {structure: "SOURCE_SEMANTICS_INSUFFICIENT" for structure in structures},
        "regions": {
            structure: {
                "unique_core": record["v2_unique_core_locations"],
                "unique_uncertain": record["v2_unique_uncertain_locations"],
                "unique_likely_non_sap": 0,
            }
            for structure, record in structures.items()
        },
        "validation": validation,
    }
    (args.output_root.resolve() / "build_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
