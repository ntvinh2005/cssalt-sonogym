#!/usr/bin/env python3
"""Generate privileged s0010 anatomy annotations and bounded review figures on CPU."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import trimesh
import yaml
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.anatomical_ground_truth import (  # noqa: E402
    S0010_STRUCTURES,
    apply_affine,
    count_exact_shared_points,
    file_sha256,
    index_to_patient_local_m,
    patient_local_to_world_m,
    point_summary,
    quantile_region,
    validate_annotation,
)


PATIENT_ID = "s0010"
PATIENT_POSITION_M = np.array([0.2, -0.45, 1.0], dtype=np.float64)
PATIENT_ROTATION = Rotation.from_euler(
    "yxz", [-90.0, -90.0, 0.0], degrees=True
).as_matrix()
LABEL_SPACING_M = 0.0015
HOME_Z_INDEX = 195.0
LEVEL_COLORS = {
    "sacrum": "#6a3d9a",
    "S1": "#cab2d6",
    "L5": "#1f78b4",
    "L4": "#e31a1c",
    "L3": "#33a02c",
    "L2": "#ff7f00",
    "L1": "#b15928",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def source_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def artifact_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def mesh_label_alignment(
    mask: np.ndarray, vertices_index: np.ndarray, spacing_mm: np.ndarray
) -> dict[str, Any]:
    label_points = np.argwhere(mask)
    lower = np.maximum(np.min(label_points, axis=0) - 3, 0)
    upper = np.minimum(np.max(label_points, axis=0) + 4, np.array(mask.shape))
    cropped = mask[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))]
    outside_distance = distance_transform_edt(~cropped, sampling=spacing_mm)
    indices = np.clip(
        np.rint(vertices_index - lower).astype(int), 0, np.array(cropped.shape) - 1
    )
    mesh_to_label = outside_distance[tuple(indices.T)]
    boundary_points = np.argwhere(mask & ~binary_erosion(mask))
    label_to_mesh = cKDTree(vertices_index * spacing_mm).query(
        boundary_points * spacing_mm
    )[0]
    return {
        "mesh_vertex_to_segmentation_mm": {
            "median": float(np.median(mesh_to_label)),
            "p95": float(np.quantile(mesh_to_label, 0.95)),
            "maximum": float(np.max(mesh_to_label)),
        },
        "segmentation_boundary_to_mesh_vertex_mm": {
            "median": float(np.median(label_to_mesh)),
            "p95": float(np.quantile(label_to_mesh, 0.95)),
            "maximum": float(np.max(label_to_mesh)),
        },
    }


def geometry_in_frames(points_index: np.ndarray, affine: np.ndarray) -> dict[str, Any]:
    patient_local = index_to_patient_local_m(points_index, LABEL_SPACING_M)
    return {
        "mesh_index": point_summary(points_index),
        "nifti_world_RAS_mm": point_summary(apply_affine(points_index, affine)),
        "sonogym_patient_local_m": point_summary(patient_local),
        "sonogym_world_m": point_summary(
            patient_local_to_world_m(
                patient_local, PATIENT_ROTATION, PATIENT_POSITION_M
            )
        ),
    }


def annotation(
    *,
    level: str,
    raw_label: Any,
    name: str,
    laterality: str,
    annotation_type: str,
    geometry: Any,
    sources: list[dict[str, str]],
    method: str,
    confidence: str,
    uncertainty_mm: float | None,
    figures: list[str],
    limitations: list[str],
    status: str,
) -> dict[str, Any]:
    record = {
        "patient_id": PATIENT_ID,
        "anatomy_level": level,
        "raw_source_label": raw_label,
        "landmark_name": name,
        "laterality": laterality,
        "annotation_type": annotation_type,
        "coordinate_frame": (
            "mesh_index_with_nifti_RAS_and_sonogym_patient_world_transforms"
        ),
        "units": {
            "mesh_index": "voxel_index",
            "nifti_world_RAS": "mm",
            "sonogym_patient_local": "m",
            "sonogym_world": "m",
        },
        "geometry": geometry,
        "source_files": sources,
        "generation_method": method,
        "confidence": confidence,
        "positional_uncertainty_mm": uncertainty_mm,
        "supporting_visualizations": figures,
        "limitations": limitations,
        "status": status,
    }
    validate_annotation(record)
    return record


def group_a_annotations(
    segmentation: np.ndarray,
    affine: np.ndarray,
    spacing_mm: np.ndarray,
    mesh_root: Path,
    segmentation_source: dict[str, str],
    figure_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, trimesh.Trimesh], dict[str, Any]]:
    records = []
    meshes = {}
    measurements = {}
    for raw_label, (level, filename) in S0010_STRUCTURES.items():
        mesh_path = mesh_root / filename
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        label_points = np.argwhere(segmentation == raw_label).astype(np.float64)
        alignment = mesh_label_alignment(
            segmentation == raw_label, vertices, spacing_mm
        )
        posterior_indices, posterior_vertices = quantile_region(
            vertices, y_fraction=(0.0, 0.20)
        )
        label_centroid = np.mean(label_points, axis=0)
        mesh_centroid = np.asarray(mesh.centroid, dtype=np.float64)
        centroid_disagreement_mm = float(
            np.linalg.norm((label_centroid - mesh_centroid) * spacing_mm)
        )
        p95 = max(
            alignment["mesh_vertex_to_segmentation_mm"]["p95"],
            alignment["segmentation_boundary_to_mesh_vertex_mm"]["p95"],
        )
        uncertainty = float(
            max(
                math.sqrt(3.0) * spacing_mm[0] / 2.0,
                p95,
                centroid_disagreement_mm,
            )
        )
        figures = [artifact_path(figure_root / "lumbar_mesh_regions.png")]
        if level in {"L3", "L4", "L5"}:
            figures += [
                artifact_path(figure_root / f"ct_mpr_{level}.png"),
                artifact_path(figure_root / "ultrasound_correspondence.png"),
            ]
        geometry = {
            "segmentation": geometry_in_frames(label_points, affine),
            "mesh": geometry_in_frames(vertices, affine),
            "candidate_ultrasound_visible_posterior_surface": {
                "selector": "mesh_vertex_y_at_or_below_level_specific_20th_percentile",
                "selected_vertex_count": int(len(posterior_indices)),
                "frames": geometry_in_frames(posterior_vertices, affine),
            },
            "reference_centroid_home_relative_mm": float(
                (label_centroid[2] - HOME_Z_INDEX) * spacing_mm[2]
            ),
        }
        records.append(
            annotation(
                level=level,
                raw_label=raw_label,
                name=f"{level}_vertebral_reference",
                laterality="MIDLINE",
                annotation_type="surface",
                geometry=geometry,
                sources=[segmentation_source, source_record(mesh_path)],
                method="raw_label_and_matching_mesh_with_posterior_surface_region",
                confidence="HIGH",
                uncertainty_mm=uncertainty,
                figures=figures,
                limitations=[
                    "reference centroid is not an MBB target",
                    "posterior surface is a geometry candidate, not an image-recognition result",
                ],
                status="PROVISIONAL",
            )
        )
        meshes[level] = mesh
        measurements[level] = {
            "raw_label": raw_label,
            "segmentation_voxel_count": int(len(label_points)),
            "segmentation_centroid_index": label_centroid.tolist(),
            "segmentation_bounds_index": [
                np.min(label_points, axis=0).tolist(),
                np.max(label_points, axis=0).tolist(),
            ],
            "mesh_centroid_index": mesh_centroid.tolist(),
            "mesh_bounds_index": np.asarray(mesh.bounds).tolist(),
            "centroid_disagreement_mm": centroid_disagreement_mm,
            "surface_alignment": alignment,
            "positional_uncertainty_mm": uncertainty,
        }
    return records, meshes, measurements


def selected_region(
    mesh: trimesh.Trimesh,
    affine: np.ndarray,
    fractions: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ],
) -> tuple[np.ndarray, dict[str, Any]]:
    indices, points = quantile_region(
        np.asarray(mesh.vertices),
        x_fraction=fractions[0],
        y_fraction=fractions[1],
        z_fraction=fractions[2],
    )
    return points, {
        "selector": {
            "x_quantile_fraction": list(fractions[0]),
            "y_quantile_fraction": list(fractions[1]),
            "z_quantile_fraction": list(fractions[2]),
        },
        "selected_vertex_count": int(len(indices)),
        "frames": geometry_in_frames(points, affine),
    }


def group_b_annotations(
    meshes: dict[str, trimesh.Trimesh],
    affine: np.ndarray,
    mesh_root: Path,
    figure_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    records = []
    plot_regions = {}
    rejected_facet_candidates = []
    label_by_level = {level: label for label, (level, _) in S0010_STRUCTURES.items()}
    file_by_level = {
        level: filename for _, (level, filename) in S0010_STRUCTURES.items()
    }
    definitions = {
        "transverse_process": {
            "LEFT": ((0.00, 0.16), (0.20, 0.80), (0.25, 0.75)),
            "RIGHT": ((0.84, 1.00), (0.20, 0.80), (0.25, 0.75)),
        },
        "superior_articular_process": {
            "LEFT": ((0.18, 0.48), (0.00, 0.55), (0.72, 1.00)),
            "RIGHT": ((0.52, 0.82), (0.00, 0.55), (0.72, 1.00)),
        },
        "inferior_articular_process": {
            "LEFT": ((0.18, 0.48), (0.00, 0.55), (0.00, 0.28)),
            "RIGHT": ((0.52, 0.82), (0.00, 0.55), (0.00, 0.28)),
        },
    }
    for level in ("L3", "L4", "L5"):
        source = [source_record(mesh_root / file_by_level[level])]
        for name, sides in definitions.items():
            for side, fractions in sides.items():
                points, geometry = selected_region(meshes[level], affine, fractions)
                plot_regions[f"{level}_{side}_{name}"] = points
                status = (
                    "PROVISIONAL" if name == "transverse_process" else "UNRESOLVED"
                )
                records.append(
                    annotation(
                        level=level,
                        raw_label=label_by_level[level],
                        name=name,
                        laterality=side,
                        annotation_type="region",
                        geometry=geometry,
                        sources=source,
                        method="level_specific_mesh_quantile_region",
                        confidence="LOW" if status == "PROVISIONAL" else "UNRESOLVED",
                        uncertainty_mm=6.0 if status == "PROVISIONAL" else None,
                        figures=[
                            artifact_path(figure_root / "lumbar_mesh_regions.png")
                        ],
                        limitations=[
                            "no dedicated process segmentation exists",
                            "region requires anatomical review before supervised use",
                        ],
                        status=status,
                    )
                )
        for name in ("mammillary_process", "accessory_process"):
            records.append(
                annotation(
                    level=level,
                    raw_label=label_by_level[level],
                    name=name,
                    laterality="BILATERAL",
                    annotation_type="region",
                    geometry=None,
                    sources=source,
                    method="mesh_and_CT_review_no_reliable_separate_boundary",
                    confidence="UNRESOLVED",
                    uncertainty_mm=None,
                    figures=[
                        artifact_path(figure_root / "lumbar_mesh_regions.png")
                    ],
                    limitations=[
                        "available segmentation does not isolate this process",
                        "mesh quantiles cannot establish the named process",
                    ],
                    status="UNRESOLVED",
                )
            )

    sacrum_source = [source_record(mesh_root / file_by_level["sacrum"])]
    for side, x_range in (("LEFT", (0.0, 0.22)), ("RIGHT", (0.78, 1.0))):
        points, geometry = selected_region(
            meshes["sacrum"], affine, (x_range, (0.0, 0.75), (0.55, 1.0))
        )
        plot_regions[f"sacral_ala_{side}"] = points
        records.append(
            annotation(
                level="sacrum",
                raw_label=25,
                name="sacral_ala_region",
                laterality=side,
                annotation_type="region",
                geometry=geometry,
                sources=sacrum_source,
                method="superolateral_sacral_mesh_region",
                confidence="LOW",
                uncertainty_mm=8.0,
                figures=[artifact_path(figure_root / "lumbar_mesh_regions.png")],
                limitations=[
                    "sacral ala has no dedicated source label",
                    "region requires anatomical review before procedural use",
                ],
                status="PROVISIONAL",
            )
        )

    for side in ("LEFT", "RIGHT"):
        records.append(
            annotation(
                level="L4-L5",
                raw_label=[27, 28],
                name="MBB_TARGET_UNRESOLVED",
                laterality=side,
                annotation_type="region",
                geometry=None,
                sources=[
                    source_record(mesh_root / file_by_level["L4"]),
                    source_record(mesh_root / file_by_level["L5"]),
                ],
                method="no_nerve_segmentation_and_unresolved_process_boundaries",
                confidence="UNRESOLVED",
                uncertainty_mm=None,
                figures=[artifact_path(figure_root / "lumbar_mesh_regions.png")],
                limitations=[
                    "medial branch nerve is absent from the anatomy assets",
                    "pedicle screw trajectories are not MBB targets",
                    "no patient-specific procedural target is inferred",
                ],
                status="UNRESOLVED",
            )
        )
    for superior, inferior in (("L3", "L4"), ("L4", "L5"), ("L5", "S1")):
        full_mesh_shared_vertex_count = count_exact_shared_points(
            np.asarray(meshes[superior].vertices),
            np.asarray(meshes[inferior].vertices),
        )
        for side, x_range in (("LEFT", (0.18, 0.48)), ("RIGHT", (0.52, 0.82))):
            upper_points, _ = selected_region(
                meshes[superior], affine, (x_range, (0.0, 0.55), (0.0, 0.30))
            )
            lower_points, _ = selected_region(
                meshes[inferior], affine, (x_range, (0.0, 0.55), (0.70, 1.0))
            )
            distances, indices = cKDTree(lower_points).query(upper_points)
            upper_point = upper_points[int(np.argmin(distances))]
            lower_point = lower_points[int(indices[int(np.argmin(distances))])]
            midpoint = 0.5 * (upper_point + lower_point)
            points = np.stack([upper_point, lower_point, midpoint])
            separation_mm = float(
                np.linalg.norm(upper_point - lower_point) * LABEL_SPACING_M * 1000
            )
            shared_vertex_count = count_exact_shared_points(
                upper_points, lower_points
            )
            rejected_facet_candidates.append(
                {
                    "anatomy_level": f"{superior}-{inferior}",
                    "laterality": side,
                    "candidate_name": "facet_joint_region",
                    "raw_source_label": [
                        label_by_level[superior],
                        label_by_level[inferior],
                    ],
                    "candidate_surface_points": geometry_in_frames(points, affine),
                    "surface_separation_mm": separation_mm,
                    "shared_vertex_count_in_selected_regions": shared_vertex_count,
                    "shared_vertex_count_in_full_meshes": (
                        full_mesh_shared_vertex_count
                    ),
                    "selected_vertex_counts": {
                        "superior": int(len(upper_points)),
                        "inferior": int(len(lower_points)),
                    },
                    "generation_method": "closest_candidate_articular_regions",
                    "outcome": "REJECTED",
                    "reason": (
                        "The selected adjacent meshes share exact boundary vertices; "
                        "a nearest-point midpoint does not identify a facet joint."
                    ),
                    "source_files": [
                        source_record(mesh_root / file_by_level[superior]),
                        source_record(mesh_root / file_by_level[inferior]),
                    ],
                }
            )
            records.append(
                annotation(
                    level=f"{superior}-{inferior}",
                    raw_label=[
                        label_by_level[superior],
                        label_by_level[inferior],
                    ],
                    name="facet_joint_region",
                    laterality=side,
                    annotation_type="region",
                    geometry=None,
                    sources=[
                        source_record(mesh_root / file_by_level[superior]),
                        source_record(mesh_root / file_by_level[inferior]),
                    ],
                    method="shared_boundary_candidate_rejected_facet_unresolved",
                    confidence="UNRESOLVED",
                    uncertainty_mm=None,
                    figures=[
                        artifact_path(figure_root / "lumbar_mesh_regions.png")
                    ],
                    limitations=[
                        "the previous nearest-point candidate was rejected because adjacent meshes share exact boundary vertices",
                        "the available labels do not isolate the facet articular surfaces",
                        "cartilage and joint capsule are not separately segmented",
                        "the true facet joint remains unresolved and is not a procedural target",
                    ],
                    status="UNRESOLVED",
                )
            )
    return records, plot_regions, rejected_facet_candidates


def plot_mpr(
    ct: np.ndarray,
    segmentation: np.ndarray,
    level: str,
    raw_label: int,
    centroid: np.ndarray,
    output: Path,
) -> None:
    x, y, z = np.rint(centroid).astype(int)
    views = [
        ("Axial", ct[:, :, z].T, segmentation[:, :, z].T, (x, y), "Right (+x)", "Anterior (+y)"),
        ("Sagittal", ct[x, :, :].T, segmentation[x, :, :].T, (y, z), "Anterior (+y)", "Superior (+z)"),
        ("Coronal", ct[:, y, :].T, segmentation[:, y, :].T, (x, z), "Right (+x)", "Superior (+z)"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (title, image, labels, marker, xlabel, ylabel) in zip(axes, views):
        axis.imshow(np.clip(image, -300, 1200), cmap="gray", origin="lower")
        axis.contour(
            labels == raw_label,
            levels=[0.5],
            colors=[LEVEL_COLORS[level]],
            linewidths=1.2,
        )
        axis.scatter([marker[0]], [marker[1]], c="cyan", marker="+", s=80)
        axis.set_title(f"{title}: {level}")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
    figure.suptitle(
        "Evaluator-only CT/segmentation overlay; cyan is the raw-label centroid"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def plot_meshes(
    meshes: dict[str, trimesh.Trimesh],
    regions: dict[str, np.ndarray],
    output: Path,
) -> None:
    figure = plt.figure(figsize=(12, 9))
    axis = figure.add_subplot(111, projection="3d")
    rng = np.random.default_rng(20261111)
    for level in ("sacrum", "S1", "L5", "L4", "L3", "L2", "L1"):
        vertices = np.asarray(meshes[level].vertices)
        chosen = rng.choice(
            len(vertices), size=min(1800, len(vertices)), replace=False
        )
        points = vertices[chosen]
        axis.scatter(
            points[:, 0],
            points[:, 2],
            points[:, 1],
            s=0.8,
            alpha=0.30,
            color=LEVEL_COLORS[level],
            label=level,
        )
    for key, points in regions.items():
        if "L4" in key or "sacral_ala" in key:
            axis.scatter(
                points[:, 0],
                points[:, 2],
                points[:, 1],
                s=8,
                alpha=0.9,
                label=key,
            )
    axis.set_xlabel("Right (+x), mesh index")
    axis.set_ylabel("Superior (+z), mesh index")
    axis.set_zlabel("Anterior (+y), mesh index")
    axis.set_title("Lumbar geometry and provisional L4/sacral regions")
    axis.view_init(elev=12, azim=-90)
    axis.legend(loc="upper left", fontsize=6, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def ultrasound_cases(capture_root: Path) -> list[dict[str, Any]]:
    session_root = capture_root / "sessions" / "td_00"
    public = [
        json.loads(line)
        for line in (session_root / "public" / "frames.jsonl").read_text().splitlines()
        if line
    ]
    images = np.load(session_root / "public" / "ultrasound.npz")["ultrasound"]
    raw = np.load(session_root / "privileged" / "raw_labels.npz")["raw_labels"]
    definitions = [
        ("L4 positive forward", "OUTBOUND_CAUDAL", 28, None),
        ("L4 positive reverse", "RETURN_CRANIAL", 28, None),
        ("L3 difficult negative", "OUTBOUND_CAUDAL", 29, 28),
        ("L5 difficult negative", "RETURN_CRANIAL", 27, 28),
    ]
    cases = []
    for title, phase, wanted, forbidden in definitions:
        eligible = []
        for index, (row, labels) in enumerate(zip(public, raw)):
            if row["phase"] != phase:
                continue
            if forbidden is not None and np.any(labels == forbidden):
                continue
            eligible.append((int(np.count_nonzero(labels == wanted)), index))
        score, index = max(eligible)
        cases.append(
            {
                "title": title,
                "session_id": "td_00",
                "frame_sequence": int(index),
                "frame_sha256": public[index]["frame_sha256"],
                "phase": phase,
                "display_label": wanted,
                "raw_label_pixel_count": int(score),
                "image": images[index],
                "raw": raw[index],
            }
        )
    return cases


def plot_ultrasound(
    cases: list[dict[str, Any]], output: Path
) -> list[dict[str, Any]]:
    figure, axes = plt.subplots(len(cases), 2, figsize=(9, 13), constrained_layout=True)
    records = []
    for row_axes, case in zip(axes, cases):
        image = case["image"]
        mask = case["raw"] == case["display_label"]
        row_axes[0].imshow(image, cmap="gray", origin="upper")
        row_axes[0].set_title(case["title"] + " — actor-visible ultrasound")
        row_axes[1].imshow(image, cmap="gray", origin="upper")
        row_axes[1].contour(mask, levels=[0.5], colors=["red"], linewidths=1.0)
        row_axes[1].set_title("Evaluator-only raw-label footprint")
        for axis in row_axes:
            axis.set_xlabel("Image column")
            axis.set_ylabel("Image depth row")
        records.append(
            {key: value for key, value in case.items() if key not in {"image", "raw"}}
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return records


def capture_visibility(capture_root: Path) -> dict[str, Any]:
    totals = {
        label: {
            "name": name,
            "visible_frames": 0,
            "forward_frames": 0,
            "reverse_frames": 0,
            "sessions_visible": 0,
        }
        for label, (name, _) in S0010_STRUCTURES.items()
    }
    for session_root in sorted((capture_root / "sessions").iterdir()):
        if not session_root.is_dir():
            continue
        privileged = [
            json.loads(line)
            for line in (session_root / "privileged" / "annotations.jsonl")
            .read_text()
            .splitlines()
            if line
        ]
        public = [
            json.loads(line)
            for line in (session_root / "public" / "frames.jsonl")
            .read_text()
            .splitlines()
            if line
        ]
        seen = set()
        for actor, evaluator in zip(public, privileged):
            if actor["phase"] == "HOME_BASELINE":
                continue
            direction = (
                "forward_frames"
                if actor["phase"] == "OUTBOUND_CAUDAL"
                else "reverse_frames"
            )
            for label in totals:
                if int(evaluator["raw_label_pixel_counts"][str(label)]) > 0:
                    totals[label]["visible_frames"] += 1
                    totals[label][direction] += 1
                    seen.add(label)
        for label in seen:
            totals[label]["sessions_visible"] += 1
    return {str(label): values for label, values in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument(
        "--unity-commit",
        default="f7e182b5e2ac71cc73ac41f8b2a928ce10e220c0",
    )
    args = parser.parse_args()

    ct_path = args.asset_root / "selected_dataset" / PATIENT_ID / "ct.nii.gz"
    mesh_root = args.asset_root / "selected_dataset_stl" / PATIENT_ID
    segmentation_path = mesh_root / "combined_label_map.nii.gz"
    ct_image = nib.load(ct_path)
    segmentation_image = nib.load(segmentation_path)
    ct = np.asarray(ct_image.dataobj)
    segmentation = np.asarray(segmentation_image.dataobj)
    spacing_mm = np.asarray(ct_image.header.get_zooms()[:3], dtype=np.float64)
    affine_difference = float(
        np.max(np.abs(ct_image.affine - segmentation_image.affine))
    )
    if ct.shape != segmentation.shape or affine_difference > 1e-9:
        raise RuntimeError("CT and segmentation geometry disagree")
    if tuple(nib.aff2axcodes(ct_image.affine)) != ("R", "A", "S"):
        raise RuntimeError("unexpected CT orientation")
    label_conversion_path = (
        REPOSITORY_ROOT
        / "source"
        / "spinal_surgery"
        / "spinal_surgery"
        / "lab"
        / "sensors"
        / "cfgs"
        / "label_conversion.yaml"
    )
    label_conversion = yaml.safe_load(
        label_conversion_path.read_text(encoding="utf-8")
    )
    resolved_acoustic_labels = {
        str(label): int(label_conversion[label]) for label in S0010_STRUCTURES
    }
    if set(resolved_acoustic_labels.values()) != {13}:
        raise RuntimeError("vertebral labels do not share the expected bone acoustic class")

    segmentation_source = source_record(segmentation_path)
    group_a, meshes, measurements = group_a_annotations(
        segmentation,
        ct_image.affine,
        spacing_mm,
        mesh_root,
        segmentation_source,
        args.figure_dir,
    )
    group_b, regions, rejected_facet_candidates = group_b_annotations(
        meshes, ct_image.affine, mesh_root, args.figure_dir
    )
    for raw_label, (level, _) in S0010_STRUCTURES.items():
        if level in {"L3", "L4", "L5"}:
            plot_mpr(
                ct,
                segmentation,
                level,
                raw_label,
                np.asarray(measurements[level]["segmentation_centroid_index"]),
                args.figure_dir / f"ct_mpr_{level}.png",
            )
    plot_meshes(meshes, regions, args.figure_dir / "lumbar_mesh_regions.png")
    cases = ultrasound_cases(args.capture_root)
    ultrasound_records = plot_ultrasound(
        cases, args.figure_dir / "ultrasound_correspondence.png"
    )
    figure_paths = [
        args.figure_dir / "ct_mpr_L3.png",
        args.figure_dir / "ct_mpr_L4.png",
        args.figure_dir / "ct_mpr_L5.png",
        args.figure_dir / "lumbar_mesh_regions.png",
        args.figure_dir / "ultrasound_correspondence.png",
    ]

    sample = np.asarray(
        [measurements[level]["segmentation_centroid_index"] for level in measurements]
    )
    patient_local = index_to_patient_local_m(sample, LABEL_SPACING_M)
    patient_world = patient_local_to_world_m(
        patient_local, PATIENT_ROTATION, PATIENT_POSITION_M
    )
    reconstructed = (patient_world - PATIENT_POSITION_M) @ PATIENT_ROTATION
    round_trip_error_m = float(np.max(np.abs(reconstructed - patient_local)))
    if round_trip_error_m > 1e-9:
        raise RuntimeError("patient transform round trip failed")
    alignment_p95 = max(
        max(
            values["surface_alignment"]["mesh_vertex_to_segmentation_mm"]["p95"],
            values["surface_alignment"][
                "segmentation_boundary_to_mesh_vertex_mm"
            ]["p95"],
        )
        for values in measurements.values()
    )
    alignment_limit_mm = float(np.linalg.norm(spacing_mm))
    if alignment_p95 > alignment_limit_mm:
        raise RuntimeError("mesh and segmentation surface alignment exceeds one voxel diagonal")
    annotations = group_a + group_b
    for record in annotations:
        validate_annotation(record)

    result = {
        "schema": "sonogym-s0010-anatomical-annotations-v2",
        "status": "PROVISIONAL",
        "patient_id": PATIENT_ID,
        "provenance": {
            "cssalt_sonogym_commit": git_revision(),
            "unity_verified_commit": args.unity_commit,
            "ct": source_record(ct_path),
            "segmentation": segmentation_source,
            "capture_root": str(args.capture_root.resolve()),
            "capture_protocol_sha256": file_sha256(
                args.capture_root / "context" / "protocol.yaml"
            ),
            "label_conversion": source_record(
                label_conversion_path
            ),
            "resolved_raw_to_acoustic_labels": resolved_acoustic_labels,
            "generation_command": [
                sys.executable,
                str(Path(__file__).resolve()),
                "--asset-root",
                str(args.asset_root.resolve()),
                "--capture-root",
                str(args.capture_root.resolve()),
                "--output-json",
                str(args.output_json.resolve()),
                "--figure-dir",
                str(args.figure_dir.resolve()),
                "--unity-commit",
                args.unity_commit,
            ],
            "software": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "nibabel": importlib.metadata.version("nibabel"),
                "trimesh": importlib.metadata.version("trimesh"),
                "matplotlib": importlib.metadata.version("matplotlib"),
                "scipy": importlib.metadata.version("scipy"),
            },
            "visualizations": [
                {
                    "path": artifact_path(path),
                    "sha256": file_sha256(path),
                }
                for path in figure_paths
            ],
        },
        "coordinate_systems": {
            "nifti_shape": list(ct.shape),
            "voxel_spacing_mm": spacing_mm.tolist(),
            "nifti_affine": ct_image.affine.tolist(),
            "nifti_orientation": list(nib.aff2axcodes(ct_image.affine)),
            "mesh_frame": "NIfTI voxel-index coordinates",
            "sonogym_patient_local": {
                "scale_m_per_index": LABEL_SPACING_M,
                "nifti_affine_translation_not_applied": True,
            },
            "sonogym_patient_to_world": {
                "translation_m": PATIENT_POSITION_M.tolist(),
                "rotation_matrix": PATIENT_ROTATION.tolist(),
                "source_euler_yxz_degrees": [-90.0, -90.0, 0.0],
            },
            "left_right": "NIfTI RAS +x is patient right before simulator root transform",
            "cranial_caudal": "NIfTI RAS +z is cranial before simulator root transform",
        },
        "validation": {
            "ct_segmentation_shape_equal": ct.shape == segmentation.shape,
            "ct_segmentation_affine_max_abs_difference": affine_difference,
            "patient_transform_round_trip_max_abs_error_m": round_trip_error_m,
            "maximum_bidirectional_mesh_segmentation_p95_mm": alignment_p95,
            "mesh_segmentation_p95_limit_mm": alignment_limit_mm,
            "major_structure_measurements": measurements,
            "rejected_facet_candidates": rejected_facet_candidates,
        },
        "annotations": {
            "group_A_unity_method_reproduction": group_a,
            "group_B_future_MBB_bony_geometry": group_b,
        },
        "ultrasound_correspondence": {
            "capture_visibility": capture_visibility(args.capture_root),
            "representative_frames": ultrasound_records,
            "interpretation": (
                "Raw-label presence and evaluator overlays establish geometric "
                "correspondence only; they do not establish actor-visible level "
                "discrimination."
            ),
        },
        "anatomical_references": [
            {
                "purpose": (
                    "medial branch course relative to superior articular, transverse, "
                    "mammillary, and accessory processes"
                ),
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7977091/",
            },
            {
                "purpose": (
                    "three-dimensional proximity to the superior articular process "
                    "and mamillo-accessory notch"
                ),
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11372964/",
            },
            {
                "purpose": (
                    "facet innervation and distinct L5 dorsal ramus relationship "
                    "to S1 and sacral ala"
                ),
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7362874/",
            },
        ],
        "gates": {
            "A0_source_anatomy_integrity": "PASS",
            "A1_major_vertebral_annotation_coverage": "PASS",
            "A2_fine_bony_landmark_feasibility": (
                "PASS_WITH_PROVISIONAL_AND_UNRESOLVED_LABELS"
            ),
            "A3_ultrasound_observability": (
                "INCONCLUSIVE_DISCRIMINATIVE_RECOGNITION_NOT_TESTED"
            ),
            "A4_readiness": {
                "unity_stage1_ultrasound_landmark_localization": "READY",
                "L4_best_valid_view_acquisition": "PROVISIONAL",
                "provisional_bony_MBB_target_region_evaluation": "BLOCKED",
                "simulated_instrument_positioning": "BLOCKED",
                "nerve_specific_MBB_validation": "BLOCKED",
            },
        },
        "information_boundary": {
            "privileged_annotation_only": True,
            "actor_inputs_unchanged": True,
            "raw_labels_or_geometry_permitted_at_inference": False,
        },
        "limitations": [
            "single fixed anatomy s0010",
            "fine bony regions use mesh quantiles rather than clinician-reviewed labels",
            "no medial branch nerve segmentation",
            "no image classifier was trained or evaluated",
            "no GPU capture was launched",
        ],
    }
    write_json(args.output_json, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "annotation_count": len(annotations),
                "output_json": str(args.output_json.resolve()),
                "figure_dir": str(args.figure_dir.resolve()),
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
