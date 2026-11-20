#!/usr/bin/env python3
"""Build the s0010 RIGHT L4/L5 SAP semantic review package.

The external model output and whole-vertebra labels are privileged annotation
evidence.  This script does not modify actor observations or promote a proposal
to semantic ground truth.
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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes, mesh_surface_area
import yaml


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_NAME = "s0010_right_l4l5_sap_semantic_review_v3"
LABEL_VALUE = {"L3": 29, "L4": 28, "L5": 27}
VERTEBRAL_SOURCE_LABELS = tuple(range(25, 32))
UPPER_LEVEL = {"L4": "L3", "L5": "L4"}
MODEL_LABEL = {
    0: "background",
    1: "vertebral_body",
    2: "pedicles",
    3: "lamina",
    4: "spinous_process",
    5: "transverse_process",
    6: "superior_articular_process",
    7: "inferior_articular_process",
}
COLORS = {
    "whole_lower": "#d7dee8",
    "whole_upper": "#a99abd",
    "sap": "#00bcd4",
    "iap": "#d627a8",
    "tp": "#2ca02c",
    "v2": "#ff8c00",
    "agreement": "#1b9e77",
    "model_only": "#377eb8",
    "v2_only": "#e41a1c",
    "conflict": "#984ea3",
}


def _load_geometry_module() -> Any:
    path = (
        ROOT
        / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation"
        / "ultrasound_plane_geometry.py"
    )
    spec = importlib.util.spec_from_file_location("sap_v3_ultrasound_geometry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GEOMETRY = _load_geometry_module()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_label_conversion(path: Path) -> dict[int, int]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"invalid label conversion: {path}")
    conversion = {int(source): int(target) for source, target in raw.items()}
    missing = sorted(set(VERTEBRAL_SOURCE_LABELS) - set(conversion))
    if missing:
        raise RuntimeError(f"label conversion misses vertebral labels: {missing}")
    if any(conversion[label] != 13 for label in VERTEBRAL_SOURCE_LABELS):
        raise RuntimeError("vertebral labels do not share acoustic bone class 13")
    return conversion


def anatomical_consistency_report(
    prediction: np.ndarray,
    segmentation: np.ndarray,
    label_conversion: dict[int, int],
) -> dict[str, Any]:
    foreground = prediction > 0
    vertebral = np.isin(segmentation, VERTEBRAL_SOURCE_LABELS)
    acoustic_bone_source_labels = tuple(
        sorted(label for label, acoustic in label_conversion.items() if acoustic == 13)
    )
    acoustic_bone = np.isin(segmentation, acoustic_bone_source_labels)

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        inside = foreground & mask
        outside = foreground & ~mask
        return {
            "foreground_voxel_count": int(np.count_nonzero(foreground)),
            "inside_count": int(np.count_nonzero(inside)),
            "outside_count": int(np.count_nonzero(outside)),
            "inside_fraction": float(
                np.count_nonzero(inside) / max(1, np.count_nonzero(foreground))
            ),
            "per_model_class": {
                MODEL_LABEL[label]: {
                    "total": int(np.count_nonzero(prediction == label)),
                    "inside": int(np.count_nonzero((prediction == label) & mask)),
                    "outside": int(np.count_nonzero((prediction == label) & ~mask)),
                }
                for label in range(1, 8)
            },
        }

    per_level: dict[str, Any] = {}
    for level, source_label in LABEL_VALUE.items():
        level_mask = segmentation == source_label
        per_level[level] = {
            "source_label": source_label,
            "foreground_voxel_count": int(np.count_nonzero(foreground & level_mask)),
            "per_model_class": {
                MODEL_LABEL[label]: int(
                    np.count_nonzero((prediction == label) & level_mask)
                )
                for label in range(1, 8)
            },
        }
    source_label_counts = {
        str(int(label)): int(np.count_nonzero(foreground & (segmentation == label)))
        for label in np.unique(segmentation[foreground])
    }
    return {
        "vertebral_source_labels": list(VERTEBRAL_SOURCE_LABELS),
        "acoustic_bone_class": 13,
        "acoustic_bone_source_labels": list(acoustic_bone_source_labels),
        "vertebral_labels_only": summarize(vertebral),
        "acoustic_bone_labels": summarize(acoustic_bone),
        "per_level": per_level,
        "foreground_by_source_label": source_label_counts,
        "note": (
            "Source label > 0 is not a bone definition. Vertebral consistency uses "
            "the explicit s0010 spinal labels 25-31; acoustic-bone consistency uses "
            "only source labels mapped to renderer class 13."
        ),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _json_float_list(values: Any) -> list[float]:
    return [float(value) for value in values]


def validate_ct(ct_path: Path, segmentation_path: Path) -> dict[str, Any]:
    ct_image = nib.load(ct_path)
    segmentation_image = nib.load(segmentation_path)
    ct = np.asanyarray(ct_image.dataobj)
    segmentation = np.asanyarray(segmentation_image.dataobj)
    if ct.shape != segmentation.shape:
        raise RuntimeError("CT and segmentation shapes differ")
    affine_error = float(np.max(np.abs(ct_image.affine - segmentation_image.affine)))
    if affine_error != 0.0:
        raise RuntimeError(f"CT and segmentation affines differ by {affine_error}")
    orientation = tuple(nib.aff2axcodes(ct_image.affine))
    if orientation != ("R", "A", "S"):
        raise RuntimeError(f"expected RAS CT orientation, got {orientation}")
    if not np.issubdtype(ct.dtype, np.signedinteger):
        raise RuntimeError(f"expected signed CT intensities, got {ct.dtype}")
    if not np.isfinite(ct).all() or float(np.ptp(ct)) < 1000.0:
        raise RuntimeError("CT intensities are invalid or effectively constant")
    levels: dict[str, Any] = {}
    for level, label in LABEL_VALUE.items():
        indices = np.argwhere(segmentation == label)
        if not len(indices):
            raise RuntimeError(f"missing {level} label {label}")
        world = nib.affines.apply_affine(ct_image.affine, indices)
        levels[level] = {
            "label": label,
            "voxel_count": int(len(indices)),
            "index_min": [int(value) for value in indices.min(axis=0)],
            "index_max": [int(value) for value in indices.max(axis=0)],
            "world_min_mm": _json_float_list(world.min(axis=0)),
            "world_max_mm": _json_float_list(world.max(axis=0)),
        }
    percentiles = np.percentile(ct, [0, 0.1, 1, 5, 50, 95, 99, 99.9, 100])
    return {
        "ct_path": str(ct_path.resolve()),
        "ct_sha256": sha256(ct_path),
        "segmentation_path": str(segmentation_path.resolve()),
        "segmentation_sha256": sha256(segmentation_path),
        "shape": [int(value) for value in ct.shape],
        "spacing_mm": _json_float_list(ct_image.header.get_zooms()[:3]),
        "orientation": list(orientation),
        "affine": [[float(value) for value in row] for row in ct_image.affine],
        "affine_max_abs_difference": affine_error,
        "intensity_dtype": str(ct.dtype),
        "intensity_min_hu": float(np.min(ct)),
        "intensity_max_hu": float(np.max(ct)),
        "intensity_mean_hu": float(np.mean(ct)),
        "intensity_std_hu": float(np.std(ct)),
        "intensity_percentiles_hu": _json_float_list(percentiles),
        "finite": True,
        "levels": levels,
    }


def render_ct_preflight(
    ct: np.ndarray, segmentation: np.ndarray, output: Path
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(11, 10), dpi=170)
    for row, level in enumerate(("L3", "L4", "L5")):
        points = np.argwhere(segmentation == LABEL_VALUE[level])
        center = np.rint(np.median(points, axis=0)).astype(int)
        slices = (
            (ct[center[0], :, :].T, segmentation[center[0], :, :].T, "Sagittal"),
            (ct[:, center[1], :].T, segmentation[:, center[1], :].T, "Coronal"),
            (ct[:, :, center[2]].T, segmentation[:, :, center[2]].T, "Axial"),
        )
        for column, (image, labels, name) in enumerate(slices):
            axis = axes[row, column]
            axis.imshow(np.clip(image, -300, 1200), cmap="gray", origin="lower")
            axis.contour(labels == LABEL_VALUE[level], [0.5], colors=["cyan"], linewidths=0.8)
            axis.set_title(f"{level} {name}")
            axis.set_axis_off()
    figure.suptitle("s0010 raw CT preflight — cyan is the existing whole-vertebra label")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def run_preflight(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = validate_ct(args.ct, args.segmentation)
    report["model_archive_path"] = str(args.model_archive.resolve())
    report["model_archive_sha256"] = sha256(args.model_archive)
    report["model_archive_expected_sha256"] = args.expected_model_sha256
    report["model_archive_checksum_pass"] = (
        report["model_archive_sha256"] == args.expected_model_sha256
    )
    if not report["model_archive_checksum_pass"]:
        raise RuntimeError("model archive checksum mismatch")
    ct = np.asanyarray(nib.load(args.ct).dataobj)
    segmentation = np.asanyarray(nib.load(args.segmentation).dataobj)
    render_ct_preflight(ct, segmentation, args.output_dir / "ct_preflight.png")
    write_json(args.output_dir / "ct_preflight.json", report)
    print(json.dumps(report, sort_keys=True))


def validate_prediction(
    prediction_path: Path, ct_path: Path, segmentation_path: Path
) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray, dict[str, Any]]:
    prediction_image = nib.load(prediction_path)
    ct_image = nib.load(ct_path)
    segmentation_image = nib.load(segmentation_path)
    prediction = np.asanyarray(prediction_image.dataobj).astype(np.uint8)
    segmentation = np.asanyarray(segmentation_image.dataobj)
    labels = sorted(int(value) for value in np.unique(prediction))
    unsupported = sorted(set(labels) - set(MODEL_LABEL))
    affine_error = float(np.max(np.abs(prediction_image.affine - ct_image.affine)))
    shape_match = prediction.shape == ct_image.shape == segmentation.shape
    report = {
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": sha256(prediction_path),
        "prediction_shape": [int(value) for value in prediction.shape],
        "prediction_spacing_mm": _json_float_list(prediction_image.header.get_zooms()[:3]),
        "prediction_orientation": list(nib.aff2axcodes(prediction_image.affine)),
        "prediction_affine": [
            [float(value) for value in row] for row in prediction_image.affine
        ],
        "input_output_affine_max_abs_difference": affine_error,
        "input_output_shape_match": shape_match,
        "labels_present": labels,
        "unsupported_labels": unsupported,
    }
    if not shape_match or affine_error > 1e-5:
        raise RuntimeError(f"prediction alignment failed: {report}")
    if unsupported or not set(range(1, 8)).issubset(labels):
        raise RuntimeError(f"prediction labels failed validation: {report}")
    return prediction_image, prediction, segmentation, report


def _world_x_by_index(affine: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if not np.allclose(affine[0, 1:3], 0.0):
        raise RuntimeError("laterality requires an axis-aligned RAS x coordinate")
    return affine[0, 0] * np.arange(shape[0], dtype=np.float64) + affine[0, 3]


def split_level_and_laterality(
    prediction: np.ndarray, segmentation: np.ndarray, affine: np.ndarray
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float]]:
    x_world = _world_x_by_index(affine, prediction.shape)
    output: dict[str, dict[str, np.ndarray]] = {}
    midlines: dict[str, float] = {}
    for level, whole_label in LABEL_VALUE.items():
        whole = segmentation == whole_label
        coordinates = np.argwhere(whole)
        midline = float(np.median(x_world[coordinates[:, 0]]))
        right = x_world[:, None, None] > midline
        output[level] = {
            name: (prediction == model_label) & whole & right
            for model_label, name in MODEL_LABEL.items()
            if model_label != 0
        }
        output[level]["whole_vertebra"] = whole
        output[level]["right_half"] = whole & right
        midlines[level] = midline
    return output, midlines


def boundary_world(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float64)
    boundary = mask & ~ndimage.binary_erosion(mask)
    return nib.affines.apply_affine(affine, np.argwhere(boundary))


def mask_geometry(mask: np.ndarray, affine: np.ndarray) -> dict[str, Any]:
    spacing = nib.affines.voxel_sizes(affine)
    voxel_count = int(np.count_nonzero(mask))
    if voxel_count == 0:
        return {
            "voxel_count": 0,
            "volume_mm3": 0.0,
            "surface_area_mm2": 0.0,
            "component_count": 0,
        }
    _, component_count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    vertices, faces, _, _ = marching_cubes(mask.astype(np.uint8), 0.5, spacing=spacing)
    return {
        "voxel_count": voxel_count,
        "volume_mm3": float(voxel_count * np.prod(spacing)),
        "surface_area_mm2": float(mesh_surface_area(vertices, faces)),
        "component_count": int(component_count),
    }


def load_v2_core(v2_package: Path, level: str) -> dict[str, Any]:
    path = v2_package / f"geometry/RIGHT_{level}_SAP_v2_regions.npz"
    archive = np.load(path)
    vertices = archive["source_vertices_mesh_index"].astype(np.float64)
    core_mask = archive["proposed_sap_surface_core_vertex_mask"].astype(bool)
    core = np.unique(vertices[core_mask], axis=0)
    return {
        "path": path,
        "sha256": sha256(path),
        "core_index": core,
        "source_mesh_sha256": str(archive["source_mesh_sha256"]),
    }


def compare_v2_model(
    *, level: str, model_mask: np.ndarray, prediction: np.ndarray,
    v2: dict[str, Any], affine: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    spacing = nib.affines.voxel_sizes(affine)
    tolerance_mm = float(np.max(spacing))
    model_boundary = boundary_world(model_mask, affine)
    v2_world = nib.affines.apply_affine(affine, v2["core_index"])
    if len(model_boundary):
        distances = cKDTree(model_boundary).query(v2_world)[0]
    else:
        distances = np.full(len(v2_world), np.inf)
    v2_voxels = np.rint(v2["core_index"]).astype(int)
    inside = np.all(
        (v2_voxels >= 0) & (v2_voxels < np.asarray(model_mask.shape)), axis=1
    )
    v2_seed = np.zeros(model_mask.shape, dtype=bool)
    v2_seed[tuple(v2_voxels[inside].T)] = True
    distance_to_v2 = ndimage.distance_transform_edt(~v2_seed, sampling=spacing)
    agreement = model_mask & (distance_to_v2 <= tolerance_mm)
    model_only = model_mask & ~agreement
    v2_agreement = distances <= tolerance_mm
    sampled_labels = np.full(len(v2_voxels), -1, dtype=np.int16)
    sampled_labels[inside] = prediction[tuple(v2_voxels[inside].T)].astype(np.int16)
    categories = np.empty(len(v2_voxels), dtype="<U48")
    for index, (proximal, sampled_label) in enumerate(
        zip(v2_agreement, sampled_labels)
    ):
        if sampled_label == 7:
            categories[index] = "CLASS_CONFLICT_SAMPLED_IAP"
        elif sampled_label not in (-1, 0, 6):
            categories[index] = "CLASS_CONFLICT_SAMPLED_OTHER"
        elif sampled_label == 6 and proximal:
            categories[index] = "PROXIMAL_SAMPLED_SAP"
        elif sampled_label == 6:
            categories[index] = "DISTANT_SAMPLED_SAP_UNRESOLVED"
        elif sampled_label == 0 and proximal:
            categories[index] = "PROXIMAL_BACKGROUND_UNCERTAIN"
        elif sampled_label == 0:
            categories[index] = "DISTANT_BACKGROUND_UNRESOLVED"
        else:
            categories[index] = "OUTSIDE_VOLUME_UNRESOLVED"
    voxel_volume_mm3 = float(np.prod(spacing))
    result = {
        "structure_id": f"RIGHT_{level}_SAP",
        "v2_regions_sha256": v2["sha256"],
        "v2_core_unique_location_count": int(len(v2_world)),
        "physical_consistency_tolerance_mm": tolerance_mm,
        "v2_core_within_tolerance_location_count": int(np.count_nonzero(v2_agreement)),
        "v2_core_outside_tolerance_location_count": int(np.count_nonzero(~v2_agreement)),
        "v2_core_within_tolerance_fraction": float(np.mean(v2_agreement)) if len(v2_agreement) else 0.0,
        "distance_proximity_is_not_semantic_agreement": True,
        "v2_sampled_model_label_counts": {
            MODEL_LABEL.get(int(label), "outside_or_unknown"): int(count)
            for label, count in zip(*np.unique(sampled_labels, return_counts=True))
        },
        "v2_to_model_surface_median_mm": float(np.median(distances)),
        "v2_to_model_surface_p95_mm": float(np.percentile(distances, 95)),
        "v2_to_model_surface_maximum_mm": float(np.max(distances)),
        "model_near_v2_core_voxel_count": int(np.count_nonzero(agreement)),
        "model_near_v2_core_volume_mm3": float(np.count_nonzero(agreement) * voxel_volume_mm3),
        "model_not_near_v2_core_voxel_count": int(np.count_nonzero(model_only)),
        "model_not_near_v2_core_volume_mm3": float(np.count_nonzero(model_only) * voxel_volume_mm3),
        "v2_only_volume_mm3": None,
        "v2_only_surface_area_mm2": None,
        "v2_only_measurement_note": "V2 is a sparse surface-point proposal; volumetric and surface-area measures are not defined.",
    }
    return result, {
        "agreement": agreement,
        "model_only": model_only,
        "v2_agreement": v2_agreement,
        "v2_distances_mm": distances,
        "v2_sampled_model_labels": sampled_labels,
        "v2_categories": categories,
    }


def _slice(array: np.ndarray, fixed_axis: int, coordinate: int) -> np.ndarray:
    if fixed_axis == 0:
        return array[coordinate, :, :].T
    if fixed_axis == 1:
        return array[:, coordinate, :].T
    return array[:, :, coordinate].T


def _project_points_to_slice(
    points: np.ndarray, fixed_axis: int, coordinate: int, tolerance: float = 0.75
) -> tuple[np.ndarray, np.ndarray]:
    if not len(points):
        return np.empty(0), np.empty(0)
    keep = np.abs(points[:, fixed_axis] - coordinate) <= tolerance
    horizontal, vertical = ((1, 2), (0, 2), (0, 1))[fixed_axis]
    return points[keep, horizontal], points[keep, vertical]


def render_ct_plane(
    *, ct: np.ndarray, structures: dict[str, np.ndarray], v2_points: np.ndarray,
    axis_index: int, coordinate: int, output: Path, title: str, zoom_center: np.ndarray | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 7), dpi=180)
    axis.imshow(np.clip(_slice(ct, axis_index, coordinate), -300, 1200), cmap="gray", origin="lower")
    color_items = (("sap", COLORS["sap"]), ("iap", COLORS["iap"]), ("tp", COLORS["tp"]))
    for name, color in color_items:
        plane = _slice(structures[name], axis_index, coordinate)
        if np.any(plane):
            axis.contour(plane, [0.5], colors=[color], linewidths=1.2)
    x, y = _project_points_to_slice(v2_points, axis_index, coordinate)
    axis.scatter(x, y, s=18, facecolors="none", edgecolors=COLORS["v2"], linewidths=0.9)
    if zoom_center is not None:
        horizontal, vertical = ((1, 2), (0, 2), (0, 1))[axis_index]
        axis.set_xlim(zoom_center[horizontal] - 25, zoom_center[horizontal] + 25)
        axis.set_ylim(zoom_center[vertical] - 25, zoom_center[vertical] + 25)
    axis.set_title(title)
    axis.set_xlabel("voxel coordinate; 1 voxel = 1.5 mm")
    axis.legend(
        handles=[
            Line2D([0], [0], color=COLORS["sap"], label="model SAP"),
            Line2D([0], [0], color=COLORS["iap"], label="model IAP"),
            Line2D([0], [0], color=COLORS["tp"], label="model TP"),
            Line2D([0], [0], marker="o", color="none", markeredgecolor=COLORS["v2"], label="V2 core"),
        ],
        fontsize=7,
    )
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def render_disagreement_sheet(
    *, ct: np.ndarray, level: str, structures: dict[str, np.ndarray],
    comparison_masks: dict[str, np.ndarray], v2: dict[str, Any], output: Path,
) -> None:
    model_points = np.argwhere(structures["sap"])
    center = np.rint(
        np.median(model_points if len(model_points) else v2["core_index"], axis=0)
    ).astype(int)
    figure, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=170)
    for column, (fixed, name) in enumerate(((2, "Axial"), (0, "Sagittal"), (1, "Coronal"))):
        coordinate = int(center[fixed])
        for row in range(2):
            axis = axes[row, column]
            axis.imshow(np.clip(_slice(ct, fixed, coordinate), -300, 1200), cmap="gray", origin="lower")
            if row == 0:
                for key, color in (("sap", COLORS["sap"]), ("iap", COLORS["iap"]), ("tp", COLORS["tp"])):
                    plane = _slice(structures[key], fixed, coordinate)
                    if np.any(plane):
                        axis.contour(plane, [0.5], colors=[color], linewidths=1.0)
                x, y = _project_points_to_slice(v2["core_index"], fixed, coordinate)
                axis.scatter(x, y, s=15, facecolors="none", edgecolors=COLORS["v2"])
                axis.set_title(f"{name}: model classes and V2")
            else:
                overlay = np.zeros(_slice(ct, fixed, coordinate).shape, dtype=np.uint8)
                overlay[_slice(comparison_masks["model_only"], fixed, coordinate)] = 1
                overlay[_slice(comparison_masks["agreement"], fixed, coordinate)] = 2
                masked = np.ma.masked_where(overlay == 0, overlay)
                axis.imshow(masked, cmap=ListedColormap([COLORS["model_only"], COLORS["agreement"]]), alpha=0.70, origin="lower", vmin=1, vmax=2)
                distances = comparison_masks["v2_distances_mm"]
                v2_only = v2["core_index"][distances > 1.5]
                x, y = _project_points_to_slice(v2_only, fixed, coordinate)
                axis.scatter(x, y, s=20, c=COLORS["v2_only"], marker="x")
                axis.set_title(f"{name}: distance proximity and non-proximity")
            horizontal, vertical = ((1, 2), (0, 2), (0, 1))[fixed]
            axis.set_xlim(center[horizontal] - 25, center[horizontal] + 25)
            axis.set_ylim(center[vertical] - 25, center[vertical] + 25)
            axis.set_axis_off()
    figure.suptitle(
        f"RIGHT {level} SAP review — green=MODEL_NEAR_V2_CORE; blue=MODEL_NOT_NEAR_V2_CORE; red x=V2_OUTSIDE_TOLERANCE"
    )
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def _surface(mask: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(mask):
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
    vertices, faces, _, _ = marching_cubes(mask.astype(np.uint8), 0.5)
    return nib.affines.apply_affine(affine, vertices), faces.astype(np.int64)


def _equal_3d(axis: Any, points: np.ndarray) -> None:
    center = np.mean(np.vstack((points.min(axis=0), points.max(axis=0))), axis=0)
    radius = max(5.0, float(np.max(np.ptp(points, axis=0)) / 2.0))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def render_3d(
    *, level: str, affine: np.ndarray, level_masks: dict[str, dict[str, np.ndarray]],
    v2: dict[str, Any], output: Path, elevation: float, azimuth: float,
    closeup: bool, title: str,
) -> None:
    upper = UPPER_LEVEL[level]
    surfaces = [
        (*_surface(level_masks[level]["whole_vertebra"], affine), COLORS["whole_lower"], 0.30),
        (*_surface(level_masks[upper]["whole_vertebra"], affine), COLORS["whole_upper"], 0.22),
        (*_surface(level_masks[level]["superior_articular_process"], affine), COLORS["sap"], 0.95),
        (*_surface(level_masks[upper]["inferior_articular_process"], affine), COLORS["iap"], 0.90),
        (*_surface(level_masks[level]["transverse_process"], affine), COLORS["tp"], 0.78),
    ]
    figure = plt.figure(figsize=(8, 7), dpi=180)
    axis = figure.add_subplot(111, projection="3d")
    all_points = []
    for vertices, faces, color, alpha in surfaces:
        if not len(vertices):
            continue
        selected_faces = faces[:: max(1, len(faces) // 12000)]
        axis.add_collection3d(
            Poly3DCollection(vertices[selected_faces], facecolor=color, edgecolor="none", alpha=alpha)
        )
        all_points.append(vertices)
    v2_world = nib.affines.apply_affine(affine, v2["core_index"])
    axis.scatter(*v2_world.T, s=15, c=COLORS["v2"], edgecolors="black", linewidths=0.25, depthshade=False)
    if closeup:
        target = np.vstack((boundary_world(level_masks[level]["superior_articular_process"], affine), v2_world))
        _equal_3d(axis, target)
    else:
        _equal_3d(axis, np.vstack(all_points))
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_xlabel("R/L (mm)")
    axis.set_ylabel("A/P (mm)")
    axis.set_zlabel("S/I (mm)")
    axis.set_title(f"RIGHT {level} SAP — {title}")
    axis.legend(
        handles=[
            Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["sap"], label="model SAP"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["iap"], label=f"model {upper} IAP"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["tp"], label="model TP"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["v2"], label="V2 core"),
        ], fontsize=7,
    )
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def collect_ultrasound_examples(
    *, v3_root: Path, points_by_level: dict[str, np.ndarray], output_dir: Path
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, list[dict[str, Any]]] = {"L4": [], "L5": []}
    for session_dir in sorted((v3_root / "stage2").glob("mbbpoc_v3_*")):
        frames = read_jsonl(session_dir / "public/frames.jsonl")
        poses = read_jsonl(session_dir / "privileged/full_pose_replay.jsonl")
        images = np.load(session_dir / "public/ultrasound.npz")["ultrasound"]
        coordinate_archive = np.load(session_dir / "privileged/pixel_to_volume_coordinates.npz")
        native_coordinates = coordinate_archive["human_img_coords"].astype(np.float64)
        local = coordinate_archive["native_local_image_coordinates_m"].astype(np.float64)
        for index, (frame, pose, image, native) in enumerate(zip(frames, poses, images, native_coordinates)):
            plane = GEOMETRY.plane_from_full_pose(
                end_effector_position_m=np.asarray(pose["human_to_ee_position_m"]),
                rotation_columns=np.asarray(pose["human_to_ee_rotation_columns"]),
                image_origin_height_m=float(pose["image_origin_height_m"]),
                local_coordinates_m=local,
                label_resolution_m=float(pose["label_resolution_m"]),
            )
            coordinate_error = float(np.max(np.abs(plane.coordinates_index - native)))
            if coordinate_error > 0.001:
                raise RuntimeError("archived exact-pose replay failed")
            for level, points in points_by_level.items():
                projection = GEOMETRY.project_points_to_pixels(
                    points_index=points,
                    image_origin_index=plane.origin_index,
                    rotation_columns=plane.rotation_columns,
                    width=150,
                    depth=200,
                    image_resolution_m=float(pose["image_resolution_m"]),
                    label_resolution_m=float(pose["label_resolution_m"]),
                    plane_tolerance_mm=1.0,
                )
                inside = projection["inside"]
                pixels = sorted(set(zip(projection["rows"][inside].tolist(), projection["columns"][inside].tolist())))
                if not pixels:
                    continue
                rows = np.asarray([pixel[0] for pixel in pixels], dtype=int)
                columns = np.asarray([pixel[1] for pixel in pixels], dtype=int)
                margin = int(min(rows.min(), 199 - rows.max(), columns.min(), 149 - columns.max()))
                local_values = [
                    float(np.max(image[max(0, row - 2):row + 3, max(0, column - 2):column + 3]))
                    for row, column in pixels
                ]
                candidates[level].append({
                    "session_id": session_dir.name,
                    "frame_sequence": index,
                    "producer_sequence": frame["producer_sequence"],
                    "acquisition_id": frame["acquisition_id"],
                    "frame_sha256": frame["frame_sha256"],
                    "coordinate_error_index": coordinate_error,
                    "rows": rows,
                    "columns": columns,
                    "unique_projected_pixels": len(pixels),
                    "field_of_view_margin_pixels": margin,
                    "local_intensity": float(np.mean(local_values)),
                    "image": np.asarray(image),
                })
    exported: list[dict[str, Any]] = []
    for level, rows in candidates.items():
        ranked = sorted(rows, key=lambda row: (-row["field_of_view_margin_pixels"], -row["unique_projected_pixels"], -row["local_intensity"]))
        selected = []
        used_sessions: set[str] = set()
        for row in ranked:
            if row["session_id"] not in used_sessions or not selected:
                selected.append(row)
                used_sessions.add(row["session_id"])
            if len(selected) == 2:
                break
        for ordinal, row in enumerate(selected, start=1):
            stem = f"RIGHT_{level}_SAP_model_projection_{ordinal:02d}"
            image = row.pop("image")
            vmax = float(np.percentile(image, 99.5))
            raw = output_dir / f"{stem}_raw.png"
            overlay = output_dir / f"{stem}_overlay.png"
            comparison = output_dir / f"{stem}_comparison.png"
            plt.imsave(raw, image, cmap="gray", vmin=0.0, vmax=vmax)
            figure, axis = plt.subplots(figsize=(5, 6), dpi=170)
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=vmax)
            axis.scatter(row["columns"], row["rows"], s=15, facecolors="none", edgecolors=COLORS["sap"])
            axis.set_title(f"Evaluator-only model SAP projection\n{row['session_id']} frame {row['frame_sequence']}")
            axis.set_axis_off()
            figure.tight_layout(); figure.savefig(overlay); plt.close(figure)
            figure, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=170)
            axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=vmax); axes[0].set_title("Clean ultrasound")
            axes[1].imshow(image, cmap="gray", vmin=0.0, vmax=vmax)
            axes[1].scatter(row["columns"], row["rows"], s=15, facecolors="none", edgecolors=COLORS["sap"])
            axes[1].set_title("Privileged geometric projection")
            for axis in axes: axis.set_axis_off()
            figure.suptitle("Projection does not establish acoustic visibility or SAP identity")
            figure.tight_layout(); figure.savefig(comparison); plt.close(figure)
            exported.append({
                **{key: value for key, value in row.items() if key not in {"rows", "columns"}},
                "structure_id": f"RIGHT_{level}_SAP",
                "raw_image": str(raw.relative_to(output_dir.parent.parent)),
                "overlay_image": str(overlay.relative_to(output_dir.parent.parent)),
                "comparison_image": str(comparison.relative_to(output_dir.parent.parent)),
                "acoustic_visibility": "NOT_REVIEWED",
                "semantic_identity": "NOT_REVIEWED",
            })
    return exported


def save_crop(
    *, ct_image: nib.Nifti1Image, segmentation: np.ndarray, prediction: np.ndarray,
    level_masks: dict[str, dict[str, np.ndarray]], comparisons: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    support = np.isin(segmentation, list(LABEL_VALUE.values()))
    indices = np.argwhere(support)
    start = np.maximum(indices.min(axis=0) - 5, 0)
    stop = np.minimum(indices.max(axis=0) + 6, np.asarray(segmentation.shape))
    slices = tuple(slice(int(start[axis]), int(stop[axis])) for axis in range(3))
    transform = np.eye(4)
    transform[:3, 3] = start
    crop_affine = ct_image.affine @ transform
    arrays: dict[str, np.ndarray] = {
        "model_prediction": prediction[slices].astype(np.uint8),
        "whole_vertebra_labels": segmentation[slices].astype(np.uint8),
    }
    for level in ("L4", "L5"):
        arrays[f"RIGHT_{level}_SAP"] = level_masks[level]["superior_articular_process"][slices].astype(np.uint8)
        arrays[f"RIGHT_{level}_IAP"] = level_masks[level]["inferior_articular_process"][slices].astype(np.uint8)
        arrays[f"RIGHT_{level}_TP"] = level_masks[level]["transverse_process"][slices].astype(np.uint8)
        arrays[f"RIGHT_{level}_MODEL_NEAR_V2_CORE"] = comparisons[level]["agreement"][slices].astype(np.uint8)
        arrays[f"RIGHT_{level}_MODEL_NOT_NEAR_V2_CORE"] = comparisons[level]["model_only"][slices].astype(np.uint8)
    records: dict[str, Any] = {
        "crop_start_index": [int(value) for value in start],
        "crop_stop_index_exclusive": [int(value) for value in stop],
        "crop_affine": [[float(value) for value in row] for row in crop_affine],
        "files": {},
    }
    for name, array in arrays.items():
        path = output_dir / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(array, crop_affine), path)
        roundtrip = nib.load(path)
        if roundtrip.shape != array.shape or not np.allclose(roundtrip.affine, crop_affine):
            raise RuntimeError(f"NIfTI round-trip failed for {path}")
        records["files"][name] = {"path": str(path.name), "sha256": sha256(path)}
    return records


def validate_package(package: Path, referenced: list[str]) -> dict[str, Any]:
    missing = sorted(path for path in referenced if not (package / path).is_file())
    required = [
        "00_EXECUTIVE_SUMMARY.md", "01_CT_PREFLIGHT.md", "02_MODEL_INFERENCE.md",
        "03_COMPARISON_RESULTS.csv", "03_COMPARISON_RESULTS.json", "04_ANATOMICAL_REVIEW.md",
        "05_RIGHT_L4_SAP_DECISION_SHEET.md", "06_RIGHT_L5_SAP_DECISION_SHEET.md",
        "07_REVIEW_RESPONSE_TEMPLATE.md", "08_PROVENANCE_AND_CHECKSUMS.json",
    ]
    missing_required = [path for path in required if not (package / path).is_file()]
    prohibited = [path.name for path in package.rglob("*") if path.is_file() and (path.suffix in {".model", ".pkl"} or "checkpoint" in path.name.lower())]
    return {
        "pass": not missing and not missing_required and not prohibited,
        "missing_referenced_files": missing,
        "missing_required_files": missing_required,
        "prohibited_weight_files": prohibited,
        "referenced_file_count": len(referenced),
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
            raise RuntimeError("ZIP integrity test failed")


def write_documents(
    *, package: Path, source_revision: str, preflight: dict[str, Any],
    inference: dict[str, Any], comparisons: list[dict[str, Any]],
    ultrasound_rows: list[dict[str, Any]], verdict: str, job_id: str,
) -> None:
    by_level = {row["structure_id"].split("_")[1]: row for row in comparisons}
    (package / "00_EXECUTIVE_SUMMARY.md").write_text(
        f"""# s0010 RIGHT L4/L5 SAP semantic review V3

V3 applies the independently pretrained Spine_Subregions model to the original s0010 CT and compares the semantic result with the frozen V2 surface proposals.

- Source revision: `{source_revision}`
- Inference job: `{job_id}`
- Verdict: **`{verdict}`**
- RIGHT L4 SAP: model voxels={by_level['L4']['model_sap']['voxel_count']}; V2 locations within 1.5 mm={by_level['L4']['v2_core_within_tolerance_location_count']} / {by_level['L4']['v2_core_unique_location_count']}.
- RIGHT L5 SAP: model voxels={by_level['L5']['model_sap']['voxel_count']}; V2 locations within 1.5 mm={by_level['L5']['v2_core_within_tolerance_location_count']} / {by_level['L5']['v2_core_unique_location_count']}.

The result is a model-generated research proposal for independent review. It is not accepted semantic ground truth, a nerve annotation, an MBB target, or an actor-visible input.
""",
        encoding="utf-8",
    )
    (package / "01_CT_PREFLIGHT.md").write_text(
        f"""# CT preflight

- CT: `{preflight['ct_path']}`
- CT SHA-256: `{preflight['ct_sha256']}`
- Whole-vertebra segmentation: `{preflight['segmentation_path']}`
- Segmentation SHA-256: `{preflight['segmentation_sha256']}`
- Shape: `{preflight['shape']}`
- Spacing: `{preflight['spacing_mm']}` mm
- Orientation: `{preflight['orientation']}`
- CT/segmentation affine maximum absolute difference: `{preflight['affine_max_abs_difference']}`
- Intensity range: `{preflight['intensity_min_hu']}` to `{preflight['intensity_max_hu']}` HU (`{preflight['intensity_dtype']}`)
- L3/L4/L5 voxel counts: `{preflight['levels']['L3']['voxel_count']}`, `{preflight['levels']['L4']['voxel_count']}`, `{preflight['levels']['L5']['voxel_count']}`

The actual CT intensity volume, not a vertebral mask or STL rasterization, was supplied to inference.
""",
        encoding="utf-8",
    )
    (package / "02_MODEL_INFERENCE.md").write_text(
        f"""# Model inference

- Model: MICN-Lab Spine_Subregions, release `V1` (pre-release, 2026-03-13).
- Official release archive SHA-256: `{inference['model_archive_sha256']}`.
- Checkpoint: `Task005`, `nnUNetTrainerV2`, `nnUNetPlansv2.1`, `3d_fullres`, `fold_0`, `model_final_checkpoint`.
- Runtime implementation: nnU-Net `1.7.1`; the official archive is a legacy nnU-Net checkpoint despite current README wording that names nnU-Net v2.
- License: University of Zurich academic non-commercial license; weights are not included in this package.
- Input/output affine maximum absolute difference: `{inference['input_output_affine_max_abs_difference']}`.
- Prediction labels present: `{inference['labels_present']}`.
- Model foreground inside explicit vertebral source labels 25-31: `{inference['anatomical_consistency']['vertebral_labels_only']['inside_count']}` / `{inference['anatomical_consistency']['vertebral_labels_only']['foreground_voxel_count']}` voxels.
- Model foreground inside source labels mapped to acoustic bone class 13: `{inference['anatomical_consistency']['acoustic_bone_labels']['inside_count']}` / `{inference['anatomical_consistency']['acoustic_bone_labels']['foreground_voxel_count']}` voxels.
- Runtime: `{inference['elapsed_seconds']}` seconds on `{inference['gpu_name']}` with `{inference['cpus']}` CPUs.
- Command: `{inference['command']}`

The raw full-volume prediction remains in HiPerGator runtime storage. This package contains only compact derived masks and review evidence.
""",
        encoding="utf-8",
    )
    (package / "04_ANATOMICAL_REVIEW.md").write_text(
        """# Anatomical review

The model predicts global SAP, IAP and TP classes. V3 assigns a prediction to a level only where it intersects the existing whole-vertebra label, then assigns RIGHT using increasing physical RAS x relative to that vertebra's median x coordinate.

- RIGHT L4 SAP is reviewed with RIGHT L3 IAP across the L3-L4 facet articulation.
- RIGHT L5 SAP is reviewed with RIGHT L4 IAP across the L4-L5 facet articulation.
- Existing whole-vertebra masks support level identity and alignment, not independent SAP correctness.
- `MODEL_NEAR_V2_CORE` and `MODEL_NOT_NEAR_V2_CORE` are distance categories, not semantic agreement or accuracy categories. Sampled model class is reported independently.
- CT resolution is 1.5 mm isotropic; partial volume and model domain shift can affect small articular-process boundaries.
- Ultrasound projection confirms only where proposed geometry intersects the archived plane. Acoustic visibility and semantic identity remain unreviewed.
""",
        encoding="utf-8",
    )
    for number, level in ((5, "L4"), (6, "L5")):
        row = by_level[level]
        us = [item for item in ultrasound_rows if item["structure_id"] == f"RIGHT_{level}_SAP"]
        us_lines = "\n".join(f"- [{Path(item['comparison_image']).name}]({item['comparison_image']}): {item['unique_projected_pixels']} projected pixels, field-of-view margin {item['field_of_view_margin_pixels']} pixels." for item in us) or "- No archived frame intersected the model proposal."
        (package / f"0{number}_RIGHT_{level}_SAP_DECISION_SHEET.md").write_text(
            f"""# RIGHT {level} SAP decision sheet

## Evidence

- Model SAP: {row['model_sap']['voxel_count']} voxels, {row['model_sap']['volume_mm3']:.3f} mm³, {row['model_sap']['component_count']} connected components.
- V2 locations within 1.5 mm: {row['v2_core_within_tolerance_location_count']} / {row['v2_core_unique_location_count']} locations; this is distance proximity, not semantic agreement.
- V2-to-model surface distance: median {row['v2_to_model_surface_median_mm']:.3f} mm, p95 {row['v2_to_model_surface_p95_mm']:.3f} mm, maximum {row['v2_to_model_surface_maximum_mm']:.3f} mm.
- Adjacent model IAP minimum surface distance: {row['sap_to_adjacent_iap_minimum_mm']:.3f} mm.

## Doubts

- Spine_Subregions has not been independently validated on s0010.
- Whole-vertebra overlap does not verify SAP versus lamina, IAP or TP.
- V2 remains a sparse geometric surface proposal without semantic acceptance.

## Inspect

- [Disagreement sheet](figures/disagreement/RIGHT_{level}_SAP_disagreement_sheet.png)
- [Facet-focused CT](figures/ct/RIGHT_{level}_SAP_facet_focus.png)
- [Paired 3D facet view](figures/3d/RIGHT_{level}_SAP_paired_facet.png)

## Ultrasound

{us_lines}

## Reviewer questions

1. Does the model SAP surface belong to the RIGHT {level} SAP rather than lamina or another posterior element?
2. Does the opposing RIGHT {UPPER_LEVEL[level]} IAP support the paired-facet interpretation?
3. Which model-only or V2-only regions require correction?
4. Is any archived ultrasound projection a distinguishable SAP cortical interface?
""",
            encoding="utf-8",
        )
    (package / "07_REVIEW_RESPONSE_TEMPLATE.md").write_text(
        """# Review response template

Reviewer name and qualification:

Review date:

- Structure: `RIGHT_L4_SAP` / `RIGHT_L5_SAP`
- Prediction artifact SHA-256:
- Decision: accept for provisional simulator research / revise / reject / unresolved
- Figure and numbered region:
- Requested correction in physical coordinates or attached mask:
- SAP/IAP/lamina/TP interpretation:
- Ultrasound visibility assessment, if any:
- Rationale and uncertainty:

A decision applies only to the exact artifact checksums in `08_PROVENANCE_AND_CHECKSUMS.json`.
""",
        encoding="utf-8",
    )


def run_build(args: argparse.Namespace) -> None:
    source_revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    package = args.output_root.resolve() / PACKAGE_NAME
    if package.exists() and any(package.iterdir()):
        raise RuntimeError(f"refusing nonempty package directory: {package}")
    for relative in ("figures/ct", "figures/3d", "figures/disagreement", "figures/ultrasound", "masks"):
        (package / relative).mkdir(parents=True, exist_ok=True)
    preflight = validate_ct(args.ct, args.segmentation)
    ct_image = nib.load(args.ct)
    ct = np.asanyarray(ct_image.dataobj)
    prediction_image, prediction, segmentation, prediction_report = validate_prediction(args.prediction, args.ct, args.segmentation)
    prediction_report["anatomical_consistency"] = anatomical_consistency_report(
        prediction, segmentation, load_label_conversion(args.label_conversion)
    )
    level_masks, midlines = split_level_and_laterality(prediction, segmentation, prediction_image.affine)
    comparisons: list[dict[str, Any]] = []
    comparison_masks: dict[str, dict[str, np.ndarray]] = {}
    v2_by_level: dict[str, dict[str, Any]] = {}
    figure_files: list[str] = []
    for level in ("L4", "L5"):
        v2 = load_v2_core(args.v2_package, level)
        v2_by_level[level] = v2
        model_sap = level_masks[level]["superior_articular_process"]
        comparison, masks = compare_v2_model(
            level=level,
            model_mask=model_sap,
            prediction=prediction,
            v2=v2,
            affine=prediction_image.affine,
        )
        comparison_masks[level] = masks
        upper = UPPER_LEVEL[level]
        adjacent_iap = level_masks[upper]["inferior_articular_process"]
        sap_surface = boundary_world(model_sap, prediction_image.affine)
        iap_surface = boundary_world(adjacent_iap, prediction_image.affine)
        if len(sap_surface) and len(iap_surface):
            paired_distances = cKDTree(iap_surface).query(sap_surface)[0]
            paired_minimum = float(np.min(paired_distances))
            paired_median = float(np.median(paired_distances))
        else:
            paired_minimum = paired_median = float("inf")
        comparison.update({
            "model_sap": mask_geometry(model_sap, prediction_image.affine),
            "adjacent_structure": f"RIGHT_{upper}_IAP",
            "adjacent_iap": mask_geometry(adjacent_iap, prediction_image.affine),
            "sap_to_adjacent_iap_minimum_mm": paired_minimum,
            "sap_to_adjacent_iap_median_mm": paired_median,
            "right_midline_world_x_mm": midlines[level],
            "model_sap_left_voxel_count": int(
                np.count_nonzero(
                    (prediction == 6)
                    & (segmentation == LABEL_VALUE[level])
                    & ~level_masks[level]["right_half"]
                )
            ),
        })
        comparisons.append(comparison)
        point_rows = []
        v2_world = nib.affines.apply_affine(prediction_image.affine, v2["core_index"])
        for index, (point_index, point_world) in enumerate(zip(v2["core_index"], v2_world)):
            sampled_label = int(masks["v2_sampled_model_labels"][index])
            point_rows.append({
                "structure_id": f"RIGHT_{level}_SAP",
                "x_index": point_index[0],
                "y_index": point_index[1],
                "z_index": point_index[2],
                "x_world_mm": point_world[0],
                "y_world_mm": point_world[1],
                "z_world_mm": point_world[2],
                "nearest_model_sap_surface_distance_mm": float(masks["v2_distances_mm"][index]),
                "sampled_model_label": sampled_label,
                "sampled_model_structure": MODEL_LABEL.get(sampled_label, "outside_or_unknown"),
                "review_category": str(masks["v2_categories"][index]),
            })
        point_path = package / f"masks/RIGHT_{level}_SAP_v2_point_comparison.csv"
        write_csv(point_path, point_rows)
        figure_files.append(str(point_path.relative_to(package)))
        points = np.argwhere(model_sap)
        center = np.rint(np.median(points if len(points) else v2["core_index"], axis=0)).astype(int)
        structures = {
            "sap": model_sap,
            "iap": adjacent_iap,
            "tp": level_masks[level]["transverse_process"],
        }
        for axis_index, name in ((2, "axial"), (0, "sagittal"), (1, "coronal")):
            output = package / f"figures/ct/RIGHT_{level}_SAP_{name}.png"
            render_ct_plane(ct=ct, structures=structures, v2_points=v2["core_index"], axis_index=axis_index, coordinate=int(center[axis_index]), output=output, title=f"RIGHT {level} SAP {name} CT overlay", zoom_center=None)
            figure_files.append(str(output.relative_to(package)))
        focus = package / f"figures/ct/RIGHT_{level}_SAP_facet_focus.png"
        render_ct_plane(ct=ct, structures=structures, v2_points=v2["core_index"], axis_index=2, coordinate=int(center[2]), output=focus, title=f"RIGHT {level} facet-focused axial CT", zoom_center=center)
        figure_files.append(str(focus.relative_to(package)))
        disagreement = package / f"figures/disagreement/RIGHT_{level}_SAP_disagreement_sheet.png"
        render_disagreement_sheet(ct=ct, level=level, structures=structures, comparison_masks=masks, v2=v2, output=disagreement)
        figure_files.append(str(disagreement.relative_to(package)))
        for stem, elevation, azimuth, closeup, title in (
            ("posterior", 10, -90, False, "posterior"),
            ("posterior_oblique", 20, -55, False, "posterior-oblique"),
            ("lateral", 10, 0, False, "right lateral"),
            ("facet_closeup", 20, -55, True, "facet close-up"),
            ("paired_facet", 30, -70, True, f"paired with RIGHT {upper} IAP"),
            ("v2_v3", 70, -90, True, "V2 versus model"),
        ):
            output = package / f"figures/3d/RIGHT_{level}_SAP_{stem}.png"
            render_3d(level=level, affine=prediction_image.affine, level_masks=level_masks, v2=v2, output=output, elevation=elevation, azimuth=azimuth, closeup=closeup, title=title)
            figure_files.append(str(output.relative_to(package)))
    points_by_level = {
        level: np.argwhere(
            level_masks[level]["superior_articular_process"]
            & ~ndimage.binary_erosion(level_masks[level]["superior_articular_process"])
        ).astype(np.float64)
        for level in ("L4", "L5")
    }
    ultrasound_rows = collect_ultrasound_examples(v3_root=args.archived_ultrasound_root, points_by_level=points_by_level, output_dir=package / "figures/ultrasound")
    if ultrasound_rows:
        write_csv(package / "figures/ultrasound/ULTRASOUND_INDEX.csv", ultrasound_rows)
        figure_files.append("figures/ultrasound/ULTRASOUND_INDEX.csv")
        for row in ultrasound_rows:
            figure_files.extend([row["raw_image"], row["overlay_image"], row["comparison_image"]])
    masks = save_crop(ct_image=ct_image, segmentation=segmentation, prediction=prediction, level_masks=level_masks, comparisons=comparison_masks, output_dir=package / "masks")
    for record in masks["files"].values():
        record["path"] = f"masks/{record['path']}"
        figure_files.append(record["path"])
    write_csv(package / "03_COMPARISON_RESULTS.csv", comparisons)
    write_json(package / "03_COMPARISON_RESULTS.json", comparisons)
    model_archive_sha = sha256(args.model_archive)
    if model_archive_sha != args.expected_model_sha256:
        raise RuntimeError("model archive checksum changed")
    inference = {
        **prediction_report,
        "model_release": "V1",
        "model_archive_sha256": model_archive_sha,
        "task": "Task005",
        "trainer": "nnUNetTrainerV2",
        "plans": "nnUNetPlansv2.1",
        "configuration": "3d_fullres",
        "fold": 0,
        "checkpoint": "model_final_checkpoint",
        "command": args.inference_command,
        "gpu_name": args.gpu_name,
        "cpus": args.cpus,
        "elapsed_seconds": args.elapsed_seconds,
    }
    nonempty = all(row["model_sap"]["voxel_count"] > 0 and row["adjacent_iap"]["voxel_count"] > 0 for row in comparisons)
    verdict = "MODEL_SAP_PROPOSAL_READY_FOR_REVIEW" if nonempty else "MODEL_PREDICTION_UNRELIABLE"
    write_documents(package=package, source_revision=source_revision, preflight=preflight, inference=inference, comparisons=comparisons, ultrasound_rows=ultrasound_rows, verdict=verdict, job_id=args.job_id)
    provenance = {
        "schema": "sonogym-s0010-right-l4l5-sap-semantic-review-v3",
        "source_revision": source_revision,
        "anatomy_id": "s0010",
        "laterality": "RIGHT",
        "verdict": verdict,
        "actor_usable": False,
        "semantic_ground_truth_accepted": False,
        "preflight": preflight,
        "inference": inference,
        "right_midlines_world_x_mm": midlines,
        "v2_sources": {level: {"path": str(v2_by_level[level]["path"]), "sha256": v2_by_level[level]["sha256"]} for level in ("L4", "L5")},
        "crop": masks,
        "figure_files": sorted(set(figure_files)),
    }
    write_json(package / "08_PROVENANCE_AND_CHECKSUMS.json", provenance)
    validation = validate_package(package, sorted(set(figure_files)))
    if not validation["pass"]:
        raise RuntimeError(f"package validation failed: {validation}")
    write_json(package / "validation.json", validation)
    write_checksums(package)
    zip_path = args.output_root.resolve() / f"{PACKAGE_NAME}.zip"
    create_zip(package, zip_path)
    result = {
        "schema": "sonogym-s0010-right-l4l5-sap-semantic-review-v3-build",
        "verdict": verdict,
        "package_directory": str(package),
        "zip_path": str(zip_path),
        "zip_sha256": sha256(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "comparisons": comparisons,
        "ultrasound_example_count": len(ultrasound_rows),
        "validation": validation,
    }
    write_json(args.output_root.resolve() / "build_result.json", result)
    print(json.dumps(result, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--ct", type=Path, required=True)
    preflight.add_argument("--segmentation", type=Path, required=True)
    preflight.add_argument("--model-archive", type=Path, required=True)
    preflight.add_argument("--expected-model-sha256", required=True)
    preflight.add_argument("--output-dir", type=Path, required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--ct", type=Path, required=True)
    build.add_argument("--segmentation", type=Path, required=True)
    build.add_argument("--prediction", type=Path, required=True)
    build.add_argument("--label-conversion", type=Path, required=True)
    build.add_argument("--model-archive", type=Path, required=True)
    build.add_argument("--expected-model-sha256", required=True)
    build.add_argument("--v2-package", type=Path, required=True)
    build.add_argument("--archived-ultrasound-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--job-id", required=True)
    build.add_argument("--inference-command", required=True)
    build.add_argument("--gpu-name", required=True)
    build.add_argument("--cpus", type=int, required=True)
    build.add_argument("--elapsed-seconds", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        run_preflight(args)
    else:
        run_build(args)


if __name__ == "__main__":
    main()
