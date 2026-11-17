#!/usr/bin/env python3
"""Create an editable, adjacent-slice RIGHT SAP/TP review package.

This responds to review of package v1.  It adds context and editable geometry without changing
any annotation status.  Privileged CT, segmentation, and mesh products are review-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import trimesh
import yaml

import review_s0010_right_landmarks as base


SLICE_OFFSETS = (-4, -2, 0, 2, 4)
ROI_MARGIN_VOXELS = 12
CANDIDATE_IDS = (
    "L4_RIGHT_superior_articular_process",
    "L4_RIGHT_transverse_process",
    "L5_RIGHT_superior_articular_process",
    "L5_RIGHT_transverse_process",
)


def cropped_affine(affine: np.ndarray, lower_index: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(lower_index, dtype=np.float64)
    return np.asarray(affine, dtype=np.float64) @ transform


def crop_l4_l5(
    ct_image: nib.Nifti1Image,
    segmentation_image: nib.Nifti1Image,
    *,
    margin: int,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image, dict[str, Any]]:
    segmentation = np.asanyarray(segmentation_image.dataobj)
    mask = np.isin(segmentation, [base.LABEL_BY_LEVEL["L4"], base.LABEL_BY_LEVEL["L5"]])
    indices = np.argwhere(mask)
    if not len(indices):
        raise RuntimeError("L4/L5 labels are absent from the segmentation")
    lower = np.maximum(np.min(indices, axis=0) - margin, 0)
    upper = np.minimum(np.max(indices, axis=0) + margin + 1, segmentation.shape)
    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))
    affine = cropped_affine(segmentation_image.affine, lower)
    ct_crop = np.asanyarray(ct_image.dataobj)[slices]
    segmentation_crop = segmentation[slices]
    ct_result = nib.Nifti1Image(ct_crop, affine, header=ct_image.header.copy())
    segmentation_result = nib.Nifti1Image(
        segmentation_crop, affine, header=segmentation_image.header.copy()
    )
    metadata = {
        "lower_source_index_inclusive": lower.tolist(),
        "upper_source_index_exclusive": upper.tolist(),
        "shape": list(ct_crop.shape),
        "voxel_spacing_mm": [float(value) for value in ct_image.header.get_zooms()[:3]],
        "resampled": False,
        "margin_voxels": margin,
    }
    return ct_result, segmentation_result, metadata


def patch_roi(
    patch: np.ndarray, horizontal: int, vertical: int, image_shape: tuple[int, int]
) -> tuple[slice, slice]:
    low_h = max(int(np.floor(np.min(patch[:, horizontal]))) - ROI_MARGIN_VOXELS, 0)
    high_h = min(
        int(np.ceil(np.max(patch[:, horizontal]))) + ROI_MARGIN_VOXELS + 1,
        image_shape[1],
    )
    low_v = max(int(np.floor(np.min(patch[:, vertical]))) - ROI_MARGIN_VOXELS, 0)
    high_v = min(
        int(np.ceil(np.max(patch[:, vertical]))) + ROI_MARGIN_VOXELS + 1,
        image_shape[0],
    )
    return slice(low_v, high_v), slice(low_h, high_h)


def plot_adjacent_slices(
    *,
    ct: np.ndarray,
    segmentation: np.ndarray,
    level: str,
    landmark: str,
    patch: np.ndarray,
    output: Path,
) -> None:
    color = base.COLORS[landmark]
    centroid = np.rint(np.mean(patch, axis=0)).astype(int)
    raw_label = base.LABEL_BY_LEVEL[level]
    figure, axes = plt.subplots(
        3, len(SLICE_OFFSETS), figsize=(18, 10), constrained_layout=True
    )
    plane_specs = (
        ("Axial", 2, 0, 1, "Right (+x)", "Anterior (+y)"),
        ("Sagittal", 0, 1, 2, "Anterior (+y)", "Superior (+z)"),
        ("Coronal", 1, 0, 2, "Right (+x)", "Superior (+z)"),
    )
    for row, (plane, fixed_axis, horizontal, vertical, xlabel, ylabel) in enumerate(
        plane_specs
    ):
        center = int(centroid[fixed_axis])
        for column, offset in enumerate(SLICE_OFFSETS):
            coordinate = center + offset
            if fixed_axis == 2:
                image = ct[:, :, coordinate].T
                labels = segmentation[:, :, coordinate].T
            elif fixed_axis == 0:
                image = ct[coordinate, :, :].T
                labels = segmentation[coordinate, :, :].T
            else:
                image = ct[:, coordinate, :].T
                labels = segmentation[:, coordinate, :].T
            local = base.projected_points(
                patch, fixed_axis, coordinate, tolerance=1.0
            )
            axis = axes[row, column]
            axis.imshow(np.clip(image, -300, 1200), cmap="gray", origin="lower")
            axis.contour(
                labels == raw_label,
                levels=[0.5],
                colors=["lime"],
                linewidths=0.8,
            )
            if len(local):
                axis.scatter(
                    local[:, horizontal],
                    local[:, vertical],
                    s=9,
                    color=color,
                    alpha=0.70,
                )
            roi = patch_roi(patch, horizontal, vertical, image.shape)
            axis.set_xlim(roi[1].start, roi[1].stop)
            axis.set_ylim(roi[0].start, roi[0].stop)
            offset_mm = offset * base.LABEL_SPACING_M * 1000.0
            axis.set_title(f"{plane} {offset_mm:+.1f} mm")
            if column == 0:
                axis.set_ylabel(ylabel)
            if row == 2:
                axis.set_xlabel(xlabel)
    figure.suptitle(
        f"Evaluator-only adjacent slices: {level} RIGHT {landmark}\n"
        "green=raw vertebral label; blue/orange=quantile candidate; review-only"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def export_candidate_geometry(
    *,
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    source_sha256: str,
    stem: Path,
) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    selected_indices = np.flatnonzero(mask).astype(np.int32)
    selected = vertices[mask]
    npz_path = stem.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        vertices_mesh_index=vertices.astype(np.float32),
        faces=faces,
        selected_vertex_mask=mask,
        selected_vertex_indices=selected_indices,
        source_mesh_sha256=np.asarray(source_sha256),
    )
    csv_path = stem.with_suffix(".csv")
    patient_local = selected * base.LABEL_SPACING_M
    world = patient_local @ base.PATIENT_ROTATION.T + base.PATIENT_POSITION_M
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_vertex_index",
                "mesh_x_index",
                "mesh_y_index",
                "mesh_z_index",
                "patient_x_m",
                "patient_y_m",
                "patient_z_m",
                "world_x_m",
                "world_y_m",
                "world_z_m",
            ]
        )
        for index, point, patient_point, world_point in zip(
            selected_indices, selected, patient_local, world
        ):
            writer.writerow(
                [int(index), *point.tolist(), *patient_point.tolist(), *world_point.tolist()]
            )
    ply_path = stem.with_suffix(".ply")
    trimesh.points.PointCloud(selected).export(ply_path)
    return {
        "npz": {"path": npz_path.name, "sha256": base.sha256(npz_path)},
        "csv": {"path": csv_path.name, "sha256": base.sha256(csv_path)},
        "point_cloud_ply": {"path": ply_path.name, "sha256": base.sha256(ply_path)},
        "selected_vertex_count": int(len(selected_indices)),
        "source_vertex_count": int(len(vertices)),
    }


def response_template() -> dict[str, Any]:
    return {
        "schema": "sonogym-s0010-right-sap-tp-review-response-v2",
        "reviewer": {"name": None, "qualification": None, "review_date": None},
        "candidate_decisions": [
            {
                "candidate_id": identifier,
                "decision": None,
                "correct_level": None,
                "correct_laterality": None,
                "semantic_structure_isolated": None,
                "adjacent_structure_contamination": None,
                "estimated_boundary_uncertainty_mm": None,
                "corrected_geometry_path": None,
                "comments": None,
            }
            for identifier in CANDIDATE_IDS
        ],
        "allowed_decisions": [
            "ACCEPT_BOUNDARY",
            "REVISE_BOUNDARY",
            "REJECT_CANDIDATE",
        ],
    }


def build_expanded_review(
    *,
    asset_root: Path,
    annotation_source: Path,
    protocol_path: Path,
    package_root: Path,
) -> dict[str, Any]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["schema"] != "sonogym-s0010-right-sap-tp-review-protocol-v2":
        raise RuntimeError("unexpected expanded review protocol")
    overview_dir = package_root / "overview"
    base_manifest = base.build_review(
        asset_root=asset_root,
        annotation_source=annotation_source,
        protocol_path=protocol_path,
        output_dir=overview_dir,
    )
    ct_path = asset_root / "selected_dataset" / base.PATIENT_ID / "ct.nii.gz"
    segmentation_path = (
        asset_root
        / "selected_dataset_stl"
        / base.PATIENT_ID
        / "combined_label_map.nii.gz"
    )
    mesh_root = asset_root / "selected_dataset_stl" / base.PATIENT_ID
    ct_image = nib.load(ct_path)
    segmentation_image = nib.load(segmentation_path)
    ct = np.asanyarray(ct_image.dataobj)
    segmentation = np.asanyarray(segmentation_image.dataobj)
    ct_crop, segmentation_crop, crop_metadata = crop_l4_l5(
        ct_image, segmentation_image, margin=ROI_MARGIN_VOXELS
    )
    volume_dir = package_root / "volumes"
    volume_dir.mkdir(parents=True, exist_ok=True)
    ct_crop_path = volume_dir / "s0010_L4_L5_ct_crop.nii.gz"
    segmentation_crop_path = volume_dir / "s0010_L4_L5_segmentation_crop.nii.gz"
    nib.save(ct_crop, ct_crop_path)
    nib.save(segmentation_crop, segmentation_crop_path)

    source = json.loads(annotation_source.read_text(encoding="utf-8"))
    annotations = source["annotations"]["group_B_future_MBB_bony_geometry"]
    editable_dir = package_root / "editable_geometry"
    source_mesh_dir = package_root / "source_meshes"
    adjacent_dir = package_root / "adjacent_slices"
    editable_dir.mkdir(parents=True, exist_ok=True)
    source_mesh_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for level in ("L4", "L5"):
        mesh_path = mesh_root / f"vertebrae_{level}.stl"
        copied_mesh = source_mesh_dir / mesh_path.name
        shutil.copy2(mesh_path, copied_mesh)
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        for landmark in ("superior_articular_process", "transverse_process"):
            identifier = f"{level}_RIGHT_{landmark}"
            source_row = base.source_candidate(annotations, level, landmark)
            mask = base.quantile_vertex_mask(vertices, source_row["geometry"]["selector"])
            expected_count = int(source_row["geometry"]["selected_vertex_count"])
            if int(np.sum(mask)) != expected_count:
                raise RuntimeError(f"{identifier} no longer reconstructs exactly")
            patch = vertices[mask]
            adjacent_name = f"{identifier}_adjacent_slices.png"
            plot_adjacent_slices(
                ct=ct,
                segmentation=segmentation,
                level=level,
                landmark=landmark,
                patch=patch,
                output=adjacent_dir / adjacent_name,
            )
            geometry_files = export_candidate_geometry(
                mesh=mesh,
                mask=mask,
                source_sha256=base.sha256(mesh_path),
                stem=editable_dir / identifier,
            )
            records.append(
                {
                    "candidate_id": identifier,
                    "source_status": source_row["status"],
                    "review_status": "MORE_EVIDENCE_REQUIRED",
                    "annotation_promoted": False,
                    "actor_usable": False,
                    "adjacent_slice_figure": {
                        "path": f"adjacent_slices/{adjacent_name}",
                        "sha256": base.sha256(adjacent_dir / adjacent_name),
                    },
                    "editable_geometry": geometry_files,
                }
            )

    response_path = package_root / "review_response_template.json"
    response_path.write_text(
        json.dumps(response_template(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "sonogym-s0010-right-sap-tp-expanded-review-v2",
        "patient_id": base.PATIENT_ID,
        "facet_level": "L4-L5",
        "laterality": base.SIDE,
        "reviewer_feedback": {
            "v1_integrity_files_verified": 7,
            "v1_boundaries_accepted": 0,
            "TP_concern": "candidate may include anterior adjacent bone surface",
            "SAP_concern": "axis-aligned cuts do not isolate SAP from posterior elements",
            "requested_expansion": [
                "zoomed adjacent CT slices",
                "editable full selected-point geometry",
                "source L4/L5 meshes",
                "cropped CT and segmentation volumes",
            ],
        },
        "base_review_manifest": base_manifest,
        "crop": {
            **crop_metadata,
            "ct": {
                "path": f"volumes/{ct_crop_path.name}",
                "sha256": base.sha256(ct_crop_path),
            },
            "segmentation": {
                "path": f"volumes/{segmentation_crop_path.name}",
                "sha256": base.sha256(segmentation_crop_path),
            },
        },
        "source_meshes": {
            level: {
                "path": f"source_meshes/vertebrae_{level}.stl",
                "sha256": base.sha256(source_mesh_dir / f"vertebrae_{level}.stl"),
            }
            for level in ("L4", "L5")
        },
        "records": records,
        "response_template": response_path.name,
        "review_decision": "MORE_EVIDENCE_REQUIRED",
        "annotation_promotion": "NOT_PERFORMED",
        "model_training_authorized": False,
        "actor_usable": False,
        "ultrasound_observability": "NOT_EVALUATED_IN_THIS_PACKAGE",
        "mbb_target_region": "UNRESOLVED",
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", required=True, choices=[base.SIDE])
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--annotation-source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    package_root.mkdir(parents=True, exist_ok=True)
    result = build_expanded_review(
        asset_root=args.asset_root.resolve(),
        annotation_source=args.annotation_source.resolve(),
        protocol_path=args.protocol.resolve(),
        package_root=package_root,
    )
    output = package_root / "expanded_review_manifest.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "records": len(result["records"]),
                "decision": result["review_decision"],
            }
        )
    )


if __name__ == "__main__":
    main()
