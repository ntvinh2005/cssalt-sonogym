#!/usr/bin/env python3
"""Audit s0010 RIGHT L4/L5 SAP source geometry and ultrasound correspondence.

All mesh, label and coordinate data used here are privileged evaluator data.
The script never writes them into an actor record and never promotes a heuristic
surface region to semantic ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import trimesh


def _load_geometry_module():
    source = (
        Path(__file__).resolve().parents[2]
        / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation"
        / "ultrasound_plane_geometry.py"
    )
    spec = importlib.util.spec_from_file_location("ultrasound_plane_geometry_cpu", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CPU geometry helper from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_GEOMETRY = _load_geometry_module()
plane_from_full_pose = _GEOMETRY.plane_from_full_pose
project_points_to_pixels = _GEOMETRY.project_points_to_pixels


LEVELS = ("L4", "L5")
SAP_IDS = {level: f"{level}_RIGHT_superior_articular_process" for level in LEVELS}
TP_IDS = {level: f"{level}_RIGHT_transverse_process" for level in LEVELS}
REFERENCE_URLS = (
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7362874/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11372964/",
    "https://pubmed.ncbi.nlm.nih.gov/26717379/",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def locate_geometry(root: Path, candidate_id: str) -> Path:
    matches = list(root.rglob(f"{candidate_id}_revision.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one geometry for {candidate_id}, found {len(matches)}")
    return matches[0]


def load_geometry(path: Path) -> dict[str, Any]:
    archive = np.load(path)
    return {
        "path": path,
        "sha256": sha256(path),
        "vertices": archive["vertices_mesh_index"].astype(np.float64),
        "faces": archive["faces"].astype(np.int64),
        "original": archive["original_selected_vertex_mask"].astype(bool),
        "proposed": archive["proposed_revised_vertex_mask"].astype(bool),
        "uncertain": archive["boundary_uncertainty_vertex_mask"].astype(bool),
        "source_mesh_sha256": str(archive["source_mesh_sha256"]),
    }


def unique_pixel_count(rows: np.ndarray, columns: np.ndarray) -> int:
    if len(rows) == 0:
        return 0
    return int(len(set(zip(rows.tolist(), columns.tolist()))))


def audit_asset_semantics(asset_root: Path) -> dict[str, Any]:
    patient = asset_root / "selected_dataset_stl/s0010"
    body = asset_root / "selected_dataset_body_from_urdf/s0010/combined_wrapwrap/combined_wrapwrap.usd"
    urdf = patient / "s0010.urdf"
    files = sorted(path.name for path in patient.iterdir() if path.is_file())
    semantic_tokens = ("superior_articular", "facet", "mammillary", "accessory_process")
    semantic_files = [name for name in files if any(token in name.lower() for token in semantic_tokens)]
    urdf_text = urdf.read_text(errors="ignore")
    usd_text = body.read_text(errors="ignore")
    return {
        "patient_asset_directory": str(patient),
        "patient_file_count": len(files),
        "level_meshes": {
            level: {
                "path": str(patient / f"vertebrae_{level}.stl"),
                "sha256": sha256(patient / f"vertebrae_{level}.stl"),
            }
            for level in LEVELS
        },
        "segmentation": {
            "path": str(patient / "combined_label_map.nii.gz"),
            "sha256": sha256(patient / "combined_label_map.nii.gz"),
        },
        "urdf": {"path": str(urdf), "sha256": sha256(urdf)},
        "usd": {"path": str(body), "sha256": sha256(body)},
        "process_or_facet_named_files": semantic_files,
        "urdf_has_separate_process_or_facet_name": any(
            token in urdf_text.lower() for token in semantic_tokens
        ),
        "usd_has_separate_process_or_facet_name": any(
            token in usd_text.lower() for token in semantic_tokens
        ),
        "urdf_link_count": urdf_text.count("<link "),
        "urdf_uses_combined_body_mesh": "combined_wrapwrap.stl" in urdf_text,
    }


def audit_labels(asset_root: Path) -> dict[str, Any]:
    patient = asset_root / "selected_dataset_stl/s0010"
    ct = nib.load(asset_root / "selected_dataset/s0010/ct.nii.gz")
    labels = nib.load(patient / "combined_label_map.nii.gz")
    volume = np.asanyarray(labels.dataobj)
    values, counts = np.unique(volume, return_counts=True)
    return {
        "ct_shape": list(ct.shape),
        "segmentation_shape": list(labels.shape),
        "affine_max_abs_difference": float(np.max(np.abs(ct.affine - labels.affine))),
        "label_values": {str(int(value)): int(count) for value, count in zip(values, counts)},
        "l4_label": 28,
        "l5_label": 27,
        "process_specific_label_count": 0,
    }


def scan_v3(
    v3_root: Path,
    geometries: dict[str, dict[str, dict[str, Any]]],
    tolerance_mm: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    maximum_coordinate_error = 0.0
    best: dict[str, dict[str, Any]] = {}
    session_dirs = sorted((v3_root / "stage2").glob("mbbpoc_v3_*"))
    for session_dir in session_dirs:
        frames = read_jsonl(session_dir / "public/frames.jsonl")
        poses = read_jsonl(session_dir / "privileged/full_pose_replay.jsonl")
        images = np.load(session_dir / "public/ultrasound.npz")["ultrasound"]
        coordinate_archive = np.load(session_dir / "privileged/pixel_to_volume_coordinates.npz")
        native_coordinates = coordinate_archive["human_img_coords"].astype(np.float64)
        local = coordinate_archive["native_local_image_coordinates_m"].astype(np.float64)
        if not (len(frames) == len(poses) == len(images) == len(native_coordinates)):
            raise RuntimeError(f"misaligned acquisition records in {session_dir}")
        for index, (frame, pose, image, native) in enumerate(
            zip(frames, poses, images, native_coordinates)
        ):
            if frame["frame_sequence"] != index or pose["frame_sequence"] != index:
                raise RuntimeError("frame sequence mismatch")
            if frame["frame_sha256"] != pose["frame_sha256"]:
                raise RuntimeError("frame identity mismatch")
            plane = plane_from_full_pose(
                end_effector_position_m=np.asarray(pose["human_to_ee_position_m"]),
                rotation_columns=np.asarray(pose["human_to_ee_rotation_columns"]),
                image_origin_height_m=float(pose["image_origin_height_m"]),
                local_coordinates_m=local,
                label_resolution_m=float(pose["label_resolution_m"]),
            )
            coordinate_error = float(np.max(np.abs(plane.coordinates_index - native)))
            maximum_coordinate_error = max(maximum_coordinate_error, coordinate_error)
            for level in LEVELS:
                sap = geometries[level]["sap"]
                projection = project_points_to_pixels(
                    points_index=sap["vertices"][sap["uncertain"]],
                    image_origin_index=plane.origin_index,
                    rotation_columns=plane.rotation_columns,
                    width=150,
                    depth=200,
                    image_resolution_m=float(pose["image_resolution_m"]),
                    label_resolution_m=float(pose["label_resolution_m"]),
                    plane_tolerance_mm=tolerance_mm,
                )
                inside = projection["inside"]
                projected_count = int(np.count_nonzero(inside))
                pixel_count = unique_pixel_count(
                    projection["rows"][inside], projection["columns"][inside]
                )
                record = {
                    "session_id": session_dir.name,
                    "frame_sequence": index,
                    "frame_sha256": frame["frame_sha256"],
                    "acquisition_id": frame["acquisition_id"],
                    "target_name": frame["target_name"],
                    "phase": frame["phase"],
                    "sap_candidate_id": SAP_IDS[level],
                    "geometry_sha256": sap["sha256"],
                    "transform_mode": "EXACT_POSE_REPLAY",
                    "maximum_coordinate_error_index": coordinate_error,
                    "projection_status": "VERIFIED",
                    "candidate_intersection": "PRESENT" if pixel_count else "ABSENT",
                    "projected_vertex_entries": projected_count,
                    "unique_projected_pixels": pixel_count,
                    "acoustic_visibility": "NOT_REVIEWED",
                    "semantic_identity": "NOT_REVIEWED",
                    "training_label": "EXCLUDED",
                }
                rows.append(record)
                if level not in best or pixel_count > best[level]["record"]["unique_projected_pixels"]:
                    best[level] = {
                        "record": record,
                        "image": np.asarray(image),
                        "rows": projection["rows"][inside],
                        "columns": projection["columns"][inside],
                    }
    transform = {
        "mode": "EXACT_POSE_REPLAY",
        "session_count": len(session_dirs),
        "frame_count": sum(1 for row in rows if row["sap_candidate_id"] == SAP_IDS["L4"]),
        "candidate_frame_rows": len(rows),
        "maximum_coordinate_error_index": maximum_coordinate_error,
        "required_maximum_coordinate_error_index": 0.001,
        "passes_coordinate_gate": maximum_coordinate_error <= 0.001,
        "local_axis_order": ["width", "elevation", "depth"],
        "array_axis_order": ["depth_row", "width_column"],
        "coordinate_frame": "patient-local volume index",
    }
    return rows, transform, best


def plot_surface_context(
    geometries: dict[str, dict[str, dict[str, Any]]], output: Path
) -> None:
    fig = plt.figure(figsize=(12, 6), dpi=180)
    for column, level in enumerate(LEVELS, start=1):
        ax = fig.add_subplot(1, 2, column, projection="3d")
        sap = geometries[level]["sap"]
        tp = geometries[level]["tp"]
        vertices = sap["vertices"]
        stride = max(1, len(vertices) // 7000)
        ax.scatter(*vertices[::stride].T, s=0.3, c="#aaaaaa", alpha=0.12)
        sap_points = vertices[sap["uncertain"]]
        tp_points = tp["vertices"][tp["proposed"]]
        ax.scatter(*sap_points.T, s=2, c="#2878b5", alpha=0.65, label="SAP unresolved")
        ax.scatter(*tp_points.T, s=2, c="#f28e2b", alpha=0.65, label="TP proposed core")
        ax.set_title(f"{level} RIGHT review geometry")
        ax.set_xlabel("volume index x")
        ax.set_ylabel("volume index y")
        ax.set_zlabel("volume index z")
        ax.view_init(elev=20, azim=-50)
        ax.set_box_aspect(np.ptp(vertices, axis=0))
        ax.legend(loc="upper left", fontsize=7)
    fig.suptitle("Privileged geometry review only; blue SAP boundary is unresolved")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_mpr(
    asset_root: Path,
    geometries: dict[str, dict[str, dict[str, Any]]],
    output: Path,
) -> None:
    ct = np.asanyarray(nib.load(asset_root / "selected_dataset/s0010/ct.nii.gz").dataobj)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=180)
    for row, level in enumerate(LEVELS):
        points = geometries[level]["sap"]["vertices"][geometries[level]["sap"]["uncertain"]]
        center = np.rint(np.median(points, axis=0)).astype(int)
        views = (
            (ct[center[0], :, :].T, points[:, 1], points[:, 2], center[0], points[:, 0], "sagittal"),
            (ct[:, center[1], :].T, points[:, 0], points[:, 2], center[1], points[:, 1], "coronal"),
            (ct[:, :, center[2]].T, points[:, 0], points[:, 1], center[2], points[:, 2], "axial"),
        )
        for column, (image, xx, yy, section, normal, title) in enumerate(views):
            ax = axes[row, column]
            lo, hi = np.percentile(image[np.isfinite(image)], [2, 98])
            ax.imshow(image, cmap="gray", origin="lower", vmin=lo, vmax=hi)
            slab = np.abs(normal - section) <= 0.75
            ax.scatter(xx[slab], yy[slab], s=5, c="#2878b5", alpha=0.7)
            ax.set_title(f"{level} {title}; 1.5-index slab")
            ax.set_axis_off()
    fig.suptitle("CT context with unresolved SAP surface candidate (blue)")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_ultrasound(best: dict[str, dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 11), dpi=180)
    for row, level in enumerate(LEVELS):
        item = best[level]
        record = item["record"]
        axes[row, 0].imshow(item["image"], cmap="gray")
        axes[row, 0].set_title(f"{level} clean ultrasound\n{record['session_id']} frame {record['frame_sequence']}")
        axes[row, 1].imshow(item["image"], cmap="gray")
        axes[row, 1].scatter(item["columns"], item["rows"], s=5, c="#2878b5", alpha=0.6)
        axes[row, 1].set_title(
            f"Evaluator-only unresolved SAP overlay\n{record['unique_projected_pixels']} unique pixels"
        )
        for ax in axes[row]:
            ax.set_axis_off()
    fig.suptitle("Plane intersection is verified; acoustic visibility and SAP identity are not reviewed")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    annotations = json.loads(args.annotations.read_text())
    assets = audit_asset_semantics(args.asset_root.resolve())
    labels = audit_labels(args.asset_root.resolve())
    geometries: dict[str, dict[str, dict[str, Any]]] = {}
    boundary_rows = []
    for level in LEVELS:
        sap = load_geometry(locate_geometry(args.geometry_root.resolve(), SAP_IDS[level]))
        tp = load_geometry(locate_geometry(args.geometry_root.resolve(), TP_IDS[level]))
        geometries[level] = {"sap": sap, "tp": tp}
        mesh = trimesh.load(assets["level_meshes"][level]["path"], force="mesh", process=False)
        merged_mesh = trimesh.load(
            assets["level_meshes"][level]["path"], force="mesh", process=True
        )
        boundary_rows.append(
            {
                "candidate_id": SAP_IDS[level],
                "source_mesh_vertex_count": int(len(mesh.vertices)),
                "source_mesh_merged_vertex_count": int(len(merged_mesh.vertices)),
                "source_mesh_face_count": int(len(mesh.faces)),
                "source_mesh_is_watertight": bool(merged_mesh.is_watertight),
                "original_candidate_vertices": int(np.count_nonzero(sap["original"])),
                "proposed_semantic_vertices": int(np.count_nonzero(sap["proposed"])),
                "uncertain_vertices": int(np.count_nonzero(sap["uncertain"])),
                "separate_semantic_object": False,
                "status": "UNRESOLVED_BOUNDARY",
            }
        )
    projection_rows, transform, best = scan_v3(
        args.v3_root.resolve(), geometries, args.surface_slab_tolerance_mm
    )
    write_csv(output / "sap_projection_table.csv", projection_rows)
    write_csv(output / "sap_boundary_inventory.csv", boundary_rows)
    plot_surface_context(geometries, output / "sap_tp_surface_context.png")
    plot_mpr(args.asset_root.resolve(), geometries, output / "sap_ct_mpr.png")
    plot_ultrasound(best, output / "sap_ultrasound_correspondence.png")
    intersections = {
        level: {
            "frames_present": sum(
                row["candidate_intersection"] == "PRESENT"
                for row in projection_rows
                if row["sap_candidate_id"] == SAP_IDS[level]
            ),
            "best_record": best[level]["record"],
        }
        for level in LEVELS
    }
    result = {
        "schema": "sonogym-s0010-sap-geometry-audit-v1",
        "source_revision": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "patient_id": "s0010",
        "laterality": "RIGHT",
        "verdict": "SAP_CANDIDATES_NEED_REVIEW",
        "case": "geometry_exists_but_is_not_semantically_separated",
        "asset_inventory": assets,
        "semantic_label_inventory": labels,
        "boundary_inventory": boundary_rows,
        "coordinate_transform_audit": transform,
        "ultrasound_correspondence": intersections,
        "annotation_source_status": annotations["status"],
        "mbb_surrogate_regions": {
            "status": "UNAVAILABLE",
            "facet_level": "L4-L5",
            "required_branch_sites": [
                {
                    "branch": "L3 medial branch",
                    "vertebral_bony_context": ["L4 RIGHT SAP", "L4 RIGHT TP root"],
                    "geometry": None,
                },
                {
                    "branch": "L4 medial branch",
                    "vertebral_bony_context": ["L5 RIGHT SAP", "L5 RIGHT TP root"],
                    "geometry": None,
                },
            ],
            "reason": "SAP boundaries, mammillary/accessory landmarks and nerve course are not represented as validated s0010 geometry",
            "actual_nerve_location_claim": False,
            "needle_endpoint_claim": False,
        },
        "actor_information_boundary": {
            "all_outputs_privileged": True,
            "runtime_actor_fields_added": [],
            "semantic_training_labels_created": 0,
        },
        "authoritative_references": list(REFERENCE_URLS),
        "limitations": [
            "whole-vertebra labels do not isolate SAP, facet surface, mammillary process or accessory process",
            "every existing RIGHT L4/L5 SAP candidate vertex remains in the uncertainty mask",
            "verified scan-plane intersection does not establish acoustic visibility or semantic SAP identity",
            "no medial-branch nerve segmentation exists",
        ],
        "files": {
            "projection_table": "sap_projection_table.csv",
            "boundary_inventory": "sap_boundary_inventory.csv",
            "surface_context": "sap_tp_surface_context.png",
            "ct_mpr": "sap_ct_mpr.png",
            "ultrasound_correspondence": "sap_ultrasound_correspondence.png",
        },
    }
    (output / "sap_audit_result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
