#!/usr/bin/env python3
"""Export a bounded RIGHT L4/L5 SAP/TP candidate package for human review.

The script reconstructs the existing v2 mesh-quantile candidates exactly.  It does not promote
them to semantic ground truth.  All CT, segmentation, and mesh overlays are privileged,
evaluator-only review artifacts and must never enter actor inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation


PATIENT_ID = "s0010"
SIDE = "RIGHT"
LABEL_BY_LEVEL = {"L4": 28, "L5": 27}
PATIENT_POSITION_M = np.asarray([0.2, -0.45, 1.0], dtype=np.float64)
PATIENT_ROTATION = Rotation.from_euler(
    "yxz", [-90.0, -90.0, 0.0], degrees=True
).as_matrix()
LABEL_SPACING_M = 0.0015
COLORS = {
    "transverse_process": "#ff7f0e",
    "superior_articular_process": "#1f77b4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(points: np.ndarray) -> dict[str, Any]:
    return {
        "point_count": int(len(points)),
        "centroid": np.mean(points, axis=0).tolist(),
        "bounds": [np.min(points, axis=0).tolist(), np.max(points, axis=0).tolist()],
    }


def geometry_frames(points: np.ndarray, affine: np.ndarray) -> dict[str, Any]:
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    ras_mm = (homogeneous @ affine.T)[:, :3]
    patient_local_m = points * LABEL_SPACING_M
    world_m = patient_local_m @ PATIENT_ROTATION.T + PATIENT_POSITION_M
    return {
        "mesh_index": summary(points),
        "nifti_world_RAS_mm": summary(ras_mm),
        "sonogym_patient_local_m": summary(patient_local_m),
        "sonogym_world_m": summary(world_m),
    }


def quantile_vertex_mask(
    vertices: np.ndarray, selector: dict[str, list[float]]
) -> np.ndarray:
    mask = np.ones(len(vertices), dtype=bool)
    axis_keys = (
        "x_quantile_fraction",
        "y_quantile_fraction",
        "z_quantile_fraction",
    )
    for axis, key in enumerate(axis_keys):
        low, high = selector[key]
        lower, upper = np.quantile(vertices[:, axis], [low, high])
        mask &= (vertices[:, axis] >= lower) & (vertices[:, axis] <= upper)
    return mask


def quantile_candidate(
    vertices: np.ndarray, selector: dict[str, list[float]]
) -> np.ndarray:
    mask = quantile_vertex_mask(vertices, selector)
    points = vertices[mask]
    if not len(points):
        raise RuntimeError("candidate selector produced no vertices")
    return points


def source_candidate(
    annotations: list[dict[str, Any]], level: str, landmark: str
) -> dict[str, Any]:
    matches = [
        row
        for row in annotations
        if row["anatomy_level"] == level
        and row["landmark_name"] == landmark
        and row["laterality"] == SIDE
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {level} {SIDE} {landmark}, found {len(matches)}")
    return matches[0]


def projected_points(
    points: np.ndarray, axis: int, coordinate: int, tolerance: float = 1.0
) -> np.ndarray:
    return points[np.abs(points[:, axis] - coordinate) <= tolerance]


def plot_review(
    *,
    ct: np.ndarray,
    segmentation: np.ndarray,
    level: str,
    landmark: str,
    full_mesh: np.ndarray,
    patch: np.ndarray,
    output: Path,
) -> None:
    color = COLORS[landmark]
    centroid = np.mean(patch, axis=0)
    x, y, z = np.rint(centroid).astype(int)
    rng = np.random.default_rng(20261117)
    unique_mesh = np.unique(full_mesh, axis=0)
    chosen = rng.choice(
        len(unique_mesh), size=min(5500, len(unique_mesh)), replace=False
    )
    background = unique_mesh[chosen]

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    projection_specs = (
        (0, 2, "Posterior projection", "Right (+x)", "Superior (+z)"),
        (1, 2, "Right-lateral projection", "Anterior (+y)", "Superior (+z)"),
        (0, 1, "Superior projection", "Right (+x)", "Anterior (+y)"),
    )
    for axis, (horizontal, vertical, title, xlabel, ylabel) in zip(
        axes[0], projection_specs
    ):
        axis.scatter(
            background[:, horizontal],
            background[:, vertical],
            s=1.2,
            color="#8f8f8f",
            alpha=0.25,
        )
        axis.scatter(
            patch[:, horizontal],
            patch[:, vertical],
            s=6,
            color=color,
            alpha=0.65,
        )
        axis.scatter(
            [centroid[horizontal]],
            [centroid[vertical]],
            s=80,
            marker="+",
            color="black",
        )
        axis.set_title(title)
        axis.set_xlabel(f"{xlabel}, mesh index")
        axis.set_ylabel(f"{ylabel}, mesh index")
        axis.set_aspect("equal", adjustable="box")

    ct_views = (
        (
            ct[:, :, z].T,
            segmentation[:, :, z].T,
            projected_points(patch, 2, z)[:, (0, 1)],
            "Axial CT",
            "Right (+x)",
            "Anterior (+y)",
        ),
        (
            ct[x, :, :].T,
            segmentation[x, :, :].T,
            projected_points(patch, 0, x)[:, (1, 2)],
            "Sagittal CT",
            "Anterior (+y)",
            "Superior (+z)",
        ),
        (
            ct[:, y, :].T,
            segmentation[:, y, :].T,
            projected_points(patch, 1, y)[:, (0, 2)],
            "Coronal CT",
            "Right (+x)",
            "Superior (+z)",
        ),
    )
    raw_label = LABEL_BY_LEVEL[level]
    for axis, (image, labels, local_patch, title, xlabel, ylabel) in zip(
        axes[1], ct_views
    ):
        axis.imshow(np.clip(image, -300, 1200), cmap="gray", origin="lower")
        axis.contour(labels == raw_label, levels=[0.5], colors=["lime"], linewidths=0.8)
        if len(local_patch):
            axis.scatter(
                local_patch[:, 0],
                local_patch[:, 1],
                s=8,
                color=color,
                alpha=0.65,
            )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)

    figure.suptitle(
        f"Evaluator-only review candidate: {level} RIGHT {landmark}\n"
        "gray=mesh; orange/blue=candidate; green=raw vertebral label; not actor ground truth"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def build_review(
    *,
    asset_root: Path,
    annotation_source: Path,
    protocol_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    ct_path = asset_root / "selected_dataset" / PATIENT_ID / "ct.nii.gz"
    segmentation_path = (
        asset_root / "selected_dataset_stl" / PATIENT_ID / "combined_label_map.nii.gz"
    )
    mesh_root = asset_root / "selected_dataset_stl" / PATIENT_ID
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["patient_id"] != PATIENT_ID or protocol["laterality"] != SIDE:
        raise RuntimeError("protocol patient or laterality does not match exporter")
    actual_hashes = {
        "annotation": sha256(annotation_source),
        "ct": sha256(ct_path),
        "segmentation": sha256(segmentation_path),
        "L4_mesh": sha256(mesh_root / "vertebrae_L4.stl"),
        "L5_mesh": sha256(mesh_root / "vertebrae_L5.stl"),
    }
    expected_hashes = {
        "annotation": protocol["inputs"]["annotation"]["sha256"],
        **protocol["inputs"]["assets"],
    }
    for name, actual in actual_hashes.items():
        expected_key = name if name == "annotation" else f"{name}_sha256"
        if actual != expected_hashes[expected_key]:
            raise RuntimeError(f"{name} SHA-256 mismatch")
    ct_image = nib.load(ct_path)
    segmentation_image = nib.load(segmentation_path)
    ct = np.asarray(ct_image.dataobj)
    segmentation = np.asarray(segmentation_image.dataobj)
    if ct.shape != segmentation.shape or not np.allclose(
        ct_image.affine, segmentation_image.affine
    ):
        raise RuntimeError("CT and segmentation grids do not agree")

    source = json.loads(annotation_source.read_text(encoding="utf-8"))
    annotations = source["annotations"]["group_B_future_MBB_bony_geometry"]
    records: list[dict[str, Any]] = []
    for level in ("L4", "L5"):
        mesh_path = mesh_root / f"vertebrae_{level}.stl"
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        for landmark in ("superior_articular_process", "transverse_process"):
            source_row = source_candidate(annotations, level, landmark)
            selector = source_row["geometry"]["selector"]
            patch = quantile_candidate(vertices, selector)
            expected_count = int(source_row["geometry"]["selected_vertex_count"])
            if len(patch) != expected_count:
                raise RuntimeError(
                    f"{level} {landmark} reconstruction changed: {len(patch)} != {expected_count}"
                )
            figure_name = f"{level}_RIGHT_{landmark}_review.png"
            plot_review(
                ct=ct,
                segmentation=segmentation,
                level=level,
                landmark=landmark,
                full_mesh=vertices,
                patch=patch,
                output=output_dir / figure_name,
            )
            records.append(
                {
                    "patient_id": PATIENT_ID,
                    "facet_level": "L4-L5",
                    "vertebral_level": level,
                    "laterality": SIDE,
                    "landmark": landmark,
                    "source_annotation_status": source_row["status"],
                    "review_candidate_status": "PENDING_INDEPENDENT_REVIEW",
                    "evidence_class": "GEOMETRY_CANDIDATE_ONLY",
                    "can_train_semantic_detector": False,
                    "selector": selector,
                    "geometry": geometry_frames(patch, segmentation_image.affine),
                    "source_mesh": {
                        "path": str(mesh_path),
                        "sha256": sha256(mesh_path),
                    },
                    "visualization": figure_name,
                    "review_questions": [
                        f"Does the colored region isolate the RIGHT {level} {landmark}?",
                        "Does it include lamina, pedicle, vertebral body, or the adjacent process?",
                        (
                            "Are the proposed boundaries supportable in all three CT "
                            "planes and mesh projections?"
                        ),
                    ],
                    "limitations": [
                        "candidate reproduces the existing mesh-quantile heuristic",
                        "no dedicated semantic process segmentation exists",
                        "no clinician or qualified anatomical reviewer decision is recorded",
                        "candidate is privileged and forbidden at actor inference",
                    ],
                }
            )

    return {
        "schema": "sonogym-s0010-right-sap-tp-review-v1",
        "patient_id": PATIENT_ID,
        "laterality": SIDE,
        "facet_level": "L4-L5",
        "source_annotation": {
            "path": str(annotation_source),
            "sha256": actual_hashes["annotation"],
        },
        "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
        "source_assets": {
            "ct": {"path": str(ct_path), "sha256": actual_hashes["ct"]},
            "segmentation": {
                "path": str(segmentation_path),
                "sha256": actual_hashes["segmentation"],
            },
        },
        "records": records,
        "review_decision": "PENDING_INDEPENDENT_REVIEW",
        "annotation_promotion": "NOT_PERFORMED",
        "mbb_target_region": "UNRESOLVED",
        "actor_usable": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", required=True, choices=[SIDE])
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--annotation-source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = build_review(
        asset_root=args.asset_root.resolve(),
        annotation_source=args.annotation_source.resolve(),
        protocol_path=args.protocol.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output_json),
                "records": len(result["records"]),
                "decision": result["review_decision"],
            }
        )
    )


if __name__ == "__main__":
    main()
