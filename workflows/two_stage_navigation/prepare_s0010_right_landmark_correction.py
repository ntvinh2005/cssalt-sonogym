#!/usr/bin/env python3
"""Repair ultrasound correspondence and propose auditable RIGHT TP/SAP revisions.

This workflow preserves the historical v1/v2 packages.  It uses native-consistent
image axes, reports reduced-pose reconstruction explicitly, and never promotes an
agent-generated boundary or ultrasound intersection to semantic training truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
import trimesh
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_s0010_right_landmark_review_v2 as historical  # noqa: E402
import review_s0010_right_landmarks as base  # noqa: E402

GEOMETRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/ultrasound_plane_geometry.py"
)
SPEC = importlib.util.spec_from_file_location("ultrasound_plane_geometry", GEOMETRY_PATH)
assert SPEC is not None and SPEC.loader is not None
geometry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = geometry
SPEC.loader.exec_module(geometry)

IMAGE_WIDTH = 150
IMAGE_DEPTH = 200
IMAGE_RESOLUTION_M = 0.0005
LABEL_RESOLUTION_M = 0.0015
PROBE_HEIGHT_M = 0.13
IMAGE_ORIGIN_HEIGHT_M = 0.133
EXACT_MIN = 0.985
BONE_IOU_MIN = 0.90
PLANE_TOLERANCE_MM = 1.5
LEVEL_LABEL = {"L4": 28, "L5": 27}
LANDMARKS = ("superior_articular_process", "transverse_process")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_torch(path: Path) -> np.ndarray:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


@dataclass(frozen=True)
class CandidateRevision:
    candidate_id: str
    level: str
    landmark: str
    source_status: str
    vertices: np.ndarray
    faces: np.ndarray
    original_mask: np.ndarray
    proposed_mask: np.ndarray
    uncertain_mask: np.ndarray
    removed_mask: np.ndarray
    added_mask: np.ndarray
    status: str
    method: str
    base_transition_x_index: int | None
    source_mesh_path: Path
    source_mesh_sha256: str


def _original_mask(
    annotations: list[dict[str, Any]], vertices: np.ndarray, level: str, landmark: str
) -> tuple[np.ndarray, dict[str, Any]]:
    row = base.source_candidate(annotations, level, landmark)
    mask = base.quantile_vertex_mask(vertices, row["geometry"]["selector"])
    if int(mask.sum()) != int(row["geometry"]["selected_vertex_count"]):
        raise RuntimeError(f"historical geometry changed: {level} {landmark}")
    return mask, row


def _tp_base_transition(
    *, segmentation: np.ndarray, label: int, original_points: np.ndarray
) -> int:
    """Locate the largest CT cross-section narrowing within the original TP span."""

    lower = np.floor(original_points.min(axis=0)).astype(int)
    upper = np.ceil(original_points.max(axis=0)).astype(int)
    spans: list[tuple[int, int]] = []
    for x_index in range(lower[0], upper[0] + 1):
        slab = segmentation[
            x_index,
            :,
            max(lower[2], 0) : min(upper[2] + 1, segmentation.shape[2]),
        ]
        y_indices = np.flatnonzero(np.any(slab == label, axis=1))
        span = int(y_indices[-1] - y_indices[0] + 1) if len(y_indices) else 0
        spans.append((x_index, span))
    losses = [spans[index - 1][1] - spans[index][1] for index in range(1, len(spans))]
    if not losses or max(losses) <= 0:
        raise RuntimeError("no CT-supported lateral-process base transition")
    return spans[int(np.argmax(losses)) + 1][0]


def load_revisions(
    *, asset_root: Path, annotation_source: Path, segmentation: np.ndarray
) -> list[CandidateRevision]:
    source = json.loads(annotation_source.read_text(encoding="utf-8"))
    annotations = source["annotations"]["group_B_future_MBB_bony_geometry"]
    result: list[CandidateRevision] = []
    mesh_root = asset_root / "selected_dataset_stl" / "s0010"
    for level in ("L4", "L5"):
        mesh_path = mesh_root / f"vertebrae_{level}.stl"
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        for landmark in LANDMARKS:
            original, row = _original_mask(annotations, vertices, level, landmark)
            proposed = np.zeros(len(vertices), dtype=bool)
            uncertain = np.zeros(len(vertices), dtype=bool)
            removed = np.zeros(len(vertices), dtype=bool)
            added = np.zeros(len(vertices), dtype=bool)
            transition: int | None = None
            if landmark == "transverse_process":
                transition = _tp_base_transition(
                    segmentation=segmentation,
                    label=LEVEL_LABEL[level],
                    original_points=vertices[original],
                )
                # The transition voxel and one adjacent layer remain uncertain.  The
                # lateral surface is a conservative core, not a whole-process mask.
                proposed = original & (vertices[:, 0] > transition + 1.0)
                uncertain = original & (np.abs(vertices[:, 0] - transition) <= 1.0)
                removed = original & ~(proposed | uncertain)
                status = "PROPOSED_REVISION_PENDING_ANATOMICAL_REVIEW"
                method = (
                    "CT-supported largest lateral cross-section narrowing; retained distal "
                    "surface core, explicit two-voxel base uncertainty; not whole-process segmentation"
                )
            else:
                uncertain = original.copy()
                status = "UNRESOLVED_BOUNDARY"
                method = (
                    "original quantile region retained only as uncertainty; CT/mesh lacks an "
                    "independent boundary separating SAP from lamina and adjacent posterior elements"
                )
            result.append(
                CandidateRevision(
                    candidate_id=f"{level}_RIGHT_{landmark}",
                    level=level,
                    landmark=landmark,
                    source_status=row["status"],
                    vertices=vertices,
                    faces=faces,
                    original_mask=original,
                    proposed_mask=proposed,
                    uncertain_mask=uncertain,
                    removed_mask=removed,
                    added_mask=added,
                    status=status,
                    method=method,
                    base_transition_x_index=transition,
                    source_mesh_path=mesh_path,
                    source_mesh_sha256=sha256(mesh_path),
                )
            )
    return result


def export_revision(candidate: CandidateRevision, root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{candidate.candidate_id}_revision.npz"
    np.savez_compressed(
        path,
        vertices_mesh_index=candidate.vertices.astype(np.float32),
        faces=candidate.faces,
        original_selected_vertex_mask=candidate.original_mask,
        proposed_revised_vertex_mask=candidate.proposed_mask,
        boundary_uncertainty_vertex_mask=candidate.uncertain_mask,
        removed_vertex_mask=candidate.removed_mask,
        added_vertex_mask=candidate.added_mask,
        source_mesh_sha256=np.asarray(candidate.source_mesh_sha256),
        coordinate_frame=np.asarray("segmentation_mesh_index_xyz"),
        unit=np.asarray("label_voxel_index_1.5_mm"),
    )
    csv_path = root / f"{candidate.candidate_id}_revision_vertices.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_vertex_index", "x_index", "y_index", "z_index", "membership"])
        for index in np.flatnonzero(
            candidate.original_mask | candidate.proposed_mask | candidate.uncertain_mask
        ):
            memberships = []
            for name, mask in (
                ("ORIGINAL", candidate.original_mask),
                ("PROPOSED", candidate.proposed_mask),
                ("UNCERTAIN", candidate.uncertain_mask),
                ("REMOVED", candidate.removed_mask),
                ("ADDED", candidate.added_mask),
            ):
                if mask[index]:
                    memberships.append(name)
            writer.writerow([int(index), *candidate.vertices[index].tolist(), "+".join(memberships)])
    return {
        "npz": {"path": path.name, "sha256": sha256(path)},
        "csv": {"path": csv_path.name, "sha256": sha256(csv_path)},
        "original_count": int(candidate.original_mask.sum()),
        "proposed_count": int(candidate.proposed_mask.sum()),
        "uncertain_count": int(candidate.uncertain_mask.sum()),
        "removed_count": int(candidate.removed_mask.sum()),
        "added_count": int(candidate.added_mask.sum()),
    }


def render_revision(candidate: CandidateRevision, output: Path) -> None:
    unique = np.unique(candidate.vertices, axis=0)
    rng = np.random.default_rng(20261117)
    background = unique[rng.choice(len(unique), min(8000, len(unique)), replace=False)]
    figure, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    specs = ((0, 2, "Posterior"), (1, 2, "Right lateral"), (0, 1, "Superior"))
    layers = (
        (candidate.removed_mask, "#d62728", "removed"),
        (candidate.proposed_mask, "#2ca02c", "proposed core"),
        (candidate.uncertain_mask, "#ffbf00", "uncertain boundary"),
    )
    for row in range(2):
        for column, (horizontal, vertical, title) in enumerate(specs):
            axis = axes[row, column]
            axis.scatter(background[:, horizontal], background[:, vertical], s=1, c="#aaaaaa", alpha=0.2)
            if row == 0:
                points = candidate.vertices[candidate.original_mask]
                axis.scatter(points[:, horizontal], points[:, vertical], s=5, c="#1f77b4", alpha=0.65, label="original")
            else:
                for mask, color, label in layers:
                    points = candidate.vertices[mask]
                    if len(points):
                        axis.scatter(points[:, horizontal], points[:, vertical], s=5, c=color, alpha=0.72, label=label)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(f"{title}: {'original' if row == 0 else 'revision'}")
            if column == 0:
                axis.legend(loc="best", fontsize=8)
    figure.suptitle(
        f"{candidate.candidate_id}\n{candidate.status} — evaluator-only surface proposal",
        fontsize=16,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def sample_volume(volume: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    indices = np.trunc(coordinates).astype(np.int64)
    for axis in range(3):
        indices[..., axis] = np.clip(indices[..., axis], 0, volume.shape[axis] - 1)
    return volume[indices[..., 0], indices[..., 1], indices[..., 2]]


def integrity(reconstructed: np.ndarray, stored: np.ndarray) -> dict[str, Any]:
    exact = float(np.mean(reconstructed == stored))
    reconstructed_bone = np.isin(reconstructed, np.arange(25, 44))
    stored_bone = np.isin(stored, np.arange(25, 44))
    union = reconstructed_bone | stored_bone
    bone_iou = float(np.count_nonzero(reconstructed_bone & stored_bone) / np.count_nonzero(union)) if np.any(union) else 1.0
    local = ndimage.binary_dilation(union, iterations=2)
    local_exact = float(np.mean(reconstructed[local] == stored[local])) if np.any(local) else 1.0
    return {
        "exact_raw_label_agreement": exact,
        "bone_mask_iou": bone_iou,
        "local_bone_exact_agreement_diagnostic": local_exact,
        "verified": bool(exact >= EXACT_MIN and bone_iou >= BONE_IOU_MIN),
    }


def correspondence_state(
    *, projection_status: str, intersects: bool, raw_level_pixels: int
) -> dict[str, Any]:
    """Return fail-closed correspondence semantics without inferring visibility."""

    if projection_status not in {"VERIFIED", "UNVERIFIED", "MISSING_METADATA"}:
        raise ValueError("invalid projection status")
    intersection = (
        "UNKNOWN"
        if projection_status != "VERIFIED"
        else ("PRESENT" if intersects else "ABSENT")
    )
    return {
        "projection_status": projection_status,
        "candidate_intersection": intersection,
        "raw_level_geometric_presence": bool(raw_level_pixels > 0),
        "acoustic_visibility": "NOT_REVIEWED",
        "semantic_identity": "NOT_REVIEWED",
        "training_label": "EXCLUDED",
        "training_eligible": False,
    }


def iter_sessions(sources: list[dict[str, Any]]):
    for source in sources:
        for public in sorted(Path(source["root"]).glob("sessions/*/public/frames.jsonl")):
            yield source, public.parents[1]


def replay(
    *,
    sources: list[dict[str, Any]],
    raw_volume: np.ndarray,
    surface_y: np.ndarray,
    surface_normals: np.ndarray,
    candidates: list[CandidateRevision],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    frame_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for source, session in iter_sessions(sources):
        public = read_jsonl(session / "public" / "frames.jsonl")
        privileged = read_jsonl(session / "privileged" / "annotations.jsonl")
        joined = {(int(row["frame_sequence"]), row["frame_sha256"]): row for row in privileged}
        images = np.load(session / "public" / "ultrasound.npz")["ultrasound"]
        labels = np.load(session / "privileged" / "raw_labels.npz")["raw_labels"]
        if not (len(public) == len(images) == len(labels)):
            raise RuntimeError(f"capture lengths differ: {session}")
        for index, row in enumerate(public):
            annotation = joined.get((int(row["frame_sequence"]), row["frame_sha256"]))
            common = {
                "source_id": source["id"],
                "session_id": row["session_id"],
                "frame_sequence": int(row["frame_sequence"]),
                "producer_sequence": int(row["producer_sequence"]),
                "frame_sha256": row["frame_sha256"],
                "recorded_split": row.get("split", "development"),
                "reconstruction_mode": "REDUCED_POSE_RECONSTRUCTION",
            }
            if annotation is None or "realized_pose_index" not in annotation:
                frame_rows.append({**common, "projection_status": "MISSING_METADATA"})
                continue
            plane = geometry.plane_from_reduced_pose(
                realized_pose_index=np.asarray(annotation["realized_pose_index"], dtype=np.float64),
                surface_y_index=surface_y,
                surface_normals=surface_normals,
                width=IMAGE_WIDTH,
                depth=IMAGE_DEPTH,
                image_resolution_m=IMAGE_RESOLUTION_M,
                label_resolution_m=LABEL_RESOLUTION_M,
                probe_height_m=PROBE_HEIGHT_M,
                image_origin_height_m=IMAGE_ORIGIN_HEIGHT_M,
                volume_shape=raw_volume.shape,
            )
            reconstructed = sample_volume(raw_volume, plane.coordinates_index)
            metric = integrity(reconstructed, np.asarray(labels[index]))
            projection_status = "VERIFIED" if metric["verified"] else "UNVERIFIED"
            frame_rows.append(
                {
                    **common,
                    **metric,
                    "projection_status": projection_status,
                    "clipped_sample_count": plane.clipped_sample_count,
                    "realized_pose_index": annotation["realized_pose_index"],
                    "image_archive_index": index,
                    "image_archive": str(session / "public" / "ultrasound.npz"),
                    "raw_label_archive": str(session / "privileged" / "raw_labels.npz"),
                }
            )
            for candidate in candidates:
                mask = candidate.proposed_mask if np.any(candidate.proposed_mask) else candidate.original_mask
                projected = geometry.project_points_to_pixels(
                    points_index=candidate.vertices[mask],
                    image_origin_index=plane.origin_index,
                    rotation_columns=plane.rotation_columns,
                    width=IMAGE_WIDTH,
                    depth=IMAGE_DEPTH,
                    image_resolution_m=IMAGE_RESOLUTION_M,
                    label_resolution_m=LABEL_RESOLUTION_M,
                    plane_tolerance_mm=PLANE_TOLERANCE_MM,
                )
                state = correspondence_state(
                    projection_status=projection_status,
                    intersects=bool(np.any(projected["inside"])),
                    raw_level_pixels=int(np.count_nonzero(labels[index] == LEVEL_LABEL[candidate.level])),
                )
                inside = projected["inside"]
                candidate_rows.append(
                    {
                        **common,
                        "candidate_id": candidate.candidate_id,
                        "geometry_status": candidate.status,
                        "geometry_version": "proposed_revision" if np.any(candidate.proposed_mask) else "original_unresolved",
                        **state,
                        "image_archive": str(session / "public" / "ultrasound.npz"),
                        "raw_label_archive": str(session / "privileged" / "raw_labels.npz"),
                        "image_archive_index": index,
                        "candidate_vertices_in_slab": int(np.count_nonzero(inside)),
                        "minimum_elevation_distance_mm": float(np.min(projected["elevation_distance_mm"])) if len(projected["elevation_distance_mm"]) else None,
                        "patch_row_min": int(np.min(projected["rows"][inside])) if np.any(inside) else None,
                        "patch_row_max": int(np.max(projected["rows"][inside])) if np.any(inside) else None,
                        "patch_column_min": int(np.min(projected["columns"][inside])) if np.any(inside) else None,
                        "patch_column_max": int(np.max(projected["columns"][inside])) if np.any(inside) else None,
                    }
                )
    sessions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in frame_rows:
        sessions.setdefault((row["source_id"], row["session_id"]), []).append(row)
    per_session = []
    for (source_id, session_id), rows in sorted(sessions.items()):
        valid = [row for row in rows if row["projection_status"] != "MISSING_METADATA"]
        per_session.append(
            {
                "source_id": source_id,
                "session_id": session_id,
                "frames": len(rows),
                "joined": len(valid),
                "verified": sum(row["projection_status"] == "VERIFIED" for row in valid),
                "median_exact_agreement": float(np.median([row["exact_raw_label_agreement"] for row in valid])) if valid else None,
                "median_bone_iou": float(np.median([row["bone_mask_iou"] for row in valid])) if valid else None,
                "distinct_hashes": len({row["frame_sha256"] for row in rows}),
                "distinct_reduced_poses": len({tuple(np.round(row.get("realized_pose_index", []), 6)) for row in valid}),
            }
        )
    summary = {
        "frames_available": len(frame_rows),
        "frames_joined": sum(row["projection_status"] != "MISSING_METADATA" for row in frame_rows),
        "exact_pose_replay_frames": 0,
        "reduced_pose_reconstruction_frames": sum(row["projection_status"] != "MISSING_METADATA" for row in frame_rows),
        "verified_frames": sum(row["projection_status"] == "VERIFIED" for row in frame_rows),
        "missing_metadata_frames": sum(row["projection_status"] == "MISSING_METADATA" for row in frame_rows),
        "unique_image_hashes": len({(row["source_id"], row["frame_sha256"]) for row in frame_rows}),
        "clipped_sample_total": sum(int(row.get("clipped_sample_count", 0)) for row in frame_rows),
        "per_session": per_session,
    }
    return frame_rows, candidate_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value for key, value in row.items()}
            )


def render_integrated_examples(
    *, package_root: Path, candidate_rows: list[dict[str, Any]], candidates: list[CandidateRevision]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    for candidate_id in sorted(by_id):
        rows = [
            row for row in candidate_rows
            if row["candidate_id"] == candidate_id and row["candidate_intersection"] == "PRESENT"
        ]
        seen: set[str] = set()
        for row in rows:
            if row["frame_sha256"] in seen:
                continue
            seen.add(row["frame_sha256"])
            selected.append(row)
            if len([x for x in selected if x["candidate_id"] == candidate_id]) == 3:
                break
    raw_root = package_root / "verification_bundle"
    raw_root.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(selected):
        image = np.load(row["image_archive"])["ultrasound"][int(row["image_archive_index"])]
        labels = np.load(row["raw_label_archive"])["raw_labels"][int(row["image_archive_index"])]
        stem = f"{index:02d}_{row['candidate_id']}_{row['session_id']}_{row['frame_sequence']}"
        np.savez_compressed(raw_root / f"{stem}.npz", ultrasound=image, raw_labels=labels)
        figure, axes = plt.subplots(1, 2, figsize=(9, 5), constrained_layout=True)
        axes[0].imshow(image, cmap="gray", origin="upper")
        axes[0].set_title("Clean actor-visible ultrasound")
        axes[1].imshow(image, cmap="gray", origin="upper")
        rectangle = plt.Rectangle(
            (row["patch_column_min"], row["patch_row_min"]),
            row["patch_column_max"] - row["patch_column_min"] + 1,
            row["patch_row_max"] - row["patch_row_min"] + 1,
            fill=False,
            color="yellow",
            linewidth=2,
        )
        axes[1].add_patch(rectangle)
        axes[1].set_title("Reviewer-only geometric slab bounds\nnot semantic visibility")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(f"{row['candidate_id']} | {row['session_id']} frame {row['frame_sequence']}")
        figure.savefig(raw_root / f"{stem}.png", dpi=180)
        plt.close(figure)
        row["verification_npz"] = f"verification_bundle/{stem}.npz"
        row["verification_png"] = f"verification_bundle/{stem}.png"
    return selected


def write_training_manifest(
    *, package_root: Path, candidate_rows: list[dict[str, Any]], selected: list[dict[str, Any]]
) -> dict[str, Any]:
    root = package_root / "training_preparation"
    root.mkdir(parents=True, exist_ok=True)
    # Only verified intersections have coordinates worth reviewing.  None is eligible
    # until anatomy and ultrasound semantic identity are independently reviewed.
    potential = [row for row in candidate_rows if row["candidate_intersection"] == "PRESENT"]
    write_csv(root / "candidate_sample_manifest.csv", potential)
    sessions = sorted({(row["source_id"], row["session_id"]) for row in candidate_rows})
    split = {
        "schema": "sonogym-sap-tp-session-splits-correction-v1",
        "all_sources_are_previously_opened_development": True,
        "ua_00_through_ua_03_role": "previously_opened_development_not_untouched_audit",
        "sessions": [
            {
                "source_id": source,
                "session_id": session,
                "role": "review_only" if any(row["session_id"] == session for row in selected) else "opened_development",
            }
            for source, session in sessions
        ],
        "future_unseen_confirmation": "requires_new_sealed_capture_after_model_selection",
    }
    (root / "session_split_manifest.yaml").write_text(yaml.safe_dump(split, sort_keys=False), encoding="utf-8")
    target = {
        "supervision": "image_patch",
        "patch_source_during_training": "reviewed_candidate_projection_only",
        "patch_source_during_inference": "actor_visible_bone_interface_morphology_proposals_not_privileged_coordinates",
        "patch_size_pixels": [48, 80],
        "feature_size": [16, 16],
        "multiple_landmarks": "AMBIGUOUS_EXCLUDED unless multilabel review explicitly supports all identities",
        "unknown_or_unreviewed": "EXCLUDED_never_negative",
        "hard_negatives": "only independently reviewed verified nonintersection or wrong-structure patches",
        "training_executed": False,
    }
    (root / "label_target.json").write_text(json.dumps(target, indent=2) + "\n", encoding="utf-8")
    eligible = sum(bool(row["training_eligible"]) for row in potential)
    return {
        "candidate_rows": len(potential),
        "training_eligible_rows": eligible,
        "status": "DATASET_PREPARED_PENDING_REVIEW" if potential else "TRAINING_BLOCKED_WITH_SPECIFIC_REASON",
        "reason": "anatomical approval and image-semantic review are pending",
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol["schema"] != "sonogym-s0010-right-sap-tp-correction-protocol-v1":
        raise RuntimeError("unexpected protocol")
    package_root = args.package_root
    package_root.mkdir(parents=True, exist_ok=True)
    patient = args.asset_root / "selected_dataset_stl" / "s0010"
    segmentation_path = patient / "combined_label_map.nii.gz"
    segmentation = np.asanyarray(nib.load(segmentation_path).dataobj).astype(np.uint8)
    candidates = load_revisions(
        asset_root=args.asset_root,
        annotation_source=args.annotation_source,
        segmentation=segmentation,
    )
    geometry_root = package_root / "boundary_revisions"
    boundary_records = []
    for candidate in candidates:
        files = export_revision(candidate, geometry_root / "editable_geometry")
        figure = geometry_root / "figures" / f"{candidate.candidate_id}_before_after.png"
        render_revision(candidate, figure)
        boundary_records.append(
            {
                "candidate_id": candidate.candidate_id,
                "source_status": candidate.source_status,
                "result_status": candidate.status,
                "method": candidate.method,
                "base_transition_x_index": candidate.base_transition_x_index,
                "source_mesh": str(candidate.source_mesh_path),
                "source_mesh_sha256": candidate.source_mesh_sha256,
                "geometry": files,
                "figure": str(figure.relative_to(package_root)),
                "independent_anatomical_approval": "PENDING",
                "actor_usable": False,
            }
        )
    surface_y = safe_torch(patient / "body_lowest_y_array.pt")
    surface_normals = safe_torch(patient / "body_surface_normal_array.pt")
    frame_rows, candidate_rows, replay_summary = replay(
        sources=protocol["capture_sources"],
        raw_volume=segmentation,
        surface_y=surface_y,
        surface_normals=surface_normals,
        candidates=candidates,
    )
    write_csv(package_root / "projection_repair" / "per_frame_metrics.csv", frame_rows)
    write_csv(package_root / "projection_repair" / "candidate_correspondence.csv", candidate_rows)
    (package_root / "projection_repair" / "replay_summary.json").write_text(
        json.dumps(replay_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selected = render_integrated_examples(
        package_root=package_root, candidate_rows=candidate_rows, candidates=candidates
    )
    training = write_training_manifest(
        package_root=package_root, candidate_rows=candidate_rows, selected=selected
    )
    historical_checksum = sha256(args.historical_package)
    if historical_checksum != protocol["historical_package"]["sha256"]:
        raise RuntimeError("historical package checksum changed")
    manifest = {
        "schema": "sonogym-s0010-right-sap-tp-correction-result-v1",
        "source_commit": args.source_commit,
        "historical_evidence_preserved": {
            "package": str(args.historical_package),
            "sha256": historical_checksum,
            "audited_frames": 3826,
            "verified_projections": 0,
            "approved_boundaries": 0,
        },
        "projection_repair": {
            "historical_bug_reproduced": True,
            "corrected_axes": "local (width, elevation, depth); array (row=depth, column=width)",
            "full_transform_search": "not_present_in_capture_records",
            "mode": "REDUCED_POSE_RECONSTRUCTION",
            "frozen_gate": {"exact_raw_label_agreement_minimum": EXACT_MIN, "bone_mask_iou_minimum": BONE_IOU_MIN},
            "result": replay_summary,
            "status": "PROJECTION_PARITY_PASS" if replay_summary["verified_frames"] else "PROJECTION_REPAIR_BLOCKED",
        },
        "boundary_review": boundary_records,
        "integrated_examples": selected,
        "training": training,
        "annotation_promotion": "NOT_PERFORMED",
        "model_training": "NOT_RUN",
        "actor_usable": False,
    }
    (package_root / "correction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(args.protocol, package_root / args.protocol.name)
    (package_root / "README.md").write_text(
        "# s0010 RIGHT SAP/TP correction package\n\n"
        "This package preserves the historical v2 package and records native-axis projection "
        "repair separately. All existing captures use REDUCED_POSE_RECONSTRUCTION because full "
        "end-effector transforms were not recorded. Yellow rectangles are evaluator-only geometric "
        "slab bounds, not semantic visibility. TP revisions are conservative distal surface-core "
        "proposals with an uncertain base; SAP boundaries remain unresolved. No annotation is "
        "independently approved, no row is a training label, and no model was fit.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--annotation-source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--historical-package", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    for name in ("asset_root", "annotation_source", "protocol", "historical_package", "package_root"):
        setattr(args, name, getattr(args, name).resolve())
    result = build(args)
    print(json.dumps({
        "projection_status": result["projection_repair"]["status"],
        "verified_frames": result["projection_repair"]["result"]["verified_frames"],
        "training_status": result["training"]["status"],
        "package_root": str(args.package_root),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
