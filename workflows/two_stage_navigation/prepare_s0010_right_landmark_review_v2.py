#!/usr/bin/env python3
"""Build the final RIGHT SAP/TP review-v2 package with real SonoGym ultrasound.

The four geometry candidates are reconstructed exactly from annotation v2.  This workflow adds
review evidence; it never promotes a candidate or makes it actor-usable.  Candidate projections
are drawn only when a pose-derived image plane reproduces the stored raw-label slice closely
enough to satisfy the frozen projection-integrity check.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
import trimesh
import yaml

import expand_s0010_right_landmark_review as expanded
import review_s0010_right_landmarks as base


IMAGE_WIDTH = 150
IMAGE_DEPTH = 200
IMAGE_RESOLUTION_M = 0.0005
LABEL_RESOLUTION_M = 0.0015
PROBE_HEIGHT_M = 0.13
IMAGE_ORIGIN_HEIGHT_M = 0.133
PROJECTION_LABEL_AGREEMENT_MINIMUM = 0.985
PROJECTION_BONE_IOU_MINIMUM = 0.90
PLANE_DISTANCE_TOLERANCE_MM = 1.5
REVIEW_FRAMES_PER_CANDIDATE = 3
LEVEL_LABEL = {"L4": 28, "L5": 27}
CANDIDATE_COLOR = {
    "superior_articular_process": "#2676b8",
    "transverse_process": "#f28e2b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_torch_array(path: Path) -> np.ndarray:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older pinned torch
        value = torch.load(path, map_location="cpu")
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("cannot normalize a zero/nonfinite vector")
    return vector / norm


def image_plane_from_realized_pose(
    pose_index: np.ndarray,
    surface_y_index: np.ndarray,
    surface_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the label-volume coordinates sampled by a SonoGym ultrasound frame.

    Returns (coords_index_hwc, image_origin_index, rotation_columns).  This mirrors
    SurfaceMotionPlanner/LabelImgSlicer, but is accepted only after stored-label round-trip
    validation.  The realized pose provides surface x/z and in-plane angle; the patient surface
    map supplies y and the normal.
    """

    pose = np.asarray(pose_index, dtype=np.float64)
    if pose.shape != (3,) or not np.all(np.isfinite(pose)):
        raise ValueError("realized pose must be a finite length-three vector")
    ix = int(np.clip(np.trunc(pose[0]), 0, surface_y_index.shape[0] - 1))
    iz = int(np.clip(np.trunc(pose[1]), 0, surface_y_index.shape[1] - 1))
    normal = normalize(np.asarray(surface_normals[ix, iz], dtype=np.float64))
    x_projection = np.asarray([math.cos(pose[2]), 0.0, math.sin(pose[2])])
    y_axis = normalize(np.cross(normal, x_projection))
    x_axis = normalize(np.cross(y_axis, normal))
    rotation = np.column_stack([x_axis, y_axis, normal])

    surface_point_m = np.asarray(
        [
            pose[0] * LABEL_RESOLUTION_M,
            float(surface_y_index[ix, iz]) * LABEL_RESOLUTION_M,
            pose[1] * LABEL_RESOLUTION_M,
        ]
    )
    image_origin_m = surface_point_m + (
        IMAGE_ORIGIN_HEIGHT_M - PROBE_HEIGHT_M
    ) * normal
    rows, columns = np.meshgrid(
        np.arange(IMAGE_DEPTH), np.arange(IMAGE_WIDTH), indexing="ij"
    )
    local = np.stack(
        [
            (columns - IMAGE_WIDTH // 2) * IMAGE_RESOLUTION_M,
            rows * IMAGE_RESOLUTION_M,
            np.zeros_like(rows, dtype=np.float64),
        ],
        axis=-1,
    )
    coordinates_m = image_origin_m + local @ rotation.T
    return coordinates_m / LABEL_RESOLUTION_M, image_origin_m / LABEL_RESOLUTION_M, rotation


def sample_volume(volume: np.ndarray, coordinates_index: np.ndarray) -> np.ndarray:
    indices = np.trunc(coordinates_index).astype(np.int64)
    for axis in range(3):
        indices[..., axis] = np.clip(indices[..., axis], 0, volume.shape[axis] - 1)
    return volume[indices[..., 0], indices[..., 1], indices[..., 2]]


def projection_integrity(
    reconstructed: np.ndarray, stored: np.ndarray
) -> dict[str, float | bool]:
    if reconstructed.shape != stored.shape:
        raise ValueError("reconstructed/stored label shapes differ")
    exact = float(np.mean(reconstructed == stored))
    predicted_bone = np.isin(reconstructed, list(range(25, 44)))
    stored_bone = np.isin(stored, list(range(25, 44)))
    union = int(np.count_nonzero(predicted_bone | stored_bone))
    bone_iou = (
        float(np.count_nonzero(predicted_bone & stored_bone) / union) if union else 1.0
    )
    return {
        "exact_label_agreement": exact,
        "bone_mask_iou": bone_iou,
        "verified": bool(
            exact >= PROJECTION_LABEL_AGREEMENT_MINIMUM
            and bone_iou >= PROJECTION_BONE_IOU_MINIMUM
        ),
    }


def candidate_projection(
    points_index: np.ndarray,
    image_origin_index: np.ndarray,
    rotation: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    local_m = (
        (np.asarray(points_index) - image_origin_index) * LABEL_RESOLUTION_M
    ) @ rotation
    columns = np.rint(local_m[:, 0] / IMAGE_RESOLUTION_M + IMAGE_WIDTH // 2).astype(int)
    rows = np.rint(local_m[:, 1] / IMAGE_RESOLUTION_M).astype(int)
    plane_distance_mm = np.abs(local_m[:, 2]) * 1000.0
    inside = (
        (columns >= 0)
        & (columns < IMAGE_WIDTH)
        & (rows >= 0)
        & (rows < IMAGE_DEPTH)
        & (plane_distance_mm <= PLANE_DISTANCE_TOLERANCE_MM)
    )
    mask = np.zeros((IMAGE_DEPTH, IMAGE_WIDTH), dtype=bool)
    mask[rows[inside], columns[inside]] = True
    if np.any(mask):
        mask = ndimage.binary_dilation(mask, iterations=2)
    return mask, {
        "candidate_vertices_in_plane": int(np.count_nonzero(inside)),
        "projected_pixels": int(np.count_nonzero(mask)),
        "minimum_plane_distance_mm": float(np.min(plane_distance_mm)),
        "intersects_plane": bool(np.any(inside)),
    }


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    level: str
    landmark: str
    source_status: str
    points_index: np.ndarray
    full_vertices: np.ndarray
    faces: np.ndarray


@dataclass
class FrameEvidence:
    source_id: str
    source_root: str
    session_id: str
    split: str
    frame_sequence: int
    producer_sequence: int
    frame_sha256: str
    phase: str
    realized_pose_index: np.ndarray
    ultrasound: np.ndarray
    raw_labels: np.ndarray
    ct_plane: np.ndarray
    candidate_mask: np.ndarray
    projection: dict[str, Any]
    integrity: dict[str, Any]
    raw_level_pixels: int
    image_archive_sha256: str
    label_archive_sha256: str

    def public_record(self, candidate_id: str, level: str) -> dict[str, Any]:
        semantic = (
            "AMBIGUOUS_NOT_SEMANTICALLY_VERIFIED"
            if self.integrity["verified"] and self.projection["intersects_plane"]
            else "NOT_OBSERVED_IN_EXISTING_DATA"
        )
        return {
            "candidate_id": candidate_id,
            "vertebral_level": level,
            "source_id": self.source_id,
            "source_root": self.source_root,
            "session_id": self.session_id,
            "split": self.split,
            "frame_sequence": self.frame_sequence,
            "producer_sequence": self.producer_sequence,
            "frame_sha256": self.frame_sha256,
            "phase": self.phase,
            "realized_pose_index_evaluator_only": self.realized_pose_index.tolist(),
            "projection_integrity": self.integrity,
            "candidate_projection": self.projection,
            "landmark_geometry_intersects_plane": bool(
                self.integrity["verified"] and self.projection["intersects_plane"]
            ),
            "vertebral_bone_interface_present": bool(self.raw_level_pixels > 0),
            "specific_landmark_distinguishable": False,
            "ultrasound_observability_status": semantic,
            "training_label_eligible": False,
            "image_archive_sha256": self.image_archive_sha256,
            "raw_label_archive_sha256": self.label_archive_sha256,
        }


def load_candidates(
    asset_root: Path, annotation_source: Path
) -> tuple[list[Candidate], dict[str, trimesh.Trimesh]]:
    source = json.loads(annotation_source.read_text(encoding="utf-8"))
    annotations = source["annotations"]["group_B_future_MBB_bony_geometry"]
    meshes: dict[str, trimesh.Trimesh] = {}
    candidates: list[Candidate] = []
    for level in ("L4", "L5"):
        mesh = trimesh.load(
            asset_root / "selected_dataset_stl" / base.PATIENT_ID / f"vertebrae_{level}.stl",
            force="mesh",
            process=False,
        )
        meshes[level] = mesh
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        for landmark in ("superior_articular_process", "transverse_process"):
            row = base.source_candidate(annotations, level, landmark)
            mask = base.quantile_vertex_mask(vertices, row["geometry"]["selector"])
            if int(np.count_nonzero(mask)) != int(row["geometry"]["selected_vertex_count"]):
                raise RuntimeError(f"candidate reconstruction changed: {level} {landmark}")
            candidates.append(
                Candidate(
                    candidate_id=f"{level}_RIGHT_{landmark}",
                    level=level,
                    landmark=landmark,
                    source_status=row["status"],
                    points_index=vertices[mask],
                    full_vertices=vertices,
                    faces=np.asarray(mesh.faces, dtype=np.int32),
                )
            )
    return candidates, meshes


def render_anatomical_sheet(
    *,
    candidate: Candidate,
    ct: np.ndarray,
    segmentation: np.ndarray,
    spacing_mm: np.ndarray,
    output: Path,
) -> None:
    color = CANDIDATE_COLOR[candidate.landmark]
    centroid = np.mean(candidate.points_index, axis=0)
    figure = plt.figure(figsize=(28, 19), constrained_layout=True)
    grid = figure.add_gridspec(4, 5, height_ratios=[1.15, 1, 1, 1])
    views = ((25, -70, "Posterior-oblique overview"), (55, -95, "Close-up"))
    for column, (elev, azim, title) in enumerate(views):
        axis = figure.add_subplot(grid[0, column * 2 : column * 2 + 2], projection="3d")
        rng = np.random.default_rng(20261117)
        unique = np.unique(candidate.full_vertices, axis=0)
        chosen = rng.choice(len(unique), min(12000, len(unique)), replace=False)
        background = unique[chosen]
        axis.scatter(
            background[:, 0], background[:, 1], background[:, 2],
            s=0.35, c="#aaaaaa", alpha=0.16, depthshade=False,
        )
        axis.scatter(
            candidate.points_index[:, 0], candidate.points_index[:, 1],
            candidate.points_index[:, 2], s=2.0, c=color, alpha=0.85,
            depthshade=False,
        )
        axis.view_init(elev=elev, azim=azim)
        axis.set_xlabel("RIGHT +x (index)")
        axis.set_ylabel("ANTERIOR +y (index)")
        axis.set_zlabel("CRANIAL +z (index)")
        axis.set_title(title)
        axis.set_box_aspect(np.ptp(candidate.full_vertices, axis=0))
        if column == 1:
            low = np.min(candidate.points_index, axis=0) - 9
            high = np.max(candidate.points_index, axis=0) + 9
            axis.set_xlim(low[0], high[0])
            axis.set_ylim(low[1], high[1])
            axis.set_zlim(low[2], high[2])
    info_axis = figure.add_subplot(grid[0, 4])
    info_axis.axis("off")
    info_axis.text(
        0.0,
        1.0,
        "REVIEW-ONLY CANDIDATE\n\n"
        f"{candidate.candidate_id}\n"
        f"source status: {candidate.source_status}\n"
        f"candidate vertices: {len(candidate.points_index):,}\n"
        f"CT spacing: {spacing_mm.tolist()} mm\n\n"
        "Gray = complete vertebra mesh\n"
        f"{color} = proposed candidate\n"
        "Green contour = complete vertebral label\n\n"
        "Check contamination by lamina, pedicle,\n"
        "vertebral body and adjacent posterior elements.\n"
        "Straight edges reflect the quantile heuristic,\n"
        "not a semantic anatomical boundary.",
        va="top",
        fontsize=12,
    )

    center = np.rint(centroid).astype(int)
    specs = (
        ("Axial", 2, 0, 1, "RIGHT +x", "ANTERIOR +y"),
        ("Sagittal", 0, 1, 2, "ANTERIOR +y", "CRANIAL +z"),
        ("Coronal", 1, 0, 2, "RIGHT +x", "CRANIAL +z"),
    )
    offsets = (-4, -2, 0, 2, 4)
    raw_label = LEVEL_LABEL[candidate.level]
    for row_index, (name, fixed, horizontal, vertical, xlabel, ylabel) in enumerate(specs, 1):
        for column, offset in enumerate(offsets):
            coordinate = int(np.clip(center[fixed] + offset, 0, ct.shape[fixed] - 1))
            if fixed == 2:
                image = ct[:, :, coordinate].T
                labels = segmentation[:, :, coordinate].T
            elif fixed == 0:
                image = ct[coordinate, :, :].T
                labels = segmentation[coordinate, :, :].T
            else:
                image = ct[:, coordinate, :].T
                labels = segmentation[:, coordinate, :].T
            local = base.projected_points(candidate.points_index, fixed, coordinate, 1.0)
            axis = figure.add_subplot(grid[row_index, column])
            axis.imshow(np.clip(image, -300, 1200), cmap="gray", origin="lower")
            axis.contour(labels == raw_label, [0.5], colors=["#39ff14"], linewidths=1.1)
            if len(local):
                axis.scatter(local[:, horizontal], local[:, vertical], s=5, c=color, alpha=0.75)
            roi = expanded.patch_roi(candidate.points_index, horizontal, vertical, image.shape)
            axis.set_xlim(roi[1].start, roi[1].stop)
            axis.set_ylim(roi[0].start, roi[0].stop)
            scale_voxels = 10.0 / spacing_mm[horizontal]
            x0 = roi[1].start + 3
            y0 = roi[0].start + 3
            axis.plot([x0, x0 + scale_voxels], [y0, y0], color="white", lw=3)
            axis.text(x0, y0 + 2, "10 mm", color="white", fontsize=8)
            axis.set_title(f"{name} {offset * spacing_mm[fixed]:+.1f} mm")
            if column == 0:
                axis.set_ylabel(ylabel)
            if row_index == 3:
                axis.set_xlabel(xlabel)
            axis.tick_params(labelsize=7)
    figure.suptitle(
        f"s0010 {candidate.level} RIGHT {candidate.landmark.replace('_', ' ').upper()} — anatomical boundary review\n"
        "Evaluator-only CT/mesh evidence; candidate is not semantic ground truth",
        fontsize=18,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def iter_capture_sessions(capture_roots: list[dict[str, str]]) -> Iterable[tuple[str, Path]]:
    for source in capture_roots:
        root = Path(source["root"])
        for path in sorted(root.glob("sessions/*/public/frames.jsonl")):
            yield source["id"], path.parents[1]


def collect_frame_evidence(
    *,
    candidates: list[Candidate],
    capture_roots: list[dict[str, str]],
    raw_volume: np.ndarray,
    ct_volume: np.ndarray,
    surface_y: np.ndarray,
    surface_normals: np.ndarray,
) -> tuple[dict[str, list[FrameEvidence]], dict[str, Any]]:
    evidence: dict[str, list[FrameEvidence]] = {row.candidate_id: [] for row in candidates}
    audit = {
        "sources": {},
        "frames_scanned": 0,
        "frames_with_joined_pose": 0,
        "projection_verified_frames": 0,
    }
    for source_id, session_root in iter_capture_sessions(capture_roots):
        public_path = session_root / "public" / "frames.jsonl"
        privileged_path = session_root / "privileged" / "annotations.jsonl"
        ultrasound_path = session_root / "public" / "ultrasound.npz"
        labels_path = session_root / "privileged" / "raw_labels.npz"
        if not all(path.exists() for path in (public_path, privileged_path, ultrasound_path, labels_path)):
            continue
        public = read_jsonl(public_path)
        privileged = read_jsonl(privileged_path)
        annotations = {
            (int(row["frame_sequence"]), row["frame_sha256"]): row for row in privileged
        }
        ultrasound = np.load(ultrasound_path)["ultrasound"]
        raw_labels = np.load(labels_path)["raw_labels"]
        if len(public) != len(ultrasound) or len(public) != len(raw_labels):
            raise RuntimeError(f"archive/manifest length mismatch: {session_root}")
        source_audit = audit["sources"].setdefault(
            source_id, {"sessions": 0, "frames": 0, "projection_verified": 0}
        )
        source_audit["sessions"] += 1
        image_archive_sha = sha256(ultrasound_path)
        label_archive_sha = sha256(labels_path)
        for index, row in enumerate(public):
            audit["frames_scanned"] += 1
            source_audit["frames"] += 1
            key = (int(row["frame_sequence"]), row["frame_sha256"])
            annotation = annotations.get(key)
            if annotation is None or "realized_pose_index" not in annotation:
                continue
            audit["frames_with_joined_pose"] += 1
            coordinates, origin, rotation = image_plane_from_realized_pose(
                np.asarray(annotation["realized_pose_index"], dtype=np.float64),
                surface_y,
                surface_normals,
            )
            reconstructed = sample_volume(raw_volume, coordinates)
            stored = np.asarray(raw_labels[index])
            integrity = projection_integrity(reconstructed, stored)
            if integrity["verified"]:
                audit["projection_verified_frames"] += 1
                source_audit["projection_verified"] += 1
            ct_plane = sample_volume(ct_volume, coordinates)
            for candidate in candidates:
                raw_level_pixels = int(np.count_nonzero(stored == LEVEL_LABEL[candidate.level]))
                mask, projection = candidate_projection(candidate.points_index, origin, rotation)
                if raw_level_pixels == 0 and not projection["intersects_plane"]:
                    continue
                evidence[candidate.candidate_id].append(
                    FrameEvidence(
                        source_id=source_id,
                        source_root=str(session_root.parents[1]),
                        session_id=str(row["session_id"]),
                        split=str(row.get("split", "unknown")),
                        frame_sequence=int(row["frame_sequence"]),
                        producer_sequence=int(row["producer_sequence"]),
                        frame_sha256=row["frame_sha256"],
                        phase=str(row.get("phase", "unknown")),
                        realized_pose_index=np.asarray(
                            annotation["realized_pose_index"], dtype=np.float64
                        ),
                        ultrasound=np.asarray(ultrasound[index], dtype=np.float32),
                        raw_labels=stored,
                        ct_plane=ct_plane,
                        candidate_mask=mask,
                        projection=projection,
                        integrity=integrity,
                        raw_level_pixels=raw_level_pixels,
                        image_archive_sha256=image_archive_sha,
                        label_archive_sha256=label_archive_sha,
                    )
                )
    return evidence, audit


def select_review_frames(rows: list[FrameEvidence]) -> list[FrameEvidence]:
    ranked = sorted(
        rows,
        key=lambda row: (
            not bool(row.integrity["verified"] and row.projection["intersects_plane"]),
            -int(row.projection["projected_pixels"]),
            -row.raw_level_pixels,
            row.session_id,
            row.frame_sequence,
        ),
    )
    selected: list[FrameEvidence] = []
    seen_sessions: set[str] = set()
    seen_hashes: set[str] = set()
    for row in ranked:
        if row.frame_sha256 in seen_hashes:
            continue
        if row.session_id in seen_sessions and len(seen_sessions) < REVIEW_FRAMES_PER_CANDIDATE:
            continue
        selected.append(row)
        seen_sessions.add(row.session_id)
        seen_hashes.add(row.frame_sha256)
        if len(selected) == REVIEW_FRAMES_PER_CANDIDATE:
            break
    if len(selected) < REVIEW_FRAMES_PER_CANDIDATE:
        for row in ranked:
            if row.frame_sha256 in seen_hashes:
                continue
            selected.append(row)
            seen_hashes.add(row.frame_sha256)
            if len(selected) == REVIEW_FRAMES_PER_CANDIDATE:
                break
    return selected


def render_ultrasound_sheet(
    candidate: Candidate, rows: list[FrameEvidence], output: Path
) -> None:
    color = CANDIDATE_COLOR[candidate.landmark]
    if not rows:
        figure, axis = plt.subplots(figsize=(14, 7))
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            f"{candidate.candidate_id}\n\nNOT_OBSERVED_IN_EXISTING_DATA\n\n"
            "No stored frame could be joined to a verified pose/label record near this candidate.\n"
            "No candidate overlay was fabricated.",
            ha="center",
            va="center",
            fontsize=16,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=220)
        plt.close(figure)
        return
    figure, axes = plt.subplots(len(rows), 3, figsize=(18, 5.5 * len(rows)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_index, row in enumerate(rows):
        clean = axes[row_index, 0]
        overlay = axes[row_index, 1]
        ct_axis = axes[row_index, 2]
        clean.imshow(row.ultrasound, cmap="gray", origin="upper", aspect="auto")
        clean.set_title("Actor-visible clean conv ultrasound")
        overlay.imshow(row.ultrasound, cmap="gray", origin="upper", aspect="auto")
        verified_intersection = bool(
            row.integrity["verified"] and row.projection["intersects_plane"]
        )
        if verified_intersection and np.any(row.candidate_mask):
            overlay.contour(row.candidate_mask, [0.5], colors=[color], linewidths=1.4)
            overlay.set_title("Evaluator-only candidate projection")
        else:
            level_mask = row.raw_labels == LEVEL_LABEL[candidate.level]
            if np.any(level_mask):
                overlay.contour(level_mask, [0.5], colors=["#39ff14"], linewidths=1.0)
            overlay.set_title("Evaluator-only vertebral context; NO candidate overlay")
        if row.integrity["verified"]:
            ct_axis.imshow(row.ct_plane, cmap="gray", origin="upper", aspect="auto")
            level_mask = row.raw_labels == LEVEL_LABEL[candidate.level]
            if np.any(level_mask):
                ct_axis.contour(level_mask, [0.5], colors=["#39ff14"], linewidths=0.9)
        if verified_intersection and np.any(row.candidate_mask):
            ct_axis.contour(row.candidate_mask, [0.5], colors=[color], linewidths=1.4)
            ct_axis.set_title("Verified CT imaging plane (evaluator-only)")
        elif row.integrity["verified"]:
            ct_axis.set_title("Verified CT plane; candidate does not intersect")
        else:
            ct_axis.axis("off")
            ct_axis.text(
                0.5,
                0.58,
                "VERIFIED CT PLANE UNAVAILABLE",
                ha="center",
                va="center",
                fontsize=14,
                weight="bold",
                color="#b22222",
            )
            ct_axis.text(
                0.5,
                0.39,
                "Stored captures contain reduced (x, z, angle) pose,\n"
                "but not the full end-effector transform or per-pixel\n"
                "human_img_coords needed for exact replay.\n\n"
                f"Round-trip label agreement: {row.integrity['exact_label_agreement']:.4f}\n"
                f"Bone-mask IoU: {row.integrity['bone_mask_iou']:.4f}\n\n"
                "Candidate projection intentionally withheld.",
                ha="center",
                va="center",
                fontsize=11,
            )
        metadata = (
            f"{row.source_id}/{row.session_id} frame={row.frame_sequence} "
            f"producer={row.producer_sequence}\n"
            f"pose index={np.round(row.realized_pose_index, 4).tolist()}  "
            f"hash={row.frame_sha256[:16]}...\n"
            f"label agreement={row.integrity['exact_label_agreement']:.4f}; "
            f"bone IoU={row.integrity['bone_mask_iou']:.4f}; "
            f"candidate vertices in plane={row.projection['candidate_vertices_in_plane']}\n"
            "Green=whole vertebral label context; colored=candidate only when verified. "
            "Intersection does not prove semantic recognizability."
        )
        clean.set_xlabel(metadata, fontsize=9)
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        f"s0010 {candidate.candidate_id} — real SonoGym conv ultrasound correspondence\n"
        "Three claims are separate: plane intersection / bone interface / semantic SAP-TP distinction",
        fontsize=16,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def render_contact_sheet(
    candidates: list[Candidate], anatomical_paths: dict[str, Path], ultrasound_paths: dict[str, Path], output: Path
) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(18, 25), constrained_layout=True)
    for row, candidate in enumerate(candidates):
        for column, path in enumerate((anatomical_paths[candidate.candidate_id], ultrasound_paths[candidate.candidate_id])):
            image = plt.imread(path)
            axes[row, column].imshow(image)
            axes[row, column].axis("off")
            axes[row, column].set_title(
                f"{candidate.candidate_id}\n{'Anatomy' if column == 0 else 'Ultrasound'}",
                fontsize=12,
            )
    figure.suptitle(
        "s0010 RIGHT L4/L5 SAP/TP review v2 — rapid review contact sheet\n"
        "All candidate boundaries remain advisory and actor_usable=false",
        fontsize=18,
    )
    figure.savefig(output, dpi=110)
    plt.close(figure)


def write_training_preparation(
    package_root: Path,
    candidates: list[Candidate],
    selected: dict[str, list[FrameEvidence]],
    all_sessions: list[dict[str, str]],
) -> dict[str, Any]:
    root = package_root / "training_preparation"
    root.mkdir(parents=True, exist_ok=True)
    review_sessions = {row.session_id for rows in selected.values() for row in rows}
    schema = {
        "schema": "sonogym-sap-tp-semantic-label-schema-v1",
        "patient_id": "s0010",
        "laterality": "RIGHT",
        "classes": [candidate.candidate_id for candidate in candidates],
        "negative_class": "NONE_OF_REQUESTED_LANDMARKS",
        "allowed_label_status": ["VERIFIED_POSITIVE", "VERIFIED_NEGATIVE", "AMBIGUOUS", "IGNORE"],
        "current_candidate_status": {
            candidate.candidate_id: candidate.source_status for candidate in candidates
        },
        "training_allowed_now": False,
        "promotion_requires": [
            "qualified_reviewer_identity_and_qualification",
            "per_landmark_ACCEPT_BOUNDARY_or_corrected_geometry",
            "ultrasound_correspondence_review",
            "uncertainty_record",
        ],
    }
    (root / "semantic_landmark_label_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    correspondence_path = root / "image_landmark_correspondence.csv"
    fieldnames = [
        "candidate_id", "source_id", "session_id", "split", "frame_sequence",
        "producer_sequence", "frame_sha256", "projection_verified",
        "intersects_plane", "raw_level_pixels", "specific_landmark_distinguishable",
        "training_label_status", "training_eligible",
    ]
    with correspondence_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            for row in selected[candidate.candidate_id]:
                writer.writerow(
                    {
                        "candidate_id": candidate.candidate_id,
                        "source_id": row.source_id,
                        "session_id": row.session_id,
                        "split": row.split,
                        "frame_sequence": row.frame_sequence,
                        "producer_sequence": row.producer_sequence,
                        "frame_sha256": row.frame_sha256,
                        "projection_verified": row.integrity["verified"],
                        "intersects_plane": row.projection["intersects_plane"],
                        "raw_level_pixels": row.raw_level_pixels,
                        "specific_landmark_distinguishable": False,
                        "training_label_status": "AMBIGUOUS",
                        "training_eligible": False,
                    }
                )
    split_rows = []
    for row in sorted(all_sessions, key=lambda item: (item["source_id"], item["session_id"])):
        assignment = "review_only_excluded" if row["session_id"] in review_sessions else row["recorded_split"]
        split_rows.append({**row, "prepared_assignment": assignment})
    split_manifest = {
        "schema": "sonogym-sap-tp-session-split-preparation-v1",
        "development_only": True,
        "patient_count": 1,
        "review_sessions_excluded_from_unseen_evaluation": sorted(review_sessions),
        "sessions": split_rows,
        "future_confirmation": "REQUIRES_NEW_SEALED_SESSION_AFTER_MODEL_SELECTION",
    }
    (root / "session_disjoint_split_manifest.yaml").write_text(
        yaml.safe_dump(split_manifest, sort_keys=False), encoding="utf-8"
    )
    config = {
        "schema": "sonogym-sap-tp-minimal-cpu-training-config-v1",
        "execution": "PREPARED_NOT_RUN",
        "model": "ExtraTreesClassifier",
        "n_estimators": 240,
        "random_state": 20261117,
        "n_jobs": 4,
        "features": ["normalized_intensity_16x16", "normalized_gradient_16x16"],
        "sample_unit": "reviewer_verified_image_patch",
        "group_split": "session_id",
        "ambiguous_handling": "exclude_from_fit_retain_for_error_analysis",
        "class_weight": "balanced",
        "gates_before_fit": [
            "at_least_one_ACCEPT_BOUNDARY_with_corrected_geometry_if_needed",
            "verified_ultrasound_correspondence",
            "independent_negative_review",
            "zero_actor_evaluator_field_leakage",
        ],
        "metrics": [
            "per_class_precision_recall", "false_detections_per_frame",
            "image_pixel_or_point_localization_error_mm", "session_stratified_calibration",
            "abstention_rate",
        ],
    }
    (root / "minimal_cpu_training_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    readme = """# Training preparation (not authorization to train)

The files in this directory define the semantic classes, provenance-rich image correspondence,
session-disjoint development split, ambiguity handling, and the smallest CPU ExtraTrees baseline.
Every current correspondence row is `AMBIGUOUS` and `training_eligible=false`.  A qualified
anatomical reviewer must accept or correct a boundary and review its ultrasound correspondence
before any positive or negative label is promoted.  Review-selected sessions are excluded from
any claim of unseen evaluation; a future sealed confirmation needs new sessions.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    return {
        "schema": str((root / "semantic_landmark_label_schema.json").relative_to(package_root)),
        "correspondence": str(correspondence_path.relative_to(package_root)),
        "split_manifest": str((root / "session_disjoint_split_manifest.yaml").relative_to(package_root)),
        "cpu_config": str((root / "minimal_cpu_training_config.yaml").relative_to(package_root)),
        "training_executed": False,
        "training_ready": False,
    }


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol["schema"] != "sonogym-s0010-right-sap-tp-ultrasound-review-protocol-v2":
        raise RuntimeError("unexpected review protocol")
    package_root = args.package_root
    package_root.mkdir(parents=True, exist_ok=True)
    asset_root = args.asset_root
    patient_root = asset_root / "selected_dataset_stl" / "s0010"
    ct_path = asset_root / "selected_dataset" / "s0010" / "ct.nii.gz"
    segmentation_path = patient_root / "combined_label_map.nii.gz"
    ct_image = nib.load(ct_path)
    segmentation_image = nib.load(segmentation_path)
    ct = np.asanyarray(ct_image.dataobj)
    raw_volume = np.asanyarray(segmentation_image.dataobj).astype(np.uint8)
    spacing_mm = np.asarray(ct_image.header.get_zooms()[:3], dtype=np.float64)
    if ct.shape != raw_volume.shape or not np.allclose(ct_image.affine, segmentation_image.affine):
        raise RuntimeError("CT and raw segmentation grids disagree")

    candidates, _meshes = load_candidates(asset_root, args.annotation_source)
    expanded_manifest = expanded.build_expanded_review(
        asset_root=asset_root,
        annotation_source=args.annotation_source,
        protocol_path=args.expanded_protocol,
        package_root=package_root / "editable_review_material",
    )
    (package_root / "editable_review_material" / "expanded_review_manifest.json").write_text(
        json.dumps(expanded_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    anatomical_paths: dict[str, Path] = {}
    for candidate in candidates:
        path = package_root / "anatomical_sheets" / f"{candidate.candidate_id}_anatomical_review.png"
        render_anatomical_sheet(
            candidate=candidate,
            ct=ct,
            segmentation=raw_volume,
            spacing_mm=spacing_mm,
            output=path,
        )
        anatomical_paths[candidate.candidate_id] = path

    surface_y = safe_torch_array(patient_root / "body_lowest_y_array.pt")
    surface_normals = safe_torch_array(patient_root / "body_surface_normal_array.pt")
    evidence, capture_audit = collect_frame_evidence(
        candidates=candidates,
        capture_roots=protocol["capture_sources"],
        raw_volume=raw_volume,
        ct_volume=ct,
        surface_y=surface_y,
        surface_normals=surface_normals,
    )
    selected = {key: select_review_frames(value) for key, value in evidence.items()}
    ultrasound_paths: dict[str, Path] = {}
    correspondence_records: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        rows = selected[candidate.candidate_id]
        path = package_root / "ultrasound_sheets" / f"{candidate.candidate_id}_ultrasound_review.png"
        render_ultrasound_sheet(candidate, rows, path)
        ultrasound_paths[candidate.candidate_id] = path
        public_rows = [row.public_record(candidate.candidate_id, candidate.level) for row in rows]
        correspondence_records.extend(public_rows)
        verified_intersections = sum(
            int(row["landmark_geometry_intersects_plane"]) for row in public_rows
        )
        observability = (
            "AMBIGUOUS_NOT_SEMANTICALLY_VERIFIED"
            if verified_intersections
            else "NOT_OBSERVED_IN_EXISTING_DATA"
        )
        recommendation = (
            "REVISE_BOUNDARY"
            if candidate.landmark in {"superior_articular_process", "transverse_process"}
            else "REJECT_CANDIDATE"
        )
        suspected = (
            "axis-aligned candidate may include lamina/adjacent posterior elements"
            if candidate.landmark == "superior_articular_process"
            else "candidate may include anterior adjacent surface or pedicle/body-facing bone"
        )
        records.append(
            {
                "candidate_id": candidate.candidate_id,
                "source_annotation_status": candidate.source_status,
                "anatomical_boundary_status": "PENDING_QUALIFIED_REVIEW",
                "ultrasound_observability_status": observability,
                "agent_advisory_recommendation": recommendation,
                "recommendation_is_not_approval": True,
                "suspected_contamination": suspected,
                "boundary_uncertainty": "UNQUANTIFIED_PENDING_CORRECTED_SEMANTIC_BOUNDARY",
                "review_frames": public_rows,
                "anatomical_sheet": str(anatomical_paths[candidate.candidate_id].relative_to(package_root)),
                "ultrasound_sheet": str(path.relative_to(package_root)),
                "editable_geometry": f"editable_review_material/editable_geometry/{candidate.candidate_id}.npz",
                "semantic_training_eligible": False,
            }
        )

    correspondence_path = package_root / "ultrasound_correspondence_records.json"
    correspondence_path.write_text(
        json.dumps(correspondence_records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contact_path = package_root / "contact_sheet.png"
    render_contact_sheet(candidates, anatomical_paths, ultrasound_paths, contact_path)

    all_sessions: list[dict[str, str]] = []
    for source_id, session_root in iter_capture_sessions(protocol["capture_sources"]):
        rows = read_jsonl(session_root / "public" / "frames.jsonl")
        if rows:
            all_sessions.append(
                {
                    "source_id": source_id,
                    "session_id": str(rows[0]["session_id"]),
                    "recorded_split": str(rows[0].get("split", "development")),
                }
            )
    training = write_training_preparation(package_root, candidates, selected, all_sessions)

    response = expanded.response_template()
    for row in response["candidate_decisions"]:
        row["ultrasound_observability_decision"] = None
        row["corrected_geometry_sha256"] = None
    response["allowed_ultrasound_decisions"] = [
        "OBSERVABLE", "AMBIGUOUS", "NOT_OBSERVED_IN_EXISTING_DATA"
    ]
    (package_root / "review_response_template.json").write_text(
        json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(args.protocol, package_root / args.protocol.name)

    manifest = {
        "schema": "sonogym-s0010-right-sap-tp-review-manifest-v2",
        "patient_id": "s0010",
        "facet_level": "L4-L5",
        "laterality": "RIGHT",
        "source_commit": args.source_commit,
        "inputs": {
            "annotation": {"path": str(args.annotation_source), "sha256": sha256(args.annotation_source)},
            "ct": {"path": str(ct_path), "sha256": sha256(ct_path)},
            "segmentation": {"path": str(segmentation_path), "sha256": sha256(segmentation_path)},
            "capture_sources": protocol["capture_sources"],
        },
        "capture_audit": capture_audit,
        "stored_pose_metadata_limitation": {
            "full_end_effector_transform_recorded": False,
            "per_pixel_human_image_coordinates_recorded": False,
            "available": "reduced realized_pose_index [surface_x, surface_z, in_plane_angle]",
            "consequence": "candidate-specific CT-plane projection is not verified for existing captures",
        },
        "projection_contract": {
            "label_agreement_minimum": PROJECTION_LABEL_AGREEMENT_MINIMUM,
            "bone_iou_minimum": PROJECTION_BONE_IOU_MINIMUM,
            "plane_distance_tolerance_mm": PLANE_DISTANCE_TOLERANCE_MM,
            "overlay_drawn_only_when_round_trip_verified": True,
        },
        "records": records,
        "training_preparation": training,
        "anatomical_boundaries_accepted": 0,
        "semantic_training_labels_approved": 0,
        "annotation_promotion": "NOT_PERFORMED",
        "actor_usable": False,
        "model_training_authorized": False,
        "mbb_target_region": "UNRESOLVED",
        "overall_status": "REVIEW_PACKAGE_READY_TRAINING_BLOCKED_INDEPENDENT_REVIEW",
    }
    (package_root / "review_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readiness_rows = "\n".join(
        "| {candidate_id} | {source_annotation_status} | {anatomical_boundary_status} | "
        "{ultrasound_observability_status} | {agent_advisory_recommendation} | no |".format(**row)
        for row in records
    )
    readiness = f"""# Training readiness report

## Result

**Semantic SAP/TP detector training is `BLOCKED_INDEPENDENT_REVIEW_AND_CORRESPONDENCE`.**
The package is review-ready, but zero anatomical boundaries are accepted and zero stored frames
support a verified candidate-specific ultrasound projection.

| quantity | result |
|---|---:|
| real SonoGym conv frames audited | {capture_audit['frames_scanned']} |
| frames joined to recorded reduced pose | {capture_audit['frames_with_joined_pose']} |
| frames passing CT/raw-label projection round trip | {capture_audit['projection_verified_frames']} |
| boundaries accepted by qualified reviewer | 0 / 4 |
| training-eligible semantic labels | 0 / 4 |

| candidate | source status | boundary status | ultrasound status | advisory action | train now |
|---|---|---|---|---|---|
{readiness_rows}

## What the existing data establish

- The included images are actual model-based `conv` SonoGym observations for `s0010`.
- Stored raw vertebral labels show L4/L5 bone interfaces in the selected frames.
- The four editable geometry exports exactly reproduce the existing heuristic candidates.
- The existing metadata do **not** preserve the full end-effector transform or per-pixel
  `human_img_coords`; reduced `(x, z, angle)` replay failed the frozen round-trip gate.
- Therefore, candidate intersection and SAP-versus-TP semantic recognizability are not established.

## Minimum work before the first semantic landmark fit

1. A qualified anatomical reviewer records identity/qualification and accepts or corrects each
   boundary, with uncertainty and corrected-geometry checksum.
2. After separate approval, collect a small targeted set that records the full image-plane mapping
   (or exact per-pixel volume coordinates) for positive, nearby ambiguous, and independent negative
   views of the accepted subset.
3. Review the ultrasound correspondence and promote only supported rows from `AMBIGUOUS` to
   `VERIFIED_POSITIVE` or `VERIFIED_NEGATIVE`.
4. Freeze session-disjoint train/calibration/development splits; review sessions remain excluded
   from any unseen-confirmation claim.
5. Run the prepared CPU ExtraTrees baseline. A new sealed capture is still required for independent
   confirmation after model selection.

If only TP boundaries are accepted, a TP-only feasibility pilot is defensible; SAP-dependent
two-site L4-L5 MBB target-region localization remains blocked. No target region or Stage 3 action is
authorized by this package.
"""
    (package_root / "TRAINING_READINESS_REPORT.md").write_text(readiness, encoding="utf-8")
    readme = f"""# s0010 RIGHT SAP/TP anatomical review v2

This package contains four high-resolution CT/mesh review sheets, four sheets made from real
model-based `conv` SonoGym ultrasound captures, editable candidate vertex masks, a rapid contact
sheet, provenance, and a prepared-but-not-run CPU training contract.

Important: all four regions remain review candidates. No candidate is approved for semantic
training, no MBB target region is defined, and no file is actor-usable. A colored SAP/TP overlay is
shown only when pose-derived image coordinates pass the stored raw-label round-trip check. Green
whole-vertebra contours are evaluator-only context, not SAP/TP masks. Plane intersection and a
visible bone interface do not prove that SAP or TP can be distinguished from ultrasound pixels.
The existing captures store only a reduced `(surface_x, surface_z, in_plane_angle)` pose; their
pose-derived CT planes failed the raw-label round-trip gate, so no candidate-specific ultrasound
projection is shown in this package.

## Review order

1. Open `contact_sheet.png`.
2. Inspect each file in `anatomical_sheets/` at full resolution.
3. Inspect the matching file in `ultrasound_sheets/`.
4. Edit the NPZ/CSV/PLY geometry under `editable_review_material/editable_geometry/` if needed.
5. Record reviewer identity, qualification, decision, uncertainty and corrected-geometry hash in
   `review_response_template.json`.

Current outcome: `{manifest['overall_status']}`.
"""
    (package_root / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["RIGHT"], required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--annotation-source", type=Path, required=True)
    parser.add_argument("--expanded-protocol", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    args.asset_root = args.asset_root.resolve()
    args.annotation_source = args.annotation_source.resolve()
    args.expanded_protocol = args.expanded_protocol.resolve()
    args.protocol = args.protocol.resolve()
    args.package_root = args.package_root.resolve()
    result = build_package(args)
    print(json.dumps({"status": result["overall_status"], "records": len(result["records"])}))


if __name__ == "__main__":
    main()
