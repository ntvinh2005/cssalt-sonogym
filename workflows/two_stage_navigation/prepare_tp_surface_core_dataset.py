#!/usr/bin/env python3
"""Prepare auditable RIGHT L4/L5 TP surface-core review samples from verified captures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

GEOMETRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/ultrasound_plane_geometry.py"
)
GEOMETRY_SPEC = importlib.util.spec_from_file_location("tp_surface_core_geometry", GEOMETRY_PATH)
if GEOMETRY_SPEC is None or GEOMETRY_SPEC.loader is None:
    raise RuntimeError("cannot load CPU ultrasound-plane geometry")
GEOMETRY = importlib.util.module_from_spec(GEOMETRY_SPEC)
sys.modules[GEOMETRY_SPEC.name] = GEOMETRY
GEOMETRY_SPEC.loader.exec_module(GEOMETRY)
plane_from_full_pose = GEOMETRY.plane_from_full_pose
project_points_to_pixels = GEOMETRY.project_points_to_pixels


ACTIVE_CANDIDATES = (
    "L4_RIGHT_transverse_process",
    "L5_RIGHT_transverse_process",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def locate_geometry(root: Path, candidate: str) -> Path:
    matches = list(root.rglob(f"{candidate}_revision.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one editable geometry for {candidate}, found {len(matches)}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != "sonogym-s0010-tp-surface-core-dataset-v1":
        raise RuntimeError("unexpected protocol")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    geometries: dict[str, dict[str, Any]] = {}
    for candidate in ACTIVE_CANDIDATES:
        path = locate_geometry(args.geometry_root.resolve(), candidate)
        archive = np.load(path)
        vertices = archive["vertices_mesh_index"].astype(np.float64)
        core = archive["proposed_revised_vertex_mask"].astype(bool)
        uncertain = archive["boundary_uncertainty_vertex_mask"].astype(bool)
        if not core.any():
            raise RuntimeError(f"{candidate} has no proposed surface-core geometry")
        geometries[candidate] = {
            "path": path,
            "sha256": sha256(path),
            "vertices": vertices,
            "core": core,
            "uncertain": uncertain,
        }

    rows: list[dict[str, Any]] = []
    missing_sessions: list[str] = []
    processed_sessions: list[str] = []
    overlay_budget = {candidate: 12 for candidate in ACTIVE_CANDIDATES}
    for session in protocol["sessions"]:
        session_dir = args.capture_root / "sessions" / session["session_id"]
        if not session_dir.is_dir():
            missing_sessions.append(session["session_id"])
            continue
        processed_sessions.append(session["session_id"])
        report = json.loads((session_dir / "verification_report.json").read_text())
        if report["verdict"] != "PROJECTION_PARITY_PASS":
            raise RuntimeError(f"unverified session: {session['session_id']}")
        public = read_jsonl(session_dir / "public/frames.jsonl")
        poses = read_jsonl(session_dir / "privileged/full_pose_replay.jsonl")
        images = np.load(session_dir / "public/ultrasound.npz")["ultrasound"]
        if not (len(public) == len(poses) == len(images)):
            raise RuntimeError("capture records are misaligned")
        for index, (actor, pose) in enumerate(zip(public, poses)):
            if actor["acquisition_id"] != pose["acquisition_id"]:
                raise RuntimeError("acquisition identity mismatch")
            local = np.load(session_dir / "privileged/pixel_to_volume_coordinates.npz")[
                "native_local_image_coordinates_m"
            ]
            plane = plane_from_full_pose(
                end_effector_position_m=np.asarray(pose["human_to_ee_position_m"]),
                rotation_columns=np.asarray(pose["human_to_ee_rotation_columns"]),
                image_origin_height_m=float(pose["image_origin_height_m"]),
                local_coordinates_m=local,
                label_resolution_m=float(pose["label_resolution_m"]),
            )
            for class_id, candidate in enumerate(ACTIVE_CANDIDATES):
                geometry = geometries[candidate]
                projected = project_points_to_pixels(
                    points_index=geometry["vertices"][geometry["core"]],
                    image_origin_index=plane.origin_index,
                    rotation_columns=plane.rotation_columns,
                    width=150,
                    depth=200,
                    image_resolution_m=float(pose["image_resolution_m"]),
                    label_resolution_m=float(pose["label_resolution_m"]),
                    plane_tolerance_mm=float(protocol["label_preparation"]["surface_slab_tolerance_mm"]),
                )
                inside = projected["inside"]
                uncertain_projection = project_points_to_pixels(
                    points_index=geometry["vertices"][geometry["uncertain"]],
                    image_origin_index=plane.origin_index,
                    rotation_columns=plane.rotation_columns,
                    width=150,
                    depth=200,
                    image_resolution_m=float(pose["image_resolution_m"]),
                    label_resolution_m=float(pose["label_resolution_m"]),
                    plane_tolerance_mm=float(protocol["label_preparation"]["surface_slab_tolerance_mm"]),
                )
                if inside.any():
                    patch_row = float(np.median(projected["rows"][inside]))
                    patch_column = float(np.median(projected["columns"][inside]))
                    intersection = "PRESENT"
                else:
                    patch_row = patch_column = float("nan")
                    intersection = "ABSENT"
                record = {
                    "sample_id": f"{session['session_id']}:{index}:{candidate}",
                    "session_id": session["session_id"],
                    "split": session["split"],
                    "frame_sequence": index,
                    "acquisition_id": actor["acquisition_id"],
                    "frame_sha256": actor["frame_sha256"],
                    "image_archive": str((session_dir / "public/ultrasound.npz").resolve()),
                    "image_archive_index": index,
                    "candidate_id": candidate,
                    "class_id": class_id,
                    "geometry_version_sha256": geometry["sha256"],
                    "projection_status": "VERIFIED",
                    "candidate_intersection": intersection,
                    "core_projected_vertex_count": int(inside.sum()),
                    "uncertain_projected_vertex_count": int(uncertain_projection["inside"].sum()),
                    "patch_row": patch_row,
                    "patch_column": patch_column,
                    "patch_width": 48,
                    "patch_depth": 80,
                    "acoustic_visibility": "NOT_REVIEWED",
                    "semantic_identity": "NOT_REVIEWED",
                    "anatomical_approval": "PENDING",
                    "training_label": "EXCLUDED",
                    "training_eligible": False,
                    "exclusion_reason": "anatomical and image-semantic review pending",
                }
                rows.append(record)
                if inside.any() and overlay_budget[candidate] > 0:
                    overlay_budget[candidate] -= 1
                    figure_path = output / "review" / f"{record['sample_id'].replace(':', '_')}.png"
                    figure_path.parent.mkdir(parents=True, exist_ok=True)
                    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=160)
                    axes[0].imshow(images[index], cmap="gray")
                    axes[0].set_title("Clean actor-visible ultrasound")
                    axes[1].imshow(images[index], cmap="gray")
                    axes[1].scatter(projected["columns"][inside], projected["rows"][inside], s=4, c="orange")
                    axes[1].scatter(
                        uncertain_projection["columns"][uncertain_projection["inside"]],
                        uncertain_projection["rows"][uncertain_projection["inside"]],
                        s=3,
                        c="cyan",
                    )
                    axes[1].set_title("Reviewer-only core/uncertain projection")
                    for axis in axes:
                        axis.axis("off")
                    fig.suptitle(f"{candidate} | {session['session_id']} frame {index}")
                    fig.tight_layout()
                    fig.savefig(figure_path)
                    plt.close(fig)

    write_csv(output / "candidate_sample_manifest.csv", rows)
    split_rows = [
        {
            "session_id": session["session_id"],
            "role": session["role"],
            "split": session["split"],
            "prior_inspection": "new_opened_development_capture",
        }
        for session in protocol["sessions"]
        if session["role"] != "smoke"
    ]
    write_csv(output / "session_split_manifest.csv", split_rows)
    summary = {
        "schema": "sonogym-tp-surface-core-dataset-preparation-v1",
        "active_classes": list(ACTIVE_CANDIDATES),
        "task_name": "TP_SURFACE_CORE detection",
        "rows": len(rows),
        "verified_intersections": sum(row["candidate_intersection"] == "PRESENT" for row in rows),
        "training_eligible_rows": 0,
        "anatomical_approval": "PENDING",
        "processed_sessions": processed_sessions,
        "missing_sessions": missing_sessions,
        "status": (
            "DATASET_INCOMPLETE"
            if missing_sessions
            else "ENGINEERING_READY_LABEL_REVIEW_PENDING"
        ),
        "counts": {
            candidate: {
                split: sum(
                    row["candidate_id"] == candidate
                    and row["split"] == split
                    and row["candidate_intersection"] == "PRESENT"
                    for row in rows
                )
                for split in ("train", "calibration", "development_evaluation")
            }
            for candidate in ACTIVE_CANDIDATES
        },
    }
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "label_schema.json").write_text(
        json.dumps(
            {
                "task": "TP_SURFACE_CORE detection",
                "classes": list(ACTIVE_CANDIDATES),
                "excluded": ["SAP", "whole_TP", "MBB_target"],
                "supervision": "48x80 interface-centred patch classification",
                "feature_mode": "16x16 normalized intensity and gradients",
                "inference_proposals": "actor-visible bright bone-interface candidates",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
