#!/usr/bin/env python3
"""Mine geometry-supported RIGHT L5 SAP/TP viewpoints from archived captures.

This is an evaluator-only planner. Its absolute waypoints may seed a frozen
capture protocol, but none of its CT or native-coordinate fields are runtime
inputs to a later ultrasound localizer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage

import correct_s0010_sap_observability_v1 as sap_correction
import run_s0010_l4_medial_branch_target as target_workflow


MINIMUM_TRANSLATION_SEPARATION_INDEX = 3.0
MINIMUM_ANGLE_SEPARATION_RAD = np.deg2rad(2.4)
TARGET_ANCHOR_COUNT = 6
CONTROL_ANCHOR_COUNT = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def normalized_gradient(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    lo, hi = np.percentile(array, (0.5, 99.5))
    normalized = np.clip((array - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return np.hypot(ndimage.sobel(normalized, axis=0), ndimage.sobel(normalized, axis=1))


def physically_distinct(a: np.ndarray, b: np.ndarray) -> bool:
    translation = float(np.linalg.norm(a[:2] - b[:2]))
    angle = abs(float(np.arctan2(np.sin(a[2] - b[2]), np.cos(a[2] - b[2]))))
    return bool(
        translation > MINIMUM_TRANSLATION_SEPARATION_INDEX
        or angle > MINIMUM_ANGLE_SEPARATION_RAD
    )


def select_diverse(
    rows: list[dict[str, Any]], count: int, *, ranking_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: tuple(-float(row[field]) for field in ranking_fields)
        + (str(row["session_id"]), int(row["frame_sequence"])),
    )
    selected: list[dict[str, Any]] = []
    for row in ordered:
        pose = np.asarray(row["requested_waypoint_pose_index"], dtype=np.float64)
        if all(
            physically_distinct(
                pose,
                np.asarray(existing["requested_waypoint_pose_index"], dtype=np.float64),
            )
            for existing in selected
        ):
            selected.append(row)
        if len(selected) == count:
            break
    return selected


def session_directories(roots: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for candidate in sorted(root.iterdir()):
            resolved = candidate.resolve()
            required = (
                resolved / "public/frames.jsonl",
                resolved / "public/ultrasound.npz",
                resolved / "privileged/annotations.jsonl",
                resolved / "privileged/raw_labels.npz",
                resolved / "privileged/pixel_to_volume_coordinates.npz",
            )
            if resolved not in seen and all(path.is_file() for path in required):
                result.append(resolved)
                seen.add(resolved)
    return result


def visible_score(
    gradient: np.ndarray, projected: dict[str, np.ndarray], first_bone: np.ndarray
) -> tuple[int, int, float]:
    sap = projected["sap_core"] & first_bone
    tp = projected["tp_core"] & first_bone
    pixels = np.concatenate((gradient[sap], gradient[tp]))
    return (
        int(np.count_nonzero(sap)),
        int(np.count_nonzero(tp)),
        float(np.median(pixels)) if len(pixels) else 0.0,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    serial = [
        {
            key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
            for key, value in row.items()
        }
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serial[0]))
        writer.writeheader()
        writer.writerows(serial)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, action="append", required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--tp-revision", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    prediction_image, sap_mask, tp_mask, contract = target_workflow.load_target_geometry(
        prediction_path=args.prediction,
        segmentation_path=args.segmentation,
    )
    bone_ids = sap_correction.bone_source_ids(args.label_conversion)
    rows: list[dict[str, Any]] = []
    sessions = session_directories(args.reference_root)
    for session_dir in sessions:
        frames = read_jsonl(session_dir / "public/frames.jsonl")
        annotations = read_jsonl(session_dir / "privileged/annotations.jsonl")
        images = np.load(session_dir / "public/ultrasound.npz")["ultrasound"]
        coordinates = np.load(
            session_dir / "privileged/pixel_to_volume_coordinates.npz"
        )["human_img_coords"]
        raw_labels = np.load(session_dir / "privileged/raw_labels.npz")["raw_labels"]
        if not (len(frames) == len(annotations) == len(images) == len(coordinates) == len(raw_labels)):
            raise ValueError(f"session arrays differ in length: {session_dir}")
        for index, (frame, annotation, image, coordinate_map, raw) in enumerate(
            zip(frames, annotations, images, coordinates, raw_labels)
        ):
            if frame["frame_sha256"] != annotation["frame_sha256"]:
                raise ValueError(f"frame/annotation hash mismatch: {session_dir.name}:{index}")
            projected = target_workflow.projected_masks(
                coordinate_map, contract, sap_mask, tp_mask
            )
            first_bone = target_workflow.first_bone_mask(raw, bone_ids)
            label, reason = target_workflow.target_evidence_label(projected, first_bone)
            gradient = normalized_gradient(image)
            sap_count, tp_count, gradient_median = visible_score(
                gradient, projected, first_bone
            )
            all_target_pixels = np.argwhere(projected["sap_core"] | projected["tp_core"])
            if len(all_target_pixels):
                margin = int(
                    min(
                        all_target_pixels[:, 0].min(),
                        image.shape[0] - 1 - all_target_pixels[:, 0].max(),
                        all_target_pixels[:, 1].min(),
                        image.shape[1] - 1 - all_target_pixels[:, 1].max(),
                    )
                )
            else:
                margin = -1
            pose = annotation.get("requested_waypoint_pose_index")
            if pose is None:
                pose = frame.get("requested_relative_pose_index")
            if pose is None:
                raise ValueError(f"missing requested pose: {session_dir.name}:{index}")
            rows.append(
                {
                    "session_id": session_dir.name,
                    "session_path": str(session_dir),
                    "frame_sequence": index,
                    "frame_sha256": frame["frame_sha256"],
                    "requested_waypoint_pose_index": [float(value) for value in pose],
                    "evaluator_label": label,
                    "evaluator_reason": reason,
                    "sap_core_first_visible_pixels": sap_count,
                    "tp_core_first_visible_pixels": tp_count,
                    "combined_core_first_visible_pixels": sap_count + tp_count,
                    "target_core_margin_pixels": margin,
                    "surface_gradient_median": gradient_median,
                    "sap_projected_pixels": int(np.count_nonzero(projected["sap"])),
                    "tp_projected_pixels": int(np.count_nonzero(projected["tp"])),
                    "sap_core_projected_pixels": int(np.count_nonzero(projected["sap_core"])),
                    "tp_core_projected_pixels": int(np.count_nonzero(projected["tp_core"])),
                }
            )

    positives = [row for row in rows if row["evaluator_label"] == "PROXY_POSITIVE"]
    tp_only = [
        row
        for row in rows
        if row["evaluator_label"] == "PROXY_NEGATIVE"
        and row["evaluator_reason"] == "TP_ONLY_WITHOUT_SAP_IN_PLANE"
    ]
    other_controls = [
        row
        for row in rows
        if row["evaluator_label"] == "PROXY_NEGATIVE"
        and row["evaluator_reason"] == "OTHER_BONE_OR_BACKGROUND"
    ]
    ambiguous = [row for row in rows if row["evaluator_label"] == "PROXY_AMBIGUOUS"]
    near_target = [
        row
        for row in ambiguous
        if row["evaluator_reason"] == "TP_AND_SAP_VISIBLE_OUTSIDE_TARGET_CORE"
        and int(row["sap_core_projected_pixels"]) > 0
        and int(row["tp_core_projected_pixels"]) > 0
    ]
    target_pool = positives if len(positives) >= TARGET_ANCHOR_COUNT else near_target
    target_anchor_source = (
        "GEOMETRIC_POSITIVE"
        if len(positives) >= TARGET_ANCHOR_COUNT
        else "AMBIGUOUS_LOCAL_SEARCH_SEED"
    )
    targets = select_diverse(
        target_pool,
        TARGET_ANCHOR_COUNT,
        ranking_fields=(
            "combined_core_first_visible_pixels",
            "sap_core_projected_pixels",
            "tp_core_projected_pixels",
            "target_core_margin_pixels",
            "surface_gradient_median",
        ),
    )
    controls = select_diverse(
        tp_only,
        CONTROL_ANCHOR_COUNT // 2,
        ranking_fields=("tp_core_first_visible_pixels", "target_core_margin_pixels"),
    )
    controls.extend(
        select_diverse(
            other_controls,
            CONTROL_ANCHOR_COUNT - len(controls),
            ranking_fields=("target_core_margin_pixels", "surface_gradient_median"),
        )
    )
    if len(targets) == TARGET_ANCHOR_COUNT and len(controls) == CONTROL_ANCHOR_COUNT:
        status = (
            "POSE_ATLAS_READY"
            if target_anchor_source == "GEOMETRIC_POSITIVE"
            else "POSE_ATLAS_EXPLORATORY_READY"
        )
    else:
        status = "POSE_ATLAS_INSUFFICIENT"
    plan = {
        "schema": "s0010-l4mb-target-observability-pose-atlas-v1",
        "status": status,
        "information_boundary": "evaluator_only_pose_design",
        "source_session_count": len(sessions),
        "source_frame_count": len(rows),
        "candidate_counts": {
            "PROXY_POSITIVE": len(positives),
            "PROXY_AMBIGUOUS": len(ambiguous),
            "AMBIGUOUS_LOCAL_SEARCH_SEED": len(near_target),
            "TP_ONLY_WITHOUT_SAP_IN_PLANE": len(tp_only),
            "OTHER_BONE_OR_BACKGROUND": len(other_controls),
        },
        "selection": {
            "target_anchor_count": TARGET_ANCHOR_COUNT,
            "control_anchor_count": CONTROL_ANCHOR_COUNT,
            "minimum_translation_separation_index": MINIMUM_TRANSLATION_SEPARATION_INDEX,
            "minimum_angle_separation_rad": MINIMUM_ANGLE_SEPARATION_RAD,
            "target_anchor_source": target_anchor_source,
            "positive_definition_unchanged": True,
        },
        "input_hashes": {
            "prediction": sha256(args.prediction),
            "segmentation": sha256(args.segmentation),
            "tp_revision": sha256(args.tp_revision),
        },
        "target_anchors": targets,
        "control_anchors": controls,
    }
    (args.output_dir / "pose_atlas.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "all_frames.csv", rows)
    if positives:
        write_csv(args.output_dir / "positive_candidates.csv", positives)
    if ambiguous:
        write_csv(args.output_dir / "ambiguous_candidates.csv", ambiguous)
    if tp_only or other_controls:
        write_csv(args.output_dir / "negative_candidates.csv", tp_only + other_controls)
    print(json.dumps({"status": status, **plan["candidate_counts"]}, sort_keys=True))
    return 0 if status in {"POSE_ATLAS_READY", "POSE_ATLAS_EXPLORATORY_READY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
