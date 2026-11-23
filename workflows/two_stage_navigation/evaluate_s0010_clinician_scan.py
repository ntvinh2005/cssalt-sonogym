#!/usr/bin/env python3
"""Evaluate executed L4/L5 scan motions and weak visible-interface transitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import yaml

from two_stage_navigation.contracts import ultrasound_sha256
from two_stage_navigation.sonoanatomy_labels import (
    OTHER_BONE_INTERFACE,
    SAP_INTERFACE,
    TP_INTERFACE,
    right_level_masks,
    visible_interface_labels,
)


MINIMUM_INTERFACE_PIXELS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bone_source_ids(path: Path) -> set[int]:
    mapping = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {int(source) for source, acoustic in mapping.items() if int(acoustic) == 13}


def transition_evidence(tp_pixels: list[int]) -> dict[str, Any]:
    if len(tp_pixels) != 8:
        raise ValueError("a cranial-caudal sequence must contain eight frames")
    visible = [value >= MINIMUM_INTERFACE_PIXELS for value in tp_pixels]
    disappearance_index = next(
        (
            index
            for index in range(1, 5)
            if any(visible[:index]) and not visible[index]
        ),
        None,
    )
    reappeared = bool(
        disappearance_index is not None and any(visible[5:])
    )
    return {
        "forward_tp_visible_count": int(sum(visible[:5])),
        "reverse_tp_visible_count": int(sum(visible[5:])),
        "disappearance_frame_in_trajectory": disappearance_index,
        "reappeared_during_reverse": reappeared,
        "passed": bool(disappearance_index is not None and reappeared),
    }


def overlay_figure(
    path: Path,
    image: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 5), constrained_layout=True)
    for axis in axes:
        axis.imshow(image, cmap="gray", origin="upper")
        axis.axis("off")
    axes[0].set_title("clean ultrasound")
    colors = {
        int(TP_INTERFACE): ("TP", "#00b4d8"),
        int(SAP_INTERFACE): ("SAP", "#f72585"),
        int(OTHER_BONE_INTERFACE): ("other bone", "#ffd166"),
    }
    for value, (name, color) in colors.items():
        rows, columns = np.where(labels == value)
        axes[1].scatter(columns, rows, s=10, c=color, label=name)
    axes[1].set_title("evaluator-only weak labels")
    axes[1].legend(loc="upper right", fontsize=7)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    figures = args.output_dir / "figures"
    figures.mkdir()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "sonogym-s0010-clinician-scan-feasibility-v1":
        raise ValueError("unexpected protocol")
    prediction_image = nib.load(str(args.prediction))
    segmentation_image = nib.load(str(args.segmentation))
    if prediction_image.shape != segmentation_image.shape or not np.allclose(
        prediction_image.affine, segmentation_image.affine
    ):
        raise ValueError("prediction and segmentation grids differ")
    prediction = np.asarray(prediction_image.dataobj, dtype=np.uint8)
    segmentation = np.asarray(segmentation_image.dataobj, dtype=np.uint8)
    masks = {
        level: right_level_masks(
            prediction, segmentation, prediction_image.affine, level
        )
        for level in ("L4", "L5")
    }
    bone_ids = bone_source_ids(args.label_conversion)
    session_specs = {str(row["session_id"]): row for row in protocol["sessions"]}
    frame_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    motion_passes = []

    for session_id, session_spec in session_specs.items():
        session = args.run_root / "sessions" / session_id
        summary = json.loads((session / "summary.json").read_text(encoding="utf-8"))
        verification = json.loads(
            (session / "verification_report.json").read_text(encoding="utf-8")
        )
        actor_validation = json.loads(
            (session / "actor_6dof_validation.json").read_text(encoding="utf-8")
        )
        public = read_jsonl(session / "public/frames.jsonl")
        privileged = read_jsonl(session / "privileged/annotations.jsonl")
        images = np.load(session / "public/ultrasound.npz")["ultrasound"]
        coordinates = np.load(
            session / "privileged/pixel_to_volume_coordinates.npz"
        )["human_img_coords"]
        raw_labels = np.load(session / "privileged/raw_labels.npz")["raw_labels"]
        expected = int(session_spec["expected_frame_count"])
        if len({len(public), len(privileged), len(images), len(coordinates), len(raw_labels)}) != 1:
            raise ValueError(f"session arrays differ: {session_id}")
        if len(public) != expected:
            raise ValueError(f"session frame count differs: {session_id}")
        verification_passed = bool(
            verification.get("gates")
            and all(bool(value) for value in verification["gates"].values())
        )
        motion_passes.append(
            bool(
                summary.get("status") == "PASSED"
                and verification_passed
                and bool(actor_validation.get("passed"))
            )
        )
        level = str(session_spec["vertebral_level"])
        bank_id = str(session_spec["absolute_waypoint_bank_id"])
        planned = protocol["sampling"]["planned_waypoint_evidence_by_bank"][bank_id]
        current_rows = []
        for index, (frame, annotation, image, coordinate_map, raw, plan) in enumerate(
            zip(public, privileged, images, coordinates, raw_labels, planned)
        ):
            image = np.asarray(image, dtype=np.float32)
            if ultrasound_sha256(image) != str(frame["frame_sha256"]):
                raise ValueError(f"image hash mismatch: {session_id}:{index}")
            result = visible_interface_labels(
                image=image,
                raw_labels=raw,
                coordinate_map_index=coordinate_map,
                masks=masks[level],
                bone_source_ids=bone_ids,
                maximum_track_distance_pixels=3.0,
            )
            tp_pixels = int(np.count_nonzero(result.labels == TP_INTERFACE))
            sap_pixels = int(np.count_nonzero(result.labels == SAP_INTERFACE))
            other_pixels = int(np.count_nonzero(result.labels == OTHER_BONE_INTERFACE))
            requested_pose = np.asarray(
                annotation["requested_waypoint_pose_index"], dtype=np.float64
            )
            realized_pose = np.asarray(annotation["realized_pose_index"], dtype=np.float64)
            tracking_position_error_mm = float(
                np.linalg.norm(realized_pose[:2] - requested_pose[:2]) * 1.5
            )
            tracking_orientation_error_deg = math.degrees(
                abs(
                    float(
                        (realized_pose[2] - requested_pose[2] + math.pi)
                        % (2.0 * math.pi)
                        - math.pi
                    )
                )
            )
            row = {
                "session_id": session_id,
                "level": level,
                "frame_sequence": int(frame["frame_sequence"]),
                "trajectory_id": str(frame["trajectory_id"]),
                "trajectory_frame_sequence": int(frame["trajectory_frame_sequence"]),
                "scan_kind": str(plan["declared_role"]),
                "frame_sha256": str(frame["frame_sha256"]),
                "tp_interface_pixels": tp_pixels,
                "sap_interface_pixels": sap_pixels,
                "other_bone_interface_pixels": other_pixels,
                "tp_supported": tp_pixels >= MINIMUM_INTERFACE_PIXELS,
                "sap_supported": sap_pixels >= MINIMUM_INTERFACE_PIXELS,
                "both_supported": tp_pixels >= MINIMUM_INTERFACE_PIXELS
                and sap_pixels >= MINIMUM_INTERFACE_PIXELS,
                "requested_pose_index": json.dumps(
                    annotation["requested_waypoint_pose_index"]
                ),
                "realized_pose_index": json.dumps(annotation["realized_pose_index"]),
                "tracking_position_error_mm": tracking_position_error_mm,
                "tracking_orientation_error_deg": tracking_orientation_error_deg,
                "image_valid": bool(frame["image_valid"]),
                "action_accepted": bool(frame["action_accepted"]),
                "action_position_clamped": bool(frame["action_position_clamped"]),
                "action_angle_clamped": bool(frame["action_angle_clamped"]),
            }
            frame_rows.append(row)
            current_rows.append(row)
            if index in (0, 7, 8, 12, 15, 16, 21, 23):
                overlay_figure(
                    figures / f"{session_id}_frame_{index:02d}.png",
                    image,
                    result.labels,
                    title=f"{session_id} frame {index} {row['scan_kind']}",
                )

        for trajectory_index, start in enumerate((0, 8, 16)):
            rows = current_rows[start : start + 8]
            scan_kind = rows[0]["scan_kind"]
            transition = (
                transition_evidence([int(row["tp_interface_pixels"]) for row in rows])
                if trajectory_index == 1
                else {}
            )
            trajectory_rows.append(
                {
                    "session_id": session_id,
                    "level": level,
                    "scan_kind": scan_kind,
                    "tp_supported_frames": sum(bool(row["tp_supported"]) for row in rows),
                    "sap_supported_frames": sum(bool(row["sap_supported"]) for row in rows),
                    "both_supported_frames": sum(bool(row["both_supported"]) for row in rows),
                    **transition,
                }
            )

    level_results = {}
    for level in ("L4", "L5"):
        level_frames = [row for row in frame_rows if row["level"] == level]
        level_trajectories = [row for row in trajectory_rows if row["level"] == level]
        session_ids = sorted({row["session_id"] for row in level_frames})
        session_sap_counts = {
            session_id: sum(
                bool(row["sap_supported"])
                for row in level_frames
                if row["session_id"] == session_id
            )
            for session_id in session_ids
        }
        long_pass = any(
            row["scan_kind"] == "LONGITUDINAL_LATERAL_SWEEP"
            and int(row["tp_supported_frames"]) >= 2
            and int(row["sap_supported_frames"]) >= 2
            for row in level_trajectories
        )
        transverse_pass = any(
            row["scan_kind"] == "TRANSVERSE_CRANIAL_CAUDAL_SWEEP"
            and int(row["both_supported_frames"]) >= 2
            for row in level_trajectories
        )
        transition_pass = any(
            row["scan_kind"] == "TRANSVERSE_CRANIAL_CAUDAL_SWEEP"
            and bool(row.get("passed"))
            for row in level_trajectories
        )
        level_results[level] = {
            "frame_count": len(level_frames),
            "session_count": len(session_ids),
            "tp_supported_frame_count": sum(bool(row["tp_supported"]) for row in level_frames),
            "sap_supported_frame_count": sum(bool(row["sap_supported"]) for row in level_frames),
            "both_supported_frame_count": sum(bool(row["both_supported"]) for row in level_frames),
            "session_sap_supported_frame_counts": session_sap_counts,
            "longitudinal_scanning_passed": long_pass,
            "transverse_scanning_passed": transverse_pass,
            "cranial_caudal_transition_passed": transition_pass,
            "sap_observability_passed": all(value >= 2 for value in session_sap_counts.values()),
        }

    write_csv(args.output_dir / "frame_results.csv", frame_rows)
    write_csv(args.output_dir / "trajectory_results.csv", trajectory_rows)
    result = {
        "protocol_sha256": sha256(args.protocol),
        "capture_root": str(args.run_root.resolve()),
        "capture_job_id": args.run_root.name,
        "frame_count": len(frame_rows),
        "session_count": len(session_specs),
        "minimum_interface_pixels": MINIMUM_INTERFACE_PIXELS,
        "motion_feasibility_passed": bool(all(motion_passes) and len(frame_rows) == 96),
        "levels": level_results,
        "anatomical_label_status": "SIMULATOR_WEAK_ANATOMICAL_LABEL",
        "independent_confirmation": False,
    }
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
