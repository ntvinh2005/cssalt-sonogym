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
from two_stage_navigation.stage1_interfaces import track_bright_interfaces
from two_stage_navigation.ultrasound_surface_features import normalized_image


MINIMUM_INTERFACE_PIXELS = 3
LOW_INTENSITY_LIMIT = 0.20
SHADOW_DEPTH_PIXELS = 42
MINIMUM_SHADOW_CHANGE = 0.05
MINIMUM_DEEP_VISIBILITY_CHANGE = 0.015


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


def image_transition_measurements(image: np.ndarray) -> dict[str, float | int | None]:
    normalized = normalized_image(image)
    tracks = track_bright_interfaces(normalized)
    deep_start = int(round(0.72 * normalized.shape[0]))
    result: dict[str, float | int | None] = {
        "interface_track_count": len(tracks),
        "deep_window_visibility": float(np.mean(normalized[deep_start:])),
        "cortical_interface_median_row_fraction": None,
        "posterior_shadow_fraction": None,
    }
    if not tracks:
        return result
    track = tracks[0]
    columns = np.arange(int(track.columns[0]), int(track.columns[-1]) + 1)
    rows = np.rint([track.row_at(float(column)) for column in columns]).astype(int)
    shadow_values = []
    for row, column in zip(rows, columns):
        distal = normalized[
            min(normalized.shape[0], row + 3) : min(
                normalized.shape[0], row + 3 + SHADOW_DEPTH_PIXELS
            ),
            column,
        ]
        shadow_values.extend(distal.tolist())
    result["cortical_interface_median_row_fraction"] = float(
        np.median(rows) / normalized.shape[0]
    )
    result["posterior_shadow_fraction"] = (
        float(np.mean(np.asarray(shadow_values) < LOW_INTENSITY_LIMIT))
        if shadow_values
        else None
    )
    return result


def transition_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 8:
        raise ValueError("a cranial-caudal sequence must contain eight frames")
    tp_pixels = [int(row["tp_interface_pixels"]) for row in rows]
    visible = [value >= MINIMUM_INTERFACE_PIXELS for value in tp_pixels]
    disappearance_index = next(
        (
            index
            for index in range(1, 5)
            if any(visible[:index]) and not visible[index]
        ),
        None,
    )
    reverse_visible_index = next(
        (index for index in range(5, 8) if visible[index]), None
    )
    reappeared = bool(disappearance_index is not None and reverse_visible_index is not None)
    realized = np.asarray([row["realized_pose_index"] for row in rows], dtype=np.float64)
    if realized.shape != (8, 3) or not np.all(np.isfinite(realized)):
        raise ValueError("realized poses must be finite eight-by-three values")
    forward_cranial = bool(np.all(np.diff(realized[:5, 1]) > 0.0))
    reverse_caudal = bool(np.all(np.diff(realized[4:, 1]) < 0.0))
    acquisition_ids = [str(row["acquisition_id"]) for row in rows]
    frame_hashes = [str(row["frame_sha256"]) for row in rows]
    producer_sequences = [int(row["producer_sequence"]) for row in rows]
    freshness_passed = bool(
        len(set(acquisition_ids)) == 8
        and len(set(frame_hashes)) == 8
        and all(
            right > left
            for left, right in zip(producer_sequences, producer_sequences[1:])
        )
    )
    technical_validity_passed = bool(
        freshness_passed
        and all(bool(row["contact_ok"]) for row in rows)
        and all(bool(row["image_valid"]) for row in rows)
        and all(bool(row["action_accepted"]) for row in rows)
    )
    actor_morphology_passed = False
    reference_index = None
    shadow_change = None
    shadow_recovery = None
    deep_visibility_change = None
    deep_visibility_recovery = None
    if disappearance_index is not None and reverse_visible_index is not None:
        reference_index = max(
            index for index in range(disappearance_index) if visible[index]
        )
        shadow = [row["posterior_shadow_fraction"] for row in rows]
        deep = [float(row["deep_window_visibility"]) for row in rows]
        required_shadow = (
            shadow[reference_index],
            shadow[disappearance_index],
            shadow[reverse_visible_index],
        )
        if all(value is not None for value in required_shadow):
            shadow_change = float(shadow[disappearance_index] - shadow[reference_index])
            shadow_recovery = float(
                shadow[disappearance_index] - shadow[reverse_visible_index]
            )
            deep_visibility_change = float(
                deep[reference_index] - deep[disappearance_index]
            )
            deep_visibility_recovery = float(
                deep[reverse_visible_index] - deep[disappearance_index]
            )
            actor_morphology_passed = bool(
                shadow_change >= MINIMUM_SHADOW_CHANGE
                and shadow_recovery >= MINIMUM_SHADOW_CHANGE
                and deep_visibility_change >= MINIMUM_DEEP_VISIBILITY_CHANGE
                and deep_visibility_recovery >= MINIMUM_DEEP_VISIBILITY_CHANGE
            )
    privileged_tp_pattern_passed = bool(disappearance_index is not None and reappeared)
    return {
        "forward_tp_visible_count": int(sum(visible[:5])),
        "reverse_tp_visible_count": int(sum(visible[5:])),
        "disappearance_frame_in_trajectory": disappearance_index,
        "reappeared_during_reverse": reappeared,
        "reverse_visible_frame_in_trajectory": reverse_visible_index,
        "reference_visible_frame_in_trajectory": reference_index,
        "forward_cranial_motion_passed": forward_cranial,
        "reverse_caudal_motion_passed": reverse_caudal,
        "freshness_passed": freshness_passed,
        "technical_validity_passed": technical_validity_passed,
        "privileged_tp_pattern_passed": privileged_tp_pattern_passed,
        "shadow_change_at_disappearance": shadow_change,
        "shadow_recovery_during_reverse": shadow_recovery,
        "deep_visibility_change_at_disappearance": deep_visibility_change,
        "deep_visibility_recovery_during_reverse": deep_visibility_recovery,
        "actor_morphology_passed": actor_morphology_passed,
        "passed": bool(
            privileged_tp_pattern_passed
            and actor_morphology_passed
            and forward_cranial
            and reverse_caudal
            and technical_validity_passed
        ),
    }


def longitudinal_transition_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the v2 lateral-loss/reverse-return/medial-region sequence."""

    if len(rows) != 8:
        raise ValueError("a reversible longitudinal sequence must contain eight frames")
    realized = np.asarray([row["realized_pose_index"] for row in rows], dtype=np.float64)
    if realized.shape != (8, 3) or not np.all(np.isfinite(realized)):
        raise ValueError("realized poses must be finite eight-by-three values")
    outward_lateral = bool(np.all(np.diff(realized[:3, 0]) > 0.0))
    parallel_turn = bool(
        abs(float(realized[3, 0] - realized[2, 0])) < 1.0
        and abs(float(realized[3, 1] - realized[2, 1])) > 1.0
    )
    return_medial = bool(np.all(np.diff(realized[3:, 0]) < 0.0))
    acquisition_ids = [str(row["acquisition_id"]) for row in rows]
    frame_hashes = [str(row["frame_sha256"]) for row in rows]
    producer_sequences = [int(row["producer_sequence"]) for row in rows]
    freshness_passed = bool(
        len(set(acquisition_ids)) == 8
        and len(set(frame_hashes)) == 8
        and all(
            right > left
            for left, right in zip(producer_sequences, producer_sequences[1:])
        )
    )
    technical_validity_passed = bool(
        freshness_passed
        and all(bool(row["contact_ok"]) for row in rows)
        and all(bool(row["image_valid"]) for row in rows)
        and all(bool(row["action_accepted"]) for row in rows)
    )
    tp_visible = [
        int(row["tp_interface_pixels"]) >= MINIMUM_INTERFACE_PIXELS for row in rows
    ]
    sap_visible = [
        int(row["sap_interface_pixels"]) >= MINIMUM_INTERFACE_PIXELS for row in rows
    ]
    tp_start = bool(tp_visible[0])
    lateral_loss = bool(not tp_visible[2] and not tp_visible[3])
    reverse_return = bool(tp_visible[4] or tp_visible[5])
    ap_index = next((index for index in (6, 7) if sap_visible[index]), None)
    medial_ap_candidate = ap_index is not None

    def morphology_change(left: int, right: int) -> tuple[float | None, float, bool]:
        left_shadow = rows[left]["posterior_shadow_fraction"]
        right_shadow = rows[right]["posterior_shadow_fraction"]
        shadow_delta = (
            None
            if left_shadow is None or right_shadow is None
            else float(right_shadow - left_shadow)
        )
        deep_delta = float(
            rows[right]["deep_window_visibility"]
            - rows[left]["deep_window_visibility"]
        )
        passed = bool(
            (shadow_delta is not None and abs(shadow_delta) >= MINIMUM_SHADOW_CHANGE)
            or abs(deep_delta) >= MINIMUM_DEEP_VISIBILITY_CHANGE
        )
        return shadow_delta, deep_delta, passed

    loss_shadow_delta, loss_deep_delta, loss_morphology = morphology_change(0, 2)
    medial_shadow_delta = None
    medial_deep_delta = None
    medial_morphology = False
    if ap_index is not None:
        medial_shadow_delta, medial_deep_delta, medial_morphology = morphology_change(
            5, ap_index
        )
    privileged_pattern_passed = bool(
        tp_start and lateral_loss and reverse_return and medial_ap_candidate
    )
    return {
        "longitudinal_outward_lateral_motion_passed": outward_lateral,
        "longitudinal_parallel_turn_passed": parallel_turn,
        "longitudinal_return_medial_motion_passed": return_medial,
        "longitudinal_freshness_passed": freshness_passed,
        "longitudinal_technical_validity_passed": technical_validity_passed,
        "longitudinal_tp_start_supported": tp_start,
        "longitudinal_lateral_tp_loss_supported": lateral_loss,
        "longitudinal_reverse_tp_return_supported": reverse_return,
        "longitudinal_medial_ap_candidate_supported": medial_ap_candidate,
        "longitudinal_ap_candidate_frame_in_trajectory": ap_index,
        "longitudinal_privileged_pattern_passed": privileged_pattern_passed,
        "longitudinal_loss_shadow_delta": loss_shadow_delta,
        "longitudinal_loss_deep_visibility_delta": loss_deep_delta,
        "longitudinal_loss_morphology_passed": loss_morphology,
        "longitudinal_medial_shadow_delta": medial_shadow_delta,
        "longitudinal_medial_deep_visibility_delta": medial_deep_delta,
        "longitudinal_medial_morphology_passed": medial_morphology,
        "longitudinal_transition_passed": bool(
            outward_lateral
            and parallel_turn
            and return_medial
            and technical_validity_passed
            and privileged_pattern_passed
            and loss_morphology
            and medial_morphology
        ),
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
    if protocol.get("protocol_id") not in {
        "sonogym-s0010-clinician-scan-feasibility-v1",
        "sonogym-s0010-clinician-scan-feasibility-v2",
    }:
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
    evaluator_label_images: list[np.ndarray] = []
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
            image_measurements = image_transition_measurements(image)
            row = {
                "session_id": session_id,
                "level": level,
                "frame_sequence": int(frame["frame_sequence"]),
                "trajectory_id": str(frame["trajectory_id"]),
                "trajectory_frame_sequence": int(frame["trajectory_frame_sequence"]),
                "scan_kind": str(plan["declared_role"]),
                "frame_sha256": str(frame["frame_sha256"]),
                "acquisition_id": str(frame["acquisition_id"]),
                "producer_sequence": int(frame["producer_sequence"]),
                "tp_interface_pixels": tp_pixels,
                "sap_interface_pixels": sap_pixels,
                "other_bone_interface_pixels": other_pixels,
                "tp_supported": tp_pixels >= MINIMUM_INTERFACE_PIXELS,
                "sap_supported": sap_pixels >= MINIMUM_INTERFACE_PIXELS,
                "both_supported": tp_pixels >= MINIMUM_INTERFACE_PIXELS
                and sap_pixels >= MINIMUM_INTERFACE_PIXELS,
                "requested_pose_index": annotation["requested_waypoint_pose_index"],
                "realized_pose_index": annotation["realized_pose_index"],
                "tracking_position_error_mm": tracking_position_error_mm,
                "tracking_orientation_error_deg": tracking_orientation_error_deg,
                "image_valid": bool(frame["image_valid"]),
                "contact_ok": bool(frame["contact_ok"]),
                "action_accepted": bool(frame["action_accepted"]),
                "action_position_clamped": bool(frame["action_position_clamped"]),
                "action_angle_clamped": bool(frame["action_angle_clamped"]),
                **image_measurements,
            }
            frame_rows.append(row)
            current_rows.append(row)
            evaluator_label_images.append(result.labels.astype(np.uint8, copy=True))
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
            transition = {}
            if trajectory_index == 1:
                transition = transition_evidence(rows)
            elif scan_kind == "LONGITUDINAL_MEDIAL_LATERAL_REVERSIBLE_SWEEP":
                transition = longitudinal_transition_evidence(rows)
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
            (
                row["scan_kind"] == "LONGITUDINAL_LATERAL_SWEEP"
                and int(row["tp_supported_frames"]) >= 2
                and int(row["sap_supported_frames"]) >= 2
            )
            or (
                row["scan_kind"]
                == "LONGITUDINAL_MEDIAL_LATERAL_REVERSIBLE_SWEEP"
                and bool(row.get("longitudinal_transition_passed"))
            )
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
    np.savez_compressed(
        args.output_dir / "weak_interface_labels.npz",
        labels=np.stack(evaluator_label_images),
    )
    result = {
        "protocol_sha256": sha256(args.protocol),
        "capture_root": str(args.run_root.resolve()),
        "capture_job_id": args.run_root.name,
        "frame_count": len(frame_rows),
        "weak_interface_label_archive": "weak_interface_labels.npz",
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
