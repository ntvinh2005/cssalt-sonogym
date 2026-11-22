#!/usr/bin/env python3
"""Evaluate fresh RIGHT L5 SAP/TP target-region observability captures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

import correct_s0010_sap_observability_v1 as sap_correction
import run_s0010_l4_medial_branch_target as target_workflow
from two_stage_navigation.stage1_interfaces import (
    InterfaceTrack,
    track_bright_interfaces,
)
from two_stage_navigation.ultrasound_surface_features import (
    normalized_image,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sessions_for_evaluation(
    protocol: dict[str, Any], requested_session_ids: list[str] | None
) -> list[dict[str, Any]]:
    sessions = list(protocol["sessions"])
    if not requested_session_ids:
        return sessions
    requested = set(requested_session_ids)
    selected = [row for row in sessions if row["session_id"] in requested]
    missing = requested - {row["session_id"] for row in selected}
    if missing:
        raise ValueError(f"requested sessions are not in the protocol: {sorted(missing)}")
    return selected


def nearest_track_distance(mask: np.ndarray, tracks: list[InterfaceTrack]) -> float | None:
    pixels = np.argwhere(np.asarray(mask, dtype=bool))
    distances: list[float] = []
    for row, column in pixels:
        for track in tracks:
            if track.columns[0] <= column <= track.columns[-1]:
                distances.append(abs(float(row) - track.row_at(float(column))))
    return float(min(distances)) if distances else None


def posterior_shadow_contrast(image: np.ndarray, mask: np.ndarray) -> float | None:
    array = normalized_image(image)
    values: list[float] = []
    for row, column in np.argwhere(np.asarray(mask, dtype=bool)):
        proximal = array[max(0, row - 2) : min(array.shape[0], row + 2), column]
        distal = array[min(array.shape[0], row + 3) : min(array.shape[0], row + 11), column]
        if len(proximal) and len(distal):
            values.append(float(np.mean(proximal) - np.mean(distal)))
    return float(np.median(values)) if values else None


def target_margin(projected: dict[str, np.ndarray]) -> int:
    pixels = np.argwhere(projected["sap_core"] | projected["tp_core"])
    if not len(pixels):
        return -1
    height, width = projected["sap_core"].shape
    return int(
        min(
            pixels[:, 0].min(),
            height - 1 - pixels[:, 0].max(),
            pixels[:, 1].min(),
            width - 1 - pixels[:, 1].max(),
        )
    )


def image_supported(
    *,
    image: np.ndarray,
    projected: dict[str, np.ndarray],
    first_bone: np.ndarray,
    maximum_track_distance: float,
    minimum_shadow: float,
    minimum_margin: int,
) -> dict[str, Any]:
    tracks = track_bright_interfaces(image)
    sap_visible = projected["sap_core"] & first_bone
    tp_visible = projected["tp_core"] & first_bone
    sap_distance = nearest_track_distance(sap_visible, tracks)
    tp_distance = nearest_track_distance(tp_visible, tracks)
    sap_shadow = posterior_shadow_contrast(image, sap_visible)
    tp_shadow = posterior_shadow_contrast(image, tp_visible)
    margin = target_margin(projected)
    supported = bool(
        np.any(sap_visible)
        and np.any(tp_visible)
        and sap_distance is not None
        and tp_distance is not None
        and sap_distance <= maximum_track_distance
        and tp_distance <= maximum_track_distance
        and sap_shadow is not None
        and tp_shadow is not None
        and sap_shadow >= minimum_shadow
        and tp_shadow >= minimum_shadow
        and margin >= minimum_margin
    )
    return {
        "image_supported": supported,
        "track_count": len(tracks),
        "sap_track_distance_pixels": sap_distance,
        "tp_track_distance_pixels": tp_distance,
        "sap_shadow_contrast": sap_shadow,
        "tp_shadow_contrast": tp_shadow,
        "target_image_margin_pixels": margin,
        "sap_core_first_visible_pixels": int(np.count_nonzero(sap_visible)),
        "tp_core_first_visible_pixels": int(np.count_nonzero(tp_visible)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_contact_sheet(path: Path, rows: list[dict[str, Any]], images: dict[str, np.ndarray]) -> None:
    selected = sorted(
        rows,
        key=lambda row: (
            not bool(row["image_supported"]),
            row["declared_role"] != "TARGET_REGION_CANDIDATE",
            -int(row["sap_core_first_visible_pixels"] + row["tp_core_first_visible_pixels"]),
        ),
    )[:12]
    figure, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
    for axis, row in zip(axes.flat, selected):
        axis.imshow(images[row["sample_id"]], cmap="gray", origin="upper")
        axis.set_title(
            f"{row['sample_id']}\n{row['declared_role']} | {row['evaluator_label']}\n"
            f"image_support={row['image_supported']}"
        )
        axis.axis("off")
    for axis in axes.flat[len(selected) :]:
        axis.axis("off")
    figure.suptitle("Clean ultrasound only; geometry labels shown as evaluator captions")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-id", action="append", dest="session_ids")
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    sessions = sessions_for_evaluation(protocol, args.session_ids)
    capture_complete = len(sessions) == len(protocol["sessions"])
    gate = protocol["observability_gate"]
    _, sap_mask, tp_mask, contract = target_workflow.load_target_geometry(
        args.prediction, args.segmentation
    )
    bone_ids = sap_correction.bone_source_ids(args.label_conversion)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    images_by_id: dict[str, np.ndarray] = {}
    all_verification_passed = True
    for session in sessions:
        session_id = session["session_id"]
        session_dir = args.run_root / "sessions" / session_id
        frames = read_jsonl(session_dir / "public/frames.jsonl")
        images = np.load(session_dir / "public/ultrasound.npz")["ultrasound"]
        coordinates = np.load(
            session_dir / "privileged/pixel_to_volume_coordinates.npz"
        )["human_img_coords"]
        raw_labels = np.load(session_dir / "privileged/raw_labels.npz")["raw_labels"]
        roles = protocol["sampling"]["planned_waypoint_evidence_by_bank"][
            session["absolute_waypoint_bank_id"]
        ]
        verification = json.loads(
            (session_dir / "verification_report.json").read_text(encoding="utf-8")
        )
        verification_passed = bool(
            verification["verdict"] == "PROJECTION_PARITY_PASS"
            and all(bool(value) for value in verification["gates"].values())
            and verification["passed_frame_count"] == verification["frame_count"]
        )
        all_verification_passed &= verification_passed
        for index, (frame, image, coordinate_map, raw, role) in enumerate(
            zip(frames, images, coordinates, raw_labels, roles)
        ):
            projected = target_workflow.projected_masks(
                coordinate_map, contract, sap_mask, tp_mask
            )
            first_bone = target_workflow.first_bone_mask(raw, bone_ids)
            label, reason = target_workflow.target_evidence_label(projected, first_bone)
            evidence = image_supported(
                image=image,
                projected=projected,
                first_bone=first_bone,
                maximum_track_distance=float(gate["maximum_interface_track_distance_pixels"]),
                minimum_shadow=float(gate["minimum_posterior_shadow_contrast"]),
                minimum_margin=int(gate["minimum_target_image_margin_pixels"]),
            )
            sample_id = f"{session_id}:{index:02d}"
            images_by_id[sample_id] = image
            rows.append(
                {
                    "sample_id": sample_id,
                    "session_id": session_id,
                    "frame_sequence": index,
                    "frame_sha256": frame["frame_sha256"],
                    "declared_role": role["declared_role"],
                    "anchor_evaluator_label": role["anchor_evaluator_label"],
                    "evaluator_label": label,
                    "evaluator_reason": reason,
                    "verification_passed": verification_passed,
                    **evidence,
                }
            )

    session_results: list[dict[str, Any]] = []
    for session in sessions:
        session_rows = [row for row in rows if row["session_id"] == session["session_id"]]
        target_rows = [
            row for row in session_rows if row["declared_role"] == "TARGET_REGION_CANDIDATE"
        ]
        control_rows = [
            row for row in session_rows if row["declared_role"] == "NEGATIVE_CONTROL"
        ]
        supported_targets = [
            row
            for row in target_rows
            if row["evaluator_label"] == "PROXY_POSITIVE" and row["image_supported"]
        ]
        supported_controls = [
            row
            for row in control_rows
            if row["evaluator_label"] == "PROXY_POSITIVE" and row["image_supported"]
        ]
        session_results.append(
            {
                "session_id": session["session_id"],
                "target_frame_count": len(target_rows),
                "supported_target_frame_count": len(supported_targets),
                "positive_control_frame_count": len(supported_controls),
                "ambiguous_frame_count": sum(
                    row["evaluator_label"] == "PROXY_AMBIGUOUS" for row in session_rows
                ),
                "negative_frame_count": sum(
                    row["evaluator_label"] == "PROXY_NEGATIVE" for row in session_rows
                ),
            }
        )
    qualifying_sessions = sum(
        row["supported_target_frame_count"]
        >= int(gate["minimum_positive_target_frames_per_session"])
        for row in session_results
    )
    positive_controls = sum(row["positive_control_frame_count"] for row in session_results)
    image_supported_positive_count = sum(
        row["evaluator_label"] == "PROXY_POSITIVE" and row["image_supported"]
        for row in rows
    )
    geometry_positive_count = sum(row["evaluator_label"] == "PROXY_POSITIVE" for row in rows)
    if (
        capture_complete
        and
        all_verification_passed
        and qualifying_sessions >= int(gate["minimum_positive_sessions"])
        and positive_controls <= int(gate["maximum_positive_control_frames"])
    ):
        verdict = "TARGET_REGION_OBSERVABLE"
    elif geometry_positive_count or image_supported_positive_count:
        verdict = "PARTIALLY_OBSERVABLE"
    else:
        verdict = "TARGET_REGION_NOT_OBSERVABLE"
    result = {
        "schema": "s0010-l4mb-target-observability-result-v1",
        "verdict": verdict,
        "anatomical_target_contract": "PARTIAL",
        "capture_complete": capture_complete,
        "evaluated_session_count": len(sessions),
        "expected_session_count": len(protocol["sessions"]),
        "all_capture_verification_passed": all_verification_passed,
        "frame_count": len(rows),
        "geometry_positive_frame_count": geometry_positive_count,
        "image_supported_positive_frame_count": image_supported_positive_count,
        "qualifying_session_count": qualifying_sessions,
        "positive_control_frame_count": positive_controls,
        "label_counts": {
            label: sum(row["evaluator_label"] == label for row in rows)
            for label in (
                "PROXY_POSITIVE",
                "PROXY_AMBIGUOUS",
                "PROXY_NEGATIVE",
                "TECHNICALLY_INVALID",
            )
        },
        "session_results": session_results,
        "robot_search_eligible": bool(
            capture_complete and verdict == "TARGET_REGION_OBSERVABLE"
        ),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "frame_results.csv", rows)
    write_csv(args.output_dir / "session_results.csv", session_results)
    save_contact_sheet(args.output_dir / "clean_ultrasound_summary.png", rows, images_by_id)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
