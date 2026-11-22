#!/usr/bin/env python3
"""Evaluate clean-image evidence for geometry-designed RIGHT L5 SAP/TP views."""

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
import evaluate_s0010_l4mb_target_observability as historical_evaluator
import run_s0010_l4_medial_branch_target as target_workflow


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def surface_image_evidence(
    image: np.ndarray,
    projected: dict[str, np.ndarray],
    first_bone: np.ndarray,
    gate: dict[str, Any],
) -> dict[str, Any]:
    surface_projected = {
        **projected,
        "sap_core": projected["sap"],
        "tp_core": projected["tp"],
    }
    evidence = historical_evaluator.image_supported(
        image=image,
        projected=surface_projected,
        first_bone=first_bone,
        maximum_track_distance=float(gate["maximum_interface_track_distance_pixels"]),
        minimum_shadow=float(gate["minimum_posterior_shadow_contrast"]),
        minimum_margin=int(gate["minimum_target_image_margin_pixels"]),
    )
    sap_count = int(evidence.pop("sap_core_first_visible_pixels"))
    tp_count = int(evidence.pop("tp_core_first_visible_pixels"))
    enough_geometry = bool(
        sap_count >= int(gate["minimum_sap_first_visible_pixels"])
        and tp_count >= int(gate["minimum_tp_first_visible_pixels"])
    )
    evidence["sap_first_visible_pixels"] = sap_count
    evidence["tp_first_visible_pixels"] = tp_count
    evidence["geometry_surface_pair_supported"] = enough_geometry
    evidence["image_supported"] = bool(evidence["image_supported"] and enough_geometry)
    return evidence


def save_review_sheet(
    path: Path,
    rows: list[dict[str, Any]],
    images: dict[str, np.ndarray],
    overlays: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    selected = sorted(
        rows,
        key=lambda row: (
            not bool(row["image_supported"]),
            row["declared_role"] != "SAP_TP_SURFACE_CANDIDATE",
            -min(int(row["sap_first_visible_pixels"]), int(row["tp_first_visible_pixels"])),
        ),
    )[:8]
    figure, axes = plt.subplots(4, 4, figsize=(13, 12), constrained_layout=True)
    for item, row in enumerate(selected):
        clean_axis = axes.flat[item * 2]
        overlay_axis = axes.flat[item * 2 + 1]
        sample_id = row["sample_id"]
        image = images[sample_id]
        sap_visible, tp_visible = overlays[sample_id]
        clean_axis.imshow(image, cmap="gray", origin="upper")
        clean_axis.set_title(f"{sample_id} clean\n{row['declared_role']}", fontsize=8)
        overlay_axis.imshow(image, cmap="gray", origin="upper")
        sap_rows, sap_columns = np.where(sap_visible)
        tp_rows, tp_columns = np.where(tp_visible)
        overlay_axis.scatter(sap_columns, sap_rows, s=8, c="#00d4ff", label="SAP first hit")
        overlay_axis.scatter(tp_columns, tp_rows, s=8, c="#ff9f1c", label="TP first hit")
        overlay_axis.set_title(
            f"evaluator overlay | image={row['image_supported']}\n"
            f"SAP={len(sap_rows)} TP={len(tp_rows)}",
            fontsize=8,
        )
        overlay_axis.legend(fontsize=6, loc="upper right")
        for axis in (clean_axis, overlay_axis):
            axis.set_xlim(0, image.shape[1] - 1)
            axis.set_ylim(image.shape[0] - 1, 0)
            axis.axis("off")
    for axis in axes.flat[len(selected) * 2 :]:
        axis.axis("off")
    figure.suptitle(
        "Clean ultrasound (left) and privileged first-hit anatomy (right); not runtime input"
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != "sonogym-s0010-l4mb-sap-tp-signal-v1":
        raise ValueError("unexpected protocol identifier")
    gate = protocol["observability_gate"]
    _, sap_mask, tp_mask, contract = target_workflow.load_target_geometry(
        args.prediction, args.segmentation
    )
    bone_ids = sap_correction.bone_source_ids(args.label_conversion)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    images_by_id: dict[str, np.ndarray] = {}
    overlays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_verification_passed = True
    for session in protocol["sessions"]:
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
        if not (len(frames) == len(images) == len(coordinates) == len(raw_labels) == len(roles)):
            raise ValueError(f"capture arrays differ for {session_id}")
        for index, (frame, image, coordinate_map, raw, role) in enumerate(
            zip(frames, images, coordinates, raw_labels, roles)
        ):
            projected = target_workflow.projected_masks(
                coordinate_map, contract, sap_mask, tp_mask
            )
            first_bone = target_workflow.first_bone_mask(raw, bone_ids)
            evaluator_label, evaluator_reason = target_workflow.target_evidence_label(
                projected, first_bone
            )
            evidence = surface_image_evidence(image, projected, first_bone, gate)
            sample_id = f"{session_id}:{index:02d}"
            images_by_id[sample_id] = image
            overlays[sample_id] = (
                projected["sap"] & first_bone,
                projected["tp"] & first_bone,
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "session_id": session_id,
                    "frame_sequence": index,
                    "frame_sha256": frame["frame_sha256"],
                    "declared_role": role["declared_role"],
                    "target_core_evaluator_label": evaluator_label,
                    "target_core_evaluator_reason": evaluator_reason,
                    "verification_passed": verification_passed,
                    **evidence,
                }
            )

    session_results: list[dict[str, Any]] = []
    for session in protocol["sessions"]:
        session_rows = [row for row in rows if row["session_id"] == session["session_id"]]
        targets = [
            row for row in session_rows if row["declared_role"] == "SAP_TP_SURFACE_CANDIDATE"
        ]
        controls = [row for row in session_rows if row["declared_role"] == "NEGATIVE_CONTROL"]
        session_results.append(
            {
                "session_id": session["session_id"],
                "target_frame_count": len(targets),
                "geometry_supported_target_frame_count": sum(
                    bool(row["geometry_surface_pair_supported"]) for row in targets
                ),
                "image_supported_target_frame_count": sum(
                    bool(row["image_supported"]) for row in targets
                ),
                "image_supported_control_frame_count": sum(
                    bool(row["image_supported"]) for row in controls
                ),
            }
        )
    qualifying_sessions = sum(
        row["image_supported_target_frame_count"]
        >= int(gate["minimum_supported_target_frames_per_session"])
        for row in session_results
    )
    supported_controls = sum(
        row["image_supported_control_frame_count"] for row in session_results
    )
    if (
        all_verification_passed
        and qualifying_sessions >= int(gate["minimum_supported_sessions"])
        and supported_controls <= int(gate["maximum_supported_control_frames"])
    ):
        verdict = "SAP_TP_IMAGE_SIGNAL_SUPPORTED"
    elif any(bool(row["image_supported"]) for row in rows):
        verdict = "SAP_TP_IMAGE_SIGNAL_PARTIAL"
    else:
        verdict = "SAP_TP_IMAGE_SIGNAL_NOT_SUPPORTED"
    result = {
        "schema": "s0010-l4mb-sap-tp-signal-result-v1",
        "verdict": verdict,
        "all_capture_verification_passed": all_verification_passed,
        "session_count": len(session_results),
        "frame_count": len(rows),
        "geometry_supported_target_frame_count": sum(
            row["declared_role"] == "SAP_TP_SURFACE_CANDIDATE"
            and bool(row["geometry_surface_pair_supported"])
            for row in rows
        ),
        "image_supported_target_frame_count": sum(
            row["declared_role"] == "SAP_TP_SURFACE_CANDIDATE"
            and bool(row["image_supported"])
            for row in rows
        ),
        "image_supported_control_frame_count": supported_controls,
        "qualifying_session_count": qualifying_sessions,
        "session_results": session_results,
        "target_core_observability_not_implied": True,
        "robot_localizer_training_not_authorized_by_this_result": True,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "frame_results.csv", rows)
    write_csv(args.output_dir / "session_results.csv", session_results)
    save_review_sheet(
        args.output_dir / "sap_tp_signal_review.png", rows, images_by_id, overlays
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
