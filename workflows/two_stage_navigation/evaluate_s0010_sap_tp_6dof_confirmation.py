#!/usr/bin/env python3
"""Evaluate frozen image evidence plus public robot 6DoF on fresh captures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml

import correct_s0010_sap_observability_v1 as sap_correction
import evaluate_s0010_sap_tp_pair_confirmation as pair_evaluation
import run_s0010_l4_medial_branch_target as target_workflow
import run_s0010_sap_tp_6dof_development as geometry
import run_s0010_sap_tp_pair_development as development
from two_stage_navigation.acquisition_integrity import validate_actor_6dof_record
from two_stage_navigation.ultrasound_sap_tp_configuration import (
    track_window_pair_candidates,
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
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(
    path: Path, rows: list[dict[str, Any]], images: dict[str, np.ndarray]
) -> None:
    selected: list[dict[str, Any]] = []
    for session_id in sorted({row["session_id"] for row in rows}):
        session = sorted(
            [row for row in rows if row["session_id"] == session_id],
            key=lambda row: float(row["accumulated_score"]),
            reverse=True,
        )
        selected.extend(session[:2])
    figure, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for axis, row in zip(axes.flat, selected):
        axis.imshow(images[row["sample_id"]], cmap="gray", origin="upper")
        pair = row.get("selected_pair")
        if pair is not None:
            axis.plot(
                pair["first_segment"].columns,
                pair["first_segment"].rows,
                color="#ff9f1c",
                linewidth=2,
            )
            axis.plot(
                pair["second_segment"].columns,
                pair["second_segment"].rows,
                color="#00d4ff",
                linewidth=2,
            )
        axis.set_title(
            f"{row['sample_id']} actual={row['actual_state']}\n"
            f"score={row['accumulated_score']:.2f} support={row['support_count']} "
            f"correct={row['selected_pair_correct']}",
            fontsize=8,
        )
        axis.axis("off")
    figure.suptitle("Actor inputs only: clean ultrasound and public base-to-EE pose")
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
    if protocol["protocol_id"] != "sonogym-s0010-sap-tp-public-6dof-confirmation-v1":
        raise ValueError("unexpected protocol identifier")
    checkpoint_path = Path(protocol["source"]["checkpoint"])
    if sha256(checkpoint_path) != protocol["source"]["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint hash changed")
    model = joblib.load(checkpoint_path)["model"]
    calibration = protocol["public_probe_calibration"]
    _, sap_mask, tp_mask, contract = target_workflow.load_target_geometry(
        args.prediction, args.segmentation
    )
    bone_ids = sap_correction.bone_source_ids(args.label_conversion)
    rows: list[dict[str, Any]] = []
    images_by_id: dict[str, np.ndarray] = {}
    all_verification_passed = True
    for session in protocol["sessions"]:
        session_id = session["session_id"]
        session_dir = args.run_root / "sessions" / session_id
        replay = json.loads(
            (session_dir / "verification_report.json").read_text(encoding="utf-8")
        )
        actor_validation = json.loads(
            (session_dir / "actor_6dof_validation.json").read_text(encoding="utf-8")
        )
        verification_passed = bool(
            replay["verdict"] == "PROJECTION_PARITY_PASS"
            and all(bool(value) for value in replay["gates"].values())
            and replay["passed_frame_count"] == replay["frame_count"]
            and replay["protocol_sha256"] == sha256(args.protocol)
            and actor_validation["passed"]
            and actor_validation["protocol_sha256"] == sha256(args.protocol)
        )
        all_verification_passed &= verification_passed
        frames = read_jsonl(session_dir / "public/frames.jsonl")
        images = np.load(session_dir / "public/ultrasound.npz")["ultrasound"]
        coordinates = np.load(
            session_dir / "privileged/pixel_to_volume_coordinates.npz"
        )["human_img_coords"]
        raw_labels = np.load(session_dir / "privileged/raw_labels.npz")["raw_labels"]
        roles = protocol["sampling"]["planned_waypoint_evidence_by_bank"][
            session["absolute_waypoint_bank_id"]
        ]
        if not (
            len(frames)
            == len(images)
            == len(coordinates)
            == len(raw_labels)
            == len(roles)
        ):
            raise ValueError("capture arrays differ")
        for index, (frame, image, coordinate_map, raw, role) in enumerate(
            zip(frames, images, coordinates, raw_labels, roles)
        ):
            validate_actor_6dof_record(frame)
            projected = target_workflow.projected_masks(
                coordinate_map, contract, sap_mask, tp_mask
            )
            first_bone = target_workflow.first_bone_mask(raw, bone_ids)
            sap_visible = projected["sap"] & first_bone
            tp_visible = projected["tp"] & first_bone
            margin = pair_evaluation.target_margin(projected)
            pair_rows = [
                development._pair_record(
                    first,
                    second,
                    feature,
                    tp_visible,
                    sap_visible,
                    protocol["pair_label_gate"],
                )
                for first, second, feature in track_window_pair_candidates(
                    image, **protocol["track_window_config"]
                )
            ]
            sap_count = int(np.count_nonzero(sap_visible))
            tp_count = int(np.count_nonzero(tp_visible))
            geometry_target = bool(
                sap_count
                >= int(protocol["pair_label_gate"]["minimum_sap_first_visible_pixels"])
                and tp_count
                >= int(protocol["pair_label_gate"]["minimum_tp_first_visible_pixels"])
                and margin
                >= int(protocol["pair_label_gate"]["minimum_target_image_margin_pixels"])
            )
            evaluator_label, evaluator_reason = target_workflow.target_evidence_label(
                projected, first_bone
            )
            if geometry_target:
                actual_state = "TARGET"
            elif evaluator_label == "PROXY_NEGATIVE" and evaluator_reason in {
                "TP_ONLY_WITHOUT_SAP_IN_PLANE",
                "OTHER_BONE_OR_BACKGROUND",
            }:
                actual_state = "CLEAR_NEGATIVE"
            else:
                actual_state = "AMBIGUOUS"
            actor_pose = dict(frame)
            actor_pose["image_resolution_m"] = calibration["pixel_resolution_m"]
            actor_pose["image_origin_height_m"] = calibration["image_origin_height_m"]
            pair = geometry.selected_pair(image, actor_pose, model, protocol)
            selected_index = None
            if pair["first_segment"] is not None:
                selected_index = int(
                    np.argmax(
                        model.predict_proba(
                            np.asarray(
                                [candidate[2] for candidate in track_window_pair_candidates(
                                    image, **protocol["track_window_config"]
                                )],
                                dtype=np.float64,
                            )
                        )[:, 1]
                    )
                )
            selected_pair_correct = bool(
                selected_index is not None and pair_rows[selected_index]["positive"]
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
                    "actual_state": actual_state,
                    "evaluator_reason": evaluator_reason,
                    "sap_first_visible_pixels": sap_count,
                    "tp_first_visible_pixels": tp_count,
                    "target_image_margin_pixels": margin,
                    "positive_pair_count": sum(item["positive"] for item in pair_rows),
                    "selected_pair_correct": selected_pair_correct,
                    "pose_position_m": np.asarray(
                        frame["actor_base_to_ee_position_m"], dtype=np.float64
                    ),
                    "pose_rotation": np.asarray(
                        frame["actor_base_to_ee_rotation_columns"], dtype=np.float64
                    ),
                    "verification_passed": verification_passed,
                    "selected_pair": pair,
                    **pair,
                }
            )
    gate = protocol["decision_gate"]
    geometry.add_support(
        rows,
        surface_radius_mm=float(gate["surface_radius_mm"]),
        relative_vector_tolerance_mm=float(
            gate["relative_pair_vector_tolerance_mm"]
        ),
        minimum_support=int(gate["minimum_independent_view_support"]),
    )
    summary = geometry.metrics(rows)
    final_gate = protocol["final_gate"]
    sufficient = all(
        sum(
            row["actual_state"] == "TARGET"
            for row in rows
            if row["session_id"] == session["session_id"]
        )
        >= int(final_gate["minimum_actual_targets_per_session"])
        and sum(
            row["actual_state"] == "CLEAR_NEGATIVE"
            for row in rows
            if row["session_id"] == session["session_id"]
        )
        >= int(final_gate["minimum_actual_clear_negatives_per_session"])
        for session in protocol["sessions"]
    )
    passed = bool(
        all_verification_passed
        and sufficient
        and summary["top1_correct_sessions"]
        >= int(final_gate["minimum_top1_correct_sessions"])
        and summary["top3_correct_sessions"]
        >= int(final_gate["minimum_top3_correct_sessions"])
        and summary["pair_precision"] >= float(final_gate["minimum_pair_precision"])
        and summary["clear_negative_false_accept_rate"]
        <= float(final_gate["maximum_clear_negative_false_accept_rate"])
        and all(
            row["correct_accepted_target_count"]
            >= int(final_gate["minimum_correct_accepted_targets_per_session"])
            for row in summary["session_results"]
        )
    )
    result = {
        "schema": "s0010-sap-tp-public-6dof-confirmation-result-v1",
        "passed": passed,
        "all_capture_verification_passed": all_verification_passed,
        "sample_sufficiency_passed": sufficient,
        "checkpoint_sha256": sha256(checkpoint_path),
        "protocol_sha256": sha256(args.protocol),
        "frame_count": len(rows),
        **summary,
        "actor_inputs": ["clean_ultrasound", "public_robot_base_to_ee_pose"],
        "robot_navigation_executed": False,
        "junction_or_nerve_endpoint_implied": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    serializable = []
    excluded = {
        "first",
        "second",
        "first_segment",
        "second_segment",
        "first_center",
        "second_center",
        "pose_position_m",
        "pose_rotation",
        "selected_pair",
    }
    for row in rows:
        serializable.append(
            {key: value for key, value in row.items() if key not in excluded}
            | {"support_sample_ids": ";".join(row["support_sample_ids"])}
        )
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "frame_results.csv", serializable)
    write_csv(args.output_dir / "session_results.csv", summary["session_results"])
    save_figure(args.output_dir / "actor_public_6dof_confirmation.png", rows, images_by_id)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
