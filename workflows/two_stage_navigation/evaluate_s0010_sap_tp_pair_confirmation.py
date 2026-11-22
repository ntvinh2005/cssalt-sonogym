#!/usr/bin/env python3
"""Evaluate the frozen image-only pair selector on fresh simulator captures."""

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
import run_s0010_l4_medial_branch_target as target_workflow
import run_s0010_sap_tp_pair_development as development
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


def target_margin(projected: dict[str, np.ndarray]) -> int:
    """Return the image-edge margin of the frozen SAP/TP core evidence."""

    pixels = np.argwhere(projected["sap_core"] | projected["tp_core"])
    if not len(pixels):
        return -1
    height, width = projected["sap"].shape
    return int(
        min(
            pixels[:, 0].min(),
            height - 1 - pixels[:, 0].max(),
            pixels[:, 1].min(),
            width - 1 - pixels[:, 1].max(),
        )
    )


def save_figure(path: Path, rows: list[dict[str, Any]], images: dict[str, np.ndarray]) -> None:
    selected: list[dict[str, Any]] = []
    for session_id in sorted({row["session_id"] for row in rows}):
        session_rows = sorted(
            [row for row in rows if row["session_id"] == session_id],
            key=lambda row: float(row["accumulated_score"]),
            reverse=True,
        )
        selected.extend(session_rows[:2])
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, row in zip(axes.flat, selected):
        axis.imshow(images[row["sample_id"]], cmap="gray", origin="upper")
        pair = row.get("selected_pair")
        if pair is not None:
            axis.plot(pair["first"].columns, pair["first"].rows, color="#ff9f1c", linewidth=2)
            axis.plot(pair["second"].columns, pair["second"].rows, color="#00d4ff", linewidth=2)
        axis.set_title(
            f"{row['sample_id']} | actual={row['actual_state']}\n"
            f"score={row['accumulated_score']:.2f} accepted={row['accepted']} "
            f"correct={row['selected_pair_correct']}",
            fontsize=8,
        )
        axis.axis("off")
    figure.suptitle("Frozen actor selector on fresh views; no geometry is displayed")
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
    if protocol["protocol_id"] != "sonogym-s0010-sap-tp-pair-confirmation-v1r3":
        raise ValueError("unexpected protocol identifier")
    checkpoint_path = Path(protocol["source"]["checkpoint"])
    if sha256(checkpoint_path) != protocol["source"]["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint hash changed")
    checkpoint = joblib.load(checkpoint_path)
    model = checkpoint["model"]
    feature_scale = np.asarray(checkpoint["feature_scale"], dtype=np.float64)
    _, sap_mask, tp_mask, contract = target_workflow.load_target_geometry(
        args.prediction, args.segmentation
    )
    bone_ids = sap_correction.bone_source_ids(args.label_conversion)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    images_by_id: dict[str, np.ndarray] = {}
    all_verification_passed = True
    carry_forward = protocol["source"].get("carry_forward") or {}
    carried_sessions = set(carry_forward.get("session_ids", []))
    for session in protocol["sessions"]:
        session_id = session["session_id"]
        session_dir = args.run_root / "sessions" / session_id
        verification = json.loads(
            (session_dir / "verification_report.json").read_text(encoding="utf-8")
        )
        expected_protocol_sha = (
            carry_forward.get("protocol_sha256")
            if session_id in carried_sessions
            else sha256(args.protocol)
        )
        verification_passed = bool(
            verification["verdict"] == "PROJECTION_PARITY_PASS"
            and all(bool(value) for value in verification["gates"].values())
            and verification["passed_frame_count"] == verification["frame_count"]
            and verification["protocol_sha256"] == expected_protocol_sha
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
        poses = development._cumulative_pose(frames)
        if not (len(frames) == len(images) == len(coordinates) == len(raw_labels) == len(roles)):
            raise ValueError("capture arrays differ")
        for index, (frame, image, coordinate_map, raw, role) in enumerate(
            zip(frames, images, coordinates, raw_labels, roles)
        ):
            projected = target_workflow.projected_masks(
                coordinate_map, contract, sap_mask, tp_mask
            )
            first_bone = target_workflow.first_bone_mask(raw, bone_ids)
            sap_visible = projected["sap"] & first_bone
            tp_visible = projected["tp"] & first_bone
            margin = target_margin(projected)
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
                sap_count >= int(protocol["pair_label_gate"]["minimum_sap_first_visible_pixels"])
                and tp_count >= int(protocol["pair_label_gate"]["minimum_tp_first_visible_pixels"])
                and margin >= int(protocol["pair_label_gate"]["minimum_target_image_margin_pixels"])
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
            base_score, selected_index = development._score_row(
                {"pairs": pair_rows}, model
            )
            selected_pair = pair_rows[selected_index] if selected_index is not None else None
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
                    "positive_pair_count": sum(pair["positive"] for pair in pair_rows),
                    "pairs": pair_rows,
                    "base_score": base_score,
                    "selected_pair_index": selected_index,
                    "selected_pair": selected_pair,
                    "selected_pair_correct": bool(
                        selected_pair is not None and selected_pair["positive"]
                    ),
                    "training_feature_scale": feature_scale,
                    "cumulative_pose": poses[index],
                    "verification_passed": verification_passed,
                }
            )
    development._add_prior_view_support(rows, protocol)

    session_results: list[dict[str, Any]] = []
    top1_correct = 0
    top3_correct = 0
    sufficient_sessions = 0
    for session in protocol["sessions"]:
        session_rows = [row for row in rows if row["session_id"] == session["session_id"]]
        actual_targets = [row for row in session_rows if row["actual_state"] == "TARGET"]
        actual_negatives = [
            row for row in session_rows if row["actual_state"] == "CLEAR_NEGATIVE"
        ]
        accepted = sorted(
            [row for row in session_rows if row["accepted"]],
            key=lambda row: float(row["accumulated_score"]),
            reverse=True,
        )
        correct_accepted = [row for row in accepted if row["selected_pair_correct"]]
        session_top1 = bool(accepted and accepted[0]["selected_pair_correct"])
        session_top3 = bool(any(row["selected_pair_correct"] for row in accepted[:3]))
        sufficient = bool(
            len(actual_targets) >= int(protocol["final_gate"]["minimum_actual_targets_per_session"])
            and len(actual_negatives)
            >= int(protocol["final_gate"]["minimum_actual_clear_negatives_per_session"])
        )
        top1_correct += session_top1
        top3_correct += session_top3
        sufficient_sessions += sufficient
        session_results.append(
            {
                "session_id": session["session_id"],
                "actual_target_count": len(actual_targets),
                "actual_clear_negative_count": len(actual_negatives),
                "ambiguous_count": len(session_rows) - len(actual_targets) - len(actual_negatives),
                "accepted_count": len(accepted),
                "correct_accepted_target_count": len(correct_accepted),
                "top1_correct": session_top1,
                "top3_contains_correct": session_top3,
                "sample_sufficiency_passed": sufficient,
            }
        )
    nonambiguous_accepted = [
        row for row in rows if row["accepted"] and row["actual_state"] != "AMBIGUOUS"
    ]
    correct_accepted = [row for row in nonambiguous_accepted if row["selected_pair_correct"]]
    negatives = [row for row in rows if row["actual_state"] == "CLEAR_NEGATIVE"]
    accepted_negatives = [row for row in negatives if row["accepted"]]
    pair_precision = len(correct_accepted) / max(len(nonambiguous_accepted), 1)
    negative_false_accept_rate = len(accepted_negatives) / max(len(negatives), 1)
    final_gate = protocol["final_gate"]
    passed = bool(
        all_verification_passed
        and sufficient_sessions == len(protocol["sessions"])
        and top1_correct >= int(final_gate["minimum_top1_correct_sessions"])
        and top3_correct >= int(final_gate["minimum_top3_correct_sessions"])
        and pair_precision >= float(final_gate["minimum_pair_precision"])
        and negative_false_accept_rate
        <= float(final_gate["maximum_clear_negative_false_accept_rate"])
        and all(
            row["correct_accepted_target_count"]
            >= int(final_gate["minimum_correct_accepted_targets_per_session"])
            for row in session_results
        )
    )
    result = {
        "schema": "s0010-sap-tp-pair-confirmation-result-v1",
        "passed": passed,
        "all_capture_verification_passed": all_verification_passed,
        "checkpoint_sha256": sha256(checkpoint_path),
        "frame_count": len(rows),
        "actual_target_count": sum(row["actual_state"] == "TARGET" for row in rows),
        "actual_clear_negative_count": len(negatives),
        "ambiguous_count": sum(row["actual_state"] == "AMBIGUOUS" for row in rows),
        "accepted_count": len([row for row in rows if row["accepted"]]),
        "correct_accepted_pair_count": len(correct_accepted),
        "pair_precision": pair_precision,
        "accepted_clear_negative_count": len(accepted_negatives),
        "clear_negative_false_accept_rate": negative_false_accept_rate,
        "top1_correct_sessions": top1_correct,
        "top3_correct_sessions": top3_correct,
        "sufficient_session_count": sufficient_sessions,
        "session_results": session_results,
        "robot_navigation_executed": False,
        "junction_or_needle_endpoint_implied": False,
    }
    serializable_rows = []
    for row in rows:
        serializable_rows.append(
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "pairs",
                    "selected_pair",
                    "training_feature_scale",
                    "cumulative_pose",
                }
            }
            | {"support_sample_ids": ";".join(row["support_sample_ids"])}
        )
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "frame_results.csv", serializable_rows)
    write_csv(args.output_dir / "session_results.csv", session_results)
    save_figure(args.output_dir / "actor_pair_confirmation.png", rows, images_by_id)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
