#!/usr/bin/env python3
"""Evaluate image-only SAP/TP interface-pair selection on opened development views."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import ndimage
from sklearn.ensemble import ExtraTreesClassifier

import correct_s0010_sap_observability_v1 as sap_correction
import run_s0010_l4_medial_branch_target as target_workflow
from two_stage_navigation.ultrasound_sap_tp_configuration import (
    CorticalSegment,
    SegmentConfig,
    pair_candidates,
    track_window_pair_candidates,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def segment_mask_distances(segment: CorticalSegment, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.full(len(segment.rows), np.inf, dtype=np.float64)
    distance = ndimage.distance_transform_edt(~mask)
    rows = np.clip(np.rint(segment.rows).astype(int), 0, mask.shape[0] - 1)
    columns = np.clip(np.rint(segment.columns).astype(int), 0, mask.shape[1] - 1)
    return distance[rows, columns]


def pair_matches_geometry(
    tp_segment: CorticalSegment,
    sap_segment: CorticalSegment,
    tp_mask: np.ndarray,
    sap_mask: np.ndarray,
    maximum_distance_pixels: float,
    minimum_aligned_points: int,
) -> bool:
    tp_distance = segment_mask_distances(tp_segment, tp_mask)
    sap_distance = segment_mask_distances(sap_segment, sap_mask)
    return bool(
        np.sum(tp_distance <= maximum_distance_pixels) >= minimum_aligned_points
        and np.sum(sap_distance <= maximum_distance_pixels) >= minimum_aligned_points
    )


def _pair_record(
    first: CorticalSegment,
    second: CorticalSegment,
    feature: np.ndarray,
    tp_mask: np.ndarray,
    sap_mask: np.ndarray,
    gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "first": first,
        "second": second,
        "feature": feature,
        "positive": pair_matches_geometry(
            first,
            second,
            tp_mask,
            sap_mask,
            float(gate["maximum_segment_surface_distance_pixels"]),
            int(gate["minimum_aligned_segment_points"]),
        ),
    }


def _fit_model(
    rows: list[dict[str, Any]],
    training_groups: set[int],
    seed: int,
    trees: int,
    maximum_negative_pairs_per_frame: int,
) -> tuple[ExtraTreesClassifier, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for row in rows:
        if int(row["frame_sequence"]) not in training_groups:
            continue
        positives = [pair for pair in row["pairs"] if bool(pair["positive"])]
        negatives = [pair for pair in row["pairs"] if not bool(pair["positive"])]
        negatives.sort(
            key=lambda pair: pair["first"].ridge_score + pair["second"].ridge_score,
            reverse=True,
        )
        for pair in positives + negatives[:maximum_negative_pairs_per_frame]:
            features.append(pair["feature"])
            labels.append(int(pair["positive"]))
    feature_matrix = np.asarray(features, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    if feature_matrix.ndim != 2 or feature_matrix.shape[1] != 21:
        raise ValueError("training pair features are incomplete")
    if set(label_array.tolist()) != {0, 1}:
        raise ValueError("training fold lacks a pair class")
    model = ExtraTreesClassifier(
        n_estimators=trees,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
    )
    model.fit(feature_matrix, label_array)
    return model, np.mean(feature_matrix, axis=0), np.maximum(np.std(feature_matrix, axis=0), 1e-6)


def _score_row(
    row: dict[str, Any],
    model: ExtraTreesClassifier,
) -> tuple[float, int | None]:
    if not row["pairs"]:
        return 0.0, None
    features = np.asarray([pair["feature"] for pair in row["pairs"]], dtype=np.float64)
    probabilities = model.predict_proba(features)[:, 1]
    selected = int(np.argmax(probabilities))
    return float(probabilities[selected]), selected


def _cumulative_pose(frames: list[dict[str, Any]]) -> np.ndarray:
    result = np.zeros((len(frames), 3), dtype=np.float64)
    for index, frame in enumerate(frames):
        if index:
            result[index] = result[index - 1]
        delta = np.asarray(frame["executed_delta_relative_mm"], dtype=np.float64)
        result[index, :2] += delta
        result[index, 2] += float(frame["executed_delta_angle_rad"])
    return result


def _add_prior_view_support(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> None:
    gate = protocol["decision_gate"]
    for session_id in sorted({str(row["session_id"]) for row in rows}):
        session_rows = sorted(
            [row for row in rows if row["session_id"] == session_id],
            key=lambda row: int(row["frame_sequence"]),
        )
        for index, row in enumerate(session_rows):
            selected = row.get("selected_pair")
            support: list[str] = []
            if selected is not None:
                current = selected["feature"]
                for previous in session_rows[:index]:
                    previous_selected = previous.get("selected_pair")
                    if previous_selected is None:
                        continue
                    translation = float(
                        np.linalg.norm(row["cumulative_pose"][:2] - previous["cumulative_pose"][:2])
                    )
                    angle = abs(float(row["cumulative_pose"][2] - previous["cumulative_pose"][2]))
                    distinct = bool(
                        translation >= float(gate["minimum_support_translation_mm"])
                        or np.degrees(angle) >= float(gate["minimum_support_angle_deg"])
                    )
                    nearby = bool(
                        translation <= float(gate["maximum_support_translation_mm"])
                        and np.degrees(angle) <= float(gate["maximum_support_angle_deg"])
                    )
                    scaled_distance = float(
                        np.mean(
                            np.abs(
                                (current - previous_selected["feature"])
                                / row["training_feature_scale"]
                            )
                        )
                    )
                    if (
                        distinct
                        and nearby
                        and float(previous["base_score"]) >= float(gate["minimum_pair_probability"])
                        and scaled_distance <= float(gate["maximum_scaled_pair_distance"])
                    ):
                        support.append(str(previous["sample_id"]))
            row["support_sample_ids"] = support
            row["support_count"] = len(support)
            row["accumulated_score"] = min(
                1.0,
                float(row["base_score"])
                + float(gate["support_score_increment"])
                * min(len(support), int(gate["maximum_counted_support"])),
            )
            row["accepted"] = bool(
                float(row["base_score"]) >= float(gate["minimum_pair_probability"])
                and len(support) >= int(gate["minimum_prior_view_support"])
            )


def save_figure(path: Path, rows: list[dict[str, Any]], images: dict[str, np.ndarray]) -> None:
    ordered = sorted(rows, key=lambda row: float(row["accumulated_score"]), reverse=True)
    examples = ordered[:4] + ordered[-2:]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, row in zip(axes.flat, examples):
        image = images[row["sample_id"]]
        axis.imshow(image, cmap="gray", origin="upper")
        selected = row.get("selected_pair")
        if selected is not None:
            first = selected["first"]
            second = selected["second"]
            axis.plot(first.columns, first.rows, color="#ff9f1c", linewidth=2)
            axis.plot(second.columns, second.rows, color="#00d4ff", linewidth=2)
        axis.set_title(
            f"{row['sample_id']} | {row['declared_role']}\n"
            f"p={row['base_score']:.2f} support={row['support_count']} "
            f"correct={row['selected_pair_correct']}",
            fontsize=8,
        )
        axis.axis("off")
    figure.suptitle("Actor image-derived pair selection; no geometry is shown")
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
    if protocol["protocol_id"] != "sonogym-s0010-sap-tp-pair-development-v1":
        raise ValueError("unexpected protocol identifier")
    source_protocol_path = Path(protocol["inputs"]["source_protocol"])
    source_protocol = yaml.safe_load(source_protocol_path.read_text(encoding="utf-8"))
    config = SegmentConfig(**protocol["segment_config"])
    _, sap_mask, tp_mask, contract = target_workflow.load_target_geometry(
        args.prediction, args.segmentation
    )
    bone_ids = sap_correction.bone_source_ids(args.label_conversion)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    images_by_id: dict[str, np.ndarray] = {}
    for session in source_protocol["sessions"]:
        session_id = str(session["session_id"])
        session_dir = args.run_root / "sessions" / session_id
        frames = read_jsonl(session_dir / "public/frames.jsonl")
        images = np.load(session_dir / "public/ultrasound.npz")["ultrasound"]
        coordinates = np.load(
            session_dir / "privileged/pixel_to_volume_coordinates.npz"
        )["human_img_coords"]
        raw_labels = np.load(session_dir / "privileged/raw_labels.npz")["raw_labels"]
        roles = source_protocol["sampling"]["planned_waypoint_evidence_by_bank"][
            session["absolute_waypoint_bank_id"]
        ]
        poses = _cumulative_pose(frames)
        for index, (frame, image, coordinate_map, raw, role) in enumerate(
            zip(frames, images, coordinates, raw_labels, roles)
        ):
            projected = target_workflow.projected_masks(
                coordinate_map, contract, sap_mask, tp_mask
            )
            first_bone = target_workflow.first_bone_mask(raw, bone_ids)
            sap_visible = projected["sap"] & first_bone
            tp_visible = projected["tp"] & first_bone
            if protocol.get("candidate_mode", "bounded_components") == "track_windows":
                candidate_pairs = track_window_pair_candidates(
                    image, **protocol["track_window_config"]
                )
            else:
                candidate_pairs = pair_candidates(image, config)
            pairs = [
                _pair_record(
                    first,
                    second,
                    feature,
                    tp_visible,
                    sap_visible,
                    protocol["pair_label_gate"],
                )
                for first, second, feature in candidate_pairs
            ]
            sample_id = f"{session_id}:{index:02d}"
            images_by_id[sample_id] = image
            rows.append(
                {
                    "sample_id": sample_id,
                    "session_id": session_id,
                    "frame_sequence": index,
                    "declared_role": role["declared_role"],
                    "frame_sha256": frame["frame_sha256"],
                    "pairs": pairs,
                    "positive_pair_count": sum(bool(pair["positive"]) for pair in pairs),
                    "cumulative_pose": poses[index],
                }
            )

    evaluation_groups: set[int] = set()
    for fold_index, fold in enumerate(protocol["cross_validation_folds"]):
        held_out = {int(value) for value in fold["held_out_waypoint_groups"]}
        if evaluation_groups & held_out:
            raise ValueError("waypoint group appears in more than one evaluation fold")
        evaluation_groups |= held_out
        training = set(range(12)) - held_out
        model, feature_mean, feature_scale = _fit_model(
            rows,
            training,
            int(protocol["model"]["random_seed"]) + fold_index,
            int(protocol["model"]["tree_count"]),
            int(protocol["model"]["maximum_negative_pairs_per_frame"]),
        )
        for row in rows:
            if int(row["frame_sequence"]) not in held_out:
                continue
            base_score, selected_index = _score_row(row, model)
            row["fold_id"] = fold["fold_id"]
            row["base_score"] = base_score
            row["selected_pair_index"] = selected_index
            row["selected_pair"] = (
                row["pairs"][selected_index] if selected_index is not None else None
            )
            row["selected_pair_correct"] = bool(
                selected_index is not None and row["pairs"][selected_index]["positive"]
            )
            row["training_feature_mean"] = feature_mean
            row["training_feature_scale"] = feature_scale
    if evaluation_groups != set(range(12)) or any("base_score" not in row for row in rows):
        raise ValueError("cross-validation did not evaluate every waypoint group once")
    _add_prior_view_support(rows, protocol)

    clear_negative_groups = set(protocol["labels"]["clear_negative_waypoint_groups"])
    ambiguous_groups = set(protocol["labels"]["ambiguous_waypoint_groups"])
    target_groups = set(protocol["labels"]["target_waypoint_groups"])
    group_results: list[dict[str, Any]] = []
    for group in range(12):
        group_rows = [row for row in rows if int(row["frame_sequence"]) == group]
        if group in target_groups:
            label = "TARGET"
            passed = sum(bool(row["accepted"] and row["selected_pair_correct"]) for row in group_rows) >= 2
        elif group in clear_negative_groups:
            label = "CLEAR_NEGATIVE"
            passed = sum(bool(row["accepted"]) for row in group_rows) < 2
        elif group in ambiguous_groups:
            label = "AMBIGUOUS"
            passed = True
        else:
            raise ValueError("waypoint group lacks an evaluation label")
        group_results.append(
            {
                "waypoint_group": group,
                "label": label,
                "accepted_frames": sum(bool(row["accepted"]) for row in group_rows),
                "correct_pair_frames": sum(
                    bool(row["accepted"] and row["selected_pair_correct"])
                    for row in group_rows
                ),
                "frame_count": len(group_rows),
                "group_gate_passed": passed,
            }
        )
    target_groups_passed = sum(
        row["label"] == "TARGET" and bool(row["group_gate_passed"])
        for row in group_results
    )
    negative_groups_failed = sum(
        row["label"] == "CLEAR_NEGATIVE" and not bool(row["group_gate_passed"])
        for row in group_results
    )
    accepted_targets = [
        row for row in rows if row["frame_sequence"] in target_groups and row["accepted"]
    ]
    correct_pair_fraction = (
        float(np.mean([row["selected_pair_correct"] for row in accepted_targets]))
        if accepted_targets
        else 0.0
    )
    gate = protocol["final_gate"]
    passed = bool(
        target_groups_passed >= int(gate["minimum_target_waypoint_groups_passed"])
        and negative_groups_failed <= int(gate["maximum_clear_negative_waypoint_groups_failed"])
        and correct_pair_fraction >= float(gate["minimum_accepted_target_pair_precision"])
    )
    result = {
        "schema": "s0010-sap-tp-pair-development-result-v1",
        "verdict": "DEVELOPMENT_SIGNAL_SUPPORTED" if passed else "DEVELOPMENT_SIGNAL_NOT_SUPPORTED",
        "opened_development_only": True,
        "training_or_independent_confirmation_not_implied": True,
        "segment_config": asdict(config),
        "frame_count": len(rows),
        "target_waypoint_groups_passed": target_groups_passed,
        "target_waypoint_group_count": len(target_groups),
        "clear_negative_waypoint_groups_failed": negative_groups_failed,
        "clear_negative_waypoint_group_count": len(clear_negative_groups),
        "ambiguous_waypoint_group_count": len(ambiguous_groups),
        "accepted_target_frame_count": len(accepted_targets),
        "accepted_target_pair_precision": correct_pair_fraction,
        "group_results": group_results,
    }
    serializable_rows: list[dict[str, Any]] = []
    for row in rows:
        serializable_rows.append(
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "pairs",
                    "selected_pair",
                    "training_feature_mean",
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
    write_csv(args.output_dir / "waypoint_group_results.csv", group_results)
    save_figure(args.output_dir / "actor_pair_selection.png", rows, images_by_id)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
