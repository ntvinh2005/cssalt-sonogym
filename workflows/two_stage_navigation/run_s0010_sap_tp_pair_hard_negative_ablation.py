#!/usr/bin/env python3
"""Test session-separated hard-negative training for the SAP/TP pair selector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesClassifier

import correct_s0010_sap_observability_v1 as sap_correction
import evaluate_s0010_sap_tp_pair_confirmation as confirmation
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


def fit_rows(rows: list[dict[str, Any]], seed: int) -> tuple[Any, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for row in rows:
        if row["actual_state"] == "AMBIGUOUS":
            continue
        positives = [pair for pair in row["pairs"] if pair["positive"]]
        negatives = [pair for pair in row["pairs"] if not pair["positive"]]
        negatives.sort(
            key=lambda pair: pair["first"].ridge_score + pair["second"].ridge_score,
            reverse=True,
        )
        for pair in positives + negatives[:20]:
            features.append(pair["feature"])
            labels.append(int(pair["positive"]))
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[1] != 21 or set(target.tolist()) != {0, 1}:
        raise ValueError("hard-negative training rows do not contain both pair classes")
    model = ExtraTreesClassifier(
        n_estimators=240,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
    )
    model.fit(matrix, target)
    return model, np.maximum(np.std(matrix, axis=0), 1e-6)


def load_rows(
    protocol: dict[str, Any],
    run_root: Path,
    sap_mask: np.ndarray,
    tp_mask: np.ndarray,
    contract: Any,
    bone_ids: set[int],
    pair_gate: dict[str, Any],
    track_config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gate = pair_gate
    for session in protocol["sessions"]:
        session_id = session["session_id"]
        session_dir = run_root / "sessions" / session_id
        frames = confirmation.read_jsonl(session_dir / "public/frames.jsonl")
        images = np.load(session_dir / "public/ultrasound.npz")["ultrasound"]
        coordinates = np.load(
            session_dir / "privileged/pixel_to_volume_coordinates.npz"
        )["human_img_coords"]
        raw_labels = np.load(session_dir / "privileged/raw_labels.npz")["raw_labels"]
        poses = development._cumulative_pose(frames)
        for index, (image, coordinate_map, raw) in enumerate(
            zip(images, coordinates, raw_labels)
        ):
            projected = target_workflow.projected_masks(
                coordinate_map, contract, sap_mask, tp_mask
            )
            first_bone = target_workflow.first_bone_mask(raw, bone_ids)
            sap_visible = projected["sap"] & first_bone
            tp_visible = projected["tp"] & first_bone
            margin = confirmation.target_margin(projected)
            sap_count = int(np.count_nonzero(sap_visible))
            tp_count = int(np.count_nonzero(tp_visible))
            evaluator_label, evaluator_reason = target_workflow.target_evidence_label(
                projected, first_bone
            )
            if (
                sap_count >= int(gate["minimum_sap_first_visible_pixels"])
                and tp_count >= int(gate["minimum_tp_first_visible_pixels"])
                and margin >= int(gate["minimum_target_image_margin_pixels"])
            ):
                actual_state = "TARGET"
            elif evaluator_label == "PROXY_NEGATIVE" and evaluator_reason in {
                "TP_ONLY_WITHOUT_SAP_IN_PLANE",
                "OTHER_BONE_OR_BACKGROUND",
            }:
                actual_state = "CLEAR_NEGATIVE"
            else:
                actual_state = "AMBIGUOUS"
            pairs = [
                development._pair_record(
                    first, second, feature, tp_visible, sap_visible, gate
                )
                for first, second, feature in track_window_pair_candidates(
                    image, **track_config
                )
            ]
            rows.append(
                {
                    "sample_id": f"{session_id}:{index:02d}",
                    "session_id": session_id,
                    "frame_sequence": index,
                    "actual_state": actual_state,
                    "pairs": pairs,
                    "cumulative_pose": poses[index],
                }
            )
    return rows


def score_session(
    rows: list[dict[str, Any]], model: Any, feature_scale: np.ndarray, protocol: dict[str, Any]
) -> dict[str, Any]:
    for row in rows:
        score, selected_index = development._score_row(row, model)
        selected = row["pairs"][selected_index] if selected_index is not None else None
        row["base_score"] = score
        row["selected_pair"] = selected
        row["selected_pair_correct"] = bool(
            row["actual_state"] == "TARGET" and selected is not None and selected["positive"]
        )
        row["training_feature_scale"] = feature_scale
    development._add_prior_view_support(rows, protocol)
    accepted = sorted(
        [row for row in rows if row["accepted"]],
        key=lambda row: row["accumulated_score"],
        reverse=True,
    )
    nonambiguous = [row for row in accepted if row["actual_state"] != "AMBIGUOUS"]
    correct = [row for row in nonambiguous if row["selected_pair_correct"]]
    negatives = [row for row in rows if row["actual_state"] == "CLEAR_NEGATIVE"]
    accepted_negatives = [row for row in negatives if row["accepted"]]
    targets = [row for row in rows if row["actual_state"] == "TARGET"]
    result = {
        "session_id": rows[0]["session_id"],
        "target_count": len(targets),
        "clear_negative_count": len(negatives),
        "ambiguous_count": len(rows) - len(targets) - len(negatives),
        "accepted_count": len(accepted),
        "correct_accepted_target_count": len(correct),
        "pair_precision": len(correct) / max(len(nonambiguous), 1),
        "clear_negative_false_accept_rate": len(accepted_negatives) / max(len(negatives), 1),
        "top1_correct": bool(accepted and accepted[0]["selected_pair_correct"]),
        "top3_contains_correct": bool(
            any(row["selected_pair_correct"] for row in accepted[:3])
        ),
    }
    result["passed"] = bool(
        result["target_count"] >= 4
        and result["clear_negative_count"] >= 4
        and result["correct_accepted_target_count"] >= 2
        and result["pair_precision"] >= 0.75
        and result["clear_negative_false_accept_rate"] <= 0.10
        and result["top1_correct"]
        and result["top3_contains_correct"]
    )
    return result


def frame_feature(row: dict[str, Any], pair_model: Any) -> np.ndarray:
    """Summarize image-only pair evidence without evaluator geometry."""

    if not row["pairs"]:
        return np.zeros(31, dtype=np.float64)
    pair_features = np.asarray(
        [pair["feature"] for pair in row["pairs"]], dtype=np.float64
    )
    probabilities = pair_model.predict_proba(pair_features)[:, 1]
    ordered = np.sort(probabilities)[::-1]
    top = np.pad(ordered[:3], (0, max(0, 3 - len(ordered))))
    summary = np.asarray(
        [
            *top,
            float(np.mean(probabilities)),
            float(np.std(probabilities)),
            float(np.quantile(probabilities, 0.75)),
            float(np.mean(probabilities >= 0.50)),
            float(np.mean(probabilities >= 0.70)),
            float(np.mean(probabilities >= 0.90)),
            float(np.log1p(len(probabilities))),
        ],
        dtype=np.float64,
    )
    return np.concatenate((summary, pair_features[int(np.argmax(probabilities))]))


def fit_rejection_model(
    rows: list[dict[str, Any]], pair_model: Any, seed: int
) -> Any:
    usable = [row for row in rows if row["actual_state"] != "AMBIGUOUS"]
    matrix = np.asarray([frame_feature(row, pair_model) for row in usable])
    target = np.asarray(
        [int(row["actual_state"] == "TARGET") for row in usable], dtype=np.int64
    )
    if matrix.shape[1] != 31 or set(target.tolist()) != {0, 1}:
        raise ValueError("frame rejection training lacks a class")
    model = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
    )
    model.fit(matrix, target)
    return model


def score_session_with_rejection(
    rows: list[dict[str, Any]],
    pair_model: Any,
    feature_scale: np.ndarray,
    rejection_model: Any,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    for row in rows:
        score, selected_index = development._score_row(row, pair_model)
        selected = row["pairs"][selected_index] if selected_index is not None else None
        row["base_score"] = score
        row["selected_pair"] = selected
        row["selected_pair_correct"] = bool(
            row["actual_state"] == "TARGET" and selected is not None and selected["positive"]
        )
        row["training_feature_scale"] = feature_scale
    development._add_prior_view_support(rows, protocol)
    for row in rows:
        probability = float(
            rejection_model.predict_proba(frame_feature(row, pair_model)[None, :])[0, 1]
        )
        row["rejection_probability"] = probability
        row["accepted"] = bool(row["accepted"] and probability >= 0.50)
        row["combined_score"] = float(row["accumulated_score"] * probability)
    accepted = sorted(
        [row for row in rows if row["accepted"]],
        key=lambda row: row["combined_score"],
        reverse=True,
    )
    nonambiguous = [row for row in accepted if row["actual_state"] != "AMBIGUOUS"]
    correct = [row for row in nonambiguous if row["selected_pair_correct"]]
    negatives = [row for row in rows if row["actual_state"] == "CLEAR_NEGATIVE"]
    accepted_negatives = [row for row in negatives if row["accepted"]]
    targets = [row for row in rows if row["actual_state"] == "TARGET"]
    result = {
        "session_id": rows[0]["session_id"],
        "target_count": len(targets),
        "clear_negative_count": len(negatives),
        "ambiguous_count": len(rows) - len(targets) - len(negatives),
        "accepted_count": len(accepted),
        "correct_accepted_target_count": len(correct),
        "pair_precision": len(correct) / max(len(nonambiguous), 1),
        "clear_negative_false_accept_rate": len(accepted_negatives) / max(len(negatives), 1),
        "top1_correct": bool(accepted and accepted[0]["selected_pair_correct"]),
        "top3_contains_correct": bool(
            any(row["selected_pair_correct"] for row in accepted[:3])
        ),
    }
    result["passed"] = bool(
        result["target_count"] >= 4
        and result["clear_negative_count"] >= 4
        and result["correct_accepted_target_count"] >= 2
        and result["pair_precision"] >= 0.75
        and result["clear_negative_false_accept_rate"] <= 0.10
        and result["top1_correct"]
        and result["top3_contains_correct"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--development-protocol", type=Path, required=True)
    parser.add_argument("--development-run", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    training_protocol = yaml.safe_load(args.training_protocol.read_text(encoding="utf-8"))
    development_protocol = yaml.safe_load(
        args.development_protocol.read_text(encoding="utf-8")
    )
    _, sap_mask, tp_mask, contract = target_workflow.load_target_geometry(
        args.prediction, args.segmentation
    )
    bone_ids = sap_correction.bone_source_ids(args.label_conversion)
    training_rows = load_rows(
        training_protocol,
        args.training_run,
        sap_mask,
        tp_mask,
        contract,
        bone_ids,
        development_protocol["pair_label_gate"],
        development_protocol["track_window_config"],
    )
    development_rows = load_rows(
        development_protocol,
        args.development_run,
        sap_mask,
        tp_mask,
        contract,
        bone_ids,
        development_protocol["pair_label_gate"],
        development_protocol["track_window_config"],
    )
    sessions = sorted({row["session_id"] for row in development_rows})
    fold_results = []
    for fold_index, held_out in enumerate(sessions):
        fit = training_rows + [
            row for row in development_rows if row["session_id"] != held_out
        ]
        model, scale = fit_rows(fit, seed=20261130 + fold_index)
        test = [dict(row) for row in development_rows if row["session_id"] == held_out]
        fold_results.append(score_session(test, model, scale, development_protocol))
    hard_negative_passed = all(row["passed"] for row in fold_results)
    base_checkpoint_path = Path(development_protocol["source"]["checkpoint"])
    if sha256(base_checkpoint_path) != development_protocol["source"]["checkpoint_sha256"]:
        raise ValueError("frozen pair checkpoint hash changed")
    base_checkpoint = joblib.load(base_checkpoint_path)
    base_model = base_checkpoint["model"]
    base_scale = np.asarray(base_checkpoint["feature_scale"], dtype=np.float64)
    rejection_results = []
    for fold_index, held_out in enumerate(sessions):
        fit = training_rows + [
            row for row in development_rows if row["session_id"] != held_out
        ]
        rejection = fit_rejection_model(fit, base_model, seed=20261140 + fold_index)
        test = [dict(row) for row in development_rows if row["session_id"] == held_out]
        rejection_results.append(
            score_session_with_rejection(
                test, base_model, base_scale, rejection, development_protocol
            )
        )
    rejection_passed = all(row["passed"] for row in rejection_results)
    result = {
        "schema": "s0010-sap-tp-hard-negative-ablation-v1",
        "passed": rejection_passed,
        "method": "original_training_plus_two_development_sessions_leave_one_session_out",
        "training_frame_count": len(training_rows),
        "opened_development_frame_count": len(development_rows),
        "fold_results": fold_results,
        "hard_negative_passed": hard_negative_passed,
        "frame_rejection_fold_results": rejection_results,
        "frame_rejection_passed": rejection_passed,
        "final_checkpoint_written": rejection_passed,
    }
    if rejection_passed:
        rejection = fit_rejection_model(
            training_rows + development_rows, base_model, seed=20261143
        )
        checkpoint = args.output_dir / "sap_tp_pair_rejection_v2.joblib"
        joblib.dump(
            {
                "pair_model": base_model,
                "pair_feature_scale": base_scale,
                "rejection_model": rejection,
                "pair_threshold": 0.50,
                "rejection_threshold": 0.50,
                "base_checkpoint_sha256": sha256(base_checkpoint_path),
                "training_protocol_sha256": sha256(args.training_protocol),
                "development_protocol_sha256": sha256(args.development_protocol),
                "training_frame_count": len(training_rows) + len(development_rows),
            },
            checkpoint,
        )
        result["checkpoint"] = str(checkpoint)
        result["checkpoint_sha256"] = sha256(checkpoint)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
