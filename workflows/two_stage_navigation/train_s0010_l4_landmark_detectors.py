#!/usr/bin/env python3
"""Train session-separated L4 TP and SAP proposal rankers on opened captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import yaml

import correct_s0010_sap_observability_v1 as sap_correction
import run_s0010_l4_medial_branch_target as target_workflow
import run_s0010_multiview_landmark_graph_development as graph
from two_stage_navigation.stage2_tp_active import positive_probability
from two_stage_navigation.tp_surface_core_dataset import actor_visible_patch_proposals


CLEAR_NEGATIVE_REASONS = {
    "TP_ONLY_WITHOUT_SAP_IN_PLANE",
    "OTHER_BONE_OR_BACKGROUND",
}


def proposal_training_state(
    distance_pixels: float,
    *,
    visible_pixels: int,
    frame_is_clear_negative: bool,
    positive_distance_pixels: float = 4.0,
    negative_distance_pixels: float = 12.0,
) -> str:
    """Return a tri-state offline label without turning ambiguity into negative."""

    if visible_pixels >= 3:
        if distance_pixels <= positive_distance_pixels:
            return "POSITIVE"
        if distance_pixels >= negative_distance_pixels:
            return "NEGATIVE"
        return "AMBIGUOUS"
    return "NEGATIVE" if frame_is_clear_negative else "AMBIGUOUS"


def fit_detector(
    rows: list[dict[str, Any]], *, heldout_session: str | None, seed: int
) -> ExtraTreesClassifier:
    training = [
        row
        for row in rows
        if row["state"] != "AMBIGUOUS"
        and (heldout_session is None or row["session_id"] != heldout_session)
    ]
    positives = [row for row in training if row["state"] == "POSITIVE"]
    negatives = sorted(
        (row for row in training if row["state"] == "NEGATIVE"),
        key=lambda row: (-row["baseline_score"], row["sample_id"]),
    )[: max(40, 8 * len(positives))]
    if not positives or not negatives:
        raise ValueError("detector fold lacks a trainable class")
    selected = positives + negatives
    model = ExtraTreesClassifier(
        n_estimators=240,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
    )
    model.fit(
        np.asarray([row["feature"] for row in selected]),
        np.asarray([row["state"] == "POSITIVE" for row in selected], dtype=np.int64),
    )
    return model


def rank_frames(
    rows: list[dict[str, Any]], scores: dict[str, float], kind: str
) -> dict[str, Any]:
    frame_results = []
    keys = sorted({(row["session_id"], row["frame_sequence"]) for row in rows})
    for session_id, frame_sequence in keys:
        frame = [
            row
            for row in rows
            if row["session_id"] == session_id
            and row["frame_sequence"] == frame_sequence
        ]
        ranked = sorted(frame, key=lambda row: scores[row["sample_id"]], reverse=True)
        positive_frame = any(row["state"] == "POSITIVE" for row in frame)
        clear_negative_frame = all(
            row["frame_state"] == "CLEAR_NEGATIVE" for row in frame
        )
        frame_results.append(
            {
                "session_id": session_id,
                "frame_sequence": frame_sequence,
                "positive_frame": positive_frame,
                "clear_negative_frame": clear_negative_frame,
                "top1_correct": bool(
                    positive_frame and ranked and ranked[0]["state"] == "POSITIVE"
                ),
                "top3_contains_correct": bool(
                    positive_frame
                    and any(row["state"] == "POSITIVE" for row in ranked[:3])
                ),
                "top5_contains_correct": bool(
                    positive_frame
                    and any(row["state"] == "POSITIVE" for row in ranked[:5])
                ),
                "maximum_score": float(scores[ranked[0]["sample_id"]]),
            }
        )
    positives = [row for row in frame_results if row["positive_frame"]]
    negatives = [row for row in frame_results if row["clear_negative_frame"]]
    return {
        "kind": kind,
        "frame_count": len(frame_results),
        "positive_frame_count": len(positives),
        "clear_negative_frame_count": len(negatives),
        "top1_correct_positive_frames": sum(row["top1_correct"] for row in positives),
        "top3_correct_positive_frames": sum(
            row["top3_contains_correct"] for row in positives
        ),
        "top5_correct_positive_frames": sum(
            row["top5_contains_correct"] for row in positives
        ),
        "per_session": {
            session_id: {
                "positive_frames": sum(
                    row["positive_frame"]
                    for row in frame_results
                    if row["session_id"] == session_id
                ),
                "top1_correct": sum(
                    row["top1_correct"]
                    for row in frame_results
                    if row["session_id"] == session_id
                ),
                "top3_correct": sum(
                    row["top3_contains_correct"]
                    for row in frame_results
                    if row["session_id"] == session_id
                ),
                "top5_correct": sum(
                    row["top5_contains_correct"]
                    for row in frame_results
                    if row["session_id"] == session_id
                ),
                "clear_negative_frames": sum(
                    row["clear_negative_frame"]
                    for row in frame_results
                    if row["session_id"] == session_id
                ),
            }
            for session_id in sorted({row["session_id"] for row in frame_results})
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--tp-baseline", type=Path, required=True)
    parser.add_argument("--sap-baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    graph.register_tp_pickle_module()
    tp_bundle = joblib.load(args.tp_baseline)
    sap_bundle = joblib.load(args.sap_baseline)
    patch_spec = graph.patch_spec(tp_bundle["patch_spec"])
    if patch_spec != graph.patch_spec(sap_bundle["patch_spec"]):
        raise ValueError("baseline landmark checkpoints use different patch contracts")
    baseline_models = {
        "TP": (tp_bundle["estimator"], 1),
        "SAP": (sap_bundle["estimator"], 1),
    }
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    _, sap_mask, tp_mask, contract = target_workflow.load_target_geometry(
        args.prediction, args.segmentation, level_label=28, level_name="L4"
    )
    bone_ids = sap_correction.bone_source_ids(args.label_conversion)
    rows_by_kind: dict[str, list[dict[str, Any]]] = {"TP": [], "SAP": []}
    frame_count = 0
    for session in protocol["sessions"]:
        session_id = session["session_id"]
        root = args.run_root / "sessions" / session_id
        frames = graph.read_jsonl(root / "public/frames.jsonl")
        images = np.load(root / "public/ultrasound.npz")["ultrasound"]
        coordinates = np.load(root / "privileged/pixel_to_volume_coordinates.npz")[
            "human_img_coords"
        ]
        raw_labels = np.load(root / "privileged/raw_labels.npz")["raw_labels"]
        for frame_sequence, (frame, image, coordinate_map, raw) in enumerate(
            zip(frames, images, coordinates, raw_labels)
        ):
            frame_count += 1
            proposals = actor_visible_patch_proposals(
                image, patch_spec=patch_spec, column_stride_pixels=12
            )
            features = np.stack([proposal.feature for proposal in proposals])
            projected = target_workflow.projected_masks(
                coordinate_map, contract, sap_mask, tp_mask
            )
            first_bone = target_workflow.first_bone_mask(raw, bone_ids)
            label, reason = target_workflow.target_evidence_label(projected, first_bone)
            frame_is_clear_negative = bool(
                label == "PROXY_NEGATIVE" and reason in CLEAR_NEGATIVE_REASONS
            )
            visible = {
                "TP": projected["tp"] & first_bone,
                "SAP": projected["sap"] & first_bone,
            }
            for kind in ("TP", "SAP"):
                model, positive_class = baseline_models[kind]
                baseline_scores = positive_probability(model, features, positive_class)
                visible_pixels = int(np.count_nonzero(visible[kind]))
                for proposal_index, proposal in enumerate(proposals):
                    distance = graph.proposal_distance(proposal, visible[kind])
                    state = proposal_training_state(
                        distance,
                        visible_pixels=visible_pixels,
                        frame_is_clear_negative=frame_is_clear_negative,
                    )
                    rows_by_kind[kind].append(
                        {
                            "sample_id": (
                                f"{session_id}:{frame_sequence}:{kind}:{proposal_index}"
                            ),
                            "session_id": session_id,
                            "frame_sequence": frame_sequence,
                            "frame_sha256": frame["frame_sha256"],
                            "frame_state": (
                                "CLEAR_NEGATIVE"
                                if frame_is_clear_negative
                                else "TARGET_OR_AMBIGUOUS"
                            ),
                            "state": state,
                            "feature": proposal.feature,
                            "baseline_score": float(baseline_scores[proposal_index]),
                        }
                    )

    session_ids = sorted(session["session_id"] for session in protocol["sessions"])
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    results: dict[str, Any] = {}
    for kind, rows in rows_by_kind.items():
        oof_scores: dict[str, float] = {}
        for fold, heldout in enumerate(session_ids):
            model = fit_detector(rows, heldout_session=heldout, seed=2026112300 + fold)
            heldout_rows = [row for row in rows if row["session_id"] == heldout]
            probability = model.predict_proba(
                np.asarray([row["feature"] for row in heldout_rows])
            )
            positive_column = list(model.classes_).index(1)
            oof_scores.update(
                {
                    row["sample_id"]: float(score)
                    for row, score in zip(heldout_rows, probability[:, positive_column])
                }
            )
        metrics = rank_frames(rows, oof_scores, kind)
        final_model = fit_detector(
            rows, heldout_session=None, seed=2026112390 + (kind == "SAP")
        )
        checkpoint = output / f"s0010_l4_{kind.lower()}_proposal_ranker.joblib"
        bundle = {
            "schema": "s0010-l4-landmark-proposal-ranker-v1",
            "kind": kind,
            "estimator": final_model,
            "patch_spec": patch_spec,
            "positive_class": 1,
            "opened_development_session_ids": session_ids,
            "positive_distance_pixels": 4.0,
            "negative_distance_pixels": 12.0,
            "ambiguous_samples_excluded": True,
            "privileged_geometry_used_at_runtime": False,
        }
        joblib.dump(bundle, checkpoint)
        reloaded = joblib.load(checkpoint)
        sample = np.asarray([row["feature"] for row in rows[:100]])
        difference = float(
            np.max(
                np.abs(
                    final_model.predict_proba(sample)
                    - reloaded["estimator"].predict_proba(sample)
                ),
                initial=0.0,
            )
        )
        if difference > 1.0e-15:
            raise RuntimeError(f"{kind} checkpoint reload changed predictions")
        results[kind] = {
            **metrics,
            "positive_patch_count": sum(row["state"] == "POSITIVE" for row in rows),
            "negative_patch_count": sum(row["state"] == "NEGATIVE" for row in rows),
            "ambiguous_patch_count": sum(row["state"] == "AMBIGUOUS" for row in rows),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": graph.sha256(checkpoint),
            "checkpoint_reload_max_abs_difference": difference,
        }
    result = {
        "schema": "s0010-l4-landmark-proposal-ranker-development-v1",
        "opened_development_only": True,
        "frame_count": frame_count,
        "session_count": len(session_ids),
        "session_ids": session_ids,
        "tp_baseline_sha256": graph.sha256(args.tp_baseline),
        "sap_transfer_baseline_sha256": graph.sha256(args.sap_baseline),
        "results": results,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
