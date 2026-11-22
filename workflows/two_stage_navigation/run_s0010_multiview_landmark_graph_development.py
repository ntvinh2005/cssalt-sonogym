#!/usr/bin/env python3
"""Test TP-anchored, cross-view SAP/TP geometry on opened s0010 captures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import ExtraTreesClassifier
import yaml

import correct_s0010_sap_observability_v1 as sap_correction
import run_s0010_l4_medial_branch_target as target_workflow
from two_stage_navigation.stage1_features import PatchSpec
from two_stage_navigation.stage2_tp_active import positive_probability
from two_stage_navigation.tp_surface_core_dataset import actor_visible_patch_proposals
from two_stage_navigation.ultrasound_spatial_map import (
    calibrated_image_coordinates_mm,
    calibrated_pose_is_distinct,
    SpatialMapConfig,
)


FEATURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/stage1_features.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def register_tp_pickle_module() -> None:
    if "tp_proxy_features" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location("tp_proxy_features", FEATURE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot register frozen TP feature module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def patch_spec(value: Any) -> PatchSpec:
    if isinstance(value, PatchSpec):
        return value
    if hasattr(value, "__dict__"):
        value = value.__dict__
    if "patch_width" in value:
        return PatchSpec(
            width_pixels=int(value["patch_width"]),
            depth_pixels=int(value["patch_depth"]),
            interface_offset_pixels=int(value["interface_offset"]),
            feature_size_pixels=16,
        )
    return PatchSpec(**{key: value[key] for key in PatchSpec.__dataclass_fields__})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def proposal_distance(proposal: Any, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.inf
    distance = distance_transform_edt(~mask)
    row = int(np.clip(round(proposal.row), 0, mask.shape[0] - 1))
    column = int(np.clip(round(proposal.column), 0, mask.shape[1] - 1))
    return float(distance[row, column])


def pose_separation(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, float]:
    left_position = np.asarray(left["actor_base_to_ee_position_m"], dtype=np.float64)
    right_position = np.asarray(right["actor_base_to_ee_position_m"], dtype=np.float64)
    left_rotation = np.asarray(
        left["actor_base_to_ee_rotation_columns"], dtype=np.float64
    )
    right_rotation = np.asarray(
        right["actor_base_to_ee_rotation_columns"], dtype=np.float64
    )
    translation = float(np.linalg.norm(left_position - right_position) * 1000.0)
    relative = left_rotation.T @ right_rotation
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return translation, math.degrees(math.acos(cosine))


def proposal_coordinate(
    proposal: Any, frame: dict[str, Any], calibration: dict[str, Any]
) -> np.ndarray:
    rotation = np.asarray(frame["actor_base_to_ee_rotation_columns"], dtype=np.float64)
    coordinates = calibrated_image_coordinates_mm(
        end_effector_position_m=frame["actor_base_to_ee_position_m"],
        rotation_columns=rotation,
        width=int(calibration["image_width_pixels"]),
        depth=int(calibration["image_depth_pixels"]),
        pixel_resolution_mm=float(calibration["pixel_resolution_m"]) * 1000.0,
        image_origin_height_mm=float(calibration["image_origin_height_m"]) * 1000.0,
        home_up_axis=rotation[:, 2],
    )
    row = int(np.clip(round(proposal.row), 0, coordinates.shape[0] - 1))
    column = int(np.clip(round(proposal.column), 0, coordinates.shape[1] - 1))
    return coordinates[row, column]


def candidate_feature(tp: dict[str, Any], sap: dict[str, Any]) -> np.ndarray:
    vector = sap["coordinate_mm"] - tp["coordinate_mm"]
    translation, angle = pose_separation(tp["frame"], sap["frame"])
    return np.asarray(
        [
            tp["score"],
            sap["score"],
            *vector,
            np.linalg.norm(vector),
            translation,
            angle,
            tp["proposal"].row,
            tp["proposal"].column,
            sap["proposal"].row,
            sap["proposal"].column,
        ],
        dtype=np.float64,
    )


def fit_fold(rows: list[dict[str, Any]], heldout: str, seed: int) -> ExtraTreesClassifier:
    training = [row for row in rows if row["session_id"] != heldout]
    positives = [row for row in training if row["correct"]]
    negatives = sorted(
        [row for row in training if not row["correct"]],
        key=lambda row: row["image_score"],
        reverse=True,
    )[: max(20, 10 * len(positives))]
    selected = positives + negatives
    labels = np.asarray([row["correct"] for row in selected], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("cross-validation fold lacks one class")
    model = ExtraTreesClassifier(
        n_estimators=240,
        max_depth=7,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
    )
    model.fit(np.asarray([row["feature"] for row in selected]), labels)
    return model


def ranking_metrics(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    sessions = []
    for session_id in sorted({row["session_id"] for row in rows}):
        ranked = sorted(
            [row for row in rows if row["session_id"] == session_id],
            key=lambda row: row[score_key],
            reverse=True,
        )
        sessions.append(
            {
                "session_id": session_id,
                "candidate_count": len(ranked),
                "correct_candidate_count": sum(row["correct"] for row in ranked),
                "top1_correct": bool(ranked and ranked[0]["correct"]),
                "top3_contains_correct": bool(any(row["correct"] for row in ranked[:3])),
                "top1_score": float(ranked[0][score_key]) if ranked else 0.0,
            }
        )
    return {
        "top1_correct_sessions": sum(row["top1_correct"] for row in sessions),
        "top3_correct_sessions": sum(row["top3_contains_correct"] for row in sessions),
        "sessions": sessions,
    }


def shuffled_fold_candidates(
    rows: list[dict[str, Any]],
    session_id: str,
    model: ExtraTreesClassifier,
    *,
    shuffle: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Return a deterministic held-out input-shuffle control.

    Labels are never used to construct the permutation.  The control preserves
    either image evidence or public-pose geometry while deliberately breaking
    its correspondence with the other input family.
    """

    heldout = [row for row in rows if row["session_id"] == session_id]
    if shuffle not in {"geometry", "appearance"}:
        raise ValueError(f"unsupported shuffle control: {shuffle}")
    permutation = np.random.default_rng(seed).permutation(len(heldout))
    controlled: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    for index, row in enumerate(heldout):
        donor = heldout[int(permutation[index])]
        value = dict(row)
        feature = np.asarray(row["feature"], dtype=np.float64).copy()
        if shuffle == "geometry":
            feature[2:8] = donor["feature"][2:8]
            value["tp_coordinate_mm"] = donor["tp_coordinate_mm"]
            value["sap_coordinate_mm"] = donor["sap_coordinate_mm"]
        else:
            feature[[0, 1, 8, 9, 10, 11]] = donor["feature"][
                [0, 1, 8, 9, 10, 11]
            ]
        value["feature"] = feature
        controlled.append(value)
        features.append(feature)
    probability = model.predict_proba(np.asarray(features))
    positive_column = list(model.classes_).index(1)
    for row, score in zip(controlled, probability[:, positive_column]):
        row["graph_score"] = float(score)
    return controlled


def cluster_ranking_metrics(
    rows: list[dict[str, Any]], *, radius_mm: float = 3.0, candidate_limit: int = 250
) -> dict[str, Any]:
    distinct_config = SpatialMapConfig(
        minimum_view_translation_mm=2.0, minimum_view_angle_deg=1.0
    )

    def diverse_support(neighbors: list[dict[str, Any]], kind: str) -> int:
        selected: list[dict[str, Any]] = []
        frames: dict[int, dict[str, Any]] = {}
        for candidate in sorted(
            neighbors, key=lambda item: item["graph_score"], reverse=True
        ):
            frames.setdefault(int(candidate[f"{kind}_frame"]), candidate)
        for candidate in frames.values():
            position = candidate[f"{kind}_pose_position_m"]
            rotation = candidate[f"{kind}_pose_rotation"]
            if all(
                calibrated_pose_is_distinct(
                    position,
                    rotation,
                    accepted[f"{kind}_pose_position_m"],
                    accepted[f"{kind}_pose_rotation"],
                    distinct_config,
                )
                for accepted in selected
            ):
                selected.append(candidate)
        return len(selected)

    sessions = []
    for session_id in sorted({row["session_id"] for row in rows}):
        session = sorted(
            [row for row in rows if row["session_id"] == session_id],
            key=lambda row: row["graph_score"],
            reverse=True,
        )[:candidate_limit]
        ranked = []
        for row in session:
            neighbors = [
                candidate
                for candidate in session
                if np.linalg.norm(
                    row["tp_coordinate_mm"] - candidate["tp_coordinate_mm"]
                )
                <= radius_mm
                and np.linalg.norm(
                    row["sap_coordinate_mm"] - candidate["sap_coordinate_mm"]
                )
                <= radius_mm
            ]
            independent_support = min(
                diverse_support(neighbors, "tp"), diverse_support(neighbors, "sap")
            )
            strongest = sorted(
                (candidate["graph_score"] for candidate in neighbors), reverse=True
            )[:10]
            consensus_score = float(np.mean(strongest)) if strongest else 0.0
            ranked.append(
                (
                    (independent_support, consensus_score, row["graph_score"]),
                    row,
                )
            )
        ranked.sort(key=lambda value: value[0], reverse=True)
        sessions.append(
            {
                "session_id": session_id,
                "top1_correct": bool(ranked and ranked[0][1]["correct"]),
                "top3_contains_correct": bool(
                    any(value[1]["correct"] for value in ranked[:3])
                ),
                "top1_independent_support": int(ranked[0][0][0]) if ranked else 0,
                "top1_consensus_score": float(ranked[0][0][1]) if ranked else 0.0,
            }
        )
    return {
        "radius_mm": radius_mm,
        "candidate_limit": candidate_limit,
        "top1_correct_sessions": sum(row["top1_correct"] for row in sessions),
        "top3_correct_sessions": sum(row["top3_contains_correct"] for row in sessions),
        "sessions": sessions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--tp-checkpoint", type=Path, required=True)
    parser.add_argument("--sap-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    register_tp_pickle_module()
    tp_bundle = joblib.load(args.tp_checkpoint)
    sap_bundle = joblib.load(args.sap_checkpoint)
    tp_model = tp_bundle["estimator"]
    sap_model = sap_bundle["estimator"]
    tp_spec = patch_spec(tp_bundle["patch_spec"])
    sap_spec = patch_spec(sap_bundle["patch_spec"])
    if tp_spec != sap_spec:
        raise ValueError("TP and SAP checkpoints use different patch contracts")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    calibration = protocol["public_probe_calibration"]
    _, sap_mask, tp_mask, contract = target_workflow.load_target_geometry(
        args.prediction, args.segmentation
    )
    bone_ids = sap_correction.bone_source_ids(args.label_conversion)
    frame_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    per_session_landmarks: dict[str, list[dict[str, Any]]] = {}
    distinct_config = SpatialMapConfig(
        minimum_view_translation_mm=2.0, minimum_view_angle_deg=1.0
    )
    for session in protocol["sessions"]:
        session_id = session["session_id"]
        root = args.run_root / "sessions" / session_id
        frames = read_jsonl(root / "public/frames.jsonl")
        images = np.load(root / "public/ultrasound.npz")["ultrasound"]
        coordinates = np.load(root / "privileged/pixel_to_volume_coordinates.npz")[
            "human_img_coords"
        ]
        raw_labels = np.load(root / "privileged/raw_labels.npz")["raw_labels"]
        landmarks = []
        for index, (frame, image, coordinate_map, raw) in enumerate(
            zip(frames, images, coordinates, raw_labels)
        ):
            proposals = actor_visible_patch_proposals(
                image, patch_spec=tp_spec, column_stride_pixels=12
            )
            feature_matrix = np.stack([proposal.feature for proposal in proposals])
            tp_scores = positive_probability(tp_model, feature_matrix, 2)
            sap_scores = positive_probability(sap_model, feature_matrix, 1)
            tp_index = int(np.argmax(tp_scores))
            sap_index = int(np.argmax(sap_scores))
            projected = target_workflow.projected_masks(
                coordinate_map, contract, sap_mask, tp_mask
            )
            first_bone = target_workflow.first_bone_mask(raw, bone_ids)
            visible = {
                "TP": projected["tp"] & first_bone,
                "SAP": projected["sap"] & first_bone,
            }
            proposal_distances = {
                kind: np.asarray(
                    [proposal_distance(proposal, visible[kind]) for proposal in proposals],
                    dtype=np.float64,
                )
                for kind in ("TP", "SAP")
            }
            records_by_kind: dict[str, list[dict[str, Any]]] = {}
            for kind, scores in (("TP", tp_scores), ("SAP", sap_scores)):
                kind_records = []
                for rank, proposal_index in enumerate(np.argsort(-scores)[:5]):
                    proposal = proposals[int(proposal_index)]
                    record = {
                        "kind": kind,
                        "rank": rank + 1,
                        "session_id": session_id,
                        "frame_sequence": index,
                        "frame_sha256": frame["frame_sha256"],
                        "frame": frame,
                        "proposal": proposal,
                        "score": float(scores[proposal_index]),
                        "coordinate_mm": proposal_coordinate(
                            proposal, frame, calibration
                        ),
                        "surface_distance_pixels": float(
                            proposal_distances[kind][proposal_index]
                        ),
                    }
                    record["correct"] = record["surface_distance_pixels"] <= 4.0
                    kind_records.append(record)
                records_by_kind[kind] = kind_records
            records = {kind: values[0] for kind, values in records_by_kind.items()}
            landmarks.append(records_by_kind)
            frame_rows.append(
                {
                    "session_id": session_id,
                    "frame_sequence": index,
                    "frame_sha256": frame["frame_sha256"],
                    "tp_score": records["TP"]["score"],
                    "tp_distance_pixels": records["TP"]["surface_distance_pixels"],
                    "tp_correct": records["TP"]["correct"],
                    "sap_score": records["SAP"]["score"],
                    "sap_distance_pixels": records["SAP"]["surface_distance_pixels"],
                    "sap_correct": records["SAP"]["correct"],
                    "tp_top3_contains_correct": bool(
                        np.any(proposal_distances["TP"][np.argsort(-tp_scores)[:3]] <= 4.0)
                    ),
                    "tp_top5_contains_correct": bool(
                        np.any(proposal_distances["TP"][np.argsort(-tp_scores)[:5]] <= 4.0)
                    ),
                    "sap_top3_contains_correct": bool(
                        np.any(
                            proposal_distances["SAP"][np.argsort(-sap_scores)[:3]] <= 4.0
                        )
                    ),
                    "sap_top5_contains_correct": bool(
                        np.any(
                            proposal_distances["SAP"][np.argsort(-sap_scores)[:5]] <= 4.0
                        )
                    ),
                }
            )
        per_session_landmarks[session_id] = landmarks
        for tp_records in landmarks:
            for sap_records in landmarks:
                for tp in tp_records["TP"]:
                    for sap in sap_records["SAP"]:
                        if tp["frame_sequence"] == sap["frame_sequence"]:
                            continue
                        if not calibrated_pose_is_distinct(
                            tp["frame"]["actor_base_to_ee_position_m"],
                            np.asarray(
                                tp["frame"]["actor_base_to_ee_rotation_columns"]
                            ),
                            sap["frame"]["actor_base_to_ee_position_m"],
                            np.asarray(
                                sap["frame"]["actor_base_to_ee_rotation_columns"]
                            ),
                            distinct_config,
                        ):
                            continue
                        feature = candidate_feature(tp, sap)
                        candidates.append(
                            {
                                "session_id": session_id,
                                "tp_frame": tp["frame_sequence"],
                                "sap_frame": sap["frame_sequence"],
                                "tp_rank": tp["rank"],
                                "sap_rank": sap["rank"],
                                "tp_correct": tp["correct"],
                                "sap_correct": sap["correct"],
                                "correct": bool(tp["correct"] and sap["correct"]),
                                "image_score": float(tp["score"] * sap["score"]),
                                "tp_coordinate_mm": tp["coordinate_mm"],
                                "sap_coordinate_mm": sap["coordinate_mm"],
                                "tp_pose_position_m": np.asarray(
                                    tp["frame"]["actor_base_to_ee_position_m"],
                                    dtype=np.float64,
                                ),
                                "tp_pose_rotation": np.asarray(
                                    tp["frame"]["actor_base_to_ee_rotation_columns"],
                                    dtype=np.float64,
                                ),
                                "sap_pose_position_m": np.asarray(
                                    sap["frame"]["actor_base_to_ee_position_m"],
                                    dtype=np.float64,
                                ),
                                "sap_pose_rotation": np.asarray(
                                    sap["frame"]["actor_base_to_ee_rotation_columns"],
                                    dtype=np.float64,
                                ),
                                "feature": feature,
                            }
                        )

    session_ids = sorted(per_session_landmarks)
    cross_validation_ready = all(
        set(
            int(row["correct"])
            for row in candidates
            if row["session_id"] != heldout
        )
        == {0, 1}
        for heldout in session_ids
    )
    pose_only_candidates = []
    shuffled_geometry_candidates = []
    shuffled_appearance_candidates = []
    checkpoint_path = args.output_dir / "s0010_l5_multiview_landmark_graph.joblib"
    checkpoint_sha256 = None
    if cross_validation_ready:
        for fold, session_id in enumerate(session_ids):
            model = fit_fold(candidates, session_id, 20261122 + fold)
            heldout = [row for row in candidates if row["session_id"] == session_id]
            probability = model.predict_proba(
                np.asarray([row["feature"] for row in heldout])
            )
            positive_column = list(model.classes_).index(1)
            for row, score in zip(heldout, probability[:, positive_column]):
                row["graph_score"] = float(score)
            shuffled_geometry_candidates.extend(
                shuffled_fold_candidates(
                    candidates,
                    session_id,
                    model,
                    shuffle="geometry",
                    seed=20261322 + fold,
                )
            )
            shuffled_appearance_candidates.extend(
                shuffled_fold_candidates(
                    candidates,
                    session_id,
                    model,
                    shuffle="appearance",
                    seed=20261422 + fold,
                )
            )

        for row in candidates:
            value = dict(row)
            value["feature"] = row["feature"][[6, 7]]
            pose_only_candidates.append(value)
        for fold, session_id in enumerate(session_ids):
            model = fit_fold(pose_only_candidates, session_id, 20261222 + fold)
            heldout = [
                row for row in pose_only_candidates if row["session_id"] == session_id
            ]
            probability = model.predict_proba(
                np.asarray([row["feature"] for row in heldout])
            )
            positive_column = list(model.classes_).index(1)
            for row, score in zip(heldout, probability[:, positive_column]):
                row["pose_only_score"] = float(score)
        final_model = fit_fold(candidates, "__no_heldout_session__", 20261122)
        bundle = {
            "schema": "s0010-l5-multiview-landmark-graph-v1",
            "estimator": final_model,
            "tp_checkpoint": str(args.tp_checkpoint.resolve()),
            "tp_checkpoint_sha256": sha256(args.tp_checkpoint),
            "sap_checkpoint": str(args.sap_checkpoint.resolve()),
            "sap_checkpoint_sha256": sha256(args.sap_checkpoint),
            "top_k_per_landmark": 5,
            "cluster_radius_mm": 3.0,
            "cluster_candidate_limit": 250,
            "minimum_independent_support": 2,
            "minimum_consensus_score": 0.75,
            "opened_development_session_ids": session_ids,
        }
        args.output_dir.mkdir(parents=True, exist_ok=False)
        joblib.dump(bundle, checkpoint_path)
        reloaded = joblib.load(checkpoint_path)
        matrix = np.asarray([row["feature"] for row in candidates[:100]])
        before_reload = final_model.predict_proba(matrix)
        after_reload = reloaded["estimator"].predict_proba(matrix)
        reload_max_abs_difference = float(
            np.max(np.abs(before_reload - after_reload), initial=0.0)
        )
        if reload_max_abs_difference > 1.0e-15:
            raise RuntimeError("landmark graph checkpoint reload changed predictions")
        checkpoint_sha256 = sha256(checkpoint_path)

    consensus = (
        cluster_ranking_metrics(candidates) if cross_validation_ready else None
    )
    accepted_development = (
        [
            row
            for row in consensus["sessions"]
            if row["top1_independent_support"] >= 2
            and row["top1_consensus_score"] >= 0.75
        ]
        if consensus is not None
        else []
    )
    result = {
        "schema": "s0010-multiview-landmark-graph-development-v1",
        "opened_development_only": True,
        "tp_checkpoint_sha256": sha256(args.tp_checkpoint),
        "sap_checkpoint_sha256": sha256(args.sap_checkpoint),
        "frame_count": len(frame_rows),
        "tp_top_proposal_correct": sum(row["tp_correct"] for row in frame_rows),
        "sap_top_proposal_correct": sum(row["sap_correct"] for row in frame_rows),
        "tp_top3_contains_correct": sum(
            row["tp_top3_contains_correct"] for row in frame_rows
        ),
        "tp_top5_contains_correct": sum(
            row["tp_top5_contains_correct"] for row in frame_rows
        ),
        "sap_top3_contains_correct": sum(
            row["sap_top3_contains_correct"] for row in frame_rows
        ),
        "sap_top5_contains_correct": sum(
            row["sap_top5_contains_correct"] for row in frame_rows
        ),
        "both_top_proposals_correct": sum(
            row["tp_correct"] and row["sap_correct"] for row in frame_rows
        ),
        "per_session_landmark_counts": {
            session_id: {
                "tp_top_proposal_correct": sum(
                    row["tp_correct"]
                    for row in frame_rows
                    if row["session_id"] == session_id
                ),
                "sap_top_proposal_correct": sum(
                    row["sap_correct"]
                    for row in frame_rows
                    if row["session_id"] == session_id
                ),
                "both_top_proposals_correct": sum(
                    row["tp_correct"] and row["sap_correct"]
                    for row in frame_rows
                    if row["session_id"] == session_id
                ),
            }
            for session_id in session_ids
        },
        "candidate_count": len(candidates),
        "correct_candidate_count": sum(row["correct"] for row in candidates),
        "cross_validation_ready": cross_validation_ready,
        "image_only_ranking": ranking_metrics(candidates, "image_score"),
        "geometry_appearance_ranking": (
            ranking_metrics(candidates, "graph_score")
            if cross_validation_ready
            else None
        ),
        "geometry_consensus_ranking": consensus,
        "frozen_development_acceptance": {
            "minimum_independent_support": 2,
            "minimum_consensus_score": 0.75,
            "accepted_session_count": len(accepted_development),
            "correct_accepted_session_count": sum(
                row["top1_correct"] for row in accepted_development
            ),
            "wrong_accepted_session_count": sum(
                not row["top1_correct"] for row in accepted_development
            ),
            "abstained_session_count": len(session_ids) - len(accepted_development),
        },
        "pose_only_ranking": (
            ranking_metrics(pose_only_candidates, "pose_only_score")
            if cross_validation_ready
            else None
        ),
        "shuffled_geometry_consensus": (
            cluster_ranking_metrics(shuffled_geometry_candidates)
            if cross_validation_ready
            else None
        ),
        "shuffled_appearance_consensus": (
            cluster_ranking_metrics(shuffled_appearance_candidates)
            if cross_validation_ready
            else None
        ),
        "independent_confirmation": False,
        "checkpoint": str(checkpoint_path) if checkpoint_sha256 else None,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_reload_max_abs_difference": (
            reload_max_abs_difference if checkpoint_sha256 else None
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "frame_landmarks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    serializable = [
        {
            key: (
                ";".join(f"{item:.9f}" for item in value)
                if key
                in {
                    "tp_coordinate_mm",
                    "sap_coordinate_mm",
                    "tp_pose_position_m",
                    "sap_pose_position_m",
                    "tp_pose_rotation",
                    "sap_pose_rotation",
                }
                and np.asarray(value).ndim == 1
                else ";".join(
                    f"{item:.9f}" for item in np.asarray(value).reshape(-1)
                )
                if key in {"tp_pose_rotation", "sap_pose_rotation"}
                else value
            )
            for key, value in row.items()
            if key != "feature"
        }
        for row in candidates
    ]
    with (args.output_dir / "pair_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable[0]))
        writer.writeheader()
        writer.writerows(serializable)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
