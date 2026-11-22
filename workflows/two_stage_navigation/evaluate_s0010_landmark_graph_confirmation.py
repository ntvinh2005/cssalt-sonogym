#!/usr/bin/env python3
"""Evaluate the frozen L5 multiview landmark graph on fresh sessions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

import correct_s0010_sap_observability_v1 as sap_correction
import evaluate_s0010_sap_tp_pair_confirmation as pair_evaluation
import run_s0010_l4_medial_branch_target as target_workflow
import run_s0010_multiview_landmark_graph_development as graph
from two_stage_navigation.acquisition_integrity import validate_actor_6dof_record
from two_stage_navigation.stage2_tp_active import positive_probability
from two_stage_navigation.tp_surface_core_dataset import actor_visible_patch_proposals
from two_stage_navigation.ultrasound_spatial_map import (
    calibrated_pose_is_distinct,
    SpatialMapConfig,
)


PROTOCOL_ID = "sonogym-s0010-l5-landmark-graph-confirmation-v1r4"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def diverse_support(
    neighbors: list[dict[str, Any]], kind: str, config: SpatialMapConfig
) -> int:
    selected: list[dict[str, Any]] = []
    frames: dict[int, dict[str, Any]] = {}
    for candidate in sorted(
        neighbors, key=lambda item: item["graph_score"], reverse=True
    ):
        frames.setdefault(int(candidate[f"{kind}_frame"]), candidate)
    for candidate in frames.values():
        if all(
            calibrated_pose_is_distinct(
                candidate[f"{kind}_pose_position_m"],
                candidate[f"{kind}_pose_rotation"],
                accepted[f"{kind}_pose_position_m"],
                accepted[f"{kind}_pose_rotation"],
                config,
            )
            for accepted in selected
        ):
            selected.append(candidate)
    return len(selected)


def consensus_ranking(
    candidates: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    radius = float(config["cluster_radius_mm"])
    limited = sorted(
        candidates, key=lambda row: row["graph_score"], reverse=True
    )[: int(config["cluster_candidate_limit"])]
    distinct = SpatialMapConfig(
        minimum_view_translation_mm=float(config["minimum_view_translation_mm"]),
        minimum_view_angle_deg=float(config["minimum_view_angle_deg"]),
    )
    ranked: list[dict[str, Any]] = []
    for candidate in limited:
        neighbors = [
            row
            for row in limited
            if np.linalg.norm(
                candidate["tp_coordinate_mm"] - row["tp_coordinate_mm"]
            )
            <= radius
            and np.linalg.norm(
                candidate["sap_coordinate_mm"] - row["sap_coordinate_mm"]
            )
            <= radius
        ]
        support = min(
            diverse_support(neighbors, "tp", distinct),
            diverse_support(neighbors, "sap", distinct),
        )
        strongest = sorted(
            (row["graph_score"] for row in neighbors), reverse=True
        )[:10]
        consensus = float(np.mean(strongest)) if strongest else 0.0
        ranked.append(
            {
                "candidate": candidate,
                "independent_support": support,
                "consensus_score": consensus,
            }
        )
    ranked.sort(
        key=lambda row: (
            row["independent_support"],
            row["consensus_score"],
            row["candidate"]["graph_score"],
        ),
        reverse=True,
    )
    return ranked


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
    if protocol["protocol_id"] != PROTOCOL_ID:
        raise ValueError("unexpected protocol identity")
    checkpoint = Path(protocol["source"]["graph_checkpoint"])
    if graph.sha256(checkpoint) != protocol["source"]["graph_checkpoint_sha256"]:
        raise ValueError("landmark-graph checkpoint hash changed")
    graph.register_tp_pickle_module()
    bundle = joblib.load(checkpoint)
    model = bundle["estimator"]
    tp_bundle = joblib.load(bundle["tp_checkpoint"])
    sap_bundle = joblib.load(bundle["sap_checkpoint"])
    if graph.sha256(Path(bundle["tp_checkpoint"])) != bundle["tp_checkpoint_sha256"]:
        raise ValueError("TP checkpoint hash changed")
    if graph.sha256(Path(bundle["sap_checkpoint"])) != bundle["sap_checkpoint_sha256"]:
        raise ValueError("SAP checkpoint hash changed")
    tp_model = tp_bundle["estimator"]
    sap_model = sap_bundle["estimator"]
    patch_spec = graph.patch_spec(tp_bundle["patch_spec"])
    if patch_spec != graph.patch_spec(sap_bundle["patch_spec"]):
        raise ValueError("landmark checkpoints use different patch contracts")
    calibration = protocol["public_probe_calibration"]
    graph_config = protocol["landmark_graph"]
    distinct = SpatialMapConfig(
        minimum_view_translation_mm=float(
            graph_config["minimum_view_translation_mm"]
        ),
        minimum_view_angle_deg=float(graph_config["minimum_view_angle_deg"]),
    )

    public_landmarks: dict[str, list[dict[str, Any]]] = {}
    frame_rows: list[dict[str, Any]] = []
    all_capture_verification_passed = True
    for session in protocol["sessions"]:
        session_id = session["session_id"]
        root = args.run_root / "sessions" / session_id
        replay = json.loads(
            (root / "verification_report.json").read_text(encoding="utf-8")
        )
        actor = json.loads(
            (root / "actor_6dof_validation.json").read_text(encoding="utf-8")
        )
        verified = bool(
            replay["verdict"] == "PROJECTION_PARITY_PASS"
            and all(bool(value) for value in replay["gates"].values())
            and replay["passed_frame_count"] == replay["frame_count"]
            and replay["protocol_sha256"] == graph.sha256(args.protocol)
            and actor["passed"]
            and actor["protocol_sha256"] == graph.sha256(args.protocol)
        )
        all_capture_verification_passed &= verified
        frames = read_jsonl(root / "public/frames.jsonl")
        images = np.load(root / "public/ultrasound.npz")["ultrasound"]
        if len(frames) != len(images):
            raise ValueError("public frame arrays differ")
        session_landmarks: list[dict[str, Any]] = []
        for frame_sequence, (frame, image) in enumerate(zip(frames, images)):
            validate_actor_6dof_record(frame)
            proposals = actor_visible_patch_proposals(
                image, patch_spec=patch_spec, column_stride_pixels=12
            )
            features = np.stack([proposal.feature for proposal in proposals])
            scores = {
                "TP": positive_probability(tp_model, features, 2),
                "SAP": positive_probability(sap_model, features, 1),
            }
            records_by_kind: dict[str, list[dict[str, Any]]] = {}
            for kind in ("TP", "SAP"):
                records = []
                for rank, proposal_index in enumerate(
                    np.argsort(-scores[kind])[: int(graph_config["top_k_per_landmark"])]
                ):
                    proposal = proposals[int(proposal_index)]
                    records.append(
                        {
                            "kind": kind,
                            "rank": rank + 1,
                            "session_id": session_id,
                            "frame_sequence": frame_sequence,
                            "frame_sha256": frame["frame_sha256"],
                            "frame": frame,
                            "proposal": proposal,
                            "score": float(scores[kind][proposal_index]),
                            "coordinate_mm": graph.proposal_coordinate(
                                proposal, frame, calibration
                            ),
                        }
                    )
                records_by_kind[kind] = records
            session_landmarks.append(records_by_kind)
            frame_rows.append(
                {
                    "session_id": session_id,
                    "frame_sequence": frame_sequence,
                    "frame_sha256": frame["frame_sha256"],
                    "split": session["split"],
                    "capture_verified": verified,
                }
            )
        public_landmarks[session_id] = session_landmarks

    candidates_by_session: dict[str, list[dict[str, Any]]] = {}
    for session_id, landmarks in public_landmarks.items():
        candidates: list[dict[str, Any]] = []
        for tp_frame in landmarks:
            for sap_frame in landmarks:
                for tp in tp_frame["TP"]:
                    for sap in sap_frame["SAP"]:
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
                            distinct,
                        ):
                            continue
                        feature = graph.candidate_feature(tp, sap)
                        candidates.append(
                            {
                                "session_id": session_id,
                                "tp_frame": tp["frame_sequence"],
                                "sap_frame": sap["frame_sequence"],
                                "tp_rank": tp["rank"],
                                "sap_rank": sap["rank"],
                                "tp_coordinate_mm": tp["coordinate_mm"],
                                "sap_coordinate_mm": sap["coordinate_mm"],
                                "tp_pose_position_m": np.asarray(
                                    tp["frame"]["actor_base_to_ee_position_m"]
                                ),
                                "tp_pose_rotation": np.asarray(
                                    tp["frame"]["actor_base_to_ee_rotation_columns"]
                                ),
                                "sap_pose_position_m": np.asarray(
                                    sap["frame"]["actor_base_to_ee_position_m"]
                                ),
                                "sap_pose_rotation": np.asarray(
                                    sap["frame"]["actor_base_to_ee_rotation_columns"]
                                ),
                                "feature": feature,
                            }
                        )
        probability = model.predict_proba(
            np.asarray([row["feature"] for row in candidates])
        )
        positive_column = list(model.classes_).index(1)
        for row, score in zip(candidates, probability[:, positive_column]):
            row["graph_score"] = float(score)
        candidates_by_session[session_id] = candidates

    actor_rankings = {
        session_id: consensus_ranking(candidates, graph_config)
        for session_id, candidates in candidates_by_session.items()
    }
    actor_decisions = {}
    for session_id, ranked in actor_rankings.items():
        top = ranked[0] if ranked else None
        accepted = bool(
            top
            and top["independent_support"]
            >= int(graph_config["minimum_independent_support"])
            and top["consensus_score"]
            >= float(graph_config["minimum_consensus_score"])
        )
        actor_decisions[session_id] = {"accepted": accepted, "top": top}

    # Evaluator-only geometry is applied only after actor rankings are frozen.
    _, sap_mask, tp_mask, contract = target_workflow.load_target_geometry(
        args.prediction, args.segmentation
    )
    bone_ids = sap_correction.bone_source_ids(args.label_conversion)
    actual_by_frame: dict[tuple[str, int], str] = {}
    correctness: dict[tuple[str, int, str, int], bool] = {}
    evaluator_rows: list[dict[str, Any]] = []
    for session in protocol["sessions"]:
        session_id = session["session_id"]
        root = args.run_root / "sessions" / session_id
        coordinates = np.load(root / "privileged/pixel_to_volume_coordinates.npz")[
            "human_img_coords"
        ]
        raw_labels = np.load(root / "privileged/raw_labels.npz")["raw_labels"]
        for frame_sequence, (coordinate_map, raw) in enumerate(
            zip(coordinates, raw_labels)
        ):
            projected = target_workflow.projected_masks(
                coordinate_map, contract, sap_mask, tp_mask
            )
            first_bone = target_workflow.first_bone_mask(raw, bone_ids)
            visible = {
                "TP": projected["tp"] & first_bone,
                "SAP": projected["sap"] & first_bone,
            }
            label, reason = target_workflow.target_evidence_label(projected, first_bone)
            margin = pair_evaluation.target_margin(projected)
            geometry_target = bool(
                np.count_nonzero(visible["SAP"]) >= 3
                and np.count_nonzero(visible["TP"]) >= 3
                and margin >= 4
            )
            actual = (
                "TARGET"
                if geometry_target
                else "CLEAR_NEGATIVE"
                if label == "PROXY_NEGATIVE"
                and reason
                in {"TP_ONLY_WITHOUT_SAP_IN_PLANE", "OTHER_BONE_OR_BACKGROUND"}
                else "AMBIGUOUS"
            )
            actual_by_frame[(session_id, frame_sequence)] = actual
            records = public_landmarks[session_id][frame_sequence]
            for kind in ("TP", "SAP"):
                for record in records[kind]:
                    correct = bool(
                        graph.proposal_distance(record["proposal"], visible[kind]) <= 4.0
                    )
                    correctness[(session_id, frame_sequence, kind, record["rank"])] = correct
            evaluator_rows.append(
                {
                    "session_id": session_id,
                    "frame_sequence": frame_sequence,
                    "actual_state": actual,
                    "evaluator_reason": reason,
                    "tp_visible_pixels": int(np.count_nonzero(visible["TP"])),
                    "sap_visible_pixels": int(np.count_nonzero(visible["SAP"])),
                    "target_image_margin_pixels": margin,
                }
            )

    session_rows: list[dict[str, Any]] = []
    for session in protocol["sessions"]:
        session_id = session["session_id"]
        ranked = actor_rankings[session_id]
        for item in ranked:
            candidate = item["candidate"]
            candidate["correct"] = bool(
                correctness[
                    (session_id, candidate["tp_frame"], "TP", candidate["tp_rank"])
                ]
                and correctness[
                    (session_id, candidate["sap_frame"], "SAP", candidate["sap_rank"])
                ]
            )
        decision = actor_decisions[session_id]
        top = decision["top"]
        top_correct = bool(top and top["candidate"]["correct"])
        target_count = sum(
            actual_by_frame[(session_id, index)] == "TARGET"
            for index in range(session["expected_frame_count"])
        )
        negative_count = sum(
            actual_by_frame[(session_id, index)] == "CLEAR_NEGATIVE"
            for index in range(session["expected_frame_count"])
        )
        session_rows.append(
            {
                "session_id": session_id,
                "split": session["split"],
                "candidate_count": len(candidates_by_session[session_id]),
                "actual_target_frames": target_count,
                "actual_clear_negative_frames": negative_count,
                "accepted": decision["accepted"],
                "top1_correct": top_correct,
                "top3_contains_correct": bool(
                    any(item["candidate"]["correct"] for item in ranked[:3])
                ),
                "top1_independent_support": int(top["independent_support"])
                if top
                else 0,
                "top1_consensus_score": float(top["consensus_score"]) if top else 0.0,
                "accepted_correct": bool(decision["accepted"] and top_correct),
                "wrong_accepted": bool(decision["accepted"] and not top_correct),
            }
        )

    target_rows = [row for row in session_rows if "target" in row["split"]]
    negative_rows = [row for row in session_rows if "negative" in row["split"]]
    accepted = [row for row in session_rows if row["accepted"]]
    correct_accepted = sum(row["accepted_correct"] for row in accepted)
    pair_precision = correct_accepted / len(accepted) if accepted else 0.0
    clear_negative_far = (
        sum(row["accepted"] for row in negative_rows) / len(negative_rows)
        if negative_rows
        else 1.0
    )
    gate = protocol["final_gate"]
    sufficient = bool(
        all(
            row["actual_target_frames"]
            >= int(gate["minimum_actual_targets_per_target_session"])
            and row["actual_clear_negative_frames"]
            >= int(gate["minimum_actual_clear_negatives_per_target_session"])
            for row in target_rows
        )
        and all(
            row["actual_clear_negative_frames"]
            >= int(gate["minimum_actual_clear_negatives_per_negative_session"])
            for row in negative_rows
        )
    )
    top1 = sum(row["top1_correct"] for row in target_rows)
    top3 = sum(row["top3_contains_correct"] for row in target_rows)
    accepted_correct_sessions = sum(row["accepted_correct"] for row in target_rows)
    wrong_accepted_sessions = sum(row["wrong_accepted"] for row in session_rows)
    passed = bool(
        all_capture_verification_passed
        and sufficient
        and top1 >= int(gate["minimum_top1_correct_sessions"])
        and top3 >= int(gate["minimum_top3_correct_sessions"])
        and accepted_correct_sessions
        >= int(gate["minimum_accepted_correct_sessions"])
        and wrong_accepted_sessions <= int(gate["maximum_wrong_accepted_sessions"])
        and pair_precision >= float(gate["minimum_pair_precision"])
        and clear_negative_far
        <= float(gate["maximum_clear_negative_false_accept_rate"])
    )
    result = {
        "schema": "s0010-l5-landmark-graph-confirmation-result-v1",
        "passed": passed,
        "protocol_sha256": graph.sha256(args.protocol),
        "graph_checkpoint_sha256": graph.sha256(checkpoint),
        "all_capture_verification_passed": all_capture_verification_passed,
        "sample_sufficiency_passed": sufficient,
        "target_session_count": len(target_rows),
        "negative_session_count": len(negative_rows),
        "top1_correct_target_sessions": top1,
        "top3_correct_target_sessions": top3,
        "accepted_correct_target_sessions": accepted_correct_sessions,
        "wrong_accepted_sessions": wrong_accepted_sessions,
        "pair_precision": pair_precision,
        "clear_negative_false_accept_rate": clear_negative_far,
        "actor_inputs": ["clean_ultrasound", "public_robot_base_to_ee_pose"],
        "privileged_geometry_used_by_actor": False,
        "robot_navigation_executed": False,
        "independent_confirmation": True,
        "junction_or_nerve_endpoint_implied": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "session_results.csv", session_rows)
    write_csv(args.output_dir / "frame_evaluator_states.csv", evaluator_rows)
    write_csv(args.output_dir / "public_frame_index.csv", frame_rows)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
