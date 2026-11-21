#!/usr/bin/env python3
"""Rank recorded SAP views from interface correspondence across each session."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

from two_stage_navigation.contracts import assert_no_privileged_fields, ultrasound_sha256
from two_stage_navigation.ultrasound_multiview import (
    EvidenceConfig,
    accumulate_evidence,
    correspondence_matrix,
    pose_distance_matrices,
)

import run_s0010_ultrasound_representation_experiment as representation


POSITIVE = representation.POSITIVE
NEGATIVE = representation.NEGATIVE
AMBIGUOUS = "AMBIGUOUS"
ABSTAIN = "ABSTAIN"


@dataclass
class SessionData:
    session_id: str
    frame_sequences: np.ndarray
    frame_hashes: list[str]
    statuses: list[str]
    images: list[np.ndarray]
    poses: np.ndarray
    base_scores: np.ndarray
    correspondences: np.ndarray
    translation_mm: np.ndarray
    angle_deg: np.ndarray


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cumulative_pose_by_frame(public_rows: list[dict[str, Any]]) -> dict[int, np.ndarray]:
    ordered = sorted(public_rows, key=lambda row: int(row["frame_sequence"]))
    pose = np.zeros(3, dtype=np.float64)
    result: dict[int, np.ndarray] = {}
    previous_sequence = -1
    previous_producer = -1
    hashes: set[str] = set()
    for row in ordered:
        assert_no_privileged_fields(row)
        sequence = int(row["frame_sequence"])
        producer = int(row["producer_sequence"])
        if sequence <= previous_sequence or producer <= previous_producer:
            raise ValueError("frame and producer sequences must increase within a session")
        frame_hash = str(row["frame_sha256"])
        if frame_hash in hashes:
            raise ValueError("duplicate image hash cannot create another viewpoint vote")
        delta = np.asarray(row["executed_delta_relative_mm"], dtype=np.float64)
        angle = float(row["executed_delta_angle_rad"])
        if delta.shape != (2,) or not np.all(np.isfinite(delta)) or not math.isfinite(angle):
            raise ValueError("executed motion is missing or nonfinite")
        if not bool(
            row["action_accepted"]
            and row["action_completed"]
            and not row["action_clamped"]
            and row["contact_ok"]
            and row["workspace_ok"]
            and row["image_valid"]
        ):
            raise ValueError("multi-view replay contains an inadmissible actor frame")
        pose[:2] += delta
        pose[2] = (pose[2] + angle + np.pi) % (2.0 * np.pi) - np.pi
        result[sequence] = pose.copy()
        hashes.add(frame_hash)
        previous_sequence = sequence
        previous_producer = producer
    return result


def build_sessions(
    *,
    sample_rows: list[dict[str, str]],
    proposal_rows: list[dict[str, str]],
    level: str,
    split: str,
    bundle: dict[str, Any],
    recorded: representation.RecordedImages,
) -> list[SessionData]:
    clear = {
        row["frame_sha256"]: row["provisional_label"]
        for row in sample_rows
        if row["conditioned_level"] == level and row["split"] == split
    }
    selected = representation.unique_level_rows(proposal_rows, level, split)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        grouped[row["session_id"]].append(row)
    estimator = bundle["estimator"]
    positive_column = list(estimator.classes_).index(1)
    sessions: list[SessionData] = []
    for session_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["frame_sequence"]))
        archives = {Path(row["image_archive"]) for row in ordered}
        if len(archives) != 1:
            raise ValueError(f"session {session_id} spans multiple image archives")
        public_rows = recorded.public_rows(next(iter(archives)))
        pose_lookup = cumulative_pose_by_frame(public_rows)
        images: list[np.ndarray] = []
        poses: list[np.ndarray] = []
        statuses: list[str] = []
        hashes: list[str] = []
        sequences: list[int] = []
        features: list[np.ndarray] = []
        for row in ordered:
            public = recorded.public_row(row)
            image = recorded.image(row)
            if str(public["frame_sha256"]) != row["frame_sha256"]:
                raise ValueError("manifest/public frame hash mismatch")
            if ultrasound_sha256(np.asarray(image, dtype=np.float32)) != row["frame_sha256"]:
                raise ValueError("saved image does not match its frame hash")
            sequence = int(row["frame_sequence"])
            images.append(image)
            poses.append(pose_lookup[sequence])
            statuses.append(clear.get(row["frame_sha256"], AMBIGUOUS))
            hashes.append(row["frame_sha256"])
            sequences.append(sequence)
            features.append(recorded.feature(row, "topology"))
        base_scores = estimator.predict_proba(np.asarray(features))[:, positive_column]
        pose_array = np.asarray(poses, dtype=np.float64)
        translation, angle = pose_distance_matrices(pose_array)
        sessions.append(
            SessionData(
                session_id=session_id,
                frame_sequences=np.asarray(sequences, dtype=np.int64),
                frame_hashes=hashes,
                statuses=statuses,
                images=images,
                poses=pose_array,
                base_scores=np.asarray(base_scores, dtype=np.float64),
                correspondences=correspondence_matrix(images),
                translation_mm=translation,
                angle_deg=angle,
            )
        )
    return sessions


def config_from_protocol(
    row: dict[str, Any], viewpoint: dict[str, Any]
) -> EvidenceConfig:
    return EvidenceConfig(
        aggregation=str(row["aggregation"]),
        maximum_neighbor_translation_mm=float(row["maximum_neighbor_translation_mm"]),
        maximum_neighbor_angle_deg=float(row["maximum_neighbor_angle_deg"]),
        minimum_view_translation_mm=float(viewpoint["minimum_translation_mm"]),
        minimum_view_angle_deg=float(viewpoint["minimum_orientation_deg"]),
        minimum_correspondence=float(row["minimum_correspondence"]),
        maximum_neighbors=int(row["maximum_neighbors"]),
        neighbor_weight=float(row["neighbor_weight"]),
        minimum_support=int(row["minimum_support"]),
    )


def assess_session(
    session: SessionData,
    scores: np.ndarray,
    support_counts: np.ndarray,
    support_indices: list[list[int]],
    *,
    threshold: float,
    margin_threshold: float,
    minimum_support: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eligible = np.flatnonzero(support_counts >= minimum_support)
    if eligible.size:
        ordered = eligible[np.argsort(scores[eligible])[::-1]]
        top = int(ordered[0])
        second_score = float(scores[int(ordered[1])]) if len(ordered) > 1 else 0.0
        margin = float(scores[top] - second_score)
        accepted = bool(scores[top] >= threshold and margin >= margin_threshold)
        top1_status = session.statuses[top]
        top3_contains_positive = any(
            session.statuses[int(index)] == POSITIVE for index in ordered[:3]
        )
    else:
        ordered = np.asarray([], dtype=np.int64)
        top = -1
        margin = 0.0
        accepted = False
        top1_status = ABSTAIN
        top3_contains_positive = False
    selected_status = session.statuses[top] if accepted else ABSTAIN
    positive_present = POSITIVE in session.statuses
    frame_rows: list[dict[str, Any]] = []
    for index, (status, score, support) in enumerate(
        zip(session.statuses, scores, support_counts)
    ):
        activated = bool(score >= threshold and support >= minimum_support)
        frame_rows.append(
            {
                "session_id": session.session_id,
                "frame_sequence": int(session.frame_sequences[index]),
                "frame_sha256": session.frame_hashes[index],
                "status": status,
                "base_score": float(session.base_scores[index]),
                "multiview_score": float(score),
                "support_count": int(support),
                "support_frame_sequences": ";".join(
                    str(int(session.frame_sequences[value]))
                    for value in support_indices[index]
                ),
                "activated": activated,
                "selected": bool(accepted and index == top),
            }
        )
    record = {
        "session_id": session.session_id,
        "images": len(session.statuses),
        "positive_images": session.statuses.count(POSITIVE),
        "negative_images": session.statuses.count(NEGATIVE),
        "ambiguous_images": session.statuses.count(AMBIGUOUS),
        "positive_present": positive_present,
        "eligible_images": int(len(eligible)),
        "top1_status": top1_status,
        "top1_score": None if top < 0 else float(scores[top]),
        "top1_margin": margin,
        "top3_contains_positive": top3_contains_positive,
        "accepted": accepted,
        "selected_status": selected_status,
        "selected_frame_sequence": None if not accepted else int(session.frame_sequences[top]),
        "accepted_positive": bool(accepted and selected_status == POSITIVE),
        "wrong_accepted_positive_session": bool(
            positive_present and accepted and selected_status != POSITIVE
        ),
        "accepted_no_positive_session": bool(not positive_present and accepted),
        "ambiguous_frame_activations": sum(
            row["status"] == AMBIGUOUS and row["activated"] for row in frame_rows
        ),
    }
    return record, frame_rows


def summarize_sessions(
    records: list[dict[str, Any]], frame_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    clear = [row for row in frame_rows if row["status"] != AMBIGUOUS]
    tp = sum(row["status"] == POSITIVE and row["activated"] for row in clear)
    fn = sum(row["status"] == POSITIVE and not row["activated"] for row in clear)
    fp = sum(row["status"] == NEGATIVE and row["activated"] for row in clear)
    tn = sum(row["status"] == NEGATIVE and not row["activated"] for row in clear)
    return {
        "sessions": len(records),
        "positive_sessions": sum(row["positive_present"] for row in records),
        "no_positive_sessions": sum(not row["positive_present"] for row in records),
        "accepted_sessions": sum(row["accepted"] for row in records),
        "abstained_sessions": sum(not row["accepted"] for row in records),
        "accepted_positive_sessions": sum(row["accepted_positive"] for row in records),
        "wrong_accepted_positive_sessions": sum(
            row["wrong_accepted_positive_session"] for row in records
        ),
        "accepted_no_positive_sessions": sum(
            row["accepted_no_positive_session"] for row in records
        ),
        "top1_positive_sessions": sum(row["top1_status"] == POSITIVE for row in records),
        "top3_positive_sessions": sum(row["top3_contains_positive"] for row in records),
        "ambiguous_top1_sessions": sum(row["top1_status"] == AMBIGUOUS for row in records),
        "negative_top1_sessions": sum(row["top1_status"] == NEGATIVE for row in records),
        "ambiguous_frame_activations": sum(
            row["status"] == AMBIGUOUS and row["activated"] for row in frame_rows
        ),
        "clear_frame_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def evaluate_setting(
    sessions: list[SessionData],
    config: EvidenceConfig,
    *,
    threshold: float,
    margin_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for session in sessions:
        scores, support_counts, support_indices = accumulate_evidence(
            session.base_scores,
            session.correspondences,
            session.translation_mm,
            session.angle_deg,
            config,
        )
        record, frames = assess_session(
            session,
            scores,
            support_counts,
            support_indices,
            threshold=threshold,
            margin_threshold=margin_threshold,
            minimum_support=config.minimum_support,
        )
        records.append(record)
        frame_rows.extend(frames)
    return summarize_sessions(records, frame_rows), records, frame_rows


def calibration_gate(summary: dict[str, Any], minimum_fraction: float) -> bool:
    required = math.ceil(summary["positive_sessions"] * minimum_fraction)
    return bool(
        summary["accepted_positive_sessions"] >= required
        and summary["wrong_accepted_positive_sessions"] == 0
        and summary["accepted_no_positive_sessions"] == 0
    )


def select_setting(
    sessions: list[SessionData], protocol: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(protocol["candidates"]):
        config = config_from_protocol(candidate, protocol["viewpoint_contract"])
        for threshold in protocol["selection"]["score_thresholds"]:
            for margin in protocol["selection"]["top1_margin_thresholds"]:
                summary, _, _ = evaluate_setting(
                    sessions,
                    config,
                    threshold=float(threshold),
                    margin_threshold=float(margin),
                )
                summary.update(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "candidate_index": candidate_index,
                        "threshold": float(threshold),
                        "margin_threshold": float(margin),
                        "calibration_gate_pass": calibration_gate(
                            summary,
                            float(
                                protocol["selection"][
                                    "calibration_minimum_positive_session_fraction"
                                ]
                            ),
                        ),
                    }
                )
                diagnostics.append(summary)
    selected = max(
        diagnostics,
        key=lambda row: (
            row["calibration_gate_pass"],
            row["accepted_positive_sessions"],
            row["top1_positive_sessions"],
            row["top3_positive_sessions"],
            -row["ambiguous_frame_activations"],
            row["accepted_sessions"],
            -row["candidate_index"],
            row["threshold"],
            row["margin_threshold"],
        ),
    )
    return selected, diagnostics


def selected_config(protocol: dict[str, Any], selected: dict[str, Any]) -> EvidenceConfig:
    row = next(
        candidate
        for candidate in protocol["candidates"]
        if candidate["candidate_id"] == selected["candidate_id"]
    )
    return config_from_protocol(row, protocol["viewpoint_contract"])


def verify_split_hashes(
    development_rows: list[dict[str, str]], evaluation_rows: list[dict[str, str]]
) -> dict[str, Any]:
    roles: dict[str, set[str]] = defaultdict(set)
    for row in development_rows + evaluation_rows:
        roles[row["split"]].add(row["frame_sha256"])
    overlaps: dict[str, int] = {}
    names = sorted(roles)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            overlaps[f"{first}:{second}"] = len(roles[first] & roles[second])
    if any(overlaps.values()):
        raise ValueError(f"exact image hashes cross split roles: {overlaps}")
    return {"unique_hashes_by_split": {key: len(value) for key, value in roles.items()}, "overlaps": overlaps}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--development-dataset", type=Path, required=True)
    parser.add_argument("--evaluation-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol["scope"]["runtime_geometry_access"]:
        raise ValueError("runtime geometry must remain disabled")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    development_samples = representation.read_csv(
        args.development_dataset / "sample_manifest.csv"
    )
    development_proposals = representation.read_csv(
        args.development_dataset / "proposal_manifest.csv"
    )
    evaluation_samples = representation.read_csv(
        args.evaluation_dataset / "sample_manifest.csv"
    )
    evaluation_proposals = representation.read_csv(
        args.evaluation_dataset / "proposal_manifest.csv"
    )
    split_check = verify_split_hashes(development_proposals, evaluation_proposals)
    recorded = representation.RecordedImages()
    base_results: dict[str, Any] = {}
    bundles: dict[str, dict[str, Any]] = {}
    selections: dict[str, dict[str, Any]] = {}
    calibration_diagnostics: dict[str, list[dict[str, Any]]] = {}
    calibration_sessions: dict[str, list[SessionData]] = {}
    for level in representation.LEVELS:
        base_result, bundle = representation.variant_fit(
            level=level,
            variant="topology",
            development_rows=development_samples,
            images=recorded,
            threshold_grid=[
                float(value) for value in protocol["selection"]["score_thresholds"]
            ],
            random_seed=int(protocol["base_model"]["random_seed"]),
        )
        sessions = build_sessions(
            sample_rows=development_samples,
            proposal_rows=development_proposals,
            level=level,
            split="calibration",
            bundle=bundle,
            recorded=recorded,
        )
        selected, diagnostics = select_setting(sessions, protocol)
        base_results[level] = base_result
        bundles[level] = bundle
        calibration_sessions[level] = sessions
        selections[level] = selected
        calibration_diagnostics[level] = diagnostics

    frozen = {
        "schema": "s0010-stage2-multiview-evidence-selection-v1",
        "source_commit": args.source_commit,
        "protocol_sha256": representation.sha256(args.protocol),
        "evaluation_opened": False,
        "primary_level": protocol["selection"]["primary_level"],
        "base_models": base_results,
        "selected_settings": selections,
        "calibration_diagnostics": calibration_diagnostics,
        "split_check": split_check,
    }
    (output / "frozen_selection.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for level, bundle in bundles.items():
        joblib.dump(bundle, output / f"s0010_{level.lower()}_topology_base.joblib")

    evaluation_results: dict[str, Any] = {}
    session_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for level in representation.LEVELS:
        sessions = build_sessions(
            sample_rows=evaluation_samples,
            proposal_rows=evaluation_proposals,
            level=level,
            split="evaluation",
            bundle=bundles[level],
            recorded=recorded,
        )
        selected = selections[level]
        summary, records, frames = evaluate_setting(
            sessions,
            selected_config(protocol, selected),
            threshold=float(selected["threshold"]),
            margin_threshold=float(selected["margin_threshold"]),
        )
        for row in records:
            row.update({"level": level, "candidate_id": selected["candidate_id"]})
        for row in frames:
            row.update({"level": level, "candidate_id": selected["candidate_id"]})
        evaluation_results[level] = {
            "diagnostic_only": level == "L4",
            "selected_setting": selected,
            "summary": summary,
        }
        session_rows.extend(records)
        frame_rows.extend(frames)

    l5 = evaluation_results["L5"]["summary"]
    gate = protocol["evaluation"]["development_gate"]
    development_gate_pass = bool(
        l5["top1_positive_sessions"] >= int(gate["required_top1_positive_sessions"])
        and l5["accepted_positive_sessions"]
        >= int(gate["required_accepted_positive_sessions"])
        and l5["wrong_accepted_positive_sessions"]
        <= int(gate["maximum_wrong_accepted_sessions"])
        and l5["accepted_no_positive_sessions"]
        <= int(gate["maximum_accepted_ambiguous_or_negative_sessions"])
    )
    write_csv(output / "evaluation_sessions.csv", session_rows)
    write_csv(output / "evaluation_frames.csv", frame_rows)
    result = {
        "schema": "s0010-stage2-multiview-evidence-result-v1",
        "source_commit": args.source_commit,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": representation.sha256(args.protocol),
        "evaluation_was_previously_opened": True,
        "independent_confirmation": False,
        "actor_feature_privileged_fields": [],
        "split_check": split_check,
        "evaluation": evaluation_results,
        "l5_development_gate_pass": development_gate_pass,
        "scientific_verdict": (
            "MULTIVIEW_DEVELOPMENT_GATE_PASS"
            if development_gate_pass
            else "MULTIVIEW_DEVELOPMENT_GATE_FAIL"
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
