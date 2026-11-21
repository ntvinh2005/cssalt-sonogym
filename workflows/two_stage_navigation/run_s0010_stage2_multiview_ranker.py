#!/usr/bin/env python3
"""Fit a session-separated ranker from actor-visible multi-view evidence."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from two_stage_navigation.ultrasound_multiview import (
    MULTIVIEW_FEATURE_NAMES,
    EvidenceConfig,
    multiview_feature_matrix,
)

import run_s0010_stage2_multiview_evidence as multiview
import run_s0010_ultrasound_representation_experiment as representation


def fit_base_model(
    rows: list[dict[str, str]],
    *,
    level: str,
    included_sessions: set[str],
    recorded: representation.RecordedImages,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["split"] == "train"
        and row["conditioned_level"] == level
        and row["session_id"] in included_sessions
    ]
    matrix = np.asarray([recorded.feature(row, "topology") for row in selected])
    labels = representation.labels(selected)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("each base-model fit must contain both clear classes")
    estimator = ExtraTreesClassifier(
        n_estimators=int(protocol["base_model"]["trees"]),
        class_weight=protocol["base_model"]["class_weight"],
        random_state=int(protocol["base_model"]["random_seed"]),
        n_jobs=int(protocol["execution"]["maximum_cpu_threads"]),
    )
    estimator.fit(matrix, labels)
    return {
        "estimator": estimator,
        "variant": "topology",
        "feature_width": int(matrix.shape[1]),
        "level": level,
        "fit_images": len(selected),
        "fit_positive": int(np.count_nonzero(labels == 1)),
        "fit_negative": int(np.count_nonzero(labels == 0)),
    }


def evidence_config(protocol: dict[str, Any]) -> EvidenceConfig:
    return multiview.config_from_protocol(
        protocol["correspondence"], protocol["viewpoint_contract"]
    )


def graph_features(
    session: multiview.SessionData, config: EvidenceConfig
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    return multiview_feature_matrix(
        session.base_scores,
        session.correspondences,
        session.translation_mm,
        session.angle_deg,
        config,
    )


def training_session_folds(
    proposal_rows: list[dict[str, str]], *, level: str, fold_count: int
) -> list[tuple[set[str], set[str]]]:
    sessions = sorted(
        {
            row["session_id"]
            for row in proposal_rows
            if row["split"] == "train" and row["conditioned_level"] == level
        }
    )
    if fold_count < 2 or len(sessions) < fold_count:
        raise ValueError("session folds require at least two nonempty held-out groups")
    folds: list[tuple[set[str], set[str]]] = []
    for fold_index in range(fold_count):
        held = set(sessions[fold_index::fold_count])
        fitted = set(sessions) - held
        if not held or not fitted or held & fitted:
            raise AssertionError("invalid session-separated fold")
        folds.append((fitted, held))
    return folds


def out_of_fold_training_data(
    *,
    level: str,
    sample_rows: list[dict[str, str]],
    proposal_rows: list[dict[str, str]],
    recorded: representation.RecordedImages,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fold_count = int(protocol["base_model"]["out_of_fold_session_folds"])
    folds = training_session_folds(
        proposal_rows, level=level, fold_count=fold_count
    )
    all_features: list[np.ndarray] = []
    all_labels: list[int] = []
    accounting: list[dict[str, Any]] = []
    config = evidence_config(protocol)
    for fold_index, (fit_sessions, held) in enumerate(folds):
        bundle = fit_base_model(
            sample_rows,
            level=level,
            included_sessions=fit_sessions,
            recorded=recorded,
            protocol=protocol,
        )
        held_samples = [
            row
            for row in sample_rows
            if row["split"] == "train"
            and row["conditioned_level"] == level
            and row["session_id"] in held
        ]
        held_proposals = [
            row
            for row in proposal_rows
            if row["split"] == "train"
            and row["conditioned_level"] == level
            and row["session_id"] in held
        ]
        session_data = multiview.build_sessions(
            sample_rows=held_samples,
            proposal_rows=held_proposals,
            level=level,
            split="train",
            bundle=bundle,
            recorded=recorded,
        )
        kept = 0
        ambiguous = 0
        unsupported = 0
        for session in session_data:
            matrix, support_counts, _ = graph_features(session, config)
            for index, status in enumerate(session.statuses):
                if status == multiview.AMBIGUOUS:
                    ambiguous += 1
                    continue
                if support_counts[index] < config.minimum_support:
                    unsupported += 1
                    continue
                all_features.append(matrix[index])
                all_labels.append(int(status == multiview.POSITIVE))
                kept += 1
        accounting.append(
            {
                "fold": fold_index,
                "held_sessions": sorted(held),
                "fit_sessions": len(fit_sessions),
                "kept_clear_frames": kept,
                "excluded_ambiguous_frames": ambiguous,
                "excluded_unsupported_clear_frames": unsupported,
            }
        )
    matrix = np.asarray(all_features, dtype=np.float64)
    labels = np.asarray(all_labels, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[1] != len(MULTIVIEW_FEATURE_NAMES):
        raise ValueError("out-of-fold feature matrix has the wrong shape")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("ranker training requires both clear classes")
    return matrix, labels, {
        "folds": accounting,
        "images": len(labels),
        "positive": int(np.count_nonzero(labels == 1)),
        "negative": int(np.count_nonzero(labels == 0)),
    }


def fit_ranker(
    matrix: np.ndarray, labels: np.ndarray, protocol: dict[str, Any]
) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classify",
                LogisticRegression(
                    C=float(protocol["ranker"]["regularization_c"]),
                    class_weight=protocol["ranker"]["class_weight"],
                    max_iter=int(protocol["ranker"]["maximum_iterations"]),
                    random_state=int(protocol["ranker"]["random_seed"]),
                ),
            ),
        ]
    )
    model.fit(matrix, labels)
    return model


def ranker_session_scores(
    session: multiview.SessionData,
    ranker: Pipeline,
    config: EvidenceConfig,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    matrix, support_counts, support_indices = graph_features(session, config)
    positive_column = list(ranker.classes_).index(1)
    scores = ranker.predict_proba(matrix)[:, positive_column]
    return np.asarray(scores), support_counts, support_indices


def evaluate_ranker(
    sessions: list[multiview.SessionData],
    ranker: Pipeline,
    config: EvidenceConfig,
    *,
    threshold: float,
    margin_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    for session in sessions:
        scores, support_counts, support_indices = ranker_session_scores(
            session, ranker, config
        )
        record, frame_rows = multiview.assess_session(
            session,
            scores,
            support_counts,
            support_indices,
            threshold=threshold,
            margin_threshold=margin_threshold,
            minimum_support=config.minimum_support,
        )
        records.append(record)
        frames.extend(frame_rows)
    return multiview.summarize_sessions(records, frames), records, frames


def select_threshold(
    sessions: list[multiview.SessionData],
    ranker: Pipeline,
    config: EvidenceConfig,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    for threshold in protocol["selection"]["score_thresholds"]:
        for margin in protocol["selection"]["top1_margin_thresholds"]:
            summary, _, _ = evaluate_ranker(
                sessions,
                ranker,
                config,
                threshold=float(threshold),
                margin_threshold=float(margin),
            )
            summary.update(
                {
                    "threshold": float(threshold),
                    "margin_threshold": float(margin),
                    "calibration_gate_pass": multiview.calibration_gate(
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
            row["threshold"],
            row["margin_threshold"],
        ),
    )
    return selected, diagnostics


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
    samples = representation.read_csv(args.development_dataset / "sample_manifest.csv")
    proposals = representation.read_csv(args.development_dataset / "proposal_manifest.csv")
    recorded = representation.RecordedImages()
    config = evidence_config(protocol)
    bundles: dict[str, dict[str, Any]] = {}
    rankers: dict[str, Pipeline] = {}
    selections: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    accounting: dict[str, Any] = {}
    for level in representation.LEVELS:
        matrix, labels, level_accounting = out_of_fold_training_data(
            level=level,
            sample_rows=samples,
            proposal_rows=proposals,
            recorded=recorded,
            protocol=protocol,
        )
        ranker = fit_ranker(matrix, labels, protocol)
        train_sessions = {
            row["session_id"]
            for row in proposals
            if row["split"] == "train" and row["conditioned_level"] == level
        }
        base_bundle = fit_base_model(
            samples,
            level=level,
            included_sessions=train_sessions,
            recorded=recorded,
            protocol=protocol,
        )
        calibration = multiview.build_sessions(
            sample_rows=samples,
            proposal_rows=proposals,
            level=level,
            split="calibration",
            bundle=base_bundle,
            recorded=recorded,
        )
        selected, level_diagnostics = select_threshold(
            calibration, ranker, config, protocol
        )
        bundles[level] = base_bundle
        rankers[level] = ranker
        selections[level] = selected
        diagnostics[level] = level_diagnostics
        accounting[level] = level_accounting

    frozen = {
        "schema": "s0010-stage2-multiview-ranker-selection-v1",
        "source_commit": args.source_commit,
        "protocol_sha256": representation.sha256(args.protocol),
        "evaluation_opened": False,
        "feature_names": list(MULTIVIEW_FEATURE_NAMES),
        "training_accounting": accounting,
        "selected_settings": selections,
        "calibration_diagnostics": diagnostics,
    }
    (output / "frozen_selection.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for level in representation.LEVELS:
        checkpoint = {
            "schema": "s0010-stage2-multiview-ranker-v1",
            "level": level,
            "base_model": bundles[level],
            "ranker": rankers[level],
            "evidence_config": config,
            "feature_names": MULTIVIEW_FEATURE_NAMES,
            "selection": selections[level],
            "actor_inputs_only": True,
        }
        joblib.dump(checkpoint, output / f"s0010_{level.lower()}_multiview_ranker.joblib")

    if not any(row["calibration_gate_pass"] for row in selections.values()):
        result = {
            "schema": "s0010-stage2-multiview-ranker-result-v1",
            "source_commit": args.source_commit,
            "protocol_sha256": representation.sha256(args.protocol),
            "evaluation_opened": False,
            "selected_settings": selections,
            "scientific_verdict": "MULTIVIEW_RANKER_CALIBRATION_FAIL",
            "elapsed_seconds": time.monotonic() - started,
        }
        (output / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    evaluation_samples = representation.read_csv(
        args.evaluation_dataset / "sample_manifest.csv"
    )
    evaluation_proposals = representation.read_csv(
        args.evaluation_dataset / "proposal_manifest.csv"
    )
    split_check = multiview.verify_split_hashes(proposals, evaluation_proposals)
    evaluation: dict[str, Any] = {}
    session_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for level in representation.LEVELS:
        selected = selections[level]
        if not selected["calibration_gate_pass"]:
            evaluation[level] = {"opened": False, "reason": "calibration_gate_failed"}
            continue
        sessions = multiview.build_sessions(
            sample_rows=evaluation_samples,
            proposal_rows=evaluation_proposals,
            level=level,
            split="evaluation",
            bundle=bundles[level],
            recorded=recorded,
        )
        summary, records, frames = evaluate_ranker(
            sessions,
            rankers[level],
            config,
            threshold=float(selected["threshold"]),
            margin_threshold=float(selected["margin_threshold"]),
        )
        for row in records:
            row["level"] = level
        for row in frames:
            row["level"] = level
        evaluation[level] = {"opened": True, "summary": summary}
        session_rows.extend(records)
        frame_rows.extend(frames)
    if session_rows:
        multiview.write_csv(output / "evaluation_sessions.csv", session_rows)
        multiview.write_csv(output / "evaluation_frames.csv", frame_rows)
    l5_summary = evaluation.get("L5", {}).get("summary")
    l5_pass = bool(
        l5_summary is not None
        and multiview.passes_development_gate(
            l5_summary,
            selections["L5"],
            protocol["evaluation"]["development_gate"],
        )
    )
    result = {
        "schema": "s0010-stage2-multiview-ranker-result-v1",
        "source_commit": args.source_commit,
        "protocol_sha256": representation.sha256(args.protocol),
        "evaluation_was_previously_opened": True,
        "independent_confirmation": False,
        "actor_feature_privileged_fields": [],
        "split_check": split_check,
        "selected_settings": selections,
        "evaluation": evaluation,
        "l5_development_gate_pass": l5_pass,
        "scientific_verdict": (
            "MULTIVIEW_RANKER_DEVELOPMENT_GATE_PASS"
            if l5_pass
            else "MULTIVIEW_RANKER_DEVELOPMENT_GATE_FAIL"
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
