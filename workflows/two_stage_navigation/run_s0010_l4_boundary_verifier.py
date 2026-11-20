#!/usr/bin/env python3
"""Fit and diagnose an image-only verifier for the frozen L4 proxy detector."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesClassifier


ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = ROOT / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/stage1_features.py"
FEATURE_SPEC = importlib.util.spec_from_file_location("tp_boundary_features", FEATURE_PATH)
if FEATURE_SPEC is None or FEATURE_SPEC.loader is None:
    raise RuntimeError("cannot load production patch features")
FEATURES = importlib.util.module_from_spec(FEATURE_SPEC)
sys.modules[FEATURE_SPEC.name] = FEATURES
FEATURE_SPEC.loader.exec_module(FEATURES)
PATCH_SPEC = FEATURES.UNITY_REFERENCE_PATCH_SPEC
extract_patch_feature = FEATURES.extract_patch_feature

EVAL_DIR = ROOT / "source/spinal_surgery/spinal_surgery/evaluation"
sys.path.insert(0, str(EVAL_DIR))
from two_stage_navigation.tp_surface_core_dataset import (  # noqa: E402
    L4_CLASS,
    L5_CLASS,
    PatchProposal,
    count_points_in_patch,
    projected_pixel_diagnostics,
)
from two_stage_navigation.ultrasound_plane_geometry import (  # noqa: E402
    plane_from_full_pose,
    project_points_to_pixels,
)


NEGATIVE = "PROXY_NEGATIVE"
POSITIVE = "PROXY_POSITIVE"
AMBIGUOUS = "PROXY_AMBIGUOUS"
STATE_ID = {NEGATIVE: 0, POSITIVE: 1, AMBIGUOUS: 2}
STATE_ORDER = (NEGATIVE, POSITIVE, AMBIGUOUS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def ultrasound_sha256(image: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(image, dtype="<f4"))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def as_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text not in {"true", "false"}:
        raise ValueError(f"invalid boolean: {value}")
    return text == "true"


def row_state(row: dict[str, Any]) -> str:
    """Keep uncertainty evidence separate from both positive and negative labels."""

    if row["projection_status"] != "VERIFIED":
        return AMBIGUOUS
    if int(row["l4_uncertain_points_in_patch"]) > 0:
        return AMBIGUOUS
    l4_core = int(row["l4_core_points_in_patch"])
    if l4_core <= 0:
        return NEGATIVE
    complete = bool(
        as_bool(row["adequate_projected_support"])
        and int(row["l5_core_points_in_patch"]) == 0
        and int(row["l5_uncertain_points_in_patch"]) == 0
    )
    return POSITIVE if complete else AMBIGUOUS


def image_state(states: Iterable[str]) -> str:
    values = set(states)
    if POSITIVE in values:
        return POSITIVE
    if AMBIGUOUS in values:
        return AMBIGUOUS
    if values == {NEGATIVE}:
        return NEGATIVE
    raise ValueError("image has no proxy state")


def class_probability(estimator: Any, matrix: np.ndarray, class_id: int) -> np.ndarray:
    classes = np.asarray(estimator.classes_)
    columns = np.flatnonzero(classes == class_id)
    if len(columns) != 1:
        raise ValueError(f"estimator lacks class {class_id}")
    return np.asarray(estimator.predict_proba(matrix)[:, int(columns[0])], dtype=np.float64)


def primary_accepts(estimator: Any, matrix: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    scores = class_probability(estimator, matrix, 1)
    classes = np.asarray(estimator.classes_)
    predictions = classes[np.argmax(estimator.predict_proba(matrix), axis=1)]
    # Binary L4 checkpoints use {0,1}; the historical three-class checkpoint used argmax too.
    return (predictions == 1) & (scores >= threshold), scores


def validate_split_isolation(rows: list[dict[str, Any]]) -> None:
    session_splits: dict[str, set[str]] = defaultdict(set)
    frame_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        session_splits[row["session_id"]].add(row["split"])
        frame_splits[row["frame_sha256"]].add(row["split"])
    if any(len(value) > 1 for value in session_splits.values()):
        raise ValueError("session crosses splits")
    if any(len(value) > 1 for value in frame_splits.values()):
        raise ValueError("exact image hash crosses splits")


def load_features(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    archives: dict[Path, np.ndarray] = {}
    features: dict[str, np.ndarray] = {}
    for row in rows:
        archive = Path(row["image_archive"])
        if archive not in archives:
            archives[archive] = np.load(archive)["ultrasound"]
        image = np.asarray(archives[archive][int(row["image_archive_index"])])
        if image.shape != (200, 150) or ultrasound_sha256(image) != row["frame_sha256"]:
            raise ValueError(f"source image mismatch: {row['sample_id']}")
        feature = extract_patch_feature(
            image, float(row["patch_row"]), float(row["patch_column"]), PATCH_SPEC
        )
        if feature is None or not np.all(np.isfinite(feature)):
            raise ValueError(f"invalid feature: {row['sample_id']}")
        if row.get("feature_sha256") and array_sha256(feature) != row["feature_sha256"]:
            raise ValueError(f"feature mismatch: {row['sample_id']}")
        features[row["sample_id"]] = feature
    return features


def add_scores(
    rows: list[dict[str, Any]], features: dict[str, np.ndarray], primary: Any, threshold: float
) -> list[dict[str, Any]]:
    matrix = np.stack([features[row["sample_id"]] for row in rows])
    accepted, probabilities = primary_accepts(primary, matrix, threshold)
    return [
        {
            **row,
            "proxy_state": row_state(row),
            "primary_probability": float(probability),
            "primary_accepted": bool(accept),
        }
        for row, accept, probability in zip(rows, accepted, probabilities)
    ]


def select_training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train" and row["primary_accepted"]:
            grouped[(row["frame_sha256"], row["proxy_state"])].append(row)
    selected = [
        min(values, key=lambda row: (-row["primary_probability"], row["sample_id"]))
        for _, values in sorted(grouped.items())
    ]
    if set(row["proxy_state"] for row in selected) != set(STATE_ORDER):
        raise ValueError("training rows do not cover all three states")
    return selected


def fit_verifier(
    rows: list[dict[str, Any]], features: dict[str, np.ndarray], settings: dict[str, Any]
) -> ExtraTreesClassifier:
    if any(row["split"] != "train" for row in rows):
        raise ValueError("non-training row reached fit")
    estimator = ExtraTreesClassifier(**settings)
    estimator.fit(
        np.stack([features[row["sample_id"]] for row in rows]),
        np.asarray([STATE_ID[row["proxy_state"]] for row in rows], dtype=np.int64),
    )
    return estimator


def decisions_by_image(
    rows: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    verifier: Any | None,
    verifier_threshold: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["frame_sha256"]].append(row)
    records: list[dict[str, Any]] = []
    for frame, proposals in sorted(grouped.items()):
        eligible = [row for row in proposals if row["primary_accepted"]]
        verifier_scores: dict[str, float] = {}
        verifier_predictions: dict[str, int] = {}
        if verifier is not None and eligible:
            matrix = np.stack([features[row["sample_id"]] for row in eligible])
            probabilities = verifier.predict_proba(matrix)
            positive = class_probability(verifier, matrix, STATE_ID[POSITIVE])
            predictions = np.asarray(verifier.classes_)[np.argmax(probabilities, axis=1)]
            verifier_scores = {
                row["sample_id"]: float(score) for row, score in zip(eligible, positive)
            }
            verifier_predictions = {
                row["sample_id"]: int(prediction)
                for row, prediction in zip(eligible, predictions)
            }
            eligible = [
                row
                for row in eligible
                if verifier_predictions[row["sample_id"]] == STATE_ID[POSITIVE]
                and verifier_scores[row["sample_id"]] >= float(verifier_threshold)
            ]
        selected = None
        if eligible:
            selected = min(
                eligible,
                key=lambda row: (
                    -row["primary_probability"]
                    * (verifier_scores.get(row["sample_id"], 1.0)),
                    row["sample_id"],
                ),
            )
        truth = image_state(row["proxy_state"] for row in proposals)
        records.append(
            {
                "frame_sha256": frame,
                "session_id": proposals[0]["session_id"],
                "split": proposals[0]["split"],
                "truth_state": truth,
                "predicted_positive": selected is not None,
                "selected_sample_id": None if selected is None else selected["sample_id"],
                "selected_state": None if selected is None else selected["proxy_state"],
                "selected_primary_probability": None
                if selected is None
                else selected["primary_probability"],
                "selected_verifier_probability": None
                if selected is None or verifier is None
                else verifier_scores[selected["sample_id"]],
                "positive_available": any(row["proxy_state"] == POSITIVE for row in proposals),
                "top3_primary_contains_positive": any(
                    row["proxy_state"] == POSITIVE
                    for row in sorted(
                        [row for row in proposals if row["primary_accepted"]],
                        key=lambda row: (-row["primary_probability"], row["sample_id"]),
                    )[:3]
                ),
            }
        )
    metrics = summarize_image_decisions(records)
    return metrics, records


def summarize_image_decisions(records: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["truth_state"] == POSITIVE and row["predicted_positive"] for row in records)
    fn = sum(row["truth_state"] == POSITIVE and not row["predicted_positive"] for row in records)
    fp = sum(row["truth_state"] == NEGATIVE and row["predicted_positive"] for row in records)
    tn = sum(row["truth_state"] == NEGATIVE and not row["predicted_positive"] for row in records)
    ambiguous = sum(
        row["truth_state"] == AMBIGUOUS and row["predicted_positive"] for row in records
    )
    selected_positive = sum(row["selected_state"] == POSITIVE for row in records)
    selected_ambiguous = sum(row["selected_state"] == AMBIGUOUS for row in records)
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    return {
        "images": len(records),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "ambiguous_activation_images": ambiguous,
        "selected_positive_images": selected_positive,
        "selected_ambiguous_images": selected_ambiguous,
        "abstentions": sum(not row["predicted_positive"] for row in records),
        "top1_selection_accuracy_when_positive_available": (
            None
            if not any(row["positive_available"] for row in records)
            else sum(
                row["positive_available"] and row["selected_state"] == POSITIVE
                for row in records
            )
            / sum(row["positive_available"] for row in records)
        ),
        "top3_primary_coverage_when_positive_available": (
            None
            if not any(row["positive_available"] for row in records)
            else sum(
                row["positive_available"] and row["top3_primary_contains_positive"]
                for row in records
            )
            / sum(row["positive_available"] for row in records)
        ),
    }


def image_metrics_by_session(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["session_id"]].append(row)
    return {
        session: summarize_image_decisions(values)
        for session, values in sorted(grouped.items())
    }


def summarize_proposal_decisions(records: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [row for row in records if row["proxy_state"] == POSITIVE]
    negative = [row for row in records if row["proxy_state"] == NEGATIVE]
    ambiguous = [row for row in records if row["proxy_state"] == AMBIGUOUS]
    tp = sum(row["predicted_positive"] for row in positive)
    fn = len(positive) - tp
    fp = sum(row["predicted_positive"] for row in negative)
    tn = len(negative) - fp
    ambiguous_activations = sum(row["predicted_positive"] for row in ambiguous)
    clear = positive + negative
    brier = None
    if clear:
        brier = float(
            np.mean(
                [
                    (float(row["positive_probability"]) - float(row["proxy_state"] == POSITIVE))
                    ** 2
                    for row in clear
                ]
            )
        )
    calibration_error = None
    if clear:
        weighted_error = 0.0
        edges = np.linspace(0.0, 1.0, 6)
        for lower, upper in zip(edges[:-1], edges[1:]):
            bucket = [
                row
                for row in clear
                if lower <= float(row["positive_probability"])
                < (upper if upper < 1.0 else upper + 1e-12)
            ]
            if bucket:
                confidence = float(np.mean([float(row["positive_probability"]) for row in bucket]))
                frequency = float(np.mean([row["proxy_state"] == POSITIVE for row in bucket]))
                weighted_error += len(bucket) / len(clear) * abs(confidence - frequency)
        calibration_error = weighted_error
    return {
        "runtime_primary_accepted_proposals": len(records),
        "positive_proposals": len(positive),
        "negative_proposals": len(negative),
        "ambiguous_proposals": len(ambiguous),
        "tp": tp,
        "fp_on_clear_negative": fp,
        "fn": fn,
        "tn": tn,
        "precision_excluding_ambiguous": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
        "clear_negative_false_positive_rate": None if not negative else fp / len(negative),
        "ambiguous_activation_count": ambiguous_activations,
        "ambiguous_activation_rate": None
        if not ambiguous
        else ambiguous_activations / len(ambiguous),
        "brier_excluding_ambiguous": brier,
        "five_bin_calibration_error_excluding_ambiguous": calibration_error,
        "ambiguous_mean_positive_probability": None
        if not ambiguous
        else float(np.mean([float(row["positive_probability"]) for row in ambiguous])),
    }


def proposal_discrimination(
    rows: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    verifier: Any,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eligible = [row for row in rows if row["primary_accepted"]]
    if not eligible:
        return summarize_proposal_decisions([]), []
    matrix = np.stack([features[row["sample_id"]] for row in eligible])
    probabilities = class_probability(verifier, matrix, STATE_ID[POSITIVE])
    predictions = np.asarray(verifier.classes_)[
        np.argmax(verifier.predict_proba(matrix), axis=1)
    ]
    records = [
        {
            "session_id": row["session_id"],
            "sample_id": row["sample_id"],
            "proxy_state": row["proxy_state"],
            "positive_probability": float(probability),
            "predicted_positive": bool(
                int(prediction) == STATE_ID[POSITIVE] and probability >= threshold
            ),
        }
        for row, probability, prediction in zip(eligible, probabilities, predictions)
    ]
    return summarize_proposal_decisions(records), records


def choose_threshold(
    baseline: dict[str, Any], candidates: dict[float, dict[str, Any]], gate: dict[str, Any]
) -> float | None:
    minimum_tp = math.ceil(
        float(gate["minimum_positive_recall_retention_fraction"]) * baseline["tp"]
    )
    eligible = [
        threshold
        for threshold, metrics in candidates.items()
        if metrics["tp"] >= minimum_tp
        and metrics["fp"]
        <= baseline["fp"] + int(gate["maximum_proxy_negative_false_positive_delta"])
        and metrics["selected_ambiguous_images"]
        <= baseline["selected_ambiguous_images"]
        - int(gate["minimum_ambiguous_selection_reduction"])
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda threshold: (
            candidates[threshold]["selected_ambiguous_images"],
            -candidates[threshold]["selected_positive_images"],
            candidates[threshold]["fp"],
            threshold,
        ),
    )


def locate_geometry(root: Path, name: str) -> Path:
    matches = list(root.rglob(f"{name}_revision.npz"))
    if len(matches) != 1:
        raise ValueError(f"expected one {name} geometry, found {len(matches)}")
    return matches[0]


def project_geometry(
    pose: dict[str, Any], local: np.ndarray, geometry: dict[str, Any], tolerance_mm: float
) -> dict[str, np.ndarray]:
    plane = plane_from_full_pose(
        end_effector_position_m=np.asarray(pose["human_to_ee_position_m"]),
        rotation_columns=np.asarray(pose["human_to_ee_rotation_columns"]),
        image_origin_height_m=float(pose["image_origin_height_m"]),
        local_coordinates_m=local,
        label_resolution_m=float(pose["label_resolution_m"]),
    )
    result: dict[str, np.ndarray] = {}
    for label, mask in (("core", geometry["core"]), ("uncertain", geometry["uncertain"])):
        projected = project_points_to_pixels(
            points_index=geometry["vertices"][mask],
            image_origin_index=plane.origin_index,
            rotation_columns=plane.rotation_columns,
            width=150,
            depth=200,
            image_resolution_m=float(pose["image_resolution_m"]),
            label_resolution_m=float(pose["label_resolution_m"]),
            plane_tolerance_mm=tolerance_mm,
        )
        result[f"{label}_rows"] = projected["rows"][projected["inside"]]
        result[f"{label}_columns"] = projected["columns"][projected["inside"]]
    return result


def state_from_projection(
    proposal: PatchProposal,
    l4: dict[str, np.ndarray],
    l5: dict[str, np.ndarray],
) -> tuple[str, dict[str, Any]]:
    l4_count = count_points_in_patch(
        proposal, rows=l4["core_rows"], columns=l4["core_columns"], patch_spec=PATCH_SPEC
    )
    l5_count = count_points_in_patch(
        proposal, rows=l5["core_rows"], columns=l5["core_columns"], patch_spec=PATCH_SPEC
    )
    l4_uncertain = count_points_in_patch(
        proposal,
        rows=l4["uncertain_rows"],
        columns=l4["uncertain_columns"],
        patch_spec=PATCH_SPEC,
    )
    l5_uncertain = count_points_in_patch(
        proposal,
        rows=l5["uncertain_rows"],
        columns=l5["uncertain_columns"],
        patch_spec=PATCH_SPEC,
    )
    diagnostics = projected_pixel_diagnostics(
        proposal,
        core_rows=l4["core_rows"],
        core_columns=l4["core_columns"],
        uncertain_rows=l4["uncertain_rows"],
        uncertain_columns=l4["uncertain_columns"],
        patch_spec=PATCH_SPEC,
    )
    if l4_uncertain > 0 or l5_uncertain > 0 or (l4_count > 0 and l5_count > 0):
        state = AMBIGUOUS
    elif l4_count > 0 and diagnostics["adequate_projected_support"]:
        state = POSITIVE
    elif l4_count > 0:
        state = AMBIGUOUS
    else:
        state = NEGATIVE
    return state, {
        **diagnostics,
        "l4_core_points": l4_count,
        "l5_core_points": l5_count,
        "l4_uncertain_points": l4_uncertain,
        "l5_uncertain_points": l5_uncertain,
    }


def proposal_available(frame: dict[str, Any]) -> bool:
    return bool(
        frame.get("proposal_row") is not None
        and frame.get("proposal_column") is not None
        and frame.get("proposal_track_id") is not None
    )


def combined_verifier_pass(
    *,
    primary_probability: float,
    primary_threshold: float,
    verifier_prediction: int,
    verifier_probability: float,
    verifier_threshold: float,
) -> bool:
    return bool(
        primary_probability >= primary_threshold
        and verifier_prediction == STATE_ID[POSITIVE]
        and verifier_probability >= verifier_threshold
    )


def reconstruct_v3(
    protocol: dict[str, Any], verifier: Any, threshold: float, output: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(protocol["inputs"]["v3_run_root"])
    primary_threshold = float(protocol["inputs"]["active_l4_checkpoint"]["threshold"])
    v3_protocol = yaml.safe_load((ROOT / protocol["inputs"]["v3_protocol"]["path"]).read_text())
    tolerance = float(v3_protocol["evaluator"]["surface_slab_tolerance_mm"])
    geometry_root = Path(protocol["inputs"]["geometry_root"])
    geometries: dict[str, dict[str, Any]] = {}
    for class_name, candidate in (
        (L4_CLASS, "L4_RIGHT_transverse_process"),
        (L5_CLASS, "L5_RIGHT_transverse_process"),
    ):
        path = locate_geometry(geometry_root, candidate)
        if sha256(path) != protocol["inputs"]["geometry_sha256"][class_name]:
            raise ValueError("geometry hash mismatch")
        archive = np.load(path)
        geometries[class_name] = {
            "vertices": archive["vertices_mesh_index"].astype(np.float64),
            "core": archive["proposed_revised_vertex_mask"].astype(bool),
            "uncertain": archive["boundary_uncertainty_vertex_mask"].astype(bool),
        }
    records: list[dict[str, Any]] = []
    session_results: dict[str, Any] = {}
    for session_dir in sorted((root / "stage2").glob("mbbpoc_v3_*")):
        frames = read_jsonl(session_dir / "public/frames.jsonl")
        poses = read_jsonl(session_dir / "privileged/full_pose_replay.jsonl")
        images = np.load(session_dir / "public/ultrasound.npz")["ultrasound"]
        coordinate_archive = np.load(session_dir / "privileged/pixel_to_volume_coordinates.npz")
        coordinates = coordinate_archive["human_img_coords"]
        local = coordinate_archive["native_local_image_coordinates_m"].astype(np.float64)
        actor = json.loads((session_dir / "results/actor_result.json").read_text())
        l4_actor = next(row for row in actor["targets"] if row["target_name"] == L4_CLASS)
        selected_sequence = int(l4_actor["selected_frame_sequence"])
        confirmation_sequences = {int(value) for value in l4_actor["fresh_confirmation_frame_sequences"]}
        session_rows: list[dict[str, Any]] = []
        for frame, pose, image in zip(frames, poses, images):
            if frame["target_name"] != L4_CLASS:
                continue
            if frame["frame_sha256"] != pose["frame_sha256"] or ultrasound_sha256(image) != frame["frame_sha256"]:
                raise ValueError("V3 frame join mismatch")
            l4_projection = project_geometry(pose, local, geometries[L4_CLASS], tolerance)
            l5_projection = project_geometry(pose, local, geometries[L5_CLASS], tolerance)
            if not proposal_available(frame):
                record = {
                    "session_id": session_dir.name,
                    "frame_sequence": int(frame["frame_sequence"]),
                    "frame_sha256": frame["frame_sha256"],
                    "phase": frame["phase"],
                    "pose_x_index": frame["executed_relative_pose_index"][0],
                    "pose_z_index": frame["executed_relative_pose_index"][1],
                    "pose_angle_rad": frame["executed_relative_pose_index"][2],
                    "primary_probability": float(frame["target_probability"]),
                    "proposal_available": False,
                    "proxy_state": None,
                    "interface_to_core_mm": None,
                    "selected": int(frame["frame_sequence"]) == selected_sequence,
                    "fresh_confirmation": int(frame["frame_sequence"]) in confirmation_sequences,
                    "verifier_positive_probability": None,
                    "verifier_pass": False,
                    "candidate_metric_eligible": False,
                    "exclusion_reason": "missing_actor_visible_proposal",
                }
                records.append(record)
                session_rows.append(record)
                continue
            proposal = PatchProposal(
                str(frame["proposal_track_id"]),
                float(frame["proposal_row"]),
                float(frame["proposal_column"]),
                np.empty(0),
            )
            state, diagnostics = state_from_projection(proposal, l4_projection, l5_projection)
            feature = extract_patch_feature(
                image, float(frame["proposal_row"]), float(frame["proposal_column"]), PATCH_SPEC
            )
            if feature is None:
                raise ValueError("V3 proposal cannot be extracted")
            matrix = np.asarray(feature)[None, :]
            verifier_probability = float(class_probability(verifier, matrix, STATE_ID[POSITIVE])[0])
            verifier_prediction = int(
                np.asarray(verifier.classes_)[np.argmax(verifier.predict_proba(matrix), axis=1)][0]
            )
            row_index = int(np.clip(round(float(frame["proposal_row"])), 0, 199))
            column_index = int(np.clip(round(float(frame["proposal_column"])), 0, 149))
            point = coordinates[int(frame["frame_sequence"]), column_index * 200 + row_index]
            core = geometries[L4_CLASS]["vertices"][geometries[L4_CLASS]["core"]]
            distance = float(np.min(np.linalg.norm(core - point, axis=1)) * 1.5)
            record = {
                "session_id": session_dir.name,
                "frame_sequence": int(frame["frame_sequence"]),
                "frame_sha256": frame["frame_sha256"],
                "phase": frame["phase"],
                "pose_x_index": frame["executed_relative_pose_index"][0],
                "pose_z_index": frame["executed_relative_pose_index"][1],
                "pose_angle_rad": frame["executed_relative_pose_index"][2],
                "primary_probability": float(frame["target_probability"]),
                "proposal_available": True,
                "proxy_state": state,
                "interface_to_core_mm": distance,
                "selected": int(frame["frame_sequence"]) == selected_sequence,
                "fresh_confirmation": int(frame["frame_sequence"]) in confirmation_sequences,
                "verifier_positive_probability": verifier_probability,
                "verifier_pass": combined_verifier_pass(
                    primary_probability=float(frame["target_probability"]),
                    primary_threshold=primary_threshold,
                    verifier_prediction=verifier_prediction,
                    verifier_probability=verifier_probability,
                    verifier_threshold=threshold,
                ),
                "candidate_metric_eligible": True,
                "exclusion_reason": "",
                **diagnostics,
            }
            records.append(record)
            session_rows.append(record)
        search = [row for row in session_rows if "fresh_confirmation" not in row["phase"]]
        candidate_search = [row for row in search if row["proposal_available"]]
        actual = next(row for row in search if row["selected"])
        if not actual["proposal_available"]:
            raise ValueError("actor selected a frame without a proposal")
        available_positive = [row for row in candidate_search if row["proxy_state"] == POSITIVE]
        actor_eligible = [
            row
            for row in candidate_search
            if row["primary_probability"] >= primary_threshold
        ]
        actor_eligible_positive = [
            row for row in actor_eligible if row["proxy_state"] == POSITIVE
        ]
        verified = [row for row in candidate_search if row["verifier_pass"]]
        verifier_selected = None if not verified else max(
            verified,
            key=lambda row: (
                row["primary_probability"] * row["verifier_positive_probability"],
                -row["frame_sequence"],
            ),
        )
        session_results[session_dir.name] = {
            "search_frames": len(search),
            "search_candidate_frames": len(candidate_search),
            "missing_proposal_search_frames": len(search) - len(candidate_search),
            "proxy_positive_search_frames": len(available_positive),
            "actor_eligible_proxy_positive_search_frames": len(actor_eligible_positive),
            "actor_eligible_search_frame_proxy_counts": dict(
                Counter(row["proxy_state"] for row in actor_eligible)
            ),
            "verifier_pass_search_frame_proxy_counts": dict(
                Counter(row["proxy_state"] for row in verified)
            ),
            "actual_selected_frame": actual["frame_sequence"],
            "actual_selected_state": actual["proxy_state"],
            "actual_selected_probability": actual["primary_probability"],
            "verifier_eligible_search_frames": len(verified),
            "verifier_hindsight_selected_frame": None
            if verifier_selected is None
            else verifier_selected["frame_sequence"],
            "verifier_hindsight_selected_state": None
            if verifier_selected is None
            else verifier_selected["proxy_state"],
            "fresh_confirmation_states": [
                row["proxy_state"]
                for row in sorted(session_rows, key=lambda item: item["frame_sequence"])
                if row["fresh_confirmation"]
            ],
            "alternative_return_and_confirmation_not_run": True,
        }
    write_csv(output / "v3_l4_candidate_diagnostics.csv", records)
    return records, session_results


def save_plots(output: Path, records: list[dict[str, Any]], calibration: list[dict[str, Any]]) -> None:
    colors = {NEGATIVE: "#666666", POSITIVE: "#2ca02c", AMBIGUOUS: "#ff7f0e"}
    fig, axis = plt.subplots(figsize=(7.5, 5.5), dpi=150)
    for state in STATE_ORDER:
        subset = [
            row
            for row in records
            if row["proxy_state"] == state
            and row["verifier_positive_probability"] is not None
        ]
        axis.scatter(
            [row["primary_probability"] for row in subset],
            [row["verifier_positive_probability"] for row in subset],
            c=colors[state], label=state, s=24, alpha=0.72,
        )
    selected = [
        row
        for row in records
        if row["selected"] and row["verifier_positive_probability"] is not None
    ]
    axis.scatter(
        [row["primary_probability"] for row in selected],
        [row["verifier_positive_probability"] for row in selected],
        facecolors="none", edgecolors="red", s=120, linewidths=1.5, label="V3 selected",
    )
    axis.set_xlabel("Frozen L4 probability")
    axis.set_ylabel("Verifier positive probability")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "v3_confidence_vs_proxy_state.png")
    plt.close(fig)

    bins = np.linspace(0.0, 1.0, 6)
    xs, observed, counts = [], [], []
    for low, high in zip(bins[:-1], bins[1:]):
        subset = [
            row for row in calibration
            if row["selected_verifier_probability"] is not None
            and low <= row["selected_verifier_probability"] <= (high if high == 1.0 else high - 1e-12)
        ]
        if subset:
            xs.append((low + high) / 2)
            observed.append(sum(row["truth_state"] == POSITIVE for row in subset) / len(subset))
            counts.append(len(subset))
    fig, axis = plt.subplots(figsize=(6.5, 5), dpi=150)
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.plot(xs, observed, "o-", color="#2ca02c")
    for x, y, count in zip(xs, observed, counts):
        axis.annotate(f"n={count}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.set_xlabel("Selected verifier positive probability")
    axis.set_ylabel("Observed PROXY_POSITIVE fraction")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "calibration_reliability.png")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.protocol.resolve()
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol["protocol_id"] != "sonogym-s0010-l4-boundary-verifier-v1":
        raise ValueError("unexpected protocol")
    for key in ("historical_tristate_manifest", "coverage_repair_manifest", "active_l4_checkpoint"):
        path = Path(protocol["inputs"][key]["path"])
        if sha256(path) != protocol["inputs"][key]["sha256"]:
            raise ValueError(f"input hash mismatch: {key}")
    v3_protocol = ROOT / protocol["inputs"]["v3_protocol"]["path"]
    if sha256(v3_protocol) != protocol["inputs"]["v3_protocol"]["sha256"]:
        raise ValueError("V3 protocol hash mismatch")

    manifests = [
        ("historical", Path(protocol["inputs"]["historical_tristate_manifest"]["path"])),
        ("coverage", Path(protocol["inputs"]["coverage_repair_manifest"]["path"])),
    ]
    rows: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    for source, path in manifests:
        for row in read_csv(path):
            if row["split"] not in {"train", "calibration", "development_evaluation"}:
                continue
            identity = f"{source}:{row['sample_id']}"
            if identity in seen_samples:
                raise ValueError("duplicate sample identity")
            seen_samples.add(identity)
            rows.append({**row, "sample_id": identity, "source_manifest": source})
    validate_split_isolation(rows)

    bundle = joblib.load(Path(protocol["inputs"]["active_l4_checkpoint"]["path"]))
    primary = bundle["estimator"]
    primary_threshold = float(protocol["inputs"]["active_l4_checkpoint"]["threshold"])
    design_rows = [row for row in rows if row["split"] in {"train", "calibration"}]
    design_features = load_features(design_rows)
    design_scored = add_scores(design_rows, design_features, primary, primary_threshold)
    fit_rows = select_training_rows(design_scored)
    verifier = fit_verifier(fit_rows, design_features, protocol["training"]["estimator"])

    calibration_rows = [row for row in design_scored if row["split"] == "calibration"]
    baseline_calibration, baseline_calibration_records = decisions_by_image(
        calibration_rows, design_features, None, None
    )
    threshold_metrics: dict[float, dict[str, Any]] = {}
    threshold_records: dict[float, list[dict[str, Any]]] = {}
    for threshold in protocol["calibration"]["threshold_candidates"]:
        metrics, records = decisions_by_image(
            calibration_rows, design_features, verifier, float(threshold)
        )
        threshold_metrics[float(threshold)] = metrics
        threshold_records[float(threshold)] = records
    selected_threshold = choose_threshold(
        baseline_calibration, threshold_metrics, protocol["calibration"]
    )
    result: dict[str, Any] = {
        "schema": "sonogym-s0010-l4-boundary-verifier-v1",
        "status": "CALIBRATION_NO_GO" if selected_threshold is None else "CALIBRATION_PASS",
        "source_revision": __import__("subprocess").check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "protocol_sha256": sha256(protocol_path),
        "fit_rows": len(fit_rows),
        "fit_unique_images": len({row["frame_sha256"] for row in fit_rows}),
        "fit_state_counts": dict(Counter(row["proxy_state"] for row in fit_rows)),
        "fit_splits": sorted({row["split"] for row in fit_rows}),
        "baseline_calibration": baseline_calibration,
        "candidate_calibration": {str(key): value for key, value in threshold_metrics.items()},
        "selected_threshold": selected_threshold,
        "development_used_for_fit_or_calibration": False,
        "v3_used_for_fit_or_calibration": False,
        "geometry_used_by_verifier_features": False,
        "l5_modified": False,
        "independent_holdout_available": False,
    }
    if selected_threshold is None:
        (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return 0

    calibration_proposal_metrics, calibration_proposal_records = proposal_discrimination(
        calibration_rows, design_features, verifier, selected_threshold
    )

    checkpoint = output / "l4_boundary_verifier.joblib"
    joblib.dump(
        {
            "schema": result["schema"],
            "estimator": verifier,
            "threshold": selected_threshold,
            "patch_spec": PATCH_SPEC.__dict__,
            "fit_sample_ids": [row["sample_id"] for row in fit_rows],
            "primary_checkpoint_sha256": protocol["inputs"]["active_l4_checkpoint"]["sha256"],
        },
        checkpoint,
    )
    reload_matrix = np.stack([design_features[row["sample_id"]] for row in fit_rows[:32]])
    reloaded = joblib.load(checkpoint)["estimator"]
    if not np.array_equal(verifier.predict_proba(reload_matrix), reloaded.predict_proba(reload_matrix)):
        raise RuntimeError("verifier checkpoint reload mismatch")

    # Opened development rows are materialized only after calibration selection.
    development_rows_raw = [row for row in rows if row["split"] == "development_evaluation"]
    development_features = load_features(development_rows_raw)
    development_rows = add_scores(
        development_rows_raw, development_features, primary, primary_threshold
    )
    baseline_development, baseline_development_records = decisions_by_image(
        development_rows, development_features, None, None
    )
    candidate_development, candidate_development_records = decisions_by_image(
        development_rows, development_features, verifier, selected_threshold
    )
    development_proposal_metrics, development_proposal_records = proposal_discrimination(
        development_rows, development_features, verifier, selected_threshold
    )
    v3_records, v3_sessions = reconstruct_v3(protocol, verifier, selected_threshold, output)
    selected_v3 = [row for row in v3_records if row["selected"]]
    result.update(
        status="VERIFIER_INCONCLUSIVE",
        checkpoint_sha256=sha256(checkpoint),
        checkpoint_reload_verified=True,
        candidate_calibration_selected=threshold_metrics[selected_threshold],
        candidate_calibration_by_session=image_metrics_by_session(
            threshold_records[selected_threshold]
        ),
        candidate_calibration_proposal_discrimination=calibration_proposal_metrics,
        baseline_opened_development=baseline_development,
        baseline_opened_development_by_session=image_metrics_by_session(
            baseline_development_records
        ),
        candidate_opened_development=candidate_development,
        candidate_opened_development_by_session=image_metrics_by_session(
            candidate_development_records
        ),
        candidate_opened_development_proposal_discrimination=development_proposal_metrics,
        opened_development_is_independent=False,
        v3_sessions=v3_sessions,
        v3_selected_state_counts=dict(Counter(row["proxy_state"] for row in selected_v3)),
        v3_selected_verifier_pass_count=sum(row["verifier_pass"] for row in selected_v3),
        v3_full_active_acquisition_success="NOT_RUN",
        verdict_reason=(
            "The image-only verifier has calibration and opened-data diagnostics, but no "
            "session-separate unopened holdout or physical return/confirmation run."
        ),
    )
    write_csv(
        output / "image_decisions.csv",
        [
            *({**row, "arm": "baseline_calibration"} for row in baseline_calibration_records),
            *({**row, "arm": "verifier_calibration"} for row in threshold_records[selected_threshold]),
            *({**row, "arm": "baseline_opened_development"} for row in baseline_development_records),
            *({**row, "arm": "verifier_opened_development"} for row in candidate_development_records),
        ],
    )
    write_csv(
        output / "proposal_discrimination.csv",
        [
            *({**row, "split_role": "calibration"} for row in calibration_proposal_records),
            *(
                {**row, "split_role": "opened_development"}
                for row in development_proposal_records
            ),
        ],
    )
    save_plots(output, v3_records, threshold_records[selected_threshold])
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({"status": result["status"], "threshold": selected_threshold}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
