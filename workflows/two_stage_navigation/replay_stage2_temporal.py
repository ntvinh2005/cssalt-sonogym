#!/usr/bin/env python3
"""Replay frozen Stage 2 scores with causal temporal and multi-view decisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.contracts import (  # noqa: E402
    assert_no_privileged_fields,
    ultrasound_sha256,
)
from two_stage_navigation.stage2_features import (  # noqa: E402
    extract_stage2_feature,
    normalize_ultrasound,
)
from two_stage_navigation.stage2_models import (  # noqa: E402
    Stage2ImageValidityConfig,
    Stage2LabelConfig,
    image_validity_target,
    supervised_targets,
)
from two_stage_navigation.stage2_temporal_decision import (  # noqa: E402
    DecisionState,
    TemporalDecisionConfig,
    reconstruct_trajectory_observations,
    replay_decision,
)


POLICIES = (
    "baseline_a_single_frame",
    "baseline_b_quality_ranked",
    "candidate_c_temporal",
    "candidate_d_multi_view",
    "diagnostic_quality_only",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def seal(path: Path) -> str:
    digest = file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    truth = labels.astype(bool)
    predicted = predictions.astype(bool)
    true_positive = int(np.count_nonzero(truth & predicted))
    false_positive = int(np.count_nonzero(~truth & predicted))
    false_negative = int(np.count_nonzero(truth & ~predicted))
    true_negative = int(np.count_nonzero(~truth & ~predicted))
    denominator = true_positive + false_positive
    positive_count = true_positive + false_negative
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": None if denominator == 0 else true_positive / denominator,
        "recall": None if positive_count == 0 else true_positive / positive_count,
    }


def _near_duplicate_groups(
    images: np.ndarray, public_rows: list[dict[str, Any]], *, maximum_rmse: float
) -> list[list[int]]:
    small = []
    for image in images:
        normalized = normalize_ultrasound(image)
        small.append(normalized.reshape(20, 10, 15, 10).mean(axis=(1, 3)).reshape(-1))
    features = np.stack(small)
    parent = list(range(len(images)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_trajectory: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(public_rows):
        by_trajectory[str(row["trajectory_id"])].append(index)
    for indexes in by_trajectory.values():
        for offset, left in enumerate(indexes):
            for right in indexes[offset + 1 :]:
                if (
                    public_rows[left]["frame_sha256"]
                    == public_rows[right]["frame_sha256"]
                ):
                    union(left, right)
                    continue
                rmse = float(np.sqrt(np.mean((features[left] - features[right]) ** 2)))
                if rmse <= maximum_rmse:
                    union(left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(images)):
        groups[find(index)].append(index)
    return [indexes for indexes in groups.values() if len(indexes) > 1]


def _trajectory_groups(
    public_rows: list[dict[str, Any]],
) -> list[tuple[str, list[int]]]:
    grouped: dict[str, list[int]] = {}
    order: list[str] = []
    for index, row in enumerate(public_rows):
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id not in grouped:
            grouped[trajectory_id] = []
            order.append(trajectory_id)
        grouped[trajectory_id].append(index)
    seen: set[str] = set()
    last: str | None = None
    for row in public_rows:
        current = str(row["trajectory_id"])
        if current != last:
            if current in seen:
                raise RuntimeError("trajectory rows are not contiguous")
            seen.add(current)
            last = current
    return [(trajectory_id, grouped[trajectory_id]) for trajectory_id in order]


def _path_length(public_rows: list[dict[str, Any]]) -> tuple[float, float]:
    translation = sum(
        math.hypot(*[float(value) for value in row["executed_delta_relative_mm"]])
        for row in public_rows
    )
    rotation_deg = sum(
        abs(float(row["executed_delta_angle_rad"])) * 180.0 / math.pi
        for row in public_rows
    )
    return translation, rotation_deg


def _aggregate_decisions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row["outcome"] == DecisionState.ACCEPT.value]
    correct = sum(int(row["selected_image_validity"] == 1) for row in accepted)
    false_accepted = sum(int(row["selected_image_validity"] == 0) for row in accepted)
    geometric = sum(int(row["selected_geometric_validity"] == 1) for row in accepted)
    latencies = [int(row["decision_latency_frames"]) for row in rows]
    return {
        "trajectory_count": len(rows),
        "accepted_candidate_count": len(accepted),
        "abstention_count": len(rows) - len(accepted),
        "correct_selected_candidate_count": correct,
        "false_accepted_candidate_count": false_accepted,
        "accepted_candidate_precision": None
        if not accepted
        else correct / len(accepted),
        "acceptance_coverage": None if not rows else len(accepted) / len(rows),
        "geometric_success_among_selected_count": geometric,
        "geometric_success_among_selected_rate": (
            None if not accepted else geometric / len(accepted)
        ),
        "median_decision_latency_frames": (
            None if not latencies else float(np.median(latencies))
        ),
        "median_distinct_supporting_hashes": (
            None
            if not accepted
            else float(
                np.median([row["distinct_supporting_hashes"] for row in accepted])
            )
        ),
        "median_distinct_supporting_viewpoints": (
            None
            if not accepted
            else float(
                np.median([row["distinct_supporting_viewpoints"] for row in accepted])
            )
        ),
    }


def _plot_policy_comparison(policy_metrics: dict[str, Any], output: Path) -> None:
    names = list(POLICIES)
    false_accepts = [
        policy_metrics[name]["false_accepted_candidate_count"] for name in names
    ]
    accepted = [policy_metrics[name]["accepted_candidate_count"] for name in names]
    labels = [name.replace("candidate_", "").replace("baseline_", "") for name in names]
    x = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(x - 0.18, accepted, width=0.36, label="accepted trajectories")
    axis.bar(x + 0.18, false_accepts, width=0.36, label="false accepted")
    axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.set_ylabel("trajectory count")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_examples(
    examples: list[dict[str, Any]],
    image_lookup: dict[tuple[str, str, int], np.ndarray],
    output: Path,
) -> None:
    if not examples:
        return
    figure, axes = plt.subplots(len(examples), 1, figsize=(6, 3.2 * len(examples)))
    axes_array = np.atleast_1d(axes)
    for axis, example in zip(axes_array, examples):
        key = (example["source"], example["session_id"], example["frame_sequence"])
        axis.imshow(image_lookup[key], cmap="gray", aspect="auto")
        axis.set_title(
            f"{example['kind']}: {example['source']} / {example['session_id']} / "
            f"frame {example['frame_sequence']}"
        )
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()

    protocol_hash = file_sha256(args.protocol)
    expected_hash = (
        args.protocol.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    )
    if protocol_hash != expected_hash:
        raise RuntimeError("temporal replay protocol seal mismatch")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "sonogym-stage2-temporal-replay-v1":
        raise RuntimeError("unexpected temporal replay protocol")

    bundle_path = Path(protocol["source"]["frozen_model_bundle"])
    if file_sha256(bundle_path) != protocol["source"]["frozen_model_bundle_sha256"]:
        raise RuntimeError("frozen model bundle hash mismatch")
    bundle = joblib.load(bundle_path)
    if bundle.get("classifier_target") != "l4_image_visibility":
        raise RuntimeError("unexpected frozen classifier target")
    classifier = bundle["classifier"]
    regressor = bundle["regressor"]
    class_columns = {
        int(value): index for index, value in enumerate(classifier.classes_)
    }
    if 1 not in class_columns:
        raise RuntimeError("frozen classifier has no positive class")

    image_config = Stage2ImageValidityConfig(**bundle["image_validity_config"])
    label_config = Stage2LabelConfig(**bundle["label_config"])
    rule = protocol["decision_rules"]
    decision_config = TemporalDecisionConfig(
        validity_probability_threshold=float(rule["validity_probability_threshold"]),
        maximum_neighbor_translation_mm=float(rule["maximum_neighbor_translation_mm"]),
        maximum_neighbor_angle_deg=float(rule["maximum_neighbor_angle_deg"]),
        maximum_quality_drop=float(rule["maximum_quality_drop"]),
        required_distinct_hash_observations=int(
            rule["required_distinct_hash_observations"]
        ),
        minimum_distinct_view_translation_mm=float(
            rule["minimum_distinct_view_translation_mm"]
        ),
        minimum_distinct_view_angle_deg=float(rule["minimum_distinct_view_angle_deg"]),
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    trajectory_results: list[dict[str, Any]] = []
    dataset_audit: list[dict[str, Any]] = []
    image_lookup: dict[tuple[str, str, int], np.ndarray] = {}
    exact_duplicate_example: dict[str, Any] | None = None
    frame_false_accept_example: dict[str, Any] | None = None
    frame_false_reject_example: dict[str, Any] | None = None

    for dataset in protocol["datasets"]:
        source_name = str(dataset["name"])
        capture_root = Path(dataset["root"])
        summary_path = capture_root / "results" / "capture_summary.json"
        if file_sha256(summary_path) != dataset["capture_summary_sha256"]:
            raise RuntimeError(f"capture summary hash mismatch for {source_name}")
        capture_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_session_rows: list[dict[str, Any]] = []
        for session_root in sorted((capture_root / "sessions").iterdir()):
            if (
                not session_root.is_dir()
                or not (session_root / "summary.json").exists()
            ):
                continue
            session_summary = json.loads(
                (session_root / "summary.json").read_text(encoding="utf-8")
            )
            if session_summary.get("status") != "PASSED":
                raise RuntimeError(f"incomplete session {session_root.name}")
            public = read_jsonl(session_root / "public" / "frames.jsonl")
            privileged = read_jsonl(session_root / "privileged" / "annotations.jsonl")
            actions = read_jsonl(session_root / "privileged" / "actions.jsonl")
            images = np.load(session_root / "public" / "ultrasound.npz")["ultrasound"]
            if not len(public) == len(privileged) == len(images):
                raise RuntimeError(f"frame join mismatch in {session_root.name}")
            if [int(row["frame_sequence"]) for row in public] != list(
                range(len(public))
            ):
                raise RuntimeError(
                    f"noncontiguous session frame sequence in {session_root.name}"
                )
            for row in public:
                assert_no_privileged_fields(row)

            features = []
            image_targets = []
            geometric_targets = []
            quality_targets = []
            for index, (actor, annotation, image) in enumerate(
                zip(public, privileged, images)
            ):
                observed_hash = ultrasound_sha256(image)
                identity = (session_root.name, index, observed_hash)
                expected_actor = (
                    str(actor["session_id"]),
                    int(actor["frame_sequence"]),
                    str(actor["frame_sha256"]),
                )
                expected_annotation = (
                    str(annotation["session_id"]),
                    int(annotation["frame_sequence"]),
                    str(annotation["frame_sha256"]),
                )
                if identity != expected_actor or identity != expected_annotation:
                    raise RuntimeError(f"frame identity mismatch at {identity[:2]}")
                classifier_target, _ = image_validity_target(
                    technical_image_valid=bool(annotation["technical_image_valid"]),
                    l4_visible_fraction=float(annotation["l4_visible_fraction"]),
                    config=image_config,
                )
                geometric_target, expected_quality = supervised_targets(
                    technical_image_valid=bool(annotation["technical_image_valid"]),
                    position_error_mm=float(annotation["position_error_mm"]),
                    orientation_error_deg=float(annotation["orientation_error_deg"]),
                    l4_visible_fraction=float(annotation["l4_visible_fraction"]),
                    config=label_config,
                )
                if geometric_target != int(
                    annotation["validity_label"]
                ) or not math.isclose(
                    expected_quality, float(annotation["quality_label"]), abs_tol=1e-12
                ):
                    raise RuntimeError(
                        f"stored evaluator label mismatch at {identity[:2]}"
                    )
                features.append(extract_stage2_feature(image))
                image_targets.append(classifier_target)
                geometric_targets.append(geometric_target)
                quality_targets.append(expected_quality)
                image_lookup[(source_name, session_root.name, index)] = image

            feature_array = np.stack(features).astype(np.float32)
            probabilities = classifier.predict_proba(feature_array)[:, class_columns[1]]
            quality_predictions = regressor.predict(feature_array)
            if not np.all(np.isfinite(probabilities)) or not np.all(
                np.isfinite(quality_predictions)
            ):
                raise RuntimeError(
                    f"nonfinite frozen prediction in {session_root.name}"
                )

            predictions = (
                probabilities >= decision_config.validity_probability_threshold
            )
            for index, actor in enumerate(public):
                row = {
                    "source": source_name,
                    "session_id": session_root.name,
                    "trajectory_id": str(actor["trajectory_id"]),
                    "frame_sequence": index,
                    "trajectory_frame_sequence": int(
                        actor["trajectory_frame_sequence"]
                    ),
                    "frame_sha256": str(actor["frame_sha256"]),
                    "validity_probability": float(probabilities[index]),
                    "quality_prediction": float(quality_predictions[index]),
                    "image_validity_target": int(image_targets[index]),
                    "geometric_validity_target": int(geometric_targets[index]),
                    "quality_target": float(quality_targets[index]),
                    "single_frame_prediction": int(predictions[index]),
                }
                frame_rows.append(row)
                if (
                    frame_false_accept_example is None
                    and row["single_frame_prediction"] == 1
                    and row["image_validity_target"] == 0
                ):
                    frame_false_accept_example = {
                        "kind": "single-frame false accept",
                        **{
                            key: row[key]
                            for key in ("source", "session_id", "frame_sequence")
                        },
                    }
                if (
                    frame_false_reject_example is None
                    and row["single_frame_prediction"] == 0
                    and row["image_validity_target"] == 1
                ):
                    frame_false_reject_example = {
                        "kind": "single-frame false reject",
                        **{
                            key: row[key]
                            for key in ("source", "session_id", "frame_sequence")
                        },
                    }

            near_config = protocol["replay_integrity"]["near_duplicate_audit"]
            near_groups = _near_duplicate_groups(
                images,
                public,
                maximum_rmse=float(near_config["normalized_pixel_rmse_maximum"]),
            )
            exact_counts = Counter(str(row["frame_sha256"]) for row in public)
            exact_groups = [count for count in exact_counts.values() if count > 1]
            if exact_duplicate_example is None and exact_groups:
                repeated_hash = next(
                    key for key, count in exact_counts.items() if count > 1
                )
                repeated_index = next(
                    index
                    for index, row in enumerate(public)
                    if row["frame_sha256"] == repeated_hash
                )
                exact_duplicate_example = {
                    "kind": "exact duplicate image",
                    "source": source_name,
                    "session_id": session_root.name,
                    "frame_sequence": repeated_index,
                }

            trajectory_groups = _trajectory_groups(public)
            relative_pose_count = 0
            for trajectory_id, indexes in trajectory_groups:
                actor_rows = [public[index] for index in indexes]
                trajectory_probabilities = [
                    float(probabilities[index]) for index in indexes
                ]
                trajectory_qualities = [
                    float(quality_predictions[index]) for index in indexes
                ]
                observations = reconstruct_trajectory_observations(
                    actor_rows, trajectory_probabilities, trajectory_qualities
                )
                relative_pose_count += len(
                    {
                        tuple(round(value, 6) for value in observation.relative_pose)
                        for observation in observations
                    }
                )
                translation_mm, rotation_deg = _path_length(actor_rows)
                label_by_frame = {
                    int(public[index]["frame_sequence"]): {
                        "image": int(image_targets[index]),
                        "geometric": int(geometric_targets[index]),
                    }
                    for index in indexes
                }
                candidate_count = sum(
                    int(
                        observation.transition_valid
                        and observation.image_valid
                        and observation.validity_probability
                        >= decision_config.validity_probability_threshold
                    )
                    for observation in observations
                )
                for policy in POLICIES:
                    decision = replay_decision(
                        observations, policy=policy, config=decision_config
                    )
                    selected = decision.selected
                    labels = (
                        None
                        if selected is None
                        else label_by_frame[selected.frame_sequence]
                    )
                    trajectory_results.append(
                        {
                            "source": source_name,
                            "session_id": session_root.name,
                            "trajectory_id": trajectory_id,
                            "policy": policy,
                            "outcome": decision.state.value,
                            "candidate_count": candidate_count,
                            "selected_frame_sequence": (
                                None if selected is None else selected.frame_sequence
                            ),
                            "selected_frame_sha256": (
                                None if selected is None else selected.frame_sha256
                            ),
                            "selected_validity_probability": (
                                None
                                if selected is None
                                else selected.validity_probability
                            ),
                            "selected_quality_prediction": (
                                None if selected is None else selected.quality
                            ),
                            "selected_image_validity": (
                                None if labels is None else labels["image"]
                            ),
                            "selected_geometric_validity": (
                                None if labels is None else labels["geometric"]
                            ),
                            "distinct_supporting_hashes": (
                                1
                                if selected is not None and decision.evidence is None
                                else 0
                                if decision.evidence is None
                                else decision.evidence.distinct_hash_count
                            ),
                            "distinct_supporting_viewpoints": (
                                1
                                if selected is not None and decision.evidence is None
                                else 0
                                if decision.evidence is None
                                else decision.evidence.distinct_viewpoint_count
                            ),
                            "decision_latency_frames": decision.observed_frame_count,
                            "observed_translation_mm": translation_mm,
                            "observed_rotation_deg": rotation_deg,
                            "state_trace": [
                                state.value for state in decision.state_trace
                            ],
                        }
                    )

            source_session_rows.append(
                {
                    "session_id": session_root.name,
                    "frame_count": len(public),
                    "trajectory_count": len(trajectory_groups),
                    "distinct_exact_hash_count": len(exact_counts),
                    "exact_duplicate_group_count": len(exact_groups),
                    "near_duplicate_group_count": len(near_groups),
                    "frames_in_near_duplicate_groups": sum(
                        len(group) for group in near_groups
                    ),
                    "distinct_reconstructed_relative_pose_count": relative_pose_count,
                    "accepted_transition_frame_count": sum(
                        int(row["action_accepted"] and row["action_completed"])
                        for row in public
                    ),
                    "rejected_transition_frame_count": sum(
                        int(not (row["action_accepted"] and row["action_completed"]))
                        for row in public
                    ),
                    "clamped_transition_frame_count": sum(
                        int(row["action_clamped"]) for row in public
                    ),
                    "action_step_count": len(actions),
                    "action_step_not_progressed_count": sum(
                        int(not row.get("command_progressed", False)) for row in actions
                    ),
                    "reset_boundaries_verified": True,
                    "fresh_same_pose_reacquisition_available": False,
                    "return_to_best_execution_available": False,
                }
            )
        dataset_audit.append(
            {
                "source": source_name,
                "root": str(capture_root),
                "patient_count": 1,
                "session_count": len(source_session_rows),
                "trajectory_count": sum(
                    row["trajectory_count"] for row in source_session_rows
                ),
                "frame_count": sum(row["frame_count"] for row in source_session_rows),
                "distinct_exact_hash_count": int(
                    capture_summary["distinct_frame_hash_count"]
                ),
                "sessions": source_session_rows,
            }
        )

    policies: dict[str, Any] = {}
    source_metrics: dict[str, dict[str, Any]] = {}
    session_metrics: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        policy_rows = [row for row in trajectory_results if row["policy"] == policy]
        policies[policy] = _aggregate_decisions(policy_rows)
        source_metrics[policy] = {
            source: _aggregate_decisions(
                [row for row in policy_rows if row["source"] == source]
            )
            for source in sorted({row["source"] for row in policy_rows})
        }
        session_metrics[policy] = {
            session: _aggregate_decisions(
                [row for row in policy_rows if row["session_id"] == session]
            )
            for session in sorted({row["session_id"] for row in policy_rows})
        }

    frame_array = np.asarray([row["image_validity_target"] for row in frame_rows])
    prediction_array = np.asarray(
        [row["single_frame_prediction"] for row in frame_rows]
    )
    quality_array = np.asarray(
        [row["quality_target"] for row in frame_rows], dtype=float
    )
    quality_prediction_array = np.asarray(
        [row["quality_prediction"] for row in frame_rows], dtype=float
    )
    quality_mask = quality_array >= 0.02
    quality_spearman = spearmanr(
        quality_array[quality_mask], quality_prediction_array[quality_mask]
    ).statistic
    if not np.isfinite(quality_spearman):
        quality_spearman = 0.0

    baseline = policies["baseline_b_quality_ranked"]
    candidate = policies["candidate_d_multi_view"]
    source_nonincrease = all(
        source_metrics["candidate_d_multi_view"][source][
            "false_accepted_candidate_count"
        ]
        <= source_metrics["baseline_b_quality_ranked"][source][
            "false_accepted_candidate_count"
        ]
        for source in source_metrics["candidate_d_multi_view"]
    )
    accepted_sources = sum(
        int(metrics["accepted_candidate_count"] > 0)
        for metrics in source_metrics["candidate_d_multi_view"].values()
    )
    d3_conditions = {
        "false_accepts_less_than_baseline_b": (
            candidate["false_accepted_candidate_count"]
            < baseline["false_accepted_candidate_count"]
        ),
        "acceptance_count_at_least_half_baseline_b": (
            candidate["accepted_candidate_count"] * 2
            >= baseline["accepted_candidate_count"]
        ),
        "false_accepts_not_higher_in_any_source": source_nonincrease,
        "accepts_in_at_least_three_sources": accepted_sources >= 3,
    }

    examples = [
        example
        for example in (
            frame_false_accept_example,
            frame_false_reject_example,
            exact_duplicate_example,
        )
        if example is not None
    ]
    report = {
        "schema": "sonogym-stage2-temporal-replay-report-v1",
        "decision": (
            "EXPLORATORY_PROMISING"
            if all(d3_conditions.values())
            else "EXPLORATORY_NO_GO"
        ),
        "protocol_sha256": protocol_hash,
        "source_commit": protocol["source"]["design_commit"],
        "model_bundle_sha256": protocol["source"]["frozen_model_bundle_sha256"],
        "model_retrained": False,
        "validity_probability_threshold": decision_config.validity_probability_threshold,
        "dataset_audit": dataset_audit,
        "frame_diagnostics": {
            "metrics": binary_metrics(frame_array, prediction_array),
            "frame_count": len(frame_rows),
            "distinct_exact_hash_count": len(
                {row["frame_sha256"] for row in frame_rows}
            ),
            "quality_support": int(np.count_nonzero(quality_mask)),
            "quality_spearman": float(quality_spearman),
            "quality_mse": float(
                np.mean(
                    (
                        quality_array[quality_mask]
                        - quality_prediction_array[quality_mask]
                    )
                    ** 2
                )
            ),
        },
        "policy_metrics": policies,
        "source_policy_metrics": source_metrics,
        "session_policy_metrics": session_metrics,
        "d3_conditions": d3_conditions,
        "support_classification": {
            "causal_temporal_replay": "SUPPORTED",
            "within_trajectory_multi_view_replay": "SUPPORTED",
            "same_pose_return_repeatability": "NOT_TESTABLE_FROM_EXISTING_LOGS",
            "physical_return_to_best": "NOT_TESTABLE_FROM_EXISTING_LOGS",
            "live_coarse_fine_micro_control": "NOT_TESTABLE_FROM_EXISTING_LOGS",
            "fresh_physical_confirmation": "NOT_TESTABLE_FROM_EXISTING_LOGS",
        },
        "scientific_gates": {
            "D0": "PASS",
            "D1": "PASS",
            "D2": "PENDING_TEST_JOB",
            "D3": (
                "EXPLORATORY_PROMISING"
                if all(d3_conditions.values())
                else "EXPLORATORY_NO_GO"
            ),
            "D4": "PENDING_TEST_JOB",
            "D5": "NOT_RUN",
            "D6": "NOT_RUN",
        },
        "representative_examples": examples,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "limitations": {
            "opened_datasets_only": True,
            "fixed_anatomy_only": True,
            "fresh_confirmation_missing": True,
            "return_to_best_missing": True,
            "physical_viewpoint_difference_does_not_prove_error_independence": True,
            "full_stage2_closed_loop": "NOT_RUN",
        },
    }

    frame_path = output / "frame_scores.csv"
    with frame_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    trajectory_path = output / "trajectory_results.jsonl"
    write_jsonl(trajectory_path, trajectory_results)
    report_path = output / "replay_report.json"
    write_json(report_path, report)
    write_json(output / "representative_examples.json", examples)
    _plot_policy_comparison(policies, output / "policy_comparison.png")
    _plot_examples(examples, image_lookup, output / "representative_examples.png")
    for path in (frame_path, trajectory_path, report_path):
        seal(path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
