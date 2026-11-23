#!/usr/bin/env python3
"""Evaluate frozen actor-side sonoanatomy models on clinician-scan captures."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from two_stage_navigation.contracts import (
    assert_no_privileged_fields,
    ultrasound_sha256,
)
from two_stage_navigation.sonoanatomy_multiview import (
    MultiviewConfig,
    PairDecision,
    SemanticObservation,
    cluster_observations,
    decide_tp_sap_pair,
    semantic_observations,
)
from two_stage_navigation.sonoanatomy_segmentation import (
    SmallUNet,
    normalized_ultrasound,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--frame-results", type=Path, required=True)
    parser.add_argument("--l4-checkpoint", type=Path, required=True)
    parser.add_argument("--l5-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_frame_results(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {(row["session_id"], int(row["frame_sequence"])): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("frame results contain duplicate frame identities")
    return result


def load_model(path: Path) -> tuple[SmallUNet, dict[str, Any]]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("schema") != "s0010-sonoanatomy-unet-v1":
        raise ValueError(f"unexpected checkpoint schema: {path}")
    if bundle.get("input_contract") != "clean_ultrasound_only":
        raise ValueError(f"unexpected input contract: {path}")
    model = SmallUNet(base_channels=int(bundle["base_channels"]))
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, bundle


@torch.no_grad()
def predict(model: SmallUNet, images: np.ndarray) -> np.ndarray:
    values = torch.from_numpy(np.asarray(images, dtype=np.float32))
    normalized = normalized_ultrasound(values)[:, None]
    return torch.softmax(model(normalized), dim=1).cpu().numpy()


def predicted_presence(
    probabilities: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float | int]],
) -> dict[str, bool | int]:
    result: dict[str, bool | int] = {}
    for class_id, name in ((1, "TP"), (2, "SAP"), (3, "OTHER_BONE")):
        pixels = int(
            np.count_nonzero(
                probabilities[class_id] >= float(thresholds[name]["threshold"])
            )
        )
        result[f"{name.lower()}_predicted_pixels"] = pixels
        result[f"{name.lower()}_predicted"] = bool(
            pixels >= int(thresholds[name]["minimum_pixels"])
        )
    return result


def observations_for(
    probabilities: np.ndarray,
    frames: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Mapping[str, float | int]],
    *,
    pose_indices: Sequence[int] | None = None,
    stale_hash: str | None = None,
) -> tuple[SemanticObservation, ...]:
    indices = list(range(len(frames))) if pose_indices is None else list(pose_indices)
    if len(indices) != len(frames):
        raise ValueError("pose association count differs from frame count")
    result: list[SemanticObservation] = []
    for index, (probability, frame, pose_index) in enumerate(
        zip(probabilities, frames, indices)
    ):
        pose = frames[pose_index]
        result.extend(
            semantic_observations(
                probability,
                thresholds=thresholds,
                end_effector_position_m=pose["actor_base_to_ee_position_m"],
                rotation_columns=np.asarray(
                    pose["actor_base_to_ee_rotation_columns"], dtype=np.float64
                ),
                frame_sequence=int(frame["frame_sequence"]),
                frame_sha256=stale_hash or str(frame["frame_sha256"]),
            )
        )
    return tuple(result)


def decision_payload(decision: PairDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "reason": decision.reason,
        "selected": None if decision.selected is None else asdict(decision.selected),
        "hypothesis_count": len(decision.hypotheses),
    }


def decide(observations: Sequence[SemanticObservation]) -> PairDecision:
    clusters = cluster_observations(observations, association_radius_mm=5.0)
    return decide_tp_sap_pair(clusters, config=MultiviewConfig())


def controls(
    probabilities: np.ndarray,
    frames: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Mapping[str, float | int]],
) -> dict[str, Any]:
    original = observations_for(probabilities, frames, thresholds)
    count = len(frames)
    shifted = list(range(1, count)) + [0]
    shuffled_pose = observations_for(
        probabilities, frames, thresholds, pose_indices=shifted
    )
    duplicate_pose = observations_for(
        probabilities, frames, thresholds, pose_indices=[0] * count
    )
    stale = tuple(
        replace(row, frame_sha256=str(frames[0]["frame_sha256"])) for row in original
    )
    without_sap = tuple(row for row in original if row.semantic_class != "SAP")
    return {
        "original": decision_payload(decide(original)),
        "reversed_observation_order": decision_payload(decide(tuple(reversed(original)))),
        "cyclic_pose_image_mismatch": decision_payload(decide(shuffled_pose)),
        "duplicate_viewpoint": decision_payload(decide(duplicate_pose)),
        "stale_image_hash": decision_payload(decide(stale)),
        "missing_sap_evidence": decision_payload(decide(without_sap)),
        "competing_bone_identity_control": "NOT_RESOLVED_BY_CURRENT_PAIR_RULE",
    }


def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, int | float]:
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if truth.shape != prediction.shape:
        raise ValueError("truth and prediction shapes differ")
    tp = int(np.count_nonzero(truth & prediction))
    fp = int(np.count_nonzero(~truth & prediction))
    fn = int(np.count_nonzero(truth & ~prediction))
    tn = int(np.count_nonzero(~truth & ~prediction))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "accepted_count": int(np.count_nonzero(prediction)),
        "total_count": int(len(prediction)),
    }


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    actor_dir = args.output_dir / "actor"
    evaluator_dir = args.output_dir / "evaluator"
    actor_dir.mkdir(parents=True)
    evaluator_dir.mkdir()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") not in {
        "sonogym-s0010-clinician-scan-feasibility-v1",
        "sonogym-s0010-clinician-scan-feasibility-v2",
    }:
        raise ValueError("unexpected protocol")
    evaluator_frames = read_frame_results(args.frame_results)
    checkpoints = {"L4": args.l4_checkpoint, "L5": args.l5_checkpoint}
    models = {}
    bundles = {}
    for level, path in checkpoints.items():
        model, bundle = load_model(path)
        if bundle.get("level") != level:
            raise ValueError(f"checkpoint level mismatch: {path}")
        models[level] = model
        bundles[level] = bundle

    actor_frame_rows: list[dict[str, Any]] = []
    actor_sessions: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    joined_rows: list[dict[str, Any]] = []
    for session_spec in protocol["sessions"]:
        session_id = str(session_spec["session_id"])
        level = str(session_spec["vertebral_level"])
        session = args.run_root / "sessions" / session_id
        frames = read_jsonl(session / "public/frames.jsonl")
        images = np.load(session / "public/ultrasound.npz")["ultrasound"]
        if len(frames) != len(images):
            raise ValueError(f"frame/image count mismatch: {session_id}")
        if len({str(row["frame_sha256"]) for row in frames}) != len(frames):
            raise ValueError(f"capture contains stale/duplicate images: {session_id}")
        for frame, image in zip(frames, images):
            if ultrasound_sha256(image) != str(frame["frame_sha256"]):
                raise ValueError(f"image hash mismatch: {session_id}")
            if not all(
                bool(frame[name])
                for name in ("contact_ok", "image_valid", "action_accepted")
            ):
                raise ValueError(f"invalid actor acquisition: {session_id}")

        probabilities = predict(models[level], images)
        thresholds = bundles[level]["thresholds"]
        observations = observations_for(probabilities, frames, thresholds)
        observations_by_frame: dict[int, list[SemanticObservation]] = {}
        for observation in observations:
            observations_by_frame.setdefault(observation.frame_sequence, []).append(observation)
        for probability, frame in zip(probabilities, frames):
            sequence = int(frame["frame_sequence"])
            presence = predicted_presence(probability, thresholds)
            actor_row = {
                "schema": "s0010-clinician-scan-perception-actor-frame-v1",
                "session_id": session_id,
                "vertebral_level": level,
                "frame_sequence": sequence,
                "frame_sha256": str(frame["frame_sha256"]),
                "producer_sequence": int(frame["producer_sequence"]),
                "trajectory_id": str(frame["trajectory_id"]),
                "trajectory_frame_sequence": int(frame["trajectory_frame_sequence"]),
                "contact_ok": bool(frame["contact_ok"]),
                "image_valid": bool(frame["image_valid"]),
                **presence,
                "semantic_observations": [
                    asdict(row) for row in observations_by_frame.get(sequence, ())
                ],
            }
            assert_no_privileged_fields(actor_row)
            actor_frame_rows.append(actor_row)
            evidence = evaluator_frames[(session_id, sequence)]
            joined_rows.append(
                {
                    "session_id": session_id,
                    "level": level,
                    "frame_sequence": sequence,
                    "tp_truth": evidence["tp_supported"].lower() == "true",
                    "sap_truth": evidence["sap_supported"].lower() == "true",
                    "pair_truth": evidence["both_supported"].lower() == "true",
                    "tp_prediction": bool(presence["tp_predicted"]),
                    "sap_prediction": bool(presence["sap_predicted"]),
                    "pair_prediction": bool(
                        presence["tp_predicted"] and presence["sap_predicted"]
                    ),
                }
            )
        clusters = cluster_observations(observations, association_radius_mm=5.0)
        pair_decision = decide_tp_sap_pair(clusters, config=MultiviewConfig())
        actor_session = {
            "schema": "s0010-clinician-scan-perception-actor-session-v1",
            "session_id": session_id,
            "vertebral_level": level,
            "frame_count": len(frames),
            "distinct_frame_hash_count": len({row["frame_sha256"] for row in frames}),
            "semantic_observation_count": len(observations),
            "semantic_clusters": [asdict(row) for row in clusters],
            "pair_decision": decision_payload(pair_decision),
            "checkpoint_sha256": file_sha256(checkpoints[level]),
            "decision_mode": "causal_prefix_eligible_but_full_scan_consumed",
            "oracle_waypoint_capture": True,
        }
        assert_no_privileged_fields(actor_session)
        actor_sessions.append(actor_session)
        control_rows.append(
            {
                "session_id": session_id,
                "vertebral_level": level,
                **controls(probabilities, frames, thresholds),
            }
        )

    write_jsonl(actor_dir / "frames.jsonl", actor_frame_rows)
    write_json(actor_dir / "sessions.json", actor_sessions)
    assert_no_privileged_fields({"sessions": actor_sessions, "frames": actor_frame_rows})

    level_results = {}
    for level in ("L4", "L5"):
        rows = [row for row in joined_rows if row["level"] == level]
        level_results[level] = {
            name: binary_metrics(
                np.asarray([row[f"{name.lower()}_truth"] for row in rows]),
                np.asarray([row[f"{name.lower()}_prediction"] for row in rows]),
            )
            for name in ("TP", "SAP", "PAIR")
        }
    accepted_sessions = sum(
        row["pair_decision"]["status"] == "ACCEPT" for row in actor_sessions
    )
    evaluator_result = {
        "schema": "s0010-clinician-scan-perception-evaluation-v1",
        "scientific_status": "OPENED_FEASIBILITY_NOT_INDEPENDENT_CONFIRMATION",
        "capture_protocol_sha256": file_sha256(args.protocol),
        "frame_results_sha256": file_sha256(args.frame_results),
        "checkpoints": {
            level: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "frozen_thresholds": bundles[level]["thresholds"],
            }
            for level, path in checkpoints.items()
        },
        "frame_metrics_against_provisional_weak_labels": level_results,
        "multiview_pair_acceptance": {
            "accepted_sessions": accepted_sessions,
            "total_sessions": len(actor_sessions),
            "coverage": accepted_sessions / len(actor_sessions),
            "pair_identity_correctness": "NOT_MEASURED_BY_FRAME_PRESENCE_LABELS",
        },
        "controls": control_rows,
        "limitations": [
            "Capture waypoints were evaluator-designed and are not an autonomous policy.",
            "Weak interface labels are simulator proxies, not clinical anatomical validation.",
            "Frame presence does not verify the identity of a selected three-dimensional pair.",
            "The current pair rule does not use OTHER_BONE to reject competing bone.",
        ],
    }
    write_json(evaluator_dir / "result.json", evaluator_result)
    print(json.dumps(evaluator_result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
