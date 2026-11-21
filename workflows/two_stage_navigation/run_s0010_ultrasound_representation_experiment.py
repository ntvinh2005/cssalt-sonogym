#!/usr/bin/env python3
"""Compare full-image interface features with the frozen SAP patch baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesClassifier

from two_stage_navigation.ultrasound_surface_features import (
    DEFAULT_SURFACE_FEATURE_SPEC,
    causal_feature,
    single_view_feature,
    topology_feature,
)

import train_s0010_sap_classifier as baseline


LEVELS = ("L4", "L5")
VARIANTS = ("single", "causal")
POSITIVE = baseline.POSITIVE
NEGATIVE = baseline.NEGATIVE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class RecordedImages:
    def __init__(self) -> None:
        self.archives: dict[Path, np.ndarray] = {}
        self.public: dict[Path, list[dict[str, Any]]] = {}
        self.features: dict[tuple[str, str], np.ndarray] = {}

    def image(self, row: dict[str, str]) -> np.ndarray:
        path = Path(row["image_archive"])
        if path not in self.archives:
            self.archives[path] = np.load(path)["ultrasound"]
        return np.asarray(self.archives[path][int(row["image_archive_index"])])

    def public_rows(self, archive: Path) -> list[dict[str, Any]]:
        path = archive.parent / "frames.jsonl"
        if path not in self.public:
            self.public[path] = read_jsonl(path)
        return self.public[path]

    def public_row(self, row: dict[str, str]) -> dict[str, Any]:
        archive = Path(row["image_archive"])
        frame_sequence = int(row["frame_sequence"])
        matches = [
            value
            for value in self.public_rows(archive)
            if int(value["frame_sequence"]) == frame_sequence
        ]
        if len(matches) != 1:
            raise ValueError(f"frame manifest join failed: {row['sample_id']}")
        return matches[0]

    def preceding_image(
        self, row: dict[str, str]
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        archive = Path(row["image_archive"])
        current = self.public_row(row)
        trajectory = current.get("trajectory_id")
        sequence = int(current.get("trajectory_frame_sequence", 0))
        candidates = [
            value
            for value in self.public_rows(archive)
            if value.get("trajectory_id") == trajectory
            and int(value.get("trajectory_frame_sequence", -1)) < sequence
        ]
        if not candidates:
            return None, current
        previous = max(candidates, key=lambda value: int(value["trajectory_frame_sequence"]))
        return (
            np.asarray(self.archives[archive][int(previous["image_archive_index"])]),
            current,
        )

    def feature(self, row: dict[str, str], variant: str) -> np.ndarray:
        key = (row["frame_sha256"], variant)
        if key in self.features:
            return self.features[key]
        image = self.image(row)
        if variant == "single":
            value = single_view_feature(image)
        elif variant == "causal":
            previous, public = self.preceding_image(row)
            value = causal_feature(
                image,
                previous,
                public.get("executed_delta_relative_mm", [0.0, 0.0]),
                float(public.get("executed_delta_angle_rad", 0.0)),
                int(public.get("action_steps_since_previous_frame", 0)),
            )
        elif variant == "topology":
            value = topology_feature(image)
        elif variant == "context":
            value = single_view_feature(image)[
                DEFAULT_SURFACE_FEATURE_SPEC.topology_width :
            ]
        else:
            raise ValueError(f"unsupported representation: {variant}")
        self.features[key] = value
        return value


def unique_level_rows(
    rows: list[dict[str, str]], level: str, split: str | None = None
) -> list[dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["conditioned_level"] != level:
            continue
        if split is not None and row["split"] != split:
            continue
        selected.setdefault(row["frame_sha256"], row)
    return list(selected.values())


def labels(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [baseline.LABEL_ID[row["provisional_label"]] for row in rows], dtype=np.int64
    )


def threshold_selection(
    truth: np.ndarray, scores: np.ndarray, grid: list[float]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diagnostics = [baseline.metrics(truth, scores, value) for value in grid]
    passing = [row for row in diagnostics if row["gate_pass"]]
    pool = passing if passing else diagnostics
    selected = max(
        pool,
        key=lambda row: (
            row["f1"],
            row["recall"],
            row["precision"],
            row["threshold"],
        ),
    )
    return selected, diagnostics


def variant_fit(
    *,
    level: str,
    variant: str,
    development_rows: list[dict[str, str]],
    images: RecordedImages,
    threshold_grid: list[float],
    random_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_rows = unique_level_rows(development_rows, level, "train")
    calibration_rows = unique_level_rows(development_rows, level, "calibration")
    train_x = np.asarray([images.feature(row, variant) for row in train_rows])
    train_y = labels(train_rows)
    calibration_x = np.asarray(
        [images.feature(row, variant) for row in calibration_rows]
    )
    calibration_y = labels(calibration_rows)
    estimator = ExtraTreesClassifier(
        n_estimators=240,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=4,
    )
    estimator.fit(train_x, train_y)
    positive_column = list(estimator.classes_).index(1)
    scores = estimator.predict_proba(calibration_x)[:, positive_column]
    selected, diagnostics = threshold_selection(calibration_y, scores, threshold_grid)
    result = {
        "variant": variant,
        "feature_width": int(train_x.shape[1]),
        "train_images": len(train_rows),
        "train_positive": int(np.count_nonzero(train_y == 1)),
        "train_negative": int(np.count_nonzero(train_y == 0)),
        "calibration_images": len(calibration_rows),
        "calibration_positive": int(np.count_nonzero(calibration_y == 1)),
        "calibration_negative": int(np.count_nonzero(calibration_y == 0)),
        "calibration": selected,
        "calibration_thresholds": diagnostics,
    }
    bundle = {
        "schema": "s0010-ultrasound-full-image-extratrees-v1",
        "level": level,
        "variant": variant,
        "estimator": estimator,
        "threshold": float(selected["threshold"]),
        "feature_spec": DEFAULT_SURFACE_FEATURE_SPEC,
        "feature_width": int(train_x.shape[1]),
        "random_seed": random_seed,
        "actor_inputs_only": True,
    }
    return result, bundle


def evaluation_metrics(
    *,
    rows: list[dict[str, str]],
    level: str,
    variant: str,
    bundle: dict[str, Any],
    images: RecordedImages,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = unique_level_rows(rows, level, "evaluation")
    matrix = np.asarray([images.feature(row, variant) for row in selected])
    truth = labels(selected)
    positive_column = list(bundle["estimator"].classes_).index(1)
    scores = bundle["estimator"].predict_proba(matrix)[:, positive_column]
    threshold = float(bundle["threshold"])
    decisions = []
    for row, label, score in zip(selected, truth, scores):
        predicted = int(score >= threshold)
        decisions.append(
            {
                "level": level,
                "variant": variant,
                "session_id": row["session_id"],
                "frame_sequence": int(row["frame_sequence"]),
                "frame_sha256": row["frame_sha256"],
                "label": int(label),
                "score": float(score),
                "predicted": predicted,
                "outcome": (
                    "TP" if label == 1 and predicted == 1
                    else "FN" if label == 1
                    else "FP" if predicted == 1
                    else "TN"
                ),
            }
        )
    result = baseline.metrics(truth, scores, threshold)
    result["images"] = len(selected)
    result["previous_observation_images"] = sum(
        images.public_row(row).get("trajectory_frame_sequence", 0) > 0
        for row in selected
    )
    return result, decisions


def score_all_images(
    *,
    clear_rows: list[dict[str, str]],
    proposal_rows: list[dict[str, str]],
    level: str,
    variant: str,
    bundle: dict[str, Any],
    images: RecordedImages,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clear = {
        row["frame_sha256"]: row["provisional_label"]
        for row in clear_rows
        if row["conditioned_level"] == level and row["split"] == "evaluation"
    }
    candidates = unique_level_rows(proposal_rows, level, "evaluation")
    matrix = np.asarray([images.feature(row, variant) for row in candidates])
    positive_column = list(bundle["estimator"].classes_).index(1)
    scores = bundle["estimator"].predict_proba(matrix)[:, positive_column]
    threshold = float(bundle["threshold"])
    result: list[dict[str, Any]] = []
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, score in zip(candidates, scores):
        status = clear.get(row["frame_sha256"], "AMBIGUOUS")
        item = {
            "level": level,
            "variant": variant,
            "session_id": row["session_id"],
            "frame_sequence": int(row["frame_sequence"]),
            "frame_sha256": row["frame_sha256"],
            "status": status,
            "score": float(score),
            "activated": bool(score >= threshold),
        }
        result.append(item)
        by_session[row["session_id"]].append(item)
    ranking_rows = []
    for session, session_rows in sorted(by_session.items()):
        ordered = sorted(session_rows, key=lambda row: row["score"], reverse=True)
        ranking_rows.append(
            {
                "session_id": session,
                "top1_status": ordered[0]["status"],
                "top1_score": ordered[0]["score"],
                "top3_contains_positive": any(
                    row["status"] == POSITIVE for row in ordered[:3]
                ),
            }
        )
    summary = {
        "acquired_images": len(result),
        "clear_images": sum(row["status"] != "AMBIGUOUS" for row in result),
        "ambiguous_images": sum(row["status"] == "AMBIGUOUS" for row in result),
        "ambiguous_activations": sum(
            row["status"] == "AMBIGUOUS" and row["activated"] for row in result
        ),
        "session_rankings": ranking_rows,
        "top1_positive_sessions": sum(
            row["top1_status"] == POSITIVE for row in ranking_rows
        ),
        "top3_positive_sessions": sum(
            row["top3_contains_positive"] for row in ranking_rows
        ),
    }
    return result, summary


def frozen_l5_baseline(
    *,
    checkpoint: Path,
    evaluation_dir: Path,
) -> dict[str, Any]:
    bundle = joblib.load(checkpoint)
    rows = read_csv(evaluation_dir / "sample_manifest.csv")
    proposals = read_csv(evaluation_dir / "proposal_manifest.csv")
    arrays = np.load(evaluation_dir / "dataset_arrays.npz")
    truth, scores, _ = baseline.full_image_scores(
        estimator=bundle["estimator"],
        level="L5",
        split="evaluation",
        sample_rows=rows,
        proposal_rows=proposals,
        proposal_features=arrays["proposal_features"],
    )
    return baseline.metrics(truth, scores, float(bundle["threshold"]))


def select_variant(results: list[dict[str, Any]]) -> str:
    selected = max(
        results,
        key=lambda row: (
            row["calibration"]["gate_pass"],
            row["calibration"]["f1"],
            row["calibration"]["recall"],
            row["calibration"]["precision"],
        ),
    )
    return str(selected["variant"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--development-dataset", type=Path, required=True)
    parser.add_argument("--evaluation-dataset", type=Path, required=True)
    parser.add_argument("--frozen-l5-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol["scope"]["runtime_geometry_access"]:
        raise ValueError("runtime geometry must remain disabled")
    if sha256(args.frozen_l5_checkpoint) != protocol["inputs"][
        "frozen_l5_checkpoint_sha256"
    ]:
        raise ValueError("frozen L5 checkpoint hash mismatch")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    development_rows = read_csv(args.development_dataset / "sample_manifest.csv")
    evaluation_rows = read_csv(args.evaluation_dataset / "sample_manifest.csv")
    evaluation_proposals = read_csv(args.evaluation_dataset / "proposal_manifest.csv")
    images = RecordedImages()
    threshold_grid = [float(value) for value in protocol["selection"]["threshold_grid"]]
    random_seed = int(protocol["model"]["random_seed"])

    calibration_results: dict[str, list[dict[str, Any]]] = {}
    bundles: dict[tuple[str, str], dict[str, Any]] = {}
    selected_variants: dict[str, str] = {}
    for level in LEVELS:
        calibration_results[level] = []
        for variant in VARIANTS:
            result, bundle = variant_fit(
                level=level,
                variant=variant,
                development_rows=development_rows,
                images=images,
                threshold_grid=threshold_grid,
                random_seed=random_seed,
            )
            calibration_results[level].append(result)
            bundles[(level, variant)] = bundle
        selected_variants[level] = select_variant(calibration_results[level])

    # Candidate selection is complete before any evaluation feature is loaded.
    (output / "frozen_selection.json").write_text(
        json.dumps(
            {
                "protocol_sha256": sha256(args.protocol),
                "selected_variants": selected_variants,
                "calibration": calibration_results,
                "evaluation_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    evaluation_results: dict[str, Any] = {}
    decision_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    for level in LEVELS:
        variant = selected_variants[level]
        bundle = bundles[(level, variant)]
        level_metrics, decisions = evaluation_metrics(
            rows=evaluation_rows,
            level=level,
            variant=variant,
            bundle=bundle,
            images=images,
        )
        all_rows, coverage = score_all_images(
            clear_rows=evaluation_rows,
            proposal_rows=evaluation_proposals,
            level=level,
            variant=variant,
            bundle=bundle,
            images=images,
        )
        evaluation_results[level] = {
            "selected_variant": variant,
            "metrics": level_metrics,
            "all_frame_coverage": coverage,
        }
        decision_rows.extend(decisions)
        all_frame_rows.extend(all_rows)
        checkpoint = output / f"s0010_{level.lower()}_{variant}_full_image.joblib"
        joblib.dump(bundle, checkpoint)
        reloaded = joblib.load(checkpoint)
        if reloaded["variant"] != variant or reloaded["feature_width"] != bundle["feature_width"]:
            raise AssertionError("checkpoint reload failed")
        evaluation_results[level]["checkpoint"] = str(checkpoint)
        evaluation_results[level]["checkpoint_sha256"] = sha256(checkpoint)

    l5_baseline = frozen_l5_baseline(
        checkpoint=args.frozen_l5_checkpoint,
        evaluation_dir=args.evaluation_dataset,
    )
    l5_candidate = evaluation_results["L5"]["metrics"]
    better = bool(
        l5_candidate["fp"] <= l5_baseline["fp"]
        and l5_candidate["recall"] > l5_baseline["recall"]
        and l5_candidate["f1"] > l5_baseline["f1"]
    )
    selected_l5_calibration = next(
        row
        for row in calibration_results["L5"]
        if row["variant"] == selected_variants["L5"]
    )["calibration"]
    promising = bool(
        selected_l5_calibration["gate_pass"]
        and l5_candidate["fp"] <= l5_baseline["fp"]
        and (
            l5_candidate["recall"] > l5_baseline["recall"]
            or l5_candidate["f1"] > l5_baseline["f1"]
        )
    )
    verdict = (
        "BETTER_REPRESENTATION_DEMONSTRATED"
        if better
        else "PROMISING_BUT_INCONCLUSIVE"
        if promising
        else "CURRENT_OBJECTIVE_NOT_SUFFICIENTLY_OBSERVABLE"
    )
    write_csv(output / "evaluation_clear_image_decisions.csv", decision_rows)
    write_csv(output / "evaluation_all_frame_scores.csv", all_frame_rows)
    result = {
        "schema": "s0010-ultrasound-representation-comparison-v1",
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256(args.protocol),
        "source_commit": args.source_commit,
        "development_manifest_sha256": sha256(
            args.development_dataset / "sample_manifest.csv"
        ),
        "evaluation_manifest_sha256": sha256(
            args.evaluation_dataset / "sample_manifest.csv"
        ),
        "evaluation_was_previously_opened": True,
        "independent_confirmation": False,
        "actor_feature_privileged_fields": [],
        "calibration": calibration_results,
        "selected_variants": selected_variants,
        "evaluation": evaluation_results,
        "frozen_l5_baseline": l5_baseline,
        "scientific_verdict": verdict,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
