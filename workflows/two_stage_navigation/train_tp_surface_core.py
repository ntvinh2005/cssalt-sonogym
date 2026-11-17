#!/usr/bin/env python3
"""Validate or fit the bounded CPU ExtraTrees TP_SURFACE_CORE patch pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from spinal_surgery.evaluation.two_stage_navigation.stage1_features import (
    UNITY_REFERENCE_PATCH_SPEC,
    extract_patch_feature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-fit", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    dataset = args.dataset_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(dataset / "candidate_sample_manifest.csv")
    if not rows:
        raise RuntimeError("candidate manifest is empty")
    present = [row for row in rows if row["candidate_intersection"] == "PRESENT"]
    features: list[np.ndarray] = []
    feature_rows: list[dict[str, str]] = []
    archives: dict[Path, np.ndarray] = {}
    failures: list[str] = []
    for row in present:
        path = Path(row["image_archive"])
        if not path.is_file():
            failures.append(f"missing archive: {path}")
            continue
        if path not in archives:
            archives[path] = np.load(path)["ultrasound"]
        image = archives[path][int(row["image_archive_index"])]
        feature = extract_patch_feature(
            image,
            float(row["patch_row"]),
            float(row["patch_column"]),
            UNITY_REFERENCE_PATCH_SPEC,
        )
        if feature is None:
            failures.append(f"patch outside image: {row['sample_id']}")
            continue
        if not np.all(np.isfinite(feature)):
            failures.append(f"nonfinite feature: {row['sample_id']}")
            continue
        features.append(feature)
        feature_rows.append(row)
    if not features:
        failures.append("no resolvable candidate patches")

    split_hashes: dict[str, set[str]] = {}
    for row in feature_rows:
        split_hashes.setdefault(row["split"], set()).add(row["frame_sha256"])
    split_names = sorted(split_hashes)
    cross_split_duplicates = sorted(
        {
            value
            for left_index, left in enumerate(split_names)
            for right in split_names[left_index + 1 :]
            for value in split_hashes[left].intersection(split_hashes[right])
        }
    )
    if cross_split_duplicates:
        failures.append(f"cross-split exact image duplicates: {len(cross_split_duplicates)}")

    eligibility = [row for row in feature_rows if row["training_eligible"].lower() == "true"]
    approvals = {row["anatomical_approval"] for row in feature_rows}
    if approvals != {"APPROVED"}:
        failures.append("exact TP surface-core geometry approval is pending")
    if any(row["acoustic_visibility"] != "VISIBLE" for row in eligibility):
        failures.append("eligible image rows lack reviewed visible-interface evidence")
    if any(row["semantic_identity"] != "SUPPORTED" for row in eligibility):
        failures.append("eligible image rows lack supported semantic identity")

    minimums = {"train": 40, "calibration": 10, "development_evaluation": 10}
    class_names = sorted({row["candidate_id"] for row in rows})
    eligible_counts = {
        class_name: {
            split: sum(
                row["candidate_id"] == class_name
                and row["split"] == split
                and row["training_eligible"].lower() == "true"
                for row in feature_rows
            )
            for split in minimums
        }
        for class_name in class_names
    }
    for class_name, counts in eligible_counts.items():
        for split, minimum in minimums.items():
            if counts[split] < minimum:
                failures.append(f"{class_name} {split}: {counts[split]} < {minimum}")

    estimator = ExtraTreesClassifier(
        n_estimators=240,
        class_weight="balanced",
        random_state=20261117,
        n_jobs=4,
    )
    estimated_feature_bytes = int(len(features) * UNITY_REFERENCE_PATCH_SPEC.feature_width * 8)
    report = {
        "schema": "sonogym-tp-surface-core-training-preflight-v1",
        "mode": "NO_FIT" if args.no_fit else "FIT",
        "candidate_rows": len(rows),
        "present_candidate_rows": len(present),
        "resolved_feature_rows": len(features),
        "feature_width": UNITY_REFERENCE_PATCH_SPEC.feature_width,
        "estimated_feature_bytes": estimated_feature_bytes,
        "eligible_counts": eligible_counts,
        "cross_split_duplicate_count": len(cross_split_duplicates),
        "estimator": estimator.get_params(),
        "blocking_conditions": sorted(set(failures)),
        "fit_called": False,
    }
    if args.no_fit:
        report["status"] = (
            "ENGINEERING_PREFLIGHT_ONLY" if failures else "TRAINING_READY_FOR_APPROVAL"
        )
    elif failures:
        report["status"] = "TRAINING_BLOCKED"
    else:
        matrix = np.stack(
            [feature for feature, row in zip(features, feature_rows) if row in eligibility]
        )
        targets = np.asarray([int(row["class_id"]) for row in eligibility])
        estimator.fit(matrix, targets)
        report["fit_called"] = True
        report["status"] = "TRAINING_COMPLETED"
    (output / "preflight_result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("status", "resolved_feature_rows", "fit_called")}, sort_keys=True))
    return 3 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
