#!/usr/bin/env python3
"""Apply checksum-bound human TP surface-core decisions to a prepared dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


SPLITS = ("train", "calibration", "development_evaluation")
CLASS_NAMES = (
    "BACKGROUND_OR_OTHER_STRUCTURE",
    "L4_RIGHT_TP_SURFACE_CORE",
    "L5_RIGHT_TP_SURFACE_CORE",
)
MINIMUM_UNIQUE_IMAGES = {"train": 40, "calibration": 10, "development_evaluation": 10}
IDENTITY_FIELD_PAIRS = (
    ("sample_id", "sample_id"),
    ("session_id", "session_id"),
    ("split", "split"),
    ("frame_sha256", "frame_sha256"),
    ("source_array_index", "image_archive_index"),
    ("track_id", "track_id"),
    ("patch_row", "patch_row"),
    ("patch_column", "patch_column"),
    ("patch_row_start", "patch_row_start"),
    ("patch_column_start", "patch_column_start"),
    ("raw_crop_sha256", "raw_crop_sha256"),
    ("preprocessed_intensity_sha256", "preprocessed_intensity_sha256"),
    ("feature_sha256", "feature_sha256"),
    ("preprocessing_spec_sha256", "preprocessing_spec_sha256"),
    ("geometry_version_sha256", "geometry_version_sha256"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def promote(row: dict[str, str], decision: dict[str, str]) -> bool:
    candidate = row["candidate_id"]
    reviewer = decision.get("reviewer_identity", "").strip()
    qualification = decision.get("reviewer_qualification", "").strip()
    boundary = decision.get("anatomical_boundary_decision", "").strip()
    visibility = decision.get("acoustic_visibility", "").strip()
    identity = decision.get("semantic_identity", "").strip()
    if candidate == "BACKGROUND_OR_OTHER_STRUCTURE":
        accepted = (
            boundary == "NOT_APPLICABLE"
            and visibility == "VISIBLE_OTHER_STRUCTURE"
            and identity == "REJECTED_AS_TP"
        )
        if accepted:
            row.update(
                class_id="0",
                acoustic_visibility=visibility,
                semantic_identity=identity,
                anatomical_approval="NOT_APPLICABLE",
                training_label="VERIFIED_NEGATIVE",
                training_eligible="True",
                exclusion_reason="",
            )
    else:
        accepted = boundary == "ACCEPT_BOUNDARY" and visibility == "VISIBLE" and identity == "SUPPORTED"
        if accepted:
            row.update(
                acoustic_visibility=visibility,
                semantic_identity=identity,
                anatomical_approval="APPROVED",
                training_label="VERIFIED_POSITIVE",
                training_eligible="True",
                exclusion_reason="",
            )
    if accepted and (not reviewer or not qualification):
        raise ValueError(f"accepted decision lacks reviewer provenance: {row['sample_id']}")
    if not accepted:
        row.update(
            acoustic_visibility=visibility or "NOT_REVIEWED",
            semantic_identity=identity or "NOT_REVIEWED",
            training_label="EXCLUDED",
            training_eligible="False",
            exclusion_reason="review did not satisfy the frozen promotion contract",
        )
    row["reviewer_identity"] = reviewer
    row["reviewer_qualification"] = qualification
    row["review_notes"] = decision.get("notes", "")
    return accepted


def apply_review(dataset: Path, decisions_path: Path, output: Path) -> dict[str, Any]:
    dataset = dataset.resolve()
    output = output.resolve()
    if output == dataset or output.exists():
        raise ValueError("output directory must be new and distinct from the source dataset")
    rows = read_csv(dataset / "candidate_sample_manifest.csv")
    decisions = read_csv(decisions_path)
    if not rows or not decisions:
        raise ValueError("dataset manifest and review decisions must be nonempty")
    indexed_rows = {row["sample_id"]: row for row in rows}
    expected_primary = {
        row["sample_id"]
        for row in rows
        if row.get("review_asset") and row.get("review_selection") == "PRIMARY"
    }
    allowed_review = {row["sample_id"] for row in rows if row.get("review_asset")}
    decision_ids = [row["sample_id"] for row in decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("duplicate review decisions")
    if not expected_primary.issubset(decision_ids) or not set(decision_ids).issubset(allowed_review):
        missing = sorted(expected_primary.difference(decision_ids))
        extra = sorted(set(decision_ids).difference(allowed_review))
        raise ValueError(f"review decision coverage mismatch: missing={missing[:3]}, extra={extra[:3]}")
    promoted = 0
    for decision in decisions:
        row = indexed_rows[decision["sample_id"]]
        for decision_field, manifest_field in IDENTITY_FIELD_PAIRS:
            if decision.get(decision_field, "") != row.get(manifest_field, ""):
                raise ValueError(
                    f"review identity mismatch for {decision_field}: {row['sample_id']}"
                )
        if decision.get("class_name", "") != row["candidate_id"]:
            raise ValueError(f"review class mismatch: {row['sample_id']}")
        promoted += int(promote(row, decision))

    eligible = [row for row in rows if row["training_eligible"].lower() == "true"]
    unique_counts = {
        class_name: {
            split: len(
                {
                    row["frame_sha256"]
                    for row in eligible
                    if row["candidate_id"] == class_name and row["split"] == split
                }
            )
            for split in SPLITS
        }
        for class_name in CLASS_NAMES
    }
    deficits = [
        f"{class_name} {split}: {unique_counts[class_name][split]} unique reviewed images < {minimum}"
        for class_name in CLASS_NAMES
        for split, minimum in MINIMUM_UNIQUE_IMAGES.items()
        if unique_counts[class_name][split] < minimum
    ]
    shutil.copytree(dataset, output)
    write_csv(output / "candidate_sample_manifest.csv", rows)
    summary_path = output / "dataset_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        training_eligible_rows=len(eligible),
        reviewed_unique_image_counts=unique_counts,
        review_coverage_deficits=deficits,
        anatomical_approval="IMPORTED_EXACT_GEOMETRY_DECISIONS",
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    result = {
        "schema": "sonogym-tp-surface-core-review-application-v1",
        "decision_rows": len(decisions),
        "promoted_rows": promoted,
        "eligible_unique_image_counts": unique_counts,
        "blocking_conditions": deficits,
        "status": "TRAINING_READY_FOR_PREFLIGHT" if not deficits else "LABEL_REVIEW_INCOMPLETE",
    }
    (output / "review_application_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    args = parse_args()
    result = apply_review(args.dataset_dir, args.decisions, args.output_dir)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "TRAINING_READY_FOR_PREFLIGHT" else 3


if __name__ == "__main__":
    raise SystemExit(main())
