#!/usr/bin/env python3
"""Merge independently captured SAP development datasets without duplicating images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LEVELS = ("L4", "L5")
POSITIVE = "SAP_RELATED_CORTICAL_INTERFACE"
NEGATIVE = "OTHER_OR_BACKGROUND"
MINIMUM = {
    "train": {POSITIVE: 30, NEGATIVE: 30},
    "calibration": {POSITIVE: 10, NEGATIVE: 10},
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    gate = protocol["label_gate"]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, Any]] = []
    merged_features: list[np.ndarray] = []
    merged_patches: list[np.ndarray] = []
    merged_proposal_rows: list[dict[str, Any]] = []
    merged_proposal_features: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for dataset in [path.resolve() for path in args.input]:
        report_path = dataset / "readiness_report.json"
        manifest_path = dataset / "sample_manifest.csv"
        proposal_manifest_path = dataset / "proposal_manifest.csv"
        arrays_path = dataset / "dataset_arrays.npz"
        report = json.loads(report_path.read_text())
        if report.get("phase") != "development":
            raise ValueError("only development datasets may be merged for fitting")
        rows = read_csv(manifest_path)
        arrays = np.load(arrays_path)
        features = arrays["features"]
        patches = arrays["raw_patches"]
        proposal_features = arrays["proposal_features"]
        if len(features) != len(patches):
            raise ValueError("dataset feature/patch arrays are misaligned")
        for row in rows:
            old_index = int(row["feature_index"])
            row = dict(row)
            row["feature_index"] = len(merged_features)
            merged_rows.append(row)
            merged_features.append(features[old_index])
            merged_patches.append(patches[old_index])
        for proposal_row in read_csv(proposal_manifest_path):
            old_index = int(proposal_row["proposal_feature_index"])
            proposal_row = dict(proposal_row)
            proposal_row["proposal_feature_index"] = len(merged_proposal_features)
            merged_proposal_rows.append(proposal_row)
            merged_proposal_features.append(proposal_features[old_index])
        reports.append(report)
        sources.append(
            {
                "dataset": str(dataset),
                "manifest_sha256": file_sha256(manifest_path),
                "proposal_manifest_sha256": file_sha256(proposal_manifest_path),
                "arrays_sha256": file_sha256(arrays_path),
                "readiness_sha256": file_sha256(report_path),
            }
        )

    if len({row["sample_id"] for row in merged_rows}) != len(merged_rows):
        raise ValueError("duplicate sample IDs across source datasets")
    hash_splits: dict[str, set[str]] = {}
    for row in merged_rows:
        hash_splits.setdefault(row["frame_sha256"], set()).add(row["split"])
    cross_split = {key: values for key, values in hash_splits.items() if len(values) > 1}
    if cross_split:
        raise ValueError(f"exact image leakage across splits: {len(cross_split)} hashes")

    counts: dict[str, dict[str, dict[str, int]]] = {}
    correspondence: dict[str, Any] = {}
    readiness: dict[str, Any] = {}
    for level in LEVELS:
        counts[level] = {}
        for split in ("train", "calibration"):
            counts[level][split] = {}
            for label in (POSITIVE, NEGATIVE):
                counts[level][split][label] = len(
                    {
                        row["frame_sha256"]
                        for row in merged_rows
                        if row["conditioned_level"] == level
                        and row["split"] == split
                        and row["provisional_label"] == label
                    }
                )
        matched_pass = sum(
            int(report["correspondence"][level]["matched_pass"]) for report in reports
        )
        matched_total = sum(
            int(report["correspondence"][level]["matched_total"]) for report in reports
        )
        mismatch_pass = sum(
            int(report["correspondence"][level]["mismatched_pass"]) for report in reports
        )
        mismatch_total = sum(
            int(report["correspondence"][level]["mismatched_total"]) for report in reports
        )
        ranking_wins = sum(
            int(report["correspondence"][level]["ranking_wins"]) for report in reports
        )
        ranking_total = sum(
            int(report["correspondence"][level]["ranking_total"]) for report in reports
        )
        matched_rate = matched_pass / matched_total
        mismatch_rate = mismatch_pass / mismatch_total
        ranking_rate = ranking_wins / ranking_total
        label_pass = bool(
            matched_rate >= float(gate["minimum_matched_acceptance_rate"])
            and mismatch_rate <= float(gate["maximum_mismatched_false_alignment_rate"])
            and ranking_rate >= float(gate["minimum_correct_projection_ranking_rate"])
        )
        correspondence[level] = {
            "matched_pass": matched_pass,
            "matched_total": matched_total,
            "matched_acceptance_rate": matched_rate,
            "mismatched_pass": mismatch_pass,
            "mismatched_total": mismatch_total,
            "mismatched_false_alignment_rate": mismatch_rate,
            "ranking_wins": ranking_wins,
            "ranking_total": ranking_total,
            "correct_projection_ranking_rate": ranking_rate,
            "label_validation_pass": label_pass,
        }
        deficits: list[str] = []
        for split, requirements in MINIMUM.items():
            for label, minimum in requirements.items():
                actual = counts[level][split][label]
                if actual < minimum:
                    deficits.append(f"{split}:{label}={actual}<{minimum}")
        if not label_pass:
            deficits.append("matched/mismatched correspondence gate failed")
        readiness[level] = {
            "annotation_status": (
                "SIMULATOR_PROXY_SAP_LABEL_READY" if not deficits else "SIMULATOR_PROXY_SAP_LABEL_NOT_READY"
            ),
            "dataset_status": "READY" if not deficits else "BLOCKED",
            "deficits": deficits,
        }

    write_csv(output / "sample_manifest.csv", merged_rows)
    write_csv(output / "proposal_manifest.csv", merged_proposal_rows)
    np.savez_compressed(
        output / "dataset_arrays.npz",
        features=np.asarray(merged_features, dtype=np.float64),
        proposal_features=np.asarray(merged_proposal_features, dtype=np.float64),
        raw_patches=np.asarray(merged_patches, dtype=np.float32),
    )
    report = {
        "schema": "s0010-sap-level-conditioned-dataset-v1",
        "phase": "development",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_sha256(args.protocol),
        "sources": sources,
        "preprocessing": reports[0]["preprocessing"],
        "counts": counts,
        "correspondence": correspondence,
        "readiness": readiness,
        "actor_feature_privileged_fields": [],
        "cross_split_exact_duplicate_count": 0,
    }
    (output / "readiness_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if all(value["dataset_status"] == "READY" for value in readiness.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
