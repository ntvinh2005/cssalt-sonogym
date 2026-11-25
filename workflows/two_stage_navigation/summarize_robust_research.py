"""Aggregate development effects and measure overlap with the pretraining pose inventory."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from two_stage_navigation.incremental_landmark_map import LANDMARK_IDENTITIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--original-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for relative_name in ("mapping/results.json", "E3-mapping/results.json"):
        path = args.review / relative_name
        if path.exists():
            rows.extend(row for row in json.loads(path.read_text())["sessions"]
                        if row["arm"] == "E3" or relative_name.startswith("mapping/"))
    by_arm = defaultdict(list)
    for row in rows:
        if "point" not in row:
            continue
        for mapper in ("point", "surface"):
            target = "E4" if mapper == "surface" and row["arm"] == "E0" else row["arm"]
            target += "-surface" if mapper == "surface" and row["arm"] == "E3" else ""
            for perception, values in row.get(mapper, {}).items():
                score = values["score"]
                by_arm[(target, perception)].append({
                    "seed": row["seed"], "valid_frames": row["valid_frames"], "planned_frames": row["planned_frames"],
                    "capture_integrity": row["capture_integrity"],
                    "identity_complete": score["complete_map"],
                    "capture_complete_map": row["capture_integrity"] and score["complete_map"],
                    "sweep_seconds": row["sweep_seconds_through_last_valid_capture"],
                    "mapping_seconds": values["mapping_seconds"],
                    "outcomes": {k: v["outcome"] for k, v in score["landmarks"].items()},
                    "coverage": row["oracle_coverage"], "pose_error": row["tracking"],
                })
    summaries = []
    for (arm, perception), cases in by_arm.items():
        counts = Counter(outcome for row in cases for outcome in row["outcomes"].values())
        summaries.append({"arm": arm, "perception": perception,
                          "cases": cases, "counts": dict(counts),
                          "valid_frames": sum(row["valid_frames"] for row in cases),
                          "planned_frames": sum(row["planned_frames"] for row in cases),
                          "identity_complete": sum(row["identity_complete"] for row in cases),
                          "capture_and_map": sum(row["capture_complete_map"] for row in cases),
                          "mapping_seconds": sum(row["mapping_seconds"] for row in cases),
                          "outcomes_per_identity": {identity: dict(Counter(row["outcomes"][identity] for row in cases))
                                                    for identity in LANDMARK_IDENTITIES}})
    report = {"arms": summaries}
    path = args.review / "E3-dataset/split_manifest.json"
    if path.exists():
        metadata = json.loads(path.read_text())
        val = {tuple(row["position_m"]) for row in metadata["records"] if row["split"] == "validation"}
        with args.original_manifest.open() as handle:
            pretraining = [json.loads(row["split_pose_position_m"]) for row in csv.DictReader(handle)
                           if row["included"] == "True" and row["split"] == "train"]
        distances = cdist(list(val), pretraining).min(axis=1)
        report["l5_pretraining_pose_overlap"] = {
            "unique_pilot_validation_poses": len(val), "original_checkpoint_training_poses": len(pretraining),
            "within_4_mm_of_original_checkpoint_training": int((distances < .004).sum()),
            "min_mm": float(1000 * distances.min()), "median_mm": float(1000 * np.median(distances)),
        }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
