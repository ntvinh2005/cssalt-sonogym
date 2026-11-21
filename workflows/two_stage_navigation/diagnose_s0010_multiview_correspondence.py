#!/usr/bin/env python3
"""Diagnose interface association on calibration sessions only."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import run_s0010_stage2_multiview_evidence as multiview
import run_s0010_ultrasound_representation_experiment as representation


def quantiles(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--development-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    samples = representation.read_csv(args.development_dataset / "sample_manifest.csv")
    proposals = representation.read_csv(args.development_dataset / "proposal_manifest.csv")
    recorded = representation.RecordedImages()
    report: dict[str, Any] = {
        "schema": "s0010-multiview-correspondence-calibration-diagnostic-v1",
        "split": "calibration",
        "evaluation_opened": False,
        "used_for_final_claim": False,
        "levels": {},
    }
    pair_rows: list[dict[str, Any]] = []
    for level in representation.LEVELS:
        _, bundle = representation.variant_fit(
            level=level,
            variant="topology",
            development_rows=samples,
            images=recorded,
            threshold_grid=[float(value) for value in protocol["selection"]["score_thresholds"]],
            random_seed=int(protocol["base_model"]["random_seed"]),
        )
        sessions = multiview.build_sessions(
            sample_rows=samples,
            proposal_rows=proposals,
            level=level,
            split="calibration",
            bundle=bundle,
            recorded=recorded,
        )
        by_pair: dict[str, list[float]] = defaultdict(list)
        neighbor_values: list[float] = []
        support_by_threshold = {"0.20": 0, "0.25": 0, "0.30": 0, "0.35": 0, "0.45": 0}
        total_frames = 0
        for session in sessions:
            total_frames += len(session.statuses)
            for first in range(len(session.statuses)):
                for second in range(first + 1, len(session.statuses)):
                    distinct = bool(
                        session.translation_mm[first, second]
                        >= float(protocol["viewpoint_contract"]["minimum_translation_mm"])
                        or session.angle_deg[first, second]
                        >= float(protocol["viewpoint_contract"]["minimum_orientation_deg"])
                    )
                    broad = bool(
                        session.translation_mm[first, second] <= 40.0
                        and session.angle_deg[first, second] <= 30.0
                    )
                    if not distinct or not broad:
                        continue
                    value = float(session.correspondences[first, second])
                    pair = ":".join(sorted((session.statuses[first], session.statuses[second])))
                    by_pair[pair].append(value)
                    neighbor_values.append(value)
                    pair_rows.append(
                        {
                            "level": level,
                            "session_id": session.session_id,
                            "first_frame_sequence": int(session.frame_sequences[first]),
                            "second_frame_sequence": int(session.frame_sequences[second]),
                            "first_status": session.statuses[first],
                            "second_status": session.statuses[second],
                            "translation_mm": float(session.translation_mm[first, second]),
                            "angle_deg": float(session.angle_deg[first, second]),
                            "correspondence": value,
                        }
                    )
            for threshold in support_by_threshold:
                supported = np.sum(
                    (
                        (session.correspondences >= float(threshold))
                        & (session.translation_mm <= 40.0)
                        & (session.angle_deg <= 30.0)
                        & (
                            (session.translation_mm >= 2.0)
                            | (session.angle_deg >= 1.0)
                        )
                    ),
                    axis=1,
                )
                support_by_threshold[threshold] += int(np.count_nonzero(supported >= 2))
        report["levels"][level] = {
            "sessions": len(sessions),
            "admissible_frames": total_frames,
            "broad_neighbor_correspondence": quantiles(neighbor_values),
            "correspondence_by_status_pair": {
                key: quantiles(values) for key, values in sorted(by_pair.items())
            },
            "frames_with_at_least_two_supporting_neighbors": support_by_threshold,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    csv_path = args.output.with_suffix(".csv")
    fields = list(pair_rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pair_rows)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
