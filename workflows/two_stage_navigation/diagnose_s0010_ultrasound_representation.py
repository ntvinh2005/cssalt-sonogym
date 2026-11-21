#!/usr/bin/env python3
"""Run post-selection feature ablations on already opened evaluation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

import run_s0010_ultrasound_representation_experiment as experiment


DIAGNOSTIC_VARIANTS = ("topology", "context", "causal")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--development-dataset", type=Path, required=True)
    parser.add_argument("--evaluation-dataset", type=Path, required=True)
    parser.add_argument("--frozen-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    frozen = json.loads(args.frozen_result.read_text(encoding="utf-8"))
    development_rows = experiment.read_csv(
        args.development_dataset / "sample_manifest.csv"
    )
    evaluation_rows = experiment.read_csv(
        args.evaluation_dataset / "sample_manifest.csv"
    )
    evaluation_proposals = experiment.read_csv(
        args.evaluation_dataset / "proposal_manifest.csv"
    )
    threshold_grid = [
        float(value) for value in protocol["selection"]["threshold_grid"]
    ]
    images = experiment.RecordedImages()
    results: dict[str, object] = {}
    for level in experiment.LEVELS:
        level_results = []
        for variant in DIAGNOSTIC_VARIANTS:
            calibration, bundle = experiment.variant_fit(
                level=level,
                variant=variant,
                development_rows=development_rows,
                images=images,
                threshold_grid=threshold_grid,
                random_seed=int(protocol["model"]["random_seed"]),
            )
            evaluation, _ = experiment.evaluation_metrics(
                rows=evaluation_rows,
                level=level,
                variant=variant,
                bundle=bundle,
                images=images,
            )
            _, coverage = experiment.score_all_images(
                clear_rows=evaluation_rows,
                proposal_rows=evaluation_proposals,
                level=level,
                variant=variant,
                bundle=bundle,
                images=images,
            )
            level_results.append(
                {
                    "variant": variant,
                    "calibration": calibration["calibration"],
                    "evaluation": evaluation,
                    "all_frame_coverage": coverage,
                }
            )
        results[level] = level_results
    report = {
        "schema": "s0010-ultrasound-representation-post-selection-diagnostic-v1",
        "evaluation_was_previously_opened": True,
        "used_for_candidate_selection": False,
        "changes_frozen_selection": False,
        "frozen_selected_variants": frozen["selected_variants"],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
