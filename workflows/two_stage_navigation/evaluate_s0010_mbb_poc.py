#!/usr/bin/env python3
"""Aggregate the fixed-s0010 Stage 1 and TP active-acquisition evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage1-evaluation", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    stage1 = json.loads(args.stage1_evaluation.read_text())
    stage1_pass = stage1.get("decision") == "PASS"
    stage2_reports: list[dict[str, Any]] = []
    for session in protocol["execution"]["sessions"]:
        path = args.run_dir / "stage2" / session["session_id"] / "results/evaluation_report.json"
        if not path.is_file():
            raise RuntimeError(f"missing Stage 2 evaluation: {path}")
        stage2_reports.append(json.loads(path.read_text()))
    target_rows = [row for report in stage2_reports for row in report["target_reports"]]
    target_successes = sum(bool(row["proxy_target_view_successful"]) for row in target_rows)
    all_target_views = target_successes == len(target_rows) == 8
    partial_pass = bool(stage1_pass and all_target_views)
    errors = [
        float(row["localization_error_mm"])
        for row in target_rows
        if row["localization_error_mm"] is not None
    ]
    actor_accepts = sum(report["actor_outcome"] == "TARGET_VIEWS_CONFIRMED" for report in stage2_reports)
    report = {
        "schema": "sonogym-s0010-mbb-end-to-end-poc-result-v1",
        "patient_id": "s0010",
        "session_count": len(stage2_reports),
        "stage1": {
            "decision": stage1.get("decision"),
            "passed_sessions": stage1.get("metrics", {}).get("complete_map_session_count"),
            "required_sessions": 4,
            "landmark_count": 7,
            "median_localization_error_mm": stage1.get("metrics", {}).get("median_localization_error_mm"),
            "maximum_localization_error_mm": stage1.get("metrics", {}).get("maximum_localization_error_mm"),
        },
        "stage2": {
            "actor_accept_sessions": actor_accepts,
            "target_attempt_count": len(target_rows),
            "target_success_count": target_successes,
            "valid_view_success_rate": target_successes / len(target_rows),
            "fresh_confirmation_success_count": sum(
                bool(row["fresh_confirmations"])
                and all(item["proxy_state"] == "PROXY_POSITIVE" for item in row["fresh_confirmations"])
                for row in target_rows
            ),
            "mean_tp_interface_localization_error_mm": None if not errors else mean(errors),
            "maximum_tp_interface_localization_error_mm": None if not errors else max(errors),
            "target_reports": target_rows,
        },
        "mbb_intervention": {
            "target_region_localized": False,
            "executed": False,
            "success": False,
            "status": "BLOCKED_GROUND_TRUTH",
            "reasons": sorted(
                {
                    reason
                    for item in stage2_reports
                    for reason in item["mbb_contract_failure_reasons"]
                }
            ),
            "simulator_capability": "no validated MBB target region or needle/injection workflow",
        },
        "end_to_end_success_rate": 0.0,
        "failure_rate": 0.0 if partial_pass else 1.0,
        "abstention_rate": 1.0 - actor_accepts / len(stage2_reports),
        "milestone_verdict": "PARTIAL_PASS" if partial_pass else "BLOCKED",
        "scientific_scope": (
            "fixed-s0010 simulator proxy; TP surface-core view acquisition only; "
            "not whole-TP, SAP, nerve, injection-target or clinical validation"
        ),
        "historical_stage2_confirmation": "CONFIRMATION_NO_GO",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"milestone_verdict": report["milestone_verdict"], "target_successes": target_successes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

