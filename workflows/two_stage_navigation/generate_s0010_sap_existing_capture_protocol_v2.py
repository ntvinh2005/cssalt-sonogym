#!/usr/bin/env python3
"""Seal opened development sessions for a second SAP development analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

import generate_s0010_sap_existing_capture_protocol as v1


ADDITIONAL_CALIBRATION = (
    ("s0010_tp_surface_core_dataset_v6/51811811/sessions/s0010tpscv6_dev_00", (("L4", 0, 25), ("L5", 25, 50))),
    ("s0010_tp_surface_core_dataset_v9/51818619/sessions/s0010tpscv9_dev_l4_00", (("L4", 0, 24),)),
    ("s0010_tp_surface_core_proxy_l4_targeted_supplement_v1/51992814/sessions/s0010tpscv12_eval_targeted_00", (("L4", 0, 24),)),
    ("s0010_tp_surface_core_proxy_l4_coverage_repair_v4/52009585/sessions/s0010tpscv16_eval_targeted_00", (("L4", 0, 48),)),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = v1.build_protocol(
        args.runtime_root.resolve(),
        args.implementation_commit,
        v1.SESSION_SPECS
        + tuple(
            (path, "calibration", ranges)
            for path, ranges in ADDITIONAL_CALIBRATION
        ),
    )
    document["protocol_id"] = (
        "sonogym-s0010-sap-validation-development-v2-existing-captures"
    )
    document["source"]["prior_cpu_job"] = 52135074
    document["source"]["additional_session_rule"] = (
        "previously opened development-evaluation sessions reassigned to calibration"
    )
    document["label_gate"]["require_projected_surface_for_matched_control"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"sessions": len(document["sessions"]), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
