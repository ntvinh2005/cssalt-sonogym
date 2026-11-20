#!/usr/bin/env python3
"""Seal previously opened synchronized captures for SAP development use."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


SESSION_SPECS = (
    ("s0010_tp_surface_core_dataset_v6/51811811/sessions/s0010tpscv6_train_00", "train", (("L4", 0, 25), ("L5", 25, 50))),
    ("s0010_tp_surface_core_dataset_v6/51811811/sessions/s0010tpscv6_train_01", "train", (("L4", 0, 25), ("L5", 25, 50))),
    ("s0010_tp_surface_core_dataset_v6/51811811/sessions/s0010tpscv6_train_02", "train", (("L4", 0, 25), ("L5", 25, 50))),
    ("s0010_tp_surface_core_dataset_v6/51811811/sessions/s0010tpscv6_train_03", "train", (("L4", 0, 25), ("L5", 25, 50))),
    ("s0010_tp_surface_core_dataset_v10/51831979/sessions/s0010tpscv10_train_l4_00", "train", (("L4", 0, 42),)),
    ("s0010_tp_surface_core_dataset_v10/51831979/sessions/s0010tpscv10_train_l5_00", "train", (("L5", 0, 72),)),
    ("s0010_tp_surface_core_proxy_l4_targeted_supplement_v1/51992814/sessions/s0010tpscv12_train_targeted_00", "train", (("L4", 0, 48),)),
    ("s0010_tp_surface_core_proxy_l4_coverage_repair_v1/52007784/sessions/s0010tpscv13_train_targeted_00", "train", (("L4", 0, 48),)),
    ("s0010_tp_surface_core_proxy_l4_coverage_repair_v3/52008931/sessions/s0010tpscv15_train_targeted_00", "train", (("L4", 0, 48),)),
    ("s0010_tp_surface_core_proxy_l4_coverage_repair_v4/52009585/sessions/s0010tpscv16_train_targeted_00", "train", (("L4", 0, 48),)),
    ("s0010_tp_surface_core_proxy_l4_coverage_repair_v4/52009585/sessions/s0010tpscv16_train_targeted_01", "train", (("L4", 0, 48),)),
    ("s0010_tp_surface_core_dataset_v6/51811811/sessions/s0010tpscv6_cal_00", "calibration", (("L4", 0, 25), ("L5", 25, 50))),
    ("s0010_tp_surface_core_dataset_v8/51817719/sessions/s0010tpscv8_cal_l4_00", "calibration", (("L4", 0, 24),)),
    ("s0010_tp_surface_core_dataset_v10/51831979/sessions/s0010tpscv10_cal_l5_00", "calibration", (("L5", 0, 36),)),
    ("s0010_tp_surface_core_proxy_l4_targeted_supplement_v1/51992814/sessions/s0010tpscv12_cal_targeted_00", "calibration", (("L4", 0, 24),)),
    ("s0010_tp_surface_core_proxy_l4_coverage_repair_v4/52009585/sessions/s0010tpscv16_cal_targeted_00", "calibration", (("L4", 0, 48),)),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_protocol(
    runtime_root: Path,
    implementation_commit: str,
    session_specs: tuple[
        tuple[str, str, tuple[tuple[str, int, int], ...]], ...
    ] = SESSION_SPECS,
) -> dict[str, Any]:
    capture_root = runtime_root / "outputs/unity_two_stage_transfer"
    sessions: list[dict[str, Any]] = []
    level_order: dict[str, list[dict[str, Any]]] = {}
    for relative, split, ranges in session_specs:
        session_path = (capture_root / relative).resolve()
        session_id = session_path.name
        required = (
            session_path / "verification_report.json",
            session_path / "public/ultrasound.npz",
            session_path / "public/frames.jsonl",
            session_path / "privileged/raw_labels.npz",
            session_path / "privileged/pixel_to_volume_coordinates.npz",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing synchronized capture files: {missing}")
        verification = json.loads(required[0].read_text(encoding="utf-8"))
        if verification.get("verdict") != "PROJECTION_PARITY_PASS":
            raise ValueError(f"projection parity did not pass: {session_path}")
        frame_count = len((session_path / "public/frames.jsonl").read_text().splitlines())
        if ranges[0][1] != 0 or ranges[-1][2] != frame_count:
            raise ValueError(f"level ranges do not cover {session_id}")
        if any(left[2] != right[1] for left, right in zip(ranges, ranges[1:])):
            raise ValueError(f"level ranges are not contiguous: {session_id}")
        sessions.append(
            {
                "session_id": session_id,
                "split": split,
                "source_session_path": str(session_path),
                "source_role": "previously_opened_development",
                "frame_count": frame_count,
                "verification_sha256": sha256(required[0]),
            }
        )
        level_order[session_id] = [
            {
                "level": level,
                "start_index": start,
                "stop_index_exclusive": stop,
            }
            for level, start, stop in ranges
        ]
    return {
        "protocol_id": "sonogym-s0010-sap-validation-development-v1-existing-captures",
        "status": "PRE_ANALYSIS_FROZEN",
        "source": {
            "implementation_commit": implementation_commit,
            "capture_status": "previously_opened_development",
            "selection_rule": "projection-parity-passed sessions with frozen source-level coverage",
        },
        "scope": {
            "patient_id": "s0010",
            "ultrasound_mode": "conv",
            "development_only": True,
            "independent_holdout": False,
            "runtime_classifier_geometry_access": False,
        },
        "sampling": {"waypoint_level_order_by_session": level_order},
        "sessions": sessions,
        "label_gate": {
            "alignment_tolerance_pixels": 4.0,
            "minimum_surface_columns": 5,
            "minimum_aligned_fraction": 0.40,
            "minimum_aligned_run_columns": 5,
            "maximum_median_distance_pixels": 4.0,
            "minimum_local_gradient": 0.10,
            "minimum_local_contrast": 0.02,
            "minimum_posterior_shadow": 0.02,
            "ambiguous_excluded_from_binary_targets": True,
            "minimum_matched_acceptance_rate": 0.50,
            "maximum_mismatched_false_alignment_rate": 0.10,
            "minimum_correct_projection_ranking_rate": 0.70,
        },
        "change_policy": [
            "Do not treat these previously opened sessions as independent holdout evidence.",
            "Do not expose privileged geometry to classifier features.",
            "Do not change label criteria after reading the generated results.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = build_protocol(args.runtime_root.resolve(), args.implementation_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    digest = sha256(args.output)
    args.output.with_suffix(".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"sessions": len(document["sessions"]), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
