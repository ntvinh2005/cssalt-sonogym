#!/usr/bin/env python3
"""Generate the immutable Phase 5A SonoGym navigation start manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()
    if args.count != 32:
        parser.error("Phase 5A requires exactly 32 initial conditions")

    configured_min = np.array([100.0, 70.0, 1.5], dtype=np.float64)
    configured_max = np.array([180.0, 200.0, 3.5], dtype=np.float64)
    operational_min = np.array([100.0, 70.0, 1.0], dtype=np.float64)
    operational_max = np.array([180.0, 200.0, 3.0], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(args.seed))
    requested = rng.uniform(configured_min, configured_max, size=(args.count, 3))
    effective = np.clip(requested, operational_min, operational_max)

    starts = []
    for index in range(args.count):
        starts.append(
            {
                "start_id": f"start_{index:03d}",
                "requested_command_x_index_z_index_angle_rad": requested[index].tolist(),
                "expected_effective_command_after_native_clamp": effective[index].tolist(),
                "native_clamp_expected": bool(
                    not np.array_equal(requested[index], effective[index])
                ),
            }
        )

    manifest = {
        "schema_version": 1,
        "protocol_id": "sonogym-navigation-metric-validation-v1",
        "immutable_after_commit": True,
        "generation": {
            "algorithm": "numpy.random.Generator(numpy.random.PCG64(seed)).uniform",
            "seed": args.seed,
            "draw_order": "32 rows x [patient_x_index, patient_z_index, angle_rad]",
            "configured_initial_range": [
                configured_min.tolist(),
                configured_max.tolist(),
            ],
            "native_operational_range": [
                operational_min.tolist(),
                operational_max.tolist(),
            ],
            "label_resolution_m_per_index": 0.0015,
        },
        "anatomy": {
            "patient_id": "s0010",
            "target": "vertebrae_L4",
            "goal_source": "mean x-z of the L4 STL via VertebraViewer; angle 3.14 rad",
        },
        "count": len(starts),
        "starts": starts,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
