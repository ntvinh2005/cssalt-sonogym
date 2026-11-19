#!/usr/bin/env python3
"""CPU tests for the L4 proxy waypoint planner."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "workflows/two_stage_navigation/plan_tp_surface_core_l4_waypoints.py"
SPEC = importlib.util.spec_from_file_location("l4_waypoint_planner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PLANNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLANNER
SPEC.loader.exec_module(PLANNER)


def candidate(sample: str, frame: str, pixels: int, *, uncertain: int = 0) -> dict[str, str]:
    return {
        "sample_id": sample,
        "frame_sha256": frame,
        "projection_status": "VERIFIED",
        "candidate_id": PLANNER.L4,
        "review_target": "REVIEW_POSITIVE",
        "l4_core_points_in_patch": str(max(1, pixels)),
        "l5_core_points_in_patch": "0",
        "l4_uncertain_points_in_patch": str(uncertain),
        "l5_uncertain_points_in_patch": "0",
        "adequate_projected_support": "True",
        "unique_core_pixels": str(pixels),
        "interior_core_feature_cells": str(pixels // 2),
    }


class WaypointPlannerTests(unittest.TestCase):
    def test_best_supported_proposal_becomes_anchor(self) -> None:
        rows = []
        annotations = {}
        for index in range(6):
            frame = f"frame-{index}"
            rows.extend(
                [candidate(f"first-{index}", frame, 2), candidate(f"better-{index}", frame, 30)]
            )
            annotations[frame] = {
                "requested_waypoint_pose_index": [145 + index, 140, 2.9],
                "realized_pose_index": [145 + index, 140, 2.9],
            }
        selected = PLANNER.select_positive_anchors(rows, annotations)
        self.assertTrue(all(row["anchor_id"].startswith("better-") for row in selected))

    def test_weak_or_uncertain_support_is_not_an_anchor(self) -> None:
        weak = candidate("weak", "w", 4)
        uncertain = candidate("uncertain", "u", 30, uncertain=1)
        self.assertTrue(PLANNER.is_l4_positive(weak))
        self.assertFalse(PLANNER.is_l4_positive(uncertain))

    def test_bank_is_balanced_unique_and_executable(self) -> None:
        positives = [
            {
                "anchor_id": f"p{i}",
                "anchor_type": PLANNER.POSITIVE,
                "frame_sha256": f"pf{i}",
                "requested_pose_index": [140 + i * 2, 140 + i, 2.86],
            }
            for i in range(7)
        ]
        confusers = [
            {
                "anchor_id": f"n{i}",
                "anchor_type": PLANNER.FALSE_TRIGGER if i % 2 else PLANNER.COMPETING_L5,
                "frame_sha256": f"nf{i}",
                "requested_pose_index": [140 + i * 2, 124 + i, 2.9],
            }
            for i in range(10)
        ]
        bank = PLANNER.build_bank(positives, confusers, session_name="train_a", scale=0.45)
        self.assertEqual(len(bank), 48)
        self.assertEqual(len({tuple(row["pose_index"]) for row in bank}), 48)
        self.assertEqual(sum(row["target_role"] == PLANNER.POSITIVE for row in bank), 24)
        for first, second in zip(bank[::2], bank[1::2]):
            translation, angle = PLANNER.pose_separation(first["pose_index"], second["pose_index"])
            self.assertTrue(translation > 3.0 or angle > 2.0)

    def test_offsets_are_split_specific(self) -> None:
        pose = [150.0, 140.0, 2.88]
        generated = {
            tuple(PLANNER.offset_pose(pose, PLANNER.BASE_OFFSETS[0], spec["offset_scale"]))
            for spec in PLANNER.SESSION_SPECS.values()
        }
        self.assertEqual(len(generated), len(PLANNER.SESSION_SPECS))


if __name__ == "__main__":
    unittest.main()
