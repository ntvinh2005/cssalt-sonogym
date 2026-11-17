#!/usr/bin/env python3
"""Focused CPU tests for the Stage 2 angular-selection audit."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import yaml

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflows" / "two_stage_navigation"))

from audit_stage2_angular_selection import (
    associate_actions,
    audit_start,
    orientation_error_deg,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class AngularSelectionAuditTests(unittest.TestCase):
    def test_angle_error_uses_wrapping(self) -> None:
        self.assertAlmostEqual(orientation_error_deg(-math.pi + 0.1, math.pi - 0.1), math.degrees(0.2))

    def test_action_association_respects_phase_and_zero_action(self) -> None:
        frames = [
            {"frame_sequence": 0, "phase": "coarse", "action_step_count": 2},
            {"frame_sequence": 1, "phase": "orientation_refine", "action_step_count": 0},
        ]
        actions = [
            {"phase": "phase1_handoff"},
            {"phase": "coarse", "step": 1},
            {"phase": "coarse", "step": 2},
        ]
        grouped = associate_actions(frames, actions)
        self.assertEqual([row["step"] for row in grouped[0]], [1, 2])
        self.assertEqual(grouped[1], ())

    def test_join_and_hindsight_detect_selection_failure(self) -> None:
        protocol = {
            "geometry": {"label_resolution_mm": 1.5, "stage2_handoff_angle_rad": 2.85},
            "decision": {
                "validity_probability_threshold": 0.5666666666666667,
                "maximum_neighbor_translation_mm": 4.5,
                "maximum_neighbor_angle_deg": 6.0,
                "maximum_quality_drop": 0.08,
                "required_distinct_hash_observations": 3,
                "minimum_distinct_view_translation_mm": 1.0,
                "minimum_distinct_view_angle_deg": 1.0,
            },
            "evaluation_gate": {
                "maximum_position_error_mm": 10.0,
                "maximum_orientation_error_deg": 10.0,
                "minimum_l4_visible_fraction": 0.2,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            start = Path(directory)
            frames = []
            annotations = []
            actions = []
            angles = [2.89, 2.91, 2.93, 2.95, 2.97]
            qualities = [0.50, 0.52, 0.54, 0.60, 0.58]
            for sequence, (angle, quality) in enumerate(zip(angles, qualities)):
                digest = f"{sequence + 1:064x}"
                frames.append(
                    {
                        "action_step_count": 1,
                        "frame_sequence": sequence,
                        "frame_sha256": digest,
                        "image_valid": True,
                        "phase": "orientation_refine",
                        "producer_advanced": True,
                        "producer_sequence": sequence + 1,
                        "quality_prediction": quality,
                        "relative_pose": [0.0, 0.0, angle - 2.85],
                        "requested_relative_pose": [0.0, 0.0, angle - 2.85],
                        "session_id": "synthetic",
                        "transition_valid": True,
                        "validity_probability": 0.95,
                    }
                )
                error = orientation_error_deg(angle, 3.14)
                annotations.append(
                    {
                        "frame_sequence": sequence,
                        "frame_sha256": digest,
                        "goal_cmd_pose": [0.0, 0.0, 3.14],
                        "l4_visible_fraction": 0.3,
                        "orientation_error_deg": error,
                        "phase": "orientation_refine",
                        "position_error_mm": 4.0,
                        "realized_pose_index": [0.0, 0.0, angle],
                    }
                )
                actions.append(
                    {
                        "phase": "orientation_refine",
                        "command_position_clamped": False,
                        "command_angle_clamped": False,
                    }
                )
            write_jsonl(start / "public" / "frames.jsonl", frames)
            write_jsonl(start / "privileged" / "annotations.jsonl", annotations)
            write_jsonl(start / "privileged" / "actions.jsonl", actions)
            (start / "results").mkdir(parents=True)
            (start / "results" / "actor_result.json").write_text(
                json.dumps(
                    {
                        "selected_frame_sequence": 3,
                        "preliminary_selected_frame_sequence": None,
                        "fresh_confirmation_frame_sequences": [],
                        "return_orientation_error_deg": 0.1,
                    }
                ),
                encoding="utf-8",
            )
            (start / "results" / "evaluation_report.json").write_text(
                json.dumps({"final_orientation_error_deg": 10.9}), encoding="utf-8"
            )
            rows, summary, _ = audit_start(start, "synthetic", protocol)
            self.assertEqual(summary["primary_cause"], "SELECTION_FAILURE")
            self.assertEqual(summary["fully_valid_frame_sequences"], [4])
            self.assertEqual(summary["fully_valid_actor_eligible_sequences"], [4])
            self.assertEqual(summary["selected"]["frame_sequence"], 3)
            self.assertFalse(summary["selected"]["full_gate"])
            self.assertEqual(
                summary["refinement_sequential_filter_counts"][
                    "actor_eligible_distinct_viewpoints"
                ],
                5,
            )
            self.assertAlmostEqual(rows[-1]["evaluator_orientation_error_deg"], orientation_error_deg(2.97, 3.14))

    def test_join_rejects_inconsistent_evaluator_angle(self) -> None:
        protocol = {
            "geometry": {"label_resolution_mm": 1.5, "stage2_handoff_angle_rad": 2.85},
            "decision": {
                "validity_probability_threshold": 0.5,
                "maximum_neighbor_translation_mm": 4.5,
                "maximum_neighbor_angle_deg": 6.0,
                "maximum_quality_drop": 0.08,
                "required_distinct_hash_observations": 3,
                "minimum_distinct_view_translation_mm": 1.0,
                "minimum_distinct_view_angle_deg": 1.0,
            },
            "evaluation_gate": {
                "maximum_position_error_mm": 10.0,
                "maximum_orientation_error_deg": 10.0,
                "minimum_l4_visible_fraction": 0.2,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            start = Path(directory)
            frame = {
                "action_step_count": 0,
                "frame_sequence": 0,
                "frame_sha256": "1" * 64,
                "image_valid": True,
                "phase": "coarse",
                "producer_advanced": True,
                "producer_sequence": 1,
                "quality_prediction": 0.5,
                "relative_pose": [0.0, 0.0, 0.0],
                "requested_relative_pose": [0.0, 0.0, 0.0],
                "session_id": "synthetic",
                "transition_valid": True,
                "validity_probability": 0.5,
            }
            annotation = {
                "frame_sequence": 0,
                "frame_sha256": "1" * 64,
                "goal_cmd_pose": [0.0, 0.0, 3.14],
                "l4_visible_fraction": 0.3,
                "orientation_error_deg": 123.0,
                "phase": "coarse",
                "position_error_mm": 4.0,
                "realized_pose_index": [0.0, 0.0, 2.85],
            }
            write_jsonl(start / "public" / "frames.jsonl", [frame])
            write_jsonl(start / "privileged" / "annotations.jsonl", [annotation])
            write_jsonl(start / "privileged" / "actions.jsonl", [])
            (start / "results").mkdir(parents=True)
            (start / "results" / "actor_result.json").write_text(
                json.dumps({"selected_frame_sequence": None, "preliminary_selected_frame_sequence": None, "fresh_confirmation_frame_sequences": [], "return_orientation_error_deg": None}),
                encoding="utf-8",
            )
            (start / "results" / "evaluation_report.json").write_text(
                json.dumps({"final_orientation_error_deg": None}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "orientation join mismatch"):
                audit_start(start, "synthetic", protocol)


if __name__ == "__main__":
    unittest.main()
