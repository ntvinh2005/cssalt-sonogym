#!/usr/bin/env python3
"""CPU tests for the frozen Stage 1 confirmation and demo."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = REPOSITORY_ROOT / "workflows" / "two_stage_navigation"
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
for path in (WORKFLOW_ROOT, EVALUATION_ROOT):
    sys.path.insert(0, str(path))

from render_stage1_confirmation_demo import canvas_frame  # noqa: E402
from run_stage1_confirmation import (  # noqa: E402
    directional_peak,
    file_sha256,
    infer,
    recursive_keys,
    validate_seal,
    write_seal,
)
from two_stage_navigation.stage1_pipeline import ActorPatchCandidate  # noqa: E402


ACTOR_PROTOCOL = (
    REPOSITORY_ROOT
    / "experiments"
    / "unity_two_stage_transfer"
    / "stage1_confirmation_actor_protocol_v2.yaml"
)
EVALUATION_PROTOCOL = (
    REPOSITORY_ROOT
    / "experiments"
    / "unity_two_stage_transfer"
    / "stage1_confirmation_evaluation_protocol_v2.yaml"
)


def candidate(coordinate: float, sequence: int, direction: str) -> ActorPatchCandidate:
    return ActorPatchCandidate(
        feature=np.asarray([coordinate], dtype=np.float32),
        coordinate_mm=coordinate,
        frame_sequence=sequence,
        direction=direction,  # type: ignore[arg-type]
        track_id=f"track-{sequence}",
        interface_row=100.0,
        column=75.0,
    )


class ConfirmationProtocolTests(unittest.TestCase):
    def test_protocols_are_sealed_and_evaluator_is_separate(self) -> None:
        actor_hash = validate_seal(ACTOR_PROTOCOL)
        validate_seal(EVALUATION_PROTOCOL)
        actor = yaml.safe_load(ACTOR_PROTOCOL.read_text(encoding="utf-8"))
        evaluator = yaml.safe_load(EVALUATION_PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(evaluator["actor_protocol"]["sha256"], actor_hash)
        self.assertNotIn("reference_coordinate_home_relative_mm", recursive_keys(actor))
        self.assertIn(
            "reference_coordinate_home_relative_mm", recursive_keys(evaluator)
        )

    def test_confirmation_starts_are_new_and_bounded(self) -> None:
        actor = yaml.safe_load(ACTOR_PROTOCOL.read_text(encoding="utf-8"))
        development = yaml.safe_load(
            (
                REPOSITORY_ROOT
                / "experiments"
                / "unity_two_stage_transfer"
                / "protocol.yaml"
            ).read_text(encoding="utf-8")
        )
        first_confirmation = yaml.safe_load(
            (
                REPOSITORY_ROOT
                / "experiments"
                / "unity_two_stage_transfer"
                / "stage1_confirmation_actor_protocol.yaml"
            ).read_text(encoding="utf-8")
        )
        sessions = actor["sessions"]
        self.assertEqual(
            [row["session_id"] for row in sessions],
            ["cf2_00", "cf2_01", "cf2_02", "cf2_03"],
        )
        old_pairs = {
            (float(row["patient_x_index"]), float(row["angle_rad"]))
            for row in development["sessions"] + first_confirmation["sessions"]
        }
        new_pairs = {
            (float(row["patient_x_index"]), float(row["angle_rad"]))
            for row in sessions
        }
        self.assertEqual(len(new_pairs), 4)
        self.assertTrue(old_pairs.isdisjoint(new_pairs))
        self.assertTrue(all(132.0 <= x <= 144.0 for x, _ in new_pairs))
        self.assertTrue(all(2.7 <= angle <= 3.0 for _, angle in new_pairs))

    def test_resource_and_scientific_bounds_are_fixed(self) -> None:
        actor = yaml.safe_load(ACTOR_PROTOCOL.read_text(encoding="utf-8"))
        evaluator = yaml.safe_load(EVALUATION_PROTOCOL.read_text(encoding="utf-8"))
        resources = actor["resource_bound_if_later_approved"]
        self.assertEqual(resources["account"], "anest-dept")
        self.assertEqual(resources["cpu_cores"], 4)
        self.assertEqual(resources["gpu_count"], 1)
        self.assertEqual(resources["memory"], "30gb")
        self.assertEqual(resources["walltime"], "00:15:00")
        self.assertEqual(
            actor["inference"]["spatial_map"]["acceptance_score"], 0.50
        )
        self.assertEqual(
            actor["inference"]["spatial_map"]["coordinate_rule"],
            "accepted_event_support_midpoint",
        )
        self.assertEqual(evaluator["criteria"]["required_correct_sessions"], 4)
        self.assertEqual(evaluator["criteria"]["maximum_wrong_stops"], 0)


class ConfirmationActorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = yaml.safe_load(ACTOR_PROTOCOL.read_text(encoding="utf-8"))

    def test_recursive_keys_detect_nested_privileged_field(self) -> None:
        self.assertIn(
            "reference_coordinates",
            recursive_keys({"nested": [{"reference_coordinates": [1.0]}]}),
        )

    def test_bidirectional_evidence_produces_one_l4_estimate(self) -> None:
        candidates = [
            candidate(-100.0, 1, "FORWARD"),
            candidate(-80.0, 2, "FORWARD"),
            candidate(-80.0, 3, "FORWARD"),
            candidate(-80.0, 4, "REVERSE"),
            candidate(-80.0, 5, "REVERSE"),
            candidate(-60.0, 6, "REVERSE"),
        ]
        shared = np.asarray([0.1, 0.9, 0.9, 0.9, 0.9, 0.1])
        anatomy = np.asarray([0.1, 0.95, 0.95, 0.95, 0.95, 0.1])
        result, events = infer(candidates, shared, anatomy, self.protocol)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["coordinate_mm"], -80.0)
        self.assertEqual(sum(event.accepted for event in events), 1)

    def test_one_direction_abstains(self) -> None:
        candidates = [
            candidate(-100.0, 1, "FORWARD"),
            candidate(-80.0, 2, "FORWARD"),
            candidate(-80.0, 3, "FORWARD"),
            candidate(-60.0, 4, "FORWARD"),
        ]
        scores = np.asarray([0.1, 0.9, 0.9, 0.1])
        result, _ = infer(candidates, scores, scores, self.protocol)
        self.assertEqual(result["status"], "ABSTAIN")

    def test_support_midpoint_rule_uses_the_accepted_interval(self) -> None:
        protocol = yaml.safe_load(ACTOR_PROTOCOL.read_text(encoding="utf-8"))
        protocol["inference"]["spatial_map"][
            "coordinate_rule"
        ] = "accepted_event_support_midpoint"
        candidates = [
            candidate(-100.0, 1, "FORWARD"),
            candidate(-80.5, 2, "FORWARD"),
            candidate(-80.5, 3, "FORWARD"),
            candidate(-80.5, 4, "REVERSE"),
            candidate(-80.5, 5, "REVERSE"),
            candidate(-80.0, 6, "FORWARD"),
            candidate(-80.0, 7, "FORWARD"),
            candidate(-80.0, 8, "REVERSE"),
            candidate(-80.0, 9, "REVERSE"),
            candidate(-60.0, 10, "REVERSE"),
        ]
        shared = np.asarray([0.1, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.1])
        anatomy = shared.copy()
        result, events = infer(candidates, shared, anatomy, protocol)
        accepted = [event for event in events if event.accepted]
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(len(accepted), 1)
        expected = sum(accepted[0].support_interval_mm) / 2.0
        self.assertEqual(result["coordinate_mm"], expected)

    def test_directional_peak_requires_distinct_frames(self) -> None:
        candidates = [
            candidate(-80.0, 1, "FORWARD"),
            candidate(-80.0, 2, "FORWARD"),
        ]
        peak = directional_peak(candidates, np.asarray([0.8, 0.9]), "FORWARD", 0.5)
        self.assertIsNotNone(peak)
        self.assertEqual(peak["coordinate_mm"], -80.0)  # type: ignore[index]

    def test_seal_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.json"
            path.write_text("{}\n", encoding="utf-8")
            digest = write_seal(path)
            self.assertEqual(digest, file_sha256(path))
            validate_seal(path)
            path.write_text('{"changed": true}\n', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_seal(path)


class ConfirmationRenderTests(unittest.TestCase):
    def test_canvas_keeps_actor_and_evaluator_labels_explicit(self) -> None:
        canvas = canvas_frame(
            np.linspace(0.0, 1.0, 200 * 150, dtype=np.float32).reshape(200, 150),
            {
                "frame_sequence": 10,
                "phase": "OUTBOUND_CAUDAL",
                "relative_position_home_mm": [0.0, -15.0],
            },
            [
                {
                    "frame_sequence": 10,
                    "direction": "FORWARD",
                    "coordinate_mm": -80.0,
                    "combined_probability": 0.9,
                }
            ],
            [
                {
                    "frame_sequence": 10,
                    "status": "SUCCESS",
                    "coordinate_mm": -80.0,
                    "reason": None,
                }
            ],
            {"actor_status": "SUCCESS"},
            {"reference_coordinate_mm_evaluator_only": -83.0},
            0.0,
            1.0,
        )
        self.assertEqual(canvas.shape, (600, 1050, 3))
        self.assertGreater(int(np.max(canvas)), 0)


if __name__ == "__main__":
    unittest.main()
