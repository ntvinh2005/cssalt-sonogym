#!/usr/bin/env python3
"""CPU tests for the Stage 1 supervised anatomical-discovery pilot."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = REPOSITORY_ROOT / "workflows" / "two_stage_navigation"
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
for path in (WORKFLOW_ROOT, EVALUATION_ROOT):
    sys.path.insert(0, str(path))

from run_stage1_supervised_pilot import (  # noqa: E402
    PilotSession,
    arrays,
    binary_metrics,
    shuffled_images,
    spatial_gate,
)
from train_stage1_group_a import GroupASessionData  # noqa: E402
from two_stage_navigation.stage1_pipeline import ActorPatchCandidate  # noqa: E402


def candidate(value: float, sequence: int, direction: str) -> ActorPatchCandidate:
    return ActorPatchCandidate(
        feature=np.asarray([value, value + 0.25], dtype=np.float32),
        coordinate_mm=value,
        frame_sequence=sequence,
        direction=direction,  # type: ignore[arg-type]
        track_id=f"track-{sequence}",
        interface_row=50.0,
        column=75.0,
    )


def session(session_id: str, offset: float) -> PilotSession:
    candidates = [
        candidate(offset + index, index, "FORWARD" if index < 2 else "REVERSE")
        for index in range(4)
    ]
    source = GroupASessionData(
        session_id=session_id,
        split="train_dev",
        candidates=candidates,
        primary_labels=np.asarray([0, 1, 1, 0], dtype=np.int8),
        anatomy_classes=np.asarray([-1, 0, 1, -1], dtype=np.int8),
    )
    position = np.asarray(
        [[offset, index, 0.0, 1.0 if index < 2 else -1.0, index / 3.0]
         for index in range(4)],
        dtype=np.float32,
    )
    return PilotSession(source, position, {})


class PilotControlTests(unittest.TestCase):
    def test_binary_metrics_count_false_positives(self) -> None:
        measured = binary_metrics(
            np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 1, 0])
        )
        self.assertEqual(measured["tp"], 1)
        self.assertEqual(measured["fp"], 1)
        self.assertEqual(measured["fn"], 1)
        self.assertEqual(measured["tn"], 1)

    def test_arrays_keep_session_level_inputs_aligned(self) -> None:
        image, position, labels = arrays(
            [session("first", 0.0), session("second", 10.0)],
            anatomy_only=True,
        )
        self.assertEqual(image.shape, (4, 2))
        self.assertEqual(position.shape, (4, 5))
        np.testing.assert_array_equal(labels, [0, 1, 0, 1])
        np.testing.assert_allclose(position[:, 0], [0.0, 0.0, 10.0, 10.0])

    def test_image_shuffle_is_deterministic_and_within_session(self) -> None:
        sessions = [session("first", 0.0), session("second", 100.0)]
        first = shuffled_images(sessions, anatomy_only=False, seed=17)
        second = shuffled_images(sessions, anatomy_only=False, seed=17)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[:4, 0] < 10.0))
        self.assertTrue(np.all(first[4:, 0] >= 100.0))

    def test_spatial_gate_requires_direction_consistency(self) -> None:
        protocol = {
            "evaluation": {
                "minimum_l4_precision": 0.90,
                "minimum_l4_recall": 0.80,
                "maximum_wrong_stops": 0,
                "maximum_median_localization_error_mm": 10.0,
                "maximum_median_forward_reverse_peak_difference_mm": 10.0,
            }
        }
        metrics = {
            "precision": 1.0,
            "recall": 1.0,
            "wrong_stop_count": 0,
            "median_localization_error_mm": 2.0,
            "median_forward_reverse_peak_difference_mm": 12.0,
        }
        self.assertFalse(spatial_gate(metrics, protocol))
        metrics["median_forward_reverse_peak_difference_mm"] = 3.0
        self.assertTrue(spatial_gate(metrics, protocol))


if __name__ == "__main__":
    unittest.main()
