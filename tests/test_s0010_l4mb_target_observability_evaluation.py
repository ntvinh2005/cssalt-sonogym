"""Tests for clean-image evidence in the L5 target observability evaluator."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PATH = Path(__file__).parents[1] / "workflows/two_stage_navigation/evaluate_s0010_l4mb_target_observability.py"
SPEC = importlib.util.spec_from_file_location("l4mb_observability_evaluation", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ObservabilityEvaluationTests(unittest.TestCase):
    def test_shadow_contrast_uses_clean_image_only(self) -> None:
        image = np.zeros((200, 150), dtype=np.float64)
        image[98:102, 40] = 1.0
        mask = np.zeros_like(image, dtype=bool)
        mask[100, 40] = True
        value = MODULE.posterior_shadow_contrast(image, mask)
        self.assertIsNotNone(value)
        self.assertGreater(value, 0.0)

    def test_target_margin_fails_closed_for_empty_projection(self) -> None:
        projected = {
            "sap_core": np.zeros((200, 150), dtype=bool),
            "tp_core": np.zeros((200, 150), dtype=bool),
        }
        self.assertEqual(MODULE.target_margin(projected), -1)

    def test_nearest_track_distance_without_tracks_is_missing(self) -> None:
        mask = np.zeros((200, 150), dtype=bool)
        mask[100, 50] = True
        self.assertIsNone(MODULE.nearest_track_distance(mask, []))


if __name__ == "__main__":
    unittest.main()
