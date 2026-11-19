#!/usr/bin/env python3
"""CPU tests for calibration-only L4 margin selection."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "workflows/two_stage_navigation/calibrate_tp_surface_core_l4_margin.py"
SPEC = importlib.util.spec_from_file_location("tp_l4_margin_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CALIBRATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CALIBRATION
SPEC.loader.exec_module(CALIBRATION)
CLASSES = np.asarray([0, 1, 2])


def geometry_row(*, l4: int = 0, l5: int = 0, l4u: int = 0, l5u: int = 0):
    return {
        "l4_core_points_in_patch": l4,
        "l5_core_points_in_patch": l5,
        "l4_uncertain_points_in_patch": l4u,
        "l5_uncertain_points_in_patch": l5u,
    }


class L4MarginCalibrationTests(unittest.TestCase):
    def test_zero_margin_matches_frozen_argmax_and_threshold_rule(self) -> None:
        self.assertTrue(
            CALIBRATION.l4_proposal_passes(
                np.asarray([0.30, 0.50, 0.20]),
                CLASSES,
                base_threshold=0.50,
                margin=0.0,
            )
        )
        self.assertFalse(
            CALIBRATION.l4_proposal_passes(
                np.asarray([0.55, 0.40, 0.05]),
                CLASSES,
                base_threshold=0.50,
                margin=0.0,
            )
        )

    def test_margin_rejects_small_l4_advantage(self) -> None:
        probabilities = np.asarray([0.40, 0.52, 0.08])
        self.assertTrue(
            CALIBRATION.l4_proposal_passes(
                probabilities, CLASSES, base_threshold=0.50, margin=0.10
            )
        )
        self.assertFalse(
            CALIBRATION.l4_proposal_passes(
                probabilities, CLASSES, base_threshold=0.50, margin=0.15
            )
        )

    def test_l5_decision_does_not_depend_on_l4_margin(self) -> None:
        probabilities = np.asarray([0.10, 0.20, 0.70])
        self.assertTrue(
            CALIBRATION.l5_proposal_passes(
                probabilities, CLASSES, base_threshold=0.50
            )
        )

    def test_false_positive_strata_remain_separate(self) -> None:
        flags = CALIBRATION.accepted_false_positive_strata(
            [geometry_row(l5=2, l4u=1)]
        )
        self.assertTrue(flags["competing_l5"])
        self.assertTrue(flags["uncertainty"])
        self.assertFalse(flags["clean_negative"])

    def test_clean_negative_requires_no_geometry_overlap(self) -> None:
        flags = CALIBRATION.accepted_false_positive_strata([geometry_row()])
        self.assertTrue(flags["clean_negative"])
        self.assertFalse(flags["other_proxy_exclusion"])

    def test_selection_requires_reduced_clean_false_positives(self) -> None:
        selected, required = CALIBRATION.select_margin(
            [
                {"margin": 0.0, "tp": 10, "fp": 4, "clean_negative_fp": 2},
                {"margin": 0.1, "tp": 9, "fp": 3, "clean_negative_fp": 2},
            ],
            retain_fraction=0.9,
        )
        self.assertEqual(required, 9)
        self.assertIsNone(selected)

    def test_selection_enforces_ninety_percent_true_positive_retention(self) -> None:
        selected, required = CALIBRATION.select_margin(
            [
                {"margin": 0.0, "tp": 10, "fp": 4, "clean_negative_fp": 2},
                {"margin": 0.1, "tp": 8, "fp": 1, "clean_negative_fp": 0},
            ],
            retain_fraction=0.9,
        )
        self.assertEqual(required, 9)
        self.assertIsNone(selected)

    def test_selection_is_deterministic(self) -> None:
        selected, _ = CALIBRATION.select_margin(
            [
                {"margin": 0.0, "tp": 10, "fp": 4, "clean_negative_fp": 2},
                {"margin": 0.1, "tp": 9, "fp": 2, "clean_negative_fp": 1},
                {"margin": 0.2, "tp": 9, "fp": 2, "clean_negative_fp": 1},
            ],
            retain_fraction=0.9,
        )
        self.assertEqual(selected["margin"], 0.1)

    def test_calibration_source_contains_no_fit_call(self) -> None:
        self.assertNotIn(".fit(", SCRIPT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
