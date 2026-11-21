"""Tests for development-only correspondence checkpoint selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflows/two_stage_navigation"))
import select_s0010_sap_correspondence_checkpoint as selection  # noqa: E402


class SapCorrespondenceSelectionTests(unittest.TestCase):
    def test_gate_pass_precedes_a_larger_ungated_score_gap(self) -> None:
        passing = {
            "calibration_gate_pass": True,
            "calibration": {
                "matched_acceptance_rate": 0.5,
                "mismatched_acceptance_rate": 0.1,
            },
            "calibration_top_one_rate": 0.7,
        }
        failing = {
            "calibration_gate_pass": False,
            "calibration": {
                "matched_acceptance_rate": 0.9,
                "mismatched_acceptance_rate": 0.0,
            },
            "calibration_top_one_rate": 0.9,
        }
        self.assertGreater(
            selection.selection_key(passing), selection.selection_key(failing)
        )


if __name__ == "__main__":
    unittest.main()
