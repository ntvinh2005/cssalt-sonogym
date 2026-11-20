"""Tests for SAP matched and mismatched cortical-interface correspondence."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

WORKFLOW = Path(__file__).resolve().parents[1] / "workflows/two_stage_navigation"
sys.path.insert(0, str(WORKFLOW))
SPEC = importlib.util.spec_from_file_location(
    "sap_correspondence", WORKFLOW / "audit_s0010_sap_correspondence.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeTrack:
    track_id = "synthetic"
    columns = np.arange(18, 132, dtype=np.float64)
    rows = np.full(114, 110.0)
    mean_ridge_score = 1.0
    maximum_jump_pixels = 0.0

    def row_at(self, column: float) -> float:
        return 110.0


class SapCorrespondenceTests(unittest.TestCase):
    def test_longest_consecutive_rejects_spatially_scattered_matches(self) -> None:
        self.assertEqual(MODULE.longest_consecutive(np.asarray([1, 2, 5, 7, 8, 9])), 3)

    def test_matched_contour_passes_and_shifted_surface_fails(self) -> None:
        image = np.zeros((200, 150), dtype=np.float64)
        image[109:111, 50:71] = 1.0
        surface = np.zeros_like(image, dtype=bool)
        surface[110, 50:71] = True
        matched = MODULE.track_alignment(image, FakeTrack(), surface)
        shifted = np.zeros_like(surface)
        shifted[130, 50:71] = True
        mismatch = MODULE.track_alignment(image, FakeTrack(), shifted)
        self.assertTrue(matched.passes)
        self.assertFalse(mismatch.passes)


if __name__ == "__main__":
    unittest.main()
