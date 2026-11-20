"""Tests for the frozen SAP validation capture design."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "workflows/two_stage_navigation/generate_s0010_sap_validation_protocols.py"
SPEC = importlib.util.spec_from_file_location("sap_validation_protocols", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SapValidationProtocolTests(unittest.TestCase):
    def test_banks_are_disjoint_and_adjacent_waypoints_exceed_tolerances(self) -> None:
        banks = MODULE.allocate_banks()
        flattened = [tuple(row) for values in banks.values() for row in values]
        self.assertEqual(len(flattened), 6 * MODULE.FRAMES_PER_SESSION)
        self.assertEqual(len(set(flattened)), len(flattened))
        for values in banks.values():
            self.assertEqual(len(values), MODULE.FRAMES_PER_SESSION)
            for left, right in zip(values[:-1], values[1:]):
                translation_mm = np.linalg.norm(np.asarray(left[:2]) - np.asarray(right[:2])) * 1.5
                angle_deg = abs(left[2] - right[2]) * 180.0 / np.pi
                self.assertTrue(translation_mm > 3.0 or angle_deg > 2.4)


if __name__ == "__main__":
    unittest.main()
