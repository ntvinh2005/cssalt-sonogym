"""Tests for targeted SAP repair and sealed evaluation capture design."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "workflows/two_stage_navigation/generate_s0010_sap_validation_followup_protocol.py"
SPEC = importlib.util.spec_from_file_location("sap_followup_protocol", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SapFollowupProtocolTests(unittest.TestCase):
    def test_targeted_banks_are_unique_and_executable(self) -> None:
        bases = {
            "L4": [np.asarray([157.0, 140.5, 2.89])],
            "L5": [np.asarray([157.0, 126.5, 2.89])],
        }
        pools = {level: MODULE.targeted_pool(level, bases[level]) for level in MODULE.LEVELS}
        banks = MODULE.allocate_banks(pools)
        all_poses = [tuple(row) for values in banks.values() for row in values]
        self.assertEqual(len(all_poses), 320)
        self.assertEqual(len(set(all_poses)), len(all_poses))
        for values in banks.values():
            for left, right in zip(values[:-1], values[1:]):
                translation_mm = np.linalg.norm(np.asarray(left[:2]) - np.asarray(right[:2])) * 1.5
                angle_deg = abs(left[2] - right[2]) * 180.0 / np.pi
                self.assertTrue(translation_mm > 3.0 or angle_deg > 2.4)

    def test_two_evaluation_sessions_are_reserved(self) -> None:
        evaluation = [row for row in MODULE.SESSION_DESIGN if row[1] == "evaluation"]
        self.assertEqual(len(evaluation), 2)
        self.assertTrue(all(row[2] == 20 for row in evaluation))


if __name__ == "__main__":
    unittest.main()
