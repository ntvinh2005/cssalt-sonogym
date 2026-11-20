"""Focused checks for the sealed use of synchronized existing captures."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / "workflows/two_stage_navigation"
PATH = WORKFLOW_DIR / "generate_s0010_sap_existing_capture_protocol.py"
SPEC = importlib.util.spec_from_file_location("sap_existing_capture_protocol", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ExistingCaptureProtocolTests(unittest.TestCase):
    def test_split_and_level_ranges_are_explicit(self) -> None:
        session_ids = [Path(row[0]).name for row in MODULE.SESSION_SPECS]
        self.assertEqual(len(session_ids), len(set(session_ids)))
        self.assertGreaterEqual(
            len({Path(row[0]).name for row in MODULE.SESSION_SPECS if row[1] == "train"}),
            3,
        )
        self.assertGreaterEqual(
            len({Path(row[0]).name for row in MODULE.SESSION_SPECS if row[1] == "calibration"}),
            1,
        )
        self.assertNotIn("evaluation", {row[1] for row in MODULE.SESSION_SPECS})
        for _, _, ranges in MODULE.SESSION_SPECS:
            self.assertEqual(ranges[0][1], 0)
            self.assertTrue(all(left[2] == right[1] for left, right in zip(ranges, ranges[1:])))
            self.assertTrue({row[0] for row in ranges} <= {"L4", "L5"})


if __name__ == "__main__":
    unittest.main()
