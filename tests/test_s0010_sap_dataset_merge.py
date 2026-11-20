"""Tests for fail-closed SAP dataset merging."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PATH = (
    Path(__file__).resolve().parents[1]
    / "workflows/two_stage_navigation/merge_s0010_sap_datasets.py"
)
SPEC = importlib.util.spec_from_file_location("sap_dataset_merge", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SapDatasetMergeTests(unittest.TestCase):
    def test_external_calibration_applies_only_to_declared_level(self) -> None:
        report = {
            "levels": {
                "L5": {"calibration_gate_pass": True},
            }
        }
        self.assertEqual(
            MODULE.label_validation_result("L5", False, report), (True, True)
        )
        self.assertEqual(
            MODULE.label_validation_result("L4", False, report), (False, False)
        )

    def test_failed_external_calibration_cannot_inherit_raw_pass(self) -> None:
        report = {
            "levels": {
                "L4": {"calibration_gate_pass": False},
            }
        }
        self.assertEqual(
            MODULE.label_validation_result("L4", True, report), (False, True)
        )


if __name__ == "__main__":
    unittest.main()
