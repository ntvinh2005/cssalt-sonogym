"""Focused tests for fail-closed SAP dataset selection."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / "workflows/two_stage_navigation"
sys.path.insert(0, str(WORKFLOW_DIR))
PATH = WORKFLOW_DIR / "prepare_s0010_sap_dataset.py"
SPEC = importlib.util.spec_from_file_location("sap_dataset", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SapDatasetTests(unittest.TestCase):
    def test_intended_level_uses_frozen_ranges(self) -> None:
        protocol = {
            "sampling": {
                "waypoint_level_order": [
                    {"level": "L4", "start_index": 0, "stop_index_exclusive": 20},
                    {"level": "L5", "start_index": 20, "stop_index_exclusive": 40},
                ]
            }
        }
        self.assertEqual(MODULE.intended_level(protocol, 19), "L4")
        self.assertEqual(MODULE.intended_level(protocol, 20), "L5")
        with self.assertRaises(ValueError):
            MODULE.intended_level(protocol, 40)

    def test_ambiguous_row_cannot_be_selected_as_binary_target(self) -> None:
        rows = [
            {
                "provisional_label": MODULE.AMBIGUOUS,
                "surface_pixels_in_patch": 20,
                "longest_aligned_run_columns": 20,
                "aligned_fraction": 1.0,
                "feature_gradient_p90": 1.0,
                "proposal_index": 0,
            }
        ]
        self.assertIsNone(MODULE.choose_candidate(rows, MODULE.POSITIVE))
        self.assertIsNone(MODULE.choose_candidate(rows, MODULE.NEGATIVE))

    def test_positive_selection_prefers_local_surface_support(self) -> None:
        def row(index: int, support: int) -> dict[str, object]:
            return {
                "provisional_label": MODULE.POSITIVE,
                "surface_pixels_in_patch": support,
                "longest_aligned_run_columns": 8,
                "aligned_fraction": 0.8,
                "feature_gradient_p90": 0.5,
                "proposal_index": index,
            }

        selected = MODULE.choose_candidate([row(0, 5), row(1, 12)], MODULE.POSITIVE)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["proposal_index"], 1)


if __name__ == "__main__":
    unittest.main()
