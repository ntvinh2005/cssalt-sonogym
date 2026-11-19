#!/usr/bin/env python3
"""CPU tests for the frozen L4 coverage-repair comparison."""

from __future__ import annotations

import importlib.util
import hashlib
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "workflows/two_stage_navigation/train_tp_surface_core_l4_coverage_repair.py"
PROTOCOL_V13 = ROOT / "experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v13.yaml"
PROTOCOL_V14 = ROOT / "experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v14.yaml"
SPEC = importlib.util.spec_from_file_location("l4_coverage_repair", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REPAIR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPAIR
SPEC.loader.exec_module(REPAIR)


class CoverageRepairTests(unittest.TestCase):
    def test_protocol_is_sealed_and_split_before_capture(self) -> None:
        expected = PROTOCOL_V13.with_suffix(".sha256").read_text().split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V13.read_bytes()).hexdigest(), expected)
        protocol = yaml.safe_load(PROTOCOL_V13.read_text())
        self.assertEqual(protocol["protocol_id"], "sonogym-s0010-tp-surface-core-dataset-v13")
        self.assertEqual(
            [row["split"] for row in protocol["sessions"]],
            ["train", "train", "calibration", "development_evaluation"],
        )
        self.assertTrue(
            all(
                len(rows) == 48
                for rows in protocol["sampling"]["absolute_waypoint_banks_index"].values()
            )
        )

    def test_v14_changes_only_convergence_budget_and_session_identity(self) -> None:
        expected = PROTOCOL_V14.with_suffix(".sha256").read_text().split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V14.read_bytes()).hexdigest(), expected)
        v13 = yaml.safe_load(PROTOCOL_V13.read_text())
        v14 = yaml.safe_load(PROTOCOL_V14.read_text())
        self.assertEqual(v13["sampling"]["maximum_initial_positioning_steps"], 350)
        self.assertEqual(v14["sampling"]["maximum_initial_positioning_steps"], 450)
        self.assertEqual(
            v13["sampling"]["absolute_waypoint_banks_index"],
            v14["sampling"]["absolute_waypoint_banks_index"],
        )
        self.assertEqual(v13["model_comparison_gate"], v14["model_comparison_gate"])

    def test_calibration_selection_is_deterministic(self) -> None:
        baseline = {"tp": 8, "fp": 4, "ambiguous_activation_images": 2}
        candidates = {
            "b": {"tp": 8, "fp": 2, "ambiguous_activation_images": 1},
            "a": {"tp": 9, "fp": 3, "ambiguous_activation_images": 2},
        }
        self.assertEqual(REPAIR.choose_candidate(baseline, candidates), "a")

    def test_recall_losing_candidate_is_ineligible(self) -> None:
        baseline = {"tp": 8, "fp": 4, "ambiguous_activation_images": 2}
        candidate = {"tp": 7, "fp": 0, "ambiguous_activation_images": 0}
        self.assertFalse(REPAIR.calibration_eligible(baseline, candidate))

    def test_coverage_gate_is_split_specific(self) -> None:
        protocol = {
            "dataset_gate": {
                "minimum_l4_proxy_positive_images": {"train": 16, "calibration": 8},
                "minimum_clear_confuser_images": {"train": 24, "calibration": 12},
            }
        }
        counts = {
            "train": {"positive": 16, "negative": 24},
            "calibration": {"positive": 8, "negative": 11},
        }
        self.assertFalse(REPAIR.coverage_passes(counts, protocol))
        counts["calibration"]["negative"] = 12
        self.assertTrue(REPAIR.coverage_passes(counts, protocol))

    def test_development_features_are_loaded_after_selection(self) -> None:
        source = SCRIPT.read_text()
        self.assertIn('"development_feature_rows_loaded_before_calibration": 0', source)
        self.assertLess(
            source.index("selected_arm = choose_candidate"),
            source.index("evaluation_features = ABLATION.load_features"),
        )
        self.assertIn('row["split"] != "train"', source)


if __name__ == "__main__":
    unittest.main()
