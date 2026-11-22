"""Tests for the frozen RIGHT L5 SAP/TP clean-image signal experiment."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from two_stage_navigation.coordinates import (
    is_supported_stage2_dataset_protocol,
    uses_coupled_stage2_dataset_controller,
    uses_segmented_stage2_dataset_capture,
)


ROOT = Path(__file__).parents[1]


def load(name: str, relative: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


GENERATOR = load(
    "sap_tp_signal_protocol", "workflows/two_stage_navigation/generate_s0010_sap_tp_signal_protocol.py"
)
EVALUATOR = load(
    "sap_tp_signal_evaluator", "workflows/two_stage_navigation/evaluate_s0010_sap_tp_signal.py"
)


class SapTpSignalTests(unittest.TestCase):
    def test_protocol_uses_synchronized_segmented_controller(self) -> None:
        protocol_id = GENERATOR.PROTOCOL_ID
        self.assertTrue(is_supported_stage2_dataset_protocol(protocol_id))
        self.assertTrue(uses_coupled_stage2_dataset_controller(protocol_id))
        self.assertTrue(uses_segmented_stage2_dataset_capture(protocol_id))

    def test_waypoint_separation_uses_tracking_tolerances(self) -> None:
        self.assertFalse(GENERATOR.separated([100, 100, 2.0], [101, 100, 2.0]))
        self.assertTrue(GENERATOR.separated([100, 100, 2.0], [102, 100, 2.0]))
        self.assertTrue(
            GENERATOR.separated([100, 100, 2.0], [100, 100, 2.0 + np.deg2rad(2.6)])
        )

    def test_clean_image_match_cannot_bypass_minimum_surface_support(self) -> None:
        projected = {
            "sap": np.zeros((5, 5), dtype=bool),
            "tp": np.zeros((5, 5), dtype=bool),
        }
        projected["sap"][1, 1] = True
        projected["tp"][2, 2] = True
        base = {
            "image_supported": True,
            "track_count": 2,
            "sap_track_distance_pixels": 0.0,
            "tp_track_distance_pixels": 0.0,
            "sap_shadow_contrast": 0.2,
            "tp_shadow_contrast": 0.2,
            "target_image_margin_pixels": 1,
            "sap_core_first_visible_pixels": 1,
            "tp_core_first_visible_pixels": 1,
        }
        gate = {
            "maximum_interface_track_distance_pixels": 4.0,
            "minimum_posterior_shadow_contrast": 0.05,
            "minimum_target_image_margin_pixels": 1,
            "minimum_sap_first_visible_pixels": 3,
            "minimum_tp_first_visible_pixels": 3,
        }
        with mock.patch.object(EVALUATOR.historical_evaluator, "image_supported", return_value=base):
            result = EVALUATOR.surface_image_evidence(
                np.ones((5, 5)), projected, np.ones((5, 5), dtype=bool), gate
            )
        self.assertFalse(result["geometry_surface_pair_supported"])
        self.assertFalse(result["image_supported"])


if __name__ == "__main__":
    unittest.main()
