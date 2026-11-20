#!/usr/bin/env python3
"""Focused tests for the s0010 SAP V3 targeted correction."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "workflows/two_stage_navigation/correct_s0010_sap_semantic_review_v3.py"
SPEC = importlib.util.spec_from_file_location("sap_v3_correction", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SapSemanticReviewV3CorrectionTests(unittest.TestCase):
    def test_targeted_sheet_accepts_distance_and_class_conflicts(self) -> None:
        self.assertEqual(MODULE.PACKAGE_NAME, "s0010_right_l4l5_sap_semantic_review_v3_correction")

    def test_explicit_vertebral_ids_exclude_nonbone_tissue(self) -> None:
        prediction = np.zeros((3, 3, 3), dtype=np.uint8)
        segmentation = np.zeros_like(prediction)
        prediction[0, 0, 0] = 6
        segmentation[0, 0, 0] = 28
        prediction[1, 1, 1] = 6
        segmentation[1, 1, 1] = 120
        conversion = {label: 13 for label in MODULE.BUILDER.VERTEBRAL_SOURCE_LABELS}
        conversion[120] = 12
        report = MODULE.BUILDER.anatomical_consistency_report(
            prediction, segmentation, conversion
        )
        self.assertEqual(report["vertebral_labels_only"]["inside_count"], 1)
        self.assertEqual(report["vertebral_labels_only"]["outside_count"], 1)


if __name__ == "__main__":
    unittest.main()
