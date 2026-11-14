#!/usr/bin/env python3
"""CPU tests for deterministic Stage 2 review selection and rendering."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "workflows" / "two_stage_navigation" / "export_stage2_review.py"
SPEC = importlib.util.spec_from_file_location("export_stage2_review", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def record(session: str, frame: int, split: str, valid: int, quality: float, l4: float) -> dict:
    return {
        "session_id": session,
        "frame_sequence": frame,
        "split": split,
        "validity_label": valid,
        "quality_label": quality,
        "position_error_mm": float(frame + 1),
        "orientation_error_deg": float(frame + 2),
        "l4_visible_fraction": l4,
        "technical_image_valid": True,
        "raw_label_pixel_counts": {"27": frame * 10, "28": int(l4 * 30000), "29": 0},
    }


class ReviewExportTests(unittest.TestCase):
    def test_selection_is_deterministic_unique_and_split_aware(self) -> None:
        rows = [
            record("train_a", 0, "train", 1, 0.9, 0.2),
            record("train_a", 1, "train", 0, 0.8, 0.3),
            record("cal_a", 0, "calibration", 1, 0.7, 0.1),
            record("cal_a", 1, "calibration", 0, 0.6, 0.01),
            record("dev_a", 0, "development_evaluation", 1, 0.5, 0.08),
            record("dev_a", 1, "development_evaluation", 0, 0.4, 0.02),
        ]
        first = MODULE.select_review_records(rows, maximum_items=6)
        second = MODULE.select_review_records(list(reversed(rows)), maximum_items=6)
        self.assertEqual([MODULE.record_key(row) for row in first], [MODULE.record_key(row) for row in second])
        self.assertEqual(len({MODULE.record_key(row) for row in first}), len(first))
        self.assertEqual({row["split"] for row in first}, {"train", "calibration", "development_evaluation"})

    def test_clean_and_overlay_render_expected_shapes(self) -> None:
        image = np.linspace(0.0, 10.0, 200 * 150, dtype=np.float32).reshape(200, 150)
        labels = np.zeros((200, 150), dtype=np.uint8)
        labels[50:80, 60:90] = 28
        clean = MODULE.clean_rgb(image)
        overlay = MODULE.evaluator_overlay(image, labels)
        self.assertEqual(clean.shape, (200, 150, 3))
        self.assertEqual(overlay.shape, clean.shape)
        self.assertFalse(np.array_equal(clean[50:80, 60:90], overlay[50:80, 60:90]))
        np.testing.assert_array_equal(clean[:20, :20], overlay[:20, :20])


if __name__ == "__main__":
    unittest.main()
