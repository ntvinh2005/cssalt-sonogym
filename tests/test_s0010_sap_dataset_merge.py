"""Tests for fail-closed SAP dataset merging."""

from __future__ import annotations

import csv
import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


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
    def test_csv_writer_preserves_union_of_revision_diagnostics(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "merged.csv"
            MODULE.write_csv(
                path,
                [
                    {"sample_id": "old", "legacy_score": "1"},
                    {"sample_id": "new", "surface_row_span_pixels": "7"},
                ],
            )
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                list(rows[0]),
                ["sample_id", "legacy_score", "surface_row_span_pixels"],
            )
            self.assertEqual(rows[0]["legacy_score"], "1")
            self.assertEqual(rows[0]["surface_row_span_pixels"], "")
            self.assertEqual(rows[1]["legacy_score"], "")
            self.assertEqual(rows[1]["surface_row_span_pixels"], "7")

    def test_missing_derived_correspondence_csv_is_represented_explicitly(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "correspondence_controls.csv"
            self.assertIsNone(MODULE.optional_file_sha256(path))
            path.write_text("comparison\nMATCHED\n", encoding="utf-8")
            self.assertEqual(MODULE.optional_file_sha256(path), MODULE.file_sha256(path))

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
