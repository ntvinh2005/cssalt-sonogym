"""Tests for deterministic SAP capture resumption after a movement-gate failure."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "workflows/two_stage_navigation/generate_s0010_sap_validation_resume_protocol.py"
SPEC = importlib.util.spec_from_file_location("sap_resume_protocol", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SapResumeProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_path = (
            ROOT
            / "experiments/unity_two_stage_transfer/s0010_sap_validation_followup_protocol_v1.yaml"
        )
        self.source = yaml.safe_load(self.source_path.read_text(encoding="utf-8"))

    def test_previous_translation_only_transition_lacks_tracking_margin(self) -> None:
        bank = self.source["sampling"]["absolute_waypoint_banks_index"][
            "s0010sapval_calibration_01"
        ]
        margin = {
            "label_resolution_mm": 1.5,
            "tracking_position_tolerance_mm": 1.5,
            "tracking_orientation_tolerance_deg": 1.2,
            "minimum_realized_translation_mm": 2.0,
            "minimum_realized_orientation_deg": 1.0,
            "extra_translation_margin_mm": 1.0,
            "extra_orientation_margin_deg": 0.5,
        }
        self.assertEqual(bank[2], [148.0, 140.0, 2.74])
        self.assertEqual(bank[3], [150.5, 140.0, 2.74])
        self.assertFalse(MODULE.transition_has_tracking_margin(bank[2], bank[3], **margin))

    def test_all_resume_transitions_retain_tracking_margin(self) -> None:
        document = MODULE.build_resume_protocol(self.source, self.source_path, "test-commit")
        margin = document["sampling"]["waypoint_tracking_margin"]
        banks = document["sampling"]["absolute_waypoint_banks_index"]
        self.assertEqual(len(document["sessions"]), 3)
        self.assertEqual(sum(row["expected_frame_count"] for row in document["sessions"]), 140)
        for rows in banks.values():
            self.assertEqual(len(rows), len(set(tuple(row) for row in rows)))
            self.assertTrue(
                all(
                    MODULE.transition_has_tracking_margin(left, right, **margin)
                    for left, right in zip(rows, rows[1:])
                )
            )

    def test_reordering_preserves_each_level_block(self) -> None:
        document = MODULE.build_resume_protocol(self.source, self.source_path, "test-commit")
        source_banks = self.source["sampling"]["absolute_waypoint_banks_index"]
        source_levels = self.source["sampling"]["waypoint_level_order_by_session"]
        for source_id, session_id, _ in MODULE.REMAINING_SESSIONS:
            revised = document["sampling"]["absolute_waypoint_banks_index"][session_id]
            for level_range in source_levels[source_id]:
                start = int(level_range["start_index"])
                stop = int(level_range["stop_index_exclusive"])
                self.assertEqual(
                    {tuple(row) for row in revised[start:stop]},
                    {tuple(row) for row in source_banks[source_id][start:stop]},
                )

    def test_failed_and_unopened_sessions_are_renamed(self) -> None:
        document = MODULE.build_resume_protocol(self.source, self.source_path, "test-commit")
        session_ids = [row["session_id"] for row in document["sessions"]]
        self.assertEqual(
            session_ids,
            [
                "s0010sapval_calibration_02",
                "s0010sapval_evaluation_02",
                "s0010sapval_evaluation_03",
            ],
        )
        self.assertNotIn(
            "s0010sapval_calibration_01",
            document["sampling"]["absolute_waypoint_banks_index"],
        )
        self.assertEqual(document["source"]["implementation_commit"], "test-commit")

    def test_capture_job_has_explicit_resume_mode(self) -> None:
        job = (ROOT / "hpg/jobs/s0010_sap_validation_capture.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("    resume)", job)
        self.assertIn(
            'PROTOCOL_BASENAME="s0010_sap_validation_resume_protocol_v1"', job
        )
        self.assertIn('EXPERIMENT_BASENAME="s0010_sap_validation_resume_v1"', job)
        for session_id in (
            "s0010sapval_calibration_02",
            "s0010sapval_evaluation_02",
            "s0010sapval_evaluation_03",
        ):
            self.assertIn(session_id, job)


if __name__ == "__main__":
    unittest.main()
