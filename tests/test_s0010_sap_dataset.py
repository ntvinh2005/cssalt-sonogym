"""Focused tests for fail-closed SAP dataset selection."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


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

    def test_session_directory_accepts_only_explicit_absolute_archives(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                MODULE.session_directory({"session_id": "session"}, root),
                root / "sessions/session",
            )
            archived = root / "archived/session"
            self.assertEqual(
                MODULE.session_directory(
                    {
                        "session_id": "session",
                        "source_session_path": str(archived),
                    },
                    root,
                ),
                archived,
            )
            with self.assertRaisesRegex(ValueError, "must be absolute"):
                MODULE.session_directory(
                    {
                        "session_id": "session",
                        "source_session_path": "relative/session",
                    },
                    root,
                )

    def test_archived_session_identity_and_frozen_hash_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            verification = root / "verification_report.json"
            verification.write_text('{"verdict":"PROJECTION_PARITY_PASS"}\n')
            session = {
                "session_id": "session",
                "frame_count": 1,
                "verification_sha256": MODULE.sha256(verification),
            }
            public = [{"session_id": "session", "frame_sequence": 0}]
            MODULE.validate_session_records(session, root, public, 1)
            with self.assertRaisesRegex(ValueError, "session identity mismatch"):
                MODULE.validate_session_records(
                    session,
                    root,
                    [{"session_id": "other", "frame_sequence": 0}],
                    1,
                )
            verification.write_text('{"verdict":"changed"}\n')
            with self.assertRaisesRegex(ValueError, "hash changed"):
                MODULE.validate_session_records(session, root, public, 1)

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

    def test_correspondence_gate_separates_empty_projections(self) -> None:
        intended = {("session", 0): "L4", ("session", 1): "L4"}
        absent = np.zeros((8, 8), dtype=bool)
        present = absent.copy()
        present[3, 1:6] = True
        surfaces = {
            ("session", 0, "L4"): absent,
            ("session", 1, "L4"): present,
        }
        retained, excluded = MODULE.correspondence_target_keys(
            intended, surfaces, "L4", 5, True
        )
        self.assertEqual(retained, [("session", 1)])
        self.assertEqual(excluded, 1)
        historical, excluded = MODULE.correspondence_target_keys(
            intended, surfaces, "L4", 5, False
        )
        self.assertEqual(historical, [("session", 0), ("session", 1)])
        self.assertEqual(excluded, 0)


if __name__ == "__main__":
    unittest.main()
