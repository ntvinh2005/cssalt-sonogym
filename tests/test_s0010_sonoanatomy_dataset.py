from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from prepare_s0010_sonoanatomy_dataset import (
    capture_verification_passed,
    selected_session_directories,
)


class SonoanatomyDatasetTests(unittest.TestCase):
    def test_session_filter_preserves_declared_order_and_level_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sessions = Path(temporary) / "sessions"
            sessions.mkdir()
            for name in ("l4_00", "l4_01", "l5_00"):
                (sessions / name).mkdir()
            selected = selected_session_directories(
                Path(temporary), ["l4_01", "l4_00"]
            )
            self.assertEqual([path.name for path in selected], ["l4_01", "l4_00"])

    def test_session_filter_rejects_missing_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sessions = Path(temporary) / "sessions"
            sessions.mkdir()
            (sessions / "l4_00").mkdir()
            with self.assertRaisesRegex(ValueError, "missing sessions"):
                selected_session_directories(Path(temporary), ["l5_00"])

    def test_direct_stage2_session_layout_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("mbbpoc_00", "mbbpoc_01"):
                (root / name).mkdir()
            selected = selected_session_directories(
                root, ["mbbpoc_01"], layout="direct_children"
            )
            self.assertEqual([path.name for path in selected], ["mbbpoc_01"])

    def test_completed_stage2_summary_requires_sealed_actor_and_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary)
            (session / "results").mkdir()
            (session / "summary.json").write_text(
                json.dumps({"status": "COMPLETED", "frame_count": 12})
            )
            self.assertFalse(
                capture_verification_passed(session, "completed_stage2_summary")
            )
            (session / "results/actor_result.json.sha256").touch()
            (session / "results/evaluation_report.json.sha256").touch()
            self.assertTrue(
                capture_verification_passed(session, "completed_stage2_summary")
            )


if __name__ == "__main__":
    unittest.main()
