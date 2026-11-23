from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from prepare_s0010_sonoanatomy_dataset import selected_session_directories


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


if __name__ == "__main__":
    unittest.main()
