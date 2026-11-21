from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / "workflows/two_stage_navigation"
if str(WORKFLOW) not in sys.path:
    sys.path.insert(0, str(WORKFLOW))
SCRIPT = WORKFLOW / "run_s0010_stage2_multiview_ranker.py"
SPEC = importlib.util.spec_from_file_location("multiview_ranker", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ranker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ranker
SPEC.loader.exec_module(ranker)


class MultiviewRankerTests(unittest.TestCase):
    def test_training_folds_are_session_disjoint(self) -> None:
        rows = [
            {
                "split": "train",
                "conditioned_level": "L5",
                "session_id": f"train-{index}",
            }
            for index in range(10)
        ]
        rows.extend(
            [
                {
                    "split": "calibration",
                    "conditioned_level": "L5",
                    "session_id": "calibration",
                },
                {
                    "split": "train",
                    "conditioned_level": "L4",
                    "session_id": "other-level",
                },
            ]
        )
        folds = ranker.training_session_folds(rows, level="L5", fold_count=5)
        held = set()
        for fitted_sessions, held_sessions in folds:
            self.assertFalse(fitted_sessions & held_sessions)
            self.assertNotIn("calibration", fitted_sessions | held_sessions)
            self.assertNotIn("other-level", fitted_sessions | held_sessions)
            held.update(held_sessions)
        self.assertEqual(held, {f"train-{index}" for index in range(10)})

    def test_too_many_folds_fail_closed(self) -> None:
        rows = [
            {"split": "train", "conditioned_level": "L5", "session_id": "one"}
        ]
        with self.assertRaisesRegex(ValueError, "nonempty held-out"):
            ranker.training_session_folds(rows, level="L5", fold_count=5)


if __name__ == "__main__":
    unittest.main()
