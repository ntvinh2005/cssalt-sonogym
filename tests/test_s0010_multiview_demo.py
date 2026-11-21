from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / "workflows/two_stage_navigation"
if str(WORKFLOW) not in sys.path:
    sys.path.insert(0, str(WORKFLOW))
SCRIPT = WORKFLOW / "render_s0010_multiview_ranker_demo.py"
SPEC = importlib.util.spec_from_file_location("multiview_demo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)


class MultiviewDemoTests(unittest.TestCase):
    def test_demo_selects_distinct_outcomes(self) -> None:
        records = [
            {
                "session_id": "correct",
                "accepted_positive": True,
                "wrong_accepted_positive_session": False,
                "positive_present": True,
                "accepted": True,
            },
            {
                "session_id": "wrong",
                "accepted_positive": False,
                "wrong_accepted_positive_session": True,
                "positive_present": True,
                "accepted": True,
            },
            {
                "session_id": "abstained",
                "accepted_positive": False,
                "wrong_accepted_positive_session": False,
                "positive_present": True,
                "accepted": False,
            },
        ]
        self.assertEqual(
            demo.select_demo_sessions(records), ["correct", "wrong", "abstained"]
        )


if __name__ == "__main__":
    unittest.main()
