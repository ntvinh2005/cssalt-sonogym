from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "workflows/two_stage_navigation/run_s0010_stage2_multiview_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("multiview_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evidence = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evidence
SPEC.loader.exec_module(evidence)


def actor_row(sequence: int, delta: tuple[float, float], angle: float) -> dict[str, object]:
    return {
        "frame_sequence": sequence,
        "producer_sequence": 10 + sequence,
        "frame_sha256": f"hash-{sequence}",
        "executed_delta_relative_mm": list(delta),
        "executed_delta_angle_rad": angle,
        "action_accepted": True,
        "action_completed": True,
        "action_clamped": False,
        "contact_ok": True,
        "workspace_ok": True,
        "image_valid": True,
    }


class MultiviewEvidenceTests(unittest.TestCase):
    def test_motion_is_accumulated_within_one_session(self) -> None:
        poses = evidence.cumulative_pose_by_frame(
            [actor_row(0, (1.0, 2.0), 0.1), actor_row(1, (3.0, -1.0), 0.2)]
        )
        np.testing.assert_allclose(poses[0], [1.0, 2.0, 0.1])
        np.testing.assert_allclose(poses[1], [4.0, 1.0, 0.3])

    def test_duplicate_hash_cannot_supply_another_vote(self) -> None:
        rows = [actor_row(0, (1.0, 0.0), 0.0), actor_row(1, (2.0, 0.0), 0.0)]
        rows[1]["frame_sha256"] = rows[0]["frame_sha256"]
        with self.assertRaisesRegex(ValueError, "duplicate image hash"):
            evidence.cumulative_pose_by_frame(rows)

    def test_ambiguous_selection_is_not_counted_as_positive(self) -> None:
        session = evidence.SessionData(
            session_id="session",
            frame_sequences=np.asarray([0, 1, 2]),
            frame_hashes=["a", "b", "c"],
            statuses=[evidence.POSITIVE, evidence.AMBIGUOUS, evidence.NEGATIVE],
            images=[np.zeros((200, 150)) for _ in range(3)],
            poses=np.zeros((3, 3)),
            base_scores=np.asarray([0.7, 0.9, 0.1]),
            correspondences=np.eye(3),
            translation_mm=np.zeros((3, 3)),
            angle_deg=np.zeros((3, 3)),
        )
        record, _ = evidence.assess_session(
            session,
            np.asarray([0.7, 0.9, 0.1]),
            np.asarray([2, 2, 2]),
            [[1, 2], [0, 2], [0, 1]],
            threshold=0.5,
            margin_threshold=0.0,
            minimum_support=2,
        )
        self.assertEqual(record["selected_status"], evidence.AMBIGUOUS)
        self.assertFalse(record["accepted_positive"])
        self.assertTrue(record["wrong_accepted_positive_session"])

    def test_cross_split_hash_overlap_fails_closed(self) -> None:
        train = [{"split": "train", "frame_sha256": "same"}]
        evaluation = [{"split": "evaluation", "frame_sha256": "same"}]
        with self.assertRaisesRegex(ValueError, "cross split"):
            evidence.verify_split_hashes(train, evaluation)


if __name__ == "__main__":
    unittest.main()
