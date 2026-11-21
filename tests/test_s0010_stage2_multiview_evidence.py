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
        poses, admissible = evidence.cumulative_pose_by_frame(
            [actor_row(0, (1.0, 2.0), 0.1), actor_row(1, (3.0, -1.0), 0.2)]
        )
        np.testing.assert_allclose(poses[0], [1.0, 2.0, 0.1])
        np.testing.assert_allclose(poses[1], [4.0, 1.0, 0.3])
        self.assertEqual(admissible, {0: True, 1: True})

    def test_duplicate_hash_cannot_supply_another_vote(self) -> None:
        rows = [actor_row(0, (1.0, 0.0), 0.0), actor_row(1, (2.0, 0.0), 0.0)]
        rows[1]["frame_sha256"] = rows[0]["frame_sha256"]
        with self.assertRaisesRegex(ValueError, "duplicate image hash"):
            evidence.cumulative_pose_by_frame(rows)

    def test_inadmissible_frame_preserves_odometry_but_cannot_vote(self) -> None:
        rows = [actor_row(0, (1.0, 0.0), 0.0), actor_row(1, (2.0, 0.0), 0.0)]
        rows[0]["action_clamped"] = True
        poses, admissible = evidence.cumulative_pose_by_frame(rows)
        np.testing.assert_allclose(poses[1], [3.0, 0.0, 0.0])
        self.assertFalse(admissible[0])
        self.assertTrue(admissible[1])

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
            excluded_frame_sequences=[],
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

    def test_evaluation_cannot_pass_after_calibration_gate_failure(self) -> None:
        evaluation = {
            "top1_positive_sessions": 2,
            "accepted_positive_sessions": 2,
            "wrong_accepted_positive_sessions": 0,
            "accepted_no_positive_sessions": 0,
        }
        calibration = {"calibration_gate_pass": False}
        gate = {
            "required_top1_positive_sessions": 2,
            "required_accepted_positive_sessions": 2,
            "maximum_wrong_accepted_sessions": 0,
            "maximum_accepted_ambiguous_or_negative_sessions": 0,
        }
        self.assertFalse(evidence.passes_development_gate(evaluation, calibration, gate))


if __name__ == "__main__":
    unittest.main()
