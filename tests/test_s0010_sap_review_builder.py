"""Tests for final SAP review coverage and split reporting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflows/two_stage_navigation"))
import build_s0010_sap_validation_training_review as review  # noqa: E402


class SapReviewBuilderTests(unittest.TestCase):
    def test_no_proposal_frame_is_counted_as_an_abstention(self) -> None:
        proposals = [
            {
                "conditioned_level": "L5",
                "split": "evaluation",
                "frame_sha256": "with-proposal",
            }
        ]
        samples = [
            {
                "conditioned_level": "L5",
                "split": "evaluation",
                "frame_sha256": "with-proposal",
            }
        ]
        summary = review.all_frame_coverage_summary(
            level="L5",
            proposal_rows=proposals,
            sample_rows=samples,
            session_frame_counts={"session": 2},
            evaluation_level={"all_frame_coverage": {"ambiguous_activations": 0}},
        )
        self.assertEqual(summary["acquired_frames"], 2)
        self.assertEqual(summary["no_actor_proposal_frames"], 1)
        self.assertEqual(summary["abstained_images"], 1)

    def test_split_integrity_reports_exact_hash_and_session_overlap(self) -> None:
        development = [
            {
                "split": "train",
                "frame_sha256": "shared-hash",
                "session_id": "train-session",
            },
            {
                "split": "calibration",
                "frame_sha256": "cal-hash",
                "session_id": "cal-session",
            },
        ]
        evaluation = [
            {
                "split": "evaluation",
                "frame_sha256": "shared-hash",
                "session_id": "eval-session",
            }
        ]
        summary = review.split_integrity_summary(development, evaluation)
        self.assertEqual(
            summary["exact_hash_overlap"]["train__evaluation"], ["shared-hash"]
        )
        self.assertEqual(summary["session_overlap"]["train__evaluation"], [])


if __name__ == "__main__":
    unittest.main()
