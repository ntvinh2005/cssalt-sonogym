#!/usr/bin/env python3
"""CPU tests for causal Stage 2 temporal and multi-view decisions."""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.stage2_temporal_decision import (  # noqa: E402
    DecisionState,
    ReplayObservation,
    TemporalDecisionConfig,
    fresh_confirmation_passes_distinct,
    reconstruct_trajectory_observations,
    replay_decision,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def config() -> TemporalDecisionConfig:
    return TemporalDecisionConfig(
        validity_probability_threshold=0.5666666666666667,
        maximum_neighbor_translation_mm=4.5,
        maximum_neighbor_angle_deg=6.0,
        maximum_quality_drop=0.08,
        required_distinct_hash_observations=3,
        minimum_distinct_view_translation_mm=1.0,
        minimum_distinct_view_angle_deg=1.0,
    )


def observation(
    index: int,
    *,
    hash_name: str | None = None,
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
    probability: float = 0.9,
    quality: float = 0.8,
    transition_valid: bool = True,
    image_valid: bool = True,
    session_id: str = "session",
    trajectory_id: str = "trajectory",
) -> ReplayObservation:
    return ReplayObservation(
        session_id=session_id,
        trajectory_id=trajectory_id,
        frame_sequence=index,
        trajectory_frame_sequence=index,
        producer_sequence=10 + index,
        frame_sha256=digest(hash_name or f"frame-{index}"),
        relative_pose=pose,
        validity_probability=probability,
        quality=quality,
        transition_valid=transition_valid,
        image_valid=image_valid,
    )


def actor_record(
    index: int,
    *,
    trajectory_id: str = "trajectory",
    delta: tuple[float, float] = (1.5, 0.0),
    accepted: bool = True,
) -> dict[str, object]:
    return {
        "schema": "sonogym-stage2-actor-frame-v1",
        "session_id": "session",
        "trajectory_id": trajectory_id,
        "trajectory_frame_sequence": index,
        "split": "confirmation",
        "frame_sequence": index,
        "producer_sequence": 20 + index,
        "frame_sha256": digest(f"record-{index}"),
        "image_archive": "public/ultrasound.npz",
        "image_archive_index": index,
        "executed_delta_relative_mm": list(delta),
        "executed_delta_angle_rad": 0.0,
        "action_steps_since_previous_frame": 1,
        "action_accepted": accepted,
        "action_completed": accepted,
        "action_clamped": False,
        "action_position_clamped": False,
        "action_angle_clamped": False,
        "contact_ok": True,
        "workspace_ok": True,
        "image_valid": True,
    }


class TemporalEvidenceTests(unittest.TestCase):
    def test_three_identical_hashes_are_not_three_confirmations(self) -> None:
        rows = [observation(index, hash_name="same") for index in range(3)]
        decision = replay_decision(rows, policy="candidate_c_temporal", config=config())
        self.assertEqual(decision.state, DecisionState.ABSTAIN)

    def test_distinct_hashes_at_same_pose_are_not_distinct_viewpoints(self) -> None:
        rows = [observation(index) for index in range(3)]
        temporal = replay_decision(rows, policy="candidate_c_temporal", config=config())
        multi_view = replay_decision(
            rows, policy="candidate_d_multi_view", config=config()
        )
        self.assertEqual(temporal.state, DecisionState.ACCEPT)
        self.assertEqual(multi_view.state, DecisionState.ABSTAIN)

    def test_distinct_physical_poses_with_invalid_movement_do_not_support(self) -> None:
        rows = [
            observation(0, pose=(0.0, 0.0, 0.0)),
            observation(1, pose=(1.5, 0.0, 0.0), transition_valid=False),
            observation(2, pose=(3.0, 0.0, 0.0)),
        ]
        decision = replay_decision(
            rows, policy="candidate_d_multi_view", config=config()
        )
        self.assertEqual(decision.state, DecisionState.ABSTAIN)

    def test_three_valid_distinct_viewpoints_support_multi_view(self) -> None:
        rows = [
            observation(0, pose=(0.0, 0.0, 0.0)),
            observation(1, pose=(1.5, 0.0, 0.0)),
            observation(2, pose=(3.0, 0.0, 0.0)),
        ]
        decision = replay_decision(
            rows, policy="candidate_d_multi_view", config=config()
        )
        self.assertEqual(decision.state, DecisionState.ACCEPT)
        self.assertEqual(
            decision.evidence.distinct_viewpoint_count, 3
        )  # type: ignore[union-attr]

    def test_reset_or_another_trajectory_breaks_history(self) -> None:
        rows = [
            observation(0),
            observation(1),
            observation(2, trajectory_id="after-reset"),
        ]
        with self.assertRaisesRegex(ValueError, "trajectory boundary"):
            replay_decision(rows, policy="candidate_c_temporal", config=config())

    def test_future_frame_is_unavailable_until_replay_advances(self) -> None:
        rows = [observation(0, probability=0.1), observation(1, probability=0.9)]
        early = replay_decision(
            rows,
            policy="baseline_a_single_frame",
            config=config(),
            available_until_frame_sequence=0,
        )
        complete = replay_decision(
            rows, policy="baseline_a_single_frame", config=config()
        )
        self.assertEqual(early.state, DecisionState.ABSTAIN)
        self.assertEqual(complete.selected, rows[1])

    def test_quality_cannot_override_classifier_rejection(self) -> None:
        row = observation(0, probability=0.1, quality=1.0)
        baseline = replay_decision(
            [row], policy="baseline_b_quality_ranked", config=config()
        )
        diagnostic = replay_decision(
            [row], policy="diagnostic_quality_only", config=config()
        )
        self.assertEqual(baseline.state, DecisionState.ABSTAIN)
        self.assertEqual(diagnostic.state, DecisionState.ACCEPT)

    def test_missing_evidence_abstains_instead_of_forcing_selection(self) -> None:
        decision = replay_decision(
            [observation(0)], policy="candidate_d_multi_view", config=config()
        )
        self.assertEqual(decision.state, DecisionState.ABSTAIN)
        self.assertIsNone(decision.selected)

    def test_privileged_field_in_actor_record_fails_closed(self) -> None:
        record = actor_record(0)
        record["goal_cmd_pose"] = [1.0, 2.0, 3.0]
        with self.assertRaisesRegex(ValueError, "privileged actor field"):
            reconstruct_trajectory_observations([record], [0.9], [0.8])

    def test_candidate_sorting_and_replay_are_deterministic(self) -> None:
        rows = [
            observation(0, pose=(0.0, 0.0, 0.0), quality=0.85),
            observation(1, pose=(1.5, 0.0, 0.0), quality=0.9),
            observation(2, pose=(3.0, 0.0, 0.0), quality=0.9),
        ]
        first = replay_decision(rows, policy="candidate_d_multi_view", config=config())
        second = replay_decision(rows, policy="candidate_d_multi_view", config=config())
        self.assertEqual(first, second)
        self.assertEqual(first.selected, rows[1])

    def test_repeated_false_positive_cannot_gain_duplicate_weight(self) -> None:
        rows = [
            observation(index, hash_name="one-false-positive", probability=0.99)
            for index in range(6)
        ]
        decision = replay_decision(rows, policy="candidate_c_temporal", config=config())
        self.assertEqual(decision.state, DecisionState.ABSTAIN)


class RecordIntegrityTests(unittest.TestCase):
    def test_relative_pose_is_reconstructed_from_executed_motion(self) -> None:
        rows = [actor_record(0), actor_record(1, delta=(-0.5, 2.0))]
        observations = reconstruct_trajectory_observations(rows, [0.9, 0.8], [0.7, 0.6])
        self.assertEqual(observations[0].relative_pose[:2], (1.5, 0.0))
        self.assertEqual(observations[1].relative_pose[:2], (1.0, 2.0))

    def test_mixed_trajectory_records_are_rejected(self) -> None:
        rows = [actor_record(0), actor_record(1, trajectory_id="other")]
        with self.assertRaisesRegex(ValueError, "trajectory boundary"):
            reconstruct_trajectory_observations(rows, [0.9, 0.9], [0.8, 0.8])

    def test_missing_pose_invalid_hash_and_rejected_action_fail_safely(self) -> None:
        missing = actor_record(0)
        del missing["executed_delta_relative_mm"]
        with self.assertRaisesRegex(ValueError, "executed_delta_relative_mm"):
            reconstruct_trajectory_observations([missing], [0.9], [0.8])

        invalid_hash = actor_record(0)
        invalid_hash["frame_sha256"] = "not-a-hash"
        with self.assertRaisesRegex(ValueError, "frame_sha256"):
            reconstruct_trajectory_observations([invalid_hash], [0.9], [0.8])

        rejected = reconstruct_trajectory_observations(
            [actor_record(0, accepted=False)], [0.9], [0.8]
        )
        decision = replay_decision(
            rejected, policy="baseline_b_quality_ranked", config=config()
        )
        self.assertEqual(decision.state, DecisionState.ABSTAIN)

    def test_empty_sequence_abstains(self) -> None:
        decision = replay_decision([], policy="candidate_d_multi_view", config=config())
        self.assertEqual(decision.state, DecisionState.ABSTAIN)

    def test_fresh_confirmation_requires_distinct_new_hashes(self) -> None:
        remembered = observation(0, hash_name="remembered")
        repeated = [
            observation(1, hash_name="confirmation"),
            observation(2, hash_name="confirmation"),
        ]
        self.assertFalse(
            fresh_confirmation_passes_distinct(
                remembered, repeated, config=config(), required_count=2
            )
        )
        distinct = [observation(1), observation(2)]
        self.assertTrue(
            fresh_confirmation_passes_distinct(
                remembered, distinct, config=config(), required_count=2
            )
        )

    def test_v18_protocol_and_result_seals_remain_unchanged(self) -> None:
        root = REPOSITORY_ROOT / "experiments" / "unity_two_stage_transfer"
        for name in (
            "stage2_confirmation_protocol_v18.yaml",
            "stage2_confirmation_v18_result.json",
        ):
            path = root / name
            expected = (
                path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
            )
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
