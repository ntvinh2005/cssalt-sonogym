#!/usr/bin/env python3
"""Focused CPU tests for the Stage 2 movement-contract audit."""

from __future__ import annotations

import hashlib
import math
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.stage2_contract_diagnostic import (  # noqa: E402
    MovementContract,
    candidate_support_profile,
    clamp_consistent_transition_valid,
    observations_with_contract,
    original_transition_valid,
)
from two_stage_navigation.stage2_temporal_decision import (  # noqa: E402
    DecisionState,
    ReplayObservation,
    TemporalDecisionConfig,
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


def record(index: int, **changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema": "sonogym-stage2-actor-frame-v1",
        "session_id": "session",
        "trajectory_id": "trajectory",
        "trajectory_frame_sequence": index,
        "split": "confirmation",
        "frame_sequence": index,
        "producer_sequence": index + 1,
        "frame_sha256": digest(f"frame-{index}"),
        "image_archive": "public/ultrasound.npz",
        "image_archive_index": index,
        "executed_delta_relative_mm": [1.5, 0.0],
        "executed_delta_angle_rad": 0.0,
        "action_steps_since_previous_frame": 1,
        "action_accepted": True,
        "action_completed": True,
        "action_clamped": True,
        "action_position_clamped": False,
        "action_angle_clamped": True,
        "contact_ok": True,
        "workspace_ok": True,
        "image_valid": True,
    }
    row.update(changes)
    return row


def action(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "finite": True,
        "surface_proxy_ok": True,
        "command_mapping_valid": True,
        "terminated": False,
        "truncated": False,
        "command_position_clamped": False,
        "command_angle_clamped": True,
    }
    row.update(changes)
    return row


def observation(
    index: int,
    *,
    pose: tuple[float, float, float] | None = None,
    hash_name: str | None = None,
    producer_advanced: bool = True,
) -> ReplayObservation:
    return ReplayObservation(
        session_id="session",
        trajectory_id="trajectory",
        frame_sequence=index,
        trajectory_frame_sequence=index,
        producer_sequence=index + 1,
        frame_sha256=digest(hash_name or f"observation-{index}"),
        relative_pose=pose or (1.5 * index, 0.0, 0.0),
        validity_probability=0.9,
        quality=0.8,
        transition_valid=False,
        image_valid=True,
        producer_advanced=producer_advanced,
    )


class MovementContractTests(unittest.TestCase):
    def test_declared_safe_angle_only_clamp_is_diagnostic_only(self) -> None:
        actor = record(0)
        self.assertFalse(original_transition_valid(actor))
        self.assertTrue(
            clamp_consistent_transition_valid(
                actor, [action()], angle_ceiling_exception_declared=True
            )
        )
        self.assertFalse(
            clamp_consistent_transition_valid(
                actor, [action()], angle_ceiling_exception_declared=False
            )
        )

    def test_position_clamp_remains_rejected(self) -> None:
        actor = record(0, action_position_clamped=True)
        self.assertFalse(
            clamp_consistent_transition_valid(
                actor,
                [action(command_position_clamped=True)],
                angle_ceiling_exception_declared=True,
            )
        )

    def test_rejected_incomplete_nonfinite_or_unsafe_actions_remain_rejected(self) -> None:
        for changes in (
            {"action_accepted": False},
            {"action_completed": False},
            {"contact_ok": False},
            {"workspace_ok": False},
            {"executed_delta_angle_rad": math.nan},
        ):
            self.assertFalse(
                clamp_consistent_transition_valid(
                    record(0, **changes),
                    [action()],
                    angle_ceiling_exception_declared=True,
                )
            )
        for changes in (
            {"finite": False},
            {"surface_proxy_ok": False},
            {"command_mapping_valid": False},
            {"terminated": True},
            {"truncated": True},
        ):
            self.assertFalse(
                clamp_consistent_transition_valid(
                    record(0),
                    [action(**changes)],
                    angle_ceiling_exception_declared=True,
                )
            )

    def test_original_behavior_is_reproduced_and_alternative_is_named(self) -> None:
        rows = [observation(index) for index in range(3)]
        records = [record(index, action_clamped=False, action_angle_clamped=False) for index in range(3)]
        actions = {index: [action(command_angle_clamped=False)] for index in range(3)}
        original = observations_with_contract(
            rows,
            records,
            actions,
            contract=MovementContract.ORIGINAL_FROZEN,
            angle_ceiling_exception_declared=True,
        )
        diagnostic = observations_with_contract(
            rows,
            records,
            actions,
            contract=MovementContract.CLAMP_CONSISTENT_DIAGNOSTIC,
            angle_ceiling_exception_declared=True,
        )
        self.assertEqual(original, diagnostic)
        self.assertEqual(
            MovementContract.CLAMP_CONSISTENT_DIAGNOSTIC.value,
            "clamp_consistent_diagnostic",
        )


class EvidenceBoundaryTests(unittest.TestCase):
    def test_identical_hashes_and_nonadvancing_frames_do_not_accumulate(self) -> None:
        identical = [observation(index, hash_name="same") for index in range(3)]
        for row in identical:
            object.__setattr__(row, "transition_valid", True)
        self.assertLess(candidate_support_profile(identical, config()).maximum_distinct_hashes, 3)
        stale = [observation(0)] + [
            observation(index, producer_advanced=False) for index in (1, 2)
        ]
        for row in stale:
            object.__setattr__(row, "transition_valid", True)
        self.assertEqual(candidate_support_profile(stale, config()).potential_candidate_count, 1)

    def test_reset_boundary_and_evaluator_leakage_remain_rejected(self) -> None:
        rows = [observation(0), observation(1)]
        object.__setattr__(rows[1], "trajectory_id", "after-reset")
        with self.assertRaisesRegex(ValueError, "trajectory boundary"):
            replay_decision(rows, policy="candidate_c_temporal", config=config())
        leaked = record(0, goal_cmd_pose=[0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "privileged actor field"):
            clamp_consistent_transition_valid(
                leaked, [action()], angle_ceiling_exception_declared=True
            )

    def test_diagnostic_can_change_only_movement_admissibility(self) -> None:
        rows = [observation(index) for index in range(3)]
        records = [record(index) for index in range(3)]
        actions = {index: [action()] for index in range(3)}
        original = observations_with_contract(
            rows,
            records,
            actions,
            contract=MovementContract.ORIGINAL_FROZEN,
            angle_ceiling_exception_declared=True,
        )
        diagnostic = observations_with_contract(
            rows,
            records,
            actions,
            contract=MovementContract.CLAMP_CONSISTENT_DIAGNOSTIC,
            angle_ceiling_exception_declared=True,
        )
        self.assertEqual(
            replay_decision(
                original, policy="candidate_d_multi_view", config=config()
            ).state,
            DecisionState.ABSTAIN,
        )
        self.assertEqual(
            replay_decision(
                diagnostic, policy="candidate_d_multi_view", config=config()
            ).state,
            DecisionState.ACCEPT,
        )


if __name__ == "__main__":
    unittest.main()
