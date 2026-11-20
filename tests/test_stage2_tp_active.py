from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.stage2_tp_active import (  # noqa: E402
    TargetView,
    group_a_target_handoff_pose,
    proxy_confirmation_requirement_met,
    select_target_view,
)


TARGET = "L4_RIGHT_TP_SURFACE_CORE"


def view(
    sequence: int,
    *,
    pose: tuple[float, float, float] = (0.0, 0.0, 2.8),
    frame_hash: str | None = None,
    probability: float = 0.8,
    valid: bool = True,
) -> TargetView:
    return TargetView(
        target=TARGET,
        frame_sequence=sequence,
        producer_sequence=sequence + 1,
        frame_sha256=frame_hash or f"{sequence:064x}",
        pose_index=pose,
        probability=probability,
        proposal_row=120.0,
        proposal_column=72.0,
        proposal_track_id="H0_BASELINE",
        transition_valid=valid,
        image_valid=valid,
        producer_advanced=valid,
    )


class GroupAHandoffTests(unittest.TestCase):
    def test_target_handoff_uses_only_actor_map_and_declared_home(self) -> None:
        actor = {
            "schema": "sonogym-stage1-group-a-confirmation-actor-v1",
            "training_or_threshold_selection_performed": False,
            "results": [
                {
                    "session_id": "s0",
                    "map_actor_status": "SUCCESS",
                    "declared_home": {"patient_x_index": 133.0, "angle_rad": 2.75},
                    "predictions": [
                        {
                            "anatomy": "L4",
                            "actor_status": "SUCCESS",
                            "actor_coordinate_mm": -81.0,
                        }
                    ],
                }
            ],
        }
        pose = group_a_target_handoff_pose(
            actor,
            session_id="s0",
            target_level="L4",
            home_z_index=195.0,
            label_resolution_mm=1.5,
            stage2_angle_rad=2.85,
        )
        np.testing.assert_allclose(pose, [133.0, 141.0, 2.85])

    def test_privileged_field_is_rejected(self) -> None:
        actor = {
            "schema": "sonogym-stage1-group-a-confirmation-actor-v1",
            "training_or_threshold_selection_performed": False,
            "goal_cmd_pose": [1, 2, 3],
        }
        with self.assertRaisesRegex(ValueError, "privileged"):
            group_a_target_handoff_pose(
                actor,
                session_id="s0",
                target_level="L4",
                home_z_index=195.0,
                label_resolution_mm=1.5,
                stage2_angle_rad=2.85,
            )


class TargetSelectionTests(unittest.TestCase):
    def settings(self) -> dict[str, object]:
        return {
            "target": TARGET,
            "probability_threshold": 0.5,
            "required_support": 3,
            "neighborhood_translation_mm": 7.0,
            "neighborhood_angle_deg": 6.0,
            "minimum_view_translation_mm": 1.0,
            "minimum_view_angle_deg": 1.0,
            "label_resolution_mm": 1.5,
        }

    def test_duplicate_hashes_do_not_create_support(self) -> None:
        rows = [view(i, pose=(float(i), 0.0, 2.8), frame_hash="a" * 64) for i in range(3)]
        result = select_target_view(rows, **self.settings())
        self.assertEqual(result.status, "ABSTAIN")
        self.assertEqual(result.distinct_hash_count, 1)

    def test_same_pose_different_hashes_are_not_distinct_viewpoints(self) -> None:
        result = select_target_view([view(0), view(1), view(2)], **self.settings())
        self.assertEqual(result.status, "ABSTAIN")
        self.assertEqual(result.distinct_viewpoint_count, 1)

    def test_three_valid_views_accept_deterministically(self) -> None:
        rows = [
            view(0, pose=(0.0, 0.0, 2.8), probability=0.7),
            view(1, pose=(1.0, 0.0, 2.8), probability=0.9),
            view(2, pose=(2.0, 0.0, 2.8), probability=0.8),
        ]
        first = select_target_view(rows, **self.settings())
        second = select_target_view(rows, **self.settings())
        self.assertEqual(first, second)
        self.assertEqual(first.status, "ACCEPT")
        self.assertEqual(first.selected.frame_sequence, 1)
        self.assertEqual(first.distinct_viewpoint_count, 3)

    def test_support_is_gate_then_highest_probability_wins(self) -> None:
        rows = [
            view(0, pose=(0.0, 0.0, 2.8), probability=0.60),
            view(1, pose=(1.0, 0.0, 2.8), probability=0.61),
            view(2, pose=(2.0, 0.0, 2.8), probability=0.62),
            view(3, pose=(3.0, 0.0, 2.8), probability=0.95),
        ]
        result = select_target_view(rows, **self.settings())
        self.assertEqual(result.status, "ACCEPT")
        self.assertEqual(result.selected.frame_sequence, 3)

    def test_invalid_movement_and_future_target_do_not_support(self) -> None:
        rows = [
            view(0, pose=(0.0, 0.0, 2.8)),
            view(1, pose=(1.0, 0.0, 2.8), valid=False),
            view(2, pose=(2.0, 0.0, 2.8)),
            TargetView(**{**view(3, pose=(3.0, 0.0, 2.8)).__dict__, "target": "L5_RIGHT_TP_SURFACE_CORE"}),
        ]
        result = select_target_view(rows, **self.settings())
        self.assertEqual(result.status, "ABSTAIN")
        self.assertEqual(result.distinct_viewpoint_count, 2)


class ProxyConfirmationEvaluationTests(unittest.TestCase):
    def test_two_of_three_positive_confirmations_pass(self) -> None:
        self.assertTrue(
            proxy_confirmation_requirement_met(
                ["PROXY_POSITIVE", "PROXY_AMBIGUOUS", "PROXY_POSITIVE"],
                required_count=3,
                minimum_positive_count=2,
            )
        )

    def test_one_of_three_positive_confirmations_fails(self) -> None:
        self.assertFalse(
            proxy_confirmation_requirement_met(
                ["PROXY_AMBIGUOUS", "PROXY_NEGATIVE", "PROXY_POSITIVE"],
                required_count=3,
                minimum_positive_count=2,
            )
        )

    def test_invalid_confirmation_requirement_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid"):
            proxy_confirmation_requirement_met(
                ["PROXY_POSITIVE"], required_count=2, minimum_positive_count=3
            )


if __name__ == "__main__":
    unittest.main()

