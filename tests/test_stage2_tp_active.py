from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.stage2_tp_active import (  # noqa: E402
    POC_PROTOCOL_IDS,
    TargetView,
    actor_confirmation_requirement_met,
    group_a_target_handoff_pose,
    perception_mode,
    proxy_confirmation_requirement_met,
    select_tp_reacquisition_confirmations,
    select_tp_refinement_anchor,
    select_target_view,
    tp_reacquisition_offsets,
    tp_refinement_is_required,
)


TARGET = "L4_RIGHT_TP_SURFACE_CORE"


class PerceptionModeTests(unittest.TestCase):
    def test_unet_only_mode_is_explicit(self) -> None:
        self.assertEqual(
            perception_mode({"perception": {"mode": "unet_only"}}),
            "unet_only",
        )

    def test_unknown_mode_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            perception_mode({"perception": {"mode": "unknown"}})


class TpReacquisitionOffsetTests(unittest.TestCase):
    SPEC = {
        "tp_local_search_offsets_index": [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
        "l5_single_view_tp_orientation_offsets_index": [
            [0.0, 0.0, -0.02],
            [0.0, 0.0, 0.02],
        ],
    }

    def test_single_view_l5_uses_bounded_orientation_rescan(self) -> None:
        self.assertEqual(
            tp_reacquisition_offsets(
                self.SPEC,
                level="L5",
                supporting_view_count=1,
                minimum_supporting_views=2,
            ),
            ((0.0, 0.0, -0.02), (0.0, 0.0, 0.02)),
        )

    def test_supported_l5_and_l4_keep_existing_offsets(self) -> None:
        expected = ((-1.5, 0.0, 0.0), (1.5, 0.0, 0.0))
        for level, count in (("L4", 1), ("L5", 2)):
            self.assertEqual(
                tp_reacquisition_offsets(
                    self.SPEC,
                    level=level,
                    supporting_view_count=count,
                    minimum_supporting_views=2,
                ),
                expected,
            )


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
    def test_supported_protocols_are_shared_by_actor_and_evaluator(self) -> None:
        for version in range(3, 7):
            self.assertIn(
                f"sonogym-s0010-mbb-single-anatomy-poc-v{version}",
                POC_PROTOCOL_IDS,
            )
        self.assertIn(
            "sonogym-s0010-mbb-single-anatomy-poc-v5r1",
            POC_PROTOCOL_IDS,
        )
        self.assertIn(
            "sonogym-s0010-mbb-single-anatomy-poc-v5r2",
            POC_PROTOCOL_IDS,
        )
        self.assertIn(
            "sonogym-s0010-mbb-single-anatomy-poc-v6r1",
            POC_PROTOCOL_IDS,
        )
        self.assertIn(
            "sonogym-s0010-mbb-single-anatomy-poc-v6r2",
            POC_PROTOCOL_IDS,
        )
        self.assertIn(
            "sonogym-s0010-mbb-single-anatomy-poc-v6r3",
            POC_PROTOCOL_IDS,
        )
        self.assertIn(
            "sonogym-s0010-mbb-single-anatomy-poc-v6r8",
            POC_PROTOCOL_IDS,
        )
        self.assertIn(
            "sonogym-s0010-mbb-single-anatomy-poc-v6r9",
            POC_PROTOCOL_IDS,
        )
        self.assertIn(
            "sonogym-s0010-mbb-single-anatomy-poc-v6r10",
            POC_PROTOCOL_IDS,
        )
        self.assertIn(
            "sonogym-s0010-mbb-single-anatomy-poc-v6r11",
            POC_PROTOCOL_IDS,
        )
        self.assertIn(
            "sonogym-s0010-mbb-single-anatomy-poc-v6r13",
            POC_PROTOCOL_IDS,
        )
        self.assertIn(
            "sonogym-s0010-mbb-single-anatomy-poc-v6r14",
            POC_PROTOCOL_IDS,
        )

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

    def test_actor_confirmation_allows_one_bounded_negative_view(self) -> None:
        rows = [
            view(0, probability=0.9, pose=(0.0, 0.0, 2.8)),
            view(1, probability=0.3, pose=(1.0, 0.0, 2.8)),
            view(2, probability=0.8, pose=(2.0, 0.0, 2.8)),
        ]
        self.assertTrue(
            actor_confirmation_requirement_met(
                rows,
                probability_threshold=0.5,
                required_count=3,
                minimum_positive_count=2,
            )
        )


class TpReacquisitionSelectionTests(unittest.TestCase):
    @staticmethod
    def row(
        sequence: int,
        *,
        pixels: int,
        kind: str = "LOCAL_SEARCH_1",
        predicted: bool = True,
    ) -> dict[str, object]:
        return {
            "confirmation_kind": kind,
            "frame_sequence": sequence,
            "frame_sha256": f"{sequence:064x}",
            "image_valid": True,
            "producer_advanced": True,
            "fresh_relative_to_source": True,
            "predicted": predicted,
            "executed_pose_index": [float(sequence), 0.0, 2.8],
            "semantic_observations": [
                {"pixel_count": pixels, "confidence": 0.9}
            ],
        }

    def test_refinement_anchor_prefers_supported_interface_over_score_spike(self) -> None:
        public = [
            {"target_probability": 0.95},
            {"target_probability": 0.75},
        ]
        anchor = select_tp_refinement_anchor(
            [self.row(0, pixels=12), self.row(1, pixels=80)],
            public,
        )
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor["frame_sequence"], 1)

    def test_refinement_runs_when_too_few_views_are_positive(self) -> None:
        rows = [
            self.row(0, pixels=80),
            self.row(1, pixels=0, predicted=False),
            self.row(2, pixels=0, predicted=False),
        ]
        self.assertTrue(
            tp_refinement_is_required(
                rows,
                minimum_predicted_count=3,
                large_interface_pixel_count=180,
            )
        )

    def test_refinement_runs_for_large_interface(self) -> None:
        rows = [self.row(sequence, pixels=200) for sequence in range(3)]
        self.assertTrue(
            tp_refinement_is_required(
                rows,
                minimum_predicted_count=3,
                large_interface_pixel_count=180,
            )
        )

    def test_refinement_skips_stable_compact_evidence(self) -> None:
        rows = [self.row(sequence, pixels=120) for sequence in range(3)]
        self.assertFalse(
            tp_refinement_is_required(
                rows,
                minimum_predicted_count=3,
                large_interface_pixel_count=180,
            )
        )

    def test_final_selection_uses_complete_refinement_bundle(self) -> None:
        public = [{"target_probability": 0.8} for _ in range(5)]
        rows = [
            self.row(0, pixels=200),
            self.row(1, pixels=80, kind="REFINEMENT_1"),
            self.row(2, pixels=90, kind="REFINEMENT_2"),
            self.row(3, pixels=100, kind="REFINEMENT_3"),
            self.row(4, pixels=120, kind="REFINEMENT_4", predicted=False),
        ]
        selected = select_tp_reacquisition_confirmations(
            rows,
            public,
            required_count=3,
            probability_threshold=0.5,
        )
        self.assertEqual(
            [row["frame_sequence"] for row in selected],
            [3, 2, 1],
        )

    def test_final_selection_falls_back_when_refinement_is_incomplete(self) -> None:
        public = [{"target_probability": 0.8} for _ in range(3)]
        rows = [
            self.row(0, pixels=70),
            self.row(1, pixels=90, kind="REFINEMENT_1"),
            self.row(2, pixels=80),
        ]
        selected = select_tp_reacquisition_confirmations(
            rows,
            public,
            required_count=3,
            probability_threshold=0.5,
        )
        self.assertEqual(
            [row["frame_sequence"] for row in selected],
            [0, 1, 2],
        )

    def test_incomplete_refinement_falls_back_to_detector_probability(self) -> None:
        public = [
            {"target_probability": 0.95},
            {"target_probability": 0.70},
            {"target_probability": 0.80},
        ]
        rows = [
            self.row(0, pixels=10),
            self.row(1, pixels=200, kind="REFINEMENT_1"),
            self.row(2, pixels=20),
        ]
        selected = select_tp_reacquisition_confirmations(
            rows,
            public,
            required_count=3,
            probability_threshold=0.5,
        )
        self.assertEqual(
            [row["frame_sequence"] for row in selected],
            [0, 2, 1],
        )


if __name__ == "__main__":
    unittest.main()

