from __future__ import annotations

import unittest

import numpy as np

from workflows.two_stage_navigation.evaluate_stage2_tp_active import (
    binary_pair_metrics,
    incremental_map_identity,
    proposal_from_frame,
    proxy_state,
    selected_proxy_or_full_independent_recovery,
    semantic_fresh_confirmation_rows,
    semantic_pair_identity,
)
from two_stage_navigation.tp_surface_core_dataset import PatchProposal


class Stage2SapTpEvaluatorTests(unittest.TestCase):
    def test_correct_but_uncredited_component_cannot_confirm_landmark(self) -> None:
        frame = dict(frame_sequence=0, frame_sha256="frame", image_valid=True,
                     producer_advanced=True, transition_valid=True, contact_ok=True)
        predicted = [
            dict(frame_sequence=0, frame_sha256="frame", row=r, column=2,
                 semantic_class="TP")
            for r in (2, 8)
        ]
        attempt = dict(
            identity="RIGHT_L4_TP", frame_sequence=0, frame_sha256="frame",
            predicted=True, image_valid=True, producer_advanced=True,
            fresh_relative_to_source=True, semantic_observations=predicted,
        )
        labels = {0: np.zeros((12, 12), dtype=np.uint8)}
        labels[0][2, 2] = 1
        data = {
            "fresh_landmark_confirmations": [attempt],
            "incremental_landmark_map": {"landmarks": {
                "RIGHT_L4_TP": {
                    "supporting_observations": [predicted[1]],
                    "fresh_confirmations": [attempt],
                },
                "RIGHT_L4_SAP": {"supporting_observations": [], "fresh_confirmations": []},
            }},
        }
        for canonical in (False, True):
            result = semantic_fresh_confirmation_rows(
                data, labels, [frame], level="L4", maximum_distance_pixels=3.0,
                canonical_evidence=canonical,
            )
            self.assertEqual(result[0]["identity_supported"], not canonical)
            self.assertEqual(result[0]["observation_support"], [False] if canonical else [True, False])

    def test_independent_map_can_recover_ambiguous_initial_tp_view(self) -> None:
        self.assertTrue(
            selected_proxy_or_full_independent_recovery(
                "PROXY_AMBIGUOUS",
                ["PROXY_POSITIVE", "PROXY_POSITIVE", "PROXY_POSITIVE"],
                independent_map_mode=True,
            )
        )
        self.assertFalse(
            selected_proxy_or_full_independent_recovery(
                "PROXY_AMBIGUOUS",
                ["PROXY_POSITIVE", "PROXY_AMBIGUOUS", "PROXY_POSITIVE"],
                independent_map_mode=True,
            )
        )
        self.assertFalse(
            selected_proxy_or_full_independent_recovery(
                "PROXY_AMBIGUOUS",
                ["PROXY_POSITIVE", "PROXY_POSITIVE", "PROXY_POSITIVE"],
                independent_map_mode=False,
            )
        )

    def test_incremental_map_identity_uses_separate_tp_and_sap_frames(self) -> None:
        labels = {
            index: np.zeros((12, 12), dtype=np.uint8) for index in range(4)
        }
        labels[0][2, 2] = 1
        labels[1][2, 3] = 1
        labels[2][7, 7] = 2
        labels[3][7, 8] = 2
        support = lambda frame, row, column: {
            "frame_sequence": frame,
            "frame_sha256": f"{frame:064d}",
            "viewpoint_id": f"view-{frame}",
            "row": row,
            "column": column,
            "coordinate_mm": [0.0, 0.0, 0.0],
            "confidence": 0.9,
        }
        sonoanatomy = {
            "incremental_landmark_map": {
                "landmarks": {
                    "RIGHT_L4_TP": {
                        "status": "CONFIRMED",
                        "supporting_observations": [
                            support(0, 2.0, 2.0),
                            support(1, 2.0, 3.0),
                        ],
                    },
                    "RIGHT_L4_SAP": {
                        "status": "CONFIRMED",
                        "supporting_observations": [
                            support(2, 7.0, 7.0),
                            support(3, 7.0, 8.0),
                        ],
                    },
                }
            }
        }

        result = incremental_map_identity(
            sonoanatomy, labels, level="L4", maximum_distance_pixels=0.0
        )

        self.assertTrue(result["accepted"])
        self.assertTrue(result["pair_correct"])

    def test_empty_projected_core_is_json_safe(self) -> None:
        empty = np.empty(0, dtype=np.float64)
        projections = {
            name: {
                "core_rows": empty,
                "core_columns": empty,
                "uncertain_rows": empty,
                "uncertain_columns": empty,
            }
            for name in ("L4_RIGHT_TP_SURFACE_CORE", "L5_RIGHT_TP_SURFACE_CORE")
        }
        state, diagnostics = proxy_state(
            "L4_RIGHT_TP_SURFACE_CORE",
            PatchProposal("track", 50.0, 50.0, np.empty(0)),
            projections,
        )

        self.assertEqual(state, "PROXY_NEGATIVE")
        self.assertIsNone(diagnostics["nearest_core_distance_feature_cells"])

    def test_missing_actor_proposal_is_explicit(self) -> None:
        self.assertIsNone(
            proposal_from_frame(
                {
                    "proposal_track_id": None,
                    "proposal_row": None,
                    "proposal_column": None,
                }
            )
        )

    def test_selected_clusters_require_majority_correct_tp_and_sap(self) -> None:
        labels = {
            index: np.zeros((12, 12), dtype=np.uint8) for index in range(4)
        }
        labels[0][2, 2] = 1
        labels[1][2, 3] = 1
        labels[2][7, 7] = 2
        labels[3][7, 8] = 2
        observations = [
            {"frame_sequence": 0, "row": 2.0, "column": 2.0},
            {"frame_sequence": 1, "row": 2.0, "column": 3.0},
            {"frame_sequence": 2, "row": 7.0, "column": 7.0},
            {"frame_sequence": 3, "row": 7.0, "column": 8.0},
        ]
        sonoanatomy = {
            "selected_trajectory_id": "sweep",
            "selected_pair_decision": {
                "selected": {"tp_cluster_index": 0, "sap_cluster_index": 1}
            },
            "trajectory_decisions": [
                {
                    "trajectory_id": "sweep",
                    "semantic_observations": observations,
                    "semantic_clusters": [
                        {"observation_indices": [0, 1]},
                        {"observation_indices": [2, 3]},
                    ],
                }
            ],
        }

        result = semantic_pair_identity(
            sonoanatomy, labels, maximum_distance_pixels=0.0
        )

        self.assertTrue(result["accepted"])
        self.assertTrue(result["pair_correct"])

    def test_pair_metrics_report_clear_negative_false_acceptance(self) -> None:
        result = binary_pair_metrics(
            [True, False, False, True],
            [True, True, False, False],
        )
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["tn"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["clear_negative_false_accept_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
