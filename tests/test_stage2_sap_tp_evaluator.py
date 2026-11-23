from __future__ import annotations

import unittest

import numpy as np

from workflows.two_stage_navigation.evaluate_stage2_tp_active import (
    binary_pair_metrics,
    proposal_from_frame,
    semantic_pair_identity,
)


class Stage2SapTpEvaluatorTests(unittest.TestCase):
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
