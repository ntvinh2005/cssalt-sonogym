"""Checks for frozen SAP/TP pair confirmation pose selection."""

from __future__ import annotations

import unittest

import numpy as np

from evaluate_s0010_sap_tp_pair_confirmation import target_margin
from prepare_s0010_sap_tp_pair_confirmation import (
    distinct,
    nearby,
    select_pairs,
)


class TestSapTpPairConfirmation(unittest.TestCase):
    def test_target_margin_uses_core_evidence(self) -> None:
        projected = {
            "sap": np.zeros((8, 10), dtype=bool),
            "tp": np.zeros((8, 10), dtype=bool),
            "sap_core": np.zeros((8, 10), dtype=bool),
            "tp_core": np.zeros((8, 10), dtype=bool),
        }
        projected["sap"][0, 0] = True
        projected["tp"][7, 9] = True
        projected["sap_core"][2, 3] = True
        projected["tp_core"][5, 6] = True

        self.assertEqual(target_margin(projected), 2)

    def test_pose_pairs_are_new_and_neighboring(self) -> None:
        rows = [
            {
                "pose_index_rad": [float(index * 4), 100.0, 2.0],
                "sap_first_visible_pixels": 10,
                "tp_first_visible_pixels": 10,
                "target_margin_pixels": 20,
            }
            for index in range(8)
        ]
        chosen = select_pairs(rows, 2, [{"pose_index_rad": [-20.0, 100.0, 2.0]}])
        self.assertEqual(len(chosen), 4)
        self.assertTrue(distinct(chosen[0], chosen[1]))
        self.assertTrue(nearby(chosen[0], chosen[1]))

    def test_duplicate_old_pose_is_excluded(self) -> None:
        rows = [
            {
                "pose_index_rad": [float(index * 4), 100.0, 2.0],
                "sap_first_visible_pixels": 10,
                "tp_first_visible_pixels": 10,
                "target_margin_pixels": 20,
            }
            for index in range(6)
        ]
        chosen = select_pairs(rows, 1, [{"pose_index_rad": [0.0, 100.0, 2.0]}])
        self.assertNotEqual(chosen[0]["pose_index_rad"], [0.0, 100.0, 2.0])


if __name__ == "__main__":
    unittest.main()
