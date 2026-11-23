"""Focused checks for clinician-scan review-video semantics."""

from __future__ import annotations

import unittest

from workflows.two_stage_navigation.build_s0010_clinician_scan_review import (
    expected_transition,
    pair_verdict,
    short_state,
)


class ClinicianScanReviewTest(unittest.TestCase):
    def test_scan_families_remain_distinct(self) -> None:
        longitudinal = "LONGITUDINAL_MEDIAL_LATERAL_REVERSIBLE_SWEEP"
        transverse = "TRANSVERSE_CRANIAL_CAUDAL_SWEEP"
        self.assertIn("TP loss", expected_transition(longitudinal))
        self.assertIn("shadow", expected_transition(transverse))
        self.assertNotEqual(short_state(longitudinal), short_state(transverse))

    def test_pair_verdict_separates_false_accept_and_abstention(self) -> None:
        accept = {"tp_predicted": True, "sap_predicted": True}
        abstain = {"tp_predicted": False, "sap_predicted": False}
        positive = {"both_supported": True}
        negative = {"both_supported": False}
        self.assertEqual(pair_verdict(accept, negative), "PAIR FALSE ACCEPT")
        self.assertEqual(pair_verdict(abstain, positive), "PAIR MISSED / ABSTAINED")


if __name__ == "__main__":
    unittest.main()
