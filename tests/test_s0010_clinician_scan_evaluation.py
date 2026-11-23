from __future__ import annotations

import unittest

from evaluate_s0010_clinician_scan import transition_evidence


class ClinicianScanEvaluationTests(unittest.TestCase):
    def test_transition_requires_disappearance_and_reverse_reappearance(self) -> None:
        result = transition_evidence([8, 7, 5, 0, 0, 0, 4, 6])
        self.assertTrue(result["passed"])
        self.assertEqual(result["disappearance_frame_in_trajectory"], 3)

    def test_always_visible_does_not_pass(self) -> None:
        self.assertFalse(transition_evidence([8, 7, 6, 5, 4, 5, 6, 7])["passed"])

    def test_disappearance_without_return_does_not_pass(self) -> None:
        self.assertFalse(transition_evidence([8, 7, 0, 0, 0, 0, 0, 0])["passed"])


if __name__ == "__main__":
    unittest.main()
