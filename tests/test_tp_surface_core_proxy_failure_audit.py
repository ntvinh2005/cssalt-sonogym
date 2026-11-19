#!/usr/bin/env python3
"""CPU tests for the frozen TP proxy failure audit."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "workflows/two_stage_navigation/audit_tp_surface_core_proxy_failures.py"
SPEC = importlib.util.spec_from_file_location("tp_proxy_failure_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def proposal(
    *,
    predicted: int = 1,
    l4: int = 0,
    l5: int = 0,
    l4_uncertain: int = 0,
    l5_uncertain: int = 0,
) -> dict[str, object]:
    return {
        "predicted_class_id": predicted,
        "l4_core_points_in_patch": l4,
        "l5_core_points_in_patch": l5,
        "l4_uncertain_points_in_patch": l4_uncertain,
        "l5_uncertain_points_in_patch": l5_uncertain,
    }


class FailureClassificationTests(unittest.TestCase):
    def test_competing_l5_overlap_has_direct_geometric_precedence(self) -> None:
        primary, evidence, _ = AUDIT.classify_false_positive([proposal(l5=2)])
        self.assertIn("competing L5", primary)
        self.assertTrue(any("competing L5" in value for value in evidence))

    def test_excluded_l4_projection_is_not_called_pure_morphology_confusion(self) -> None:
        primary, evidence, _ = AUDIT.classify_false_positive(
            [proposal(l4=2, l4_uncertain=1)]
        )
        self.assertIn("proxy label excludes projected L4", primary)
        self.assertTrue(any("uncertainty" in value for value in evidence))

    def test_multiple_accepted_proposals_are_counted(self) -> None:
        primary, evidence, _ = AUDIT.classify_false_positive(
            [proposal(l4_uncertain=1), proposal(l4_uncertain=1)]
        )
        self.assertIn("multiple accepted proposals", primary)
        self.assertTrue(any(value.startswith("2 proposals") for value in evidence))

    def test_clean_proxy_negative_is_reported_as_morphology_confusion(self) -> None:
        primary, evidence, _ = AUDIT.classify_false_positive([proposal()])
        self.assertIn("proxy-negative bone-interface", primary)
        self.assertTrue(any("no projected core" in value for value in evidence))

    def test_audit_source_contains_no_model_fit_call(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn(".fit(", source)

    def test_binary_counts_preserve_fixed_denominator(self) -> None:
        truth = {"a": {1}, "b": set(), "c": {1}, "d": set()}
        prediction = {"a": {1}, "b": {1}, "c": set(), "d": set()}
        self.assertEqual(
            AUDIT.binary_counts(truth, prediction, 1),
            {"tp": 1, "fp": 1, "fn": 1, "tn": 1},
        )


if __name__ == "__main__":
    unittest.main()
