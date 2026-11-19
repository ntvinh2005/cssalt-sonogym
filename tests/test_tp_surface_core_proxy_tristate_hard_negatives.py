#!/usr/bin/env python3
"""CPU tests for L4 proxy tri-state and hard-negative preparation."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "workflows/two_stage_navigation/prepare_tp_surface_core_proxy_tristate_hard_negatives.py"
)
SPEC = importlib.util.spec_from_file_location("tp_proxy_tristate_hard_negatives", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPARE
SPEC.loader.exec_module(PREPARE)


def row(
    *,
    split: str = "train",
    projection: str = "VERIFIED",
    proxy_class: str = "",
    eligible: str = "False",
    l4: int = 0,
    l5: int = 0,
    l4u: int = 0,
    l5u: int = 0,
    frame: str = "frame",
    probability: float = 0.6,
    sample: str = "sample",
):
    return {
        "split": split,
        "projection_status": projection,
        "proxy_class_id": proxy_class,
        "proxy_condition_eligible": eligible,
        "proposal_source": "ACTOR_VISIBLE_BRIGHT_INTERFACE",
        "l4_core_points_in_patch": l4,
        "l5_core_points_in_patch": l5,
        "l4_uncertain_points_in_patch": l4u,
        "l5_uncertain_points_in_patch": l5u,
        "frame_sha256": frame,
        "probability_l4": probability,
        "sample_id": sample,
    }


class ProxyTriStateTests(unittest.TestCase):
    def test_complete_l4_contract_is_positive(self) -> None:
        state, _ = PREPARE.proposal_state(
            row(proxy_class="1", eligible="True", l4=12)
        )
        self.assertEqual(state, PREPARE.PROXY_POSITIVE)

    def test_l5_core_without_l4_evidence_is_l4_negative(self) -> None:
        state, _ = PREPARE.proposal_state(row(l5=20))
        self.assertEqual(state, PREPARE.PROXY_NEGATIVE)

    def test_l4_uncertainty_is_ambiguous(self) -> None:
        state, _ = PREPARE.proposal_state(row(l4u=3))
        self.assertEqual(state, PREPARE.PROXY_AMBIGUOUS)

    def test_incomplete_l4_core_is_ambiguous(self) -> None:
        state, _ = PREPARE.proposal_state(row(l4=3))
        self.assertEqual(state, PREPARE.PROXY_AMBIGUOUS)

    def test_unverified_projection_is_ambiguous(self) -> None:
        state, _ = PREPARE.proposal_state(row(projection="UNVERIFIED"))
        self.assertEqual(state, PREPARE.PROXY_AMBIGUOUS)

    def test_image_state_uses_positive_then_ambiguous_precedence(self) -> None:
        self.assertEqual(
            PREPARE.image_state(
                [PREPARE.PROXY_NEGATIVE, PREPARE.PROXY_AMBIGUOUS]
            ),
            PREPARE.PROXY_AMBIGUOUS,
        )
        self.assertEqual(
            PREPARE.image_state(
                [PREPARE.PROXY_AMBIGUOUS, PREPARE.PROXY_POSITIVE]
            ),
            PREPARE.PROXY_POSITIVE,
        )

    def test_hard_negative_requires_training_and_zero_all_uncertainty(self) -> None:
        candidate = row()
        candidate["l4_proxy_state"] = PREPARE.PROXY_NEGATIVE
        self.assertTrue(
            PREPARE.hard_negative_geometry_is_eligible(
                candidate,
                source_split="train",
                source_proposal="ACTOR_VISIBLE_BRIGHT_INTERFACE",
            )
        )
        candidate["l5_uncertain_points_in_patch"] = 1
        self.assertFalse(
            PREPARE.hard_negative_geometry_is_eligible(
                candidate,
                source_split="train",
                source_proposal="ACTOR_VISIBLE_BRIGHT_INTERFACE",
            )
        )

    def test_development_row_cannot_enter_hard_negative_manifest(self) -> None:
        candidate = row(split="development_evaluation")
        candidate["l4_proxy_state"] = PREPARE.PROXY_NEGATIVE
        self.assertFalse(
            PREPARE.hard_negative_geometry_is_eligible(
                candidate,
                source_split="train",
                source_proposal="ACTOR_VISIBLE_BRIGHT_INTERFACE",
            )
        )

    def test_one_patch_per_frame_is_deterministic(self) -> None:
        selected = PREPARE.select_one_per_frame(
            [
                row(frame="a", probability=0.6, sample="a2"),
                row(frame="a", probability=0.7, sample="a1"),
                row(frame="b", probability=0.8, sample="b1"),
            ]
        )
        self.assertEqual([item["sample_id"] for item in selected], ["a1", "b1"])

    def test_frozen_l4_gate_requires_argmax_and_threshold(self) -> None:
        classes = np.asarray([0, 1, 2])
        self.assertTrue(
            PREPARE.l4_is_frozen_positive(
                np.asarray([0.3, 0.5, 0.2]), classes, threshold=0.5
            )
        )
        self.assertFalse(
            PREPARE.l4_is_frozen_positive(
                np.asarray([0.51, 0.49, 0.0]), classes, threshold=0.5
            )
        )

    def test_source_contains_no_fit_call(self) -> None:
        self.assertNotIn(".fit(", SCRIPT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
