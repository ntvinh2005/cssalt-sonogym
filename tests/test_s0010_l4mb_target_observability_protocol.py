"""Tests for the frozen RIGHT L5 target-observability protocol generator."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import yaml

from two_stage_navigation.coordinates import (
    is_supported_stage2_dataset_protocol,
    uses_coupled_stage2_dataset_controller,
    uses_segmented_stage2_dataset_capture,
)
from two_stage_navigation.stage2_models import Stage2LabelConfig


PATH = Path(__file__).parents[1] / "workflows/two_stage_navigation/generate_s0010_l4mb_target_observability_protocol.py"
SPEC = importlib.util.spec_from_file_location("l4mb_observability_protocol", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(index: int, role: str, pose: list[float]) -> dict[str, object]:
    return {
        "session_id": "source",
        "frame_sequence": index,
        "frame_sha256": str(index).zfill(64),
        "requested_waypoint_pose_index": pose,
        "evaluator_label": role,
        "evaluator_reason": "reason",
    }


class ProtocolGeneratorTests(unittest.TestCase):
    def test_capture_protocol_uses_synchronized_segmented_controller(self) -> None:
        protocol_id = MODULE.PROTOCOL_ID
        self.assertTrue(is_supported_stage2_dataset_protocol(protocol_id))
        self.assertTrue(uses_coupled_stage2_dataset_controller(protocol_id))
        self.assertTrue(uses_segmented_stage2_dataset_capture(protocol_id))

    def test_frozen_protocol_passes_label_schema_and_verifier_allowlist(self) -> None:
        root = Path(__file__).parents[1]
        protocol = yaml.safe_load(
            (root / "experiments/unity_two_stage_transfer/s0010_l4mb_target_observability_v1.yaml").read_text()
        )
        Stage2LabelConfig(**protocol["labels"]["validity"], **protocol["labels"]["quality"])
        verifier = (root / "workflows/two_stage_navigation/verify_exact_pose_capture.py").read_text()
        self.assertIn(MODULE.PROTOCOL_ID, verifier)

    def test_pose_distance_uses_physical_translation_or_rotation(self) -> None:
        self.assertFalse(MODULE.pose_distance_clear([100, 100, 2.8], [102, 100, 2.8]))
        self.assertTrue(MODULE.pose_distance_clear([100, 100, 2.8], [103, 100, 2.8]))
        self.assertTrue(MODULE.pose_distance_clear([100, 100, 2.8], [100, 100, 2.85]))

    def test_ordered_anchors_fails_when_control_is_swallowed(self) -> None:
        target = row(0, "PROXY_POSITIVE", [100, 100, 2.8])
        control = row(1, "PROXY_NEGATIVE", [101, 100, 2.8])
        with self.assertRaisesRegex(ValueError, "target-to-control"):
            MODULE.ordered_anchors([target], [control])

    def test_ordered_anchors_preserves_role_identity(self) -> None:
        target = row(0, "PROXY_POSITIVE", [100, 100, 2.8])
        control = row(1, "PROXY_NEGATIVE", [105, 100, 2.8])
        result = MODULE.ordered_anchors([target], [control])
        self.assertEqual([item["evaluator_label"] for item in result], ["PROXY_POSITIVE", "PROXY_NEGATIVE"])


if __name__ == "__main__":
    unittest.main()
