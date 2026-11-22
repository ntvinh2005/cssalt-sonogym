"""Tests for the frozen RIGHT L5 target-observability protocol generator."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
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

VERIFIER_PATH = Path(__file__).parents[1] / "workflows/two_stage_navigation/verify_exact_pose_capture.py"
VERIFIER_SPEC = importlib.util.spec_from_file_location(
    "l4mb_exact_pose_verifier", VERIFIER_PATH
)
assert VERIFIER_SPEC is not None and VERIFIER_SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(VERIFIER_SPEC)
sys.modules[VERIFIER_SPEC.name] = VERIFIER
VERIFIER_SPEC.loader.exec_module(VERIFIER)


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
        self.assertRegex(protocol["source"]["implementation_commit"], r"^[0-9a-f]{40}$")
        verifier = (root / "workflows/two_stage_navigation/verify_exact_pose_capture.py").read_text()
        self.assertIn(MODULE.PROTOCOL_ID, verifier)
        self.assertTrue(VERIFIER.requires_synchronized_capture(MODULE.PROTOCOL_ID))

    def test_verifier_recovers_commit_from_captured_run_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            session = run_root / "sessions" / "session_00"
            session.mkdir(parents=True)
            (run_root / "context").mkdir()
            commit = "3c7d656ba828df7e2230e4d15435a485e2323ab7"
            (run_root / "context/git-commit.txt").write_text(commit + "\n")
            resolved, origin = VERIFIER.resolve_source_commit({"source": {}}, session)
            self.assertEqual(resolved, commit)
            self.assertEqual(origin, "captured_run_context")

    def test_resume_job_preserves_parent_protocol_and_only_captures_missing_sessions(self) -> None:
        job = (
            Path(__file__).parents[1]
            / "hpg/jobs/s0010_l4mb_target_observability_v1_resume.sbatch"
        ).read_text()
        self.assertIn("52278401", job)
        self.assertIn("089cc97563c90847d91cc172daf9dd513a670713e75372aa0b660e45f68b540d", job)
        self.assertIn("cp -a \"$PARENT_RUN/sessions/s0010l4mbobs_00\"", job)
        self.assertNotIn("capture_session s0010l4mbobs_00", job)
        self.assertIn("capture_session s0010l4mbobs_01", job)
        self.assertIn("capture_session s0010l4mbobs_02", job)

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
