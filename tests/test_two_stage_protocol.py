#!/usr/bin/env python3
"""CPU tests for the frozen G0/G1 protocol and dataset separation."""

from __future__ import annotations

import hashlib
import sys
import unittest
from collections import Counter
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.annotations import (  # noqa: E402
    dataset_summary,
    validate_one_to_one_join,
    validate_privileged_rows,
    validate_public_rows,
    validate_split_assignments,
)
from two_stage_navigation.contracts import FORBIDDEN_ACTOR_FIELDS  # noqa: E402


PROTOCOL_PATH = REPOSITORY_ROOT / "experiments" / "unity_two_stage_transfer" / "protocol.yaml"
STAGE1_PROTOCOL_PATH = (
    REPOSITORY_ROOT / "experiments" / "unity_two_stage_transfer" / "stage1_protocol.yaml"
)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_protocol_is_bounded_and_does_not_authorize_gpu(self) -> None:
        self.assertEqual(
            self.protocol["protocol_id"], "sonogym-unity-two-stage-g0-g1-v2"
        )
        self.assertEqual(self.protocol["status"], "PRE_GPU_FROZEN")
        self.assertEqual(
            self.protocol["geometry"]["longitudinal_scan"][
                "maximum_action_steps_per_session"
            ],
            360,
        )
        self.assertEqual(
            self.protocol["amendment"]["prior_sha256"],
            "dc240b5bb333312f6bf85a9ff509f1beafd81e18f1ed458e75b498b5a017017a",
        )
        resources = self.protocol["resource_bound_if_later_approved"]
        self.assertEqual(resources["account"], "anest-dept")
        self.assertLessEqual(resources["cpu_cores"], 4)
        self.assertLessEqual(resources["gpu_count"], 1)
        self.assertFalse(resources["current_protocol_authorizes_submission"])
        self.assertEqual(self.protocol["scope"]["training"], "forbidden_during_capture")

    def test_session_matrix_and_splits_are_frozen_and_disjoint(self) -> None:
        sessions = self.protocol["sessions"]
        self.assertEqual(len(sessions), 16)
        self.assertEqual(len({session["session_id"] for session in sessions}), 16)
        self.assertEqual(len({session["seed"] for session in sessions}), 16)
        self.assertEqual(
            Counter(session["split"] for session in sessions),
            Counter({"train_dev": 8, "calibration_dev": 4, "untouched_audit": 4}),
        )
        # The 4 x-values by 4 angles occur exactly once, preventing post-hoc start selection.
        combinations = {
            (float(session["patient_x_index"]), float(session["angle_rad"]))
            for session in sessions
        }
        self.assertEqual(len(combinations), 16)
        validate_split_assignments(sessions)

    def test_stage1_protocol_is_sealed_and_preserves_locked_split(self) -> None:
        stage1 = yaml.safe_load(STAGE1_PROTOCOL_PATH.read_text(encoding="utf-8"))
        expected_hash = STAGE1_PROTOCOL_PATH.with_suffix(".sha256").read_text(
            encoding="utf-8"
        ).split()[0]
        actual_hash = hashlib.sha256(STAGE1_PROTOCOL_PATH.read_bytes()).hexdigest()
        self.assertEqual(actual_hash, expected_hash)
        self.assertEqual(
            stage1["protocol_id"], "sonogym-unity-two-stage-stage1-v2"
        )
        self.assertEqual(stage1["status"], "PRE_FIT_FROZEN")
        self.assertEqual(stage1["amendment"]["development_job_id"], 50915512)
        self.assertFalse(stage1["amendment"]["audit_opened"])
        self.assertEqual(stage1["split_policy"]["training"], "train_dev")
        self.assertEqual(
            stage1["split_policy"]["locked_evaluation"], "untouched_audit"
        )
        self.assertTrue(stage1["split_policy"]["open_audit_only_after_bundle_hash"])
        self.assertEqual(stage1["resource_bound"]["gpu_count"], 0)

    def test_protocol_preserves_information_boundary(self) -> None:
        forbidden = set(self.protocol["information_boundaries"]["actor_forbidden"])
        self.assertIn("true_l4_target_pose", forbidden)
        self.assertIn("absolute_patient_pose", forbidden)
        self.assertIn("evaluator_score", forbidden)
        self.assertTrue(forbidden.issubset(FORBIDDEN_ACTOR_FIELDS))
        self.assertTrue(
            self.protocol["labels"]["public_and_privileged_manifests_separate"]
        )


def public_row(frame_sequence: int = 0, frame_hash: str = "a" * 64) -> dict[str, object]:
    return {
        "session_id": "td_00",
        "frame_sequence": frame_sequence,
        "frame_sha256": frame_hash,
        "split": "train_dev",
        "image_path": f"public/td_00/{frame_sequence:04d}.npy",
        "executed_delta_home_mm": [0.0, -1.5],
    }


def privileged_row(frame_sequence: int = 0, frame_hash: str = "a" * 64) -> dict[str, object]:
    return {
        "session_id": "td_00",
        "frame_sequence": frame_sequence,
        "frame_sha256": frame_hash,
        "patient_id": "s0010",
        "l4_raw_label": 28,
        "command_pose_index": [132.0, 194.0, 2.7],
    }


class AnnotationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assignments = validate_split_assignments(
            [{"session_id": "td_00", "split": "train_dev"}]
        )

    def test_public_and_privileged_rows_join_only_by_identity(self) -> None:
        public = validate_public_rows([public_row()], self.assignments)
        privileged = validate_privileged_rows([privileged_row()])
        validate_one_to_one_join(public, privileged)
        summary = dataset_summary(public, privileged, "b" * 64)
        self.assertTrue(summary["one_to_one_join"])
        self.assertEqual(summary["actor_privileged_leakage"], 0)

    def test_privileged_field_in_public_manifest_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "privileged"):
            validate_public_rows([{**public_row(), "patient_id": "s0010"}], self.assignments)

    def test_orphan_or_hash_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "join mismatch"):
            validate_one_to_one_join([public_row()], [privileged_row(frame_hash="c" * 64)])

    def test_adjacent_frames_cannot_change_session_split(self) -> None:
        rows = [
            public_row(0),
            {**public_row(1, "d" * 64), "split": "calibration_dev"},
        ]
        with self.assertRaisesRegex(ValueError, "incorrect split"):
            validate_public_rows(rows, self.assignments)


if __name__ == "__main__":
    unittest.main()
