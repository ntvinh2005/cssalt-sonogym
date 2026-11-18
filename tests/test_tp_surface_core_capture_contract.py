#!/usr/bin/env python3
"""Static contract tests joining the sealed TP capture protocol and Slurm wrapper."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
COORDINATES_PATH = (
    ROOT
    / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/coordinates.py"
)
COORDINATES_SPEC = importlib.util.spec_from_file_location(
    "capture_contract_coordinates", COORDINATES_PATH
)
assert COORDINATES_SPEC is not None and COORDINATES_SPEC.loader is not None
COORDINATES = importlib.util.module_from_spec(COORDINATES_SPEC)
sys.modules[COORDINATES_SPEC.name] = COORDINATES
COORDINATES_SPEC.loader.exec_module(COORDINATES)
is_supported_stage2_dataset_protocol = COORDINATES.is_supported_stage2_dataset_protocol
uses_coupled_stage2_dataset_controller = COORDINATES.uses_coupled_stage2_dataset_controller
uses_segmented_stage2_dataset_capture = COORDINATES.uses_segmented_stage2_dataset_capture
PROTOCOL = ROOT / "experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v2.yaml"
SEAL = PROTOCOL.with_suffix(".sha256")
JOB = ROOT / "hpg/jobs/s0010_tp_surface_core_dataset_v2.sbatch"
PROTOCOL_V3 = (
    ROOT
    / "experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v3.yaml"
)
SEAL_V3 = PROTOCOL_V3.with_suffix(".sha256")
JOB_V3 = ROOT / "hpg/jobs/s0010_tp_surface_core_dataset_v3.sbatch"
PROTOCOL_V4 = (
    ROOT
    / "experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v4.yaml"
)
SEAL_V4 = PROTOCOL_V4.with_suffix(".sha256")
JOB_V4 = ROOT / "hpg/jobs/s0010_tp_surface_core_dataset_v4.sbatch"
PROTOCOL_V5 = (
    ROOT
    / "experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v5.yaml"
)
SEAL_V5 = PROTOCOL_V5.with_suffix(".sha256")
JOB_V5 = ROOT / "hpg/jobs/s0010_tp_surface_core_dataset_v5.sbatch"
PROTOCOL_V6 = (
    ROOT
    / "experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v6.yaml"
)
SEAL_V6 = PROTOCOL_V6.with_suffix(".sha256")
JOB_V6 = ROOT / "hpg/jobs/s0010_tp_surface_core_dataset_v6.sbatch"
PROTOCOL_V7 = (
    ROOT
    / "experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v7.yaml"
)
SEAL_V7 = PROTOCOL_V7.with_suffix(".sha256")
JOB_V7 = ROOT / "hpg/jobs/s0010_tp_surface_core_dataset_v7.sbatch"
PROTOCOL_V8 = (
    ROOT
    / "experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v8.yaml"
)
SEAL_V8 = PROTOCOL_V8.with_suffix(".sha256")
JOB_V8 = ROOT / "hpg/jobs/s0010_tp_surface_core_dataset_v8.sbatch"
PROTOCOL_V9 = (
    ROOT
    / "experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v9.yaml"
)
SEAL_V9 = PROTOCOL_V9.with_suffix(".sha256")
JOB_V9 = ROOT / "hpg/jobs/s0010_tp_surface_core_dataset_v9.sbatch"


class CaptureContractTests(unittest.TestCase):
    def test_protocol_seal_and_job_pointer(self) -> None:
        expected = SEAL.read_text().split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), expected)
        job = JOB.read_text()
        self.assertIn(PROTOCOL.name, job)
        self.assertIn(SEAL.name, job)
        self.assertNotIn("dataset_protocol_v1", job)

    def test_every_job_session_is_frozen_in_protocol(self) -> None:
        protocol = yaml.safe_load(PROTOCOL.read_text())
        frozen = {row["session_id"] for row in protocol["sessions"]}
        job = JOB.read_text()
        declared = set(
            re.findall(r"\bs0010tpscv2_(?:smoke|train_\d\d|cal_00|dev_00)\b", job)
        )
        self.assertEqual(declared, frozen)

    def test_new_capture_requires_serialized_hash_integrity(self) -> None:
        protocol = yaml.safe_load(PROTOCOL.read_text())
        self.assertTrue(protocol["exact_pose_replay"]["require_serialized_array_hash_integrity"])
        self.assertIn("ARRAY_HASH_INTEGRITY", JOB.read_text())


class CaptureV3ContractTests(unittest.TestCase):
    def test_protocol_seal_and_job_pointer(self) -> None:
        expected = SEAL_V3.read_text().split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V3.read_bytes()).hexdigest(), expected)
        job = JOB_V3.read_text()
        self.assertIn(PROTOCOL_V3.name, job)
        self.assertIn(SEAL_V3.name, job)
        self.assertNotIn("dataset_protocol_v2", job)

    def test_every_job_session_is_frozen_in_protocol(self) -> None:
        protocol = yaml.safe_load(PROTOCOL_V3.read_text())
        frozen = {row["session_id"] for row in protocol["sessions"]}
        job = JOB_V3.read_text()
        declared = set(
            re.findall(r"\bs0010tpscv3_(?:smoke|train_\d\d|cal_00|dev_00)\b", job)
        )
        self.assertEqual(declared, frozen)

    def test_retry_changes_only_the_declared_implementation_contract(self) -> None:
        previous = yaml.safe_load(PROTOCOL.read_text())
        current = yaml.safe_load(PROTOCOL_V3.read_text())
        self.assertEqual(current["amendment"]["prior_job"], 51798167)
        self.assertEqual(
            current["source"]["implementation_commit"],
            "31ca7e91a42678b6a28c24de7c585caca64e7987",
        )
        for key in (
            "sampling",
            "labels",
            "dataset_gate",
            "exact_pose_replay",
            "label_preparation",
            "training_preflight",
            "resource_bound",
        ):
            self.assertEqual(current[key], previous[key])

    def test_new_capture_requires_serialized_hash_integrity(self) -> None:
        protocol = yaml.safe_load(PROTOCOL_V3.read_text())
        self.assertTrue(protocol["exact_pose_replay"]["require_serialized_array_hash_integrity"])
        self.assertIn("ARRAY_HASH_INTEGRITY", JOB_V3.read_text())

    def test_collector_and_verifier_accept_the_frozen_protocol(self) -> None:
        protocol_id = yaml.safe_load(PROTOCOL_V3.read_text())["protocol_id"]
        self.assertTrue(is_supported_stage2_dataset_protocol(protocol_id))
        self.assertTrue(uses_coupled_stage2_dataset_controller(protocol_id))
        self.assertIn(protocol_id, (ROOT / "workflows/two_stage_navigation/verify_exact_pose_capture.py").read_text())


class CaptureV4ContractTests(unittest.TestCase):
    def test_protocol_seal_and_job_pointer(self) -> None:
        expected = SEAL_V4.read_text().split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V4.read_bytes()).hexdigest(), expected)
        job = JOB_V4.read_text()
        self.assertIn(PROTOCOL_V4.name, job)
        self.assertIn(SEAL_V4.name, job)
        self.assertNotIn("dataset_protocol_v3", job)

    def test_every_job_session_is_frozen_in_protocol(self) -> None:
        protocol = yaml.safe_load(PROTOCOL_V4.read_text())
        frozen = {row["session_id"] for row in protocol["sessions"]}
        job = JOB_V4.read_text()
        declared = set(
            re.findall(r"\bs0010tpscv4_(?:smoke|train_\d\d|cal_00|dev_00)\b", job)
        )
        self.assertEqual(declared, frozen)

    def test_retry_preserves_the_v3_scientific_contract(self) -> None:
        previous = yaml.safe_load(PROTOCOL_V3.read_text())
        current = yaml.safe_load(PROTOCOL_V4.read_text())
        self.assertEqual(current["amendment"]["prior_job"], 51798995)
        self.assertEqual(
            current["source"]["implementation_commit"],
            "707e9ca86d2a1b6466a9cb9d37636510a5d163a9",
        )
        for key in (
            "sampling",
            "labels",
            "dataset_gate",
            "exact_pose_replay",
            "label_preparation",
            "training_preflight",
            "resource_bound",
        ):
            self.assertEqual(current[key], previous[key])

    def test_collector_and_verifier_accept_the_frozen_protocol(self) -> None:
        protocol_id = yaml.safe_load(PROTOCOL_V4.read_text())["protocol_id"]
        self.assertTrue(is_supported_stage2_dataset_protocol(protocol_id))
        self.assertTrue(uses_coupled_stage2_dataset_controller(protocol_id))
        self.assertIn(
            protocol_id,
            (ROOT / "workflows/two_stage_navigation/verify_exact_pose_capture.py").read_text(),
        )


class CaptureV5ContractTests(unittest.TestCase):
    def test_protocol_seal_and_job_pointer(self) -> None:
        expected = SEAL_V5.read_text().split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V5.read_bytes()).hexdigest(), expected)
        job = JOB_V5.read_text()
        self.assertIn(PROTOCOL_V5.name, job)
        self.assertIn(SEAL_V5.name, job)
        self.assertNotIn("dataset_protocol_v4", job)

    def test_every_job_session_is_frozen_in_protocol(self) -> None:
        protocol = yaml.safe_load(PROTOCOL_V5.read_text())
        frozen = {row["session_id"] for row in protocol["sessions"]}
        job = JOB_V5.read_text()
        declared = set(
            re.findall(r"\bs0010tpscv5_(?:smoke|train_\d\d|cal_00|dev_00)\b", job)
        )
        self.assertEqual(declared, frozen)

    def test_retry_changes_only_segment_boundaries(self) -> None:
        previous = yaml.safe_load(PROTOCOL_V4.read_text())
        current = yaml.safe_load(PROTOCOL_V5.read_text())
        self.assertEqual(current["amendment"]["prior_job"], 51808727)
        self.assertEqual(
            current["source"]["implementation_commit"],
            "11cd5a4fc1b1825d79312ffc9312503bf24c5252",
        )
        for key in (
            "labels",
            "dataset_gate",
            "exact_pose_replay",
            "label_preparation",
            "training_preflight",
            "resource_bound",
        ):
            self.assertEqual(current[key], previous[key])
        current_sampling = dict(current["sampling"])
        starts = current_sampling.pop("trajectory_start_frames")
        self.assertEqual(starts, list(range(0, 50, 3)))
        self.assertEqual(current_sampling.pop("trajectory_seed_stride"), 1000)
        self.assertTrue(current_sampling.pop("enforce_episode_action_budget"))
        self.assertEqual(current_sampling, previous["sampling"])

    def test_collector_and_verifier_accept_segmented_protocol(self) -> None:
        protocol_id = yaml.safe_load(PROTOCOL_V5.read_text())["protocol_id"]
        self.assertTrue(is_supported_stage2_dataset_protocol(protocol_id))
        self.assertTrue(uses_coupled_stage2_dataset_controller(protocol_id))
        self.assertTrue(uses_segmented_stage2_dataset_capture(protocol_id))
        self.assertIn(
            protocol_id,
            (ROOT / "workflows/two_stage_navigation/verify_exact_pose_capture.py").read_text(),
        )


class CaptureV6ContractTests(unittest.TestCase):
    def test_protocol_seal_and_job_pointer(self) -> None:
        expected = SEAL_V6.read_text().split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V6.read_bytes()).hexdigest(), expected)
        job = JOB_V6.read_text()
        self.assertIn(PROTOCOL_V6.name, job)
        self.assertIn(SEAL_V6.name, job)
        self.assertNotIn("dataset_protocol_v5", job)

    def test_review_checksum_manifest_does_not_hash_itself(self) -> None:
        self.assertIn("! -name checksums.sha256", JOB_V6.read_text())

    def test_every_job_session_is_frozen_in_protocol(self) -> None:
        protocol = yaml.safe_load(PROTOCOL_V6.read_text())
        frozen = {row["session_id"] for row in protocol["sessions"]}
        job = JOB_V6.read_text()
        declared = set(
            re.findall(r"\bs0010tpscv6_(?:smoke|train_\d\d|cal_00|dev_00)\b", job)
        )
        self.assertEqual(declared, frozen)

    def test_retry_preserves_the_v5_acquisition_contract(self) -> None:
        previous = yaml.safe_load(PROTOCOL_V5.read_text())
        current = yaml.safe_load(PROTOCOL_V6.read_text())
        self.assertEqual(current["amendment"]["prior_job"], 51810449)
        self.assertEqual(
            current["source"]["implementation_commit"],
            "a3d0c8f5acabd65f28f0d451551b1e7d107a52ec",
        )
        for key in (
            "sampling",
            "labels",
            "dataset_gate",
            "exact_pose_replay",
            "label_preparation",
            "training_preflight",
            "resource_bound",
        ):
            self.assertEqual(current[key], previous[key])

    def test_collector_and_verifier_accept_segmented_protocol(self) -> None:
        protocol_id = yaml.safe_load(PROTOCOL_V6.read_text())["protocol_id"]
        self.assertTrue(is_supported_stage2_dataset_protocol(protocol_id))
        self.assertTrue(uses_coupled_stage2_dataset_controller(protocol_id))
        self.assertTrue(uses_segmented_stage2_dataset_capture(protocol_id))
        self.assertIn(
            protocol_id,
            (ROOT / "workflows/two_stage_navigation/verify_exact_pose_capture.py").read_text(),
        )


class CaptureV7ContractTests(unittest.TestCase):
    def test_protocol_seal_and_job_pointer(self) -> None:
        expected = SEAL_V7.read_text().split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V7.read_bytes()).hexdigest(), expected)
        job = JOB_V7.read_text()
        self.assertIn(PROTOCOL_V7.name, job)
        self.assertIn(SEAL_V7.name, job)

    def test_supplement_capture_sessions_are_frozen(self) -> None:
        protocol = yaml.safe_load(PROTOCOL_V7.read_text())
        captured = {
            row["session_id"]
            for row in protocol["sessions"]
            if row["role"] in {"smoke", "supplemental_dataset"}
        }
        declared = set(
            re.findall(
                r"\bs0010tpscv7_(?:smoke|train_l4_00|cal_l4_00|dev_l4_00)\b",
                JOB_V7.read_text(),
            )
        )
        self.assertEqual(declared, captured)

    def test_supplement_targets_only_demonstrated_l4_coverage_deficit(self) -> None:
        protocol = yaml.safe_load(PROTOCOL_V7.read_text())
        self.assertEqual(protocol["amendment"]["prior_job"], 51811811)
        self.assertEqual(
            protocol["source"]["implementation_commit"],
            "886afadf029de636faf4630b0acc940f2d43ebfd",
        )
        bank = protocol["sampling"]["absolute_waypoint_banks_index"]["l4_supplement"]
        self.assertEqual(len(bank), 24)
        supplemental = [
            row for row in protocol["sessions"] if row["role"] == "supplemental_dataset"
        ]
        self.assertEqual(
            {row["split"] for row in supplemental},
            {"train", "calibration", "development_evaluation"},
        )
        self.assertTrue(all(row["expected_frame_count"] == 24 for row in supplemental))

    def test_collector_and_verifier_accept_segmented_protocol(self) -> None:
        protocol_id = yaml.safe_load(PROTOCOL_V7.read_text())["protocol_id"]
        self.assertTrue(is_supported_stage2_dataset_protocol(protocol_id))
        self.assertTrue(uses_coupled_stage2_dataset_controller(protocol_id))
        self.assertTrue(uses_segmented_stage2_dataset_capture(protocol_id))
        self.assertIn(
            protocol_id,
            (ROOT / "workflows/two_stage_navigation/verify_exact_pose_capture.py").read_text(),
        )


class CaptureV8ContractTests(unittest.TestCase):
    def test_protocol_seal_and_job_pointer(self) -> None:
        expected = SEAL_V8.read_text().split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V8.read_bytes()).hexdigest(), expected)
        job = JOB_V8.read_text()
        self.assertIn(PROTOCOL_V8.name, job)
        self.assertIn(SEAL_V8.name, job)

    def test_recovery_preserves_tolerance_and_episode_budget(self) -> None:
        protocol = yaml.safe_load(PROTOCOL_V8.read_text())
        sampling = protocol["sampling"]
        self.assertEqual(protocol["amendment"]["prior_job"], 51815016)
        self.assertEqual(sampling["physical_tracking_position_tolerance_mm"], 1.5)
        self.assertEqual(sampling["physical_tracking_orientation_tolerance_deg"], 1.0)
        self.assertEqual(sampling["maximum_action_steps_per_waypoint"], 150)
        self.assertEqual(sampling["trajectory_start_frames"], list(range(0, 24, 2)))
        self.assertLessEqual(
            sampling["maximum_initial_positioning_steps"]
            + sampling["maximum_action_steps_per_waypoint"],
            600,
        )

    def test_only_verified_v7_train_is_inherited(self) -> None:
        protocol = yaml.safe_load(PROTOCOL_V8.read_text())
        inherited_v7 = [
            row["session_id"]
            for row in protocol["sessions"]
            if row["role"] == "inherited_supplement"
        ]
        self.assertEqual(inherited_v7, ["s0010tpscv7_train_l4_00"])
        self.assertNotIn("s0010tpscv7_cal_l4_00", inherited_v7)

    def test_collector_and_verifier_accept_segmented_protocol(self) -> None:
        protocol_id = yaml.safe_load(PROTOCOL_V8.read_text())["protocol_id"]
        self.assertTrue(is_supported_stage2_dataset_protocol(protocol_id))
        self.assertTrue(uses_coupled_stage2_dataset_controller(protocol_id))
        self.assertTrue(uses_segmented_stage2_dataset_capture(protocol_id))
        self.assertIn(
            protocol_id,
            (ROOT / "workflows/two_stage_navigation/verify_exact_pose_capture.py").read_text(),
        )


class CaptureV9ContractTests(unittest.TestCase):
    def test_protocol_seal_and_job_pointer(self) -> None:
        expected = SEAL_V9.read_text().split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V9.read_bytes()).hexdigest(), expected)
        job = JOB_V9.read_text()
        self.assertIn(PROTOCOL_V9.name, job)
        self.assertIn(SEAL_V9.name, job)

    def test_only_missing_development_session_is_captured(self) -> None:
        protocol = yaml.safe_load(PROTOCOL_V9.read_text())
        captured = [
            row for row in protocol["sessions"] if row["role"] == "supplemental_dataset"
        ]
        self.assertEqual([row["session_id"] for row in captured], ["s0010tpscv9_dev_l4_00"])
        self.assertEqual(captured[0]["split"], "development_evaluation")
        job = JOB_V9.read_text()
        self.assertIn("SESSION_ID=s0010tpscv9_dev_l4_00", job)
        self.assertNotIn("s0010tpscv8_dev_l4_00", job)

    def test_completion_bank_excludes_demonstrated_unreachable_angles(self) -> None:
        protocol = yaml.safe_load(PROTOCOL_V9.read_text())
        sampling = protocol["sampling"]
        bank = sampling["absolute_waypoint_banks_index"]["l4_dev_completion"]
        session = next(
            row for row in protocol["sessions"] if row["session_id"] == "s0010tpscv9_dev_l4_00"
        )
        requested_angles = [float(row[2]) + float(session["waypoint_offset_index"][2]) for row in bank]
        self.assertEqual(len(requested_angles), 24)
        self.assertLessEqual(max(requested_angles), 2.94)
        self.assertNotIn(2.985, requested_angles)
        self.assertEqual(sampling["physical_tracking_orientation_tolerance_deg"], 1.0)

    def test_failed_partials_are_not_inherited(self) -> None:
        protocol = yaml.safe_load(PROTOCOL_V9.read_text())
        inherited = {row["session_id"] for row in protocol["sessions"]}
        self.assertNotIn("s0010tpscv7_cal_l4_00", inherited)
        self.assertNotIn("s0010tpscv8_dev_l4_00", inherited)
        self.assertIn("s0010tpscv8_train_l4_00", inherited)
        self.assertIn("s0010tpscv8_cal_l4_00", inherited)

    def test_collector_and_verifier_accept_segmented_protocol(self) -> None:
        protocol_id = yaml.safe_load(PROTOCOL_V9.read_text())["protocol_id"]
        self.assertTrue(is_supported_stage2_dataset_protocol(protocol_id))
        self.assertTrue(uses_coupled_stage2_dataset_controller(protocol_id))
        self.assertTrue(uses_segmented_stage2_dataset_capture(protocol_id))
        self.assertIn(
            protocol_id,
            (ROOT / "workflows/two_stage_navigation/verify_exact_pose_capture.py").read_text(),
        )


if __name__ == "__main__":
    unittest.main()
