#!/usr/bin/env python3
"""Static contract tests joining the sealed TP capture protocol and Slurm wrapper."""

from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v2.yaml"
SEAL = PROTOCOL.with_suffix(".sha256")
JOB = ROOT / "hpg/jobs/s0010_tp_surface_core_dataset_v2.sbatch"
PROTOCOL_V3 = (
    ROOT
    / "experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v3.yaml"
)
SEAL_V3 = PROTOCOL_V3.with_suffix(".sha256")
JOB_V3 = ROOT / "hpg/jobs/s0010_tp_surface_core_dataset_v3.sbatch"


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


if __name__ == "__main__":
    unittest.main()
