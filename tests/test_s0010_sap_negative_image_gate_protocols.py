"""Tests for the three SAP development protocol clones."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = ROOT / "experiments/unity_two_stage_transfer"
PATH = (
    ROOT
    / "workflows/two_stage_navigation/generate_s0010_sap_negative_image_gate_protocols.py"
)
SPEC = importlib.util.spec_from_file_location("sap_negative_image_gate_protocols", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SapNegativeImageGateProtocolTests(unittest.TestCase):
    def test_clone_changes_only_identity_provenance_and_gate(self) -> None:
        for spec in MODULE.PROTOCOL_CLONES:
            source_path = PROTOCOL_DIR / spec.source_name
            source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
            cloned = MODULE.clone_protocol(
                source,
                source_path=source_path,
                protocol_id=spec.protocol_id,
                implementation_revision="test-revision",
            )
            self.assertTrue(
                cloned["label_gate"][
                    "require_all_proposals_clear_for_negative_image"
                ]
            )
            self.assertEqual(
                cloned["source"]["prior_protocol_sha256"],
                hashlib.sha256(source_path.read_bytes()).hexdigest(),
            )
            expected = deepcopy(source)
            expected["protocol_id"] = spec.protocol_id
            expected["source"] = cloned["source"]
            expected["label_gate"][
                "require_all_proposals_clear_for_negative_image"
            ] = True
            self.assertEqual(cloned, expected)

    def test_written_protocol_seal_matches(self) -> None:
        source_path = PROTOCOL_DIR / MODULE.PROTOCOL_CLONES[0].source_name
        source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        cloned = MODULE.clone_protocol(
            source,
            source_path=source_path,
            protocol_id="test-protocol",
            implementation_revision="test-revision",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "protocol.yaml"
            digest = MODULE.write_protocol(output, cloned)
            self.assertEqual(digest, hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(
                output.with_suffix(".sha256").read_text(encoding="utf-8"),
                f"{digest}  {output.name}\n",
            )


if __name__ == "__main__":
    unittest.main()
