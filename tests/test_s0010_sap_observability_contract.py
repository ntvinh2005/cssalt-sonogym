import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/unity_two_stage_transfer/s0010_sap_observability_protocol_v1.yaml"
COORDINATES_PATH = (
    ROOT
    / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/coordinates.py"
)
SPEC = importlib.util.spec_from_file_location(
    "sap_observability_coordinates", COORDINATES_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load coordinates module")
COORDINATES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COORDINATES
SPEC.loader.exec_module(COORDINATES)


class SapObservabilityContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))

    def test_protocol_is_sealed_and_supported(self) -> None:
        seal_path = PROTOCOL.with_suffix(".sha256")
        expected = seal_path.read_text(encoding="utf-8").split()[0]
        actual = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
        self.assertEqual(actual, expected)
        protocol_id = self.protocol["protocol_id"]
        self.assertTrue(COORDINATES.is_supported_stage2_dataset_protocol(protocol_id))
        self.assertTrue(COORDINATES.uses_coupled_stage2_dataset_controller(protocol_id))
        self.assertTrue(COORDINATES.uses_segmented_stage2_dataset_capture(protocol_id))

    def test_capture_is_small_and_contains_two_sessions_per_level(self) -> None:
        self.assertEqual(len(self.protocol["sessions"]), 2)
        self.assertEqual(
            self.protocol["sampling"]["trajectory_start_frames"], [0, 2, 4]
        )
        for bank_id, rows in self.protocol["observability_waypoints"].items():
            self.assertEqual(len(rows), 6)
            self.assertEqual(
                len(self.protocol["sampling"]["absolute_waypoint_banks_index"][bank_id]),
                6,
            )
            for level in ("L4", "L5"):
                positives = [
                    row
                    for row in rows
                    if row["level"] == level and row["role"] == "PSEUDO_SAP_TARGET"
                ]
                self.assertEqual(len(positives), 2)

    def test_scope_excludes_training_and_stage3(self) -> None:
        self.assertFalse(self.protocol["scope"]["model_training_authorized"])
        self.assertFalse(self.protocol["scope"]["stage3_authorized"])
        self.assertTrue(
            self.protocol["observability_gate"][
                "overlay_access_only_after_blind_decision_hash"
            ]
        )


if __name__ == "__main__":
    unittest.main()
