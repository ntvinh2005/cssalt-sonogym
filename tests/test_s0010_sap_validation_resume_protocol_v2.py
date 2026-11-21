"""Tests for the focused SAP data completion capture design."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PATH = (
    ROOT
    / "workflows/two_stage_navigation/generate_s0010_sap_validation_resume_protocol_v2.py"
)
SPEC = importlib.util.spec_from_file_location("sap_resume_protocol_v2", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
COORDINATES_PATH = (
    ROOT
    / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/coordinates.py"
)
COORDINATES_SPEC = importlib.util.spec_from_file_location(
    "sap_resume_v2_coordinates", COORDINATES_PATH
)
assert COORDINATES_SPEC is not None and COORDINATES_SPEC.loader is not None
COORDINATES = importlib.util.module_from_spec(COORDINATES_SPEC)
sys.modules[COORDINATES_SPEC.name] = COORDINATES
COORDINATES_SPEC.loader.exec_module(COORDINATES)


class SapResumeProtocolV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        session = self.root / "source_session"
        (session / "privileged").mkdir(parents=True)
        annotations: list[dict[str, object]] = []
        manifest: list[dict[str, str]] = []
        frame = 0
        for level, z_positive, z_negative in (
            ("L4", 139.0, 148.0),
            ("L5", 122.0, 133.0),
        ):
            for index in range(12):
                pose = [145.0 + 4.0 * (index % 6), z_positive + 6.0 * (index // 6), 2.75 + 0.08 * (index % 3)]
                annotations.append(
                    {"frame_sequence": frame, "realized_pose_index": pose}
                )
                manifest.append(self.row(frame, level, "SAP_RELATED_CORTICAL_INTERFACE"))
                frame += 1
            for index in range(70):
                pose = [143.0 + 2.5 * (index % 10), z_negative - 2.5 * ((index // 10) % 6), 2.74 + 0.05 * (index % 6)]
                annotations.append(
                    {"frame_sequence": frame, "realized_pose_index": pose}
                )
                manifest.append(self.row(frame, level, "OTHER_OR_BACKGROUND"))
                frame += 1
        with (session / "privileged/annotations.jsonl").open("w", encoding="utf-8") as handle:
            for row in annotations:
                handle.write(json.dumps(row) + "\n")
        self.source_protocol_path = self.root / "source.yaml"
        self.source_protocol = {
            "sessions": [
                {
                    "session_id": "source",
                    "source_session_path": str(session),
                }
            ]
        }
        self.source_protocol_path.write_text(
            yaml.safe_dump(self.source_protocol), encoding="utf-8"
        )
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        with (self.dataset / "sample_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
        self.resume_path = self.root / "resume.yaml"
        self.resume = {
            "protocol_id": "sonogym-s0010-sap-validation-resume-v1",
            "source": {},
            "scope": {"runtime_classifier_geometry_access": False},
            "sampling": {
                "minimum_realized_viewpoint_translation_mm": 2.0,
                "minimum_realized_viewpoint_orientation_deg": 1.0,
                "physical_tracking_position_tolerance_mm": 1.5,
                "physical_tracking_orientation_tolerance_deg": 1.2,
            },
            "exact_pose_replay": {"label_resolution_m": 0.0015},
            "dataset_gate": {"minimum_distinct_frame_hashes_per_session": 40},
        }
        self.resume_path.write_text(yaml.safe_dump(self.resume), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def row(frame: int, level: str, label: str) -> dict[str, str]:
        positive = label == "SAP_RELATED_CORTICAL_INTERFACE"
        return {
            "session_id": "source",
            "frame_sequence": str(frame),
            "conditioned_level": level,
            "provisional_label": label,
            "selected_for_dataset": "True",
            "training_eligible": "True",
            "passes": str(positive),
            "surface_columns": "12" if positive else "0",
            "longest_aligned_run_columns": "9" if positive else "0",
            "aligned_fraction": "0.75" if positive else "0",
            "overlapping_surface_fraction": "0.8" if positive else "0",
            "distance_p90_pixels": "2.0" if positive else "inf",
            "local_gradient_median": "0.3" if positive else "0",
            "local_contrast_median": "0.2" if positive else "0",
            "posterior_shadow_median": "0.2" if positive else "0",
            "feature_gradient_p90": "0.5",
            "feature_intensity_std": "0.4",
            "surface_pixels_in_patch": "8" if positive else "0",
            "pseudo_pixels_in_patch": "8" if positive else "0",
            "uncertainty_pixels_in_patch": "0",
            "other_level_surface_pixels_in_patch": "0",
            "frame_sha256": f"{frame:064x}",
        }

    def document(self) -> dict[str, object]:
        return MODULE.build_protocol(
            self.source_protocol,
            self.source_protocol_path,
            self.dataset,
            self.resume,
            self.resume_path,
            "test-commit",
        )

    def test_sessions_supply_disjoint_coverage_and_tracking_margin(self) -> None:
        document = self.document()
        self.assertEqual(len(document["sessions"]), 5)
        self.assertEqual(
            sum(row["expected_frame_count"] for row in document["sessions"]), 220
        )
        banks = document["sampling"]["absolute_waypoint_banks_index"]
        all_poses = [tuple(row) for bank in banks.values() for row in bank]
        self.assertEqual(len(all_poses), len(set(all_poses)))
        margin = document["sampling"]["waypoint_tracking_margin"]
        for rows in banks.values():
            self.assertTrue(
                all(
                    MODULE.has_tracking_margin(left, right, margin)
                    for left, right in zip(rows, rows[1:])
                )
            )

    def test_planned_evidence_is_not_promoted_to_a_label(self) -> None:
        document = self.document()
        roles = document["sampling"]["planned_waypoint_evidence_by_session"]
        for session_id, rows in roles.items():
            self.assertEqual(len(rows), len(document["sampling"]["absolute_waypoint_banks_index"][session_id]))
            self.assertTrue(all(row["not_a_training_label"] for row in rows))
            self.assertEqual(
                {row["planned_label"] for row in rows},
                {"SAP_RELATED_CORTICAL_INTERFACE", "OTHER_OR_BACKGROUND"},
            )

    def test_capture_and_verifier_support_the_protocol_without_actor_geometry(self) -> None:
        protocol_id = "sonogym-s0010-sap-validation-resume-v2"
        self.assertTrue(COORDINATES.is_supported_stage2_dataset_protocol(protocol_id))
        self.assertTrue(COORDINATES.uses_coupled_stage2_dataset_controller(protocol_id))
        self.assertTrue(COORDINATES.uses_segmented_stage2_dataset_capture(protocol_id))
        capture = (ROOT / "hpg/jobs/s0010_sap_validation_capture.sbatch").read_text(
            encoding="utf-8"
        )
        verifier = (
            ROOT / "workflows/two_stage_navigation/verify_exact_pose_capture.py"
        ).read_text(encoding="utf-8")
        self.assertIn("    resume_v2)", capture)
        self.assertGreaterEqual(verifier.count(protocol_id), 2)
        self.assertFalse(
            self.document()["scope"]["runtime_classifier_geometry_access"]
        )

    def test_checked_in_protocol_is_sealed_and_preserves_split_counts(self) -> None:
        path = (
            ROOT
            / "experiments/unity_two_stage_transfer/s0010_sap_validation_resume_protocol_v2.yaml"
        )
        seal = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), seal)
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(document["protocol_id"], "sonogym-s0010-sap-validation-resume-v2")
        self.assertEqual(
            sum(row["expected_frame_count"] for row in document["sessions"]), 220
        )
        splits = {row["session_id"]: row["split"] for row in document["sessions"]}
        counts: dict[tuple[str, str, str], int] = {}
        for session_id, rows in document["sampling"][
            "planned_waypoint_evidence_by_session"
        ].items():
            for row in rows:
                key = (splits[session_id], row["level"], row["planned_label"])
                counts[key] = counts.get(key, 0) + 1
        for level in MODULE.LEVELS:
            self.assertEqual(
                counts[("train", level, "SAP_RELATED_CORTICAL_INTERFACE")], 24
            )
            self.assertEqual(
                counts[("calibration", level, "SAP_RELATED_CORTICAL_INTERFACE")],
                18,
            )
            self.assertEqual(
                counts[("evaluation", level, "SAP_RELATED_CORTICAL_INTERFACE")],
                24,
            )


if __name__ == "__main__":
    unittest.main()
