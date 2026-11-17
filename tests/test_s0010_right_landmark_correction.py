"""Focused fail-closed tests for SAP/TP correspondence and boundary correction."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows/two_stage_navigation"
sys.path.insert(0, str(WORKFLOWS))
PATH = WORKFLOWS / "prepare_s0010_right_landmark_correction.py"
SPEC = importlib.util.spec_from_file_location("right_landmark_correction", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CorrespondenceSemanticsTests(unittest.TestCase):
    def test_unverified_projection_cannot_become_negative(self) -> None:
        row = MODULE.correspondence_state(
            projection_status="UNVERIFIED", intersects=False, raw_level_pixels=0
        )
        self.assertEqual(row["candidate_intersection"], "UNKNOWN")
        self.assertEqual(row["training_label"], "EXCLUDED")
        self.assertFalse(row["training_eligible"])

    def test_raw_level_presence_does_not_imply_acoustic_visibility(self) -> None:
        row = MODULE.correspondence_state(
            projection_status="VERIFIED", intersects=True, raw_level_pixels=200
        )
        self.assertTrue(row["raw_level_geometric_presence"])
        self.assertEqual(row["acoustic_visibility"], "NOT_REVIEWED")
        self.assertEqual(row["semantic_identity"], "NOT_REVIEWED")

    def test_verified_absence_is_candidate_specific_but_still_not_training_truth(self) -> None:
        row = MODULE.correspondence_state(
            projection_status="VERIFIED", intersects=False, raw_level_pixels=0
        )
        self.assertEqual(row["candidate_intersection"], "ABSENT")
        self.assertEqual(row["training_label"], "EXCLUDED")


class BoundaryRevisionTests(unittest.TestCase):
    def test_ct_transition_uses_largest_lateral_narrowing(self) -> None:
        segmentation = np.zeros((12, 12, 4), dtype=np.uint8)
        segmentation[2, 1:10, 1:3] = 28
        segmentation[3, 2:9, 1:3] = 28
        segmentation[4, 4:7, 1:3] = 28
        segmentation[5, 4:6, 1:3] = 28
        points = np.asarray([[2, 1, 1], [5, 5, 2]], dtype=float)
        self.assertEqual(
            MODULE._tp_base_transition(
                segmentation=segmentation, label=28, original_points=points
            ),
            4,
        )

    def test_export_preserves_source_indices_and_keeps_unresolved_sap_empty(self) -> None:
        vertices = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        original = np.asarray([True, True, False])
        candidate = MODULE.CandidateRevision(
            candidate_id="L4_RIGHT_superior_articular_process",
            level="L4",
            landmark="superior_articular_process",
            source_status="UNRESOLVED",
            vertices=vertices,
            faces=np.asarray([[0, 1, 2]], dtype=np.int32),
            original_mask=original,
            proposed_mask=np.zeros(3, dtype=bool),
            uncertain_mask=original.copy(),
            removed_mask=np.zeros(3, dtype=bool),
            added_mask=np.zeros(3, dtype=bool),
            status="UNRESOLVED_BOUNDARY",
            method="synthetic test",
            base_transition_x_index=None,
            source_mesh_path=Path("L4.stl"),
            source_mesh_sha256="a" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = MODULE.export_revision(candidate, Path(directory))
            archive = np.load(Path(directory) / result["npz"]["path"])
            np.testing.assert_array_equal(archive["original_selected_vertex_mask"], original)
            self.assertEqual(int(archive["proposed_revised_vertex_mask"].sum()), 0)
            self.assertEqual(str(archive["source_mesh_sha256"]), "a" * 64)

    def test_unreviewed_sap_cannot_be_a_tp_negative(self) -> None:
        state = MODULE.correspondence_state(
            projection_status="VERIFIED", intersects=False, raw_level_pixels=100
        )
        self.assertNotEqual(state["training_label"], "VERIFIED_NEGATIVE")


class InformationBoundaryTests(unittest.TestCase):
    def test_actor_patch_plan_contains_no_privileged_coordinate_source(self) -> None:
        source = (ROOT / "workflows/two_stage_navigation/prepare_s0010_right_landmark_correction.py").read_text()
        self.assertIn("actor_visible_bone_interface_morphology_proposals", source)
        self.assertNotIn('"patch_source_during_inference": "raw_segmentation', source)
        self.assertNotIn('"patch_source_during_inference": "CT', source)


if __name__ == "__main__":
    unittest.main()
