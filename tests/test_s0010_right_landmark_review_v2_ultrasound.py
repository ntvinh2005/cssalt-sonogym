#!/usr/bin/env python3
"""Focused CPU tests for review-v2 ultrasound geometry and fail-closed contracts."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows" / "two_stage_navigation"
sys.path.insert(0, str(WORKFLOWS))
PATH = WORKFLOWS / "prepare_s0010_right_landmark_review_v2.py"
SPEC = importlib.util.spec_from_file_location("right_landmark_review_v2_ultrasound", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ProjectionTests(unittest.TestCase):
    def test_axial_flat_surface_projection_has_expected_shape(self) -> None:
        surface = np.full((220, 240), 100.0)
        normals = np.zeros((220, 240, 3), dtype=np.float64)
        normals[..., 1] = -1.0
        coordinates, origin, rotation = MODULE.image_plane_from_realized_pose(
            np.asarray([130.0, 140.0, np.pi]), surface, normals
        )
        self.assertEqual(coordinates.shape, (200, 150, 3))
        self.assertEqual(origin.shape, (3,))
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        self.assertTrue(np.all(np.isfinite(coordinates)))

    def test_sample_volume_matches_integer_truncation_and_clipping(self) -> None:
        volume = np.arange(4 * 5 * 6).reshape(4, 5, 6)
        coordinates = np.asarray([[[1.9, 2.1, 3.8], [-2.0, 99.0, 5.9]]])
        sampled = MODULE.sample_volume(volume, coordinates)
        self.assertEqual(int(sampled[0, 0]), int(volume[1, 2, 3]))
        self.assertEqual(int(sampled[0, 1]), int(volume[0, 4, 5]))

    def test_projection_integrity_fails_closed(self) -> None:
        stored = np.zeros((10, 10), dtype=np.uint8)
        reconstructed = stored.copy()
        reconstructed[:5] = 28
        result = MODULE.projection_integrity(reconstructed, stored)
        self.assertFalse(result["verified"])
        self.assertLess(result["exact_label_agreement"], MODULE.PROJECTION_LABEL_AGREEMENT_MINIMUM)

    def test_candidate_projection_requires_plane_proximity(self) -> None:
        points = np.asarray([[0.0, 0.0, 4.0], [1.0, 1.0, 4.0]])
        mask, result = MODULE.candidate_projection(points, np.zeros(3), np.eye(3))
        self.assertFalse(result["intersects_plane"])
        self.assertEqual(int(np.count_nonzero(mask)), 0)


class InformationBoundaryTests(unittest.TestCase):
    def test_current_records_are_never_training_eligible(self) -> None:
        row = MODULE.FrameEvidence(
            source_id="source",
            source_root="/source",
            session_id="session",
            split="development",
            frame_sequence=1,
            producer_sequence=2,
            frame_sha256="a" * 64,
            phase="scan",
            realized_pose_index=np.asarray([1.0, 2.0, 3.0]),
            ultrasound=np.zeros((200, 150)),
            raw_labels=np.zeros((200, 150), dtype=np.uint8),
            ct_plane=np.zeros((200, 150)),
            candidate_mask=np.zeros((200, 150), dtype=bool),
            projection={"intersects_plane": True, "candidate_vertices_in_plane": 2, "projected_pixels": 3},
            integrity={"verified": True, "exact_label_agreement": 1.0, "bone_mask_iou": 1.0},
            raw_level_pixels=10,
            image_archive_sha256="b" * 64,
            label_archive_sha256="c" * 64,
        ).public_record("L4_RIGHT_transverse_process", "L4")
        self.assertFalse(row["specific_landmark_distinguishable"])
        self.assertFalse(row["training_label_eligible"])
        self.assertEqual(row["ultrasound_observability_status"], "AMBIGUOUS_NOT_SEMANTICALLY_VERIFIED")

    def test_duplicate_hashes_are_not_selected_twice(self) -> None:
        common = dict(
            source_id="source", source_root="/source", session_id="session", split="dev",
            producer_sequence=1, frame_sha256="a" * 64, phase="scan",
            realized_pose_index=np.zeros(3), ultrasound=np.zeros((2, 2)),
            raw_labels=np.zeros((2, 2), dtype=np.uint8), ct_plane=np.zeros((2, 2)),
            candidate_mask=np.zeros((2, 2), dtype=bool),
            projection={"intersects_plane": True, "candidate_vertices_in_plane": 1, "projected_pixels": 1},
            integrity={"verified": True, "exact_label_agreement": 1.0, "bone_mask_iou": 1.0},
            raw_level_pixels=1, image_archive_sha256="b" * 64, label_archive_sha256="c" * 64,
        )
        rows = [MODULE.FrameEvidence(frame_sequence=index, **common) for index in range(2)]
        selected = MODULE.select_review_frames(rows)
        self.assertEqual(len(selected), 1)


if __name__ == "__main__":
    unittest.main()
