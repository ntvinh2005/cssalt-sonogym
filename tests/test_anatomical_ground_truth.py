#!/usr/bin/env python3
"""CPU tests for privileged anatomical annotation geometry and leakage guards."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.anatomical_ground_truth import (  # noqa: E402
    apply_affine,
    count_exact_shared_points,
    index_to_patient_local_m,
    patient_local_to_world_m,
    quantile_region,
    reject_privileged_actor_payload,
    validate_annotation,
)


def valid_annotation() -> dict[str, object]:
    return {
        "patient_id": "s0010",
        "anatomy_level": "L4",
        "raw_source_label": 28,
        "landmark_name": "L4_reference",
        "laterality": "MIDLINE",
        "annotation_type": "region",
        "coordinate_frame": "mesh_index",
        "units": {"mesh_index": "voxel_index"},
        "geometry": {"centroid": [1.0, 2.0, 3.0]},
        "source_files": [{"path": "mesh.stl", "sha256": "a" * 64}],
        "generation_method": "synthetic_test",
        "confidence": "HIGH",
        "positional_uncertainty_mm": 1.5,
        "supporting_visualizations": [],
        "limitations": [],
        "status": "PROVISIONAL",
    }


class CoordinateTests(unittest.TestCase):
    def test_affine_and_patient_world_round_trip(self) -> None:
        points = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        affine = np.eye(4)
        affine[:3, :3] *= 1.5
        affine[:3, 3] = [-10.0, 2.0, 7.0]
        expected = points * 1.5 + affine[:3, 3]
        np.testing.assert_allclose(apply_affine(points, affine), expected)

        patient = index_to_patient_local_m(points, 0.0015)
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation = np.asarray([0.2, -0.45, 1.0])
        world = patient_local_to_world_m(patient, rotation, translation)
        reconstructed = (world - translation) @ rotation
        np.testing.assert_allclose(reconstructed, patient, atol=1e-12)

    def test_quantile_region_preserves_source_indices(self) -> None:
        vertices = np.asarray(
            [[float(x), float(y), float(z)] for x in range(4) for y in range(4) for z in range(4)]
        )
        indices, points = quantile_region(
            vertices,
            x_fraction=(0.0, 0.25),
            y_fraction=(0.0, 1.0),
            z_fraction=(0.75, 1.0),
        )
        self.assertGreater(len(indices), 0)
        np.testing.assert_array_equal(points, vertices[indices])
        self.assertTrue(np.all(points[:, 0] <= 0.75))
        self.assertTrue(np.all(points[:, 2] >= 2.25))

    def test_exact_shared_point_count_detects_duplicate_boundaries(self) -> None:
        first = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        )
        second = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(count_exact_shared_points(first, second), 1)


class SchemaTests(unittest.TestCase):
    def test_provisional_annotation_requires_geometry(self) -> None:
        validate_annotation(valid_annotation())
        invalid = {**valid_annotation(), "geometry": None}
        with self.assertRaisesRegex(ValueError, "require geometry"):
            validate_annotation(invalid)

    def test_unresolved_annotation_may_omit_geometry(self) -> None:
        unresolved = {
            **valid_annotation(),
            "status": "UNRESOLVED",
            "confidence": "UNRESOLVED",
            "geometry": None,
            "positional_uncertainty_mm": None,
        }
        validate_annotation(unresolved)

    def test_closest_mesh_pair_cannot_be_provisional_facet_joint(self) -> None:
        invalid = {
            **valid_annotation(),
            "landmark_name": "facet_joint_region",
            "generation_method": "closest_candidate_articular_regions",
        }
        with self.assertRaisesRegex(ValueError, "cannot support"):
            validate_annotation(invalid)

        unresolved = {
            **invalid,
            "status": "UNRESOLVED",
            "confidence": "UNRESOLVED",
            "geometry": None,
            "positional_uncertainty_mm": None,
        }
        validate_annotation(unresolved)

    def test_nested_privileged_actor_field_fails(self) -> None:
        reject_privileged_actor_payload(
            {"ultrasound": [[0.0]], "history": [{"executed_delta": [0.0, 1.0]}]}
        )
        with self.assertRaisesRegex(ValueError, "privileged"):
            reject_privileged_actor_payload(
                {"ultrasound": [[0.0]], "history": [{"raw_label": 28}]}
            )


if __name__ == "__main__":
    unittest.main()
