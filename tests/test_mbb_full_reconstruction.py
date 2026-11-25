"""Verify depth-ray provenance and anatomy-only cortical connectivity."""

import unittest

import numpy as np

from two_stage_navigation.simple_landmark_mapping import CranialCoordinate, Stage1Region, project_image
from workflows.mbb_end_to_end.reconstruct import (
    OLD_COMPONENT, SUPPLEMENTAL_TRACK, cloud_arrays, first_supported_samples,
    learned_frame, oracle_frame,
)
from workflows.mbb_end_to_end.reference import face_properties, shortest_cortical_path
from workflows.mbb_observed_surface.audit import exposed_faces


def recording(shape):
    f = {"actor_base_to_ee_position_m": [0., 0., 0.], "actor_base_to_ee_rotation_columns": np.eye(3),
         "level_hypotheses": ("L5",), "tp_eligible_for_mapping": True,
         "frame_sequence": 0, "frame_sha256": "hash", "acquisition_id": "pose"}
    return {
        "frames": [f], "coords": {0: project_image(f, shape, .5)},
        "region": Stage1Region(0., 142., 124., (129, 135)),
        "fit": CranialCoordinate((0., 1., 0.), 133., 0., 1.5),
    }


class ReconstructionTests(unittest.TestCase):
    def test_depth_support_and_uncertain_fallback(self):
        candidates = np.zeros((2, 9, 4), dtype=np.uint8)
        candidates[0, 2, 0] = 1
        candidates[1, 5, 0] = 3
        candidates[0, 8, 1] = 3
        support = np.zeros((9, 4), dtype=bool)
        support[1:6, 0] = True
        support[2:5, 1] = True
        support[6:8, 1] = True
        self.assertEqual(first_supported_samples(candidates, support), [(2, 0, True), (3, 1, False)])

    def test_learned_exports_exact_pixels_and_level_conditioned_other(self):
        shape = (9, 6)
        public = recording(shape)
        posterior = np.zeros((2, 4, *shape), dtype=np.float32)
        posterior[:, 0] = 1.
        posterior[1, 0, 2:4, 1:3] = .02
        posterior[1, 1, 2:4, 1:3] = .98
        posterior[:, 0, 4:6, 3:5] = .04
        posterior[:, 3, 4:6, 3:5] = .96
        support = np.zeros(shape, dtype=bool)
        support[1:7, 1:6] = True
        rule = {name: {"threshold": .9, "minimum_pixels": 2} for name in ("TP", "SAP", "OTHER_BONE")}
        points, historical = learned_frame(public, 0, posterior, {"L4": rule, "L5": rule}, support)
        cloud = cloud_arrays(points)
        self.assertEqual(historical, 2)
        self.assertEqual(cloud["identity"].tolist(),
                         ["RIGHT_L5_TP", "RIGHT_L5_TP", "OTHER_BONE", "OTHER_BONE", "UNCERTAIN_BONE"])
        self.assertEqual(cloud["level_hint"].tolist(), ["L5"] * 5)
        self.assertEqual(cloud["source"][0], OLD_COMPONENT)
        self.assertEqual(cloud["source"][-1], SUPPLEMENTAL_TRACK)
        for k, (row, col) in enumerate(cloud["pixel_row_col"]):
            np.testing.assert_array_equal(cloud["semantic_probabilities_by_level"][k], posterior[:, :, row, col])
            np.testing.assert_array_equal(cloud["xyz_mm"][k], public["coords"][0][row, col])

    def test_oracle_preserves_other_and_uncertainty_only_on_first_supported_ray(self):
        shape = (8, 4)
        public = recording(shape)
        raw = np.zeros(shape, dtype=np.int16)
        raw[2:6, :] = 27
        labels = {level: np.full((1, *shape), 255, dtype=np.uint8) for level in ("L4", "L5")}
        labels["L5"][0, 2, 0] = 1
        labels["L5"][0, 5, 0] = 1
        labels["L5"][0, 2, 1] = 3
        labels["L4"][0, 2, 1] = 3
        support = np.ones(shape, dtype=bool)
        support[2, 3] = False
        points, codes, _ = oracle_frame(public, 0, labels, raw, {27}, support)
        cloud = cloud_arrays(points)
        self.assertEqual(cloud["identity"].tolist(), ["RIGHT_L5_TP", "OTHER_BONE", "UNCERTAIN_BONE"])
        self.assertEqual(cloud["pixel_row_col"][:, 0].tolist(), [2, 2, 2])
        self.assertEqual(cloud["level_hint"].tolist(), ["L5", "L5", "L5"])
        self.assertEqual(codes, [27, 27, 27])
        self.assertEqual(cloud["confidence"].tolist(), [1., 1., 0.])

    def test_exposed_face_path_follows_shared_cortical_edges(self):
        volume = np.zeros((5, 5, 5), dtype=bool)
        volume[2, 2, 1:4] = True
        faces, axes = exposed_faces(volume)
        owners, normals = face_properties(volume, faces, axes)
        self.assertTrue(volume[tuple(owners.T)].all())
        start = np.flatnonzero((normals[:, 0] == 1) & (owners[:, 2] == 1))
        end = np.flatnonzero((normals[:, 0] == 1) & (owners[:, 2] == 3))
        trace, length = shortest_cortical_path(faces, axes, np.full(3, 1.5), start, end)
        self.assertAlmostEqual(length, 3.)
        self.assertEqual(owners[trace, 2].tolist(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
