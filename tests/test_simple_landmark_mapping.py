from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from two_stage_navigation.incremental_landmark_map import IncrementalLandmarkMapConfig
from two_stage_navigation.simple_landmark_mapping import (
    Stage1Region, add_outer_transverse_samples, extract_observations, project_image, sap_level_at,
    simple_map, systematic_sweep, transverse_cranial_coordinate,
)
from two_stage_navigation.sonoanatomy_multiview import SemanticObservation


IDENTITY = "RIGHT_L4_TP"
CONFIG = IncrementalLandmarkMapConfig()


def observation(frame, x, *, pose_x=None, confidence=0.95):
    return SemanticObservation(
        semantic_class="TP", frame_sequence=frame, frame_sha256=str(frame),
        viewpoint_id=str(frame), viewpoint_position_m=(frame * 0.003 if pose_x is None else pose_x, 0, 0),
        viewpoint_rotation_columns=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        row=1, column=2, coordinate_mm=(x, 0, 0),
        confidence=confidence, pixel_count=5,
    )


class SimpleLandmarkMappingTests(unittest.TestCase):
    def test_support_needs_distinct_poses_and_hashes(self):
        rows = [
            observation(0, 0), observation(1, 0.2, pose_x=0.001),
            replace(observation(2, 0.3), frame_sha256="0"),
        ]
        result = simple_map({IDENTITY: rows}, {}, CONFIG)[IDENTITY]
        self.assertEqual(result["status"], "CANDIDATE")
        self.assertEqual(result["selected"]["distinct_viewpoints"], 1)
        rows.append(observation(3, 0.4))
        result = simple_map({IDENTITY: rows}, {}, CONFIG)[IDENTITY]
        self.assertEqual(result["status"], "MAPPED")
        self.assertEqual(result["selected"]["distinct_viewpoints"], 2)

    def test_radius_bound_prevents_spatial_chain(self):
        rows = [observation(index, x) for index, x in enumerate((0.0, 3.0, 6.0))]
        result = simple_map({IDENTITY: rows}, {}, CONFIG)[IDENTITY]
        self.assertEqual(result["selected"]["distinct_viewpoints"], 2)
        self.assertEqual(len(result["competing_clusters"]), 1)

    def test_equally_supported_clusters_and_boundary_stay_unresolved(self):
        rows = [observation(index, x) for index, x in enumerate((0, 0.2, 8, 8.2))]
        result = simple_map({IDENTITY: rows}, {}, CONFIG)
        self.assertEqual(result[IDENTITY]["status"], "AMBIGUOUS")
        self.assertEqual(result["RIGHT_L4_SAP"]["status"], "NOT_OBSERVED")
        self.assertEqual(simple_map({}, {IDENTITY: rows}, CONFIG)[IDENTITY]["status"], "AMBIGUOUS")

    def test_systematic_motion_uses_two_planes_and_both_references(self):
        region = Stage1Region(135, 140, 122, (129, 133))
        scan = {
            "translation_step_index": 3., "longitudinal_angle_rad": np.pi / 2,
            "transverse_angle_rad": 2.95, "reference_offsets_index": [-6., 6.],
        }
        route = systematic_sweep(region, scan, (100, 70, 1), (180, 200, 3))
        self.assertEqual(route, systematic_sweep(region, scan, (100, 70, 1), (180, 200, 3)))
        self.assertEqual({pose[1] for plane, pose in route if plane == "longitudinal"}, {122, 140})
        transverse = [pose for plane, pose in route if plane == "transverse"]
        self.assertEqual({pose[2] for pose in transverse}, {2.95})
        self.assertEqual(len({pose[0] for pose in transverse}), 3)
        self.assertEqual((min(pose[1] for pose in transverse), max(pose[1] for pose in transverse)), (116, 146))
        self.assertEqual(region.levels_at(129), ("L4", "L5"))
        self.assertEqual(region.levels_at(134), ("L4",))
        self.assertEqual(region.levels_at(128), ("L5",))
        with self.assertRaisesRegex(ValueError, "workspace"):
            systematic_sweep(region, scan, (100, 70, 1), (150, 200, 3))

    def test_midpoints_are_only_added_to_outer_l4_transverse_segment(self):
        region = Stage1Region(133, 141.425, 122.830, (130.628, 133.795))
        scan = {
            "translation_step_index": 3., "longitudinal_angle_rad": np.pi / 2,
            "transverse_angle_rad": 2.95, "reference_offsets_index": [-6., 6.],
        }
        original = systematic_sweep(region, scan, (100, 70, 1), (180, 200, 3))
        revised = add_outer_transverse_samples(original, region, [-6, 6], 2 / 1.5)
        changes = [row for row in revised if row not in original]
        self.assertEqual(len(original), 50)
        self.assertEqual(len(changes), 4)
        self.assertEqual([row for row in revised if row in original], original)
        for phase, pose in changes:
            self.assertEqual(phase, "transverse")
            self.assertEqual(pose[0], 150.)
            self.assertLessEqual(abs(pose[1] - region.l4_z), 6.)
        self.assertEqual(add_outer_transverse_samples(original, region, [-6, 6], 2.0), original)

    def test_sap_levels_use_projected_location_and_keep_uncertainty(self):
        region = Stage1Region(133, 141, 123, (130.5, 133.5))
        frames = [
            {
                "image_valid": True, "phase": "transverse",
                "executed_relative_pose_index": [x, z, 2.95],
                "actor_base_to_ee_position_m": [-.0015 * z, .0015 * x, -.133],
                "actor_base_to_ee_rotation_columns": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
            for x in (137, 149) for z in (120, 135, 146)
        ]
        fit = transverse_cranial_coordinate(frames, 1.5, 1.5)
        self.assertIsNotNone(fit)
        np.testing.assert_allclose(fit.direction, [-1., 0., 0.], atol=1e-10)
        self.assertEqual(sap_level_at((-201., 500., 375.), region, fit)[1], ("L5",))
        self.assertEqual(sap_level_at((-211.5, 500., 375.), region, fit)[1], ("L4", "L5"))
        self.assertEqual(sap_level_at((-219., 500., 375.), region, fit)[1], ("L4",))
        self.assertEqual(sap_level_at((-172.5, 500., 375.), region, fit)[1], ())
        self.assertIsNone(transverse_cranial_coordinate(frames[:3], 1.5, 1.5))
        self.assertEqual(sap_level_at((-201., 500., 375.), region, None)[1], ("L4", "L5"))

    def test_surface_representative_is_an_observed_pixel(self):
        frame = {
            "actor_base_to_ee_position_m": [0.01, 0.02, 0.03],
            "actor_base_to_ee_rotation_columns": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            "frame_sequence": 0, "frame_sha256": "sha", "acquisition_id": "frame",
        }
        classes = np.asarray([[1, 1, 1], [1, 0, 1], [0, 0, 0]])
        grid = project_image(frame, classes.shape, 0.5)
        np.testing.assert_allclose(grid[2, 0], [10, 19.5, 164])
        rows, mask = extract_observations(classes, None, {}, frame, grid)
        self.assertEqual(len(rows), 1)
        self.assertEqual(mask[rows[0].row, rows[0].column], 1)
        np.testing.assert_allclose(rows[0].coordinate_mm, grid[rows[0].row, rows[0].column])


if __name__ == "__main__":
    unittest.main()
