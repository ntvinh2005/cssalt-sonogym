from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from two_stage_navigation.incremental_landmark_map import IncrementalLandmarkMapConfig
from two_stage_navigation.simple_landmark_mapping import (
    Stage1Region, extract_observations, project_image, simple_map, systematic_sweep,
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
