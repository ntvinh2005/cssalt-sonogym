import unittest

import numpy as np

from two_stage_navigation.incremental_landmark_map import IncrementalLandmarkMapConfig
from two_stage_navigation.sonoanatomy_multiview import SemanticObservation
from two_stage_navigation.surface_landmark_mapping import surface_clusters


def observation(index, x_mm):
    return SemanticObservation(
        semantic_class="SAP", frame_sequence=index, frame_sha256=str(index), viewpoint_id=str(index),
        viewpoint_position_m=(0., index * .01, 0.), viewpoint_rotation_columns=tuple(map(tuple, np.eye(3))),
        row=0, column=0, coordinate_mm=(x_mm, 0., 0.), confidence=.9, pixel_count=1,
    )


class SurfaceAssociationTests(unittest.TestCase):
    def test_complete_linkage_does_not_chain(self):
        rows = [observation(i, x) for i, x in enumerate((0., 3., 6.))]
        clouds = [np.array([row.coordinate_mm]) for row in rows]
        clusters = surface_clusters(rows, clouds, IncrementalLandmarkMapConfig(), .25)
        self.assertEqual(sorted(c["members"] for c in clusters), [1, 2])

    def test_surface_overlap_can_join_distant_representatives(self):
        rows = [observation(0, 0.), observation(1, 6.)]
        clouds = [np.array([[x, 0, 0] for x in [0, 1, 2, 3]]),
                  np.array([[x, 0, 0] for x in [3, 4, 5, 6]])]
        cluster, = surface_clusters(rows, clouds, IncrementalLandmarkMapConfig(), .25)
        self.assertEqual(cluster["distinct_viewpoints"], 2)
