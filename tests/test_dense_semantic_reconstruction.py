from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from workflows.reconstruction_s0010.dense import (
    cloud_from_arrays, distinct_pose_matrix, evaluate, most_proximal_pixels, raw_cloud,
    voxel_fuse, write_ply,
)
from two_stage_navigation.simple_landmark_mapping import Stage1Region, project_image


def frame(sequence, dx=0., digest=None):
    return {
        "actor_base_to_ee_position_m": [dx, 0, 0],
        "actor_base_to_ee_rotation_columns": np.eye(3).tolist(),
        "frame_sequence": sequence, "frame_sha256": str(sequence) if digest is None else digest,
        "acquisition_id": f"frame{sequence}", "level_hypotheses": ["L4"],
        "tp_eligible_for_mapping": False,
    }


class DenseReconstructionTests(unittest.TestCase):
    def test_proximal_pixels_keep_lateral_geometry(self):
        pixels = np.asarray([[2, 2], [1, 2], [6, 1], [3, 4], [5, 4]])
        np.testing.assert_array_equal(most_proximal_pixels(pixels), [[6, 1], [1, 2], [3, 4]])

    def test_learned_cloud_admits_same_component_and_projects_pixel_anchors(self):
        f = frame(0)
        coord = project_image(f, (7, 6), .5)
        public = {"valid": [0], "frames": [f], "coords": {0: coord},
                  "fit": None, "region": Stage1Region(133, 141, 123, (130, 133))}
        p = np.zeros((1, 4, 7, 6), dtype=float)
        p[:, 0] = 1.
        p[0, 0, 2:4, 2:5] = 0.
        p[0, 1, 2:4, 2:5] = .95
        frozen = {key: {"threshold": .9, "minimum_pixels": 2}
                  for key in ("TP", "SAP", "OTHER_BONE")}
        cloud, points, admission, _ = raw_cloud(
            public, probabilities={"L4": p, "L5": p}, thresholds={"L4": frozen, "L5": frozen},
        )
        self.assertEqual(admission["RIGHT_L4_TP"]["eligible_components"], 1)
        self.assertEqual(admission["RIGHT_L4_TP"]["eligible_pixels"], 6)
        self.assertEqual(admission["RIGHT_L5_TP"]["excluded_components"], 1)
        np.testing.assert_array_equal(points, [[2, 2], [2, 3], [2, 4]])
        np.testing.assert_allclose(cloud.xyz_mm, np.asarray([[-.5, 0., 134.], [0., 0., 134.], [.5, 0., 134.]]))
        oracle = np.zeros((1, 7, 6), dtype=np.uint8)
        oracle[0, 2:4, 2:5] = 1
        projection, _, _, _ = raw_cloud(public, oracle_labels={"L4": oracle, "L5": oracle})
        self.assertEqual(len(projection.code), 6)

    def test_fusion_balance_distinct_views_and_keeps_conflict(self):
        frames = [frame(0), frame(1, .0001), frame(2, .004), frame(3, .006, digest="2")]
        config = {"minimum_view_translation_mm": 2., "minimum_view_rotation_deg": 1.}
        distinct = distinct_pose_matrix(frames, config)
        self.assertFalse(distinct[0, 1])
        self.assertTrue(distinct[0, 2])
        self.assertFalse(distinct[2, 3])
        cloud = cloud_from_arrays(
            [[.1, 0, 0], [.2, 0, 0], [.7, 0, 0], [1.3, 0, 0], [.2, 0, 0]],
            [0, 0, 0, 0, 1], [0, 0, 1, 2, 3], [.95, .95, .8, .9, .91],
        )
        fused, meta = voxel_fuse(cloud, distinct, 1.5)
        np.testing.assert_allclose(fused.xyz_mm[:, 0], [.725, .2])
        np.testing.assert_array_equal(meta["unique_viewpoints"], [2, 1])
        np.testing.assert_array_equal(meta["conflicting_semantics"], [True, True])
        np.testing.assert_array_equal(meta["selected_frame_sequences"], [0, 2, 3])
        np.testing.assert_array_equal(meta["raw_offsets"], [0, 4, 5])
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "cloud.ply"
            write_ply(output, fused, meta["unique_viewpoints"], meta["conflicting_semantics"])
            self.assertIn("element vertex 2", output.read_text())

    def test_same_frame_semantics_does_not_reward_unobserved_geometry(self):
        p0 = cloud_from_arrays([[0, 0, 0], [0, 0, 0], [9, 0, 0]], [0, 1, 3], [0, 1, 1], [1, 1, 1])
        p1 = cloud_from_arrays([[0, 0, 0], [0, 0, 0], [9, 0, 0]], [0, 0, 3], [0, 1, 0], [1, 1, 1])
        ct = {key: p0.xyz_mm for key in ("RIGHT_L4_TP", "RIGHT_L4_SAP", "RIGHT_L5_TP", "RIGHT_L5_SAP")}
        metrics, _ = evaluate(p1, p0, p0, ct, 1.5, ~np.eye(2, dtype=bool), None)
        self.assertEqual(metrics["RIGHT_L4_TP"]["semantic_correctness"],
                         {"correct": 1, "wrong_class": 1, "outside_gate_or_unmatched": 0, "total": 2})
        self.assertEqual(metrics["RIGHT_L5_SAP"]["semantic_correctness"]["outside_gate_or_unmatched"], 1)
        self.assertEqual(metrics["RIGHT_L4_SAP"]["eligible_oracle_to_output"]["within_mm"], 0)


if __name__ == "__main__":
    unittest.main()
