import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from workflows.mbb_observed_surface.audit import distance_to_faces
from workflows.mbb_shape_completion.fit import fit_height, grid, mask_disk
from workflows.mbb_shape_completion.predict import disk_request, read
from workflows.mbb_shape_completion.prepare import subset
from workflows.mbb_shape_completion.score import Cortex, check_seal


PROTOCOL = read(Path(__file__).resolve().parents[1] /
                "experiments/unity_two_stage_transfer/s0010_mbb_shape_completion_v1.json")


class ShapeCompletionTests(unittest.TestCase):
    def test_disk_removes_all_depths_classes_and_views(self):
        xyz = np.array([[0., 0., 10.], [0., 0., -8.], [1., 0., 0.], [2., 0., 10.]])
        data = {"xyz_mm": xyz, "basis": np.eye(3), "probe_mm": np.ones((2, 3)),
                "identity": np.array(["RIGHT_L5_TP", "OTHER_BONE", "RIGHT_L5_SAP", "RIGHT_L5_TP"]),
                "level_hint": np.array(["L5", "L5", "L4", "L5"])}
        request = {"center_uv_mm": [0., 0.], "gap_mm": 2., "context_radius_mm": 10.,
                   "identity": "RIGHT_L5_TP"}
        with self.assertRaisesRegex(ValueError, "leaked"):
            disk_request(data, request, PROTOCOL["fit"])
        hidden = mask_disk(xyz[:, :2], request["center_uv_mm"], request["gap_mm"] / 2)
        inp = subset(data, ~hidden)
        np.testing.assert_array_equal(inp["xyz_mm"], xyz[-1:])
        np.testing.assert_array_equal(inp["probe_mm"], data["probe_mm"])
        detail, _, _, _, _, _ = disk_request(inp, request, PROTOCOL["fit"])
        self.assertEqual(detail["eligible_points"], 1)

    def test_no_extrapolation_or_support_from_repeated_view(self):
        uv = grid([(-4, 4), (-4, 4)], 1)
        points = np.column_stack((uv, .2 * uv[:, 0]))
        kept = ~mask_disk(uv, [0, 0], 1.)
        query = np.array([[0., 0.], [20., 0.]])
        for method in ("linear", "regularized"):
            groups = np.arange(len(points))[kept] % 4
            result = fit_height(points[kept], groups, query, PROTOCOL["fit"], method)
            self.assertAlmostEqual(result.height[0], 0.)
            self.assertTrue(np.isnan(result.height[1]))
            result = fit_height(points[kept], groups * 0, query[:1], PROTOCOL["fit"], method)
            self.assertEqual(result.detail["reason"], "INSUFFICIENT_DISTINCT_VIEWS")

    def test_competing_height_layers_are_rejected_before_fitting(self):
        aspect = grid([(-4, 4), (-4, 4)], 1)
        flat = np.column_stack((aspect, np.zeros(len(aspect))))
        points = np.concatenate((flat, flat + [0, 0, 5.]))
        groups = np.arange(len(points)) % 4
        outcome = fit_height(points, groups, np.array([[0., 0.]]), PROTOCOL["fit"], "regularized")
        self.assertEqual(outcome.detail["reason"], "COMPETING_HEIGHT_LAYERS")
        self.assertTrue(np.isnan(outcome.height).all())

    def test_exact_ct_face_distance_in_transformed_anisotropic_voxels(self):
        faces = np.array([[1., .5, 2.], [0., 0., .5], [.5, 1., 0.]])
        axes = np.array([1, 2, 0])
        affine = np.diag([.75, 1.25, 1.5, 1.])
        basis = (np.diag([.75, 1.25, 1.5]) @ np.array([[0, 1, 0], [-1., 0, 0], [0, 0, 1.]]))
        with patch("workflows.mbb_shape_completion.score.ras_to_base",
                   return_value=np.vstack((np.zeros(3), basis)) + [10., -20., 5.]):
            cortex = Cortex(faces, axes, affine, {})
        xyz = np.random.default_rng(15).normal(size=(60, 3))
        np.testing.assert_allclose(cortex.distance(cortex.base(xyz)),
                                   distance_to_faces(xyz, faces, axes, np.diag(affine)[:3]), atol=1e-12)

    def test_mismatched_prediction_seals_stop_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "seals.json").write_text('{"abc":"x"}')
            (folder / "inputs").mkdir()
            (folder / "inputs" / "requests.json").write_text('{"def":{}}')
            with self.assertRaisesRegex(ValueError, "differ"):
                check_seal(folder, folder)


if __name__ == "__main__":
    unittest.main()
