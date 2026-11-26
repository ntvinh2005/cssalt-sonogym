import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from workflows.mbb_ssm.benchmark import keep_mask
from workflows.mbb_ssm.fit import Model, fit_arrays, read, weights
from workflows.mbb_ssm.geometry import Mesh, face_vectors, nearest_points, rigid
from workflows.mbb_ssm.prepare import read_prior_splits
from workflows.mbb_ssm.train import guarded_update, pca


PROTOCOL = read(Path(__file__).resolve().parents[1] /
                "experiments/unity_two_stage_transfer/mbb_ssm_v1.json")


class StatisticalShapeTests(unittest.TestCase):
    def test_pca_reconstructs_size_variation_in_mm(self):
        base = np.array([[2., 0., 0.], [-2., 0., 0.], [0., 1., 0.], [0., 0., 2.]])
        mean, modes, eigen = pca([base * scale for scale in (.8, 1., 1.2)])
        direction = base / np.linalg.norm(base)
        self.assertAlmostEqual(abs(np.sum(modes[0] * direction)), 1.)
        np.testing.assert_allclose(mean, base)
        self.assertAlmostEqual(np.sqrt(eigen[0]), .2 * np.linalg.norm(base))
        model = Model(mean, np.array([[0, 1, 2]]), modes * np.sqrt(eigen)[:, None, None])
        np.testing.assert_allclose(model.shape([0.]), base)
        np.testing.assert_allclose((model.shape([1.]) + model.shape([-1.])) / 2, base)

    def test_proper_rigid_transform_preserves_size(self):
        xyz = np.array([[0., 1., 0.], [0., 0., 2.], [1., 0., 0.], [-1., 0., 0.]])
        basis = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        target = xyz @ basis + [10., -7., 2.]
        rotation, offset = rigid(xyz, target)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.)
        np.testing.assert_allclose(xyz @ rotation + offset, target, atol=1e-12)

    def test_distance_projects_to_triangle_interior(self):
        mesh = Mesh(np.array([[0., 0., 0.], [2., 0., 0.], [0., 4., 0.]]), np.array([[0, 1, 2]]))
        p, _ = nearest_points(np.array([[.2, .7, 1.], [-1., .7, .4]]), mesh)
        np.testing.assert_allclose(p, [[.2, .7, 0.], [0., .7, 0.]])

    def test_disk_excludes_every_depth_and_keeps_split_members(self):
        xyz = np.array([[0., 0., .5], [0., 8., .5], [1.5, 2., 0.]])
        keep = keep_mask(xyz, {"gap": 2., "center_uv_mm": [0, 0]})
        np.testing.assert_array_equal(keep, [False, False, True])
        with tempfile.TemporaryDirectory() as folder:
            initial = Path(folder) / "initial.json"
            initial.write_text(json.dumps({"split": {"train": ["a"], "development": ["b"], "held_out": ["c"]}}))
            results = read_prior_splits(initial, list("abcdefgh"), 2)
        self.assertIn("a", results["train"])
        self.assertIn("b", results["development"])
        self.assertIn("c", results["held_out"])
        self.assertEqual(sorted(sum(results.values(), [])), list("abcdefgh"))

    def test_fold_guard_backtracks_an_inverted_triangle(self):
        ref = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                        [7., 0., 0.], [8., 0., 0.], [7., 1., 0.]])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        nominal = face_vectors(Mesh(ref, faces))
        moved = ref.copy()
        moved[2, 1] = -2.
        moved[3:, 2] = 1.5
        fit = guarded_update(ref, moved, faces, nominal, .02)
        after = face_vectors(Mesh(fit, faces))
        np.testing.assert_array_less(.019 * np.sum(nominal * nominal, axis=1),
                                     np.sum(nominal * after, axis=1))
        np.testing.assert_array_equal(fit[3:], moved[3:])
        self.assertFalse(np.array_equal(fit[:3], ref[:3]))

    def test_dense_view_cannot_outvote_others_and_repeated_pose_abstains(self):
        groups = np.array([0, 0, 0, 1])
        w = weights(groups, np.ones(len(groups)))
        self.assertAlmostEqual(w[groups == 0].sum(), w[groups == 1].sum())
        data = np.zeros((45, 3))
        unused = Model(data[:3], np.array([[0, 1, 2]]), np.empty((0, 3, 3)))
        points, _, status = fit_arrays(data, np.eye(3), np.repeat("OTHER_BONE", len(data)),
                                       np.zeros(len(data), dtype=int), np.ones(len(data)),
                                       unused, 0., PROTOCOL["fit"])
        self.assertIsNone(points)
        self.assertEqual(status["status"], "INSUFFICIENT_OBSERVATIONS")


if __name__ == "__main__":
    unittest.main()
