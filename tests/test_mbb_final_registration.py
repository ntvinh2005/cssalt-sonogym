import unittest

import numpy as np

from workflows.mbb_final_registration.evaluate import local_ct_mesh, patch_metrics
from workflows.mbb_final_registration.register import (
    OTHER, SAP, TP, Observations, anchor, balanced_weights, deduplicate, external_proxies,
    posterior_disagreement, robust_enough,
)
from workflows.mbb_ssm.geometry import Mesh


class FinalRegistrationTests(unittest.TestCase):
    def test_voxel_sampling_keeps_distinct_views(self):
        xyz = np.array([[.1, 0., 0.], [.3, 0., 0.], [.1, 0., 0.], [2., 0., 0.]])
        sampled = deduplicate(Observations(xyz, np.full(4, TP), np.array([0, 0, 1, 1]),
                                          np.array([.1, .5, .4, .4])), 1.)
        np.testing.assert_array_equal(sampled.xyz, xyz[[1, 2, 3]])
        np.testing.assert_array_equal(sampled.group, [0, 1, 1])

    def test_class_and_view_normalization_caps_other_bone(self):
        labels = np.asarray([TP] * 4 + [SAP] * 2 + [OTHER] * 90)
        groups = np.r_[np.array([0, 0, 0, 1, 2, 4]), np.tile([0, 1, 2], 30)]
        values = balanced_weights(Observations(np.zeros((len(labels), 3)), labels, groups,
                                               np.ones(len(labels))), .3)
        np.testing.assert_allclose([values[labels == a].sum() for a in (TP, SAP, OTHER)], [.35, .35, .3])
        self.assertAlmostEqual(values[:3].sum(), values[3])

    def test_competing_anchors_do_not_pick_arbitrary_location(self):
        xyz = np.c_[np.arange(8), np.zeros((8, 2))]
        cloud = np.vstack((xyz, xyz + [0., 15., 0.]))
        point = Observations(cloud, np.full(16, TP), np.tile([0, 1], 8), np.ones(16))
        fit, cause = anchor(point, {"anchor_min_points": 8, "anchor_min_views": 2,
                                    "anchor_connectivity_mm": 4.})
        self.assertIsNone(fit)
        self.assertEqual(cause, "COMPETING_ANCHOR_CLUSTERS")

    def test_local_bidirectional_error_keeps_reverse_direction(self):
        ref = Mesh(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]), np.array([[0, 1, 2]]))
        beyond = Mesh(np.array([[0., 0., 0.], [5., 0., 0.], [0., 5., 0.]]), np.array([[0, 1, 2]]))
        result = patch_metrics(ref, beyond, ref.points, [1., 1.5, 2.])
        self.assertAlmostEqual(result["reference_to_inferred"]["p95_mm"], 0.)
        self.assertAlmostEqual(result["bidirectional_p95_mm"], 4.)
        self.assertIsNone(patch_metrics(ref, Mesh(beyond.points, np.empty((0, 3), dtype=int)),
                                        ref.points, [1.]))

    def test_local_ct_face_indices_preserve_geometry(self):
        class ReferenceFrame:
            def base(self, xyz):
                return xyz * 1.5 + [2., 3., 4.]
        mesh = local_ct_mesh(ReferenceFrame(), np.array([[0., .5, 0.], [4., 0., .5]]),
                             np.array([1, 2]), np.array([True, False]))
        self.assertEqual(mesh.faces.shape, (2, 3))
        np.testing.assert_allclose(mesh.points[:, 1], 3.75)
        np.testing.assert_allclose(np.ptp(mesh.points, axis=0), [1.5, 0., 1.5])

    def test_single_anchor_cannot_pass_and_unknown_disagreement_stays_none(self):
        meta = {"observed_p95_mm": 1., "class_observed_p95_mm": {TP: .4}}
        self.assertFalse(robust_enough(meta, {"max_observed_p95_mm": 5., "max_class_observed_p95_mm": 5.}))
        p = np.zeros((4, 3))
        self.assertIsNone(posterior_disagreement(p, p, np.zeros(4, dtype=bool)))

    def test_synthetic_semantics_depend_on_observed_coordinates_only(self):
        rng = np.random.default_rng(15)
        xyz = rng.normal(size=(1000, 3))
        labels = external_proxies(xyz)
        np.testing.assert_array_equal(labels, external_proxies(xyz + [7., -25., 10.]))
        self.assertEqual(set(labels), {TP, SAP, OTHER})


if __name__ == "__main__":
    unittest.main()
