"""Check local evaluator denominators and protected annotation perturbations."""

import unittest

import numpy as np
from scipy.ndimage import binary_dilation

from workflows.mbb_target_identifiability.geometry import (
    coverage, feature_distance, first_bone_voxels, hidden_sap_variants, transform,
)
from workflows.reconstruction_s0010.dense import cloud_from_arrays


class IdentifiabilityTests(unittest.TestCase):
    def test_asymmetric_coverage_and_empty_support(self):
        a = np.array([[0, 0, 0], [4, 0, 0]])
        b = np.array([[0, 0, 0]])
        self.assertEqual(coverage(a, b, 1.5)["covered"], 1)
        result = coverage(a, b[:0], 1.5)
        self.assertEqual((result["covered"], result["total"]), (0, 2))
        self.assertIsNone(result["median_nearest_mm"])

    def test_distances_abstain_when_one_surface_is_absent(self):
        cloud = cloud_from_arrays([[0, 0, 0], [1, 0, 0]], [2, 3], [0, 1], [.8, .7])
        self.assertIsNone(feature_distance(cloud, np.array([True, False]))["closest_visible_pair_mm"])
        self.assertEqual(feature_distance(cloud, np.ones(2,dtype=bool))["closest_visible_pair_mm"], 1.)

    def test_protected_voxels_follow_width_major_first_hit(self):
        raw = np.array([[0, 27, 0], [27, 27, 0]])
        directory = np.array([[0,0,0], [1,0,0], [2,0,0], [3,0,0], [4,0,0], [5,0,0]])
        np.testing.assert_array_equal(first_bone_voxels(directory, raw, {27}, (8,8,8)),
                                      [[2,0,0], [1,0,0]])

    def test_perturbations_preserve_original_first_hit_and_shell(self):
        sap = np.zeros((16,12,12), dtype=bool)
        tp = sap.copy()
        sap[2:6,4:8,4:8] = True
        tp[10:14,4:8,4:8] = True
        protected = np.zeros_like(sap)
        protected[2,4,4] = True
        protected[6,4,4] = True
        original_shell = binary_dilation(sap | tp) & ~(sap | tp)
        base, variants = hidden_sap_variants(sap,tp,np.ones_like(sap), protected, np.eye(4))
        self.assertGreater(len(variants), 1)
        self.assertTrue(any(abs(v.minimum_surface_separation_mm-base.minimum_surface_separation_mm)>0
                            for _, _, v in variants))
        for _, alt, _ in variants:
            altered_shell = binary_dilation(alt | tp) & ~(alt | tp)
            np.testing.assert_array_equal(alt[protected], sap[protected])
            np.testing.assert_array_equal(altered_shell[protected], original_shell[protected])

    def test_native_indices_map_to_public_base_millimeters(self):
        f = {"actor_base_to_ee_rotation_columns": np.eye(3),
             "actor_base_to_ee_position_m": [1.,0.,0.]}
        r = {"human_to_ee_rotation_columns": np.eye(3),
             "human_to_ee_position_m": [0.,0.,0.]}
        np.testing.assert_array_equal(transform(f,r,1.5)([[2,0,0]]), [[1003,0,0]])


if __name__ == "__main__":
    unittest.main()
