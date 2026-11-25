"""Exercise actor-visible geometry without CT assets or simulator state."""

from dataclasses import replace
import unittest

import numpy as np

from workflows.mbb_end_to_end.localize import (
    Cloud, OTHER, RULES, SAP, TP, cloud_integrity, find_region, localize,
)
from two_stage_navigation.simple_landmark_mapping import project_image


RIGHT, SUPERIOR = np.eye(3)[[0, 2]]


def case(*, connector=True, duplicate=False, parallel=False):
    horizontal = np.array([[x, y, 0.] for x in (1.5, 2., 2.5, 3., 3.5)
                           for y in (-1., -.5, 0., .5, 1.)])
    vertical = np.array([[0., y, z] for z in (1.5, 2., 2.5, 3., 3.5)
                         for y in (-1., -.5, 0., .5, 1.)])
    if parallel:
        vertical = vertical[:, [2, 1, 0]] + [0., 0., 1.5]
    pieces = [(TP, horizontal), (SAP, vertical)]
    if connector:
        lower = [[x, y, 0.] for x in (0., .5, 1.) for y in (-1., -.5, 0., .5, 1.)]
        upper = [[0., y, z] for z in (.5, 1.) for y in (-1., -.5, 0., .5, 1.)]
        pieces.append((OTHER, np.array(lower + upper)))

    xyz = np.concatenate([p for _, p in pieces])
    labels = [k for k, p in pieces for _ in p]
    if duplicate:
        xyz = np.concatenate((xyz, xyz + [0., 8., 0.]))
        labels = labels * 2
    n = len(xyz)
    copies = 3
    roles = np.asarray(labels * copies)
    probabilities = np.zeros((len(roles), 4))
    for code, role in enumerate((TP, SAP, OTHER), start=1):
        probabilities[roles == role, code] = 1
    return Cloud(
        np.tile(xyz, (copies, 1)), roles, np.repeat("L5", n * copies),
        np.repeat(np.arange(copies), n), np.zeros((n * copies, 2), dtype=int),
        np.ones(n * copies), probabilities, np.arange(n * copies),
    )


def actor_frames():
    # Translation deltas in the two public command axes are physically distinct.
    frames = []
    for f, (x, z) in enumerate(((0., 0.), (3., 0.), (0., 3.))):
        frames.append({
            "frame_sequence": f, "actor_base_to_ee_position_m": [(10 + x) / 1000, .015, (150 + z) / 1000],
            "actor_base_to_ee_rotation_columns": np.eye(3).tolist(),
            "executed_relative_pose_index": [x / 1.5, z / 1.5, 2.95],
            "frame_sha256": str(f), "phase": "transverse",
            "image_valid": True, "producer_advanced": True, "transition_valid": True, "contact_ok": True,
        })
    return frames


class ObservedGeometryTests(unittest.TestCase):
    def solve(self, cloud, independent=True):
        distinct = (np.ones((3, 3), dtype=bool) ^ np.eye(3, dtype=bool)) if independent else np.zeros((3, 3), bool)
        return find_region(cloud, distinct, np.tile([8., 15., 10.], (3, 1)), (RIGHT, SUPERIOR), 2)

    def test_connected_untyped_surface_supports_a_bounded_region(self):
        cloud = case()
        result, why, diag = self.solve(cloud)
        self.assertIsNone(why)
        self.assertEqual(diag["competing_region_count"], 1)
        self.assertGreater(diag["nearest_tp_sap_mm"], 1.5)
        self.assertGreater(result["diameter_mm"], 0.)
        self.assertLessEqual(result["diameter_mm"], RULES["max_region_diameter_mm"])
        points_in_cloud = {tuple(row) for row in cloud.xyz}
        self.assertTrue(all(tuple(row) in points_in_cloud for row in result["observed_region_mm"]))
        via = result["support"]["observed_cortical_route"]["cloud_rows"]
        self.assertIn(OTHER, cloud.role[via])
        self.assertEqual(result["support"]["SAP"]["distinct_viewpoint_count"], 3)
        endpoint = np.asarray(result["observed_region_mm"])
        for idx in result["inferred_region_mm"]:
            self.assertLessEqual(np.linalg.norm(endpoint - idx, axis=1).min(), .75)

    def test_missing_connector_cannot_bridge_unobserved_gap(self):
        result, why, _ = self.solve(case(connector=False))
        self.assertIsNone(result)
        self.assertEqual(why, "no_supported_cortical_transition")

    def test_two_parallel_interfaces_are_not_a_transition(self):
        result, why, _ = self.solve(case(connector=False, parallel=True))
        self.assertIsNone(result)
        self.assertEqual(why, "no_supported_cortical_transition")

    def test_l4_points_cannot_bridge_l5_surfaces(self):
        cloud = case()
        cloud = replace(cloud, level=np.where(cloud.role == OTHER, "L4", "L5"))
        result, why, _ = self.solve(cloud)
        self.assertIsNone(result)
        self.assertEqual(why, "no_supported_cortical_transition")

    def test_competing_regions_and_repeated_viewpoints_abstain(self):
        result, why, diag = self.solve(case(duplicate=True))
        self.assertIsNone(result)
        self.assertEqual(why, "competing_cortical_transitions")
        self.assertEqual(diag["competing_region_count"], 2)
        result, _, _ = self.solve(case(), independent=False)
        self.assertIsNone(result)

    def test_ambiguous_adjacent_level_support_can_block_accepted_region(self):
        cloud = case()
        mask = cloud.role == OTHER
        attributes = [np.concatenate((values, values[mask])) for values in
                      (cloud.xyz, cloud.role, cloud.level, cloud.frame, cloud.pixel, cloud.confidence,
                       cloud.probabilities, cloud.row)]
        attributes[2][len(cloud.row):] = "L4"
        attributes[-1] = np.arange(len(attributes[-1]))
        result, why, diag = self.solve(Cloud(*attributes))
        self.assertIsNone(result)
        self.assertEqual(why, "competing_level_near_transition")
        self.assertGreater(diag["ambiguous_level_seed_count"], 0)

    def test_actor_callable_returns_repeatable_view_removals(self):
        result = localize(case(), actor_frames())
        self.assertEqual(result["status"], "GEOMETRIC_CANDIDATE")
        self.assertEqual(result["diagnostics"]["omit_view_0"]["region_distance_mm"], 0.)
        self.assertIn("pose_perturbation_+1", result["diagnostics"])

    def test_public_calibration_corruption_is_rejected(self):
        frames = actor_frames()
        xyz = project_image(frames[0], tuple(RULES["image_shape"]), RULES["pixel_mm"])[[5], [7]]
        cloud = Cloud(xyz, np.array([TP]), np.array(["L5"]), np.array([0]), np.array([[5, 7]]),
                      np.ones(1), np.array([[0., 1., 0., 0.]]), np.array([0]))
        self.assertEqual(cloud_integrity(cloud, frames)["public_reprojection_max_mm"], 0.)
        with self.assertRaisesRegex(ValueError, "public reprojection mismatch"):
            cloud_integrity(replace(cloud, xyz=cloud.xyz + .1), frames)


if __name__ == "__main__":
    unittest.main()
