import json
from pathlib import Path
import unittest

import numpy as np

from workflows.mbb_observed_surface.localize import Surface, transition, view_groups
from workflows.mbb_observed_surface.replay import bidirectional_score


RULES = json.loads(Path("workflows/mbb_observed_surface/actor_rules.json").read_text())
RIGHT, SUPERIOR = np.eye(3)[[0, 2]]


def surfaces(copies=1, competing=False, separated=False):
    tp = [[x, y, 0.] for x in (1., 2., 3.) for y in (-1., 0., 1.)]
    sap = [[0., y, z] for z in (0., 1., 2., 3.) for y in (-1., 0., 1.)]

    def make(points):
        xyz = np.asarray(points)
        if competing:
            xyz = np.concatenate((xyz, xyz + [0., 16., 0.]))
        ids = np.repeat(np.arange(copies), len(xyz))
        return Surface(np.tile(xyz, (copies, 1)), ids, np.ones(len(ids)), np.zeros((len(ids), 2), int))

    if separated:
        sap = np.asarray(sap) - [9., 0., 0.]
    return make(tp), make(sap)


class ObservedSurfaceTests(unittest.TestCase):
    def evaluate(self, copies=3, independent=True, **case):
        tp, sap = surfaces(copies, **case)
        groups = np.arange(copies) if independent else np.zeros(copies, dtype=int)
        return transition(tp, sap, RIGHT, SUPERIOR, groups, np.tile([9., 30., 7.], (copies, 1)), RULES, 2)

    def test_region_retains_observed_samples_and_no_extension(self):
        evidence, why = self.evaluate()
        self.assertIsNone(why)
        region = evidence["components"][0]
        self.assertGreater(len(region["observed_region_mm"]), 1)
        self.assertTrue(all(p[0] == 0 for p in region["observed_region_mm"]))
        self.assertEqual(region["inferred_points_mm"], [])
        self.assertEqual(len(region["support"]["SAP"]["view_groups"]), 3)

    def test_separated_surface_cannot_create_an_extrapolated_region(self):
        evidence, why = self.evaluate(separated=True)
        self.assertEqual(why, "no_local_tp_sap_continuity")
        self.assertGreater(evidence["nearest_gap_mm"], RULES["max_gap_mm"])

    def test_competing_transitions_abstain(self):
        evidence, why = self.evaluate(competing=True)
        self.assertEqual(why, "competing_surface_regions")
        self.assertEqual(len(evidence["components"]), 2)

    def test_repeated_frames_do_not_supply_distinct_view_support(self):
        def frame(index, x, image):
            return {"frame_sequence": index, "actor_base_to_ee_position_m": [x, 0, 0],
                    "actor_base_to_ee_rotation_columns": np.eye(3).tolist(), "frame_sha256": image}
        np.testing.assert_array_equal(view_groups([frame(0, 0, "A"), frame(1, 0, "B"),
                                                   frame(2, .003, "C"), frame(3, .006, "C")], RULES), [0, 0, 2, 2])
        evidence, why = self.evaluate(independent=False)
        self.assertEqual(why, "no_supported_root_and_lateral_neck_transition")
        self.assertEqual(evidence["patch_pair_count"], 0)

    def test_reverse_distance_penalizes_a_partial_match(self):
        score = bidirectional_score([[0., 0., 0.]], [[0., 0., 0.], [8., 0., 0.]], 3)
        self.assertEqual(score["prediction_to_reference"]["maximum_mm"], 0.)
        self.assertEqual(score["reference_to_prediction"]["maximum_mm"], 8.)
        self.assertEqual(score["reference_to_prediction"]["within_technical_gate"], 1)


if __name__ == "__main__":
    unittest.main()
