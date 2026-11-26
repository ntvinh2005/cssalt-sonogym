import unittest

import numpy as np

from workflows.mbb_adaptive_cortical.tracker import adaptive_frame, candidates_for, echo_fields, track_path, turn_regions


CONFIG = {"step_cost_per_pixel": .07, "gap_cost_per_column": 1.25,
          "ambiguity_cost_margin": .30, "ambiguity_depth_mm": 2.0}


def run(columns, gap=0):
    return track_path(columns, gap, [True, *([False] * (len(columns) - 2)), True], 4, CONFIG, .5)


class AdaptiveTrackerTest(unittest.TestCase):
    def test_recovers_observed_ridge_without_inserting_points(self):
        path = run([{30: .2}, {29: .2, 40: .01}, {27: .2}, {26: .2}])
        self.assertEqual(path["status"], "JOINED")
        self.assertEqual(path["rows"], [30, 29, 27, 26])

    def test_large_depth_jump_does_not_connect_two_echo_layers(self):
        path = run([{25: .1}, {30: .1}, {31: .1}, {30: .1}])
        self.assertEqual(path["status"], "UNSUPPORTED")
        self.assertEqual(path["rows"], [25, None, None, None])

    def test_small_missing_run_is_explicit_and_cannot_be_a_turn(self):
        columns = [{30: .1}, {30: .1}, {}, {29: .1}, {28: .1}]
        self.assertEqual(run(columns)["status"], "UNSUPPORTED")
        accepted = run(columns, gap=2)
        self.assertEqual(accepted["rows"], [30, 30, None, 29, 28])
        self.assertEqual(turn_regions(accepted, {"start": 2, "stop": 2}, {"turn_window_px": 3}), [])

    def test_repeated_gaps_and_large_gaps_reject(self):
        columns = [{30: .1}, {}, {}, {30: .1}, {}, {}, {}, {30: .1}]
        self.assertEqual(run(columns, gap=4)["status"], "UNSUPPORTED")

    def test_two_echo_layers_of_equal_cost_report_ambiguity(self):
        path = run([{30: .1}, {28: .1, 32: .1}, {27: .1, 33: .1}, {30: .1}])
        self.assertEqual(path["status"], "AMBIGUOUS")
        self.assertIsNotNone(path["alternative_rows"])
        self.assertFalse(adaptive_frame({"direct_candidate": False}, [None], [path], {})["direct_candidate"])

    def test_no_peak_from_flat_image(self):
        terms = {"minimum_relative_echo": .05, "minimum_proximal_contrast": .02,
                 "minimum_distal_contrast": .05}
        self.assertFalse(echo_fields(np.zeros((100, 60)), terms)[-1].any())

    def test_competing_level_is_excluded_from_corridor(self):
        from types import SimpleNamespace

        columns = np.arange(3)
        b = {"pixels": np.column_stack(([7, 7, 7], columns)), "start": 0, "stop": 2}
        depth = np.arange(5, 10)
        data = np.ones((5, 3))
        fields = depth, data, data, data, data.astype(bool)
        sample = np.array([[8, 1]])
        named = {"RIGHT_L4_TP": (sample, np.ones(1)),
                 "RIGHT_L4_SAP": (np.zeros((0, 2)), np.zeros(0))}
        track = SimpleNamespace(columns=columns, rows=np.full(3, 7))
        _, score = candidates_for(b, track, fields, named, {"pixel_mm": .5, "track_label_radius_px": 1},
                                  {"corridor_half_width_mm": 4.})
        self.assertNotIn(8, score[1])
        self.assertIn(5, score[1])


if __name__ == "__main__":
    unittest.main()
