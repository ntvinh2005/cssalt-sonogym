import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from workflows.mbb_direct_sonographic.localize import (
    COMPETING, SAP, TP, audit_frame, observed_profiles, pair_regions,
    region_distance, select_views, verify_session,
)
from workflows.mbb_direct_sonographic.public_geometry import sweep_axes
from workflows.mbb_direct_sonographic.replay import compare_sessions, load_annotations


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "experiments/unity_two_stage_transfer/mbb_direct_sonographic_v2.json"
)
RULES = json.loads(CONFIG.read_text())


def straight_to_rising_profile():
    rows = np.array([120] * 8 + [119, 117, 115, 113] + [112] * 8)
    points = np.column_stack((rows, np.arange(len(rows)) + 50))
    named = {
        TP: np.isin(np.arange(len(rows)), np.arange(2, 8)),
        SAP: np.isin(np.arange(len(rows)), np.arange(12, 18)),
        **{name: np.zeros(len(rows), dtype=bool) for name in COMPETING},
    }
    return points, named


def pose(frame_id, x_mm=0., image_hash=None):
    return {
        "frame_sequence": frame_id,
        "frame_sha256": image_hash or f"hash{frame_id}",
        "actor_base_to_ee_position_m": [x_mm / 1000, 0., 0.],
        "actor_base_to_ee_rotation_columns": np.eye(3).tolist(),
    }


def known_candidate(family="transverse"):
    return {
        "view": family,
        "direct_candidate": family == "transverse",
        "candidate_pixels": [[100, 72], [100, 73], [99, 74]],
        "support_pixels": {TP: [[100, 72]], SAP: [[99, 74]]},
        "status": "DIRECT_IMAGE_CANDIDATE" if family == "transverse" else "LONGITUDINAL_CONTEXT",
        "tp_columns": 4, "sap_columns": 4, "tp_confidence": .95,
        "sap_confidence": .95, "maximum_supported_run": 12,
    }


class DirectSonographicTest(unittest.TestCase):
    def test_signal_gate_rejects_named_but_flat_bmode(self):
        image = np.zeros(RULES["image_shape"], dtype=np.float32)
        profiles = observed_profiles(image, RULES)
        points, named = straight_to_rising_profile()
        evidence = {name: (points[mask], np.ones(mask.sum())) for name, mask in named.items()}
        result = audit_frame(profiles, evidence, "transverse", -1, RULES)
        self.assertFalse(result["direct_candidate"])
        self.assertEqual(result["tp_columns"], 0)

    def test_a_supported_turn_returns_only_measured_pixels(self):
        points, named = straight_to_rising_profile()
        supported = np.ones(len(points), dtype=bool)
        candidates, checks, _ = pair_regions(points, supported, named, -1, RULES)
        self.assertEqual(checks["continuous_pairs"], 1)
        self.assertTrue(candidates)
        self.assertLessEqual(len(candidates[0]["candidate"]), 3)
        self.assertTrue(all(tuple(p) in set(map(tuple, points)) for p in candidates[0]["candidate"]))
        reversed_candidates, _, _ = pair_regions(points, supported, named, 1, RULES)
        self.assertEqual(reversed_candidates, [])

    def test_missing_or_competing_cortex_abstains(self):
        points, named = straight_to_rising_profile()
        for competing in (False, True):
            with self.subTest(competing=competing):
                supported = np.ones(len(points), dtype=bool)
                if competing:
                    named[COMPETING[0]][10] = True
                else:
                    supported[10] = False
                candidates, checks, _ = pair_regions(points, supported, named, -1, RULES)
                self.assertEqual(checks["ordered_pairs"], 1)
                self.assertEqual(checks["continuous_pairs"], 0)
                self.assertFalse(candidates)

    def test_projection_and_consistency_are_in_robot_base_mm(self):
        frames = [pose(0), pose(1), pose(2, 2)]
        rows = [known_candidate("longitudinal"), known_candidate(), known_candidate()]
        result = verify_session(frames, rows, [0, 1, 2], RULES)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["support_frames"], [1, 2])
        np.testing.assert_allclose(result["regions"][1]["candidate_xyz_mm"][0], [-1.5, 0., 183.])
        self.assertLessEqual(result["max_disagreement_mm"], 2.0)
        self.assertAlmostEqual(region_distance([[0, 0, 0]], [[1, 0, 0]]), 1.)

    def test_ambiguous_multiview_regions_are_not_silently_filtered(self):
        frames = [pose(0), pose(1), pose(2, 2), pose(3, 12)]
        rows = [known_candidate("longitudinal"), *[known_candidate() for _ in range(3)]]
        result = verify_session(frames, rows, [0, 1, 2, 3], RULES)
        self.assertEqual(result["status"], "AMBIGUOUS_MULTIVIEW_REGIONS")
        self.assertFalse(result["accepted"])
        self.assertFalse(result["region_xyz_mm"])

    def test_shared_hash_does_not_create_support(self):
        frames = [pose(0), pose(1, image_hash="same"), pose(2, 5, image_hash="same")]
        rows = [known_candidate("longitudinal"), known_candidate(), known_candidate()]
        result = verify_session(frames, rows, [0, 1, 2], RULES)
        self.assertEqual(result["status"], "INSUFFICIENT_DISTINCT_TRANSVERSE_VIEWS")
        selected, _ = select_views(frames, rows, RULES)
        self.assertEqual(selected, [0, 1])

    def test_axes_use_actor_commands_and_fk_only(self):
        coordinates = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
        frames = [
            {**pose(i), "phase": "transverse", "image_valid": True,
             "requested_relative_pose_index": [x, y, 0.],
             "actor_base_to_ee_position_m": [x * .0015, y * .0015, 0.]}
            for i, (x, y) in enumerate(coordinates)
        ]
        right, cranial = sweep_axes(frames, RULES)
        np.testing.assert_allclose(right, [1., 0., 0.], atol=1e-12)
        np.testing.assert_allclose(cranial.direction, [0., 1., 0.], atol=1e-12)
        self.assertAlmostEqual(cranial.index_at([1.5, 3., 133.]), 2.)

    def test_visible_input_schema_refuses_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pixels.npz"
            np.savez(path, frame_sequence=[0], pixel_row_col=[[1, 0]], identity=[TP],
                     confidence=[.95], xyz_mm=[[0., 0., 0.]])
            with self.assertRaisesRegex(ValueError, "only visible 2D"):
                load_annotations(path, 1)

    def test_repeatability_table_preserves_shared_prefix(self):
        left, right, _ = RULES["shared_prefix"]
        results = {"A2": {
            left: {"accepted": True, "region_xyz_mm": [[0, 0, 0]]},
            right: {"accepted": True, "region_xyz_mm": [[2, 0, 0]]},
            "unresolved": {"accepted": False},
        }}
        rows = compare_sessions(results, RULES)
        self.assertEqual(rows[0]["comparison"], "shared_prefix")
        self.assertEqual(rows[0]["bidirectional_p95_mm"], 2)
        self.assertIsNone(rows[1]["bidirectional_p95_mm"])


if __name__ == "__main__":
    unittest.main()
