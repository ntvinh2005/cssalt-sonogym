import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from workflows.mbb_end_to_end.evaluate import cortex_diagnostic, score, write_public_prefix


GATE = {"max_p95_mm": 3., "max_diameter_mm": 6.}
PROXY = np.array([[0., 0., 0.], [1., 0., 0.]])


class RegionEvaluationTests(unittest.TestCase):
    def test_partial_region_must_cover_the_reference(self):
        candidate = {"observed_region_mm": [[0., 0., 0.], [1., 0., 0.]]}
        reference = np.array([[0., 0., 0.], [1., 0., 0.], [5., 0., 0.]])
        actual = score(candidate, {"reference": reference}, PROXY, GATE)
        self.assertFalse(actual["anatomical_correct"])
        self.assertEqual(actual["reference_scores"]["reference"]["candidate_to_reference"]["p95_mm"], 0)
        self.assertGreater(actual["reference_scores"]["reference"]["reference_to_candidate"]["p95_mm"], 3)

    def test_diameter_and_legacy_proxy_cannot_fabricate_correctness(self):
        candidate = {"observed_region_mm": [[0., 0., 0.], [7., 0., 0.]]}
        actual = score(candidate, {"reference": np.array(candidate["observed_region_mm"])}, PROXY, GATE)
        self.assertEqual(actual["diameter_mm"], 7.)
        self.assertFalse(actual["anatomical_correct"])
        legacy_only = score(candidate, {}, PROXY, GATE)
        self.assertIsNone(legacy_only["anatomical_correct"])
        self.assertNotIn("passes", legacy_only["historical_diagnostic_only"])

    def test_diagnostic_keeps_uncertain_and_routed_evidence_separate(self):
        cloud = {"xyz_mm": PROXY, "identity": np.array(["OTHER_BONE", "UNCERTAIN_BONE"]),
                 "level_hint": np.array(["L4", "AMBIGUOUS"]), "frame_sequence": np.array([0, 1])}
        trace = cortex_diagnostic(cloud, np.array([[.5, 0., 0.]]), [3], 1.5)
        self.assertEqual(trace["matched_faces_with_other_or_uncertain"], 1)
        self.assertEqual(trace["matched_faces_routed_l5"], 0)

    def test_public_prefix_excludes_all_additional_pose_records(self):
        with tempfile.TemporaryDirectory() as folder:
            source, result = (Path(folder) / name for name in ("all.jsonl", "first.jsonl"))
            source.write_text("".join(json.dumps({"waypoint_index": f}) + "\n" for f in range(126)))
            self.assertEqual(write_public_prefix(source, result), 102)
            self.assertEqual(json.loads(result.read_text().splitlines()[-1])["waypoint_index"], 101)


if __name__ == "__main__":
    unittest.main()
