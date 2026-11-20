import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "workflows/two_stage_navigation/run_s0010_sap_observability.py"
)
SPEC = importlib.util.spec_from_file_location("sap_observability", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load SAP observability workflow")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SapObservabilityTests(unittest.TestCase):
    def test_one_voxel_core_and_uncertainty_are_separate(self) -> None:
        prediction = np.zeros((12, 12, 12), dtype=np.uint8)
        segmentation = np.zeros_like(prediction)
        segmentation[1:11, 1:11, 6:11] = 28
        segmentation[1:11, 1:11, 0:5] = 27
        prediction[6:11, 3:9, 7:11] = 6
        prediction[6:11, 3:9, 1:5] = 6
        masks = MODULE.pseudo_sap_masks(prediction, segmentation, np.eye(4))
        for level in ("L4", "L5"):
            self.assertTrue(np.any(masks[level]["core"]))
            self.assertFalse(
                np.any(masks[level]["core"] & masks[level]["uncertainty"])
            )
            self.assertTrue(
                np.all(masks[level]["core"] <= masks[level]["pseudo"])
            )

    def test_projection_counts_physical_voxels_and_image_pixels_separately(self) -> None:
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[5, 5, 5] = True
        coordinates = np.zeros((200 * 150, 3), dtype=np.float64)
        coordinates[:] = (1.0, 1.0, 1.0)
        coordinates[:9] = (5.2, 5.4, 5.8)
        projected, source = MODULE.project_mask(coordinates, mask)
        self.assertEqual(int(np.count_nonzero(projected)), 9)
        self.assertEqual(len(source), 1)

    def test_finalize_requires_two_sessions_and_aligned_visible_points(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            projection_rows = []
            decisions = []
            for level in ("L4", "L5"):
                for index, session in enumerate(("session_a", "session_b")):
                    review_id = f"{level}_{index}"
                    projection_rows.append(
                        {
                            "review_id": review_id,
                            "session_id": session,
                            "frame_sequence": index,
                            "split": "development",
                            "level": level,
                            "role": "PSEUDO_SAP_TARGET",
                            "acquisition_id": review_id,
                            "producer_sequence": index + 1,
                            "frame_sha256": review_id,
                            "clean_image": "x.png",
                            "privileged_overlay": "y.png",
                            "pseudo_pixel_count": 30,
                            "core_pixel_count": 24,
                            "uncertainty_pixel_count": 10,
                            "pseudo_source_voxel_count": 5,
                            "core_source_voxel_count": 3,
                            "uncertainty_source_voxel_count": 4,
                            "core_margin_pixels": 5,
                            "core_centroid_row": 100.0,
                            "core_centroid_column": 75.0,
                            "projection_gate_passed": True,
                        }
                    )
                    decisions.append(
                        {
                            "review_id": review_id,
                            "decision": "VISIBLE",
                            "interface_points_rc": "101:76",
                            "notes": "synthetic",
                        }
                    )
            for path, rows in (
                (root / "projection_manifest.csv", projection_rows),
                (root / "blind_decisions.csv", decisions),
            ):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            result = MODULE.finalize_review(root, root / "result.json")
            self.assertEqual(result["structures"]["L4"]["verdict"], "VISIBLE")
            self.assertEqual(result["structures"]["L5"]["verdict"], "VISIBLE")

    def test_unaligned_visible_point_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            decisions = []
            for index, session in enumerate(("a", "b")):
                review_id = f"r{index}"
                rows.append(
                    {
                        "review_id": review_id,
                        "session_id": session,
                        "frame_sequence": index,
                        "split": "development",
                        "level": "L4",
                        "role": "PSEUDO_SAP_TARGET",
                        "acquisition_id": review_id,
                        "producer_sequence": index,
                        "frame_sha256": review_id,
                        "clean_image": "x",
                        "privileged_overlay": "y",
                        "pseudo_pixel_count": 30,
                        "core_pixel_count": 24,
                        "uncertainty_pixel_count": 10,
                        "pseudo_source_voxel_count": 5,
                        "core_source_voxel_count": 3,
                        "uncertainty_source_voxel_count": 4,
                        "core_margin_pixels": 5,
                        "core_centroid_row": 100,
                        "core_centroid_column": 75,
                        "projection_gate_passed": True,
                    }
                )
                decisions.append(
                    {
                        "review_id": review_id,
                        "decision": "VISIBLE",
                        "interface_points_rc": "10:10",
                        "notes": "synthetic",
                    }
                )
            for name, content in (
                ("projection_manifest.csv", rows),
                ("blind_decisions.csv", decisions),
            ):
                with (root / name).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(content[0]))
                    writer.writeheader()
                    writer.writerows(content)
            result = MODULE.finalize_review(root, root / "result.json")
            self.assertEqual(result["structures"]["L4"]["verdict"], "AMBIGUOUS")


if __name__ == "__main__":
    unittest.main()
