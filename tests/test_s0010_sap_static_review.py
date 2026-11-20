from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sap_static_review",
    ROOT / "workflows/two_stage_navigation/build_s0010_sap_static_review.py",
)
assert SPEC and SPEC.loader
REVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW)


def record(index: int, *, session: str, support: int, boundary: bool = False) -> dict:
    return {
        "session_id": session,
        "frame_sequence": index,
        "frame_sha256": f"{index:064x}",
        "unique_projected_pixels": support,
        "boundary_touching": boundary,
        "maximum_coordinate_error_index": 1e-5 + index * 1e-8,
        "phase": "L4_RIGHT_TP_SURFACE_CORE:fine",
        "target_name": "L4_RIGHT_TP_SURFACE_CORE",
        "target_probability": 0.5 + index / 1000,
    }


class StaticReviewTests(unittest.TestCase):
    def test_curated_examples_are_distinct_and_include_contrast(self) -> None:
        rows = [
            record(1, session="a", support=40),
            record(2, session="a", support=25),
            record(3, session="b", support=35),
            record(4, session="b", support=12, boundary=True),
            record(5, session="c", support=7),
            record(6, session="c", support=0),
            record(7, session="d", support=0),
        ]
        selected = REVIEW.select_curated_examples(rows)
        self.assertEqual(len(selected), 6)
        self.assertEqual(len({row["frame_sha256"] for row in selected}), 6)
        self.assertEqual(selected[0]["frame_sequence"], 1)
        self.assertNotEqual(selected[1]["session_id"], selected[0]["session_id"])
        self.assertTrue(any(row["unique_projected_pixels"] == 0 for row in selected))
        self.assertTrue(any(row["boundary_touching"] for row in selected))

    def test_candidate_face_count_requires_all_vertices(self) -> None:
        faces = np.asarray([[0, 1, 2], [1, 2, 3], [3, 4, 5]])
        mask = np.asarray([True, True, True, False, True, True])
        self.assertEqual(REVIEW.candidate_face_count(faces, mask), 1)

    def test_validation_fails_closed_on_missing_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            existing = package / "exists.png"
            existing.write_bytes(b"png")
            manifest = {
                "structures": {
                    "RIGHT_L4_SAP": {
                        "figure_files": ["exists.png", "missing.png"],
                        "ultrasound_files": [],
                        "geometry_files": {},
                    }
                }
            }
            targets = {"targets": []}
            result = REVIEW.validate_package(package, manifest, targets)
            self.assertFalse(result["pass"])
            self.assertEqual(result["missing_referenced_files"], ["missing.png"])

    def test_checksum_and_zip_cover_static_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / REVIEW.PACKAGE_NAME
            package.mkdir()
            (package / "00_EXECUTIVE_SUMMARY.md").write_text("review\n")
            checksum = REVIEW.write_checksums(package)
            self.assertIn("00_EXECUTIVE_SUMMARY.md", checksum.read_text())
            archive = root / f"{REVIEW.PACKAGE_NAME}.zip"
            REVIEW.create_zip(package, archive)
            with zipfile.ZipFile(archive) as handle:
                self.assertIsNone(handle.testzip())
                names = handle.namelist()
            self.assertIn(
                f"{REVIEW.PACKAGE_NAME}/00_EXECUTIVE_SUMMARY.md", names
            )
            self.assertIn(f"{REVIEW.PACKAGE_NAME}/checksums.sha256", names)

    def test_review_targets_are_json_serializable(self) -> None:
        payload = {
            "allowed_decisions": ["ACCEPT_PROVISIONAL", "REVISE", "UNRESOLVED"],
            "targets": [{"target_id": "RIGHT_L4_SAP_main_candidate", "files": []}],
        }
        self.assertEqual(json.loads(json.dumps(payload)), payload)


if __name__ == "__main__":
    unittest.main()
