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
    "sap_static_review_v2",
    ROOT / "workflows/two_stage_navigation/build_s0010_sap_static_review_v2.py",
)
assert SPEC and SPEC.loader
REVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW)


class SapStaticReviewV2Tests(unittest.TestCase):
    def test_zero_area_is_traced_to_repeated_physical_corner(self) -> None:
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float
        )
        faces = np.asarray([[0, 1, 2], [0, 3, 1]])
        quality = REVIEW.mesh_quality(vertices, faces)
        self.assertEqual(quality["zero_area_faces"], 1)
        self.assertEqual(quality["zero_area_faces_with_repeated_physical_corner"], 1)
        self.assertEqual(quality["unique_physical_locations"], 3)

    def test_voxel_surface_is_nondegenerate_and_deduplicated(self) -> None:
        volume = np.zeros((4, 4, 4), dtype=np.uint8)
        volume[1:3, 1:3, 1:3] = 7
        vertices, faces = REVIEW.reconstruct_voxel_surface(volume, 7)
        self.assertEqual(len(vertices), len(np.unique(vertices, axis=0)))
        self.assertTrue(np.all(REVIEW.face_areas(vertices, faces) > 0.0))
        self.assertEqual(len(faces), 48)

    def test_core_requires_valid_coincident_surfaces(self) -> None:
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 0, 0]], dtype=float
        )
        sap = {
            "vertices": vertices,
            "faces": np.asarray([[0, 1, 2], [3, 3, 1]]),
            "original": np.asarray([True, True, True, True]),
        }
        upper_vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 0, 0]], dtype=float
        )
        upper_faces = np.asarray([[0, 1, 2], [3, 3, 1]])
        regions = REVIEW.derive_regions(sap, upper_vertices, upper_faces)
        self.assertEqual(regions["unique_core_locations"], 3)
        self.assertFalse(regions["core"][3])
        np.testing.assert_array_equal(
            regions["core"] | regions["uncertain"] | regions["neighbor"],
            sap["original"],
        )

    def test_candidate_partition_does_not_invent_neighbor_semantics(self) -> None:
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]], dtype=float
        )
        sap = {
            "vertices": vertices,
            "faces": np.asarray([[0, 1, 2], [1, 3, 2]]),
            "original": np.ones(4, dtype=bool),
        }
        upper_vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        upper_faces = np.asarray([[0, 1, 2]])
        regions = REVIEW.derive_regions(sap, upper_vertices, upper_faces)
        self.assertFalse(np.any(regions["neighbor"]))
        self.assertTrue(regions["uncertain"][3])

    def test_projection_margin_requires_full_interior_support(self) -> None:
        self.assertEqual(
            REVIEW.projection_margin_pixels(
                np.asarray([10, 20]), np.asarray([15, 30])
            ),
            10,
        )
        self.assertEqual(
            REVIEW.projection_margin_pixels(
                np.asarray([0, 20]), np.asarray([15, 30])
            ),
            0,
        )
        self.assertEqual(
            REVIEW.projection_margin_pixels(np.asarray([]), np.asarray([])), -1
        )

    def test_package_validation_fails_on_missing_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            manifest = {
                "structures": {
                    "RIGHT_L4_SAP": {
                        "figure_files": ["missing.png"],
                        "ultrasound_files": [],
                        "geometry_files": {},
                    }
                }
            }
            validation = REVIEW.validate_package(package, manifest)
            self.assertFalse(validation["pass"])
            self.assertEqual(validation["missing_referenced_files"], ["missing.png"])

    def test_checksums_and_zip_are_self_contained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / REVIEW.PACKAGE_NAME
            package.mkdir()
            (package / "manifest.json").write_text(json.dumps({"ok": True}))
            REVIEW.write_checksums(package)
            destination = root / f"{REVIEW.PACKAGE_NAME}.zip"
            REVIEW.create_zip(package, destination)
            with zipfile.ZipFile(destination) as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn(
                    f"{REVIEW.PACKAGE_NAME}/checksums.sha256", archive.namelist()
                )


if __name__ == "__main__":
    unittest.main()
