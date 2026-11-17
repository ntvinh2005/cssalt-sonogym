#!/usr/bin/env python3
"""CPU tests for the expanded editable RIGHT landmark review package."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = REPOSITORY_ROOT / "workflows" / "two_stage_navigation"
sys.path.insert(0, str(WORKFLOW_ROOT))
MODULE_PATH = WORKFLOW_ROOT / "expand_s0010_right_landmark_review.py"
SPEC = importlib.util.spec_from_file_location("expanded_right_landmark_review", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CropTests(unittest.TestCase):
    def test_cropped_affine_preserves_source_world_coordinates(self) -> None:
        affine = np.asarray(
            [
                [1.5, 0.0, 0.0, -10.0],
                [0.0, 1.5, 0.0, 20.0],
                [0.0, 0.0, 1.5, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        lower = np.asarray([4, 5, 6])
        crop_affine = MODULE.cropped_affine(affine, lower)
        crop_point = np.asarray([2.0, 3.0, 4.0, 1.0])
        source_point = crop_point.copy()
        source_point[:3] += lower
        np.testing.assert_allclose(crop_affine @ crop_point, affine @ source_point)

    def test_crop_contains_l4_l5_and_is_not_resampled(self) -> None:
        shape = (20, 22, 24)
        affine = np.diag([1.5, 1.5, 1.5, 1.0])
        ct = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
        segmentation = np.zeros(shape, dtype=np.int16)
        segmentation[6:9, 7:10, 8:11] = 28
        segmentation[10:13, 11:14, 12:15] = 27
        ct_crop, segmentation_crop, metadata = MODULE.crop_l4_l5(
            nib.Nifti1Image(ct, affine),
            nib.Nifti1Image(segmentation, affine),
            margin=2,
        )
        cropped_labels = np.asanyarray(segmentation_crop.dataobj)
        self.assertIn(27, cropped_labels)
        self.assertIn(28, cropped_labels)
        self.assertFalse(metadata["resampled"])
        self.assertEqual(metadata["margin_voxels"], 2)
        self.assertEqual(list(ct_crop.shape), metadata["shape"])


class EditableGeometryTests(unittest.TestCase):
    def test_exported_mask_and_csv_round_trip(self) -> None:
        mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        mask = np.arange(len(mesh.vertices)) % 2 == 0
        with tempfile.TemporaryDirectory() as directory:
            result = MODULE.export_candidate_geometry(
                mesh=mesh,
                mask=mask,
                source_sha256="a" * 64,
                stem=Path(directory) / "candidate",
            )
            loaded = np.load(Path(directory) / result["npz"]["path"])
            np.testing.assert_array_equal(loaded["selected_vertex_mask"], mask)
            self.assertEqual(result["selected_vertex_count"], int(np.sum(mask)))
            csv_rows = (Path(directory) / result["csv"]["path"]).read_text().splitlines()
            self.assertEqual(len(csv_rows), int(np.sum(mask)) + 1)

    def test_response_template_starts_without_reviewer_decisions(self) -> None:
        template = MODULE.response_template()
        self.assertEqual(
            [row["candidate_id"] for row in template["candidate_decisions"]],
            list(MODULE.CANDIDATE_IDS),
        )
        self.assertTrue(
            all(row["decision"] is None for row in template["candidate_decisions"])
        )
        self.assertIsNone(template["reviewer"]["name"])

    def test_patch_roi_is_bounded_by_image(self) -> None:
        patch = np.asarray([[1.0, 2.0, 3.0], [18.0, 19.0, 4.0]])
        vertical, horizontal = MODULE.patch_roi(patch, 0, 1, (20, 20))
        self.assertEqual((vertical.start, vertical.stop), (0, 20))
        self.assertEqual((horizontal.start, horizontal.stop), (0, 20))


if __name__ == "__main__":
    unittest.main()
