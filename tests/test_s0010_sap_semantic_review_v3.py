#!/usr/bin/env python3
"""Focused CPU tests for the s0010 SAP semantic review V3 builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "workflows/two_stage_navigation/build_s0010_sap_semantic_review_v3.py"
SPEC = importlib.util.spec_from_file_location("sap_semantic_review_v3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SapSemanticReviewV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.affine = np.diag([1.5, 1.5, 1.5, 1.0])
        self.ct = np.full((12, 12, 12), -1000, dtype=np.int16)
        self.ct[2:10, 2:10, 2:10] = 1200
        self.segmentation = np.zeros_like(self.ct, dtype=np.uint8)
        self.segmentation[2:5, 2:10, 2:5] = 29
        self.segmentation[5:8, 2:10, 5:8] = 28
        self.segmentation[8:11, 2:10, 8:11] = 27

    def tearDown(self) -> None:
        self.temp.cleanup()

    def save(self, name: str, array: np.ndarray, affine: np.ndarray | None = None) -> Path:
        path = self.root / name
        nib.save(nib.Nifti1Image(array, self.affine if affine is None else affine), path)
        return path

    def test_ct_preflight_requires_real_aligned_intensity_volume(self) -> None:
        ct = self.save("ct.nii.gz", self.ct)
        segmentation = self.save("seg.nii.gz", self.segmentation)
        report = MODULE.validate_ct(ct, segmentation)
        self.assertEqual(report["orientation"], ["R", "A", "S"])
        self.assertEqual(report["affine_max_abs_difference"], 0.0)
        self.assertEqual(report["levels"]["L4"]["label"], 28)
        wrong_affine = self.affine.copy()
        wrong_affine[0, 3] = 1.0
        wrong = self.save("wrong.nii.gz", self.segmentation, wrong_affine)
        with self.assertRaisesRegex(RuntimeError, "affines differ"):
            MODULE.validate_ct(ct, wrong)

    def test_level_and_right_laterality_use_ras_physical_x(self) -> None:
        prediction = np.zeros_like(self.segmentation)
        prediction[5, 3, 6] = 6
        prediction[7, 3, 6] = 6
        masks, midlines = MODULE.split_level_and_laterality(
            prediction, self.segmentation, self.affine
        )
        self.assertFalse(masks["L4"]["superior_articular_process"][5, 3, 6])
        self.assertTrue(masks["L4"]["superior_articular_process"][7, 3, 6])
        self.assertEqual(midlines["L4"], 9.0)

    def test_prediction_validation_rejects_unknown_labels_and_misalignment(self) -> None:
        ct = self.save("ct.nii.gz", self.ct)
        segmentation = self.save("seg.nii.gz", self.segmentation)
        prediction = np.zeros_like(self.segmentation)
        for label in range(1, 8):
            prediction[label, 0, 0] = label
        valid = self.save("prediction.nii.gz", prediction)
        _, _, _, report = MODULE.validate_prediction(valid, ct, segmentation)
        self.assertEqual(report["labels_present"], list(range(8)))
        prediction[0, 0, 0] = 8
        invalid = self.save("invalid.nii.gz", prediction)
        with self.assertRaisesRegex(RuntimeError, "labels failed"):
            MODULE.validate_prediction(invalid, ct, segmentation)

    def test_v2_comparison_uses_physical_distance(self) -> None:
        mask = np.zeros((12, 12, 12), dtype=bool)
        mask[6:8, 5:7, 5:7] = True
        v2 = {
            "core_index": np.asarray([[6.0, 5.0, 5.0], [1.0, 1.0, 1.0]]),
            "sha256": "frozen-v2",
        }
        report, masks = MODULE.compare_v2_model(
            level="L4",
            model_mask=mask,
            prediction=mask.astype(np.uint8) * 6,
            v2=v2,
            affine=self.affine,
        )
        self.assertEqual(report["physical_consistency_tolerance_mm"], 1.5)
        self.assertEqual(report["v2_core_within_tolerance_location_count"], 1)
        self.assertEqual(report["v2_core_outside_tolerance_location_count"], 1)
        self.assertGreater(np.count_nonzero(masks["agreement"]), 0)

    def test_proximity_does_not_hide_sampled_iap_conflict(self) -> None:
        mask = np.zeros((12, 12, 12), dtype=bool)
        mask[6:8, 5:7, 5:7] = True
        prediction = mask.astype(np.uint8) * 6
        prediction[6, 5, 5] = 7
        v2 = {
            "core_index": np.asarray([[6.0, 5.0, 5.0]]),
            "sha256": "frozen-v2",
        }
        report, masks = MODULE.compare_v2_model(
            level="L4", model_mask=mask, prediction=prediction,
            v2=v2, affine=self.affine,
        )
        self.assertEqual(report["v2_core_within_tolerance_location_count"], 1)
        self.assertEqual(
            masks["v2_categories"].tolist(), ["CLASS_CONFLICT_SAMPLED_IAP"]
        )

    def test_anatomical_consistency_uses_explicit_source_labels(self) -> None:
        prediction = np.zeros((4, 4, 4), dtype=np.uint8)
        segmentation = np.zeros_like(prediction)
        prediction[0, 0, 0] = 6
        segmentation[0, 0, 0] = 28
        prediction[1, 1, 1] = 6
        segmentation[1, 1, 1] = 120
        conversion = {label: 13 for label in MODULE.VERTEBRAL_SOURCE_LABELS}
        conversion[120] = 12
        report = MODULE.anatomical_consistency_report(
            prediction, segmentation, conversion
        )
        self.assertEqual(report["vertebral_labels_only"]["inside_count"], 1)
        self.assertEqual(report["vertebral_labels_only"]["outside_count"], 1)
        self.assertEqual(report["acoustic_bone_labels"]["inside_count"], 1)
        self.assertEqual(report["foreground_by_source_label"], {"28": 1, "120": 1})

    def test_compact_mask_roundtrip_preserves_crop_affine(self) -> None:
        prediction = np.zeros_like(self.segmentation)
        level_masks, _ = MODULE.split_level_and_laterality(
            prediction, self.segmentation, self.affine
        )
        comparison = {
            level: {
                "agreement": np.zeros_like(self.segmentation, dtype=bool),
                "model_only": np.zeros_like(self.segmentation, dtype=bool),
            }
            for level in ("L4", "L5")
        }
        records = MODULE.save_crop(
            ct_image=nib.Nifti1Image(self.ct, self.affine),
            segmentation=self.segmentation,
            prediction=prediction,
            level_masks=level_masks,
            comparisons=comparison,
            output_dir=self.root / "masks",
        )
        self.assertEqual(len(records["files"]), 12)
        for record in records["files"].values():
            self.assertTrue((self.root / "masks" / record["path"]).is_file())

    def test_package_validation_rejects_weights(self) -> None:
        package = self.root / "package"
        package.mkdir()
        required = [
            "00_EXECUTIVE_SUMMARY.md", "01_CT_PREFLIGHT.md", "02_MODEL_INFERENCE.md",
            "03_COMPARISON_RESULTS.csv", "03_COMPARISON_RESULTS.json", "04_ANATOMICAL_REVIEW.md",
            "05_RIGHT_L4_SAP_DECISION_SHEET.md", "06_RIGHT_L5_SAP_DECISION_SHEET.md",
            "07_REVIEW_RESPONSE_TEMPLATE.md", "08_PROVENANCE_AND_CHECKSUMS.json",
        ]
        for name in required:
            (package / name).write_text("ok", encoding="utf-8")
        self.assertTrue(MODULE.validate_package(package, required)["pass"])
        (package / "model_final_checkpoint.model").write_text("forbidden", encoding="utf-8")
        self.assertFalse(MODULE.validate_package(package, required)["pass"])

    def test_frozen_v2_artifact_is_read_only(self) -> None:
        vertices = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        path = self.root / "geometry/RIGHT_L4_SAP_v2_regions.npz"
        path.parent.mkdir()
        np.savez_compressed(
            path,
            source_vertices_mesh_index=vertices,
            proposed_sap_surface_core_vertex_mask=np.asarray([True, False]),
            source_mesh_sha256=np.asarray("source"),
        )
        before = MODULE.sha256(path)
        loaded = MODULE.load_v2_core(self.root, "L4")
        self.assertEqual(loaded["core_index"].tolist(), [[1.0, 2.0, 3.0]])
        self.assertEqual(MODULE.sha256(path), before)


if __name__ == "__main__":
    unittest.main()
