#!/usr/bin/env python3
"""CPU tests for TP surface-core proposals and leakage-separated fitting."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = ROOT / "source/spinal_surgery/spinal_surgery/evaluation"
sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.stage1_features import (  # noqa: E402
    UNITY_REFERENCE_PATCH_SPEC,
    extract_interface_patch,
    normalized_patch_intensity,
    patch_feature,
)
from two_stage_navigation.tp_surface_core_dataset import (  # noqa: E402
    L4_CLASS,
    L5_CLASS,
    PatchProposal,
    actor_visible_patch_proposals,
    projected_pixel_diagnostics,
    proposed_review_target,
)

TRAIN_PATH = ROOT / "workflows/two_stage_navigation/train_tp_surface_core.py"
SPEC = importlib.util.spec_from_file_location("tp_surface_core_training", TRAIN_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAIN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAIN
SPEC.loader.exec_module(TRAIN)

PREPARE_PATH = ROOT / "workflows/two_stage_navigation/prepare_tp_surface_core_dataset.py"
PREPARE_SPEC = importlib.util.spec_from_file_location("tp_surface_core_prepare", PREPARE_PATH)
assert PREPARE_SPEC is not None and PREPARE_SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(PREPARE_SPEC)
sys.modules[PREPARE_SPEC.name] = PREPARE
PREPARE_SPEC.loader.exec_module(PREPARE)

REVIEW_PATH = ROOT / "workflows/two_stage_navigation/apply_tp_surface_core_review.py"
REVIEW_SPEC = importlib.util.spec_from_file_location("tp_surface_core_review", REVIEW_PATH)
assert REVIEW_SPEC is not None and REVIEW_SPEC.loader is not None
REVIEW = importlib.util.module_from_spec(REVIEW_SPEC)
sys.modules[REVIEW_SPEC.name] = REVIEW
REVIEW_SPEC.loader.exec_module(REVIEW)


class ProposalTests(unittest.TestCase):
    def test_actor_visible_proposals_do_not_need_privileged_coordinates(self) -> None:
        image = np.zeros((200, 150), dtype=np.float64)
        image[118:123, :] = 1.0
        proposals = actor_visible_patch_proposals(
            image, patch_spec=UNITY_REFERENCE_PATCH_SPEC, column_stride_pixels=12
        )
        self.assertGreater(len(proposals), 0)
        self.assertTrue(all(isinstance(item, PatchProposal) for item in proposals))

    def test_negative_and_ambiguous_targets_are_explicit(self) -> None:
        self.assertEqual(
            proposed_review_target({L4_CLASS: 0, L5_CLASS: 0}, {L4_CLASS: 0, L5_CLASS: 0})[0],
            "REVIEW_HARD_NEGATIVE",
        )
        self.assertEqual(
            proposed_review_target({L4_CLASS: 2, L5_CLASS: 0}, {L4_CLASS: 1, L5_CLASS: 0})[0],
            "AMBIGUOUS",
        )

    def test_candidate_coverage_counts_unique_source_images_not_patches(self) -> None:
        rows = []
        for class_name, target in (
            (PREPARE.BACKGROUND_CLASS, "REVIEW_HARD_NEGATIVE"),
            (PREPARE.L4_CLASS, "REVIEW_POSITIVE"),
            (PREPARE.L5_CLASS, "REVIEW_POSITIVE"),
        ):
            for split, count in PREPARE.REVIEW_MINIMUM_UNIQUE_IMAGES.items():
                for index in range(count):
                    row = {
                        "candidate_id": class_name,
                        "review_target": target,
                        "split": split,
                        "frame_sha256": f"{class_name}:{split}:{index}",
                        "adequate_projected_support": True,
                    }
                    rows.extend((row, dict(row)))
        counts = PREPARE.candidate_unique_image_counts(rows)
        self.assertEqual(
            counts[PREPARE.L4_CLASS], PREPARE.REVIEW_MINIMUM_UNIQUE_IMAGES
        )
        self.assertEqual(PREPARE.candidate_coverage_deficits(counts), [])

        rows = [
            row
            for row in rows
            if not (
                row["candidate_id"] == PREPARE.L4_CLASS
                and row["split"] == "calibration"
                and row["frame_sha256"].endswith(":9")
            )
        ]
        deficits = PREPARE.candidate_coverage_deficits(
            PREPARE.candidate_unique_image_counts(rows)
        )
        self.assertEqual(
            deficits,
            ["L4_RIGHT_TP_SURFACE_CORE calibration: 9 unique candidate images < 10"],
        )

    def test_better_supported_proposal_wins_after_all_image_proposals_are_seen(self) -> None:
        common = {
            "candidate_id": PREPARE.L4_CLASS,
            "review_target": "REVIEW_POSITIVE",
            "split": "train",
            "frame_sha256": "same-image",
            "session_id": "session",
            "patch_column": 24,
            "near_interface_core_feature_cells": 1,
            "nearest_core_distance_feature_cells": 1.0,
            "border_core_feature_cell_fraction": 0.0,
            "unique_core_pixels": 20,
            "patch_gradient_p90": 0.0,
            "patch_normalized_std": 0.0,
        }
        first = dict(
            common,
            sample_id="first-border-fragment",
            adequate_projected_support=False,
            interior_core_feature_cells=0,
            core_feature_cells=1,
        )
        later = dict(
            common,
            sample_id="later-supported",
            adequate_projected_support=True,
            interior_core_feature_cells=4,
            core_feature_cells=5,
        )
        selected = PREPARE.best_candidate_per_source_image(
            [first, later], PREPARE.L4_CLASS, "train"
        )
        self.assertEqual([row["sample_id"] for row in selected], ["later-supported"])

    def test_border_fragment_is_not_adequate_projected_support(self) -> None:
        proposal = PatchProposal("H0", 100.0, 72.0, np.zeros(774))
        diagnostics = projected_pixel_diagnostics(
            proposal,
            core_rows=np.asarray([100, 101, 102]),
            core_columns=np.asarray([95, 95, 95]),
            uncertain_rows=np.asarray([], dtype=int),
            uncertain_columns=np.asarray([], dtype=int),
            patch_spec=UNITY_REFERENCE_PATCH_SPEC,
        )
        self.assertFalse(diagnostics["adequate_projected_support"])
        self.assertEqual(diagnostics["interior_core_feature_cells"], 0)

    def test_background_selection_uses_multiple_sessions_and_columns(self) -> None:
        candidates = []
        for session in ("a", "b", "c"):
            for column in (24, 60, 96):
                candidates.append(
                    {
                        "sample_id": f"{session}:{column}",
                        "session_id": session,
                        "patch_column": column,
                        "candidate_id": PREPARE.BACKGROUND_CLASS,
                        "patch_gradient_p90": 1.0,
                        "patch_normalized_std": 1.0,
                    }
                )
        selected, _ = PREPARE.diversified_selection(candidates, 6)
        self.assertEqual(len({row["session_id"] for row in selected}), 3)
        self.assertGreaterEqual(len({row["patch_column"] for row in selected}), 2)

    def test_review_export_matches_production_patch_and_feature(self) -> None:
        image = np.linspace(0.0, 1.0, 200 * 150, dtype=np.float64).reshape(200, 150)
        row = {
            "sample_id": "session:0:H0:0",
            "session_id": "session",
            "frame_sequence": 0,
            "review_selection": "PRIMARY",
            "patch_row": 100.0,
            "patch_column": 72.0,
            "patch_row_start": 66,
            "patch_column_start": 48,
            "patch_width": 48,
            "patch_depth": 80,
            "unique_core_pixels": 2,
            "core_feature_cells": 2,
            "interior_core_feature_cells": 2,
            "nearest_core_distance_feature_cells": 1.0,
            "border_core_feature_cell_fraction": 0.0,
        }
        raw = extract_interface_patch(
            image, row["patch_row"], row["patch_column"], UNITY_REFERENCE_PATCH_SPEC
        )
        self.assertIsNotNone(raw)
        intensity = normalized_patch_intensity(raw, UNITY_REFERENCE_PATCH_SPEC)
        feature = patch_feature(raw, UNITY_REFERENCE_PATCH_SPEC)
        row.update(
            raw_crop_sha256=PREPARE.array_sha256(raw),
            preprocessed_intensity_sha256=PREPARE.array_sha256(intensity),
            feature_sha256=PREPARE.array_sha256(feature),
        )
        projections = {
            PREPARE.L4_CLASS: {
                "rows": np.asarray([100, 101]),
                "columns": np.asarray([72, 73]),
                "uncertain_rows": np.asarray([], dtype=int),
                "uncertain_columns": np.asarray([], dtype=int),
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            PREPARE.export_review_sample(
                output=Path(temporary),
                row=row,
                image=image,
                projections=projections,
                npz_index=0,
            )
        self.assertEqual(row["raw_crop_sha256"], PREPARE.array_sha256(raw))
        self.assertEqual(row["feature_sha256"], PREPARE.array_sha256(feature))


class TrainingTests(unittest.TestCase):
    def test_partition_never_places_calibration_or_development_in_fit(self) -> None:
        rows = [
            {"sample_id": "train", "split": "train"},
            {"sample_id": "cal", "split": "calibration"},
            {"sample_id": "dev", "split": "development_evaluation"},
        ]
        partitions = TRAIN.partition_eligible_rows(rows)
        self.assertEqual([row["sample_id"] for row in partitions["train"]], ["train"])
        self.assertNotIn("cal", {row["sample_id"] for row in partitions["train"]})
        self.assertNotIn("dev", {row["sample_id"] for row in partitions["train"]})

    def test_synthetic_fit_saves_reloads_and_evaluates_disjoint_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            output = root / "output"
            dataset.mkdir()
            rows = []
            images = []
            split_sizes = {"train": 40, "calibration": 10, "development_evaluation": 10}
            for class_id in (0, 1, 2):
                for split_index, (split, count) in enumerate(split_sizes.items()):
                    for index in range(count):
                        image = np.zeros((200, 150), dtype=np.float32)
                        row = 70 + class_id * 32
                        image[row - 2 : row + 3, :] = 0.4 + 0.25 * class_id
                        image += np.float32((split_index * 100 + index + 1) * 1e-5)
                        archive_index = len(images)
                        images.append(image)
                        negative = class_id == 0
                        rows.append(
                            {
                                "sample_id": f"{split}:{class_id}:{index}",
                                "session_id": f"{split}_session",
                                "split": split,
                                "frame_sequence": archive_index,
                                "acquisition_id": f"a:{archive_index}",
                                "frame_sha256": TRAIN.ultrasound_sha256(image),
                                "image_archive": str(dataset / "images.npz"),
                                "image_archive_index": archive_index,
                                "proposal_source": "ACTOR_VISIBLE_BRIGHT_INTERFACE",
                                "track_id": "H0",
                                "candidate_id": TRAIN.CLASS_NAMES[class_id],
                                "class_id": class_id,
                                "patch_row": row,
                                "patch_column": 75,
                                "training_eligible": True,
                                "training_label": "VERIFIED_NEGATIVE" if negative else "VERIFIED_POSITIVE",
                                "semantic_identity": "REJECTED_AS_TP" if negative else "SUPPORTED",
                                "acoustic_visibility": "VISIBLE_OTHER_STRUCTURE" if negative else "VISIBLE",
                                "anatomical_approval": "NOT_APPLICABLE" if negative else "APPROVED",
                            }
                        )
            np.savez_compressed(dataset / "images.npz", ultrasound=np.stack(images))
            with (dataset / "candidate_sample_manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (dataset / "dataset_summary.json").write_text(
                json.dumps({"status": "ENGINEERING_READY_LABEL_REVIEW_PENDING"})
            )
            original = sys.argv
            try:
                sys.argv = ["train_tp_surface_core.py", "--dataset-dir", str(dataset), "--output-dir", str(output)]
                self.assertEqual(TRAIN.main(), 0)
            finally:
                sys.argv = original
            result = json.loads((output / "preflight_result.json").read_text())
            self.assertTrue(result["checkpoint_saved"])
            self.assertTrue(result["checkpoint_reload_verified"])
            self.assertTrue(all(value.startswith("train:") for value in result["fit_sample_ids"]))
            self.assertTrue(all(value.startswith("calibration:") for value in result["calibration_sample_ids"]))
            self.assertTrue(all(value.startswith("development_evaluation:") for value in result["development_sample_ids"]))


class ReviewApplicationTests(unittest.TestCase):
    def test_exact_review_contract_promotes_disjoint_minimum_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            rows = []
            decisions = []
            class_ids = {
                "BACKGROUND_OR_OTHER_STRUCTURE": -1,
                "L4_RIGHT_TP_SURFACE_CORE": 1,
                "L5_RIGHT_TP_SURFACE_CORE": 2,
            }
            for class_name, class_id in class_ids.items():
                for split, count in REVIEW.MINIMUM_UNIQUE_IMAGES.items():
                    for index in range(count):
                        sample_id = f"{class_name}:{split}:{index}"
                        geometry = "background" if class_id == -1 else f"geometry:{class_id}"
                        row = {
                            "sample_id": sample_id,
                            "session_id": f"{split}_session",
                            "split": split,
                            "frame_sha256": f"hash:{sample_id}",
                            "image_archive_index": index,
                            "track_id": "H0",
                            "patch_row": 100.0,
                            "patch_column": 72.0,
                            "patch_row_start": 66,
                            "patch_column_start": 48,
                            "candidate_id": class_name,
                            "class_id": class_id,
                            "geometry_version_sha256": geometry,
                            "review_asset": f"review/{sample_id}.png",
                            "review_selection": "PRIMARY",
                            "raw_crop_sha256": f"raw:{sample_id}",
                            "preprocessed_intensity_sha256": f"intensity:{sample_id}",
                            "feature_sha256": f"feature:{sample_id}",
                            "preprocessing_spec_sha256": "preprocessing",
                            "acoustic_visibility": "NOT_REVIEWED",
                            "semantic_identity": "NOT_REVIEWED",
                            "anatomical_approval": "PENDING",
                            "training_label": "EXCLUDED",
                            "training_eligible": False,
                            "exclusion_reason": "pending",
                        }
                        rows.append(row)
                        negative = class_id == -1
                        decisions.append(
                            {
                                "sample_id": sample_id,
                                "session_id": row["session_id"],
                                "split": split,
                                "frame_sha256": row["frame_sha256"],
                                "source_array_index": str(index),
                                "track_id": "H0",
                                "patch_row": "100.0",
                                "patch_column": "72.0",
                                "patch_row_start": "66",
                                "patch_column_start": "48",
                                "raw_crop_sha256": row["raw_crop_sha256"],
                                "preprocessed_intensity_sha256": row[
                                    "preprocessed_intensity_sha256"
                                ],
                                "feature_sha256": row["feature_sha256"],
                                "preprocessing_spec_sha256": "preprocessing",
                                "class_name": class_name,
                                "geometry_version_sha256": geometry,
                                "anatomical_boundary_decision": "NOT_APPLICABLE" if negative else "ACCEPT_BOUNDARY",
                                "acoustic_visibility": "VISIBLE_OTHER_STRUCTURE" if negative else "VISIBLE",
                                "semantic_identity": "REJECTED_AS_TP" if negative else "SUPPORTED",
                                "reviewer_identity": "reviewer",
                                "reviewer_qualification": "documented qualification",
                                "notes": "",
                            }
                        )
            with (dataset / "candidate_sample_manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (dataset / "dataset_summary.json").write_text(
                json.dumps({"status": "ENGINEERING_READY_LABEL_REVIEW_PENDING"})
            )
            decisions_path = root / "decisions.csv"
            with decisions_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
                writer.writeheader()
                writer.writerows(decisions)
            result = REVIEW.apply_review(dataset, decisions_path, root / "reviewed")
            self.assertEqual(result["status"], "TRAINING_READY_FOR_PREFLIGHT")
            self.assertEqual(result["promoted_rows"], 180)
            reviewed = REVIEW.read_csv(root / "reviewed/candidate_sample_manifest.csv")
            self.assertTrue(all(row["training_eligible"] == "True" for row in reviewed))
            decisions[0]["patch_column"] = "84.0"
            stale_path = root / "stale-decisions.csv"
            with stale_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
                writer.writeheader()
                writer.writerows(decisions)
            with self.assertRaisesRegex(ValueError, "patch_column"):
                REVIEW.apply_review(dataset, stale_path, root / "stale-reviewed")

    def test_identity_mismatch_and_unqualified_acceptance_fail_closed(self) -> None:
        row = {
            "sample_id": "positive",
            "candidate_id": "L4_RIGHT_TP_SURFACE_CORE",
            "class_id": "1",
        }
        decision = {
            "anatomical_boundary_decision": "ACCEPT_BOUNDARY",
            "acoustic_visibility": "VISIBLE",
            "semantic_identity": "SUPPORTED",
            "reviewer_identity": "",
            "reviewer_qualification": "",
        }
        with self.assertRaisesRegex(ValueError, "reviewer provenance"):
            REVIEW.promote(row, decision)


if __name__ == "__main__":
    unittest.main()
