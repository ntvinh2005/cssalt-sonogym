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

from two_stage_navigation.stage1_features import UNITY_REFERENCE_PATCH_SPEC  # noqa: E402
from two_stage_navigation.tp_surface_core_dataset import (  # noqa: E402
    L4_CLASS,
    L5_CLASS,
    PatchProposal,
    actor_visible_patch_proposals,
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


if __name__ == "__main__":
    unittest.main()
