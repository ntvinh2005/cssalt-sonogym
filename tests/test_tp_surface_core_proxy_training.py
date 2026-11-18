#!/usr/bin/env python3
"""CPU regression tests for the isolated simulator-geometry proxy pilot."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "workflows/two_stage_navigation/train_tp_surface_core_proxy.py"
SPEC = importlib.util.spec_from_file_location("tp_surface_core_proxy", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROXY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROXY
SPEC.loader.exec_module(PROXY)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ProxyConditionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = {
            "excluded_split": "opened_development_smoke",
            "source_candidate_id": PROXY.CLASS_NAMES[1],
            "required_review_target": "REVIEW_POSITIVE",
            "projection_status_required": "VERIFIED",
            "required_core": "L4",
            "other_core_must_be_zero": True,
            "all_uncertain_must_be_zero": True,
            "adequate_projected_support_required": True,
        }
        self.row = {
            "split": "train",
            "candidate_id": PROXY.CLASS_NAMES[1],
            "review_target": "REVIEW_POSITIVE",
            "projection_status": "VERIFIED",
            "l4_core_points_in_patch": "4",
            "l5_core_points_in_patch": "0",
            "l4_uncertain_points_in_patch": "0",
            "l5_uncertain_points_in_patch": "0",
            "adequate_projected_support": "True",
        }

    def test_exact_proxy_conditions_are_required(self) -> None:
        self.assertEqual(PROXY.proxy_condition(self.row, self.rule), (True, "eligible"))
        changed = dict(self.row, l4_uncertain_points_in_patch="1")
        self.assertEqual(
            PROXY.proxy_condition(changed, self.rule),
            (False, "uncertain_region_present"),
        )
        changed = dict(self.row, l5_core_points_in_patch="1")
        self.assertEqual(
            PROXY.proxy_condition(changed, self.rule),
            (False, "competing_core_present"),
        )

    def test_smoke_is_excluded(self) -> None:
        row = dict(self.row, split="opened_development_smoke")
        self.assertEqual(PROXY.proxy_condition(row, self.rule), (False, "excluded_split"))

    def test_best_supported_patch_is_selected_per_source_image(self) -> None:
        common = {
            "proxy_class_id": 1,
            "split": "train",
            "frame_sha256": "frame",
            "near_interface_core_feature_cells": "1",
            "nearest_core_distance_feature_cells": "1",
            "border_core_feature_cell_fraction": "0",
            "unique_core_pixels": "3",
        }
        weak = dict(
            common,
            sample_id="weak",
            interior_core_feature_cells="1",
            core_feature_cells="2",
        )
        strong = dict(
            common,
            sample_id="strong",
            interior_core_feature_cells="5",
            core_feature_cells="6",
        )
        selected = PROXY.choose_one_per_source_image([weak, strong])
        self.assertEqual([row["sample_id"] for row in selected], ["strong"])


class ProxyEndToEndTests(unittest.TestCase):
    def test_no_fit_then_fit_keeps_splits_separate_and_reloads_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            images: list[np.ndarray] = []
            rows: list[dict[str, object]] = []
            split_counts = {"train": 2, "calibration": 1, "development_evaluation": 1}
            geometry_bytes = {"L4": b"l4-geometry", "L5": b"l5-geometry"}
            geometry_hashes = {
                level: hashlib.sha256(value).hexdigest() for level, value in geometry_bytes.items()
            }
            geometry_json = json.dumps(
                {PROXY.CLASS_NAMES[1]: geometry_hashes["L4"], PROXY.CLASS_NAMES[2]: geometry_hashes["L5"]},
                sort_keys=True,
            )
            for class_id in PROXY.CLASS_NAMES:
                for split_index, (split, count) in enumerate(split_counts.items()):
                    for index in range(count):
                        image = np.zeros((200, 150), dtype=np.float32)
                        row_position = 70 + class_id * 35
                        image[row_position - 2 : row_position + 3] = np.float32(0.3 + 0.3 * class_id)
                        image += np.float32((split_index * 10 + index + 1) * 1e-4)
                        archive_index = len(images)
                        images.append(image)
                        background = class_id == 0
                        rows.append(
                            {
                                "sample_id": f"{split}_{class_id}:{archive_index}:H0:0",
                                "session_id": f"{split}_{class_id}",
                                "split": split,
                                "frame_sequence": archive_index,
                                "track_id": "H0",
                                "frame_sha256": PROXY.ultrasound_sha256(image),
                                "image_archive": str(dataset / "images.npz"),
                                "image_archive_index": archive_index,
                                "candidate_id": PROXY.CLASS_NAMES[class_id],
                                "review_target": "REVIEW_HARD_NEGATIVE" if background else "REVIEW_POSITIVE",
                                "projection_status": "VERIFIED",
                                "l4_core_points_in_patch": 3 if class_id == 1 else 0,
                                "l5_core_points_in_patch": 3 if class_id == 2 else 0,
                                "l4_uncertain_points_in_patch": 0,
                                "l5_uncertain_points_in_patch": 0,
                                "adequate_projected_support": not background,
                                "geometry_version_sha256": geometry_json,
                                "patch_row": row_position,
                                "patch_column": 75,
                                "patch_gradient_p90": 1,
                                "patch_normalized_std": 1,
                                "interior_core_feature_cells": 2,
                                "near_interface_core_feature_cells": 1,
                                "core_feature_cells": 2,
                                "nearest_core_distance_feature_cells": 1,
                                "border_core_feature_cell_fraction": 0,
                                "unique_core_pixels": 3,
                            }
                        )
            np.savez_compressed(dataset / "images.npz", ultrasound=np.stack(images))
            manifest = dataset / "candidate_sample_manifest.csv"
            write_csv(manifest, rows)
            package = root / "source.zip"
            with zipfile.ZipFile(package, "w") as archive:
                archive.write(manifest, "dataset/candidate_sample_manifest.csv")
                for level, basename in PROXY.GEOMETRY_ARCHIVES.items():
                    archive.writestr(f"geometry/{basename}", geometry_bytes[level])
            package_hash = PROXY.sha256(package)
            contract_rows = []
            for class_id, name in PROXY.CLASS_NAMES.items():
                background = class_id == 0
                contract_rows.append(
                    {
                        "experiment_id": PROXY.EXPERIMENT_ID,
                        "label_mode": PROXY.LABEL_MODE,
                        "source_zip_sha256": package_hash,
                        "class_id": class_id,
                        "class_name": name,
                        "source_candidate_id": name,
                        "required_review_target": "REVIEW_HARD_NEGATIVE" if background else "REVIEW_POSITIVE",
                        "required_core": "NONE" if background else "L4" if class_id == 1 else "L5",
                        "other_core_must_be_zero": "true",
                        "all_uncertain_must_be_zero": "true",
                        "adequate_projected_support_required": "false" if background else "true",
                        "projection_status_required": "VERIFIED",
                        "excluded_split": "opened_development_smoke",
                        "generated_proxy_label": f"PROXY_{class_id}",
                        "min_train_unique_images": 2,
                        "min_calibration_unique_images": 1,
                        "min_development_unique_images": 1,
                    }
                )
            contract = root / "contract.csv"
            write_csv(contract, contract_rows)
            original = sys.argv
            try:
                preflight = root / "preflight"
                sys.argv = [
                    "train_tp_surface_core_proxy.py",
                    "--dataset-dir",
                    str(dataset),
                    "--source-zip",
                    str(package),
                    "--label-contract",
                    str(contract),
                    "--output-dir",
                    str(preflight),
                    "--no-fit",
                ]
                self.assertEqual(PROXY.main(), 0)
                preflight_result = json.loads((preflight / "result.json").read_text())
                self.assertEqual(preflight_result["status"], "PROXY_TRAINING_READY")
                self.assertFalse(preflight_result["fit_called"])

                training = root / "training"
                sys.argv = sys.argv[:-2] + [str(training)]
                self.assertEqual(PROXY.main(), 0)
            finally:
                sys.argv = original
            result = json.loads((training / "result.json").read_text())
            self.assertEqual(result["status"], "EXPLORATORY_PROXY_TRAINING_COMPLETED")
            self.assertTrue(result["checkpoint_reload_verified"])
            self.assertTrue(all(value.startswith("train_") for value in result["fit_sample_ids"]))
            self.assertTrue(
                all(value.startswith("calibration_") for value in result["calibration_sample_ids"])
            )
            self.assertTrue(
                all(value.startswith("development_evaluation_") for value in result["development_sample_ids"])
            )


if __name__ == "__main__":
    unittest.main()
