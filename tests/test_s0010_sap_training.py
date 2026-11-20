"""Focused tests for SAP calibration gates and binary label handling."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
STAGE1_PACKAGES = PROJECT.parent / "cssalt-sonogym-runtime/envs/stage1-python"
if STAGE1_PACKAGES.exists():
    sys.path.insert(0, str(STAGE1_PACKAGES))
sys.path.insert(0, str(PROJECT / "workflows/two_stage_navigation"))
import train_s0010_sap_classifier as training  # noqa: E402


class SapTrainingTests(unittest.TestCase):
    def test_perfect_calibration_selects_a_passing_threshold(self) -> None:
        labels = np.asarray([1, 1, 0, 0])
        scores = np.asarray([0.9, 0.8, 0.2, 0.1])
        threshold, rows = training.select_threshold(labels, scores)
        self.assertIsNotNone(threshold)
        selected = training.metrics(labels, scores, float(threshold))
        self.assertTrue(selected["gate_pass"])
        self.assertTrue(any(row["gate_pass"] for row in rows))

    def test_reject_all_cannot_pass_by_precision_alone(self) -> None:
        labels = np.asarray([1, 1, 0, 0])
        scores = np.zeros(4)
        threshold, _ = training.select_threshold(labels, scores)
        self.assertIsNone(threshold)

    def test_only_two_binary_labels_are_supported(self) -> None:
        self.assertEqual(
            training.LABEL_ID,
            {
                training.NEGATIVE: 0,
                training.POSITIVE: 1,
            },
        )
        self.assertNotIn("BOUNDARY_OR_IDENTITY_AMBIGUOUS", training.LABEL_ID)

    def test_full_image_score_uses_all_actor_visible_proposals(self) -> None:
        class Estimator:
            classes_ = np.asarray([0, 1])

            def predict_proba(self, values: np.ndarray) -> np.ndarray:
                scores = values[:, 0]
                return np.column_stack((1.0 - scores, scores))

        sample_rows = [
            {
                "frame_sha256": "positive",
                "conditioned_level": "L4",
                "split": "calibration",
                "provisional_label": training.POSITIVE,
            },
            {
                "frame_sha256": "negative",
                "conditioned_level": "L4",
                "split": "calibration",
                "provisional_label": training.NEGATIVE,
            },
        ]
        proposal_rows = [
            {
                "frame_sha256": "positive",
                "conditioned_level": "L4",
                "split": "calibration",
                "proposal_feature_index": str(index),
            }
            for index in (0, 1)
        ] + [
            {
                "frame_sha256": "negative",
                "conditioned_level": "L4",
                "split": "calibration",
                "proposal_feature_index": "2",
            }
        ]
        features = np.zeros((3, 774))
        features[:, 0] = [0.2, 0.9, 0.1]
        labels, scores, hashes = training.full_image_scores(
            estimator=Estimator(),
            level="L4",
            split="calibration",
            sample_rows=sample_rows,
            proposal_rows=proposal_rows,
            proposal_features=features,
        )
        self.assertEqual(hashes, ["negative", "positive"])
        np.testing.assert_array_equal(labels, [0, 1])
        np.testing.assert_allclose(scores, [0.1, 0.9])

    def test_failed_calibration_still_persists_reloadable_estimator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            output = root / "output"
            dataset.mkdir()
            rows = []
            proposals = []
            for split in ("train", "calibration"):
                for label in (training.NEGATIVE, training.POSITIVE):
                    for index in range(30):
                        feature_index = len(rows)
                        frame_hash = f"{split}-{label}-{index}"
                        rows.append(
                            {
                                "sample_id": frame_hash,
                                "session_id": (
                                    f"train-{index % 3}"
                                    if split == "train"
                                    else "calibration-0"
                                ),
                                "split": split,
                                "frame_sha256": frame_hash,
                                "conditioned_level": "L4",
                                "provisional_label": label,
                                "training_eligible": "true",
                                "feature_index": str(feature_index),
                            }
                        )
                        proposals.append(
                            {
                                "frame_sha256": frame_hash,
                                "conditioned_level": "L4",
                                "split": split,
                                "proposal_feature_index": str(feature_index),
                            }
                        )

            def write_csv(path: Path, values: list[dict[str, str]]) -> None:
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                    writer.writeheader()
                    writer.writerows(values)

            write_csv(dataset / "sample_manifest.csv", rows)
            write_csv(dataset / "proposal_manifest.csv", proposals)
            constant_features = np.zeros((len(rows), 774), dtype=np.float64)
            np.savez_compressed(
                dataset / "dataset_arrays.npz",
                features=constant_features,
                proposal_features=constant_features,
            )
            (dataset / "readiness_report.json").write_text(
                json.dumps(
                    {
                        "phase": "development",
                        "readiness": {"L4": {"dataset_status": "READY", "deficits": []}},
                        "preprocessing": {"feature_width": 774},
                    }
                ),
                encoding="utf-8",
            )
            arguments = [
                "train_s0010_sap_classifier.py",
                "--dataset-dir",
                str(dataset),
                "--output-dir",
                str(output),
                "--level",
                "L4",
                "--fit",
            ]
            with (
                mock.patch.object(sys, "argv", arguments),
                mock.patch.object(training, "N_ESTIMATORS", 5),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                return_code = training.main()

            self.assertEqual(return_code, 3)
            result = json.loads((output / "training_result.json").read_text())
            level = result["levels"]["L4"]
            self.assertEqual(result["status"], "TRAINING_BLOCKED")
            self.assertEqual(level["status"], "SAP_TRAINING_INCONCLUSIVE")
            self.assertTrue(level["checkpoint_reload_verified"])
            checkpoint = Path(level["checkpoint"])
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(
                training.file_sha256(checkpoint), level["checkpoint_sha256"]
            )
            bundle = joblib.load(checkpoint)
            self.assertIsNone(bundle["threshold"])
            self.assertIsNone(bundle["heuristic_gradient_threshold"])
            self.assertEqual(bundle["level"], "L4")


if __name__ == "__main__":
    unittest.main()
