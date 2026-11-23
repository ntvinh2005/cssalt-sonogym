from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from evaluate_s0010_clinician_scan_perception import (
    binary_metrics,
    evaluate_selected_pair_identity,
    predicted_presence,
    read_dataset_manifest,
    trajectory_scoped_pair_decision,
)
from two_stage_navigation.sonoanatomy_multiview import SemanticObservation


class ClinicianScanPerceptionTests(unittest.TestCase):
    @staticmethod
    def observation(kind: str, frame: int, coordinate: float, confidence: float) -> SemanticObservation:
        return SemanticObservation(
            semantic_class=kind,
            frame_sequence=frame,
            frame_sha256=f"{frame:064d}",
            viewpoint_id=f"view-{frame}",
            viewpoint_position_m=(0.003 * frame, 0.0, 0.0),
            viewpoint_rotation_columns=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            row=100.0,
            column=75.0,
            coordinate_mm=(coordinate, 0.0, 0.0),
            confidence=confidence,
            pixel_count=5,
        )

    def test_binary_metrics_report_explicit_counts(self) -> None:
        result = binary_metrics(
            np.asarray([True, True, False, False]),
            np.asarray([True, False, True, False]),
        )
        self.assertEqual(
            {name: result[name] for name in ("tp", "fp", "fn", "tn")},
            {"tp": 1, "fp": 1, "fn": 1, "tn": 1},
        )
        self.assertEqual(result["accepted_count"], 2)
        self.assertEqual(result["total_count"], 4)

    def test_presence_uses_frozen_threshold_and_minimum_pixel_count(self) -> None:
        probabilities = np.zeros((4, 5, 5), dtype=np.float32)
        probabilities[1, 1, 1:3] = 0.8
        probabilities[2, 2, 2] = 0.9
        thresholds = {
            name: {"threshold": 0.7, "minimum_pixels": 2}
            for name in ("TP", "SAP", "OTHER_BONE")
        }
        result = predicted_presence(probabilities, thresholds)
        self.assertTrue(result["tp_predicted"])
        self.assertFalse(result["sap_predicted"])
        self.assertEqual(result["tp_predicted_pixels"], 2)
        self.assertEqual(result["sap_predicted_pixels"], 1)

    def test_presence_does_not_count_a_nonwinning_class(self) -> None:
        probabilities = np.zeros((4, 3, 3), dtype=np.float32)
        probabilities[1, 1, 1] = 0.8
        probabilities[2, 1, 1] = 0.9
        thresholds = {
            name: {"threshold": 0.7, "minimum_pixels": 1}
            for name in ("TP", "SAP", "OTHER_BONE")
        }

        result = predicted_presence(probabilities, thresholds)

        self.assertFalse(result["tp_predicted"])
        self.assertTrue(result["sap_predicted"])

    def test_dataset_manifest_truth_is_scoped_to_capture_root(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            capture.mkdir()
            manifest = root / "manifest.csv"
            manifest.write_text(
                "source_root,session_id,frame_sequence,has_tp_interface,has_sap_interface\n"
                f"{capture},session,0,True,False\n"
                f"{root / 'other'},other,0,True,True\n",
                encoding="utf-8",
            )

            result = read_dataset_manifest(manifest, capture)

        self.assertEqual(set(result), {("session", 0)})
        self.assertEqual(result[("session", 0)]["both_supported"], "False")

    def test_trajectory_scoping_selects_strongest_sweep_without_mixing(self) -> None:
        observations = (
            self.observation("TP", 0, 0.0, 0.7),
            self.observation("TP", 1, 0.5, 0.7),
            self.observation("SAP", 2, 8.0, 0.7),
            self.observation("SAP", 3, 8.5, 0.7),
            self.observation("TP", 4, 30.0, 0.95),
            self.observation("TP", 5, 30.5, 0.95),
            self.observation("SAP", 6, 38.0, 0.95),
            self.observation("SAP", 7, 38.5, 0.95),
        )
        frames = [
            {
                "frame_sequence": index,
                "trajectory_id": "first" if index < 4 else "second",
            }
            for index in range(8)
        ]

        result = trajectory_scoped_pair_decision(observations, frames)

        self.assertEqual(result["status"], "ACCEPT")
        self.assertEqual(result["selected_trajectory_id"], "second")
        self.assertEqual(len(result["trajectory_decisions"]), 2)

    def test_selected_pair_identity_is_evaluator_only_and_explicit(self) -> None:
        observations = (
            self.observation("TP", 0, 0.0, 0.9),
            self.observation("TP", 1, 0.5, 0.9),
            self.observation("SAP", 2, 8.0, 0.9),
            self.observation("SAP", 3, 8.5, 0.9),
        )
        frames = [
            {"session_id": "session", "frame_sequence": index, "trajectory_id": "sweep"}
            for index in range(4)
        ]
        scoped = trajectory_scoped_pair_decision(observations, frames)
        labels = np.zeros((4, 200, 150), dtype=np.uint8)
        for index, observation in enumerate(observations):
            labels[index, int(observation.row), int(observation.column)] = (
                1 if observation.semantic_class == "TP" else 2
            )

        result = evaluate_selected_pair_identity(
            scoped,
            observations,
            frames,
            labels,
            {("session", index): index for index in range(4)},
        )

        self.assertTrue(result["accepted"])
        self.assertTrue(result["pair_correct"])

        labels[:, :, :] = 0
        negative = evaluate_selected_pair_identity(
            scoped,
            observations,
            frames,
            labels,
            {("session", index): index for index in range(4)},
        )
        self.assertFalse(negative["pair_correct"])
        self.assertEqual(negative["structures"]["TP"]["distances_pixels"], [None, None])


if __name__ == "__main__":
    unittest.main()
