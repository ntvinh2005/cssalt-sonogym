#!/usr/bin/env python3
"""CPU tests for transferred Stage 1 interface, feature, and spatial-map behavior."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.stage1_features import (  # noqa: E402
    UNITY_REFERENCE_PATCH_SPEC,
    extract_interface_patch,
    extract_patch_feature,
    patch_feature,
)
from two_stage_navigation.stage1_interfaces import (  # noqa: E402
    InterfaceTrackerConfig,
    track_bright_interfaces,
)
from two_stage_navigation.stage1_map import (  # noqa: E402
    EvidenceObservation,
    SpatialEvidenceMap,
    SpatialMapConfig,
)
from two_stage_navigation.stage1_pipeline import (  # noqa: E402
    ActorPatchCandidate,
    PatchLabelConfig,
    annotate_patch_candidate,
    binary_metrics,
    project_column_to_home_longitudinal_mm,
    reconstruct_landmark,
)


def synthetic_interface_image() -> np.ndarray:
    height, width = 200, 150
    image = np.zeros((height, width), dtype=np.float64)
    columns = np.arange(width)
    rows = np.rint(130.0 + 3.0 * np.sin(columns / 18.0)).astype(int)
    for column, row in enumerate(rows):
        image[max(0, row - 2) : min(height, row + 3), column] = 1.0
        image[min(height, row + 3) : min(height, row + 30), column] = 0.05
    # Deterministic low-amplitude texture avoids a completely piecewise-constant toy image.
    image += 0.01 * np.outer(np.linspace(0.0, 1.0, height), np.linspace(1.0, 0.0, width))
    return image


class FeatureTests(unittest.TestCase):
    def test_reference_feature_shape_and_numeric_validity(self) -> None:
        image = synthetic_interface_image()
        feature = extract_patch_feature(image, 130.0, 75.0, UNITY_REFERENCE_PATCH_SPEC)
        self.assertIsNotNone(feature)
        assert feature is not None
        self.assertEqual(feature.shape, (774,))
        self.assertTrue(np.all(np.isfinite(feature)))

    def test_boundary_patch_is_rejected(self) -> None:
        image = synthetic_interface_image()
        self.assertIsNone(
            extract_interface_patch(image, 5.0, 5.0, UNITY_REFERENCE_PATCH_SPEC)
        )

    def test_constant_patch_still_produces_finite_feature(self) -> None:
        patch = np.ones(
            (
                UNITY_REFERENCE_PATCH_SPEC.depth_pixels,
                UNITY_REFERENCE_PATCH_SPEC.width_pixels,
            )
        )
        feature = patch_feature(patch, UNITY_REFERENCE_PATCH_SPEC)
        self.assertTrue(np.all(np.isfinite(feature)))
        self.assertEqual(feature.shape, (UNITY_REFERENCE_PATCH_SPEC.feature_width,))


class InterfaceTrackerTests(unittest.TestCase):
    def test_tracks_synthetic_bright_interface(self) -> None:
        tracks = track_bright_interfaces(
            synthetic_interface_image(),
            InterfaceTrackerConfig(minimum_mean_ridge_score=0.10),
        )
        self.assertGreaterEqual(len(tracks), 1)
        baseline = tracks[0]
        self.assertTrue(baseline.valid)
        center_row = baseline.row_at(75.0)
        self.assertLess(abs(center_row - (130.0 + 3.0 * np.sin(75.0 / 18.0))), 5.0)

    def test_nonfinite_image_fails(self) -> None:
        image = synthetic_interface_image()
        image[0, 0] = np.nan
        with self.assertRaises(ValueError):
            track_bright_interfaces(image)


def map_config() -> SpatialMapConfig:
    return SpatialMapConfig(
        grid_step_mm=0.5,
        minimum_event_separation_mm=10.0,
        raw_peak_minimum_score=0.55,
        acceptance_score=0.70,
        minimum_forward_frames=2,
        minimum_reverse_frames=2,
    )


def add_supported_peak(spatial_map: SpatialEvidenceMap, coordinate: float, frame_base: int) -> None:
    for offset, direction in enumerate(("FORWARD", "FORWARD", "REVERSE", "REVERSE")):
        spatial_map.add(
            EvidenceObservation(
                coordinate_mm=coordinate,
                probability=0.9 - 0.01 * offset,
                direction=direction,  # type: ignore[arg-type]
                frame_sequence=frame_base + offset,
                track_id="H0",
            )
        )


class SpatialMapTests(unittest.TestCase):
    def test_supported_ordered_events_and_target(self) -> None:
        spatial_map = SpatialEvidenceMap(-10.0, 50.0, map_config())
        add_supported_peak(spatial_map, 0.0, 0)
        add_supported_peak(spatial_map, 20.0, 10)
        add_supported_peak(spatial_map, 40.0, 20)
        events = spatial_map.events()
        self.assertEqual([event.coordinate_mm for event in events], [0.0, 20.0, 40.0])
        self.assertTrue(all(event.accepted for event in events))
        estimate = spatial_map.infer_ordered_target(
            expected_event_count=3,
            target_event_index=1,
            anchor_id="synthetic-complete-order-v1",
        )
        self.assertEqual(estimate.status, "SUCCESS")
        self.assertEqual(estimate.coordinate_mm, 20.0)

    def test_incomplete_ordered_map_abstains(self) -> None:
        spatial_map = SpatialEvidenceMap(-10.0, 50.0, map_config())
        add_supported_peak(spatial_map, 0.0, 0)
        add_supported_peak(spatial_map, 20.0, 10)
        estimate = spatial_map.infer_ordered_target(
            expected_event_count=3,
            target_event_index=1,
            anchor_id="synthetic-complete-order-v1",
        )
        self.assertEqual(estimate.status, "ABSTAIN")
        self.assertIsNone(estimate.coordinate_mm)

    def test_bidirectional_support_is_a_conjunction(self) -> None:
        spatial_map = SpatialEvidenceMap(-10.0, 10.0, map_config())
        for sequence in range(4):
            spatial_map.add(EvidenceObservation(0.0, 0.9, "FORWARD", sequence, "H0"))
        event = spatial_map.events()[0]
        self.assertFalse(event.accepted)
        self.assertIn("INSUFFICIENT_REVERSE_SUPPORT", event.rejection_reasons)

    def test_multiple_tracks_in_one_frame_do_not_inflate_support(self) -> None:
        spatial_map = SpatialEvidenceMap(-10.0, 10.0, map_config())
        spatial_map.add(EvidenceObservation(0.0, 0.8, "FORWARD", 1, "H0"))
        spatial_map.add(EvidenceObservation(0.0, 0.9, "FORWARD", 1, "H1"))
        record = next(item for item in spatial_map.records() if item.coordinate_mm == 0.0)
        self.assertEqual(record.forward_count, 1)
        self.assertAlmostEqual(record.score_q75, 0.9)


class Stage1PipelineTests(unittest.TestCase):
    def test_column_projection_uses_declared_probe_angle(self) -> None:
        center = project_column_to_home_longitudinal_mm(
            probe_longitudinal_home_mm=-50.0,
            column=74.5,
            width_pixels=150,
            pixel_resolution_mm=0.5,
            declared_home_angle_rad=math.pi / 2,
            relative_angle_rad=0.0,
        )
        self.assertAlmostEqual(center, -50.0)
        offset = project_column_to_home_longitudinal_mm(
            probe_longitudinal_home_mm=-50.0,
            column=84.5,
            width_pixels=150,
            pixel_resolution_mm=0.5,
            declared_home_angle_rad=math.pi / 2,
            relative_angle_rad=0.0,
        )
        self.assertAlmostEqual(offset, -45.0)

    def test_patch_annotation_separates_primary_and_l4_verifier(self) -> None:
        candidate = ActorPatchCandidate(
            feature=np.ones(774),
            coordinate_mm=-80.0,
            frame_sequence=1,
            direction="FORWARD",
            track_id="H0",
            interface_row=100.0,
            column=75.0,
        )
        labels = np.zeros((200, 150), dtype=np.uint8)
        labels[80:120, 60:90] = 28
        annotation = annotate_patch_candidate(
            labels, candidate, UNITY_REFERENCE_PATCH_SPEC, PatchLabelConfig()
        )
        self.assertEqual(annotation.primary_label, 1)
        self.assertEqual(annotation.verifier_label, 1)
        labels[80:120, 60:90] = 29
        annotation = annotate_patch_candidate(
            labels, candidate, UNITY_REFERENCE_PATCH_SPEC, PatchLabelConfig()
        )
        self.assertEqual(annotation.primary_label, 1)
        self.assertEqual(annotation.verifier_label, 0)

    def test_binary_metrics_require_precision_and_recall_conjunction(self) -> None:
        metrics = binary_metrics(
            np.asarray([1, 1, 0, 0]), np.asarray([0.9, 0.8, 0.7, 0.1]), 0.75
        )
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_landmark_selector_uses_top_score_margin_and_frozen_offset(self) -> None:
        candidates = []
        primary = []
        verifier = []
        for coordinate, probability, base in ((-84.0, 0.7, 0), (-72.0, 0.9, 20)):
            for offset, direction in enumerate(("FORWARD", "FORWARD", "REVERSE", "REVERSE")):
                candidates.append(
                    ActorPatchCandidate(
                        feature=np.ones(774),
                        coordinate_mm=coordinate,
                        frame_sequence=base + offset,
                        direction=direction,
                        track_id="H0",
                        interface_row=120.0,
                        column=75.0,
                    )
                )
                primary.append(probability)
                verifier.append(probability)
        # Low-score corridor endpoints keep the two supported peaks away from map edges.
        for coordinate, sequence in ((-90.0, 40), (-60.0, 41)):
            candidates.append(
                ActorPatchCandidate(
                    feature=np.ones(774),
                    coordinate_mm=coordinate,
                    frame_sequence=sequence,
                    direction="FORWARD",
                    track_id="H0",
                    interface_row=120.0,
                    column=75.0,
                )
            )
            primary.append(0.1)
            verifier.append(0.1)
        inference, _ = reconstruct_landmark(
            candidates,
            np.asarray(primary),
            np.asarray(verifier),
            map_config=map_config(),
            coordinate_offset_mm=-10.0,
            minimum_top_score_margin=0.1,
        )
        self.assertEqual(inference.status, "SUCCESS")
        self.assertEqual(inference.coordinate_mm, -82.0)
        ambiguous, _ = reconstruct_landmark(
            candidates,
            np.asarray([0.82] * 4 + [0.9] * 4 + [0.1] * 2),
            np.asarray([0.82] * 4 + [0.9] * 4 + [0.1] * 2),
            map_config=map_config(),
            minimum_top_score_margin=0.1,
        )
        self.assertEqual(ambiguous.status, "ABSTAIN")
        self.assertEqual(ambiguous.reason, "AMBIGUOUS_TOP_EVENT_SCORE")


if __name__ == "__main__":
    unittest.main()
