#!/usr/bin/env python3
"""CPU tests for supervised best-view features, labels, and selection."""

from __future__ import annotations

import sys
import unittest
import hashlib
from pathlib import Path

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
sys.path.insert(0, str(EVALUATION_ROOT))
PROTOCOL = (
    REPOSITORY_ROOT
    / "experiments"
    / "unity_two_stage_transfer"
    / "stage2_dataset_protocol_v1.yaml"
)
PROTOCOL_V2 = PROTOCOL.with_name("stage2_dataset_protocol_v2.yaml")

from two_stage_navigation.stage2_features import (  # noqa: E402
    FEATURE_WIDTH,
    extract_stage2_feature,
    normalize_ultrasound,
)
from two_stage_navigation.stage2_models import (  # noqa: E402
    ScoredView,
    Stage2LabelConfig,
    fresh_confirmation_passes,
    select_best_view,
    supervised_targets,
)


def image() -> np.ndarray:
    y, x = np.mgrid[0:200, 0:150]
    return (0.1 * x + 0.2 * y + 3.0 * np.sin(x / 9.0)).astype(np.float32)


def view(
    candidate_id: str,
    probability: float,
    quality: float,
    *,
    transition_valid: bool = True,
    image_valid: bool = True,
    digest_character: str = "a",
) -> ScoredView:
    return ScoredView(
        candidate_id=candidate_id,
        relative_pose=(0.0, 0.0, 0.0),
        validity_probability=probability,
        quality=quality,
        transition_valid=transition_valid,
        image_valid=image_valid,
        frame_sha256=digest_character * 64,
    )


class Stage2FeatureTests(unittest.TestCase):
    def test_feature_shape_and_numeric_validity(self) -> None:
        feature = extract_stage2_feature(image())
        self.assertEqual(feature.shape, (FEATURE_WIDTH,))
        self.assertTrue(np.all(np.isfinite(feature)))

    def test_normalization_is_bounded(self) -> None:
        normalized = normalize_ultrasound(image())
        self.assertEqual(normalized.shape, (200, 150))
        self.assertGreaterEqual(float(normalized.min()), 0.0)
        self.assertLessEqual(float(normalized.max()), 1.0)

    def test_constant_or_wrong_shape_image_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extract_stage2_feature(np.ones((200, 150), dtype=np.float32))
        with self.assertRaises(ValueError):
            extract_stage2_feature(np.ones((20, 15), dtype=np.float32))


class Stage2ProtocolTests(unittest.TestCase):
    def test_protocol_is_sealed_and_session_separated(self) -> None:
        expected = PROTOCOL.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        observed = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
        self.assertEqual(observed, expected)
        protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
        splits = [row["split"] for row in protocol["sessions"]]
        self.assertEqual(splits.count("train"), 4)
        self.assertEqual(splits.count("calibration"), 1)
        self.assertEqual(splits.count("development_evaluation"), 1)
        self.assertEqual(len({row["session_id"] for row in protocol["sessions"]}), 6)

    def test_protocol_preserves_resource_and_information_bounds(self) -> None:
        protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
        resources = protocol["resource_bound"]
        self.assertEqual(resources["account"], "anest-dept")
        self.assertEqual(resources["gpu_count"], 1)
        self.assertEqual(resources["cpu_cores"], 4)
        self.assertEqual(resources["memory"], "30gb")
        forbidden = set(protocol["information_boundaries"]["actor_record_forbidden"])
        self.assertIn("goal_cmd_pose", forbidden)
        self.assertIn("command_pose_index", forbidden)
        self.assertIn("raw_labels", forbidden)

    def test_v2_changes_only_the_documented_convergence_bound(self) -> None:
        expected = PROTOCOL_V2.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V2.read_bytes()).hexdigest(), expected)
        first = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
        second = yaml.safe_load(PROTOCOL_V2.read_text(encoding="utf-8"))
        self.assertEqual(second["amendment"]["evidence_job"], 51231922)
        self.assertEqual(first["sampling"]["maximum_action_steps_per_waypoint"], 8)
        self.assertEqual(second["sampling"]["maximum_action_steps_per_waypoint"], 16)
        for key in (
            "sessions",
            "labels",
            "features",
            "model",
            "development_gate",
        ):
            self.assertEqual(first[key], second[key])
        self.assertEqual(
            first["sampling"]["maximum_total_action_steps_per_session"],
            second["sampling"]["maximum_total_action_steps_per_session"],
        )


class Stage2LabelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Stage2LabelConfig()

    def test_validity_requires_all_conditions(self) -> None:
        valid, quality = supervised_targets(
            technical_image_valid=True,
            position_error_mm=10.0,
            orientation_error_deg=10.0,
            l4_visible_fraction=0.05,
            config=self.config,
        )
        self.assertEqual(valid, 1)
        self.assertGreater(quality, 0.0)
        invalid, _ = supervised_targets(
            technical_image_valid=True,
            position_error_mm=10.01,
            orientation_error_deg=9.0,
            l4_visible_fraction=0.05,
            config=self.config,
        )
        self.assertEqual(invalid, 0)

    def test_quality_is_zero_for_technical_failure(self) -> None:
        valid, quality = supervised_targets(
            technical_image_valid=False,
            position_error_mm=0.0,
            orientation_error_deg=0.0,
            l4_visible_fraction=0.2,
            config=self.config,
        )
        self.assertEqual(valid, 0)
        self.assertEqual(quality, 0.0)

    def test_quality_decreases_with_geometric_error(self) -> None:
        _, near = supervised_targets(
            technical_image_valid=True,
            position_error_mm=1.0,
            orientation_error_deg=1.0,
            l4_visible_fraction=0.2,
            config=self.config,
        )
        _, far = supervised_targets(
            technical_image_valid=True,
            position_error_mm=9.0,
            orientation_error_deg=9.0,
            l4_visible_fraction=0.2,
            config=self.config,
        )
        self.assertGreater(near, far)


class Stage2SelectionTests(unittest.TestCase):
    def test_quality_cannot_override_invalidity(self) -> None:
        invalid_high_quality = view("invalid", 0.49, 1.0)
        valid_lower_quality = view("valid", 0.9, 0.3)
        selected = select_best_view(
            [invalid_high_quality, valid_lower_quality], threshold=0.5
        )
        self.assertEqual(selected, valid_lower_quality)

    def test_physical_failure_is_ineligible(self) -> None:
        failed = view("failed", 0.99, 0.99, transition_valid=False)
        self.assertIsNone(select_best_view([failed], threshold=0.5))

    def test_fresh_confirmation_rejects_reused_hash(self) -> None:
        remembered = view("best", 0.9, 0.8, digest_character="a")
        reused = view("confirm", 0.9, 0.8, digest_character="a")
        self.assertFalse(
            fresh_confirmation_passes(
                remembered,
                [reused, reused],
                threshold=0.5,
                maximum_quality_drop=0.1,
                required_count=2,
            )
        )

    def test_fresh_confirmation_requires_quality_and_count(self) -> None:
        remembered = view("best", 0.9, 0.8, digest_character="a")
        first = view("c1", 0.9, 0.72, digest_character="b")
        second = view("c2", 0.8, 0.75, digest_character="c")
        self.assertTrue(
            fresh_confirmation_passes(
                remembered,
                [first, second],
                threshold=0.5,
                maximum_quality_drop=0.1,
                required_count=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
