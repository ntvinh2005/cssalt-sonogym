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
WORKFLOW_ROOT = REPOSITORY_ROOT / "workflows" / "two_stage_navigation"
sys.path.insert(0, str(EVALUATION_ROOT))
sys.path.insert(0, str(WORKFLOW_ROOT))
PROTOCOL = (
    REPOSITORY_ROOT
    / "experiments"
    / "unity_two_stage_transfer"
    / "stage2_dataset_protocol_v1.yaml"
)
PROTOCOL_V2 = PROTOCOL.with_name("stage2_dataset_protocol_v2.yaml")
PROTOCOL_V3 = PROTOCOL.with_name("stage2_dataset_protocol_v3.yaml")
PROTOCOL_V4 = PROTOCOL.with_name("stage2_dataset_protocol_v4.yaml")
PROTOCOL_V5 = PROTOCOL.with_name("stage2_dataset_protocol_v5.yaml")
PROTOCOL_V6 = PROTOCOL.with_name("stage2_dataset_protocol_v6.yaml")
PROTOCOL_V7 = PROTOCOL.with_name("stage2_dataset_protocol_v7.yaml")
PROTOCOL_V8 = PROTOCOL.with_name("stage2_dataset_protocol_v8.yaml")
PROTOCOL_V9 = PROTOCOL.with_name("stage2_dataset_protocol_v9.yaml")
PROTOCOL_V10 = PROTOCOL.with_name("stage2_dataset_protocol_v10.yaml")
PROTOCOL_V11 = PROTOCOL.with_name("stage2_dataset_protocol_v11.yaml")
PROTOCOL_V12 = PROTOCOL.with_name("stage2_dataset_protocol_v12.yaml")
PROTOCOL_V13 = PROTOCOL.with_name("stage2_dataset_protocol_v13.yaml")
PROTOCOL_V14 = PROTOCOL.with_name("stage2_dataset_protocol_v14.yaml")
PROTOCOL_V15 = PROTOCOL.with_name("stage2_dataset_protocol_v15.yaml")
CONFIRMATION_V16 = PROTOCOL.with_name("stage2_confirmation_protocol_v16.yaml")
CALIBRATION_V17 = PROTOCOL.with_name("stage2_threshold_calibration_protocol_v17.yaml")
CONFIRMATION_V17 = PROTOCOL.with_name("stage2_confirmation_protocol_v17.yaml")

from two_stage_navigation.stage2_features import (  # noqa: E402
    FEATURE_WIDTH,
    extract_stage2_feature,
    normalize_ultrasound,
)
from two_stage_navigation.stage2_models import (  # noqa: E402
    ScoredView,
    Stage2ImageValidityConfig,
    Stage2LabelConfig,
    fresh_confirmation_passes,
    image_validity_target,
    select_best_view,
    supervised_targets,
)
from train_stage2 import PROTOCOL_IDS as TRAIN_STAGE2_PROTOCOL_IDS  # noqa: E402
from evaluate_stage2_confirmation import confirmation_gate_passes  # noqa: E402
from summarize_stage2_dataset import dataset_gate_passes  # noqa: E402


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

    def test_v3_is_the_final_bounded_convergence_retry(self) -> None:
        expected = PROTOCOL_V3.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V3.read_bytes()).hexdigest(), expected)
        second = yaml.safe_load(PROTOCOL_V2.read_text(encoding="utf-8"))
        third = yaml.safe_load(PROTOCOL_V3.read_text(encoding="utf-8"))
        self.assertEqual(third["amendment"]["evidence_job"], 51245393)
        self.assertIn("final retry", third["amendment"]["retry_policy"])
        self.assertEqual(second["sampling"]["maximum_action_steps_per_waypoint"], 16)
        self.assertEqual(third["sampling"]["maximum_action_steps_per_waypoint"], 24)
        for key in (
            "sessions",
            "labels",
            "features",
            "model",
            "development_gate",
        ):
            self.assertEqual(second[key], third[key])

    def test_v4_preserves_data_design_and_changes_only_physical_control(self) -> None:
        expected = PROTOCOL_V4.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V4.read_bytes()).hexdigest(), expected)
        third = yaml.safe_load(PROTOCOL_V3.read_text(encoding="utf-8"))
        fourth = yaml.safe_load(PROTOCOL_V4.read_text(encoding="utf-8"))
        self.assertEqual(fourth["amendment"]["evidence_jobs"], [51245827, 51334984])
        for key in ("sessions", "labels", "features", "model", "development_gate"):
            self.assertEqual(third[key], fourth[key])
        for key in (
            "goal_relative_x_offsets_index",
            "goal_relative_z_offsets_index",
            "command_angles_rad",
            "expected_frames_per_session",
            "maximum_total_action_steps_per_session",
            "physical_tracking_position_tolerance_mm",
            "physical_tracking_orientation_tolerance_deg",
        ):
            self.assertEqual(third["sampling"][key], fourth["sampling"][key])
        self.assertEqual(
            fourth["sampling"]["physical_controller"]["ratio"], [0.02, 0.02, 0.6]
        )
        self.assertEqual(fourth["sampling"]["maximum_initial_positioning_steps"], 300)
        self.assertEqual(fourth["sampling"]["maximum_action_steps_per_waypoint"], 60)

    def test_v5_segments_capture_without_disabling_task_timeout(self) -> None:
        expected = PROTOCOL_V5.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V5.read_bytes()).hexdigest(), expected)
        fourth = yaml.safe_load(PROTOCOL_V4.read_text(encoding="utf-8"))
        fifth = yaml.safe_load(PROTOCOL_V5.read_text(encoding="utf-8"))
        self.assertEqual(fifth["amendment"]["evidence_job"], 51335739)
        for key in ("sessions", "labels", "features", "model", "development_gate"):
            self.assertEqual(fourth[key], fifth[key])
        self.assertEqual(
            fourth["sampling"]["physical_controller"],
            fifth["sampling"]["physical_controller"],
        )
        self.assertEqual(fifth["sampling"]["trajectory_start_frames"], [0, 20, 40, 60])
        self.assertEqual(fifth["sampling"]["expected_trajectories_per_session"], 4)
        self.assertEqual(fifth["sampling"]["expected_total_trajectories"], 24)
        self.assertEqual(fifth["sampling"]["maximum_total_action_steps_per_session"], 1800)
        self.assertIn("task_timeout_and_dynamics", fifth["amendment"]["unchanged_design"])

    def test_v6_only_repairs_protocol_dispatch(self) -> None:
        expected = PROTOCOL_V6.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V6.read_bytes()).hexdigest(), expected)
        fifth = yaml.safe_load(PROTOCOL_V5.read_text(encoding="utf-8"))
        sixth = yaml.safe_load(PROTOCOL_V6.read_text(encoding="utf-8"))
        self.assertEqual(sixth["amendment"]["evidence_job"], 51337309)
        for key in ("sessions", "labels", "features", "model", "development_gate", "sampling"):
            self.assertEqual(fifth[key], sixth[key])
        for key in ("account", "gpu_count", "cpu_cores", "memory", "walltime"):
            self.assertEqual(fifth["resource_bound"][key], sixth["resource_bound"][key])
        self.assertIn("all_action_bounds", sixth["amendment"]["unchanged_design"])

    def test_v7_only_repairs_the_protocol_registry(self) -> None:
        expected = PROTOCOL_V7.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V7.read_bytes()).hexdigest(), expected)
        sixth = yaml.safe_load(PROTOCOL_V6.read_text(encoding="utf-8"))
        seventh = yaml.safe_load(PROTOCOL_V7.read_text(encoding="utf-8"))
        self.assertEqual(seventh["amendment"]["evidence_job"], 51337864)
        for key in ("sessions", "labels", "features", "model", "development_gate", "sampling"):
            self.assertEqual(sixth[key], seventh[key])
        for key in ("account", "gpu_count", "cpu_cores", "memory", "walltime"):
            self.assertEqual(sixth["resource_bound"][key], seventh["resource_bound"][key])
        self.assertIn("trajectory_boundaries_and_seeds", seventh["amendment"]["unchanged_design"])

    def test_v8_only_repairs_segmented_capture_dispatch(self) -> None:
        expected = PROTOCOL_V8.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V8.read_bytes()).hexdigest(), expected)
        seventh = yaml.safe_load(PROTOCOL_V7.read_text(encoding="utf-8"))
        eighth = yaml.safe_load(PROTOCOL_V8.read_text(encoding="utf-8"))
        self.assertEqual(eighth["amendment"]["evidence_job"], 51339010)
        for key in ("sessions", "labels", "features", "model", "development_gate", "sampling"):
            self.assertEqual(seventh[key], eighth[key])
        for key in ("account", "gpu_count", "cpu_cores", "memory", "walltime"):
            self.assertEqual(seventh["resource_bound"][key], eighth["resource_bound"][key])
        self.assertIn("task_timeout_and_dynamics", eighth["amendment"]["unchanged_design"])

    def test_v9_applies_the_existing_initial_bound_to_each_segment(self) -> None:
        expected = PROTOCOL_V9.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V9.read_bytes()).hexdigest(), expected)
        eighth = yaml.safe_load(PROTOCOL_V8.read_text(encoding="utf-8"))
        ninth = yaml.safe_load(PROTOCOL_V9.read_text(encoding="utf-8"))
        self.assertEqual(ninth["amendment"]["evidence_job"], 51340679)
        for key in ("sessions", "labels", "features", "model", "development_gate", "sampling"):
            self.assertEqual(eighth[key], ninth[key])
        for key in ("account", "gpu_count", "cpu_cores", "memory", "walltime"):
            self.assertEqual(eighth["resource_bound"][key], ninth["resource_bound"][key])
        self.assertIn("initial_and_ordinary_action_bounds", ninth["amendment"]["unchanged_design"])

    def test_v10_settles_high_angle_and_extends_only_ordinary_motion_bound(self) -> None:
        expected = PROTOCOL_V10.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V10.read_bytes()).hexdigest(), expected)
        ninth = yaml.safe_load(PROTOCOL_V9.read_text(encoding="utf-8"))
        tenth = yaml.safe_load(PROTOCOL_V10.read_text(encoding="utf-8"))
        self.assertEqual(tenth["amendment"]["evidence_job"], 51341988)
        for key in ("sessions", "labels", "features", "model", "development_gate"):
            self.assertEqual(ninth[key], tenth[key])
        for key in (
            "goal_relative_x_offsets_index",
            "goal_relative_z_offsets_index",
            "command_angles_rad",
            "expected_frames_per_session",
            "trajectory_start_frames",
            "trajectory_seed_stride",
            "expected_trajectories_per_session",
            "expected_total_trajectories",
            "maximum_initial_positioning_steps",
            "maximum_total_action_steps_per_session",
            "physical_tracking_position_tolerance_mm",
            "physical_tracking_orientation_tolerance_deg",
        ):
            self.assertEqual(ninth["sampling"][key], tenth["sampling"][key])
        self.assertEqual(ninth["sampling"]["maximum_action_steps_per_waypoint"], 60)
        self.assertEqual(tenth["sampling"]["maximum_action_steps_per_waypoint"], 90)
        self.assertTrue(
            tenth["sampling"]["physical_controller"][
                "accept_against_evaluator_goal_angle"
            ]
        )

    def test_v11_records_angle_ceiling_as_invalid_candidate_without_changing_labels(self) -> None:
        expected = PROTOCOL_V11.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V11.read_bytes()).hexdigest(), expected)
        tenth = yaml.safe_load(PROTOCOL_V10.read_text(encoding="utf-8"))
        eleventh = yaml.safe_load(PROTOCOL_V11.read_text(encoding="utf-8"))
        self.assertEqual(eleventh["amendment"]["evidence_job"], 51343674)
        for key in ("sessions", "labels", "features", "model", "development_gate"):
            self.assertEqual(tenth[key], eleventh[key])
        for key in (
            "goal_relative_x_offsets_index",
            "goal_relative_z_offsets_index",
            "command_angles_rad",
            "expected_frames_per_session",
            "trajectory_start_frames",
            "trajectory_seed_stride",
            "expected_trajectories_per_session",
            "expected_total_trajectories",
            "maximum_initial_positioning_steps",
            "maximum_action_steps_per_waypoint",
            "maximum_total_action_steps_per_session",
            "physical_tracking_position_tolerance_mm",
            "physical_tracking_orientation_tolerance_deg",
        ):
            self.assertEqual(tenth["sampling"][key], eleventh["sampling"][key])
        self.assertTrue(
            eleventh["sampling"]["physical_controller"][
                "capture_angle_ceiling_limited_samples"
            ]
        )

    def test_v12_segments_same_grid_before_native_timeout(self) -> None:
        expected = PROTOCOL_V12.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V12.read_bytes()).hexdigest(), expected)
        eleventh = yaml.safe_load(PROTOCOL_V11.read_text(encoding="utf-8"))
        twelfth = yaml.safe_load(PROTOCOL_V12.read_text(encoding="utf-8"))
        self.assertEqual(twelfth["amendment"]["evidence_job"], 51344965)
        for key in ("sessions", "labels", "features", "model", "development_gate"):
            self.assertEqual(eleventh[key], twelfth[key])
        for key in (
            "goal_relative_x_offsets_index",
            "goal_relative_z_offsets_index",
            "command_angles_rad",
            "expected_frames_per_session",
            "physical_controller",
            "maximum_initial_positioning_steps",
            "maximum_action_steps_per_waypoint",
            "physical_tracking_position_tolerance_mm",
            "physical_tracking_orientation_tolerance_deg",
        ):
            self.assertEqual(eleventh["sampling"][key], twelfth["sampling"][key])
        self.assertEqual(
            twelfth["sampling"]["trajectory_start_frames"],
            [0, 10, 20, 30, 40, 50, 60, 70],
        )
        self.assertEqual(twelfth["sampling"]["expected_total_trajectories"], 48)
        self.assertEqual(twelfth["sampling"]["maximum_total_action_steps_per_session"], 4000)
        self.assertIn("sonogym-stage2-dataset-v12", TRAIN_STAGE2_PROTOCOL_IDS)

    def test_v13_focuses_positive_coverage_without_changing_labels_or_models(self) -> None:
        expected = PROTOCOL_V13.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V13.read_bytes()).hexdigest(), expected)
        twelfth = yaml.safe_load(PROTOCOL_V12.read_text(encoding="utf-8"))
        thirteenth = yaml.safe_load(PROTOCOL_V13.read_text(encoding="utf-8"))
        self.assertEqual(thirteenth["amendment"]["evidence_job"], 51367445)
        for key in ("labels", "features", "model", "development_gate"):
            self.assertEqual(twelfth[key], thirteenth[key])
        self.assertEqual(
            thirteenth["sampling"]["goal_relative_x_offsets_index"],
            [-4.0, -2.0, 0.0, 2.0, 4.0],
        )
        self.assertEqual(
            thirteenth["sampling"]["goal_relative_z_offsets_index"],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        )
        self.assertEqual(
            thirteenth["sampling"]["command_angles_rad"], [2.9, 2.95, 3.0]
        )
        self.assertEqual(
            thirteenth["dataset_gate"]["minimum_unique_positive_hashes"],
            {"train": 60, "calibration": 20, "development_evaluation": 20},
        )
        assisted_minimum = thirteenth["sampling"]["physical_controller"][
            "ceiling_assisted_minimum_requested_angle_rad"
        ]
        for session in thirteenth["sessions"]:
            self.assertGreaterEqual(
                max(thirteenth["sampling"]["command_angles_rad"])
                + session["angle_jitter_rad"],
                assisted_minimum,
            )
            self.assertLess(
                sorted(thirteenth["sampling"]["command_angles_rad"])[-2]
                + session["angle_jitter_rad"],
                assisted_minimum,
            )
            maximum_radius_index = max(
                np.hypot(
                    x + session["x_jitter_index"],
                    z + session["z_jitter_index"],
                )
                for x in thirteenth["sampling"]["goal_relative_x_offsets_index"]
                for z in thirteenth["sampling"]["goal_relative_z_offsets_index"]
            )
            self.assertLess(maximum_radius_index * 1.5, 10.0)
        self.assertIn("sonogym-stage2-dataset-v13", TRAIN_STAGE2_PROTOCOL_IDS)

    def test_v13_dataset_gate_requires_hash_and_trajectory_coverage(self) -> None:
        gate = yaml.safe_load(PROTOCOL_V13.read_text(encoding="utf-8"))["dataset_gate"]
        hashes = {
            "train": {f"t{index}" for index in range(60)},
            "calibration": {f"c{index}" for index in range(20)},
            "development_evaluation": {f"d{index}" for index in range(20)},
        }
        trajectories = {
            "train": {f"t{index}" for index in range(8)},
            "calibration": {f"c{index}" for index in range(4)},
            "development_evaluation": {f"d{index}" for index in range(4)},
        }
        self.assertTrue(dataset_gate_passes(gate, hashes, trajectories))
        hashes["development_evaluation"].pop()
        self.assertFalse(dataset_gate_passes(gate, hashes, trajectories))

    def test_v14_changes_only_the_per_session_distinct_image_precondition(self) -> None:
        expected = PROTOCOL_V14.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V14.read_bytes()).hexdigest(), expected)
        thirteenth = yaml.safe_load(PROTOCOL_V13.read_text(encoding="utf-8"))
        fourteenth = yaml.safe_load(PROTOCOL_V14.read_text(encoding="utf-8"))
        self.assertEqual(fourteenth["amendment"]["evidence_job"], 51374942)
        for key in (
            "scope",
            "information_boundaries",
            "labels",
            "features",
            "model",
            "development_gate",
            "sampling",
            "sessions",
        ):
            self.assertEqual(thirteenth[key], fourteenth[key])
        self.assertEqual(
            fourteenth["dataset_gate"]["minimum_distinct_frame_hashes_per_session"],
            20,
        )
        for key in ("minimum_unique_positive_hashes", "minimum_positive_trajectories"):
            self.assertEqual(thirteenth["dataset_gate"][key], fourteenth["dataset_gate"][key])
        self.assertIn("sonogym-stage2-dataset-v14", TRAIN_STAGE2_PROTOCOL_IDS)

    def test_v15_changes_classifier_target_without_changing_evaluator_success(self) -> None:
        expected = PROTOCOL_V15.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(PROTOCOL_V15.read_bytes()).hexdigest(), expected)
        fourteenth = yaml.safe_load(PROTOCOL_V14.read_text(encoding="utf-8"))
        fifteenth = yaml.safe_load(PROTOCOL_V15.read_text(encoding="utf-8"))
        self.assertEqual(fifteenth["amendment"]["evidence_job"], 51378486)
        for key in ("scope", "information_boundaries", "labels", "features", "sampling", "sessions"):
            self.assertEqual(fourteenth[key], fifteenth[key])
        self.assertEqual(
            fifteenth["model"]["classifier_target"],
            {
                "name": "l4_image_visibility",
                "minimum_l4_visible_fraction": 0.20,
                "training_ambiguous_half_width": 0.02,
            },
        )
        self.assertIn("sonogym-stage2-dataset-v15", TRAIN_STAGE2_PROTOCOL_IDS)

    def test_v16_freezes_new_sessions_and_the_v15_bundle(self) -> None:
        expected = CONFIRMATION_V16.with_suffix(".sha256").read_text(
            encoding="utf-8"
        ).split()[0]
        self.assertEqual(hashlib.sha256(CONFIRMATION_V16.read_bytes()).hexdigest(), expected)
        protocol = yaml.safe_load(CONFIRMATION_V16.read_text(encoding="utf-8"))
        self.assertEqual(protocol["protocol_id"], "sonogym-stage2-confirmation-v16")
        self.assertEqual(len(protocol["sessions"]), 4)
        self.assertEqual({row["split"] for row in protocol["sessions"]}, {"confirmation"})
        self.assertEqual(len({row["seed"] for row in protocol["sessions"]}), 4)
        self.assertEqual(protocol["sampling"]["expected_total_trajectories"], 32)
        frozen = protocol["model_confirmation"]
        self.assertEqual(
            frozen["model_bundle_sha256"],
            "7d3b113ba33b1b80448264a0d2e7f2c2cd5561b26021707648c4eef43d586672",
        )
        self.assertTrue(frozen["no_retraining"])
        self.assertTrue(frozen["require_zero_reference_image_hash_overlap"])
        self.assertEqual(frozen["validity_probability_threshold"], 0.6833333333333333)

    def test_v17_threshold_calibration_is_multisession_and_model_frozen(self) -> None:
        expected = CALIBRATION_V17.with_suffix(".sha256").read_text(
            encoding="utf-8"
        ).split()[0]
        self.assertEqual(hashlib.sha256(CALIBRATION_V17.read_bytes()).hexdigest(), expected)
        protocol = yaml.safe_load(CALIBRATION_V17.read_text(encoding="utf-8"))
        self.assertEqual(
            protocol["protocol_id"], "sonogym-stage2-threshold-calibration-v17"
        )
        self.assertEqual(
            sum(len(source["session_ids"]) for source in protocol["calibration_sources"]),
            6,
        )
        self.assertEqual(len(protocol["calibration_sources"]), 2)
        self.assertEqual(protocol["selection_rule"]["minimum_pooled_precision"], 0.95)
        self.assertEqual(protocol["selection_rule"]["minimum_unique_hash_precision"], 0.95)
        self.assertEqual(protocol["selection_rule"]["minimum_session_recall"], 0.70)
        self.assertEqual(protocol["selection_rule"]["minimum_median_session_recall"], 0.80)
        self.assertEqual(protocol["frozen_model"]["retraining"], "forbidden")

    def test_v17_confirmation_freezes_recalibration_and_new_sessions(self) -> None:
        expected = CONFIRMATION_V17.with_suffix(".sha256").read_text(
            encoding="utf-8"
        ).split()[0]
        self.assertEqual(hashlib.sha256(CONFIRMATION_V17.read_bytes()).hexdigest(), expected)
        protocol = yaml.safe_load(CONFIRMATION_V17.read_text(encoding="utf-8"))
        self.assertEqual(protocol["protocol_id"], "sonogym-stage2-confirmation-v17")
        self.assertEqual(len(protocol["sessions"]), 4)
        self.assertEqual(len({row["seed"] for row in protocol["sessions"]}), 4)
        self.assertTrue(
            all(str(row["session_id"]).startswith("s2v17cf_") for row in protocol["sessions"])
        )
        frozen = protocol["model_confirmation"]
        self.assertEqual(frozen["validity_probability_threshold"], 0.40)
        self.assertEqual(len(frozen["reference_capture_roots"]), 2)
        self.assertFalse(frozen["threshold_calibration"]["model_retrained"])
        self.assertEqual(
            frozen["threshold_calibration"]["report_sha256"],
            "aebc0870cecb1872a9d18cd80525241f260f49e93f2d9978f9ad0f6b86f15a1a",
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

    def test_image_validity_uses_visibility_not_privileged_pose_error(self) -> None:
        config = Stage2ImageValidityConfig(
            minimum_l4_visible_fraction=0.20,
            training_ambiguous_half_width=0.02,
        )
        self.assertEqual(
            image_validity_target(
                technical_image_valid=True,
                l4_visible_fraction=0.25,
                config=config,
            ),
            (1, True),
        )
        self.assertEqual(
            image_validity_target(
                technical_image_valid=True,
                l4_visible_fraction=0.19,
                config=config,
            ),
            (0, False),
        )
        self.assertEqual(
            image_validity_target(
                technical_image_valid=False,
                l4_visible_fraction=0.25,
                config=config,
            ),
            (0, True),
        )


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

    def test_confirmation_gate_requires_every_frozen_condition(self) -> None:
        gate = {
            "minimum_validity_precision": 0.95,
            "minimum_validity_recall": 0.80,
            "minimum_quality_spearman": 0.30,
        }
        metrics = {"precision": 0.96, "recall": 0.81}
        self.assertTrue(confirmation_gate_passes(metrics, 0.31, 0.01, 0.02, gate))
        self.assertFalse(confirmation_gate_passes(metrics, 0.31, 0.02, 0.02, gate))
        self.assertFalse(
            confirmation_gate_passes(
                {"precision": 1.0, "recall": 0.79}, 0.31, 0.01, 0.02, gate
            )
        )


if __name__ == "__main__":
    unittest.main()
