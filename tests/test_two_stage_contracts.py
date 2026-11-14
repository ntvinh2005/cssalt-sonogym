#!/usr/bin/env python3
"""CPU tests for leakage, freshness, coordinate, and odometry contracts."""

from __future__ import annotations

import math
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "source" / "spinal_surgery" / "spinal_surgery" / "evaluation"
)
sys.path.insert(0, str(EVALUATION_ROOT))

from two_stage_navigation.contracts import (  # noqa: E402
    ACTOR_SCHEMA,
    ActorFrame,
    FrameFreshnessGuard,
    PrivilegedAnnotation,
    assert_no_privileged_fields,
    ultrasound_sha256,
    validate_actor_mapping,
)
from two_stage_navigation.coordinates import (  # noqa: E402
    ExecutedMotion,
    HomeRelativeOdometry,
    PlannedNativeAction,
    PatientFrameConvention,
    PixelGeometry,
    assess_native_command,
    command_target_from_realized_error,
    command_with_replaced_angle,
    indices_to_mm,
    mm_to_indices,
    plan_native_action,
    plan_proportional_target_action,
    pose_within_tolerance,
    uses_coupled_stage2_dataset_controller,
    validate_planar_rotation,
    wrap_to_pi,
)


def make_frame(sequence: int, producer_sequence: int | None = None) -> ActorFrame:
    image = np.full((200, 150), 0.25, dtype=np.float32)
    return ActorFrame(
        session_id="td_00",
        frame_sequence=sequence,
        producer_sequence=sequence if producer_sequence is None else producer_sequence,
        frame_sha256=ultrasound_sha256(image),
        ultrasound=image,
        requested_action=np.asarray([0.0, -0.5, 0.0]),
        executed_delta_home_mm=np.asarray([0.0, -1.5]),
        executed_delta_angle_rad=0.0,
        action_accepted=True,
        action_completed=True,
        action_clamped=False,
        contact_ok=True,
        workspace_ok=True,
        image_valid=True,
        home_prior_id="workspace-grid-v1-x132",
    )


class ActorContractTests(unittest.TestCase):
    def test_canonical_hash_and_actor_frame(self) -> None:
        frame = make_frame(0)
        self.assertEqual(len(frame.frame_sha256), 64)
        self.assertTrue(frame.transition_valid)
        self.assertNotIn("patient_id", frame.public_metadata())

    def test_hash_mismatch_and_nonfinite_image_fail(self) -> None:
        frame = make_frame(0)
        with self.assertRaises(ValueError):
            ActorFrame(**{**frame.__dict__, "frame_sha256": "0" * 64})
        image = frame.ultrasound.copy()
        image[0, 0] = np.nan
        with self.assertRaises(ValueError):
            ultrasound_sha256(image)

    def test_forbidden_fields_fail_at_any_nesting_depth(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_position"):
            assert_no_privileged_fields({"history": [{"target_position": [1, 2]}]})
        with self.assertRaisesRegex(ValueError, "reward"):
            assert_no_privileged_fields({"state": {"reward": 1.0}})

    def test_declared_actor_mapping_is_closed(self) -> None:
        frame = make_frame(0)
        payload = {
            "schema": ACTOR_SCHEMA,
            "session_id": frame.session_id,
            "frame_sequence": frame.frame_sequence,
            "producer_sequence": frame.producer_sequence,
            "frame_sha256": frame.frame_sha256,
            "ultrasound": frame.ultrasound,
            "requested_action": frame.requested_action,
            "executed_delta_home_mm": frame.executed_delta_home_mm,
            "executed_delta_angle_rad": frame.executed_delta_angle_rad,
            "action_accepted": frame.action_accepted,
            "action_completed": frame.action_completed,
            "action_clamped": frame.action_clamped,
            "contact_ok": frame.contact_ok,
            "workspace_ok": frame.workspace_ok,
            "image_valid": frame.image_valid,
            "home_prior_id": frame.home_prior_id,
        }
        validate_actor_mapping(payload)
        with self.assertRaisesRegex(ValueError, "extra"):
            validate_actor_mapping({**payload, "debug_value": 1})

    def test_freshness_uses_sequences_not_pixel_difference(self) -> None:
        guard = FrameFreshnessGuard()
        guard.accept(make_frame(0))
        # Deterministic pixels can hash identically while a producer sequence proves recapture.
        guard.accept(make_frame(1))
        with self.assertRaisesRegex(ValueError, "frame_sequence"):
            guard.accept(make_frame(1, 2))

    def test_privileged_annotation_is_separate(self) -> None:
        frame = make_frame(2)
        annotation = PrivilegedAnnotation(
            session_id=frame.session_id,
            frame_sequence=frame.frame_sequence,
            frame_sha256=frame.frame_sha256,
            patient_id="s0010",
            anatomy_id="vertebrae_L4",
            command_pose_index=np.asarray([132.0, 150.0, 3.0]),
            l4_visible_fraction=0.3,
            lumbar_interface_visible=True,
            adjacent_level="L3",
            position_error_mm=12.0,
            orientation_error_deg=4.0,
        )
        self.assertEqual(annotation.join_key, ("td_00", 2, frame.frame_sha256))


class CoordinateTests(unittest.TestCase):
    def test_all_segmented_dataset_protocols_use_coupled_control(self) -> None:
        self.assertTrue(uses_coupled_stage2_dataset_controller("sonogym-stage2-dataset-v4"))
        self.assertTrue(uses_coupled_stage2_dataset_controller("sonogym-stage2-dataset-v5"))
        self.assertTrue(uses_coupled_stage2_dataset_controller("sonogym-stage2-dataset-v6"))
        self.assertFalse(uses_coupled_stage2_dataset_controller("sonogym-stage2-dataset-v3"))
        with self.assertRaises(ValueError):
            uses_coupled_stage2_dataset_controller("")

    def test_pose_tolerance_uses_the_declared_protocol_limits(self) -> None:
        self.assertTrue(pose_within_tolerance(1.906, 3.718, 5.0, 5.0))
        self.assertFalse(pose_within_tolerance(1.906, 3.718, 3.0, 3.0))
        with self.assertRaises(ValueError):
            pose_within_tolerance(1.0, math.nan, 5.0, 5.0)

    def test_index_millimetre_round_trip(self) -> None:
        indices = np.asarray([132.0, 195.0])
        np.testing.assert_allclose(mm_to_indices(indices_to_mm(indices)), indices)

    def test_patient_home_round_trip_and_signs(self) -> None:
        convention = PatientFrameConvention(right_sign=1, cranial_sign=1)
        home = np.asarray([140.0, 195.0])
        current = np.asarray([132.0, 190.0])
        relative = convention.patient_xz_to_home_mm(current, home)
        np.testing.assert_allclose(relative, [-12.0, -7.5])
        np.testing.assert_allclose(convention.home_mm_to_patient_xz(relative, home), current)

    def test_pixel_projection_round_trip(self) -> None:
        geometry = PixelGeometry()
        rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
        origin = np.asarray([12.0, -4.0])
        point = geometry.pixel_to_home_mm(40.0, 80.0, origin, rotation)
        pixel = geometry.home_mm_to_pixel(point, origin, rotation)
        np.testing.assert_allclose(pixel, [40.0, 80.0], atol=1e-9)

    def test_nonrigid_pixel_rotation_fails(self) -> None:
        with self.assertRaises(ValueError):
            validate_planar_rotation([[1.0, 0.0], [0.0, 2.0]])

    def test_angle_wrap(self) -> None:
        self.assertAlmostEqual(float(wrap_to_pi(math.pi)), -math.pi)
        self.assertAlmostEqual(float(wrap_to_pi(3 * math.pi)), -math.pi)

    def test_rejected_and_clamped_motion_do_not_advance(self) -> None:
        odometry = HomeRelativeOdometry()
        accepted = ExecutedMotion(np.asarray([1.5, -3.0]), 0.1, True, True, False, True, True)
        rejected = ExecutedMotion(np.asarray([10.0, 10.0]), 0.3, False, False, False, True, True)
        clamped = ExecutedMotion(np.asarray([2.0, 2.0]), 0.2, True, True, True, True, True)
        self.assertTrue(odometry.apply(accepted))
        self.assertFalse(odometry.apply(rejected))
        self.assertFalse(odometry.apply(clamped))
        np.testing.assert_allclose(odometry.position_mm, [1.5, -3.0])
        self.assertAlmostEqual(odometry.angle_rad, 0.1)

    def test_return_to_best_delta(self) -> None:
        odometry = HomeRelativeOdometry(position_mm=np.asarray([5.0, 8.0]), angle_rad=0.2)
        delta_position, delta_angle = odometry.delta_to([2.0, 3.0], -0.1)
        np.testing.assert_allclose(delta_position, [-3.0, -5.0])
        self.assertAlmostEqual(delta_angle, -0.3)

    def test_native_action_inverse_mapping_and_bounds(self) -> None:
        angle = math.radians(35.0)
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [0.0, 0.0, 1.0],
                [math.sin(angle), math.cos(angle), 0.0],
            ]
        )
        small = plan_native_action([0.0, -1.0], 0.05, rotation)
        np.testing.assert_allclose(
            small.predicted_patient_delta_index, [0.0, -1.0], atol=1e-12
        )
        self.assertAlmostEqual(small.predicted_angle_delta_rad, 0.05)
        bounded = plan_native_action([20.0, -20.0], 1.0, rotation)
        applied = bounded.normalized_action * np.asarray([2.0, 2.0, 0.1])
        self.assertTrue(np.all(np.abs(applied) <= np.asarray([4.0, 4.0, 0.2]) + 1e-12))

    def test_proportional_planner_matches_coupled_expert_math(self) -> None:
        rotation = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        planned = plan_proportional_target_action(
            current_pose=[10.0, 20.0, 2.0],
            target_pose=[14.0, 15.0, 3.14],
            human_to_probe_rotation=rotation,
        )
        np.testing.assert_allclose(planned.normalized_action, [0.08, -0.10, 0.10])
        np.testing.assert_allclose(planned.predicted_patient_delta_index, [0.16, -0.20])
        self.assertAlmostEqual(planned.predicted_angle_delta_rad, 0.01)

    def test_proportional_planner_has_no_evaluator_goal_interface(self) -> None:
        parameters = inspect.signature(plan_proportional_target_action).parameters
        self.assertIn("target_pose", parameters)
        self.assertNotIn("goal_cmd_pose", parameters)

    def test_command_assessment_separates_mapping_error_from_workspace_clamp(self) -> None:
        planned = PlannedNativeAction(
            normalized_action=np.asarray([1.0, 0.0, 0.0]),
            predicted_patient_delta_index=np.asarray([-4.0, 2.0]),
            predicted_angle_delta_rad=-0.1,
        )
        assessment = assess_native_command(
            before_command=[165.0, 179.0, 2.8],
            after_command=[161.06, 181.04, 2.7],
            planned=planned,
            target_command=[132.0, 195.0, 2.7],
            command_minimum=[100.0, 70.0, 1.0],
            command_maximum=[180.0, 200.0, 3.0],
        )
        self.assertTrue(assessment.mapping_valid)
        self.assertFalse(assessment.native_clamped)
        self.assertFalse(assessment.position_clamped)
        self.assertFalse(assessment.angle_clamped)
        self.assertTrue(assessment.progressed_toward_target)
        wider = assess_native_command(
            before_command=[165.0, 179.0, 2.8],
            after_command=[161.12, 181.0, 2.7],
            planned=planned,
            target_command=[132.0, 195.0, 2.7],
            command_minimum=[100.0, 70.0, 1.0],
            command_maximum=[180.0, 200.0, 3.0],
            position_mapping_tolerance_index=0.15,
        )
        self.assertTrue(wider.mapping_valid)
        narrow = assess_native_command(
            before_command=[165.0, 179.0, 2.8],
            after_command=[161.12, 181.0, 2.7],
            planned=planned,
            target_command=[132.0, 195.0, 2.7],
            command_minimum=[100.0, 70.0, 1.0],
            command_maximum=[180.0, 200.0, 3.0],
            position_mapping_tolerance_index=0.10,
        )
        self.assertFalse(narrow.mapping_valid)

        clamped = assess_native_command(
            before_command=[101.0, 100.0, 2.0],
            after_command=[100.0, 100.0, 2.0],
            planned=PlannedNativeAction(
                normalized_action=np.asarray([-1.0, 0.0, 0.0]),
                predicted_patient_delta_index=np.asarray([-4.0, 0.0]),
                predicted_angle_delta_rad=0.0,
            ),
            target_command=[100.0, 100.0, 2.0],
            command_minimum=[100.0, 70.0, 1.0],
            command_maximum=[180.0, 200.0, 3.0],
        )
        self.assertTrue(clamped.native_clamped)
        self.assertTrue(clamped.position_clamped)
        self.assertFalse(clamped.angle_clamped)
        self.assertTrue(clamped.mapping_valid)

        angle_clamped = assess_native_command(
            before_command=[132.0, 139.0, 2.995],
            after_command=[132.0, 139.0, 3.0],
            planned=PlannedNativeAction(
                normalized_action=np.asarray([0.0, 0.0, 0.1]),
                predicted_patient_delta_index=np.asarray([0.0, 0.0]),
                predicted_angle_delta_rad=0.01,
            ),
            target_command=[132.0, 139.0, 3.14],
            command_minimum=[100.0, 70.0, 1.0],
            command_maximum=[180.0, 200.0, 3.0],
        )
        self.assertTrue(angle_clamped.native_clamped)
        self.assertFalse(angle_clamped.position_clamped)
        self.assertTrue(angle_clamped.angle_clamped)
        self.assertTrue(angle_clamped.mapping_valid)

    def test_command_target_uses_realized_pose_error(self) -> None:
        corrected = command_target_from_realized_error(
            realized_pose=[131.688, 142.354, 2.7063],
            target_pose=[132.0, 139.0, 2.7],
        )
        np.testing.assert_allclose(corrected, [132.624, 132.292, 2.6937], atol=1e-12)
        stronger = command_target_from_realized_error(
            realized_pose=[131.688, 142.354, 2.7063],
            target_pose=[132.0, 139.0, 2.7],
            position_multiplier=2.5,
        )
        np.testing.assert_allclose(stronger, [132.78, 130.615, 2.6937], atol=1e-12)
        with self.assertRaises(ValueError):
            command_target_from_realized_error(
                realized_pose=[132.0, 139.0, 2.7],
                target_pose=[132.0, 139.0, 2.7],
                position_multiplier=0.0,
            )

    def test_orientation_command_does_not_request_translation(self) -> None:
        command = command_with_replaced_angle([133.1, 145.9, 2.17], 3.0)
        np.testing.assert_allclose(command, [133.1, 145.9, 3.0])
        with self.assertRaises(ValueError):
            command_with_replaced_angle([133.1, 145.9, 2.17], math.nan)


if __name__ == "__main__":
    unittest.main()
