from __future__ import annotations

import inspect
import unittest

import numpy as np

from workflows.two_stage_navigation.build_s0010_mbb_stage2_review import (
    evaluator_panel,
    map_acquisitions_to_robot_frames,
    render_actor_panels,
)


class MbbStage2ReviewTests(unittest.TestCase):
    def test_action_history_maps_exactly_to_recorded_robot_frames(self) -> None:
        actions = [
            {"phase": "acquire_a", "after_realized_pose": [0.0, 0.0, 0.0]},
            {"phase": "acquire_a", "after_realized_pose": [1.0, 2.0, 0.1]},
            {"phase": "dither", "after_realized_pose": [2.0, 2.0, 0.1]},
            {"phase": "acquire_b", "after_realized_pose": [1.5, 2.0, 0.1]},
            {"phase": "acquire_b", "after_realized_pose": [1.2, 2.5, 0.2]},
            {"phase": "acquire_b", "after_realized_pose": [1.0, 3.0, 0.3]},
        ]
        frames = [
            {
                "phase": "acquire_a",
                "action_step_count": 2,
                "executed_relative_pose_index": [1.0, 2.0, 0.1],
            },
            {
                "phase": "acquire_b",
                "action_step_count": 3,
                "executed_relative_pose_index": [1.0, 3.0, 0.3],
            },
        ]
        self.assertEqual(
            map_acquisitions_to_robot_frames(
                frames, actions, action_stride=2, robot_frame_count=4
            ),
            [1, 3],
        )

    def test_action_mapping_rejects_pose_mismatch(self) -> None:
        frames = [
            {
                "phase": "acquire",
                "action_step_count": 1,
                "executed_relative_pose_index": [1.0, 2.0, 0.1],
            }
        ]
        actions = [
            {"phase": "acquire", "after_realized_pose": [1.0, 2.5, 0.1]}
        ]
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            map_acquisitions_to_robot_frames(
                frames, actions, action_stride=2, robot_frame_count=2
            )

    def test_actor_panel_renderer_has_no_evaluator_input(self) -> None:
        parameters = set(inspect.signature(render_actor_panels).parameters)
        self.assertEqual(
            parameters,
            {
                "robot_frame",
                "image",
                "frame",
                "observations",
                "target_actor",
                "previous_frame",
                "previous_image",
            },
        )
        self.assertFalse(any("evaluator" in value for value in parameters))

    def test_blinded_evaluator_panel_does_not_reveal_evaluator_values(self) -> None:
        image = np.arange(200 * 150, dtype=np.float32).reshape(200, 150)
        frame = {"frame_sequence": 0}
        first = evaluator_panel(
            image,
            np.ones((200, 150), dtype=np.uint8),
            frame,
            {"integrated_target_success": True},
            {"proxy_state": "PROXY_POSITIVE"},
            blinded=True,
        )
        second = evaluator_panel(
            image,
            np.full((200, 150), 2, dtype=np.uint8),
            frame,
            {"integrated_target_success": False},
            {"proxy_state": "PROXY_NEGATIVE"},
            blinded=True,
        )
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
