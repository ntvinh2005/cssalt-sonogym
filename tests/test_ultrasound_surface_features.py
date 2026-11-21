"""Tests for actor-visible full-image ultrasound features."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from two_stage_navigation.ultrasound_surface_features import (
    DEFAULT_SURFACE_FEATURE_SPEC,
    causal_feature,
    single_view_feature,
    topology_feature,
)


def synthetic_image(offset: int = 0) -> np.ndarray:
    rows = np.arange(200, dtype=np.float64)[:, None]
    columns = np.arange(150, dtype=np.float64)[None, :]
    first = np.exp(-0.5 * ((rows - (118 + offset + 0.04 * columns)) / 2.0) ** 2)
    second = 0.75 * np.exp(
        -0.5 * ((rows - (151 + offset - 0.03 * columns)) / 2.5) ** 2
    )
    shadow = 0.12 * (rows > 123 + offset)
    return first + second - shadow


class UltrasoundSurfaceFeatureTests(unittest.TestCase):
    def test_feature_widths_are_finite(self) -> None:
        image = synthetic_image()
        spec = DEFAULT_SURFACE_FEATURE_SPEC
        topology = topology_feature(image)
        single = single_view_feature(image)
        causal = causal_feature(image, None, [0.0, 0.0], 0.0, 0)
        self.assertEqual(topology.shape, (spec.topology_width,))
        self.assertEqual(single.shape, (spec.single_view_width,))
        self.assertEqual(causal.shape, (spec.causal_width,))
        self.assertTrue(np.all(np.isfinite(causal)))

    def test_normalization_removes_affine_intensity_change(self) -> None:
        image = synthetic_image()
        first = single_view_feature(image)
        second = single_view_feature(3.5 * image + 7.0)
        np.testing.assert_allclose(first, second, atol=1e-10)

    def test_previous_observation_changes_only_causal_representation(self) -> None:
        image = synthetic_image()
        previous = synthetic_image(offset=-5)
        without_history = causal_feature(image, None, [0.0, 0.0], 0.0, 0)
        with_history = causal_feature(image, previous, [2.0, -1.0], 0.04, 12)
        self.assertFalse(np.array_equal(without_history, with_history))
        self.assertEqual(without_history[-5], 0.0)
        self.assertEqual(with_history[-5], 1.0)

    def test_feature_interface_accepts_no_privileged_geometry(self) -> None:
        parameters = set(inspect.signature(causal_feature).parameters)
        self.assertEqual(
            parameters,
            {
                "image",
                "previous_image",
                "executed_delta_relative_mm",
                "executed_delta_angle_rad",
                "action_steps_since_previous_frame",
                "spec",
                "tracker_config",
            },
        )


if __name__ == "__main__":
    unittest.main()
