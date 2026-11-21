from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "workflows/two_stage_navigation/build_s0010_ultrasound_representation_review.py"
)
SPEC = importlib.util.spec_from_file_location("representation_review", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class RepresentationReviewTests(unittest.TestCase):
    def test_public_frame_lookup_uses_sequence_identity(self) -> None:
        rows = [
            {"frame_sequence": 8, "image_archive_index": 0},
            {"frame_sequence": 3, "image_archive_index": 1},
        ]
        self.assertEqual(review.public_frame_by_sequence(rows, 3)["image_archive_index"], 1)
        with self.assertRaises(ValueError):
            review.public_frame_by_sequence(rows, 4)

    def test_rendered_panels_have_expected_shape(self) -> None:
        image = np.linspace(0.0, 1.0, 150 * 200, dtype=np.float32).reshape(200, 150)
        pseudo = np.zeros_like(image, dtype=bool)
        pseudo[80:85, 70:75] = True
        first_hit = np.zeros_like(image, dtype=bool)
        first_hit[80, 70:75] = True
        fig = review.render_frame(
            image=image,
            public={
                "executed_delta_relative_mm": [1.0, -2.0],
                "executed_delta_angle_rad": 0.1,
            },
            score=0.7,
            threshold=0.55,
            status="AMBIGUOUS",
            pseudo_projection=pseudo,
            first_hit_surface=first_hit,
            title="synthetic frame",
        )
        try:
            rendered = review.figure_array(fig)
        finally:
            review.plt.close(fig)
        self.assertEqual(rendered.shape, (450, 1200, 3))
        self.assertEqual(rendered.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
