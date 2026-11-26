"""Retain visible U-Net interfaces using actor Stage 1 and calibrated probe poses."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from two_stage_navigation.simple_landmark_mapping import (
    CLASSES, LEVELS, Stage1Region, extract_observations, project_image, sap_level_at,
)
from two_stage_navigation.sonoanatomy_labels import image_cortical_support

from .public_geometry import sweep_axes

CONNECTIVITY = np.ones((3, 3), dtype=bool)


def visible_pixels(frames, images, probabilities, stage1, thresholds, rules):
    region = Stage1Region(**stage1)
    _, cranial = sweep_axes(frames, rules)
    items = []
    for i, frame in enumerate(frames):
        image = images[i]
        coordinate = project_image(frame, image.shape, rules["pixel_mm"])
        zero_depth = coordinate[0, image.shape[1] // 2]
        frame_levels = region.levels_at(cranial.index_at(zero_depth))
        signal = image_cortical_support(image)
        for level in LEVELS:
            probability = probabilities[level][i]
            rows, mask = extract_observations(
                probability.argmax(0), probability, thresholds[level], frame, coordinate,
            )
            components = {code: ndimage.label(mask == code, structure=CONNECTIVITY)[0]
                          for code in (1, 2)}
            for observation in rows:
                semantic = observation.semantic_class
                code = CLASSES.index(semantic)
                possible = (
                    sap_level_at(observation.coordinate_mm, region, cranial)[1]
                    if semantic == "SAP" else frame_levels
                )
                if possible != (level,):
                    continue
                membership = components[code]
                visible = (membership == membership[observation.row, observation.column]) & signal
                for column in np.flatnonzero(np.any(visible, axis=0)):
                    row = np.flatnonzero(visible[:, column])[0]
                    confidence = probability[code, row, column]
                    items.append((i, (row, column), f"RIGHT_{level}_{semantic}", confidence))
    return {
        "frame_sequence": np.array([item[0] for item in items], dtype=np.int32),
        "pixel_row_col": np.array([item[1] for item in items], dtype=np.int16).reshape(-1, 2),
        "identity": np.array([item[2] for item in items], dtype="U16"),
        "confidence": np.array([item[3] for item in items], dtype=np.float32),
    }
