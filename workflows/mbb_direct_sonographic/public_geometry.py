"""Relate public probe poses to Stage 1 sweep commands without simulator poses."""

from __future__ import annotations

import numpy as np

from two_stage_navigation.simple_landmark_mapping import CranialCoordinate, project_image


def sweep_axes(frames, rules):
    transverse = [frame for frame in frames if frame["phase"] == "transverse" and frame["image_valid"]]
    commands = np.asarray([frame["requested_relative_pose_index"][:2] for frame in transverse])
    origins = np.asarray([project_image(frame, (1, 1), rules["pixel_mm"])[0, 0] for frame in transverse])
    directions, _, rank, _ = np.linalg.lstsq(
        commands - commands.mean(0), origins - origins.mean(0), rcond=None,
    )
    if rank != 2:
        raise ValueError("public sweep does not identify lateral and cranial axes")
    right = directions[0] / np.linalg.norm(directions[0])
    cranial = directions[1] - np.dot(directions[1], right) * right
    cranial /= np.linalg.norm(cranial)
    spacing = rules["index_spacing_mm"]
    intercept = float(commands[:, 1].mean() - np.dot(origins.mean(0), cranial) / spacing)
    residual = float(np.max(np.abs(origins @ cranial / spacing + intercept - commands[:, 1])))
    if not np.isfinite(residual) or residual > rules["max_axis_fit_residual_index"]:
        raise ValueError("executed FK poses disagree with the actor's Stage 1-relative commands")
    return right, CranialCoordinate(tuple(cranial), intercept, residual, spacing)
