"""Describe local orientation and curvature using only observed cortical points."""

import numpy as np

from workflows.mbb_shape_completion.fit import compress


def describe(points, views, anchor, probes, fit):
    n_views = len(np.unique(views))
    result = {"observed_samples": len(points), "distinct_views": n_views}
    if n_views < fit["minimum_distinct_views"] or len(points) < fit["minimum_cells"]:
        return {**result, "status": "INSUFFICIENT_DISTINCT_SURFACE_EVIDENCE"}
    _, inverse, counts = np.unique(views, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse] / n_views
    centered = points - weights @ points
    values, basis = np.linalg.eigh((centered * weights[:, None]).T @ centered)
    if values[1] < fit["height_cell_mm"] ** 2:
        return {**result, "status": "INSUFFICIENT_TANGENT_SPREAD"}
    basis = basis[:, ::-1]
    if (weights @ probes[views] - anchor) @ basis[:, 2] < 0:
        basis[:, 2] *= -1
    local = (points - anchor) @ basis
    cells, stacked = compress(local, views, fit["height_cell_mm"], fit["rms_cell_height_limit_mm"])
    if len(cells) < fit["minimum_cells"] or stacked > fit["maximum_multilayer_fraction"]:
        return {**result, "status": "UNSUPPORTED_LOCAL_HEIGHT_FIELD"}
    u, v, z = cells.T
    system = np.column_stack((np.ones(len(u)), u, v, u * u / 2, u * v, v * v / 2))
    coefficients, _, rank, _ = np.linalg.lstsq(system, z, rcond=None)
    residual = np.quantile(np.abs(z - system @ coefficients), .9)
    if rank < 6 or residual > fit["maximum_robust_plane_residual_mm"]:
        return {**result, "status": "UNSUPPORTED_LOCAL_QUADRATIC"}
    _, dx, dy, dxx, dxy, dyy = coefficients
    derivative = np.asarray([dx, dy])
    first_form = np.eye(2) + np.outer(derivative, derivative)
    second_form = np.array([[dxx, dxy], [dxy, dyy]]) / np.linalg.norm(np.r_[derivative, 1.])
    curvatures = np.linalg.eigvals(np.linalg.solve(first_form, second_form)).real
    normal = basis @ np.r_[-derivative, 1.]
    normal /= np.linalg.norm(normal)
    return {**result, "status": "DESCRIBED", "fit_cells": len(cells),
            "normal_actor_base": normal.tolist(),
            "principal_curvature_inverse_mm": np.sort(curvatures).tolist(),
            "absolute_height_residual_p90_mm": float(residual)}
