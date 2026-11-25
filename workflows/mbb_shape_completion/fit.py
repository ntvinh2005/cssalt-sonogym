"""Fit bounded local height fields using observed cortical points only."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from scipy.spatial import QhullError, cKDTree
from scipy.spatial.transform import Rotation


METHODS = ("linear", "regularized")


@dataclass
class Fit:
    height: np.ndarray
    detail: dict


def mask_disk(uv, center, radius):
    return np.linalg.norm(uv - center, axis=1) <= radius + 1e-9


def grid(bounds, step):
    axes = [np.arange(lo, hi + 1e-8, step) for lo, hi in bounds]
    return np.column_stack([a.ravel() for a in np.meshgrid(*axes, indexing="ij")])


def compress(points, views, cell_mm, thickness):
    cells, inverse = np.unique(np.floor(points[:, :2] / cell_mm).astype(int),
                               axis=0, return_inverse=True)
    combined = []
    stacked = 0
    for i in range(len(cells)):
        take = inverse == i
        samples = points[take]
        groups = views[take]
        per_view = np.asarray([np.median(samples[groups == g], axis=0) for g in np.unique(groups)])
        # Repeated pixels from a slice contribute once per cell and view.
        if len(samples) >= 2 and np.ptp(samples[:, 2]) > 2 * thickness:
            stacked += 1
        combined.append(np.median(per_view, axis=0))
    return np.asarray(combined).reshape(-1, 3), stacked / max(1, len(cells))


def fit_height(points, views, query, cfg, method, smoothing=1., required_index=None):
    heights = np.full(len(query), np.nan)
    detail = {"observations": len(points), "distinct_views": len(np.unique(views))}
    if detail["distinct_views"] < cfg["minimum_distinct_views"]:
        return Fit(heights, {**detail, "reason": "INSUFFICIENT_DISTINCT_VIEWS"})
    samples, stacked = compress(points, views, cfg["height_cell_mm"], cfg["rms_cell_height_limit_mm"])
    detail.update(cells=len(samples), multilayer_fraction=stacked)
    if len(samples) < cfg["minimum_cells"]:
        return Fit(heights, {**detail, "reason": "INSUFFICIENT_SPATIAL_SUPPORT"})
    if stacked > cfg["maximum_multilayer_fraction"]:
        return Fit(heights, {**detail, "reason": "COMPETING_HEIGHT_LAYERS"})

    center = np.median(samples, axis=0)
    scaled = (samples - center) / cfg["coordinate_scale_mm"]
    a = np.column_stack((scaled[:, :2], np.ones(len(samples))))
    beta, _, rank, _ = np.linalg.lstsq(a, scaled[:, 2], rcond=None)
    residual = np.abs(scaled[:, 2] - a @ beta) * cfg["coordinate_scale_mm"]
    detail["plane_residual_p90_mm"] = float(np.quantile(residual, .90))
    detail["plane_gradient"] = float(np.linalg.norm(beta[:2]))
    if (rank < 3 or detail["plane_gradient"] > cfg["maximum_gradient"] or
            detail["plane_residual_p90_mm"] > cfg["maximum_robust_plane_residual_mm"]):
        return Fit(heights, {**detail, "reason": "UNSUPPORTED_HEIGHT_FIELD"})
    if len(samples) > cfg["maximum_samples"]:
        idx = np.linspace(0, len(samples) - 1, cfg["maximum_samples"], dtype=int)
        scaled = scaled[idx]
    x = scaled[:, :2]
    y = scaled[:, 2]
    q = (query - center[:2]) / cfg["coordinate_scale_mm"]
    try:
        linear = LinearNDInterpolator(x, y)
        linear_values = linear(q)
        keep = (np.isfinite(linear_values) &
                (cKDTree(samples[:, :2]).query(query)[0] <= cfg["maximum_support_distance_mm"]))
        if method == "linear":
            values = linear_values[keep]
        else:
            smooth = RBFInterpolator(x, y, kernel="thin_plate_spline",
                                     smoothing=cfg["tps_lambda"] * smoothing, degree=1)
            values = smooth(q[keep])
        # Do not bridge a discontinuity by leaving the supported local height range.
        local = cKDTree(x).query(q[keep], k=min(6, len(x)))[1]
        distance = np.linalg.norm(x[local] - q[keep, None], axis=2)
        bounded = ((values >= (y[local] - cfg["maximum_gradient"] * distance).max(axis=1)) &
                   (values <= (y[local] + cfg["maximum_gradient"] * distance).min(axis=1)))
        values[~bounded] = np.nan
        heights[keep] = values * cfg["coordinate_scale_mm"] + center[2]
    except (QhullError, np.linalg.LinAlgError):
        return Fit(heights, {**detail, "reason": "DEGENERATE_SURFACE"})
    fraction = float(np.isfinite(heights).mean()) if len(heights) else 0.
    reason = "FIT" if fraction >= cfg["minimum_fit_fraction"] else "UNSUPPORTED_GAP"
    if required_index is not None and not np.isfinite(heights[required_index]):
        reason = "UNSUPPORTED_GAP_CENTER"
    return Fit(heights, {**detail, "reason": reason, "fraction": fraction})


def perturb_points(points, views, probes, cfg, sign):
    changed = points.copy()
    for i, view in enumerate(np.unique(views)):
        take = views == view
        # Coherent stress within a viewpoint group, with deterministic alternating
        # directions across groups. This is not a sampled measurement-error model.
        angle = sign * (-1) ** i * cfg["rotation_stress_deg"]
        axis = np.eye(3)[i % 3]
        rotation = Rotation.from_rotvec(np.deg2rad(angle) * axis)
        pivot = probes[view]
        changed[take] = rotation.apply(points[take] - pivot) + pivot
        changed[take] += axis * sign * (-1) ** i * cfg["translation_stress_mm"]
    return changed


def height_ensemble(points, views, probes, query, fit_center, radius_mm, cfg, method, required_index=None):
    nominal = fit_height(points, views, query, cfg, method, required_index=required_index)
    nominal.detail["query_count"] = len(query)
    variants = {"nominal": nominal.height}
    members = {"nominal": nominal.detail.copy()}
    if nominal.detail["reason"] != "FIT":
        return nominal.detail["reason"], variants, members, np.empty((0, 4))

    def add(name, samples, groups, q=query, smoothing=1., convert=None):
        if q is query:
            index = required_index
        else:
            index = int(np.count_nonzero(valid[:required_index])) if required_index is not None else None
        trial = fit_height(samples, groups, q, cfg, method, smoothing, required_index=index)
        members[name] = trial.detail
        if trial.detail["reason"] == "FIT":
            variants[name] = trial.height if convert is None else convert(trial.height)

    group_ids, counts = np.unique(views, return_counts=True)
    for group in group_ids[np.argsort(-counts, kind="stable")[:cfg["holdout_view_groups"]]]:
        take = views != group
        add(f"omit_view_{group}", points[take], views[take])
    distance = np.linalg.norm(points[:, :2] - fit_center, axis=1)
    for scale in cfg["neighbor_scales"]:
        take = distance <= radius_mm * scale
        add(f"context_{scale}", points[take], views[take])
    if method == "regularized":
        for scale in cfg["lambda_scales"]:
            add(f"smoothing_{scale}", points, views, smoothing=scale)
    valid = np.isfinite(nominal.height)
    q3 = np.column_stack((query, nominal.height))
    pivot = np.median(points, axis=0)
    for sign in (-1, 1):
        add(f"pose_stress_{sign}", perturb_points(points, views, probes, cfg, sign), views)
        rotation = Rotation.from_euler("x", sign * cfg["chart_tilt_deg"], degrees=True)
        chart_query = rotation.apply(q3[valid] - pivot) + pivot

        def back(height):
            xyz = rotation.inv().apply(np.column_stack((chart_query[:, :2], height)) - pivot) + pivot
            adjusted = np.full(len(query), np.nan)
            adjusted[valid] = xyz[:, 2]
            return adjusted

        add(f"chart_tilt_{sign}", rotation.apply(points - pivot) + pivot, views,
            q=chart_query[:, :2], convert=back)
    array = np.stack(list(variants.values()))
    required = cfg["minimum_ensemble_members"]
    supported = valid & (np.isfinite(array).sum(axis=0) >= required)
    band = np.column_stack((np.flatnonzero(supported), np.nanmin(array[:, supported], axis=0),
                            np.nanmax(array[:, supported], axis=0),
                            np.isfinite(array[:, supported]).sum(axis=0)))
    shared = len(band) / max(1, int(valid.sum()))
    max_spread = float(np.ptp(band[:, 1:3], axis=1).max()) if len(band) else None
    # A variant agreeing on a tiny region is not a robust reconstruction.
    holdouts = {name: detail for name, detail in members.items() if name.startswith("omit_view_")}
    complete_holdouts = len(holdouts) == cfg["holdout_view_groups"] and all(
        trial["reason"] == "FIT" for trial in holdouts.values())
    members["ensemble"] = {"successful_members": len(variants), "shared_fraction": shared,
                           "maximum_disagreement_mm": max_spread,
                           "tested_view_removals": len(holdouts),
                           "view_removals_supported": sum(trial["reason"] == "FIT"
                                                          for trial in holdouts.values())}
    if (shared < cfg["minimum_ensemble_grid_fraction"] or
            (cfg["require_top_view_removals_supported"] and not complete_holdouts)):
        return "INSUFFICIENT_ROBUST_SUPPORT", variants, members, band
    if max_spread > cfg["maximum_disagreement_mm"]:
        return "GEOMETRIC_AMBIGUITY", variants, members, band
    return "INFERRED", variants, members, band
