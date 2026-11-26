"""Register a fixed L5 mean from public observations only."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from workflows.mbb_ssm.fit import LABELS, fit_arrays, initial_translation, model_regions, weights
from workflows.mbb_ssm.geometry import Mesh, nearest_points, posterior, rigid


ARMS = ("A0", "A1", "A2", "A3")
TP, SAP, OTHER = LABELS[1], LABELS[2], LABELS[0]
SEMANTIC = (TP, SAP)


@dataclass
class Observations:
    xyz: np.ndarray
    labels: np.ndarray
    group: np.ndarray
    quality: np.ndarray

    def select(self, keep):
        return Observations(self.xyz[keep], self.labels[keep], self.group[keep], self.quality[keep])


def group_weights(groups, confidence):
    _, g = np.unique(groups, return_inverse=True)
    return weights(g, np.maximum(confidence, .02))


def deduplicate(obs, voxel):
    spatial = np.floor(obs.xyz / voxel).astype(int)
    _, label = np.unique(obs.labels, return_inverse=True)
    key = np.column_stack((spatial, label, obs.group))
    order = np.argsort(-obs.quality, kind="stable")
    _, representative = np.unique(key[order], axis=0, return_index=True)
    return obs.select(np.sort(order[representative]))


def external_proxies(xyz):
    """Use only the remaining left-side geometric samples to create proxy anchors."""
    x, y, z = xyz.T
    lat = np.quantile(x, [.35, .38, .80])
    ap = np.quantile(y, [.12, .38, .72])
    result = np.full(len(xyz), OTHER, dtype="U20")
    result[(x <= lat[0]) & (y >= ap[0]) & (y <= ap[2])] = TP
    result[(x >= lat[1]) & (x <= lat[2]) & (y <= ap[1]) & (z >= np.median(z))] = SAP
    return result


def anchor(obs, settings):
    n = len(obs.xyz)
    if n < settings["anchor_min_points"]:
        return None, "TOO_FEW_ANCHOR_POINTS"
    p = cKDTree(obs.xyz).query_pairs(settings["anchor_connectivity_mm"], output_type="ndarray")
    graph = sparse.coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n, n))
    _, ids = connected_components(graph.tocsr(), directed=False)
    components = []
    for c in np.unique(ids):
        keep = ids == c
        views = len(np.unique(obs.group[keep]))
        if views >= settings["anchor_min_views"] and keep.sum() >= settings["anchor_min_points"]:
            subset = obs.select(keep)
            w = group_weights(subset.group, subset.quality)
            medians = [np.median(subset.xyz[subset.group == g], axis=0)
                       for g in np.unique(subset.group)]
            center = np.median(medians, axis=0)
            components.append((views, float(w @ subset.quality), keep.sum(), keep, center))
    if not components:
        return None, "TOO_FEW_CONNECTED_VIEWS"
    components.sort(key=lambda x: x[:3], reverse=True)
    views, _, n, mask, center = components[0]
    if any(m[0] >= .8 * views and np.linalg.norm(m[-1] - center) > 6. for m in components[1:]):
        return None, "COMPETING_ANCHOR_CLUSTERS"
    return {"center": center, "mask": mask, "views": views, "points": int(n)}, "SUPPORTED"


def supported_normal(points, center, weights_):
    near = np.linalg.norm(points - center, axis=1) <= 5.
    if near.sum() < 8 or weights_[near].sum() <= 0:
        return None
    x, w = points[near], weights_[near]
    w = w / w.sum()
    x = x - w @ x
    value, directions = np.linalg.eigh((x * w[:, None]).T @ x)
    v = directions[:, 0]
    if value[1] < 1. or value[0] > .22 * value[1] or abs(v[1]) < .2:
        return None
    return v if v[1] < 0 else -v


def regions_for_mean(model, side, other_radius):
    xyz = model.mean
    mirrored = xyz.copy()
    mirrored[:, 0] *= side
    support, tp, sap = model_regions(mirrored)
    centers = np.array([np.median(xyz[m], axis=0) for m in (tp, sap)])
    lateral = mirrored[:, 0] >= np.quantile(mirrored[:, 0], .45)
    near = np.linalg.norm(xyz[:, None] - centers, axis=-1).min(axis=1) <= other_radius
    masks = {TP: tp, SAP: sap, OTHER: support & lateral & near}
    meshes = {}
    for name, mask in masks.items():
        region = Mesh(xyz, model.faces[np.count_nonzero(mask[model.faces], axis=1) >= 2])
        if not len(region.faces):
            raise ValueError("Empty mean-shape compatibility region")
        meshes[name] = (region, cKDTree(xyz[region.faces].mean(axis=1)))
    return centers, masks, meshes


def clip_rotation(rotation, limit):
    vector = Rotation.from_matrix(rotation).as_rotvec()
    angle = np.linalg.norm(vector)
    maximum = np.deg2rad(limit)
    return Rotation.from_rotvec(vector * maximum / angle).as_matrix() if angle > maximum else rotation


def anatomical_starts(obs, observed, model, centers, masks, cfg):
    model_axis = centers[1] - centers[0]
    measured_axis = observed[1] - observed[0]
    model_axis /= np.linalg.norm(model_axis)
    measured_axis /= np.linalg.norm(measured_axis)
    fit, _ = Rotation.align_vectors(measured_axis, model_axis)
    starts = [("public_axes", np.eye(3))]
    for angle in cfg["roll_hypotheses_deg"]:
        roll = Rotation.from_rotvec(np.deg2rad(angle) * measured_axis)
        starts.append((f"anchor_axis_roll_{angle:g}", (roll * fit).as_matrix().T))

    source, target, w = [model_axis], [measured_axis], [2.]
    surface_normals = {}
    for index, label in enumerate(SEMANTIC):
        take = obs.labels == label
        w_obs = group_weights(obs.group[take], obs.quality[take])
        normal_observed = supported_normal(obs.xyz[take], observed[index], w_obs)
        points = model.mean[masks[label]]
        normal_mean = supported_normal(points, centers[index], np.ones(len(points)))
        surface_normals[label] = {"observed": normal_observed.tolist() if normal_observed is not None else None,
                                  "mean": normal_mean.tolist() if normal_mean is not None else None}
        if (normal_observed is not None and normal_mean is not None and
                abs(normal_observed @ measured_axis) < .92 and abs(normal_mean @ model_axis) < .92):
            source.append(normal_mean)
            target.append(normal_observed)
            w.append(1.)
    if len(w) > 1:
        pose, _ = Rotation.align_vectors(target, source, weights=w)
        starts.append(("anchors_and_supported_normals", pose.as_matrix().T))
    cap = cfg["max_public_rotation_deg"]
    poses = []
    for name, rotation in starts:
        r = clip_rotation(rotation, cap)
        if not any(np.allclose(r, prev, atol=1e-5) for _, prev, _ in poses):
            poses.append((name, r, observed.mean(axis=0) - centers.mean(axis=0) @ r))
    return poses, surface_normals


def legacy_starts(obs, model, old):
    labels = np.asarray([LABELS.index(name) for name in obs.labels])
    t = initial_translation(obs.xyz, labels, model.mean, model_regions(model.mean))
    poses = [("legacy_nominal", np.eye(3), t)]
    poses += [(f"legacy_ap_{shift:g}", np.eye(3), t + [0., shift, 0.])
              for shift in old["initial_y_offsets_mm"] if shift]
    poses += [(f"legacy_tilt_{angle:g}", Rotation.from_euler("z", angle, degrees=True).as_matrix(), t)
              for angle in old["initial_z_tilts_deg"] if angle]
    return poses


def match(obs, meshes, r, t):
    mapped = (obs.xyz - t) @ r.T
    nearest = np.empty_like(mapped)
    for label in np.unique(obs.labels):
        use = obs.labels == label
        region, tree = meshes[label]
        nearest[use], _ = nearest_points(mapped[use], region, tree)
    return nearest, np.linalg.norm(mapped - nearest, axis=1)


def balanced_weights(obs, other_share):
    proportions = {TP: .5, SAP: .5, OTHER: 0.}
    if OTHER in obs.labels:
        proportions = {TP: (1 - other_share) / 2, SAP: (1 - other_share) / 2, OTHER: other_share}
    values = np.zeros(len(obs.xyz))
    for label, share in proportions.items():
        take = obs.labels == label
        if take.any():
            values[take] = share * group_weights(obs.group[take], obs.quality[take])
    return values


def objective(distance, weight, huber):
    return float(weight @ np.where(distance <= huber, distance ** 2, huber * (2 * distance - huber)))


def fit_pose(obs, meshes, pose, fit_cfg, weight):
    name, r, t = pose
    huber = fit_cfg["huber_mm"]
    previous = np.inf
    for iteration in range(fit_cfg["iterations"]):
        nearest, d = match(obs, meshes, r, t)
        robust = weight * np.minimum(1., huber / np.maximum(d, 1e-6))
        r, _ = rigid(nearest, obs.xyz, robust)
        r = clip_rotation(r, fit_cfg["max_public_rotation_deg"])
        t = np.average(obs.xyz - nearest @ r, weights=robust, axis=0)
        loss = objective(d, weight, huber)
        if iteration >= 4 and abs(loss - previous) < 1e-4:
            break
        previous = loss
    _, d = match(obs, meshes, r, t)
    class_p95 = {label: float(np.quantile(d[obs.labels == label], .95)) for label in np.unique(obs.labels)}
    return r, t, {"initialization": name, "objective_mm2": objective(d, weight, huber),
                  "observed_median_mm": float(np.median(d)), "observed_p95_mm": float(np.quantile(d, .95)),
                  "class_observed_p95_mm": class_p95, "iterations": iteration + 1}


def posterior_disagreement(a, b, support):
    return float(np.quantile(np.linalg.norm(a[support] - b[support], axis=1), .95)) if support.any() else None


def robust_enough(meta, cfg):
    return (meta["observed_p95_mm"] <= cfg["max_observed_p95_mm"] and
            all(meta["class_observed_p95_mm"].get(label, np.inf) <= cfg["max_class_observed_p95_mm"]
                for label in SEMANTIC))


def solve(raw, model, side, arm, weight, protocol, legacy_cfg):
    pre, cfg = protocol["preprocess"], protocol["fit"]
    obs = deduplicate(raw, pre["voxel_mm"])
    centers, masks, meshes = regions_for_mean(model, side, pre["other_anchor_radius_mm"])
    flags, selected, observed = {}, [], []
    for label in SEMANTIC:
        group = obs.select(obs.labels == label)
        anchor_info, cause = anchor(group, pre)
        flags[label] = cause
        if anchor_info is not None:
            selected.append(group.select(anchor_info["mask"]))
            observed.append(anchor_info["center"])
    if len(selected) != 2:
        return None, [], {"status": "UNRESOLVED_ANCHORS", "anchor_status": flags}, None
    observed = np.asarray(observed)
    gap = np.linalg.norm(observed[1] - observed[0])
    if not pre["anchor_min_separation_mm"] <= gap <= pre["anchor_max_separation_mm"]:
        return None, [], {"status": "INCONSISTENT_ANCHORS", "anchor_separation_mm": float(gap)}, None
    anchors = Observations(*(np.concatenate([getattr(part, field) for part in selected])
                             for field in ("xyz", "labels", "group", "quality")))
    starts, orientation = anatomical_starts(anchors, observed, model, centers, masks, cfg)
    if arm == "A3":
        # A3 differs from A1 only in its initial rigid poses. Left-side synthetic
        # rays get the corresponding mirrored compatibility masks.
        if side == 1:
            starts = legacy_starts(anchors, model, legacy_cfg)
        else:
            shifted_mean = model.mean.copy()
            shifted_mean[:, 0] *= -1
            reflected = anchors.xyz.copy()
            reflected[:, 0] *= -1
            reflected_anchors = Observations(reflected, anchors.labels, anchors.group, anchors.quality)
            mirror_model = type(model)(shifted_mean, model.faces, model.modes)
            flip = np.diag([-1., 1., 1.])
            starts = [(name, flip @ r @ flip, t @ flip)
                      for name, r, t in legacy_starts(reflected_anchors, mirror_model, legacy_cfg)]
    other_count = 0
    if arm == "A2":
        near = np.linalg.norm(obs.xyz[:, None] - observed, axis=-1).min(axis=1) <= pre["other_anchor_radius_mm"]
        other = obs.select(near & (obs.labels == OTHER))
        if len(other.xyz):
            distance = np.min([match(other, meshes, r, t)[1] for _, r, t in starts], axis=0)
            other = other.select(distance <= pre["other_compatibility_mm"])
            if len(np.unique(other.group)) >= pre["other_min_views"]:
                other_count = len(other.xyz)
                anchors = Observations(*(np.concatenate((getattr(anchors, field), getattr(other, field)))
                                         for field in ("xyz", "labels", "group", "quality")))
    normalized = balanced_weights(anchors, weight)
    solutions = [fit_pose(anchors, meshes, pose, cfg, normalized) for pose in starts]
    solutions.sort(key=lambda x: x[2]["objective_mm2"])
    r, t, info = solutions[0]
    best = model.mean @ r + t
    hidden = posterior(model.mean) & (cKDTree(raw.xyz).query(best)[0] > 3.)
    alternatives = []
    disagreements = []
    for other_r, other_t, info2 in solutions[1:]:
        if info2["objective_mm2"] - info["objective_mm2"] <= cfg["objective_delta_mm2"] and robust_enough(info2, cfg):
            candidate = model.mean @ other_r + other_t
            alternatives.append(candidate)
            delta = posterior_disagreement(best, candidate, hidden)
            if delta is not None:
                disagreements.append(delta)
    spread = max(disagreements, default=None)
    if not robust_enough(info, cfg):
        status = "OBSERVED_RESIDUAL_TOO_LARGE"
    elif spread is not None and spread > cfg["max_alternative_hidden_p95_mm"]:
        status = "ANATOMICALLY_AMBIGUOUS"
    else:
        status = "INFERRED"
    # Preserve input anchors, region centers, and pre-fit model anchor guesses.
    result = {**info, "status": status, "anchor_status": flags,
              "anchor_separation_mm": float(gap), "orientation_support": orientation,
              "rotation_local_rows": r.tolist(), "translation_local_mm": t.tolist(),
              "other_points": other_count, "other_effective_weight": weight if other_count else 0.,
              "class_point_count": {label: int(np.count_nonzero(anchors.labels == label)) for label in np.unique(anchors.labels)},
              "solutions": [x[2] for x in solutions], "similar_solution_count": len(alternatives),
              "alternative_hidden_p95_mm": spread}
    layers = {"observed_anchors": observed,
              "initial_anchor_guesses": np.stack([centers @ r0 + t0 for _, r0, t0 in starts])}
    return best, alternatives, result, (hidden, anchors, layers)


def perturbations(obs):
    groups, count = np.unique(obs.group[np.isin(obs.labels, SEMANTIC)], return_counts=True)
    if not len(groups):
        groups, count = np.unique(obs.group, return_counts=True)
    if not len(groups):
        return {}
    return {"drop_one_strongest_anchor_view": obs.group != groups[np.argmax(count)],
            "drop_every_fifth_view": ~np.isin(obs.group, np.unique(obs.group)[::5])}


def register(data, model, arm, other_weight, protocol, legacy_fit, external):
    started = perf_counter()
    basis, xyz = data["basis"], data["xyz_mm"]
    if not np.isfinite(xyz).all() or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-6) or np.linalg.det(basis) < 0.:
        raise ValueError("Observations need a proper actor coordinate frame")
    raw = Observations(xyz, data["identity"], data["view_group"], data["confidence"])
    if arm == "A0":
        proposal, alternatives, meta = fit_arrays(xyz, basis, raw.labels, raw.group, raw.quality,
                                                   model, 0., legacy_fit)
        selected = {k: v for k, v in meta.items() if k != "prior_perturbations"}
        layers = {f"A0_alternative_{i}": a for i, a in enumerate(alternatives)}
        stress = {}
        if proposal is not None:
            layers["A0_proposal"] = proposal
            hidden = posterior(model.mean) & (cKDTree(xyz).query(proposal)[0] > 3.)
            for name, keep in perturbations(raw).items():
                obs = raw.select(keep)
                outcome, _, trial = fit_arrays(obs.xyz, basis, obs.labels, obs.group, obs.quality, model, 0., legacy_fit)
                delta = None if outcome is None else posterior_disagreement(proposal, outcome, hidden)
                stress[name] = {"status": trial["status"], "posterior_p95_mm": delta}
                if outcome is not None:
                    layers[f"A0_subset_{name}"] = outcome
        selected["subset_diagnostics"] = stress
        selected["seconds_with_subsets"] = perf_counter() - started
        return layers, selected

    origin = np.median(xyz, axis=0)
    labels = external_proxies(xyz) if external else raw.labels
    obs = Observations((xyz - origin) @ basis, labels, raw.group, raw.quality)
    side = -1 if external else 1
    proposal, alternatives, meta, details = solve(obs, model, side, arm, other_weight, protocol, legacy_fit)
    nominal_seconds = perf_counter() - started
    layers = {}
    perturb = {}
    if proposal is not None:
        hidden, chosen, anchor_layers = details
        layers = {f"{arm}_{key}": array @ basis.T + origin for key, array in anchor_layers.items()}
        layers.update({f"{arm}_proposal": proposal @ basis.T + origin,
                       f"{arm}_included_xyz": chosen.xyz @ basis.T + origin,
                       f"{arm}_included_label": chosen.labels,
                       f"{arm}_included_view": chosen.group,
                       f"{arm}_included_confidence": chosen.quality})
        layers.update({f"{arm}_alternative_{i}": a @ basis.T + origin for i, a in enumerate(alternatives)})
        for name, keep in perturbations(obs).items():
            subset = obs.select(keep)
            fit, _, case, _ = solve(subset, model, side, arm, other_weight, protocol, legacy_fit)
            delta = None if fit is None else posterior_disagreement(proposal, fit, hidden)
            # The periodic deletion is skipped only if it removes anchor support.
            skip = name == "drop_every_fifth_view" and case["status"] in ("UNRESOLVED_ANCHORS", "INCONSISTENT_ANCHORS")
            perturb[name] = {"status": case["status"], "posterior_p95_mm": delta, "skipped": skip}
            if fit is not None:
                layers[f"{arm}_subset_{name}"] = fit @ basis.T + origin
            if (meta["status"] == "INFERRED" and not skip and
                    (case["status"] != "INFERRED" or delta is None or
                     delta > protocol["fit"]["max_subset_hidden_p95_mm"])):
                meta["status"] = "UNSTABLE_UNDER_OBSERVATION_SELECTION"

    return layers, {**meta, "subset_diagnostics": perturb, "seconds": nominal_seconds,
                    "seconds_with_subsets": perf_counter() - started}
