"""Run ultrasound-only local completion on one sealed set of requests."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree, distance
from scipy.spatial.distance import pdist

from workflows.mbb_shape_completion.fit import METHODS, fit_height, grid, height_ensemble, mask_disk


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(block)
        return checksum.hexdigest()


def load_cloud(path):
    with np.load(path, allow_pickle=False) as cloud:
        return dict(cloud)


def disk_request(data, request, cfg):
    xyz = data["xyz_mm"]
    origin = np.r_[request["center_uv_mm"], 0] @ data["basis"].T
    chart = (xyz - origin) @ data["basis"]
    radius = request["gap_mm"] / 2
    if mask_disk(chart[:, :2], [0, 0], radius).any():
        raise ValueError("withheld observations leaked into the request")
    outer = request["context_radius_mm"]
    use = ((data["identity"] == request["identity"]) & (data["level_hint"] == "L5") &
           mask_disk(chart[:, :2], [0, 0], outer))
    axis = np.arange(-int(radius / cfg["grid_mm"]), int(radius / cfg["grid_mm"]) + 1) * cfg["grid_mm"]
    q = grid([(axis[0], axis[-1])] * 2, cfg["grid_mm"])
    q = q[mask_disk(q, [0, 0], radius)]
    return {"eligible_points": int(use.sum())}, use, origin, data["basis"], q, outer


def normal_at(points, near, probes, radius, minimum_views, groups, probe_ids):
    keep = np.linalg.norm(points - near, axis=1) <= radius
    views = np.unique(groups[keep])
    if len(views) < minimum_views or keep.sum() < 6:
        return None, len(views)
    samples = points[keep]
    # Each viewpoint supplies the same total weight for plane orientation.
    _, inverse, counts = np.unique(groups[keep], return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse] / len(views)
    center = weights @ samples
    centered = samples - center
    eigenvalues, axes = np.linalg.eigh((centered * weights[:, None]).T @ centered)
    if eigenvalues[1] < .25 ** 2:
        return None, len(views)
    normal = axes[:, 0]
    toward_probe = weights @ probes[probe_ids[keep]] - center
    if np.dot(normal, toward_probe) < 0:
        normal = -normal
    return normal, len(views)


def flank_request(data, cfg, fit_cfg, window):
    xyz, labels, groups = (data[key] for key in ("xyz_mm", "identity", "view_group"))
    right = data["basis"][:, 0]
    known = data["level_hint"] == "L5"
    tp, sap = np.flatnonzero(known & (labels == "RIGHT_L5_TP")), np.flatnonzero(known & (labels == "RIGHT_L5_SAP"))
    report = {"tp_points": len(tp), "sap_points": len(sap),
              "eligible_other_bone_points": int((known & (labels == "OTHER_BONE")).sum()),
              "uncertain_bone_points": int((labels == "UNCERTAIN_BONE").sum())}
    if min(len(tp), len(sap)) < 6:
        return {**report, "reason": "MISSING_SEMANTIC_FLANK"}, None
    tree = cKDTree(xyz[sap])
    report["minimum_tp_sap_gap_mm"] = float(tree.query(xyz[tp])[0].min())
    if report["minimum_tp_sap_gap_mm"] > cfg["maximum_gap_mm"]:
        return {**report, "reason": "FLANKS_TOO_FAR_APART"}, None

    dist = distance.cdist(xyz[tp], xyz[sap])
    # Relative right-side ordering is actor-visible. Avoid connecting unrelated
    # levels or a mislabeled SAP lateral to a purported TP root.
    order = xyz[tp] @ right
    lateral = order[:, None] - (xyz[sap] @ right)[None, :]
    admissible = ((dist >= cfg["minimum_gap_mm"]) & (dist <= cfg["maximum_gap_mm"]) & (lateral > 0))
    if not admissible.any():
        return {**report, "reason": "NO_ORDERED_SEMANTIC_FLANKS"}, None
    ti, si = np.unravel_index(np.argmin(np.where(admissible, dist, np.inf)), dist.shape)
    a, b = xyz[tp[ti]], xyz[sap[si]]
    normals, support = zip(*(normal_at(xyz[indices], point, data["probe_mm"], window,
                                     cfg["minimum_anchor_views"], groups[indices],
                                     data["frame_sequence"][indices])
                             for point, indices in ((a, tp), (b, sap))))
    report.update(gap_mm=float(dist[ti, si]), tp_view_groups=support[0], sap_view_groups=support[1],
                  flank_points_mm=[a.tolist(), b.tolist()], anchor_window_mm=window,
                  flank_frame_ids=[int(data["frame_sequence"][i]) for i in (tp[ti], sap[si])])
    if any(n is None for n in normals):
        return {**report, "reason": "UNSUPPORTED_FLANK_ORIENTATION"}, None
    normal_angle = np.degrees(np.arccos(np.clip(normals[0] @ normals[1], -1, 1)))
    report["flank_normal_separation_deg"] = float(normal_angle)
    if normal_angle > cfg["maximum_normal_separation_deg"]:
        return {**report, "reason": "INCOMPATIBLE_FLANK_NORMALS"}, None
    normal = np.sum(normals, axis=0)
    normal /= np.linalg.norm(normal)
    between = b - a
    axis = between - (between @ normal) * normal
    span = np.linalg.norm(axis)
    if span < cfg["minimum_gap_mm"]:
        return {**report, "reason": "INVALID_FLANK_CHART"}, None
    axis /= span
    width_axis = np.cross(normal, axis)
    basis = np.column_stack((axis, width_axis, normal))
    origin = (a + b) / 2
    local = (xyz - origin) @ basis
    near = np.linalg.norm(local, axis=1) <= cfg["context_radius_mm"]
    across = np.abs(local[:, 1]) <= cfg["half_width_mm"] + window
    semantic = np.isin(labels, ("RIGHT_L5_TP", "RIGHT_L5_SAP"))
    other = labels == "OTHER_BONE"
    local_flanks = xyz[near & across & known & semantic]
    near_flanks = cKDTree(local_flanks).query(xyz)[0] <= cfg["other_bone_max_distance_to_flank_mm"]
    use = known & near & across & (semantic | (other & near_flanks))
    # Keep the queried surface inside the observed flank interval and discard
    # already-supported UV positions. All unsupported space remains unknown.
    step = fit_cfg["grid_mm"]
    q = grid([(-span / 2, span / 2), (-cfg["half_width_mm"], cfg["half_width_mm"])], step)
    q = q[cKDTree(local[use, :2]).query(q)[0] > cfg["missing_sample_uv_mm"]]
    report.update(eligible_points=int(use.sum()), eligible_other_bone_local=int((other & use).sum()),
                  excluded_other_bone_local=int((near & across & other & ~use).sum()),
                  reason="FIT_INPUT" if len(q) else "NO_MISSING_UV_SAMPLES")
    return report, (use, origin, basis, q, cfg["context_radius_mm"])


def complete(data, request, protocol, folder):
    begin = perf_counter()
    fit = protocol["fit"]
    if request["experiment"] == "A":
        detail, *args = disk_request(data, request, fit)
    else:
        settings = protocol["actual"]
        detail, args = flank_request(data, settings, fit, settings["anchor_radius_mm"])
        alternatives = {}
        for window in settings["alternative_anchor_windows_mm"]:
            if window == settings["anchor_radius_mm"]:
                continue
            trial, trial_args = flank_request(data, settings, fit, window)
            alternatives[str(window)] = (trial, trial_args)
        detail["flank_window_diagnostics"] = {window: item[0] for window, item in alternatives.items()}
    outcomes = {"input_diagnostics": detail, "methods": {}}
    for method in METHODS:
        start = perf_counter()
        if args is None or not len(args[3]):
            outcomes["methods"][method] = {"status": detail["reason"], "accepted_points": 0,
                                           "proposal_points": 0, "runtime_seconds": perf_counter() - start}
            continue
        use, origin, basis, q, outer = args
        xyz = (data["xyz_mm"][use] - origin) @ basis
        views = data["view_group"][use]
        probes = (data["probe_mm"] - origin) @ basis
        required = None
        if request["experiment"] == "A" and protocol["masks"]["require_fit_at_mask_center"]:
            required = int(np.argmin(np.linalg.norm(q, axis=1)))
        status, variants, members, band = height_ensemble(
            xyz, views, probes, q, [0, 0], outer, fit, method, required_index=required)
        files = {}
        for name, heights in variants.items():
            coords = np.column_stack((q, heights))
            files[name] = coords @ basis.T + origin
        nominal = files["nominal"]
        proposal = nominal[np.isfinite(nominal).all(axis=1)]
        alternative_completions = []
        if request["experiment"] == "B":
            for window, (trial, trial_args) in alternatives.items():
                if trial_args is None or not len(trial_args[3]):
                    continue
                keep, o, axes, _, _ = trial_args
                valid = np.isfinite(nominal).all(axis=1)
                alternative_query = (nominal[valid] - o) @ axes
                alt = fit_height((data["xyz_mm"][keep] - o) @ axes, data["view_group"][keep],
                                 alternative_query[:, :2], fit, method)
                trial.setdefault("fits", {})[method] = alt.detail
                if alt.detail["reason"] == "FIT":
                    valid_alt = np.isfinite(alt.height)
                    alt_q = np.column_stack((alternative_query[valid_alt, :2], alt.height[valid_alt]))
                    alt_world = alt_q @ axes.T + o
                    delta = np.linalg.norm(alt_world - nominal[valid][valid_alt], axis=1)
                    trial["fits"][method]["maximum_disagreement_mm"] = float(delta.max())
                    alternative_completions.append((f"anchor_window_{window}", alt_world))
                    if status == "INFERRED" and delta.max() > fit["maximum_disagreement_mm"]:
                        status = "GEOMETRIC_AMBIGUITY"
                elif status == "INFERRED":
                    status = "INSUFFICIENT_ROBUST_SUPPORT"
            if status == "INFERRED" and any(a[1] is None for a in alternatives.values()):
                status = "INSUFFICIENT_ROBUST_SUPPORT"
        outpath = folder / f"{method}.npz"
        accepted = proposal if status == "INFERRED" else np.empty((0, 3))
        np.savez_compressed(outpath, accepted=accepted, proposed=proposal, query_uv=q, uv_basis=basis,
                            chart_origin_mm=origin, band=band, fit_obs_id=np.flatnonzero(use),
                            **{f"variant_{name}": points for name, points in files.items()},
                            **{f"variant_{name}": points for name, points in alternative_completions})
        outcomes["methods"][method] = {
            "status": status, "accepted_points": len(accepted), "proposal_points": len(proposal),
            "accepted_diameter_mm": float(pdist(accepted).max()) if len(accepted) > 1 else None,
            "supporting_view_groups": sorted(map(int, np.unique(views))),
            "supporting_frame_sequences": sorted(map(int, np.unique(data["frame_sequence"][use]))),
            "fits": members, "file": outpath.name, "sha256": digest(outpath),
            "runtime_seconds": perf_counter() - start,
        }
    outcomes["runtime_seconds"] = perf_counter() - begin
    return outcomes


def run(protocol, input_folder, output):
    requests = read(input_folder / "requests.json")
    output.mkdir(parents=True, exist_ok=False)
    seals = {}
    for key, request in requests.items():
        path = input_folder / f"{key}.npz"
        if digest(path) != request["sha256"]:
            raise ValueError(f"modified actor input: {key}")
        folder = output / key
        folder.mkdir()
        result = complete(load_cloud(path), request, protocol, folder)
        dump(folder / "result.json", result)
        seals[key] = digest(folder / "result.json")
    dump(output / "seals.json", seals)
    print(f"Sealed {len(seals)} input cases.")


if __name__ == "__main__":
    main = argparse.ArgumentParser(description=__doc__)
    main.add_argument("--protocol", type=Path, required=True)
    main.add_argument("--inputs", type=Path, required=True)
    main.add_argument("--output", type=Path, required=True)
    opts = main.parse_args()
    if opts.protocol.with_suffix(".sha256").read_text().split()[0] != digest(opts.protocol):
        raise ValueError("invalid frozen protocol")
    run(read(opts.protocol), opts.inputs, opts.output)
