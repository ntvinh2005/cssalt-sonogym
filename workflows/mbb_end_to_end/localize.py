"""Localize an observed TP–SAP cortical transition from actor-visible evidence."""

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation

from two_stage_navigation.simple_landmark_mapping import project_image
from workflows.mbb_observed_surface.localize import (
    Surface, assign_views, component_labels, distinct_poses, patch, public_axes,
)


RULES = json.loads((Path(__file__).parents[1] / "mbb_observed_surface" / "actor_rules.json").read_text())
TP = "RIGHT_L5_TP"
SAP = "RIGHT_L5_SAP"
OTHER = "OTHER_BONE"
UNCERTAIN = "UNCERTAIN_BONE"
# Samples are spaced at 0.5 mm in the image and the prior geometry uses a 1.5 mm thickness scale.
CELL_MM = RULES["pixel_mm"]
EDGE_MM = RULES["max_fit_thickness_mm"]
FLANK_LENGTH_MM = RULES["max_region_diameter_mm"]
ACQUISITION_KEYS = ("image_valid", "contact_ok", "producer_advanced", "transition_valid")


@dataclass(frozen=True)
class Cloud:
    xyz: np.ndarray
    role: np.ndarray
    level: np.ndarray
    frame: np.ndarray
    pixel: np.ndarray
    confidence: np.ndarray
    probabilities: np.ndarray
    row: np.ndarray

    @classmethod
    def from_arrays(cls, arrays):
        xyz = arrays["xyz_mm"]
        return cls(xyz, arrays["identity"], arrays["level_hint"], arrays["frame_sequence"],
                   arrays["pixel_row_col"], arrays["confidence"],
                   arrays["semantic_probabilities"], np.arange(len(xyz)))

    def take(self, indices):
        return Cloud(*(getattr(self, field)[indices] for field in self.__dataclass_fields__))

    def surface(self):
        return Surface(self.xyz, self.frame, self.confidence, self.pixel)


def cloud_integrity(cloud, frames):
    n = len(cloud.xyz)
    shape = tuple(RULES["image_shape"])
    arrays = (cloud.role, cloud.level, cloud.frame, cloud.confidence)
    if [f["frame_sequence"] for f in frames] != list(range(len(frames))):
        raise ValueError("expected the original contiguous public frame order")
    if (cloud.xyz.shape != (n, 3) or cloud.pixel.shape != (n, 2) or
            any(a.shape != (n,) for a in arrays) or cloud.probabilities.shape != (n, 4) or
            not np.isfinite(cloud.xyz).all() or not np.isfinite(cloud.confidence).all() or
            not np.isfinite(cloud.probabilities).all() or np.any((cloud.confidence < 0) | (cloud.confidence > 1)) or
            np.any((cloud.confidence == 0) & (cloud.role != UNCERTAIN)) or
            not np.issubdtype(cloud.frame.dtype, np.integer) or
            not np.issubdtype(cloud.pixel.dtype, np.integer) or
            np.any((cloud.frame < 0) | (cloud.frame >= len(frames))) or
            np.any(cloud.pixel < 0) or np.any(cloud.pixel >= shape)):
        raise ValueError("malformed actor cloud")
    errors = []
    for f in np.unique(cloud.frame):
        frame = frames[f]
        if not all(frame[key] is True for key in ACQUISITION_KEYS):
            raise ValueError(f"cloud contains a point from invalid acquisition {f}")
        same = cloud.frame == f
        r, c = cloud.pixel[same].T
        fitted = project_image(frame, shape, RULES["pixel_mm"])[r, c]
        errors.extend(np.linalg.norm(cloud.xyz[same] - fitted, axis=1))
    error = float(max(errors, default=0.0))
    if error > RULES["max_projection_parity_mm"]:
        raise ValueError(f"calibrated public reprojection mismatch: {error} mm")
    return {"public_reprojection_max_mm": error, "frames": len(frames), "cloud_rows": n,
            "frames_with_cloud_support": int(len(np.unique(cloud.frame)))}


def views(cloud, distinct):
    return np.unique(assign_views(np.unique(cloud.frame), distinct)).tolist()


def provenance(cloud, distinct):
    independent = views(cloud, distinct)
    return {
        "cloud_rows": cloud.row.tolist(),
        "frame_sequences": np.unique(cloud.frame).tolist(),
        "pixel_row_col": cloud.pixel.tolist(),
        "frame_sequence_by_row": cloud.frame.tolist(),
        "distinct_viewpoints": independent,
        "distinct_viewpoint_count": len(independent),
    }


def cortex_graph(nodes, cortex_tree):
    pairs = cKDTree(nodes.xyz).query_pairs(EDGE_MM, output_type="ndarray")
    endpoints = nodes.xyz[pairs]
    distance = np.linalg.norm(endpoints[:, 0] - endpoints[:, 1], axis=1)
    # The widest links need a separate observed point near the intervening surface.
    supported = (distance <= 2 * CELL_MM) | (cortex_tree.query(endpoints.mean(axis=1))[0] <= CELL_MM)
    pairs, distance = pairs[supported], distance[supported]
    return csr_matrix((np.maximum(distance, 1e-9), (pairs[:, 0], pairs[:, 1])),
                      shape=(len(nodes.xyz), len(nodes.xyz)))


def representatives(cloud):
    # Preserve each label represented in a cell so a route must reach real anchor evidence.
    order = np.argsort(-cloud.confidence, kind="stable")
    cells = np.floor(cloud.xyz[order] / CELL_MM).astype(np.int64)
    _, roles = np.unique(cloud.role[order], return_inverse=True)
    _, at = np.unique(np.column_stack((cells, roles)), axis=0, return_index=True)
    return cloud.take(order[at])


def fit_flank(anchor, tree, endpoint, distinct, probe, minimum_views):
    neighbors = np.asarray(sorted(tree.query_ball_point(endpoint, RULES["fit_radius_mm"])), dtype=int)
    if len(neighbors) < 3:
        return None
    seed = int(neighbors[np.argmin(np.linalg.norm(anchor.xyz[neighbors] - endpoint, axis=1))])
    fitted, _ = patch(anchor.surface(), neighbors, seed, distinct, probe, RULES, minimum_views)
    if fitted is None:
        return None
    source, normal = fitted
    return anchor.take(source), normal


def joined_cortex(node, graph, cap, tp_nodes, sap_nodes):
    distance, predecessor = dijkstra(graph, indices=node, directed=False,
                                    limit=cap, return_predecessors=True)
    ends = [int(ids[np.argmin(distance[ids])]) for ids in (tp_nodes, sap_nodes)]
    if not np.isfinite(distance[ends]).all() or distance[ends].sum() > 2 * cap:
        return None
    paths = []
    for end in ends:
        route = [end]
        while route[-1] != node:
            route.append(int(predecessor[route[-1]]))
        paths.append(route)
    return ends, np.unique(np.concatenate(paths)), float(distance[ends].sum())


def unstable_levels(controversial_tree, point, distinct):
    if controversial_tree is None:
        return False
    cloud, tree = controversial_tree
    near = tree.query_ball_point(point, EDGE_MM)
    if not near:
        return False
    local = cloud.take(np.asarray(near, dtype=int))
    return len(views(local, distinct)) >= RULES["min_views"]


def find_region(cloud, distinct, probe, axes, minimum_views):
    right, superior = axes[:2]
    anchors = [cloud.take((cloud.role == key) & (cloud.level == "L5")) for key in (TP, SAP)]
    evidence = {
        "tp_observations": len(anchors[0].xyz), "sap_observations": len(anchors[1].xyz),
        "tp_distinct_views": len(views(anchors[0], distinct)), "sap_distinct_views": len(views(anchors[1], distinct)),
        "eligible_other_bone_observations": int(np.count_nonzero((cloud.role == OTHER) & (cloud.level == "L5"))),
        "nearest_tp_sap_mm": None, "local_route_seed_count": 0, "bounded_connector_route_count": 0,
        "multi_view_route_count": 0, "fitted_flank_route_count": 0, "oriented_flank_route_count": 0,
        "supported_route_count": 0, "maximum_flank_path_mm": FLANK_LENGTH_MM,
        "maximum_combined_path_mm": 2 * FLANK_LENGTH_MM,
        "competing_region_count": 0, "ambiguous_level_seed_count": 0,
    }
    if not all(len(anchor.xyz) for anchor in anchors):
        return None, "missing_tp_or_sap", evidence

    anchor_trees = [cKDTree(a.xyz) for a in anchors]
    evidence["nearest_tp_sap_mm"] = float(anchor_trees[0].query(anchors[1].xyz)[0].min())
    eligible = (cloud.level == "L5") & np.isin(cloud.role, (TP, SAP, OTHER))
    cortex = cloud.take(eligible)
    cortex_tree = cKDTree(cortex.xyz)
    candidate = np.logical_and.reduce([tree.query(cortex.xyz)[0] <= FLANK_LENGTH_MM
                                      for tree in anchor_trees])
    if not np.any(candidate):
        return None, "no_shared_local_support", evidence

    # Include the neighborhood of each possible junction, with no route through adjacent vertebrae.
    seed_tree = cKDTree(cortex.xyz[candidate])
    nodes = representatives(cortex.take(seed_tree.query(cortex.xyz)[0] <= FLANK_LENGTH_MM))
    anchor_node_ids = [np.flatnonzero(nodes.role == name) for name in (TP, SAP)]
    if any(len(ids) == 0 for ids in anchor_node_ids):
        return None, "missing_tp_or_sap_in_local_cortex", evidence
    graph = cortex_graph(nodes, cortex_tree)
    seeds = np.flatnonzero(np.logical_and.reduce([tree.query(nodes.xyz)[0] <= FLANK_LENGTH_MM
                                                for tree in anchor_trees]))
    evidence["local_route_seed_count"] = len(seeds)
    uncertain = cloud.take(np.isin(cloud.level, ("L4", "AMBIGUOUS")))
    controversial = (uncertain, cKDTree(uncertain.xyz)) if len(uncertain.xyz) else None
    fits = [{}, {}]
    passed = {}
    for node in seeds:
        center = nodes.xyz[node]
        connection = joined_cortex(node, graph, FLANK_LENGTH_MM, *anchor_node_ids)
        if connection is None:
            continue
        evidence["bounded_connector_route_count"] += 1
        endpoints, route, path_length = connection
        nearby = np.asarray(cortex_tree.query_ball_point(center, EDGE_MM), dtype=int)
        if not len(nearby) or len(views(cortex.take(nearby), distinct)) < minimum_views:
            continue
        evidence["multi_view_route_count"] += 1
        flank = []
        for endpoint, anchor, tree, cache in zip(endpoints, anchors, anchor_trees, fits):
            if endpoint not in cache:
                cache[endpoint] = fit_flank(anchor, tree, nodes.xyz[endpoint], distinct, probe, minimum_views)
            flank.append(cache[endpoint])
        if any(f is None for f in flank):
            continue
        evidence["fitted_flank_route_count"] += 1
        (tp, n_tp), (sap, n_sap) = flank
        if (n_sap @ right < RULES["min_right_component"] or abs(n_tp @ right) > RULES["max_tp_right_component"]
                or abs(n_tp @ n_sap) > RULES["max_absolute_normal_dot"]):
            continue
        evidence["oriented_flank_route_count"] += 1
        tp_continuation, sap_continuation = tp.xyz - center, sap.xyz - center
        if (np.max(tp_continuation @ right) < RULES["min_directional_span_mm"] or
                np.max(sap_continuation @ superior) < RULES["min_directional_span_mm"] or
                tp_continuation.mean(0) @ right < -RULES["side_tolerance_mm"] or
                sap_continuation.mean(0) @ superior < -RULES["side_tolerance_mm"] or
                abs(tp_continuation.mean(0) @ n_tp) > EDGE_MM or
                abs(sap_continuation.mean(0) @ n_sap) > EDGE_MM):
            continue
        if unstable_levels(controversial, center, distinct):
            evidence["ambiguous_level_seed_count"] += 1
            continue
        passed[int(node)] = (tp, sap, nodes.take(route), cortex.take(nearby), path_length)
    evidence["supported_route_count"] = len(passed)
    if not passed:
        reason = "competing_level_near_transition" if evidence["ambiguous_level_seed_count"] else "no_supported_cortical_transition"
        return None, reason, evidence

    selected = np.asarray(sorted(passed), dtype=int)
    observed = nodes.xyz[selected]
    components = component_labels(observed, EDGE_MM)
    evidence["competing_region_count"] = len(np.unique(components))
    if evidence["competing_region_count"] != 1:
        return None, "competing_cortical_transitions", evidence
    diameter = float(pdist(observed).max()) if len(observed) > 1 else 0.0
    if not 0 < diameter <= RULES["max_region_diameter_mm"]:
        return None, "region_size_unresolved", evidence
    roles = ("TP", "SAP", "observed_cortical_route", "local_region_support")
    support = {}
    for i, role in enumerate(roles):
        rows = np.unique(np.concatenate([item[i].row for item in passed.values()]))
        support[role] = provenance(cloud.take(np.flatnonzero(np.isin(cloud.row, rows))), distinct)

    # Interpolated midpoints only fill sub-edge intervals between two observed region samples.
    observed = np.unique(observed, axis=0)
    adjacent = cKDTree(observed).query_pairs(EDGE_MM, output_type="ndarray")
    middle = observed[adjacent].mean(axis=1)
    infer = middle[cortex_tree.query(middle)[0] > CELL_MM]
    return {
        "scope": "Right L5 TP-root/lateral-SAP cortical transition candidate in actor robot-base mm",
        "observed_region_mm": observed.tolist(), "inferred_region_mm": np.unique(infer, axis=0).tolist(),
        "inference": "Midpoints of observed local surface pairs at most 1.5 mm apart, at most 0.75 mm from a measured endpoint.",
        "diameter_mm": diameter, "observed_region_samples": provenance(nodes.take(selected), distinct),
        "route_length_range_mm": [min(p[-1] for p in passed.values()), max(p[-1] for p in passed.values())],
        "support": support,
    }, None, evidence


def sensitivity_trials(cloud, frames, distinct, candidate):
    if candidate is not None:
        used_frames = np.unique(np.concatenate([v["frame_sequences"] for v in candidate["support"].values()]))
        assignments = assign_views(used_frames, distinct)
        for group in np.unique(assignments):
            exclude = used_frames[assignments == group]
            yield f"omit_view_{group}", cloud.take(~np.isin(cloud.frame, exclude)), 1
    phases = np.asarray([f["phase"] for f in frames])
    for phase in np.unique(phases):
        yield f"omit_phase_{phase}", cloud.take(phases[cloud.frame] != phase), 1

    probs = np.sort(cloud.probabilities, axis=1)
    near_tie = (probs[:, -1] - probs[:, -2]) < 0.10
    yield "omit_semantic_near_ties", cloud.take(~near_tie), RULES["min_views"]
    unassigned = (cloud.level == "AMBIGUOUS") & (cloud.role == UNCERTAIN)
    # This sensitivity trial uses the quarantined probability as a placeholder weight.
    stress_weight = np.where(unassigned & (cloud.confidence == 0),
                             cloud.probabilities.max(axis=1), cloud.confidence)
    yield "ambiguous_bone_as_l5_cortex", replace(
        cloud, level=np.where(unassigned, "L5", cloud.level),
        role=np.where(unassigned, OTHER, cloud.role), confidence=stress_weight), RULES["min_views"]

    origin = 1000 * np.asarray([f["actor_base_to_ee_position_m"] for f in frames])[cloud.frame]
    t = cloud.frame.astype(float)
    vector = np.column_stack((np.sin(t), np.cos(t), np.sin(2 * t)))
    vector /= np.linalg.norm(vector, axis=1, keepdims=True)
    for sign in (-1, 1):
        rotation = Rotation.from_rotvec(np.deg2rad(sign * .5) * vector)
        shifted = origin + rotation.apply(cloud.xyz - origin) + sign * vector
        yield f"pose_perturbation_{sign:+d}", replace(cloud, xyz=shifted), RULES["min_views"]


def region_distance(first, second):
    a, b = np.asarray(first["observed_region_mm"]), np.asarray(second["observed_region_mm"])
    return float(max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max()))


def localize(cloud, frames, *, sensitivity=True):
    """Run the same method for complete or ablated actor-visible 3D observations."""
    start = perf_counter()
    if not isinstance(cloud, Cloud):
        cloud = Cloud.from_arrays(cloud)
    usable = [f for f in frames if all(f[k] is True for k in ACQUISITION_KEYS)]
    axes = public_axes(usable, RULES)
    result = {"status": "UNRESOLVED", "candidate": None, "diagnostics": {}, "reason": None}
    if axes is None:
        result["reason"] = "actor_axes_unavailable"
    else:
        distinct = distinct_poses(frames, RULES)
        probe = np.asarray([project_image(f, (1, 1), RULES["pixel_mm"])[0, 0] for f in frames])
        candidate, reason, evidence = find_region(cloud, distinct, probe, axes, RULES["min_views"])
        result["geometry_runtime_ms"] = (perf_counter() - start) * 1000
        result["geometry"] = evidence
        result["public_axes"] = {"right": axes[0].tolist(), "superior": axes[1].tolist(),
                                 "cranial_fit_residual_index": axes[2]}
        result["reason"] = reason
        # Report label/pose/phase effects on abstention too.
        trials = sensitivity_trials(cloud, frames, distinct, candidate) if sensitivity else ()
        for name, changed, minimum_views in trials:
            trial, failed, diagnostic = find_region(changed, distinct, probe, axes, minimum_views)
            delta = region_distance(candidate, trial) if candidate is not None and trial is not None else None
            result["diagnostics"][name] = {"status": "GEOMETRIC_CANDIDATE" if trial else "UNRESOLVED",
                                           "reason": failed, "region_distance_mm": delta,
                                           "nearest_tp_sap_mm": diagnostic["nearest_tp_sap_mm"],
                                           "region_diameter_mm": trial["diameter_mm"] if trial else None}
        if candidate is not None:
            removals = [d for k, d in result["diagnostics"].items() if k.startswith("omit_view_")]
            if any(d["region_distance_mm"] is None or d["region_distance_mm"] > RULES["stability_mm"] for d in removals):
                result.update(reason="unstable_under_view_removal", proposed_candidate=candidate)
            else:
                result.update(status="GEOMETRIC_CANDIDATE", candidate=candidate)
    result["runtime_ms"] = (perf_counter() - start) * 1000
    return result


def predict(cloud_path, frames_path):
    frames = [json.loads(line) for line in frames_path.read_text().splitlines() if line.strip()]
    with np.load(cloud_path, allow_pickle=False) as arrays:
        cloud = Cloud.from_arrays(arrays)
    integrity = cloud_integrity(cloud, frames)
    return localize(cloud, frames) | {"integrity": integrity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cloud", "frames", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = predict(args.cloud, args.frames)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
