"""Find observed right L5 TP–SAP surface transitions in robot-base millimetres."""

from collections import Counter
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from two_stage_navigation.simple_landmark_mapping import project_image, transverse_cranial_coordinate


@dataclass(frozen=True)
class Surface:
    xyz: np.ndarray
    frame: np.ndarray
    confidence: np.ndarray
    pixels: np.ndarray

    def subset(self, mask):
        return Surface(self.xyz[mask], self.frame[mask], self.confidence[mask], self.pixels[mask])


def public_axes(frames, rules):
    transverse = [f for f in frames if f["phase"] == "transverse" and f["image_valid"]]
    spacing = rules["index_spacing_mm"]
    cranial_fit = transverse_cranial_coordinate(transverse, spacing, rules["max_axis_fit_residual_index"])
    if cranial_fit is None:
        return None
    positions = np.asarray([project_image(f, (1, 1), rules["pixel_mm"])[0, 0] for f in transverse])
    indices = np.asarray([f["executed_relative_pose_index"][:2] for f in transverse])
    directions, *_ = np.linalg.lstsq(indices - indices.mean(0), positions - positions.mean(0), rcond=None)
    superior = np.asarray(cranial_fit.direction)
    right = directions[0] / np.linalg.norm(directions[0])
    return right, superior, cranial_fit.maximum_fit_residual_index


def view_groups(frames, rules):
    ids = np.asarray([f["frame_sequence"] for f in frames], dtype=int)
    if not np.array_equal(ids, np.arange(len(frames))):
        raise ValueError("public frame sequence must be contiguous and ordered")
    positions = 1000 * np.asarray([f["actor_base_to_ee_position_m"] for f in frames])
    attitude = np.asarray([f["actor_base_to_ee_rotation_columns"] for f in frames]).reshape(-1, 9)
    angle = np.rad2deg(np.arccos(np.clip((attitude @ attitude.T - 1) / 2, -1, 1)))
    distance = np.linalg.norm(positions[:, None] - positions, axis=-1)
    hashes = np.asarray([f["frame_sha256"] for f in frames])
    distinct = (((distance >= rules["min_view_translation_mm"]) |
                 (angle >= rules["min_view_rotation_deg"])) &
                (hashes[:, None] != hashes))
    representatives, assignments = [], []
    for index in ids:
        if all(distinct[index, other] for other in representatives):
            representatives.append(int(index))
        assignments.append(next(other for other in representatives if not distinct[index, other]))
    return np.asarray(assignments)


def component_labels(xyz, width):
    pairs = cKDTree(xyz).query_pairs(width, output_type="ndarray")
    graph = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(xyz), len(xyz)))
    return connected_components(graph, directed=False, return_labels=True)[1]


def patch(surface, neighbors, seed, groups, probe, rules, minimum_views):
    if len(neighbors) < 3:
        return None, "too_few_surface_samples"
    xyz = surface.xyz[neighbors]
    labels = component_labels(xyz, rules["connection_mm"])
    neighbors = neighbors[labels == labels[np.searchsorted(neighbors, seed)]]
    xyz = surface.xyz[neighbors]
    if len(neighbors) < 3:
        return None, "disconnected_local_samples"
    sources = surface.frame[neighbors]
    group = groups[sources]
    independent, inverse = np.unique(group, return_inverse=True)
    if len(independent) < minimum_views:
        return None, "insufficient_distinct_views"
    weights = surface.confidence[neighbors].astype(float)
    weights /= np.bincount(inverse, weights=weights)[inverse]
    weights /= weights.sum()
    center = weights @ xyz
    centered = xyz - center
    value, vectors = np.linalg.eigh((centered * weights[:, None]).T @ centered)
    if value[1] < rules["min_spread_mm"] ** 2 or value[0] > rules["max_fit_thickness_mm"] ** 2:
        return None, "not_a_locally_supported_surface"
    normal = vectors[:, 0]
    toward_sensor = weights @ probe[sources] - center
    normal *= 1 if np.dot(normal, toward_sensor) >= 0 else -1
    return (neighbors, normal), None


def transition(tp, sap, right, superior, groups, probe, rules, minimum_views):
    evidence = {"tp_points": len(tp.xyz), "sap_points": len(sap.xyz), "support": {},
                "within_gap_pairs": 0, "patch_pair_count": 0, "oriented_pair_count": 0,
                "directional_pair_count": 0, "components": []}
    if not len(tp.xyz) or not len(sap.xyz):
        return evidence, "missing_semantic_surface"
    tp_tree, sap_tree = cKDTree(tp.xyz), cKDTree(sap.xyz)
    distances = tp_tree.query(sap.xyz)[0]
    sap_index = int(np.argmin(distances))
    evidence["nearest_gap_mm"] = float(distances[sap_index])
    nearest_tp = int(tp_tree.query(sap.xyz[sap_index])[1])
    evidence["nearest_pair_observations"] = {"TP": observations(tp, [nearest_tp], groups),
                                             "SAP": observations(sap, [sap_index], groups)}
    pairs = [(i, j) for j, indices in enumerate(sap_tree.query_ball_tree(tp_tree, rules["max_gap_mm"]))
             for i in indices]
    evidence["within_gap_pairs"] = len(pairs)
    if not pairs:
        return evidence, "no_local_tp_sap_continuity"
    valid = []
    patches = []
    for slot, (name, surface, tree) in enumerate((("TP", tp, tp_tree), ("SAP", sap, sap_tree))):
        accepted, failures = {}, Counter()
        for seed in sorted({row[slot] for row in pairs}):
            indices = np.asarray(sorted(tree.query_ball_point(surface.xyz[seed], rules["fit_radius_mm"])))
            geometry, reason = patch(surface, indices, seed, groups, probe, rules, minimum_views)
            if geometry is None:
                failures[reason] += 1
            else:
                accepted[seed] = geometry
        evidence["support"][name] = {"fitted_patches": len(accepted), "reasons": dict(failures)}
        patches.append(accepted)
    for i, j in pairs:
        if i not in patches[0] or j not in patches[1]:
            continue
        evidence["patch_pair_count"] += 1
        tp_neighbors, tp_normal = patches[0][i]
        sap_neighbors, sap_normal = patches[1][j]
        if (np.dot(sap_normal, right) < rules["min_right_component"] or
                abs(np.dot(tp_normal, right)) > rules["max_tp_right_component"] or
                abs(np.dot(sap_normal, tp_normal)) > rules["max_absolute_normal_dot"]):
            continue
        evidence["oriented_pair_count"] += 1
        separation = tp.xyz[i] - sap.xyz[j]
        if (separation @ right < -rules["side_tolerance_mm"] or
                separation @ superior > rules["side_tolerance_mm"] or
                np.max((tp.xyz[tp_neighbors] - tp.xyz[i]) @ right) < rules["min_directional_span_mm"] or
                np.max((sap.xyz[sap_neighbors] - sap.xyz[j]) @ superior) < rules["min_directional_span_mm"]):
            continue
        valid.append((i, j))
    evidence["directional_pair_count"] = len(valid)
    if not valid:
        return evidence, "no_supported_root_and_lateral_neck_transition"
    seeds = np.unique([j for _, j in valid])
    labels = component_labels(sap.xyz[seeds], rules["connection_mm"])
    for label in np.unique(labels):
        region = seeds[labels == label]
        links = [(i, j) for i, j in valid if j in region]
        tp_indices = np.unique(np.concatenate([patches[0][i][0] for i, _ in links]))
        sap_indices = np.unique(np.concatenate([patches[1][j][0] for _, j in links]))
        xyz = np.unique(sap.xyz[region], axis=0)
        diameter = np.linalg.norm(xyz[:, None] - xyz, axis=2).max()
        evidence["components"].append({
            "observed_region_mm": xyz.tolist(), "inferred_points_mm": [], "diameter_mm": float(diameter),
            "support": {"TP": observations(tp, tp_indices, groups), "SAP": observations(sap, sap_indices, groups)},
            "region_samples": observations(sap, region, groups),
        })
    if len(evidence["components"]) != 1:
        return evidence, "competing_surface_regions"
    region = evidence["components"][0]
    if len(region["observed_region_mm"]) < 2 or region["diameter_mm"] > rules["max_region_diameter_mm"]:
        return evidence, "region_size_unresolved"
    return evidence, None


def observations(surface, indices, groups):
    indices = np.asarray(indices, dtype=int)
    frames = surface.frame[indices]
    return {"frames": np.unique(frames).tolist(), "view_groups": np.unique(groups[frames]).tolist(),
            "points": [{"frame": int(f), "pixel_row_col": pixel.tolist(), "xyz_mm": xyz.tolist(),
                        "confidence": float(prob)}
                       for f, pixel, xyz, prob in zip(frames, surface.pixels[indices],
                                                      surface.xyz[indices], surface.confidence[indices])]}


def distance_both_ways(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if not len(a) or not len(b):
        raise ValueError("region comparison requires two nonempty point sets")
    return cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]


def localize(tp, sap, frames, rules):
    start = perf_counter()
    axes = public_axes(frames, rules)
    if axes is None:
        return {"status": "UNRESOLVED", "reason": "missing_actor_visible_anatomical_axes", "candidate": None,
                "runtime_ms": (perf_counter() - start) * 1000}
    right, superior, residual = axes
    groups = view_groups(frames, rules)
    probe = np.asarray([project_image(f, (1, 1), rules["pixel_mm"])[0, 0] for f in frames])
    evidence, reason = transition(tp, sap, right, superior, groups, probe, rules, rules["min_views"])
    result = {"observed_coverage": evidence, "holdouts": {}, "candidate": None,
              "public_axes": {"right": right.tolist(), "superior": superior.tolist(),
                              "fit_residual_index": residual}}
    phases = np.asarray([f["phase"] for f in frames])
    masks = {f"phase_{phase}": phases == phase for phase in np.unique(phases)}
    if reason is None:
        region = evidence["components"][0]
        used = sorted({g for support in region["support"].values() for g in support["view_groups"]})
        masks.update({f"view_{group}": groups == group for group in used})
    for name, mask in masks.items():
        held, held_reason = transition(tp.subset(~mask[tp.frame]), sap.subset(~mask[sap.frame]),
                                       right, superior, groups, probe, rules, 1)
        check = {k: held[k] for k in ("tp_points", "sap_points", "within_gap_pairs", "directional_pair_count")}
        check.update(reason=held_reason, nearest_gap_mm=held.get("nearest_gap_mm"))
        if reason is None and held_reason is None:
            a, b = distance_both_ways(region["observed_region_mm"], held["components"][0]["observed_region_mm"])
            check["max_bidirectional_distance_mm"] = float(max(a.max(), b.max()))
        result["holdouts"][name] = check
    if reason is None:
        stable = all(check.get("max_bidirectional_distance_mm", np.inf) <= rules["stability_mm"]
                     for name, check in result["holdouts"].items() if name.startswith("view_"))
        reason = None if stable else "unstable_under_view_removal"
        if stable:
            result["candidate"] = region
    result.update(status="CANDIDATE" if reason is None else "UNRESOLVED", reason=reason,
                  runtime_ms=(perf_counter() - start) * 1000)
    return result


def predict(cloud_path, frames_path, rules, frame_count):
    import json

    frames = [json.loads(line) for line in frames_path.read_text().splitlines()][:frame_count]
    required = ("image_valid", "contact_ok", "producer_advanced", "transition_valid")
    if (len(frames) != frame_count or
            any(not all(f[key] is True for key in required) for f in frames)):
        raise ValueError("invalid acquisition in the sealed cohort")
    with np.load(cloud_path, allow_pickle=False) as arrays:
        xyz = arrays["xyz_mm"]
        sequence = arrays["frame_sequence"]
        identity = arrays["identity"]
        confidence = arrays["confidence"]
        pixels = arrays["pixel_row_col"]
    shape = tuple(rules["image_shape"])
    if (xyz.shape != (len(sequence), 3) or pixels.shape != (len(sequence), 2) or
            identity.shape != sequence.shape or confidence.shape != sequence.shape or
            not np.isfinite(xyz).all() or not np.isfinite(confidence).all() or
            np.any((confidence <= 0) | (confidence > 1)) or
            np.any((sequence < 0) | (sequence >= len(frames))) or
            np.any(pixels < 0) or np.any(pixels >= shape)):
        raise ValueError("invalid cloud arrays")
    projection_errors = []
    for index in np.unique(sequence):
        frame_mask = sequence == index
        rows, cols = pixels[frame_mask].T
        calibrated = project_image(frames[index], shape, rules["pixel_mm"])[rows, cols]
        projection_errors.extend(np.linalg.norm(calibrated - xyz[frame_mask], axis=1))
    max_error = max(projection_errors, default=0)
    if max_error > rules["max_projection_parity_mm"]:
        raise ValueError(f"public reprojection does not match cached point cloud: {max_error} mm")
    surface = Surface(xyz, sequence, confidence, pixels)
    result = localize(surface.subset(identity == "RIGHT_L5_TP"),
                      surface.subset(identity == "RIGHT_L5_SAP"), frames, rules)
    result["integrity"] = {"max_public_reprojection_error_mm": float(max_error),
                           "supplied_frames": len(frames), "supplied_points": len(xyz)}
    return result


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cloud", "frames", "rules", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--frame-count", type=int, required=True)
    args = parser.parse_args()
    result = predict(args.cloud, args.frames, json.loads(args.rules.read_text()), args.frame_count)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
