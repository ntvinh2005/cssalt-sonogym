"""Evaluator-only geometry for the sealed s0010 reconstruction cohort."""

from collections import defaultdict

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from workflows.reconstruction_s0010.dense import (
    IDENTITIES, cloud_from_arrays, distinct_pose_matrix, nearest, summarize_distances, voxel_fuse,
)
from two_stage_navigation.l4_medial_branch_target import build_target_contract_geometry
from two_stage_navigation.sonoanatomy_labels import first_bone_surface


def transform(frame, replay, spacing):
    rotation = (np.asarray(frame["actor_base_to_ee_rotation_columns"])
                @ np.asarray(replay["human_to_ee_rotation_columns"]).T)
    origin = 1000 * (np.asarray(frame["actor_base_to_ee_position_m"])
                     - rotation @ replay["human_to_ee_position_m"])
    return lambda index: np.asarray(index) * spacing @ rotation.T + origin


def prepare_cloud(data):
    return cloud_from_arrays(data["xyz_mm"],
                             [IDENTITIES.index(label) for label in data["identity"]],
                             data["frame_sequence"], data["confidence"])


def subset(cloud, mask):
    return cloud_from_arrays(cloud.xyz_mm[mask], cloud.code[mask], cloud.frame_sequence[mask],
                             cloud.confidence[mask])


def coverage(points, other, gate):
    d = nearest(points, other)
    return {
        "covered": int(np.count_nonzero(d <= gate)), "total": len(d),
        "median_nearest_mm": float(np.median(d)) if len(d) and np.isfinite(d).all() else None,
        "maximum_nearest_mm": float(np.max(d)) if len(d) and np.isfinite(d).all() else None,
    }


def observed_coverage(reference, candidate, gate):
    rows = {identity: coverage(reference.xyz_mm[reference.code == code],
                               candidate.xyz_mm[candidate.code == code], gate)
            for code, identity in enumerate(IDENTITIES)}
    return {
        "covered": sum(v["covered"] for v in rows.values()),
        "total": sum(v["total"] for v in rows.values()), "by_identity": rows,
    }


def reference_coverage(reference_boundary, observed, gate):
    rows = {identity: coverage(boundary, observed.xyz_mm[observed.code == code], gate)
            for code, (identity, boundary) in enumerate(reference_boundary.items())}
    return {
        "covered": sum(v["covered"] for v in rows.values()),
        "total": sum(v["total"] for v in rows.values()), "by_identity": rows,
    }


def view_support(cloud, indices, distinct, fusion):
    if fusion is None:
        source = np.unique(cloud.frame_sequence[indices]).tolist()
    else:
        offsets = fusion["selected_frame_offsets"]
        sequences = fusion["selected_frame_sequences"]
        source = sorted({int(frame) for i in indices for frame in sequences[offsets[i]:offsets[i+1]]})
    retained = []
    for frame in source:
        if all(distinct[frame, other] for other in retained):
            retained.append(frame)
    return {"unique_sequences": source, "distinct_sequences": retained, "distinct_views": len(retained)}


def score_region(cloud, cloud_in, oracle, all_visible, boundary, distances, distinct, fusion, gate):
    local = subset(cloud, cloud_in)
    own_semantics = cloud_in[distances["semantic_vertex_ids"]]
    near = distances["semantic_distances_mm"][own_semantics]
    wrong = distances["semantic_wrong_class"][own_semantics]
    correct = distances["semantic_correct"][own_semantics]
    by_identity = {}
    pooled_surface = []
    pooled_reference = []
    for code, (identity, surface) in enumerate(boundary.items()):
        current = local.xyz_mm[local.code == code]
        by_shape = nearest(current, surface)
        by_ref = nearest(current, oracle.xyz_mm[oracle.code == code])
        pooled_surface.extend(by_shape)
        pooled_reference.extend(by_ref)
        local_view = np.flatnonzero(cloud_in & (cloud.code == code))
        membership = own_semantics & (cloud.code[distances["semantic_vertex_ids"]] == code)
        class_wrong = distances["semantic_wrong_class"][membership]
        class_correct = distances["semantic_correct"][membership]
        by_identity[identity] = {
            "points": len(current), "surface_error_to_local_ct_boundary": summarize_distances(by_shape, gate),
            "distance_to_local_eligible_visible_surface": summarize_distances(by_ref, gate),
            "same_frame_semantics": {"wrong_nearest_identity": int(class_wrong.sum()),
                                      "correct_within_gate": int(class_correct.sum()),
                                      "unmatched": int(len(class_wrong) - class_wrong.sum() - class_correct.sum()),
                                      "comparisons": len(class_wrong)},
            "views": view_support(cloud, local_view, distinct, fusion),
        }
    return {
        "points": int(cloud_in.sum()),
        "coverage_eligible_visible": observed_coverage(oracle, local, gate),
        "coverage_all_visible": observed_coverage(all_visible, local, gate),
        "full_ct_boundary_to_reconstruction": reference_coverage(boundary, local, gate),
        "surface_error_to_local_ct_boundary": summarize_distances(np.asarray(pooled_surface), gate),
        "distance_to_local_eligible_visible_surface": summarize_distances(np.asarray(pooled_reference), gate),
        "same_frame_semantics": {"wrong_nearest_identity": int(wrong.sum()),
                                  "correct_within_gate": int(correct.sum()),
                                  "unmatched": int(np.count_nonzero(near > gate)),
                                  "comparisons": len(near)},
        "views": view_support(cloud, np.flatnonzero(cloud_in), distinct, fusion),
        "by_identity": by_identity,
    }


def feature_distance(cloud, near):
    tp = cloud.xyz_mm[near & (cloud.code == IDENTITIES.index("RIGHT_L5_TP"))]
    sap = cloud.xyz_mm[near & (cloud.code == IDENTITIES.index("RIGHT_L5_SAP"))]
    if not len(tp) or not len(sap):
        return {"tp_points": len(tp), "sap_points": len(sap), "closest_visible_pair_mm": None}
    distances = cKDTree(tp).query(sap)[0]
    return {"tp_points": len(tp), "sap_points": len(sap),
            "closest_visible_pair_mm": float(distances.min())}


def leave_views(frame_rows, own, oracle, target_tree, map_config, radius, gate, fuse):
    """Use recorded source frames and hold the original evaluator region fixed."""
    distinct = distinct_pose_matrix(frame_rows, map_config)
    eligible_local = subset(oracle, target_tree.query(oracle.xyz_mm)[0] <= radius)
    local_frames = sorted({int(i) for i in own.frame_sequence[target_tree.query(own.xyz_mm)[0] <= radius]})
    frame_to_phase = {f["frame_sequence"]: f["phase"] for f in frame_rows}
    trials = [("baseline", [])] + [(f"leave_frame_{i}", [i]) for i in local_frames]
    phases = sorted(set(frame_to_phase.values()))
    trials += [(f"leave_family_{phase}", [i for i, family in frame_to_phase.items() if family == phase])
               for phase in phases]
    outputs = {}
    for name, frames in trials:
        retained = subset(own, ~np.isin(own.frame_sequence, frames))
        filtered = voxel_fuse(retained, distinct, 1.5)[0] if fuse else retained
        focus = target_tree.query(filtered.xyz_mm)[0] <= radius
        outputs[name] = {
            "frames_removed": frames, "observed_feature": feature_distance(filtered, focus),
            "local_observed_coverage": observed_coverage(eligible_local, subset(filtered, focus), gate),
        }
    return outputs


def first_bone_voxels(coordinates, raw_labels, bone_ids, shape):
    indices = np.trunc(coordinates).astype(np.int64).reshape(raw_labels.shape[1], raw_labels.shape[0], 3)
    by_pixel = indices.transpose(1, 0, 2)
    visible = by_pixel[first_bone_surface(raw_labels, bone_ids)]
    valid = np.all((visible >= 0) & (visible < shape), axis=1)
    return visible[valid]


def crop_geometry(sap, tp, affine, margin=5):
    combined = np.argwhere(sap | tp)
    ranges = tuple(slice(max(int(lo) - margin, 0), min(int(hi) + margin + 1, lim))
                   for lo, hi, lim in zip(combined.min(axis=0), combined.max(axis=0), sap.shape))
    shift = np.asarray([r.start for r in ranges])
    cropped_affine = np.asarray(affine).copy()
    cropped_affine[:3, 3] = nib.affines.apply_affine(affine, shift)
    return ranges, cropped_affine


def hidden_sap_variants(sap, tp, source_right, protected, affine):
    """Vary a semantic frontier within fixed bony geometry and protect visible evidence."""
    conn = ndimage.generate_binary_structure(3, 1)
    baseline = build_target_contract_geometry(sap_mask=sap, tp_mask=tp, affine=affine)
    spacing = nib.affines.voxel_sizes(affine)
    allowed = source_right & ~ndimage.binary_dilation(protected, structure=conn) & ~tp
    dist = ndimage.distance_transform_edt(~baseline.tp_surface, sampling=spacing)
    allowed &= dist <= baseline.core_separation_limit_mm + 2 * np.min(spacing)
    base_uncertainty = ndimage.binary_dilation(sap | tp, structure=conn) & ~(sap | tp)
    components = ndimage.label(sap, structure=conn)[1]
    variants = []
    for direction in ("expand", "retract"):
        working = sap.copy()
        for step in (1, 2):
            if direction == "expand":
                working |= ndimage.binary_dilation(working, structure=conn) & allowed
            else:
                working &= ~(allowed & ~ndimage.binary_erosion(working, structure=conn))
            changed = np.count_nonzero(working ^ sap)
            variant_components = ndimage.label(working, structure=conn)[1]
            if not changed or variant_components != components:
                continue
            uncertain = ndimage.binary_dilation(working | tp, structure=conn) & ~(working | tp)
            assert not np.any((working ^ sap) & protected)
            assert not np.any((uncertain ^ base_uncertainty) & protected)
            ref = build_target_contract_geometry(sap_mask=working, tp_mask=tp, affine=affine)
            name = f"{direction}_{step}_voxel"
            variants.append((name, working.copy(), ref))
    return baseline, variants
