"""Reconstruct the recorded TP and SAP interfaces as semantic point clouds."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import yaml

from replay import CONVERSION, context, route
from run_simple_landmark_mapping import load_evaluator, load_public, read_jsonl, sha256, write_json
from two_stage_navigation.simple_landmark_mapping import (
    CLASSES, LEVELS, Stage1Region, extract_observations, transverse_cranial_coordinate,
)


IDENTITIES = ("RIGHT_L4_TP", "RIGHT_L4_SAP", "RIGHT_L5_TP", "RIGHT_L5_SAP")
COLORS = ((36, 138, 204), (210, 73, 98), (20, 149, 111), (240, 148, 39))
CONNECTIVITY = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class Cloud:
    xyz_mm: np.ndarray
    code: np.ndarray
    frame_sequence: np.ndarray
    confidence: np.ndarray

    def arrays(self):
        return {
            "xyz_mm": self.xyz_mm,
            "identity": np.asarray(IDENTITIES)[self.code],
            "frame_sequence": self.frame_sequence,
            "confidence": self.confidence,
        }


def cloud_from_arrays(xyz, codes, frames, confidence):
    return Cloud(
        np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
        np.asarray(codes, dtype=np.uint8), np.asarray(frames, dtype=np.int32),
        np.asarray(confidence, dtype=np.float32),
    )


def nearest(points, reference):
    if not len(reference):
        return np.full(len(points), np.inf)
    return cKDTree(reference).query(points)[0]


def summarize_distances(distance, gate):
    finite = distance[np.isfinite(distance)]
    return {
        "count": len(distance),
        "finite_count": len(finite),
        "within_mm": int(np.count_nonzero(distance <= gate)),
        "median_mm": float(np.median(finite)) if len(finite) else None,
        "p95_mm": float(np.quantile(finite, .95)) if len(finite) else None,
        "outliers_or_missing": int(np.count_nonzero(distance > gate)),
    }


def arrays_from_blocks(blocks):
    if not blocks:
        return cloud_from_arrays([], [], [], []), np.zeros((0, 2), dtype=np.int16)
    xyz, codes, frames, confidence, pixels = (np.concatenate(part) for part in zip(*blocks))
    return cloud_from_arrays(xyz, codes, frames, confidence), pixels


def most_proximal_pixels(pixels):
    order = np.lexsort((pixels[:, 0], pixels[:, 1]))
    ordered = pixels[order]
    first = np.r_[True, ordered[1:, 1] != ordered[:-1, 1]]
    return ordered[first]


def public_context(path, protocol, manifest):
    frames, images, valid, coordinates, _, _ = load_public(
        path, protocol["geometry"]["ultrasound_pixel_resolution_mm"],
    )
    declared = json.loads((path / "waypoints.json").read_text())[:102]
    waypoint_hash = hashlib.sha256((json.dumps(declared, indent=2) + "\n").encode()).hexdigest()
    if waypoint_hash != manifest["common_waypoints_sha256"]:
        raise ValueError(f"waypoints do not match the frozen cohort: {path}")
    valid = [i for i in valid if frames[i]["waypoint_index"] < 102]
    actor = json.loads((path / "actor_result.json").read_text())
    fit = transverse_cranial_coordinate(
        [frames[i] for i in valid], protocol["geometry"]["label_resolution_mm"],
        protocol["sonoanatomy_scan"]["translation_step_index"] / 2,
    )
    return {"frames": frames, "images": images, "valid": valid, "coords": coordinates,
            "region": Stage1Region(**actor["stage1_region"]), "fit": fit}


def raw_cloud(public, *, probabilities=None, thresholds=None, oracle_labels=None):
    if (probabilities is None) == (oracle_labels is None):
        raise ValueError("supply either predictions or oracle labels")
    blocks = []
    admissions = defaultdict(Counter)
    ray_classes = defaultdict(set)
    for offset, sequence in enumerate(public["valid"]):
        frame, coords = public["frames"][sequence], public["coords"][sequence]
        for level_index, level in enumerate(LEVELS):
            prob = probabilities[level][offset] if probabilities is not None else None
            winner = prob.argmax(0) if prob is not None else oracle_labels[level][sequence]
            rules = thresholds[level] if thresholds is not None else {}
            rows, accepted = extract_observations(winner, prob, rules, frame, coords)
            groups = {value: ndimage.label(accepted == value, structure=CONNECTIVITY)[0]
                      for value in (1, 2)}
            for row in rows:
                value = CLASSES.index(row.semantic_class)
                code = 2 * level_index + value - 1
                identity = IDENTITIES[code]
                n = route(row, frame, level, public, tp_all=True)
                admission = {0: "excluded", 1: "eligible", 2: "ambiguous"}[n]
                admissions[identity][admission + "_components"] += 1
                admissions[identity][admission + "_pixels"] += row.pixel_count
                if n != 1:
                    continue
                group = groups[value]
                pixels = np.argwhere(group == group[row.row, row.column]).astype(np.int16)
                if len(pixels) != row.pixel_count:
                    raise ValueError(f"component pixel count changed at {sequence}/{identity}")
                if prob is not None:
                    pixels = most_proximal_pixels(pixels)
                indexing = tuple(pixels.T)
                confidence = prob[value][indexing] if prob is not None else np.ones(len(pixels))
                blocks.append((
                    coords[indexing], np.full(len(pixels), code, dtype=np.uint8),
                    np.full(len(pixels), sequence, dtype=np.int32), confidence, pixels,
                ))
                for r, c in pixels:
                    ray_classes[(sequence, int(c))].add((int(r), code))
    cloud, pixels = arrays_from_blocks(blocks)
    ray_conflicts = sum(len({code for _, code in points}) > 1 for points in ray_classes.values())
    return cloud, pixels, {key: dict(admissions[key]) for key in IDENTITIES}, ray_conflicts


def distinct_pose_matrix(frames, map_config):
    pos = 1000 * np.asarray([f["actor_base_to_ee_position_m"] for f in frames])
    rotations = np.asarray([f["actor_base_to_ee_rotation_columns"] for f in frames]).reshape(-1, 9)
    translation = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    cosine = (rotations @ rotations.T - 1) / 2
    angle = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    hashes = np.asarray([f["frame_sha256"] for f in frames])
    same_hash = hashes[:, None] == hashes[None, :]
    return ((translation >= map_config["minimum_view_translation_mm"]) |
            (angle >= map_config["minimum_view_rotation_deg"])) & ~same_hash


def voxel_fuse(cloud, distinct, voxel_mm):
    if voxel_mm <= 0:
        raise ValueError("voxel width must be positive")
    voxels = np.floor(cloud.xyz_mm / voxel_mm).astype(int)
    groups, voxel_classes = defaultdict(list), defaultdict(set)
    for i, (voxel, code) in enumerate(zip(voxels, cloud.code)):
        cell = tuple(voxel)
        groups[(cell, int(code))].append(i)
        voxel_classes[cell].add(int(code))
    xyz, codes, frames, confidence = [], [], [], []
    raw_indices, raw_end, source_frames, source_end = [], [0], [], [0]
    support_counts, spreads, conflict_flags, cells = [], [], [], []
    for (cell, code), indices in sorted(groups.items()):
        views = defaultdict(list)
        for i in indices:
            views[int(cloud.frame_sequence[i])].append(i)
        frame_conf = {seq: float(np.mean(cloud.confidence[ids])) for seq, ids in views.items()}
        selected = []
        for seq in sorted(views, key=lambda seq: (-frame_conf[seq], seq)):
            if all(distinct[seq, previous] for previous in selected):
                selected.append(seq)
        means = np.asarray([np.mean(cloud.xyz_mm[views[seq]], axis=0) for seq in selected])
        center = np.mean(means, axis=0)
        xyz.append(center)
        codes.append(code)
        frames.append(min(views))
        confidence.append(np.mean([frame_conf[seq] for seq in selected]))
        cells.append(cell)
        raw_indices.extend(indices)
        raw_end.append(len(raw_indices))
        source_frames.extend(selected)
        source_end.append(len(source_frames))
        support_counts.append(len(selected))
        spreads.append(np.sqrt(np.mean(np.sum((means - center) ** 2, axis=1))))
        conflict_flags.append(len(voxel_classes[cell]) > 1)
    return cloud_from_arrays(xyz, codes, frames, confidence), {
        "voxel_ijk": np.asarray(cells, dtype=np.int32).reshape(-1, 3),
        "raw_indices": np.asarray(raw_indices, dtype=np.int32),
        "raw_offsets": np.asarray(raw_end, dtype=np.int32),
        "selected_frame_sequences": np.asarray(source_frames, dtype=np.int32),
        "selected_frame_offsets": np.asarray(source_end, dtype=np.int32),
        "unique_viewpoints": np.asarray(support_counts, dtype=np.int16),
        "view_rms_mm": np.asarray(spreads, dtype=np.float32),
        "conflicting_semantics": np.asarray(conflict_flags, dtype=bool),
    }


def all_visible_cloud(ctx):
    blocks = []
    for seq in ctx["valid"]:
        for level_index, level in enumerate(LEVELS):
            for value in (1, 2):
                pixels = np.argwhere(ctx["labels"][level][seq] == value)
                blocks.append((
                    ctx["coords"][seq][tuple(pixels.T)],
                    np.full(len(pixels), 2 * level_index + value - 1, dtype=np.uint8),
                    np.full(len(pixels), seq, dtype=np.int32), np.ones(len(pixels)), pixels,
                ))
    return arrays_from_blocks(blocks)[0]


def full_ct_geometry(path, first_frame, volume_masks, label_resolution_mm):
    poses = read_jsonl(path / "privileged/full_pose_replay.jsonl")
    pose = poses[first_frame["frame_sequence"]]
    rotation = (np.asarray(first_frame["actor_base_to_ee_rotation_columns"])
                @ np.asarray(pose["human_to_ee_rotation_columns"]).T)
    origin = 1000 * (
        np.asarray(first_frame["actor_base_to_ee_position_m"])
        - rotation @ pose["human_to_ee_position_m"]
    )
    result = {}
    for level in LEVELS:
        for semantic in ("TP", "SAP"):
            mask = volume_masks[level][semantic.lower()]
            surface = mask & ~ndimage.binary_erosion(mask)
            result[f"RIGHT_{level}_{semantic}"] = (
                np.argwhere(surface) * label_resolution_mm @ rotation.T + origin
            )
    return result


def frame_matches(cloud, visible, gate, fusion):
    if fusion is None:
        vertex_ids = np.arange(len(cloud.code))
        sequences = cloud.frame_sequence
    else:
        vertex_ids = np.repeat(np.arange(len(cloud.code)), fusion["unique_viewpoints"])
        sequences = fusion["selected_frame_sequences"]
    dist = np.full(len(sequences), np.inf)
    match = np.full(len(sequences), -1, dtype=np.int8)
    for seq in np.unique(sequences):
        slots = np.flatnonzero(sequences == seq)
        ref = visible.frame_sequence == seq
        if ref.any():
            distances, indices = cKDTree(visible.xyz_mm[ref]).query(cloud.xyz_mm[vertex_ids[slots]])
            dist[slots] = distances
            match[slots] = visible.code[ref][indices]
    correct = (dist <= gate) & (match == cloud.code[vertex_ids])
    wrong = (dist <= gate) & ~correct
    return vertex_ids, sequences, dist, correct, wrong


def cross_view_distances(cloud, distinct):
    result = np.full(len(cloud.code), np.inf)
    for code, seq in set(zip(cloud.code.tolist(), cloud.frame_sequence.tolist())):
        source = (cloud.code == code) & (cloud.frame_sequence == seq)
        other = (cloud.code == code) & distinct[seq, cloud.frame_sequence]
        result[source] = nearest(cloud.xyz_mm[source], cloud.xyz_mm[other])
    return result


def evaluate(cloud, oracle, visible, ct, gate, distinct, fusion):
    distances = {}
    metrics = {}
    for code, identity in enumerate(IDENTITIES):
        observed = cloud.xyz_mm[cloud.code == code]
        ref = oracle.xyz_mm[oracle.code == code]
        all_ref = visible.xyz_mm[visible.code == code]
        distance_sets = {
            "output_to_eligible_oracle": nearest(observed, ref),
            "eligible_oracle_to_output": nearest(ref, observed),
            "all_visible_oracle_to_output": nearest(all_ref, observed),
            "output_to_boundary_voxel_positions": nearest(observed, ct[identity]),
            "boundary_voxel_positions_to_output": nearest(ct[identity], observed),
        }
        metrics[identity] = {key: summarize_distances(value, gate)
                             for key, value in distance_sets.items()}
        for key, value in distance_sets.items():
            distances[f"{identity}__{key}"] = value
    vertex_ids, source_frames, same_frame_d, correct, wrong = frame_matches(cloud, visible, gate, fusion)
    for code, identity in enumerate(IDENTITIES):
        slots = cloud.code[vertex_ids] == code
        matched = correct[slots]
        mismatched = wrong[slots]
        metrics[identity]["semantic_correctness"] = {
            "correct": int(matched.sum()), "wrong_class": int(mismatched.sum()),
            "outside_gate_or_unmatched": int(slots.sum() - matched.sum() - mismatched.sum()),
            "total": int(slots.sum()),
        }
    distances.update(
        semantic_vertex_ids=vertex_ids, semantic_frame_sequences=source_frames,
        semantic_distances_mm=same_frame_d, semantic_correct=correct, semantic_wrong_class=wrong,
    )
    if fusion is None:
        cross_view = cross_view_distances(cloud, distinct)
        distances["distinct_view_nearest_same_class_mm"] = cross_view
        for code, identity in enumerate(IDENTITIES):
            metrics[identity]["distinct_view_nearest_same_class"] = summarize_distances(
                cross_view[cloud.code == code], gate,
            )
    else:
        for code, identity in enumerate(IDENTITIES):
            selected = cloud.code == code
            support = fusion["unique_viewpoints"][selected]
            rms = fusion["view_rms_mm"][selected][support >= 2]
            metrics[identity]["voxel_view_support"] = {
                "one_view": int(np.count_nonzero(support == 1)),
                "multiple_views": int(np.count_nonzero(support >= 2)),
                "cross_view_rms": summarize_distances(rms, gate),
                "points_with_competing_label": int(fusion["conflicting_semantics"][selected].sum()),
            }
    return metrics, distances


def write_ply(path, cloud, unique_views=None, conflict=None):
    if unique_views is None:
        unique_views = np.ones(len(cloud.code), dtype=np.int16)
    if conflict is None:
        conflict = np.zeros(len(cloud.code), dtype=bool)
    with path.open("w", encoding="ascii") as handle:
        handle.write(
            "ply\nformat ascii 1.0\ncomment coordinate_frame robot_base_mm\n"
            f"element vertex {len(cloud.code)}\n"
            "property double x\nproperty double y\nproperty double z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "property uchar semantic_id\nproperty float confidence\n"
            "property int frame_sequence\nproperty ushort distinct_views\n"
            "property uchar competing_label\nend_header\n"
        )
        for xyz, code, sequence, confidence, views, competing in zip(
            cloud.xyz_mm, cloud.code, cloud.frame_sequence, cloud.confidence, unique_views, conflict
        ):
            r, g, b = COLORS[code]
            x, y, z = xyz
            handle.write(
                f"{x:.7f} {y:.7f} {z:.7f} {r} {g} {b} {code} {confidence:.6f} "
                f"{sequence} {views} {int(competing)}\n"
            )


def load_cache(cache, manifest_hash, models, seed, valid, shape):
    root = cache / str(seed)
    receipt = json.loads((root / "receipt.json").read_text())
    if (receipt["manifest_sha256"] != manifest_hash or receipt["valid_sequences"] != valid
            or receipt["models"] != models or receipt["seed"] != seed
            or receipt["inference_batch_size"] != 1):
        raise ValueError(f"cache receipt does not describe the requested replay: {root}")
    probabilities = {}
    for level in LEVELS:
        artifact = root / f"{level}.npy"
        expected = receipt["arrays"][level]["sha256"]
        if sha256(artifact) != expected:
            raise ValueError(f"probability artifact checksum mismatch: {artifact}")
        array = np.load(artifact, mmap_mode="r")
        if array.dtype != np.float32 or array.shape != (len(valid), 4, *shape):
            raise ValueError(f"unexpected probability shape or dtype: {artifact}")
        probabilities[level] = array
    return probabilities, receipt["thresholds"], receipt


def evaluate_variant(directory, method, seed, cloud, pixels, admissions, ray_conflicts, fusion,
                     oracle, visible, ct, gate, distinct, elapsed, prep_seconds, projection):
    out = directory / str(seed) / method
    out.mkdir(parents=True, exist_ok=False)
    predicted = method in ("B1", "B3")
    dest = out / "cloud.npz"
    arrays = cloud.arrays()
    if fusion is None:
        arrays["pixel_row_col"] = pixels
    else:
        arrays.update(fusion)
    np.savez_compressed(dest, **arrays)
    write_ply(out / "cloud.ply", cloud, None if fusion is None else fusion["unique_viewpoints"],
              None if fusion is None else fusion["conflicting_semantics"])
    start = time.perf_counter()
    metrics, distances = evaluate(cloud, oracle, visible, ct, gate, distinct, fusion)
    np.savez_compressed(out / "distances.npz", **distances)
    n = len(cloud.xyz_mm)
    coverage = [metrics[name]["eligible_oracle_to_output"] for name in IDENTITIES]
    sem = [metrics[name]["semantic_correctness"] for name in IDENTITIES]
    per_class = np.concatenate([distances[f"{name}__output_to_eligible_oracle"] for name in IDENTITIES])
    ct_error = np.concatenate([distances[f"{name}__output_to_boundary_voxel_positions"] for name in IDENTITIES])
    report = {
        "method": method, "seed": seed, "perception": "P1" if predicted else "P0",
        "representation": "1.5-mm semantic voxels" if fusion is not None else "raw interface point cloud",
        "eligible_waypoints": 102, "point_count": n,
        "observation_pool": admissions, "multiple_semantics_same_ray": ray_conflicts,
        "semantic_correctness": {key: sum(s[key] for s in sem) for key in sem[0]},
        "observed_reference_localization": summarize_distances(per_class, gate),
        "observed_coverage": {"covered": sum(row["within_mm"] for row in coverage),
                              "total": sum(row["count"] for row in coverage)},
        "full_ct_reference_localization": summarize_distances(ct_error, gate),
        "per_identity": metrics, "target_status": "NOT_EVALUABLE",
        "reconstruction_runtime_seconds": elapsed,
        "evaluation_runtime_seconds": time.perf_counter() - start,
        "shared_input_preparation_seconds": prep_seconds,
        "peak_rss_mib_process_to_date": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "projection": projection,
        "cloud_npz": str(dest), "cloud_ply": str(out / "cloud.ply"),
        "evaluation_distances_npz": str(out / "distances.npz"),
        "limitations": "Provisional observable-reference labels and finite-resolution CT boundary voxels."
        " Fused semantic counts use one match per selected distinct source view for each voxel/class.",
    }
    write_json(out / "metrics.json", report)
    return report


def validate_inputs(path, source, phase):
    for relative, expected in source["files"].items():
        if relative.startswith("privileged/") and phase == "learned":
            continue
        if sha256(path / relative) != expected:
            raise ValueError(f"sealed acquisition input changed: {path}/{relative}")


def run(args):
    manifest = json.loads(args.manifest.read_text())
    manifest_hash = sha256(args.manifest)
    if manifest_hash != args.manifest_sha256:
        raise ValueError("shared manifest checksum mismatch")
    protocol_path = Path(manifest["protocol"]["path"])
    if sha256(protocol_path) != manifest["protocol"]["sha256"]:
        raise ValueError("replay protocol checksum mismatch")
    protocol = yaml.safe_load(protocol_path.read_text())
    calibration = manifest["projection"]
    gate = manifest["evaluation"]["spatial_distance_gate_mm"]
    voxel = calibration["simulator_label_spacing_mm"]
    if IDENTITIES != tuple(manifest["classes"]):
        raise ValueError("identity taxonomy and manifest differ")
    if set(manifest["reserved_seeds"]) & {s["seed"] for s in manifest["sessions"]}:
        raise ValueError("reserved seed in recording cohort")
    if args.phase == "oracle":
        evaluator = load_evaluator(protocol, CONVERSION)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "cloud_schema.json", {
        "identities_in_semantic_id_order": IDENTITIES, "rgb_in_same_order": COLORS,
        "coordinates": "session-specific robot base in millimetres",
        "voxel_mm": voxel,
        "raw": "Frame metadata and executed pose in frames.json. N x 2 pixel_row_col indexed into original image.",
        "fused": "raw_indices/raw_offsets index contributing points of the corresponding raw method. selected_frame_sequences/selected_frame_offsets index distinct retained views. frame_sequence is earliest contributing frame.",
        "visible_reference": "Evaluator-only first-visible image-supported TP/SAP pixels.",
    })
    reports = []
    for source in manifest["sessions"]:
        seed = source["seed"]
        print(f"{args.phase} {seed}", flush=True)
        path = Path(source["path"])
        directory = args.output / str(seed)
        directory.mkdir(exist_ok=True)
        start = time.perf_counter()
        validate_inputs(path, source, args.phase)
        oracle_path = directory / "B0/cloud.npz"
        visible_path, ct_path = directory / "evaluator_visible.npz", directory / "evaluator_ct.npz"
        if args.phase == "oracle":
            ctx = context(path, protocol, evaluator, "medium")
            if not ctx["projection"]["passes"] or len(ctx["valid"]) != source["eligible_waypoint_count"]:
                raise ValueError(f"projection or eligibility check failed for {seed}")
            public = {key: ctx[key] for key in ("frames", "images", "valid", "coords", "region", "fit")}
            ct = full_ct_geometry(
                path, ctx["frames"][ctx["valid"][0]], evaluator[0], calibration["simulator_label_spacing_mm"],
            )
            visible = all_visible_cloud(ctx)
            np.savez_compressed(visible_path, **visible.arrays(), code=visible.code)
            np.savez_compressed(ct_path, **ct)
            projection = ctx["projection"]
            write_json(directory / "frames.json", {
                "source": str(path), "valid_sequences": ctx["valid"], "projection": projection,
                "frames": [{key: ctx["frames"][i][key] for key in (
                    "frame_sequence", "waypoint_index", "phase", "acquisition_id", "frame_sha256",
                    "actor_base_to_ee_position_m", "actor_base_to_ee_rotation_columns"
                )} for i in ctx["valid"]],
            })
            prep_seconds = time.perf_counter() - start
            start = time.perf_counter()
            raw, pixels, admissions, conflicts = raw_cloud(public, oracle_labels=ctx["labels"])
            oracle = raw
        else:
            public = public_context(path, protocol, manifest)
            frames = json.loads((directory / "frames.json").read_text())
            projection = frames["projection"]
            if len(public["valid"]) != source["eligible_waypoint_count"] or frames["valid_sequences"] != public["valid"]:
                raise ValueError(f"oracle and learned frame selection differ for {seed}")
            probabilities, rules, receipt = load_cache(
                Path(manifest["shared_probability_cache"]["path"]), manifest_hash, manifest["models"],
                seed, public["valid"], public["images"].shape[1:],
            )
            write_json(directory / "probability_receipt.json", receipt)
            prep_seconds = time.perf_counter() - start
            start = time.perf_counter()
            raw, pixels, admissions, conflicts = raw_cloud(public, probabilities=probabilities, thresholds=rules)
            reconstruction = time.perf_counter() - start
            with np.load(oracle_path) as artifact:
                code = np.asarray([IDENTITIES.index(s) for s in artifact["identity"]])
                oracle = cloud_from_arrays(
                    artifact["xyz_mm"], code, artifact["frame_sequence"], artifact["confidence"]
                )
            with np.load(visible_path) as artifact:
                visible = cloud_from_arrays(
                    artifact["xyz_mm"], artifact["code"], artifact["frame_sequence"], artifact["confidence"]
                )
            with np.load(ct_path) as artifact:
                ct = dict(artifact)
        if args.phase == "oracle":
            reconstruction = time.perf_counter() - start
        distinct = distinct_pose_matrix(public["frames"], protocol["landmark_map"])
        raw_method, fused_method = ("B0", "B2") if args.phase == "oracle" else ("B1", "B3")
        reports.append(evaluate_variant(
            args.output, raw_method, seed, raw, pixels, admissions, conflicts, None, oracle, visible, ct,
            gate, distinct, reconstruction, prep_seconds, projection,
        ))
        fuse_start = time.perf_counter()
        fused, metadata = voxel_fuse(raw, distinct, voxel)
        fuse_elapsed = time.perf_counter() - fuse_start
        reports.append(evaluate_variant(
            args.output, fused_method, seed, fused, pixels, admissions, conflicts, metadata,
            oracle, visible, ct, gate, distinct, reconstruction + fuse_elapsed, prep_seconds, projection,
        ))
        write_json(args.output / f"{args.phase}_summary.json", {"runs": reports})
    if args.phase == "learned":
        old = json.loads((args.output / "oracle_summary.json").read_text())
        write_json(args.output / "summary.json", {
            "manifest_sha256": manifest_hash, "runs": old["runs"] + reports,
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", choices=("oracle", "learned"), required=True)
    run(parser.parse_args())
