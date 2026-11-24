#!/usr/bin/env python3
"""Compare simple mapping and existing association on recorded SonoGym frames."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
import torch
import yaml

from two_stage_navigation.acquisition_integrity import validate_actor_6dof_record
from two_stage_navigation.contracts import ultrasound_sha256
from two_stage_navigation.incremental_landmark_map import (
    LANDMARK_IDENTITIES, IncrementalAnatomicalLandmarkMap, IncrementalLandmarkMapConfig,
)
from two_stage_navigation.simple_landmark_mapping import (
    CLASSES, LEVELS, extract_observations, project_image, simple_map, stage1_region,
)
from two_stage_navigation.sonoanatomy_labels import IGNORE, right_level_masks, visible_interface_labels
from two_stage_navigation.sonoanatomy_multiview import cluster_observations
from two_stage_navigation.sonoanatomy_segmentation import (
    load_segmentation_checkpoint, segmentation_probabilities,
)


OFFLINE_TITLE = "OFFLINE RECORDED-DATA MAPPING EXPERIMENT"
ORACLE_TITLE = "ORACLE_PERCEPTION_DIAGNOSTIC — NOT AUTONOMOUS"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_seal(path, seal):
    digest = sha256(path)
    if digest != seal.read_text().split()[0]:
        raise ValueError(f"checksum mismatch: {path}")
    return digest


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_models(protocol):
    models, manifest = {}, {}
    for level in LEVELS:
        specification = protocol["models"]["sonoanatomy"][level]
        path = Path(specification["path"])
        digest = sha256(path)
        if digest != specification["sha256"]:
            raise ValueError(f"frozen U-Net checksum mismatch: {level}")
        model, checkpoint = load_segmentation_checkpoint(path, expected_level=level)
        thresholds = checkpoint["thresholds"]
        models[level] = model, thresholds
        manifest[level] = {"path": str(path), "sha256": digest, "thresholds": thresholds}
    return models, manifest


def load_public(session, pixel_mm):
    files = [session / "public/frames.jsonl", session / "public/ultrasound.npz"]
    frames = read_jsonl(files[0])
    with np.load(files[1]) as archive:
        images = archive["ultrasound"]
    if images.shape != (len(frames), 200, 150):
        raise ValueError(f"unexpected image dimensions: {session}")
    valid, coordinates, seen_acquisitions = [], {}, set()
    previous_producer = -1
    dropped = Counter()
    for sequence, frame in enumerate(frames):
        if frame["frame_sequence"] != sequence or frame["session_id"] != session.name:
            raise ValueError(f"frame sequence/session mismatch: {session}:{sequence}")
        if ultrasound_sha256(images[sequence]) != frame["frame_sha256"]:
            raise ValueError(f"ultrasound content hash mismatch: {session}:{sequence}")
        producer = frame["producer_sequence"]
        increasing = producer > previous_producer
        previous_producer = max(producer, previous_producer)
        failed = [key for key in ("image_valid", "contact_ok", "producer_advanced", "transition_valid")
                  if frame[key] is not True]
        if failed:
            dropped["+".join(failed)] += 1
            continue
        if not increasing or frame["acquisition_id"] in seen_acquisitions:
            raise ValueError(f"invalid producer order or reused acquisition: {session}:{sequence}")
        seen_acquisitions.add(frame["acquisition_id"])
        validate_actor_6dof_record(frame)
        valid.append(sequence)
        coordinates[sequence] = project_image(frame, images.shape[1:], pixel_mm)
    return frames, images, valid, coordinates, dict(dropped), files


def unet_observations(frames, images, valid, coordinates, region, models, batch_size):
    observations, ambiguities = defaultdict(list), defaultdict(list)
    levels_by_frame = {
        sequence: region.levels_at(frames[sequence]["executed_relative_pose_index"][1])
        for sequence in valid
    }
    masks = {}
    for level in LEVELS:
        model, thresholds = models[level]
        masks[level] = np.full(images.shape, 255, dtype=np.uint8)
        indices = [index for index in valid if level in levels_by_frame[index]]
        for start in range(0, len(indices), batch_size):
            sequences = indices[start:start + batch_size]
            probabilities = segmentation_probabilities(model, images[sequences])
            for sequence, probability in zip(sequences, probabilities):
                rows, masks[level][sequence] = extract_observations(
                    probability.argmax(axis=0), probability, thresholds,
                    frames[sequence], coordinates[sequence],
                )
                destination = observations if len(levels_by_frame[sequence]) == 1 else ambiguities
                for row in rows:
                    destination[f"RIGHT_{level}_{row.semantic_class}"].append(row)
    return observations, ambiguities, masks, levels_by_frame


def load_evaluator(protocol, label_conversion):
    settings = protocol["evaluator"]
    prediction_path = Path(settings["subregion_prediction_path"])
    segmentation_path = Path(settings["source_segmentation_path"])
    prediction = nib.load(prediction_path)
    segmentation = nib.load(segmentation_path)
    if prediction.shape != segmentation.shape or not np.allclose(prediction.affine, segmentation.affine):
        raise ValueError("evaluator subregions and source segmentation use different grids")
    predicted = np.asarray(prediction.dataobj, dtype=np.uint8)
    source = np.asarray(segmentation.dataobj, dtype=np.uint8)
    masks = {level: right_level_masks(predicted, source, prediction.affine, level) for level in LEVELS}
    bone_ids = {int(source_id) for source_id, acoustic in
                yaml.safe_load(label_conversion.read_text()).items() if int(acoustic) == 13}
    files = (prediction_path, segmentation_path, label_conversion)
    return masks, bone_ids, segmentation.shape, {str(path): sha256(path) for path in files}


def oracle_observations(session, frames, images, valid, coordinates, evaluator, protocol):
    volume_masks, bone_ids, volume_shape, _ = evaluator
    files = [session / "privileged" / filename for filename in (
        "raw_labels.npz", "pixel_to_volume_coordinates.npz", "full_pose_replay.jsonl",
    )]
    with np.load(files[0]) as archive:
        raw = archive["raw_labels"]
    with np.load(files[1]) as archive:
        native = archive["human_img_coords"]
    poses = read_jsonl(files[2])
    if raw.shape != images.shape or native.shape != (
        len(images), images.shape[1] * images.shape[2], 3
    ) or len(poses) != len(frames):
        raise ValueError("evaluator and actor frame counts or dimensions differ")
    for frame, pose in zip(frames, poses):
        if any(frame[key] != pose[key] for key in (
            "frame_sequence", "frame_sha256", "producer_sequence", "acquisition_id", "session_id",
        )):
            raise ValueError("evaluator and actor acquisition keys differ")

    observations = defaultdict(list)
    visibility = {}
    labels = {level: np.full(images.shape, IGNORE, dtype=np.uint8) for level in LEVELS}
    reference = defaultdict(list)
    residuals, rotation_to_base, origin_in_base = [], None, None
    for sequence in valid:
        frame, pose = frames[sequence], poses[sequence]
        resolution_mm = pose["label_resolution_m"] * 1000
        if not (np.isclose(resolution_mm, protocol["geometry"]["label_resolution_mm"])
                and np.isclose(pose["image_resolution_m"] * 1000,
                               protocol["geometry"]["ultrasound_pixel_resolution_mm"])
                and np.isclose(pose["image_origin_height_m"], 0.133)):
            raise ValueError("recorded calibration does not match public calibration")
        if rotation_to_base is None:
            rotation_to_base = (np.asarray(frame["actor_base_to_ee_rotation_columns"])
                                @ np.asarray(pose["human_to_ee_rotation_columns"]).T)
            origin_in_base = 1000 * (
                np.asarray(frame["actor_base_to_ee_position_m"])
                - rotation_to_base @ pose["human_to_ee_position_m"]
            )
        grid = native[sequence].reshape(images.shape[2], images.shape[1], 3).transpose(1, 0, 2)
        audit_pixels = np.zeros(grid.shape[:2], dtype=bool)
        audit_pixels[::25, ::25] = True
        for level in LEVELS:
            visible = visible_interface_labels(
                image=images[sequence], raw_labels=raw[sequence],
                coordinate_map_index=native[sequence], masks=volume_masks[level],
                bone_source_ids=bone_ids,
            )
            labels[level][sequence] = visible.labels
            rows, _ = extract_observations(
                visible.labels, None, {}, frame, coordinates[sequence],
            )
            for row in rows:
                observations[f"RIGHT_{level}_{row.semantic_class}"].append(row)
            for class_id, name in enumerate(CLASSES[1:3], start=1):
                target = visible.labels == class_id
                audit_pixels |= target
                if np.any(target):
                    identity = f"RIGHT_{level}_{name}"
                    visibility[(identity, sequence)] = cKDTree(np.argwhere(target))
                    reference[identity].append(coordinates[sequence][target])
        audit_pixels &= np.all((grid > 0) & (grid < np.asarray(volume_shape) - 1), axis=-1)
        native_base = (grid[audit_pixels].astype(np.float64) * resolution_mm
                       @ rotation_to_base.T + origin_in_base)
        residuals.append(np.linalg.norm(coordinates[sequence][audit_pixels] - native_base, axis=1))
    errors = np.concatenate(residuals)
    if not len(errors):
        raise ValueError("no unclipped pixels for the projection comparison")
    gate = (protocol["geometry"]["label_resolution_mm"]
            * protocol["evaluator"]["maximum_coordinate_error_index"])
    audit = {
        "unclipped_sample_count": int(len(errors)), "maximum_error_mm": float(errors.max()),
        "median_error_mm": float(np.median(errors)), "p95_error_mm": float(np.quantile(errors, 0.95)),
        "gate_mm": gate, "passes": bool(errors.max() <= gate),
    }
    reference = {identity: cKDTree(np.concatenate(cloud)) for identity, cloud in reference.items()}
    return observations, labels, visibility, reference, audit, files


def existing_map(observations, ambiguities, config):
    mapping = IncrementalAnatomicalLandmarkMap(config)
    for level in LEVELS:
        rows = [row for name in ("TP", "SAP")
                for row in observations.get(f"RIGHT_{level}_{name}", ())]
        mapping.add_observations(level, sorted(rows, key=lambda row: row.frame_sequence))
    landmarks = {}
    for identity in LANDMARK_IDENTITIES:
        recorded = mapping.landmark(identity)
        status = {"CONFIRMED": "MAPPED", "REJECTED": "CANDIDATE"}.get(recorded.status, recorded.status)
        if status == "NOT_OBSERVED" and ambiguities.get(identity):
            status = "AMBIGUOUS"
        selected = None
        competitors = []
        if recorded.position_mm is not None:
            selected = {
                "position_mm": recorded.position_mm,
                "semantic_confidence": recorded.semantic_confidence,
                "rms_dispersion_mm": recorded.rms_uncertainty_mm,
                "distinct_viewpoints": len(recorded.supporting_viewpoint_ids),
                "supporting_observations": [asdict(row) for row in recorded.supporting_observations],
            }
            rows = observations[identity]
            clusters = cluster_observations(
                rows, association_radius_mm=config.association_radius_mm,
                minimum_view_translation_mm=config.minimum_view_translation_mm,
                minimum_view_rotation_deg=config.minimum_view_rotation_deg,
            )
            for cluster in clusters:
                support = [asdict(rows[index]) for index in cluster.observation_indices]
                if {(row["frame_sequence"], row["row"], row["column"]) for row in support} == {
                    (row["frame_sequence"], row["row"], row["column"])
                    for row in selected["supporting_observations"]
                }:
                    continue
                competitors.append({
                    "position_mm": cluster.centroid_mm, "semantic_confidence": cluster.confidence,
                    "distinct_viewpoints": len(cluster.supporting_frames),
                    "supporting_observations": support,
                })
        landmarks[identity] = {
            "status": status, "selected": selected, "existing_status": recorded.status,
            "existing_reason": recorded.reason,
            "existing_competing_hypothesis_count": recorded.competing_hypothesis_count,
            "competing_clusters": competitors,
            "ambiguous_level_observation_count": len(ambiguities.get(identity, ())),
        }
    return landmarks


def score_map(mapping, observations, visibility, reference, maximum_distance_pixels):
    result = {}
    for identity, landmark in mapping.items():
        def distance(row):
            surface = visibility.get((identity, row["frame_sequence"]))
            return None if surface is None else float(surface.query((row["row"], row["column"]))[0])

        def correct(value):
            return value is not None and value <= maximum_distance_pixels

        distances = [distance(asdict(row)) for row in observations.get(identity, ())]
        selection = landmark["selected"]
        support_distances = [] if selection is None else [
            distance(row) for row in selection["supporting_observations"]
        ]
        count_correct = sum(correct(value) for value in distances)
        support_correct = sum(correct(value) for value in support_distances)
        outcome = "UNRESOLVED"
        if landmark["status"] == "MAPPED":
            outcome = "CORRECT" if 2 * support_correct > len(support_distances) else "WRONG"
        target_surface = reference.get(identity)
        result[identity] = {
            "outcome": outcome,
            "observation_count": len(distances),
            "correct_observations": count_correct,
            "incorrect_observations": len(distances) - count_correct,
            "correct_observation_fraction": count_correct / len(distances) if distances else None,
            "supporting_viewpoint_count": 0 if selection is None else selection["distinct_viewpoints"],
            "support_correct": support_correct,
            "support_distances_pixels": support_distances,
            "competing_cluster_count": len(landmark["competing_clusters"]),
            "position_mm": None if selection is None else selection["position_mm"],
            "nearest_visible_surface_error_mm": None if selection is None or target_surface is None
            else float(target_surface.query(selection["position_mm"])[0]),
            "status": landmark["status"],
        }
    counts = Counter(row["outcome"] for row in result.values())
    return {
        "landmarks": result,
        "correct_pairs": {level: all(result[f"RIGHT_{level}_{kind}"]["outcome"] == "CORRECT"
                                    for kind in ("TP", "SAP")) for level in LEVELS},
        "correct_mapped_landmarks": counts["CORRECT"],
        "wrong_mapped_landmarks": counts["WRONG"],
        "unresolved_landmarks": counts["UNRESOLVED"],
        "complete_map": counts["CORRECT"] == len(LANDMARK_IDENTITIES),
        "observation_count": sum(row["observation_count"] for row in result.values()),
    }


def diagnose_visibility(visibility, labels, unet, ambiguities, masks, route, distance_pixels):
    diagnostic = {}
    for identity in LANDMARK_IDENTITIES:
        _, level, name = identity.split("_")
        class_id = CLASSES.index(name)
        counts = Counter()
        correct_frames = set()
        for row in (*unet.get(identity, ()), *ambiguities.get(identity, ())):
            surface = visibility.get((identity, row.frame_sequence))
            if surface is not None and surface.query((row.row, row.column))[0] <= distance_pixels:
                correct_frames.add(row.frame_sequence)
        for (visible_identity, sequence), _ in visibility.items():
            if identity != visible_identity:
                continue
            counts["oracle_visible_frames"] += 1
            if level not in route[sequence]:
                counts["true_level_checkpoint_not_applied"] += 1
            elif len(route[sequence]) > 1:
                counts["ambiguous_stage1_region"] += 1
                counts["correct_candidate_in_ambiguous_region"] += sequence in correct_frames
            elif sequence in correct_frames:
                counts["usable_correct_prediction"] += 1
            else:
                target = labels[level][sequence] == class_id
                overlaps = masks[level][sequence][target]
                common = int(np.bincount(overlaps, minlength=4).argmax())
                counts[f"miss_dominant_mask_class_{CLASSES[common]}"] += 1
        diagnostic[identity] = dict(counts)
    return diagnostic


def serialize_observations(rows):
    return {identity: [asdict(row) for row in group] for identity, group in rows.items()}


def run_session(session, stage1, protocol, models, evaluator, batch_size, output):
    started = time.perf_counter()
    actor_path = session / "results/actor_result.json"
    source_actor_hash = check_seal(actor_path, actor_path.with_suffix(".json.sha256"))
    original_actor = json.loads(actor_path.read_text())
    frames, images, valid, coordinates, dropped, public_files = load_public(
        session, protocol["geometry"]["ultrasound_pixel_resolution_mm"],
    )
    if original_actor["session_id"] != session.name or original_actor["frame_count"] != len(frames):
        raise ValueError("recorded actor result and public acquisition manifest disagree")
    if {frames[index]["stage1_session_id"] for index in valid} != {original_actor["stage1_session_id"]}:
        raise ValueError("Stage 1 references disagree with the recorded actor result")
    region = stage1_region(stage1, original_actor["stage1_session_id"], protocol)
    config = IncrementalLandmarkMapConfig(**protocol["landmark_map"])

    start = time.perf_counter()
    unet, ambiguous, masks, route = unet_observations(
        frames, images, valid, coordinates, region, models, batch_size,
    )
    unet_seconds = time.perf_counter() - start
    start = time.perf_counter()
    map_b = simple_map(unet, ambiguous, config)
    map_b_seconds = time.perf_counter() - start
    start = time.perf_counter()
    map_c = existing_map(unet, ambiguous, config)
    map_c_seconds = time.perf_counter() - start

    actor_output = output / "actor"
    actor_output.mkdir(parents=True)
    np.savez_compressed(actor_output / "unet_masks.npz", **masks)
    write_json(actor_output / "observations.json", {
        "stage1_region": asdict(region), "level_hypotheses_by_frame": route,
        "observations": serialize_observations(unet),
        "ambiguous_observations": serialize_observations(ambiguous),
    })
    write_json(actor_output / "simple_map.json", map_b)
    write_json(actor_output / "existing_map.json", map_c)

    start = time.perf_counter()
    oracle, labels, visibility, reference, projection, privileged_files = oracle_observations(
        session, frames, images, valid, coordinates, evaluator, protocol,
    )
    oracle_seconds = time.perf_counter() - start
    start = time.perf_counter()
    map_a = simple_map(oracle, {}, config)
    map_a_seconds = time.perf_counter() - start
    oracle_output = output / "oracle"
    oracle_output.mkdir()
    np.savez_compressed(oracle_output / "visible_interface_labels.npz", **labels)
    write_json(oracle_output / "observations.json", {
        "label": ORACLE_TITLE, "observations": serialize_observations(oracle),
    })
    write_json(oracle_output / "simple_map.json", {"label": ORACLE_TITLE, "landmarks": map_a})

    gate = protocol["evaluator"]["maximum_semantic_interface_distance_pixels"]
    arms = {
        "A": score_map(map_a, oracle, visibility, reference, gate),
        "B": score_map(map_b, unet, visibility, reference, gate),
        "C": score_map(map_c, unet, visibility, reference, gate),
    }
    for arm, perception, mapping in (
        ("A", oracle_seconds, map_a_seconds), ("B", unet_seconds, map_b_seconds),
        ("C", unet_seconds, map_c_seconds),
    ):
        arms[arm]["perception_seconds"] = perception
        arms[arm]["mapping_seconds"] = mapping
        arms[arm]["perception_and_mapping_seconds"] = perception + mapping
    arms["A"]["label"] = ORACLE_TITLE
    return {
        "session": session.name, "source": str(session), "seed": original_actor["seed"],
        "source_stage1_session": original_actor["stage1_session_id"], "attempted_frames": len(frames),
        "valid_frames": valid, "dropped_frames_by_flags": dropped,
        "ambiguous_level_frames": sum(len(levels) > 1 for levels in route.values()),
        "projection_check": projection, "arms": arms,
        "visible_interface_diagnostics": diagnose_visibility(visibility, labels, unet, ambiguous,
                                                            masks, route, gate),
        "input_sha256": {str(actor_path): source_actor_hash,
                         **{str(path): sha256(path) for path in (*public_files, *privileged_files)}},
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize(sessions):
    return {
        arm: {
            "per_landmark": {
                identity: dict(Counter(
                    session["arms"][arm]["landmarks"][identity]["outcome"] for session in sessions
                )) for identity in LANDMARK_IDENTITIES
            },
            "correct_pairs": {
                level: sum(session["arms"][arm]["correct_pairs"][level] for session in sessions)
                for level in LEVELS
            },
            **{metric: sum(session["arms"][arm][metric] for session in sessions) for metric in (
                "correct_mapped_landmarks", "wrong_mapped_landmarks", "unresolved_landmarks",
                "complete_map", "observation_count", "mapping_seconds", "perception_seconds",
                "perception_and_mapping_seconds",
            )},
        }
        for arm in ("A", "B", "C")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--label-conversion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    torch.set_num_interop_threads(1)
    protocol_hash = check_seal(args.protocol, args.protocol.with_suffix(".sha256"))
    protocol = yaml.safe_load(args.protocol.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    stage1_path = args.run_root / "stage1/results/actor_results.json"
    stage1_hash = check_seal(stage1_path, stage1_path.with_suffix(".sha256"))
    stage1 = json.loads(stage1_path.read_text())
    models, model_manifest = load_models(protocol)
    evaluator = load_evaluator(protocol, args.label_conversion)
    sessions = []
    for source in protocol["execution"]["sessions"]:
        session = args.run_root / "stage2" / source["session_id"]
        print(f"Processing {session.name}", flush=True)
        result = run_session(session, stage1, protocol, models, evaluator, args.batch_size,
                             args.output / session.name)
        if result["seed"] != source["seed"] or result["source_stage1_session"] != source["stage1_session_id"]:
            raise ValueError("recorded session differs from the sealed protocol")
        sessions.append(result)
        write_json(args.output / session.name / "comparison.json", result)
    result = {
        "label": OFFLINE_TITLE, "anatomy": protocol["scope"]["patient_id"],
        "source_protocol": str(args.protocol), "source_protocol_sha256": protocol_hash,
        "source_commit": (args.run_root / "context/git-commit.txt").read_text().strip(),
        "source_stage1_sha256": stage1_hash, "frozen_models": model_manifest,
        "evaluator_inputs_sha256": evaluator[-1], "configuration": protocol["landmark_map"],
        "versions": {"numpy": np.__version__, "torch": torch.__version__, "nibabel": nib.__version__},
        "sessions": sessions, "comparison": summarize(sessions), "batch_size": args.batch_size,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output / "comparison.json", result)
    print(json.dumps(result["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
