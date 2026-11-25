"""Replay observed cortical samples with preserved public registration and provenance."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy import ndimage
import yaml

from replay import CONVERSION, infer, models_for, route
from run_simple_landmark_mapping import load_evaluator, load_public, oracle_observations, sha256, write_json
from two_stage_navigation.simple_landmark_mapping import (
    CLASSES, LEVELS, Stage1Region, extract_observations, sap_level_at, transverse_cranial_coordinate,
)
from two_stage_navigation.sonoanatomy_labels import (
    LEVEL_SOURCE_LABEL, first_bone_surface, image_cortical_support,
)
from workflows.reconstruction_s0010.dense import load_cache, most_proximal_pixels


IDENTITIES = ("RIGHT_L4_TP", "RIGHT_L4_SAP", "RIGHT_L5_TP", "RIGHT_L5_SAP", "OTHER_BONE", "UNCERTAIN_BONE")
COLORS = ((36, 138, 204), (210, 73, 98), (20, 149, 111), (240, 148, 39), (157, 159, 159), (168, 121, 191))
STRUCTURE = np.ones((3, 3), dtype=bool)
OLD_COMPONENT = "historical_eligible_component"
SUPPLEMENTAL_MODEL = "supplemental_accepted_foreground"
SUPPLEMENTAL_TRACK = "supplemental_track_without_accepted_foreground"
SUPPLEMENTAL_ORACLE = "supplemental_first_visible_oracle"


def file_seal(path, expected=None):
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"asset checksum mismatch: {path}")
    return {"path": str(path.resolve()), "sha256": actual, "size_bytes": path.stat().st_size}


def load_manifest(path):
    file_seal(path, path.with_suffix(".json.sha256").read_text().split()[0])
    return json.loads(path.read_text())


def load_recording(path, protocol, count, waypoint_sha=None):
    frames, images, valid, coords, dropped, _ = load_public(
        path, protocol["geometry"]["ultrasound_pixel_resolution_mm"],
    )
    actor_file = path / "actor_result.json"
    seal = actor_file.with_suffix(".json.sha256")
    if seal.is_file():
        file_seal(actor_file, seal.read_text().split()[0])
    actor = json.loads(actor_file.read_text())
    if waypoint_sha is not None:
        plan = json.loads((path / "waypoints.json").read_text())[:count]
        design = json.dumps(plan, indent=2) + "\n"
        if hashlib.sha256(design.encode()).hexdigest() != waypoint_sha:
            raise ValueError(f"matched waypoint plan changed: {path}")
        valid = [i for i in valid if frames[i]["waypoint_index"] < count]
    if len(valid) != count or len({frames[i]["waypoint_index"] for i in valid}) != count:
        raise ValueError(f"expected {count} distinct validated waypoint frames at {path}, found {len(valid)}")
    fit = transverse_cranial_coordinate(
        [frames[i] for i in valid], protocol["geometry"]["label_resolution_mm"],
        protocol["sonoanatomy_scan"]["translation_step_index"] / 2,
    )
    return {
        "actor": actor, "frames": frames, "images": images, "valid": valid,
        "coords": coords, "region": Stage1Region(**actor["stage1_region"]), "fit": fit,
        "dropped": dict(dropped),
    }


def level_hint(public, point):
    _, possible = sap_level_at(point, public["region"], public["fit"])
    return possible[0] if len(possible) == 1 else ("AMBIGUOUS" if possible else "OTHER")


def component_maps(public, seq, level, winner, probabilities, thresholds):
    frame, coords = public["frames"][seq], public["coords"][seq]
    rows, mask = extract_observations(winner, probabilities, thresholds, frame, coords)
    candidates = np.zeros(mask.shape, dtype=np.uint8)
    historic = np.zeros(mask.shape, dtype=np.uint8)
    groups = {value: ndimage.label(mask == value, structure=STRUCTURE)[0] for value in (1, 2)}
    for row in rows:
        key = CLASSES.index(row.semantic_class)
        memberships = groups[key]
        component = memberships == memberships[row.row, row.column]
        if int(component.sum()) != row.pixel_count:
            raise ValueError("pixel associations do not reproduce the historical component")
        n = route(row, frame, level, public, tp_all=True)
        if n == 1:
            candidates[component] = key
            pixels = np.argwhere(component)
            if probabilities is not None:
                pixels = most_proximal_pixels(pixels)
            historic[tuple(pixels.T)] = key
        else:
            candidates[component] = 4
    candidates[mask == 3] = 3
    return candidates, historic


def first_supported_samples(candidates, support):
    foreground = np.any(candidates > 0, axis=0) & support
    selected = []
    for column in np.flatnonzero(support.any(axis=0)):
        appears = np.flatnonzero(foreground[:, column])
        if len(appears):
            selected.append((int(appears[0]), int(column), True))
        else:
            rows = np.flatnonzero(support[:, column])
            cuts = np.flatnonzero(np.diff(rows) > 1)
            first_band = rows[:cuts[0] + 1] if len(cuts) else rows
            selected.append((int(first_band[(len(first_band) - 1) // 2]), int(column), False))
    return selected


def learned_frame(public, seq, prediction, rules, support):
    accepted, historic = [], []
    for j, level in enumerate(LEVELS):
        maps = component_maps(public, seq, level, prediction[j].argmax(0), prediction[j], rules[level])
        accepted.append(maps[0])
        historic.append(maps[1])
    candidate_labels = np.asarray(accepted)
    history = np.asarray(historic)
    xyz = public["coords"][seq]
    points = []
    for row, col, observed in first_supported_samples(candidate_labels, support):
        predicted = prediction[:, :, row, col]
        model_index = int(np.argmax(1 - predicted[:, 0]))
        identity, hint = "UNCERTAIN_BONE", level_hint(public, xyz[row, col])
        confidence = 1 - predicted[model_index, 0]
        source = SUPPLEMENTAL_MODEL if observed else SUPPLEMENTAL_TRACK
        labels = candidate_labels[:, row, col]
        named = np.flatnonzero(np.isin(labels, (1, 2)))
        if len(named) == 1 and 4 not in labels:
            model_index = int(named[0])
            code = int(labels[model_index])
            hint = LEVELS[model_index]
            identity = f"RIGHT_{hint}_{CLASSES[code]}"
            confidence = predicted[model_index, code]
            if history[model_index, row, col] == code:
                source = OLD_COMPONENT
        elif len(named) > 1 or 4 in labels:
            hint = "AMBIGUOUS"
        elif observed:
            positive = np.flatnonzero(labels == 3)
            if not len(positive):
                raise ValueError("observed foreground has no supported semantic source")
            model_index = int(positive[np.argmax(predicted[positive, 3])])
            identity = "OTHER_BONE"
            confidence = predicted[model_index, 3]
        points.append((xyz[row, col], identity, hint, seq, [row, col], confidence,
                       predicted[model_index], predicted, LEVELS[model_index], source))
    return points, int(np.count_nonzero(history))


def oracle_frame(public, seq, labels, raw, bone_ids, support):
    first = first_bone_surface(raw, bone_ids)
    requests = np.asarray([labels[level][seq] for level in LEVELS])
    historical = np.asarray([
        component_maps(public, seq, level, labels[level][seq], None, {})[1]
        for level in LEVELS
    ])
    coords = public["coords"][seq]
    points, raw_labels = [], []
    level_by_id = {ident: level for level, ident in LEVEL_SOURCE_LABEL.items()}
    for row, col in np.argwhere(first & support):
        label_values = requests[:, row, col]
        named = np.flatnonzero(np.isin(label_values, (1, 2)))
        code = None
        hint = level_by_id.get(int(raw[row, col]), "OTHER")
        source = SUPPLEMENTAL_ORACLE
        if len(named) == 1:
            j = int(named[0])
            code = int(label_values[j])
            hint = LEVELS[j]
            identity = f"RIGHT_{hint}_{CLASSES[code]}"
            if historical[j, row, col] == code:
                source = OLD_COMPONENT
        elif len(named) == 0 and np.all(label_values == 3):
            identity, code = "OTHER_BONE", 3
        else:
            identity = "UNCERTAIN_BONE"
        posterior = np.eye(4, dtype=np.float32)[code] if code is not None else np.full(4, .25, dtype=np.float32)
        points.append((coords[row, col], identity, hint, seq, [row, col], float(code is not None),
                       posterior, np.zeros((2, 4), dtype=np.float32), "ORACLE", source))
        raw_labels.append(int(raw[row, col]))
    return points, raw_labels, int(np.count_nonzero(historical))


def cloud_arrays(points):
    return {
        "xyz_mm": np.asarray([p[0] for p in points], dtype=np.float64).reshape(-1, 3),
        "identity": np.asarray([p[1] for p in points], dtype="U16"),
        "level_hint": np.asarray([p[2] for p in points], dtype="U9"),
        "frame_sequence": np.asarray([p[3] for p in points], dtype=np.int32),
        "pixel_row_col": np.asarray([p[4] for p in points], dtype=np.int16).reshape(-1, 2),
        "confidence": np.asarray([p[5] for p in points], dtype=np.float32),
        "semantic_probabilities": np.asarray([p[6] for p in points], dtype=np.float32).reshape(-1, 4),
        "semantic_probabilities_by_level": np.asarray([p[7] for p in points], dtype=np.float32).reshape(-1, 2, 4),
        "model_level": np.asarray([p[8] for p in points], dtype="U6"),
        "model_probability_available": np.asarray([p[8] != "ORACLE" for p in points], dtype=bool),
        "source": np.asarray([p[9] for p in points], dtype="U48"),
        "historical_eligible_component": np.asarray([p[9] == OLD_COMPONENT for p in points], dtype=bool),
    }


def write_ply(path, cloud):
    with path.open("w", encoding="ascii") as handle:
        handle.write(
            "ply\nformat ascii 1.0\ncomment coordinates session robot base in millimetres\n"
            f"element vertex {len(cloud['xyz_mm'])}\n"
            "property double x\nproperty double y\nproperty double z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "property uchar semantic_id\nproperty int frame_sequence\nproperty ushort pixel_row\n"
            "property ushort pixel_col\nproperty float confidence\nend_header\n"
        )
        for xyz, identity, seq, pixel, confidence in zip(
            cloud["xyz_mm"], cloud["identity"], cloud["frame_sequence"], cloud["pixel_row_col"], cloud["confidence"],
        ):
            code = IDENTITIES.index(identity)
            rgb = " ".join(map(str, COLORS[code]))
            handle.write(" ".join([*(f"{v:.7f}" for v in xyz), rgb, str(code), str(seq),
                                    str(pixel[0]), str(pixel[1]), f"{confidence:.7f}"]) + "\n")


def save_cloud(directory, label, points, historical_admissions, elapsed):
    cloud = cloud_arrays(points)
    keys = list(zip(cloud["frame_sequence"], *cloud["pixel_row_col"].T))
    if len(keys) != len(set((s, c) for s, _, c in keys)):
        raise ValueError("one physical sample per acquisition depth ray is required")
    if not np.isfinite(cloud["xyz_mm"]).all():
        raise ValueError("nonfinite calibrated observation")
    np.savez_compressed(directory / f"{label}.npz", **cloud)
    write_ply(directory / f"{label}.ply", cloud)
    groups = Counter((p[1], p[2]) for p in points)
    return {
        "points": len(points), "frames_with_points": len(set(cloud["frame_sequence"].tolist())),
        "identity_counts": dict(Counter(cloud["identity"].tolist())),
        "identity_and_level_counts": {f"{name}/{level}": n for (name, level), n in sorted(groups.items())},
        "point_sources": dict(Counter(cloud["source"].tolist())),
        "historical_component_point_pool": historical_admissions,
        "historical_points_retained": int(cloud["historical_eligible_component"].sum()),
        "elapsed_seconds": elapsed,
        "output": {k: file_seal(directory / f"{label}.{k}") for k in ("npz", "ply")},
    }


def replay_session(path, public, protocol, evaluator_factory, probabilities, rules, directory):
    directory.mkdir(parents=True, exist_ok=False)
    valid = public["valid"]
    start = time.perf_counter()
    support = {i: image_cortical_support(public["images"][i]) for i in valid}
    learned_points, learned_total = [], 0
    for i, seq in enumerate(valid):
        predicted = np.asarray([probabilities[level][i] for level in LEVELS])
        records, historical = learned_frame(public, seq, predicted, rules, support[seq])
        learned_points.extend(records)
        learned_total += historical
    learned = save_cloud(directory, "learned", learned_points, learned_total, time.perf_counter() - start)
    # Serialize the learned output before loading anatomical labels and coordinates for scoring.
    start = time.perf_counter()
    evaluator = evaluator_factory()
    rows, labels, visible, reference, projection, _ = oracle_observations(
        path, public["frames"], public["images"], valid, public["coords"], evaluator, protocol,
    )
    del rows, visible, reference
    if not projection["passes"]:
        raise ValueError(f"calibration parity failed for {path}")
    with np.load(path / "privileged/raw_labels.npz") as file:
        raw = file["raw_labels"]
    oracle_points, oracle_labels, oracle_total = [], [], 0
    for seq in valid:
        records, labels_raw, historical = oracle_frame(
            public, seq, labels, raw[seq], evaluator[1], support[seq],
        )
        oracle_points.extend(records)
        oracle_labels.extend(labels_raw)
        oracle_total += historical
    oracle = save_cloud(directory, "oracle", oracle_points, oracle_total, time.perf_counter() - start)
    oracle_raw = directory / "evaluator_only_oracle_source_labels.npz"
    np.savez_compressed(oracle_raw, raw_source_label=np.asarray(oracle_labels))
    oracle["evaluator_only_raw_labels"] = file_seal(oracle_raw)
    return learned, oracle, projection


def reconstruct(args):
    manifest = load_manifest(args.manifest)
    cache_manifest = load_manifest(args.cache_manifest)
    if manifest["anatomy"] != "s0010" or cache_manifest["models"] != manifest["models"]:
        raise ValueError("model or anatomy manifest mismatch")
    file_seal(args.cache_manifest, manifest["historical_sweep"]["sha256"])
    protocol_path = Path(manifest["protocol"]["path"])
    file_seal(protocol_path, manifest["protocol"]["sha256"])
    protocol = yaml.safe_load(protocol_path.read_text())
    for model in manifest["models"].values():
        file_seal(Path(model["path"]), model["sha256"])
    for key in ("prediction", "segmentation"):
        entry = manifest["target"][key]
        file_seal(Path(entry["path"]), entry["sha256"])
    file_seal(CONVERSION, manifest["target"]["label_conversion"]["sha256"])
    reference = json.loads((args.reference / "reference.json").read_text())
    if reference["used_ultrasound_to_choose_reference"] is not False:
        raise ValueError("external reference provenance must be chosen independently of observations")
    for row in (args.reference / "checksums.sha256").read_text().splitlines():
        expected, file = row.split(maxsplit=1)
        file_seal(args.reference / file, expected)
    args.output.mkdir(parents=True, exist_ok=False)

    evaluator = None
    def get_evaluator():
        nonlocal evaluator
        if evaluator is None:
            evaluator = load_evaluator(protocol, CONVERSION)
        return evaluator

    if args.session is not None:
        sessions = [{"path": str(args.session), "seed": args.seed, "eligible_waypoint_count": args.expected_frames}]
    else:
        sessions = manifest["sessions"]
        if sum(s["eligible_waypoint_count"] for s in sessions) != manifest["acquisition_count"]:
            raise ValueError("frozen replay acquisition count differs from its session entries")
    reserved = set(manifest["reserved_seeds"]) | set(cache_manifest["reserved_seeds"])
    if {r["seed"] for r in sessions} & set(reserved):
        raise ValueError("reserved seed requested")
    output = []
    for source in sessions:
        path = Path(source["path"])
        seed = source["seed"]
        print(f"Reconstructing {seed}", flush=True)
        for relative, expected in source.get("files", {}).items():
            file_seal(path / relative, expected)
        waypoint_hash = cache_manifest["common_waypoints_sha256"] if args.session is None else None
        public = load_recording(path, protocol, source["eligible_waypoint_count"], waypoint_hash)
        if public["actor"]["seed"] != seed:
            raise ValueError(f"session identifier differs from the sealed actor: {path}")
        if args.session is None:
            probabilities, rules, receipt = load_cache(
                Path(cache_manifest["shared_probability_cache"]["path"]), sha256(args.cache_manifest),
                manifest["models"], seed, public["valid"], public["images"].shape[1:],
            )
        else:
            models = models_for(manifest["models"])
            probabilities = {level: np.asarray(list(infer(model, public["images"][public["valid"]], batch_size=1)))
                             for level, (model, _) in models.items()}
            rules = {level: values[1] for level, values in models.items()}
            receipt = {"inference": "uncached recorded input", "inference_batch_size": 1,
                       "models": manifest["models"]}
        directory = args.output / str(seed)
        learned, oracle, geometry = replay_session(path, public, protocol, get_evaluator,
                                                    probabilities, rules, directory)
        result = {
            "seed": seed, "base_coordinates": "session actor robot base millimetres", "valid_sequences": public["valid"],
            "actor_pose_source": file_seal(path / "public/frames.jsonl"), "ultrasound": file_seal(path / "public/ultrasound.npz"),
            "calibration_parity": geometry, "probability_order": CLASSES, "model_levels": LEVELS,
            "model_receipt": receipt, "thresholds": rules, "oracle": oracle, "learned": learned,
            "oracle_samples": "First acoustic-bone hit within existing three-pixel bright-interface support. One-hot diagnostic TP/SAP or other, uniform placeholder with zero confidence for ambiguous label. All suspected anatomy beyond first supported cortex is excluded.",
            "learned_samples": "Earliest U-Net threshold-admitted foreground within existing three-pixel clean-ultrasound support per ray. If no accepted foreground is supported, retain the middle of the first support run as semantic uncertainty.",
            "learned_routing": "Public replay.route for TP and SAP; actor-side fitted cranial coordinate for other bone. Explicit semantic conflict from multiple routed targets or discarded target-level component remains uncertain.",
            "historical_anchor_provenance": "Exact per-component old eligible pixels (Oracle full label surface; learned proximal pixel per ray) when retained by the complete replay.",
            "max_rss_mib_process_to_date": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
        receipt_path = directory / "receipt.json"
        write_json(receipt_path, result)
        (directory / "receipt.json.sha256").write_text(f"{sha256(receipt_path)}  receipt.json\n")
        output.append(result)
    write_json(args.output / "summary.json", {
        "manifest": file_seal(args.manifest), "cache_manifest": file_seal(args.cache_manifest),
        "reference": file_seal(args.reference / "reference.json"), "reference_status": reference["status"],
        "acquisitions": sum(len(r["valid_sequences"]) for r in output), "sessions": output,
        "identity_ply_order": IDENTITIES,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "cache-manifest", "reference", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--session", type=Path, help="optional separately authorized recorded session")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--expected-frames", type=int)
    args = parser.parse_args()
    if args.session is not None and (args.seed is None or args.expected_frames is None or args.expected_frames <= 0):
        parser.error("--session requires --seed and positive --expected-frames")
    reconstruct(args)


if __name__ == "__main__":
    main()
