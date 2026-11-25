"""Evaluate controlled density, stabilization, and mapping comparisons."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import generate_binary_structure, label
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import yaml

from run_simple_landmark_mapping import (
    load_evaluator, load_models, load_public, oracle_observations, read_jsonl, score_map, sha256, write_json,
)
from two_stage_navigation.incremental_landmark_map import (
    LANDMARK_IDENTITIES, IncrementalLandmarkMapConfig,
)
from two_stage_navigation.simple_landmark_mapping import (
    CLASSES, Stage1Region, extract_observations, sap_level_at, simple_map, transverse_cranial_coordinate,
)
from two_stage_navigation.sonoanatomy_multiview import SemanticObservation
from two_stage_navigation.sonoanatomy_segmentation import (
    load_segmentation_checkpoint, segmentation_probabilities,
)
from two_stage_navigation.surface_landmark_mapping import (
    observation_surfaces, surface_distance, surface_map,
)


def read_json(path):
    return json.loads(path.read_text())


def observations(data):
    return {k: [SemanticObservation(**r) for r in v] for k, v in data.items()}


def route_sap(proposals, accepted_sequences, frames, region, protocol):
    calibration = transverse_cranial_coordinate(
        [f for f in frames if f["frame_sequence"] in accepted_sequences],
        protocol["geometry"]["label_resolution_mm"], protocol["sonoanatomy_scan"]["translation_step_index"] / 2,
    )
    eligible, ambiguous = defaultdict(list), defaultdict(list)
    for level, rows in proposals.items():
        for row in rows:
            if row.frame_sequence not in accepted_sequences:
                continue
            _, candidates = sap_level_at(row.coordinate_mm, region, calibration)
            if candidates == (level,):
                eligible[f"RIGHT_{level}_SAP"].append(row)
            elif len(candidates) > 1:
                ambiguous[f"RIGHT_{level}_SAP"].append(row)
    return eligible, ambiguous, calibration


def filter_rows(data, eligible, *, kind=None):
    return {k: [r for r in v if r.frame_sequence in eligible]
            for k, v in data.items() if kind is None or k.endswith("_" + kind)}


def leave_one_view_out(mapping, obs, amb, config, assess):
    result = {}
    for identity, landmark in mapping.items():
        selected = landmark["selected"]
        if not selected:
            continue
        result[identity] = {}
        for support in selected["supporting_observations"]:
            sequence = support["frame_sequence"]
            altered, uncertain = ({k: [r for r in v if r.frame_sequence != sequence]
                                   for k, v in source.items()} for source in (obs, amb))
            outcome = assess(simple_map(altered, uncertain, config), altered)["landmarks"][identity]["outcome"]
            result[identity][str(sequence)] = outcome
    return result


def statistics(values):
    return ({stat: float(value) for stat, value in zip(
        ("mean", "median", "p95", "max"), (np.mean(values), np.median(values),
                                          np.quantile(values, .95), np.max(values)),
    )} if len(values) else None)


def tracking(frames):
    keys = ("full_nominal_position_error_mm", "full_nominal_rotation_error_deg")
    if not frames or keys[0] not in frames[0]:
        return None
    drifts = [drift for f in frames for drift in f.get("stabilization_deltas_mm_deg", ())]
    return {
        **{key: statistics([f[key] for f in frames]) for key in keys},
        "controller_xz_error_mm": statistics([f["reduced_tracking_error"][0] for f in frames]),
        "controller_angle_error_deg": statistics([f["reduced_tracking_error"][1] for f in frames]),
        "hold_translation_mm": statistics([d[0] for d in drifts]),
        "hold_rotation_deg": statistics([d[1] for d in drifts]),
        "hold_attempts": len(drifts),
        "action_steps_at_last_capture": frames[-1]["action_steps_at_capture"],
        "planned_waypoint_ids_with_full_error_above_1mm_or_half_degree": [
            f["waypoint_index"] for f in frames
            if f[keys[0]] > 1.0 or f[keys[1]] > .5
        ],
    }


def oracle_competitors(point, labels, coords, native, volume_components, quantile):
    identity = "RIGHT_L4_SAP"
    entry = point[identity]
    if not (entry["selected"] and entry["competing_clusters"]):
        return None
    a, b = entry["selected"], entry["competing_clusters"][0]
    if min(a["distinct_viewpoints"], b["distinct_viewpoints"]) < 2:
        return None
    groups = [observations({identity: c["supporting_observations"]})[identity] for c in (a, b)]
    surfaces = [observation_surfaces(identity, rows, labels, coords) for rows in groups]
    xyz = [np.concatenate(s) for s in surfaces]
    dist = cdist(*xyz)

    def voxel_components(rows):
        result = set()
        for r in rows:
            grid = native[r.frame_sequence]
            depth = labels["L4"].shape[1]
            index = np.trunc(grid[r.column * depth + r.row]).astype(int)
            result.add(int(volume_components[tuple(index)]))
        return sorted(result - {0})

    return {
        "support_frames": [[r.frame_sequence for r in rows] for rows in groups],
        "volume_subregion_components_under_support": [voxel_components(rows) for rows in groups],
        "min_intercluster_surface_distance_mm": float(dist.min()),
        "between_clusters_max_pair_surface_distance_mm": max(
            surface_distance(first, second, quantile) for first in surfaces[0] for second in surfaces[1]
        ),
        "medoid_distance_mm": float(np.linalg.norm(np.asarray(a["position_mm"]) - b["position_mm"])),
    }


def mismatched_merges(mapping, visible, gate):
    merged = []
    for identity, landmark in mapping.items():
        if landmark["status"] != "MAPPED":
            continue
        cluster = landmark["selected"]
        wrong_support = []
        for row in cluster["supporting_observations"]:
            target = visible.get((identity, row["frame_sequence"]))
            if target is None or target.query([row["row"], row["column"]])[0] > gate:
                wrong_support.append(row["frame_sequence"])
        if wrong_support:
            merged.append({"identity": identity, "unsupported_frames": wrong_support,
                           "total_distinct_frames": cluster["distinct_viewpoints"]})
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "profile", "reference", "label-conversion", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--acquisition", type=Path)
    parser.add_argument("--fine-tuned", type=Path)
    parser.add_argument("--extended-tp", action="store_true")
    parser.add_argument("--surface-on-density", action="store_true")
    parser.add_argument("--density-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = yaml.safe_load(args.protocol.read_text())
    profile = yaml.safe_load(args.profile.read_text())
    map_config = IncrementalLandmarkMapConfig(**protocol["landmark_map"])
    evaluator = load_evaluator(protocol, args.label_conversion)
    connectivity = label(evaluator[0]["L4"]["sap"], generate_binary_structure(3, 1))[0]
    quantile = profile["E4"]["surface_proximity_quantile"]
    gate = protocol["evaluator"]["maximum_semantic_interface_distance_pixels"]
    if args.extended_tp:
        models, _ = load_models(protocol)
    if args.fine_tuned:
        receipt = read_json(args.fine_tuned / "training_result.json")
        checkpoint = Path(receipt["checkpoint"])
        if sha256(checkpoint) != receipt["checkpoint_sha256"]:
            raise ValueError("fine-tuned checkpoint checksum mismatch")
        tuned, thresholds = load_segmentation_checkpoint(checkpoint, expected_level="L5")

    report = {"input_hashes": {"protocol": sha256(args.protocol), "profile": sha256(args.profile)},
              "sessions": [], "acquisition_attempts": []}

    def audit(session, category):
        actor_path = session / "actor_result.json"
        actor = read_json(actor_path)
        seal = session / "actor_result.json.sha256"
        if seal.exists() and sha256(actor_path) != seal.read_text().split()[0]:
            raise ValueError(f"actor seal mismatch at {session}")
        region = Stage1Region(**actor["stage1_region"])
        frames, images, valid, coords, dropped, _ = load_public(
            session, protocol["geometry"]["ultrasound_pixel_resolution_mm"],
        )
        if not valid:
            report["sessions"].append({"session": str(session), "seed": actor["seed"],
                                       "arm": category, "status": "NO_VALID_CAPTURES"})
            return
        with np.load(session / "public/unet_masks.npz") as archive:
            masks = dict(archive)
        oracle_obs, labels, all_visible, _, projection, _ = oracle_observations(
            session, frames, images, valid, coords, evaluator, protocol,
        )
        actor_obs = observations(actor["observations"])
        amb = observations(actor["ambiguous_observations"])
        index_to_frame = {x["waypoint_index"]:x for x in frames}
        full_pose = read_jsonl(session / "privileged/full_pose_replay.jsonl")

        def evaluation_for(sequences):
            visible = {(k, i): d for (k, i), d in all_visible.items() if i in sequences}
            trees = {}
            coverage = {}
            for identity in LANDMARK_IDENTITIES:
                _, level, semantic = identity.split("_")
                clouds = [coords[i][labels[level][i] == CLASSES.index(semantic)]
                          for k, i in visible if k == identity]
                if clouds:
                    cloud = np.concatenate(clouds)
                    trees[identity] = cKDTree(cloud)
                    coverage[identity] = {"visible_frames": len(clouds), "pixels": len(cloud),
                                          "observed_1mm_voxels": len(np.unique(np.floor(cloud), axis=0))}
                else:
                    coverage[identity] = {"visible_frames": 0, "pixels": 0, "observed_1mm_voxels": 0}

            def assess(mapping, obs):
                return score_map(mapping, obs, visible, trees, gate)
            return assess, coverage, visible

        def run_point(obs, uncertain, assess, *, sensitivity=False):
            start = time.perf_counter()
            mapping = simple_map(obs, uncertain, map_config)
            runtime = time.perf_counter() - start
            output = {"mapping": mapping, "score": assess(mapping, obs), "mapping_seconds": runtime}
            if sensitivity:
                output["leave_one_view_out"] = leave_one_view_out(mapping, obs, uncertain, map_config, assess)
            return output

        def results(tag, budget, obs, uncertain, oracle_allowed, do_surface=False, sensitivity=False):
            available = {i for i in valid if frames[i]["waypoint_index"] < budget}
            assess, coverage, visible = evaluation_for(available)
            truth = {
                identity: [r for r in rows if r.frame_sequence in available and (
                    not identity.endswith("_TP") or oracle_allowed(frames[r.frame_sequence]))]
                for identity, rows in oracle_obs.items()
            }
            selected_frames = [frames[i] for i in sorted(available)]
            row = {"session": str(session), "seed": actor["seed"], "arm": tag, "planned_frames": budget,
                   "captured_frames": sum(i < budget for i in index_to_frame), "valid_frames": len(available),
                   "tracking": tracking(selected_frames), "oracle_coverage": coverage,
                   "capture_integrity": projection["passes"] and len(available) == budget,
                   "projection": projection, "failed_moves": [x for x in actor["failed_moves"] if x["waypoint"] < budget],
                   "dropped_counts_full_capture": dropped, "elapsed_inclusive_seconds": actor["elapsed_seconds"],
                   "sweep_seconds_through_last_valid_capture": selected_frames[-1].get("sweep_elapsed_seconds"),
                   "point": {"UNet": run_point(obs, uncertain, assess, sensitivity=sensitivity),
                             "oracle": run_point(truth, {}, assess)}}
            if do_surface:
                row["surface"] = {}
                for key, source, ambiguity, mask in (("oracle", truth, {}, labels),
                                                     ("UNet", obs, uncertain, masks)):
                    t = time.perf_counter()
                    mapping = surface_map(source, ambiguity, mask, coords, map_config, quantile)
                    runtime = time.perf_counter() - t
                    row["surface"][key] = {"mapping": mapping, "score": assess(mapping, source),
                                           "mapping_seconds": runtime,
                                           "selected_support_off_interface": mismatched_merges(mapping, visible, gate)}
                native = np.load(session / "privileged/pixel_to_volume_coordinates.npz")["human_img_coords"]
                row["oracle_l4_sap_competitors"] = oracle_competitors(
                    row["point"]["oracle"]["mapping"], labels, coords, native, connectivity, quantile,
                )
            report["sessions"].append(row)
            return row

        if category == "E0":
            row = results("E0", 54, actor_obs, amb, lambda _f: True, do_surface=True, sensitivity=True)
            if row["point"]["UNet"]["mapping"] != actor["landmarks"]:
                # Compare JSON encodings to account for tuples reconstructed from observations.
                if json.dumps(row["point"]["UNet"]["mapping"], sort_keys=True) != json.dumps(actor["landmarks"], sort_keys=True):
                    raise ValueError(f"point mapper did not reproduce the baseline at {session}")
            if args.fine_tuned:
                sap_rows, tp_rows, tp_amb = [], [], []
                for i in valid:
                    f = frames[i]
                    prob = segmentation_probabilities(tuned, images[i:i+1])[0]
                    rows, masks["L5"][i] = extract_observations(prob.argmax(0), prob, thresholds, f, coords[i])
                    for observation in rows:
                        if observation.semantic_class == "SAP":
                            sap_rows.append(observation)
                        elif ("L5" in f["level_hypotheses"] and f["tp_eligible_for_mapping"]
                              and observation.semantic_class == "TP"):
                            (tp_rows if len(f["level_hypotheses"]) == 1 else tp_amb).append(observation)
                retain = {k: v for k, v in actor_obs.items() if k.startswith("RIGHT_L4")}
                keep_ambiguity = {k: v for k, v in amb.items() if k.startswith("RIGHT_L4")}
                accept, uncertain, _ = route_sap({"L5": sap_rows}, set(valid), frames, region, protocol)
                merged = dict(retain, **accept, RIGHT_L5_TP=tp_rows)
                ambiguities = dict(keep_ambiguity, **uncertain, RIGHT_L5_TP=tp_amb)
                fine = results("E3", 54, merged, ambiguities, lambda _f: True)
                assess, _, _ = evaluation_for(set(valid))
                t = time.perf_counter()
                mapping = surface_map(merged, ambiguities, masks, coords, map_config, quantile)
                runtime = time.perf_counter() - t
                fine["surface"] = {"UNet": {"mapping": mapping, "score": assess(mapping, merged),
                                             "mapping_seconds": runtime}}
        elif category == "E2":
            results("E2", 54, actor_obs, amb, lambda f: f["tp_eligible_for_mapping"], sensitivity=True)
        else:
            proposals = defaultdict(list)
            for row in actor["sap_routing"]:
                proposal = SemanticObservation(**{k:v for k,v in row.items() if k in SemanticObservation.__dataclass_fields__})
                proposals[row["checkpoint_level"]].append(proposal)
            tp_extended, amb_extended = defaultdict(list), defaultdict(list)
            if args.extended_tp:
                for i in valid:
                    f = frames[i]
                    if f["tp_eligible_for_mapping"]:
                        continue
                    for level in f["level_hypotheses"]:
                        model, rules = models[level]
                        probs = segmentation_probabilities(model, images[i:i+1])[0]
                        rows, _ = extract_observations(probs.argmax(0), probs, rules, f, coords[i])
                        target = tp_extended if len(f["level_hypotheses"]) == 1 else amb_extended
                        target[f"RIGHT_{level}_TP"].extend(r for r in rows if r.semantic_class == "TP")
            for size, tag in [(profile["E1"][k], name) for k,name in
                              (("sparse", "E1-S"), ("medium", "E1-M"), ("dense", "E1-D"))]:
                available = {i for i in valid if frames[i]["waypoint_index"] < size}
                eligible, ambiguous, calibration = route_sap(proposals, available, frames, region, protocol)
                obs = dict(filter_rows(actor_obs, available, kind="TP"), **eligible)
                uncertain = dict(filter_rows(amb, available, kind="TP"), **ambiguous)
                row = results(tag, size, obs, uncertain, lambda f: f["tp_eligible_for_mapping"],
                              do_surface=args.surface_on_density, sensitivity=True)
                row["sap_craniocaudal_fit"] = asdict(calibration) if calibration is not None else None
                match = args.reference / f"session_{actor['seed']-2026121700:02d}"
                if tag == "E1-S" and match.exists():
                    old = read_jsonl(match / "public/frames.jsonl")
                    row["exact_image_hashes_against_historical_E0"] = sum(
                        f["waypoint_index"] < len(old) and f["frame_sha256"] == old[f["waypoint_index"]]["frame_sha256"]
                        for f in frames if f["waypoint_index"] < size
                    )
                if args.extended_tp and size > 54:
                    more, more_amb = dict(obs), dict(uncertain)
                    for dest, source in ((more, tp_extended), (more_amb, amb_extended)):
                        for identity, rows in filter_rows(source, available).items():
                            dest[identity] = dest.get(identity, []) + rows
                    results(tag + "-TP-all", size, more, more_amb, lambda _f: True, sensitivity=True)
        write_json(args.output / "results.json", report)
        print(category, actor["seed"], "evaluated", flush=True)

    if not args.density_only:
        for session in sorted(args.reference.glob("session_*")):
            audit(session, "E0")
    if args.acquisition:
        attempts_path = args.acquisition / "attempts.tsv"
        if attempts_path.exists():
            report["acquisition_attempts"] = attempts_path.read_text().splitlines()
        for session in sorted(args.acquisition.glob("E[12]_*")):
            if args.density_only and not session.name.startswith("E1_"):
                continue
            if (session / "actor_result.json").exists():
                audit(session, session.name.split("_")[0])
    write_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
