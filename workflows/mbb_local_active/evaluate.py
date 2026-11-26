"""Check sealed acquisitions, image novelty and Oracle evidence after live verification."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import nibabel as nib
import yaml
from scipy.spatial.distance import cdist

from run_simple_landmark_mapping import load_evaluator, load_public, oracle_observations
from two_stage_navigation.sonoanatomy_labels import LEVEL_SOURCE_LABEL, first_bone_surface
from two_stage_navigation.ultrasound_plane_geometry import sample_label_image
from workflows.mbb_direct_sonographic.localize import acquisition_views, distinct_poses, region_distance, verify_session
from workflows.mbb_direct_sonographic.prepare import read_json, verify, write_json
from workflows.mbb_direct_sonographic.replay import NAMES, write_table
from workflows.mbb_adaptive_cortical.evaluate import LABEL_CONVERSION

from .live import BASE, decisions


def region_matches(frame_id, prediction, semantic, raw, bone_ids, rules):
    pixels = prediction["frames"][frame_id]["candidate_pixels"]
    front = first_bone_surface(raw[frame_id], bone_ids)
    categories = []
    for row, col in pixels:
        expected = np.flatnonzero(front[:, col])
        if not len(expected) or abs(expected[0] - row) * rules["pixel_mm"] > 1.0:
            categories.append("unmatched")
        elif int(raw[frame_id, expected[0], col]) != LEVEL_SOURCE_LABEL["L5"]:
            categories.append("other_first_visible_bone")
        else:
            names = {name for name, (points, _) in semantic[frame_id].items()
                     if name.startswith("RIGHT_L5") and len(points) and np.min(
                         np.linalg.norm(points - (row, col), axis=1)) * rules["pixel_mm"] <= 1.0}
            categories.append(names.pop() if len(names) == 1 else "L5_subregion_uncertain")
    return dict(Counter(categories))


def evaluate(session, protocol_path, archived, output):
    for line in (session / "local_actor.sha256").read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        verify(session / name, checksum)
    actor = read_json(session / "local_actor.json")
    protocol = yaml.safe_load(protocol_path.read_text())
    rules = read_json(BASE / "mbb_direct_sonographic_v2.json")
    config = read_json(BASE / "mbb_adaptive_cortical_tracking_v1.json")
    frames, images, valid, coords, dropped, _ = load_public(session, rules["pixel_mm"])
    output.mkdir(parents=True, exist_ok=False)
    initial = actor["initial_frame_count"]
    if initial is None or len(actor["frames"]) != len(frames):
        raise ValueError("capture did not complete the matched comparison")
    candidate_source = actor["source_sequence"]
    eligible = actor["eligible_sequences"]
    before = [i for i in eligible if i < initial]
    if any(i not in valid for i in eligible):
        raise ValueError("invalid or stale view entered verification")

    # Oracle geometry is loaded after verifying the actor prediction and public recording seals.
    evaluator = load_evaluator(protocol, LABEL_CONVERSION)
    source_volume = np.asarray(nib.load(protocol["evaluator"]["source_segmentation_path"]).dataobj, dtype=np.uint8)
    with np.load(session / "privileged/pixel_to_volume_coordinates.npz", allow_pickle=False) as native:
        coordinates = native["human_img_coords"]
    raw = np.stack([sample_label_image(volume=source_volume, coordinates_index=pose,
                                      width=images.shape[2], depth=images.shape[1])
                    for pose in coordinates]).astype(np.uint8)
    np.savez_compressed(session / "privileged/raw_labels.npz", raw_labels=raw)
    _, masks, _, _, parity, _ = oracle_observations(session, frames, images, valid, coords, evaluator, protocol)
    # Labeled pixels are already first-visible and image-supported in the existing Oracle.
    names = []
    view, _ = acquisition_views(frames, rules)
    oracle = []
    for i, (family, sign) in enumerate(view):
        named = {name: (np.zeros((0, 2), dtype=int), np.zeros(0)) for name in NAMES}
        if i in valid:
            for level in ("L4", "L5"):
                for code, name in ((1, "TP"), (2, "SAP")):
                    pixels = np.argwhere(masks[level][i] == code)
                    named[f"RIGHT_{level}_{name}"] = pixels, np.ones(len(pixels))
            _, estimate = decisions(images[i], named, family, sign, rules, config)
        else:
            estimate = {"status": "INVALID_ACQUISITION", "direct_candidate": False, "joined_profiles": 0}
        names.append(named)
        oracle.append(estimate)
    estimates = {"unet": actor["frames"], "oracle_visible": oracle}
    prior = {}
    old_metadata = read_json(archived / "replay/frame_audits.json")
    if len(frames) > initial:
        for identifier, data in old_metadata.items():
            with np.load(archived / "inputs" / identifier / "bmode.npz", allow_pickle=False) as capture:
                for p, image in zip(data["public_poses"], capture["ultrasound"]):
                    prior.setdefault(p["frame_sha256"], (identifier, p["frame_sequence"], image))
    references = list(prior.values())
    distinct = distinct_poses(frames, rules)
    novelty, frame_records, spans, pairs = [], [], [], []
    for i in range(initial, len(frames)):
        distances = [float(np.mean(np.abs(images[i] - ref[2]))) for ref in references]
        j = int(np.argmin(distances))
        pose_a, pose_b = frames[candidate_source], frames[i]
        rotation_a = np.asarray(pose_a["actor_base_to_ee_rotation_columns"])
        rotation_b = np.asarray(pose_b["actor_base_to_ee_rotation_columns"])
        geodesic = float(np.rad2deg(np.arccos(np.clip((np.sum(rotation_a * rotation_b) - 1.) / 2., -1, 1))))
        translation = float(1000 * np.linalg.norm(np.asarray(pose_a["actor_base_to_ee_position_m"])
                                                  - pose_b["actor_base_to_ee_position_m"]))
        novelty.append({
            "frame": i, "executed_translation_from_source_mm": translation,
            "executed_rotation_from_source_deg": geodesic, "valid": i in valid, "eligible_for_verification": i in eligible,
            "image_sha256": frames[i]["frame_sha256"], "nearest_archived_session": references[j][0],
            "nearest_archived_frame": references[j][1], "raw_mae_nearest_archived": distances[j],
            "equal_to_archived_pixels": frames[i]["frame_sha256"] in prior,
            "distinct_vs_fresh_source": bool(distinct[i, candidate_source]),
        })
    if novelty:
        write_table(output / "local_image_novelty.csv", novelty)
    for perception, evidence in estimates.items():
        for scope, ids in (("before", before), ("after", eligible)):
            result = verify_session(frames, evidence, ids, rules)
            accepted_pairs = []
            anatomy = {}
            for i, region in result["regions"].items():
                identity = region_matches(i, {"frames": evidence}, names, raw, evaluator[1], rules)
                anatomy[i] = identity
                frame_records.append({"perception": perception, "scope": scope, "frame": i,
                    "same_ray_anatomy": identity, "candidate_xyz_mm": region["candidate_xyz_mm"]})
            for i, j in combinations(result["candidate_frames"], 2):
                a = result["regions"][i]["candidate_xyz_mm"]
                b = result["regions"][j]["candidate_xyz_mm"]
                distances = cdist(a, b)
                pair = {"perception": perception, "scope": scope, "a": i, "b": j,
                        "distinct": bool(distinct[i, j]), "bidirectional_p95_mm": region_distance(a, b),
                        "a_within_half_mm": float(np.mean(distances.min(axis=1) <= .5)),
                        "b_within_half_mm": float(np.mean(distances.min(axis=0) <= .5))}
                pairs.append(pair)
                if pair["distinct"]:
                    accepted_pairs.append(pair)
            usable = range(initial) if scope == "before" else range(len(frames))
            spans.append({
                "perception": perception, "scope": scope, "valid_frames": sum(i in valid for i in usable),
                "eligible_frames": len(ids), "joined_profiles": sum(evidence[i]["joined_profiles"] for i in ids),
                "candidate_regions": len(result["regions"]), "support_frames": result["support_frames"],
                "candidate_frames": result["candidate_frames"],
                "distinct_candidate_pairs": len(accepted_pairs),
                "max_distinct_bidirectional_p95_mm": max((p["bidirectional_p95_mm"] for p in accepted_pairs), default=None),
                "qualified_first_visible_L5_candidates": sum(bool(stats) and all(
                    name == "L5_subregion_uncertain" or name.startswith("RIGHT_L5") for name in stats)
                    for stats in anatomy.values()),
                "acceptance": result["accepted"], "status": result["status"],
            })
    write_json(output / "evaluation.json", {
        "comparisons": spans, "pair_evidence": pairs, "candidate_projection_evidence": frame_records,
        "projection_parity": parity, "dropped": dropped, "novelty": novelty,
        "oracle_frames": oracle, "independent_target_status": "TARGET_LOCALIZATION_NOT_EVALUABLE",
        "initial_runtime_s": actor["initial_elapsed_seconds"], "additional_runtime_s": actor["local_elapsed_seconds"],
    })
    local = []
    for i in range(initial, len(frames)):
        for mode, evidence in estimates.items():
            row = evidence[i]
            local.append({"sequence": i, "perception": mode, "acquisition_id": frames[i]["acquisition_id"],
                          "requested": frames[i]["requested_relative_pose_index"],
                          "executed": frames[i]["actor_base_to_ee_position_m"],
                          "rotation": frames[i]["actor_base_to_ee_rotation_columns"],
                          "image_sha256": frames[i]["frame_sha256"], "valid": i in valid,
                          "existing_verifier_eligible": i in eligible, "status": row["status"],
                          "support_pixels": row.get("support_pixels"), "candidate_pixels": row.get("candidate_pixels")})
    write_json(output / "local_attempts.json", {"recorded": local, "failed": actor["failed_moves"]})
    print(spans, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("session", "protocol-path", "archived", "output"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    evaluate(*vars(cli.parse_args()).values())
