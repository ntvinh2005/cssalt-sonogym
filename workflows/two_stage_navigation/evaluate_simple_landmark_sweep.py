#!/usr/bin/env python3
"""Score the sealed actor map from one development sweep against visible interfaces."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import yaml

from run_simple_landmark_mapping import (
    ORACLE_TITLE, diagnose_visibility, load_evaluator, load_public, oracle_observations, score_map, sha256, write_json,
)
from two_stage_navigation.sonoanatomy_multiview import SemanticObservation
from two_stage_navigation.incremental_landmark_map import IncrementalLandmarkMapConfig
from two_stage_navigation.simple_landmark_mapping import CLASSES, simple_map

import numpy as np


def revised_sap_visibility(visible, labels, masks, rows, maximum_pixels):
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[(row["checkpoint_level"], row["frame_sequence"])].append(row)
    totals, frames = {}, {}
    for level in ("L4", "L5"):
        identity = f"RIGHT_{level}_SAP"
        count, accounting = Counter(), []
        for (name, sequence), surface in visible.items():
            if name != identity:
                continue
            matching = [
                row for row in by_frame[level, sequence]
                if surface.query([row["row"], row["column"]])[0] <= maximum_pixels
            ]
            count["oracle_visible_frames"] += 1
            reasons = sorted({row["status"] for row in matching})
            if any(row["status"] == "ELIGIBLE" for row in matching):
                reason = "usable_correct_prediction"
            elif matching:
                reason = "routing_" + "+".join(reasons)
            else:
                observed = masks[level][sequence][labels[level][sequence] == 2]
                reason = "miss_dominant_mask_class_" + CLASSES[int(np.bincount(observed).argmax())]
            count[reason] += 1
            accounting.append({"frame_sequence": sequence, "reason": reason, "matching_rows": matching})
        totals[identity], frames[identity] = dict(count), accounting
    return totals, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--oracle-diagnostic", type=Path)
    args = parser.parse_args()
    output = args.output or args.session / "evaluation.json"
    if output.exists() or (args.oracle_diagnostic and args.oracle_diagnostic.exists()):
        raise FileExistsError("evaluation output already exists")
    protocol = yaml.safe_load(args.protocol.read_text())
    actor_path = args.session / "actor_result.json"
    actor = json.loads(actor_path.read_text())
    if actor["protocol_sha256"] != sha256(args.protocol):
        raise ValueError("fresh actor map and evaluation protocol differ")
    if not actor["completed_trajectory"] or actor["valid_frame_count"] == 0:
        write_json(output, {"status": "FAIL", "reason": "incomplete acquisition"})
        return
    frames, images, valid, coordinates, dropped, _ = load_public(
        args.session, protocol["geometry"]["ultrasound_pixel_resolution_mm"],
    )
    evaluator = load_evaluator(protocol, args.label_conversion)
    oracle, labels, visible, reference, projection, _ = oracle_observations(
        args.session, frames, images, valid, coordinates, evaluator, protocol,
    )
    observations, ambiguities = (
        {identity: [SemanticObservation(**row) for row in rows] for identity, rows in actor[key].items()}
        for key in ("observations", "ambiguous_observations")
    )
    gate = protocol["evaluator"]["maximum_semantic_interface_distance_pixels"]
    score = score_map(actor["landmarks"], observations, visible, reference, gate)
    with np.load(args.session / "public/unet_masks.npz") as archive:
        masks = dict(archive)
    route = {seq: frames[seq]["level_hypotheses"] for seq in valid}
    valid_by_plane = {plane: sum(frames[seq]["phase"] == plane for seq in valid)
                      for plane in ("longitudinal", "transverse")}
    valid_capture = (projection["passes"] and all(valid_by_plane.values())
                     and not actor["failed_moves"] and not dropped)
    status = "PASS" if valid_capture and score["complete_map"] else (
        "PARTIAL" if projection["passes"] and score["correct_mapped_landmarks"] else "FAIL"
    )
    visibility_diagnostics = diagnose_visibility(
        visible, labels, observations, ambiguities, masks, route, gate,
    )
    revised_frames = None
    if actor.get("revision_sha256"):
        sap_diagnostics, revised_frames = revised_sap_visibility(
            visible, labels, masks, actor["sap_routing"], gate,
        )
        visibility_diagnostics.update(sap_diagnostics)
    write_json(output, {
        "label": actor["label"], "status": status, "actor_result_sha256": sha256(actor_path),
        "valid_frames_by_plane": valid_by_plane, "dropped_frames_by_flags": dropped,
        "projection_check": projection, "score": score,
        "visible_interface_diagnostics": visibility_diagnostics,
        **({"revised_sap_frames": revised_frames} if revised_frames is not None else {}),
        "oracle_observation_counts": {identity: len(rows) for identity, rows in oracle.items()},
    })

    if args.oracle_diagnostic:
        mapping = simple_map(oracle, {}, IncrementalLandmarkMapConfig(**protocol["landmark_map"]))
        write_json(args.oracle_diagnostic, {
            "label": ORACLE_TITLE, "source_fresh_session": str(args.session),
            "map": mapping, "score": score_map(mapping, oracle, visible, reference, gate),
        })


if __name__ == "__main__":
    main()
