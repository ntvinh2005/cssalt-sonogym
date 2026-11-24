#!/usr/bin/env python3
"""Score the sealed actor map from one development sweep against visible interfaces."""

import argparse
import json
from pathlib import Path

import yaml

from run_simple_landmark_mapping import (
    diagnose_visibility, load_evaluator, load_public, oracle_observations, score_map, sha256, write_json,
)
from two_stage_navigation.sonoanatomy_multiview import SemanticObservation

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    actor_path = args.session / "actor_result.json"
    actor = json.loads(actor_path.read_text())
    if actor["protocol_sha256"] != sha256(args.protocol):
        raise ValueError("fresh actor map and evaluation protocol differ")
    if not actor["completed_trajectory"] or actor["valid_frame_count"] == 0:
        write_json(args.session / "evaluation.json", {"status": "FAIL", "reason": "incomplete acquisition"})
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
    write_json(args.session / "evaluation.json", {
        "label": actor["label"], "status": status, "actor_result_sha256": sha256(actor_path),
        "valid_frames_by_plane": valid_by_plane, "dropped_frames_by_flags": dropped,
        "projection_check": projection, "score": score,
        "visible_interface_diagnostics": diagnose_visibility(
            visible, labels, observations, ambiguities, masks, route, gate,
        ),
        "oracle_observation_counts": {identity: len(rows) for identity, rows in oracle.items()},
    })


if __name__ == "__main__":
    main()
