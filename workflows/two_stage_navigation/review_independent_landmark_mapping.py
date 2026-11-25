#!/usr/bin/env python3
"""Summarize and render every attempted independent session using frozen scores."""

from __future__ import annotations

import argparse
from datetime import datetime
from itertools import combinations
import math
from pathlib import Path
import shutil

import numpy as np
import yaml

import render_simple_sap_demo as demo
from render_simple_landmark_mapping import actor_figure, load
from run_simple_landmark_mapping import check_seal, load_public, write_json


LANDMARKS = ("RIGHT_L4_TP", "RIGHT_L4_SAP", "RIGHT_L5_TP", "RIGHT_L5_SAP")
TITLE = "INDEPENDENT ANATOMICAL MAPPING - s0010"


def diagnose(name, summary, hypothesis, oracle, visibility):
    if summary["outcome"] == "CORRECT":
        return None
    if summary["outcome"] == "WRONG":
        return "Accepted U-Net cluster failed the frozen majority visible-interface score"
    if hypothesis["status"] == "AMBIGUOUS" and hypothesis["selected"] and hypothesis["competing_clusters"]:
        return "Competing spatial clusters had insufficient separation in support or confidence"
    if visibility.get("oracle_visible_frames", 0) <= 1:
        return f"Insufficient {name.split('_')[-1]} visibility"
    if oracle["outcome"] == "CORRECT":
        if any(key.startswith("routing_") and val for key, val in visibility.items()):
            return "SAP level assignment withheld some otherwise matching components"
        return "U-Net observations provided insufficient support for the visible-interface cluster"
    return "Visible interfaces did not provide a uniquely supported spatial cluster"


def read_session(session, index, seed, protocol, timeline, output):
    settings = protocol["acquisition"]
    row = {"session": index, "seed": seed, "attempt": timeline, "scored": False}
    if not (session / "actor_result.json").is_file():
        return row
    dest = output / f"session_{index}"
    dest.mkdir()
    for relative in ("actor_result.json", "actor_result.json.sha256", "evaluation.json",
                     "oracle_diagnostic.json", "waypoints.json", "public/frames.jsonl"):
        if (session / relative).is_file():
            (dest / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(session / relative, dest / relative)
    actor = load(session / "actor_result.json")
    row.update({"valid_frames": actor["valid_frame_count"], "observed_frames": actor["frame_count"],
                "initial_executed_pose_index": actor["initial_executed_pose_index"],
                "action_steps": actor["action_steps"], "actor_elapsed_seconds": actor["elapsed_seconds"],
                "mapping_seconds": actor["mapping_seconds"], "failed_moves": actor["failed_moves"],
                "planned_frames": settings["waypoint_count"]})
    if len(timeline) >= 2:
        row["wall_seconds"] = (datetime.fromisoformat(timeline[-1][1])
                               - datetime.fromisoformat(timeline[0][1])).total_seconds()
    if not (session / "evaluation.json").is_file() or not timeline or timeline[-1][0] != "EVALUATED":
        return row
    evidence = load(session / "evaluation.json")
    if "score" not in evidence:
        return row
    digest = check_seal(session / "actor_result.json", session / "actor_result.json.sha256")
    seal_ok = evidence["actor_result_sha256"] == digest
    frames, _, valid, _, _, _ = load_public(session, protocol["geometry"]["ultrasound_pixel_resolution_mm"])
    if len(valid) != actor["valid_frame_count"]:
        raise RuntimeError(f"valid-frame accounting mismatch in session {index}")
    row["first_acquisition_pose_index"] = frames[valid[0]]["executed_relative_pose_index"]
    row["frame_hashes"] = [frames[i]["frame_sha256"] for i in valid]
    oracle = load(session / "oracle_diagnostic.json")["score"]
    scored = evidence["score"]
    table = {}
    for name in LANDMARKS:
        landmark = scored["landmarks"][name]
        hypothesis = actor["landmarks"][name]
        selected = hypothesis["selected"]
        support = []
        if selected:
            for entry, distance in zip(selected["supporting_observations"], landmark["support_distances_pixels"],
                                       strict=True):
                frame = frames[entry["frame_sequence"]]
                support.append({
                    "acquisition_id": frame["acquisition_id"], "frame_sequence": frame["frame_sequence"],
                    "frame_sha256": frame["frame_sha256"], "nearest_visible_pixels": distance,
                })
        table[name] = {
            **landmark, "rms_dispersion_mm": selected["rms_dispersion_mm"] if selected else None,
            "support": support, "oracle": oracle["landmarks"][name],
            "diagnosis": diagnose(name, landmark, hypothesis, oracle["landmarks"][name],
                                  evidence["visible_interface_diagnostics"].get(name, {})),
        }
    integrity = (
        seal_ok and actor["completed_trajectory"] and not actor["failed_moves"]
        and actor["frame_count"] == len(valid) == settings["waypoint_count"]
        and evidence["projection_check"]["passes"] and not evidence["dropped_frames_by_flags"]
        and evidence["valid_frames_by_plane"] == settings["per_plane"]
    )
    row.update({"scored": True, "landmarks": table, "acquisition_integrity": integrity,
                "unique_ultrasound_hashes": len(set(row["frame_hashes"])),
                "projection_check": evidence["projection_check"],
                "oracle_complete_map": oracle["complete_map"],
                "complete_map": integrity and scored["complete_map"],
                **{name: scored[name] for name in ("correct_mapped_landmarks", "wrong_mapped_landmarks",
                                                  "unresolved_landmarks")}})
    return row


def render_session(session, destination, score, source_protocol, make_video):
    evaluation = load(session / "evaluation.json")
    actor = load(session / "actor_result.json")
    frames, images, valid, _, _, _ = load_public(session, source_protocol["geometry"]["ultrasound_pixel_resolution_mm"])
    actor["level_hypotheses_by_frame"] = {str(i): frames[i]["level_hypotheses"] for i in valid}
    with np.load(session / "public/unet_masks.npz") as archive:
        masks = dict(archive)
    actor_figure(destination / "final_actor_map.png", f"{session.name} - seed {score['seed']}",
                 images, frames, valid, masks, actor, actor["landmarks"], title=TITLE)
    if make_video:
        demo.TITLE = TITLE
        path = destination / "independent_systematic_sweep.mp4"
        manifest = demo.render_video(session, path, actor, frames, images, valid, masks, evaluation, source_protocol)
        manifest["path"] = str(path)
        write_json(destination / "video_manifest.json", manifest)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--source-protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    protocol_hash = check_seal(args.protocol, args.protocol.with_suffix(".sha256"))
    protocol = yaml.safe_load(args.protocol.read_text())
    source = yaml.safe_load(args.source_protocol.read_text())
    attempts = {}
    log = args.campaign / "attempts.tsv"
    if log.is_file():
        for line in log.read_text().splitlines():
            index, seed, state, stamp = line.split("\t")
            attempts.setdefault(index, []).append((state, stamp))
    rows = [
        read_session(args.campaign / f"session_{i:02d}", f"{i:02d}", seed, protocol,
                     attempts.get(f"{i:02d}", []), args.output)
        for i, seed in enumerate(protocol["seeds"])
    ]
    scored = [row for row in rows if row["scored"]]
    pairs = []
    for a, b in combinations(scored, 2):
        left, right = a["initial_executed_pose_index"], b["initial_executed_pose_index"]
        distance = 1.5 * math.dist(left[:2], right[:2])
        angle = abs(math.degrees((left[2] - right[2] + math.pi) % (2 * math.pi) - math.pi))
        pairs.append({"sessions": [a["session"], b["session"]], "initial_translation_mm": distance,
                      "initial_rotation_deg": angle, "identical_frame_hashes": a["frame_hashes"] == b["frame_hashes"]})
    distinct = len(pairs) == 6 and all(
        not p["identical_frame_hashes"]
        and (p["initial_translation_mm"] >= protocol["mapper"]["distinct_view_translation_mm"]
             or p["initial_rotation_deg"] >= protocol["mapper"]["distinct_view_rotation_deg"]) for p in pairs
    )
    totals = {name: sum(row[name] for row in scored) for name in (
        "correct_mapped_landmarks", "wrong_mapped_landmarks", "unresolved_landmarks", "complete_map",
    )}
    passed = distinct and totals["complete_map"] == 4 and totals["correct_mapped_landmarks"] == 16
    status = "INCOMPLETE" if len(scored) != 4 else "PASS" if passed else "FAIL"
    representative = next((row for row in scored if not row["complete_map"]), scored[0] if scored else None)
    result = {
        "status": status, "protocol_sha256": protocol_hash, "scored_sessions": len(scored),
        "distinct_realized_starts": distinct, "initial_pose_pairs": pairs, "totals": totals,
        "sessions": rows, "video": None,
    }
    write_json(args.output / "confirmation.json", result)
    for row in scored:
        folder = f"session_{row['session']}"
        manifest = render_session(args.campaign / folder, args.output / folder, row, source, row is representative)
        if manifest:
            result["video"] = manifest
    write_json(args.output / "confirmation.json", result)


if __name__ == "__main__":
    main()
