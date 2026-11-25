#!/usr/bin/env python3
"""Render SAP diagnosis examples and compare repeated requested poses."""

import argparse
from pathlib import Path

import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from render_simple_landmark_mapping import load, overlay
from run_simple_landmark_mapping import load_evaluator, load_public, oracle_observations, score_map, write_json
from two_stage_navigation.incremental_landmark_map import IncrementalLandmarkMapConfig
from two_stage_navigation.simple_landmark_mapping import simple_map
from two_stage_navigation.sonoanatomy_multiview import SemanticObservation


def audit_figure(args, images):
    rows = load(args.diagnosis)["L5_SAP"]
    figure, axes = plt.subplots(len(rows), 3, figsize=(10.1, 13.2), layout="constrained")
    for row, (raw, oracle, predicted) in zip(rows, axes):
        index = row["frame"]
        image = images[index]
        rgb = overlay(image, np.zeros_like(image))
        pixels = np.asarray(row["visible_sap_pixels_rc"])
        model = row["models"]["L5"]
        for axis in (raw, oracle, predicted):
            axis.imshow(rgb)
            axis.set_xlim(max(0, pixels[:, 1].min() - 32), min(image.shape[1], pixels[:, 1].max() + 32))
            axis.set_ylim(min(image.shape[0], pixels[:, 0].max() + 29), max(0, pixels[:, 0].min() - 29))
            axis.set_axis_off()
        raw.set_title(f"Frame {index:02d} · {row['phase']}", fontsize=11)
        oracle.scatter(pixels[:, 1], pixels[:, 0], s=9, c="#f3b538")
        oracle.set_title("Provisional visible SAP", fontsize=11)
        rr, cc = model["max_sap_pixel_rc"]
        predicted.scatter([cc], [rr], marker="+", s=80, c="#f25be8")
        for proposal in model["proposal_rows"]:
            if proposal["class"] == "SAP" and proposal["distance_pixels"] <= 3:
                predicted.scatter(proposal["rc"][1], proposal["rc"][0], s=80, facecolors="none",
                                  edgecolors="#14e6a8")
        predicted.set_title(f"L5 SAP peak {model['max_sap_probability_within_3px']:.3g} · "
                            f"{'+'.join(row['original_level'])} originally", fontsize=10)
    figure.suptitle("OFFLINE EVALUATOR DIAGNOSIS · s0010 · original sweep", fontsize=14)
    figure.savefig(args.output / "sap_evidence_examples.png", dpi=145)
    plt.close(figure)


def repeated_pose_figure(args, original):
    bundles = [original]
    sessions = [args.original]
    for session in args.revised:
        if not (session / "evaluation.json").is_file():
            continue
        evaluation = load(session / "evaluation.json")
        if "score" not in evaluation:
            continue
        frames, images, valid, _, _, _ = load_public(session, 0.5)
        with np.load(session / "public/unet_masks.npz") as archive:
            masks = dict(archive)
        bundles.append((frames, images, valid, masks))
        sessions.append(session)
    figure, axes = plt.subplots(2, len(bundles), squeeze=False,
                                figsize=(3.4 * len(bundles), 6.9), layout="constrained")
    comparisons = []
    for axes_row, waypoint in zip(axes, (31, 43)):
        baseline_frames = original[0]
        baseline = next(row for row in baseline_frames if row["waypoint_index"] == waypoint)
        target = baseline["requested_relative_pose_index"]
        for axis, session, (frames, images, valid, masks) in zip(axes_row, sessions, bundles):
            axis.set_axis_off()
            candidate = next((row for row in frames
                              if np.allclose(row["requested_relative_pose_index"], target, atol=1e-8, rtol=0.0)), None)
            label = "original"
            if session != args.original:
                label = session.name.rsplit("_", 1)[-1]
                if load(session / "actor_result.json").get("tp_evidence_on_added_sap_waypoints") is False:
                    label += " repair"
            if candidate is None or candidate["frame_sequence"] not in valid:
                axis.set_title(f"{label} · unavailable", fontsize=10)
                continue
            sequence = candidate["frame_sequence"]
            image = images[sequence]
            mask = masks["L5"][sequence]
            pixels = int(np.count_nonzero(mask == 2)) if not np.all(mask == 255) else None
            axis.imshow(overlay(image, mask))
            axis.set_title(f"{label} · frame {sequence}\n"
                           f"L5 SAP mask px: {pixels if pixels is not None else 'model skipped'}", fontsize=10)
            displacement = 1000 * float(np.linalg.norm(
                np.asarray(candidate["actor_base_to_ee_position_m"]) - baseline["actor_base_to_ee_position_m"]
            ))
            difference = np.asarray(candidate["actor_base_to_ee_rotation_columns"]).T @ np.asarray(
                baseline["actor_base_to_ee_rotation_columns"]
            )
            angle = float(np.degrees(np.arccos(np.clip((np.trace(difference) - 1) / 2, -1, 1))))
            comparisons.append({
                "original_waypoint_index": waypoint, "session": str(session), "frame_sequence": sequence,
                "requested_pose_index": target, "executed_pose_delta_mm": displacement,
                "executed_angle_delta_deg": angle, "image_sha256": candidate["frame_sha256"],
                "same_image_as_original": candidate["frame_sha256"] == baseline["frame_sha256"],
                "l5_sap_mask_pixels": pixels,
            })
    figure.suptitle("Repeated requested positions · reset, tracking and image variation", fontsize=13)
    figure.savefig(args.output / "repeated_requested_poses.png", dpi=145)
    plt.close(figure)
    write_json(args.output / "repeated_requested_poses.json", comparisons)


def original_tp_waypoints(args):
    protocol = yaml.safe_load(args.protocol.read_text())
    config = IncrementalLandmarkMapConfig(**protocol["landmark_map"])
    geometry = load_evaluator(protocol, Path("source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"))
    base = {tuple(pose) for _, pose in load(args.original / "waypoints.json")}
    records = []
    for session in args.revised:
        if not (session / "oracle_diagnostic.json").is_file():
            continue
        frames, images, valid, coordinates, _, _ = load_public(session, 0.5)
        actor = load(session / "actor_result.json")
        _, _, visible, reference, _, _ = oracle_observations(
            session, frames, images, valid, coordinates, geometry, protocol,
        )
        selected = {
            name: [
                SemanticObservation(**row) for row in rows
                if not name.endswith("_TP")
                or tuple(frames[row["frame_sequence"]]["requested_relative_pose_index"]) in base
            ]
            for name, rows in actor["observations"].items()
        }
        ambiguous = {
            name: [
                SemanticObservation(**row) for row in rows
                if not name.endswith("_TP")
                or tuple(frames[row["frame_sequence"]]["requested_relative_pose_index"]) in base
            ]
            for name, rows in actor["ambiguous_observations"].items()
        }
        mapping = simple_map(selected, ambiguous, config)
        gate = protocol["evaluator"]["maximum_semantic_interface_distance_pixels"]
        records.append({
            "seed": actor["seed"], "source_session": str(session), "revision_sha256": actor["revision_sha256"],
            "acquired_result": load(session / "evaluation.json")["score"],
            "tp_original_waypoints_replay": score_map(mapping, selected, visible, reference, gate),
            "l4_tp_supports": [{
                "frames": [r["frame_sequence"] for r in cluster["supporting_observations"]],
                "added_waypoint_frames": [
                    r["frame_sequence"] for r in cluster["supporting_observations"]
                    if tuple(frames[r["frame_sequence"]]["requested_relative_pose_index"]) not in base
                ],
                "mean_confidence": cluster["semantic_confidence"],
                "position_mm": cluster["position_mm"],
            } for cluster in (
                [actor["landmarks"]["RIGHT_L4_TP"]["selected"]]
                + actor["landmarks"]["RIGHT_L4_TP"]["competing_clusters"][:1]
            ) if cluster is not None],
        })
    write_json(args.output / "tp_original_waypoints_diagnostic.json", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--diagnosis", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revised", nargs="*", default=(), type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frames, images, valid, _, _, _ = load_public(args.original, 0.5)
    with np.load(args.original / "public/unet_masks.npz") as archive:
        masks = dict(archive)
    audit_figure(args, images)
    repeated_pose_figure(args, (frames, images, valid, masks))
    original_tp_waypoints(args)


if __name__ == "__main__":
    main()
