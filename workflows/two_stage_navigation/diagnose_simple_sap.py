#!/usr/bin/env python3
"""Trace SAP evidence on the original sweep, with evaluator data loaded offline."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from dataclasses import asdict

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
import torch
import yaml

from run_simple_landmark_mapping import (
    load_evaluator, load_models, load_public, oracle_observations, score_map, write_json,
)
from two_stage_navigation.incremental_landmark_map import IncrementalLandmarkMapConfig
from two_stage_navigation.simple_landmark_mapping import (
    CLASSES, Stage1Region, distinct_support, extract_observations, sap_level_at,
    simple_map, transverse_cranial_coordinate,
)
from two_stage_navigation.sonoanatomy_segmentation import segmentation_probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    torch.set_num_threads(1)
    frames, images, valid, coords, _, _ = load_public(args.session, 0.5)
    oracle, labels, visibility, reference, audit, _ = oracle_observations(
        args.session, frames, images, valid, coords,
        load_evaluator(protocol, Path("source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml")),
        protocol,
    )
    config = IncrementalLandmarkMapConfig(**protocol["landmark_map"])
    mapping = simple_map(oracle, {}, config)
    models, _ = load_models(protocol)
    old_actor = json.loads((args.session / "actor_result.json").read_text())
    old_region = old_actor["stage1_region"]
    selected_l5 = mapping["RIGHT_L5_SAP"]["selected"]["supporting_observations"]
    l4_sequences = sorted({row.frame_sequence for row in oracle["RIGHT_L4_SAP"]})
    geometry = protocol["geometry"]
    scan = protocol["sonoanatomy_scan"]
    fit = transverse_cranial_coordinate(
        frames, geometry["label_resolution_mm"], scan["translation_step_index"] / 2,
    )
    region = Stage1Region(**old_region)
    probabilities = {
        level: {index: probability for index, probability in zip(
            valid, segmentation_probabilities(model, images[valid]),
        )}
        for level, (model, _) in models.items()
    }
    rerouted, uncertain = defaultdict(list), defaultdict(list)
    from two_stage_navigation.sonoanatomy_multiview import SemanticObservation
    for name, rows in old_actor["observations"].items():
        if name.endswith("_TP"):
            rerouted[name] = [SemanticObservation(**row) for row in rows]
    for name, rows in old_actor["ambiguous_observations"].items():
        if name.endswith("_TP"):
            uncertain[name] = [SemanticObservation(**row) for row in rows]
    for level, (_, threshold) in models.items():
        for index, probability in probabilities[level].items():
            rows, _ = extract_observations(probability.argmax(axis=0), probability, threshold,
                                           frames[index], coords[index])
            for row in rows:
                if row.semantic_class != "SAP":
                    continue
                _, possible = sap_level_at(row.coordinate_mm, region, fit)
                if possible == (level,):
                    rerouted[f"RIGHT_{level}_SAP"].append(row)
                elif len(possible) > 1:
                    uncertain[f"RIGHT_{level}_SAP"].append(row)
    routing_only = score_map(simple_map(rerouted, uncertain, config), rerouted, visibility, reference, 3)
    l5_rows = []
    for row in selected_l5:
        k = row["frame_sequence"]
        coords_true = labels["L5"][k] == 2
        frame = frames[k]
        surface = visibility[("RIGHT_L5_SAP", k)]
        points = np.indices(coords_true.shape).reshape(2, -1).T
        roi = (surface.query(points)[0] <= 3).reshape(coords_true.shape)
        truth_outline = np.argwhere(coords_true)
        candidates = {}
        for level, (_, thresholds) in models.items():
            probs = probabilities[level][k]
            winner = probs.argmax(axis=0)
            options, _ = extract_observations(winner, probs, thresholds, frame, coords[k])
            best_rc = np.unravel_index(np.argmax(np.where(roi, probs[2], -1)), roi.shape)
            components, count = ndimage.label(
                (winner == 2) & (probs[2] >= thresholds["SAP"]["threshold"]), np.ones((3, 3)),
            )
            near_components = np.unique(components[roi])
            count_per_component = np.bincount(components.ravel(), minlength=count + 1)
            sizes = [int(count_per_component[j]) for j in near_components if j]
            candidates[level] = {
                "max_sap_probability_within_3px": float(probs[2][roi].max()),
                "max_sap_pixel_rc": list(map(int, best_rc)),
                "component_sizes_at_threshold_intersecting_3px": sizes,
                "winning_class_at_sap_peak": CLASSES[winner[best_rc]],
                "winning_class_at_interface_counts": {
                    name: int(np.sum(winner[coords_true] == ix)) for ix, name in enumerate(CLASSES)
                },
                "proposal_rows": [{
                    "class": p.semantic_class, "rc": [p.row, p.column], "confidence": p.confidence,
                    "pixel_count": p.pixel_count, "distance_pixels": float(surface.query([p.row, p.column])[0]),
                    "coordinate_mm": list(p.coordinate_mm),
                    "projected_z_index": sap_level_at(p.coordinate_mm, region, fit)[0],
                } for p in options if p.semantic_class in {"TP", "SAP"}],
                "sap_minimum_pixels": thresholds["SAP"]["minimum_pixels"],
                "sap_threshold": thresholds["SAP"]["threshold"],
            }
        retained = any(row["frame_sequence"] == k for row in old_actor["observations"]["RIGHT_L5_SAP"])
        l5 = candidates["L5"]
        stage = "RETAINED" if retained else "WRONG_VERTEBRAL_LEVEL" if "L5" not in frame["level_hypotheses"] else (
            "REJECTED_BY_CONFIDENCE" if l5["winning_class_at_sap_peak"] == "SAP" else "MISSED_BY_UNET"
        )
        l5_rows.append({
            "original_disposition": stage,
            "frame": k, "phase": frame["phase"], "executed_probe_pose": frame["executed_relative_pose_index"],
            "position_m": frame["actor_base_to_ee_position_m"],
            "quaternion_wxyz": frame["actor_base_to_ee_quaternion_wxyz"],
            "image_sha256": frame["frame_sha256"], "original_level": frame["level_hypotheses"],
            "oracle_rc": [row["row"], row["column"]], "oracle_coordinate_mm": row["coordinate_mm"],
            "oracle_eligible": True,
            "oracle_projected_z_index": sap_level_at(row["coordinate_mm"], region, fit)[0],
            "visible_sap_pixels_rc": truth_outline.tolist(), "models": candidates,
            "original_l5_observations": {
                key: [p for p in old_actor[key].get("RIGHT_L5_SAP", []) if p["frame_sequence"] == k]
                for key in ("observations", "ambiguous_observations")
            },
        })

    l4_pairs = []
    for k, j in zip(l4_sequences, l4_sequences[1:]):
        c1, c2 = coords[k][labels["L4"][k] == 2], coords[j][labels["L4"][j] == 2]
        r1, r2 = (max((r for r in oracle["RIGHT_L4_SAP"] if r.frame_sequence == n),
                      key=lambda r: r.pixel_count) for n in (k, j))
        l4_pairs.append({
            "frames": [k, j], "closest_interface_pixel_pair_mm": float(cdist(c1, c2).min()),
            "largest_component_representative_distance_mm": float(np.linalg.norm(
                np.asarray(r1.coordinate_mm) - r2.coordinate_mm
            )),
            "probe_translation_mm": float(1000 * np.linalg.norm(
                np.asarray(r1.viewpoint_position_m) - r2.viewpoint_position_m
            )),
        })
    write_json(args.output, {
        "projection": audit, "old_ambiguous_z_indices": old_region["ambiguous_z"],
        "projected_craniocaudal_calibration": asdict(fit) if fit else None,
        "routing_only_replay_score": routing_only,
        "L4_SAP": {
            "visible_frames": l4_sequences,
            "distinct_view_frames": [row.frame_sequence for row in distinct_support(oracle["RIGHT_L4_SAP"], config)],
            "clusters": mapping["RIGHT_L4_SAP"], "adjacent_visible_frame_pairs": l4_pairs,
        },
        "L5_SAP": l5_rows,
    })


if __name__ == "__main__":
    main()
