#!/usr/bin/env python3
"""Apply the frozen SAP processing to compatible v6r19 recorded acquisitions."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
import yaml

from run_simple_landmark_mapping import load_models, load_public, score_map, write_json
from two_stage_navigation.incremental_landmark_map import IncrementalLandmarkMapConfig
from two_stage_navigation.simple_landmark_mapping import (
    Stage1Region, extract_observations, sap_level_at, simple_map, transverse_cranial_coordinate,
)
from two_stage_navigation.sonoanatomy_multiview import SemanticObservation
from two_stage_navigation.sonoanatomy_segmentation import segmentation_probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    torch.set_num_interop_threads(1)
    protocol = yaml.safe_load(args.protocol.read_text())
    baseline = json.loads((args.baseline / "comparison.json").read_text())
    models, _ = load_models(protocol)
    config = IncrementalLandmarkMapConfig(**protocol["landmark_map"])
    scan, geometry = protocol["sonoanatomy_scan"], protocol["geometry"]
    sessions = []
    for row in baseline["sessions"]:
        folder = args.baseline / row["session"]
        frames, images, valid, coords, _, _ = load_public(Path(row["source"]), geometry["ultrasound_pixel_resolution_mm"])
        if valid != row["valid_frames"]:
            raise RuntimeError("revised replay frame set differs from original replay")
        transverse = [
            dict(frames[i], phase="transverse") for i in valid
            if frames[i].get("trajectory_id", "").split(":")[-1] == "transverse"
            and abs(frames[i]["executed_relative_pose_index"][2] - scan["transverse_angle_rad"])
            <= np.radians(protocol["motion"]["coarse_orientation_tolerance_deg"])
        ]
        fit = transverse_cranial_coordinate(
            transverse, geometry["label_resolution_mm"], scan["translation_step_index"] / 2,
        )
        actor = json.loads((folder / "actor/observations.json").read_text())
        region = Stage1Region(**actor["stage1_region"])
        pools = []
        for key in ("observations", "ambiguous_observations"):
            pools.append(defaultdict(list, {
                name: [SemanticObservation(**observation) for observation in group]
                for name, group in actor[key].items() if name.endswith("_TP")
            }))
        selected, ambiguous = pools
        counts = Counter()
        for level, (model, threshold) in models.items():
            for offset in range(0, len(valid), 8):
                for index, probability in zip(valid[offset:offset+8],
                                              segmentation_probabilities(model, images[valid[offset:offset+8]])):
                    observations, _ = extract_observations(
                        probability.argmax(axis=0), probability, threshold, frames[index], coords[index],
                    )
                    for observation in observations:
                        if observation.semantic_class != "SAP":
                            continue
                        _, possible = sap_level_at(observation.coordinate_mm, region, fit)
                        if possible == (level,):
                            selected[f"RIGHT_{level}_SAP"].append(observation)
                            counts["eligible"] += 1
                        elif len(possible) > 1:
                            ambiguous[f"RIGHT_{level}_SAP"].append(observation)
                            counts["ambiguous"] += 1
                        else:
                            counts["outside_or_checkpoint_mismatch"] += 1
        mapping = simple_map(selected, ambiguous, config)
        visibility, reference = {}, {}
        with np.load(folder / "oracle/visible_interface_labels.npz") as archive:
            for level in models:
                labels = archive[level]
                for class_id, kind in ((1, "TP"), (2, "SAP")):
                    name = f"RIGHT_{level}_{kind}"
                    points = []
                    for index in valid:
                        mask = labels[index] == class_id
                        if np.any(mask):
                            visibility[(name, index)] = cKDTree(np.argwhere(mask))
                            points.append(coords[index][mask])
                    if points:
                        reference[name] = cKDTree(np.concatenate(points))
        score = score_map(mapping, selected, visibility, reference,
                          protocol["evaluator"]["maximum_semantic_interface_distance_pixels"])
        sessions.append({
            "session": row["session"], "valid_frames": len(valid), "transverse_fit_frames": len(transverse),
            "fit_valid": fit is not None, "fit_max_error_index": fit.maximum_fit_residual_index if fit else None,
            "sap_routing_counts": dict(counts), "historical": row["arms"]["B"], "revised_sap": score,
            "revised_map": mapping,
        })
    totals = {
        key: sum(row["revised_sap"][key] for row in sessions)
        for key in ("correct_mapped_landmarks", "wrong_mapped_landmarks", "unresolved_landmarks", "complete_map")
    }
    write_json(args.output, {
        "scope": "Revised SAP with historical TP inputs. The recorded active trajectories lack the 50+4 TP partition.",
        "historical": baseline["comparison"]["B"], "revised_sap": totals, "sessions": sessions,
    })


if __name__ == "__main__":
    main()
