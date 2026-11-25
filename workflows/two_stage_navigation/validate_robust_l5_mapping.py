"""Score L5 mapping on the same held-out rendered frames for both checkpoints."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import yaml

from run_simple_landmark_mapping import read_jsonl, score_map, sha256, write_json
from two_stage_navigation.incremental_landmark_map import IncrementalLandmarkMapConfig
from two_stage_navigation.simple_landmark_mapping import (
    Stage1Region, extract_observations, project_image, sap_level_at, simple_map, transverse_cranial_coordinate,
)
from two_stage_navigation.sonoanatomy_segmentation import load_segmentation_checkpoint, segmentation_probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "training", "protocol", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    config = IncrementalLandmarkMapConfig(**protocol["landmark_map"])
    gate = protocol["evaluator"]["maximum_semantic_interface_distance_pixels"]
    report = json.loads((args.training / "training_result.json").read_text())
    original = protocol["models"]["sonoanatomy"]["L5"]
    alternatives = {"original": original, "fine_tuned": {
        "path": report["checkpoint"], "sha256": report["checkpoint_sha256"],
    }}
    models = {}
    for name, model in alternatives.items():
        if sha256(Path(model["path"])) != model["sha256"]:
            raise ValueError(f"checkpoint mismatch for {name}")
        models[name] = load_segmentation_checkpoint(model["path"], expected_level="L5")
    with np.load(args.dataset / "dataset.npz") as a:
        images, labels = a["ultrasound"], a["labels"]
    data = json.loads((args.dataset / "split_manifest.json").read_text())
    groups = defaultdict(list)
    for row in data["records"]:
        if row["split"] == "validation":
            groups[(Path(row["source_session"]), row["speckle_field"])].append(row)

    analysis = []
    for (session, field), group in groups.items():
        actor = json.loads((session / "actor_result.json").read_text())
        region = Stage1Region(**actor["stage1_region"])
        source_frames = read_jsonl(session / "public/frames.jsonl")
        fit = transverse_cranial_coordinate(
            source_frames, protocol["geometry"]["label_resolution_mm"],
            protocol["sonoanatomy_scan"]["translation_step_index"] / 2,
        )
        observed = {}
        clouds = defaultdict(list)
        eligible = defaultdict(lambda: defaultdict(list))
        uncertain = defaultdict(lambda: defaultdict(list))
        for row in group:
            index, sequence = row["archive_index"], row["source_frame_sequence"]
            f = dict(source_frames[sequence])
            f["frame_sha256"] = row["image_sha256"]
            f["acquisition_id"] += f":render:{field}"
            world = project_image(f, images[index].shape, protocol["geometry"]["ultrasound_pixel_resolution_mm"])
            for value, label in ((1, "TP"), (2, "SAP")):
                subset = labels[index] == value
                if subset.any():
                    identity = f"RIGHT_L5_{label}"
                    observed[(identity, sequence)] = cKDTree(np.argwhere(subset))
                    clouds[identity].append(world[subset])
            for name, (model, bundle) in models.items():
                probability = segmentation_probabilities(model, images[index:index+1])[0]
                proposals, _ = extract_observations(probability.argmax(0), probability, bundle["thresholds"], f, world)
                for proposal in proposals:
                    if proposal.semantic_class == "SAP":
                        _, levels = sap_level_at(proposal.coordinate_mm, region, fit)
                    elif f["tp_eligible_for_mapping"]:
                        levels = f["level_hypotheses"]
                    else:
                        continue
                    if "L5" in levels:
                        (eligible if len(levels) == 1 else uncertain)[name][f"RIGHT_L5_{proposal.semantic_class}"].append(proposal)
        reference = {name: cKDTree(np.concatenate(parts)) for name, parts in clouds.items()}
        outcomes = {}
        for name in models:
            mapping = simple_map(eligible[name], uncertain[name], config)
            score = score_map(mapping, eligible[name], observed, reference, gate)
            outcomes[name] = {k: v for k, v in score["landmarks"].items() if k.startswith("RIGHT_L5_")}
        analysis.append({"source_session": str(session), "field": field, "frames": len(group),
                         "visible_frames": {k: sum(key[0] == k for key in observed) for k in reference},
                         "maps": outcomes})
    write_json(args.output, {
        "scope": "L5 only, pose-neighborhood and speckle-held-out offline rendered acquisition subsets",
        "archive_sha256": hashlib.sha256((args.dataset / "dataset.npz").read_bytes()).hexdigest(),
        "groups": analysis,
    })


if __name__ == "__main__":
    main()
