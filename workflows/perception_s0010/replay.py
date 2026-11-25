"""Compare perception on the frozen 102-waypoint and historical acquisitions."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml
from scipy.ndimage import distance_transform_edt, label
from scipy.spatial import cKDTree

from run_simple_landmark_mapping import load_evaluator, load_public, oracle_observations, read_jsonl, score_map, sha256, write_json
from two_stage_navigation.incremental_landmark_map import IncrementalLandmarkMapConfig, LANDMARK_IDENTITIES
from two_stage_navigation.simple_landmark_mapping import (
    CLASSES, LEVELS, Stage1Region, extract_observations, project_image, sap_level_at, simple_map, transverse_cranial_coordinate,
)
from two_stage_navigation.sonoanatomy_labels import LEVEL_SOURCE_LABEL
from two_stage_navigation.sonoanatomy_segmentation import load_segmentation_checkpoint, normalized_ultrasound

STRUCTURE = np.ones((3, 3), dtype=bool)
CONVERSION = Path("source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml")


def models_for(model_specs):
    models = {}
    for level, spec in model_specs.items():
        path = Path(spec["path"])
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"checkpoint hash mismatch {path}")
        model, bundle = load_segmentation_checkpoint(path, expected_level=level)
        models[level] = model, spec.get("thresholds", bundle["thresholds"])
    return models


@torch.no_grad()
def infer(model, images, device="cpu", batch_size=12):
    model.to(device)
    for group in torch.from_numpy(np.asarray(images, dtype=np.float32)).split(batch_size):
        yield from torch.softmax(model(normalized_ultrasound(group.to(device))[:, None]), dim=1).cpu().numpy()


def context(session, protocol, evaluator, kind):
    actor = json.loads((session / "actor_result.json").read_text())
    seal = session / "actor_result.json.sha256"
    if seal.exists() and sha256(session / "actor_result.json") != seal.read_text().split()[0]:
        raise ValueError(f"actor seal mismatch {session}")
    frames, images, valid, coords, dropped, _ = load_public(
        session, protocol["geometry"]["ultrasound_pixel_resolution_mm"],
    )
    if kind == "medium":
        declared = json.loads((session / "waypoints.json").read_text())[:102]
        design_hash = hashlib.sha256((json.dumps(declared, indent=2) + "\n").encode()).hexdigest()
        assert design_hash == "b325d3c5df65ca17bda3584374d3fd6d784c246ae931b7d63c5629fb45d653e9"
        valid = [i for i in valid if frames[i]["waypoint_index"] < 102]
    oracle, labels, visible, ref, projection, _ = oracle_observations(
        session, frames, images, valid, coords, evaluator, protocol,
    )
    raw = np.load(session / "privileged/raw_labels.npz")["raw_labels"]
    fit = transverse_cranial_coordinate(
        [frames[i] for i in valid], protocol["geometry"]["label_resolution_mm"],
        protocol["sonoanatomy_scan"]["translation_step_index"] / 2,
    )
    return {"session": str(session), "actor": actor, "frames": frames, "images": images, "valid": valid,
            "coords": coords, "region": Stage1Region(**actor["stage1_region"]), "fit": fit, "oracle": oracle,
            "visible": visible, "ref": ref, "labels": labels, "raw": raw, "projection": projection,
            "dropped": dropped, "kind": kind}


def route(row, frame, level, ctx, tp_all):
    if row.semantic_class == "SAP":
        _, candidates = sap_level_at(row.coordinate_mm, ctx["region"], ctx["fit"])
    elif tp_all or frame["tp_eligible_for_mapping"]:
        candidates = frame["level_hypotheses"]
    else:
        candidates = ()
    return len(candidates) if level in candidates else 0


def bone_source(raw, pixel, level, observed_labels):
    r, c = pixel
    own = LEVEL_SOURCE_LABEL[level]
    code = int(raw[r, c])
    if observed_labels[r, c] == 255:
        return "uncertain_or_cortical_track_unsupported", code
    if code in LEVEL_SOURCE_LABEL.values() and code != own:
        return "other_vertebral_level", code
    if observed_labels[r, c] == 3 or code == own:
        return "non_target_bone", code
    return "background_or_boundary", code


def diagnose_probability(prob, rules, truth, predicted, rows, routed, frame, ctx, level, gate):
    decisions = []
    winner = prob.argmax(0)
    for code, semantic in ((1, "TP"), (2, "SAP")):
        actual = truth == code
        true_pixels = int(actual.sum())
        predicted_pixels = int((predicted == code).sum())
        candidates = [r for r in rows if r.semantic_class == semantic]
        route_sizes = [routed[r] for r in candidates]
        eligible = [r for r in candidates if routed[r] == 1]
        identity = f"RIGHT_{level}_{semantic}"
        tree = ctx["visible"].get((identity, frame["frame_sequence"]))
        distances = [None if tree is None else float(tree.query((r.row, r.column))[0]) for r in eligible]
        closest = min((d for d in distances if d is not None), default=None)
        local_best = None
        reason = "true_negative" if not true_pixels else "correct_interface"
        if not true_pixels and eligible:
            reason = "no_visible_target_accepted"
        if true_pixels:
            neighborhood = distance_transform_edt(~actual) <= gate
            local_best = float(prob[code][neighborhood].max())
            if closest is None or closest > gate:
                rule = rules[semantic]
                if eligible:
                    reason = "representative_off_interface"
                elif candidates:
                    reason = "level_routing_or_ambiguous"
                elif not ((winner == code) & neighborhood).any():
                    competing = Counter(winner[actual].tolist()).most_common(1)[0][0]
                    reason = "competing_class_" + CLASSES[competing]
                elif local_best < rule["threshold"]:
                    reason = "below_probability_threshold"
                else:
                    reason = "component_below_minimum"
        fp = [{"row": r.row, "column": r.column, "confidence": r.confidence, "distance_px": d,
               "source": bone_source(ctx["raw"][frame["frame_sequence"]], (r.row, r.column), level, truth)}
              for r, d in zip(eligible, distances) if d is None or d > gate]
        decisions.append({"identity": identity, "frame": frame["frame_sequence"], "waypoint": frame["waypoint_index"],
                          "phase": frame["phase"], "truth_pixels": true_pixels, "prediction_pixels": predicted_pixels,
                          "extremely_sparse_truth": 0 < true_pixels <= 2, "reason": reason,
                          "peak_near_interface": local_best, "closest_accepted_px": closest,
                          "all_components": len(candidates), "routing": route_sizes,
                          "false_components": fp})
    return decisions


def replay(ctx, protocol, models, *, tp_all, diagnose=False, device="cpu", probabilities=None):
    observations, ambiguity = defaultdict(list), defaultdict(list)
    diagnostic = []
    masks = {lvl: np.zeros(ctx["images"].shape, np.uint8) for lvl in LEVELS}
    valid = ctx["valid"]
    gate = protocol["evaluator"]["maximum_semantic_interface_distance_pixels"]
    for level, (model, rules) in models.items():
        predicted = (probabilities[level] if probabilities is not None else
                     infer(model, ctx["images"][valid], device, batch_size=1))
        for sequence, prob in zip(valid, predicted):
            frame = ctx["frames"][sequence]
            rows, mask = extract_observations(prob.argmax(0), prob, rules, frame, ctx["coords"][sequence])
            masks[level][sequence] = mask
            routed = {}
            for row in rows:
                n = route(row, frame, level, ctx, tp_all)
                routed[row] = n
                if n:
                    destination = observations if n == 1 else ambiguity
                    destination[f"RIGHT_{level}_{row.semantic_class}"].append(row)
            if diagnose:
                diagnostic.extend(diagnose_probability(prob, rules, ctx["labels"][level][sequence], mask,
                                                       rows, routed, frame, ctx, level, gate))
    config = IncrementalLandmarkMapConfig(**protocol["landmark_map"])
    mapping = simple_map(observations, ambiguity, config)
    score = score_map(mapping, observations, ctx["visible"], ctx["ref"], gate)
    total = 102 if ctx["kind"] == "medium" else 54
    integrity = len(valid) == total and ctx["projection"]["passes"]
    return {"mapping": mapping, "score": score, "valid": len(valid), "integrity": integrity,
            "complete": integrity and score["complete_map"], "diagnostics": diagnostic,
            "observations": {k: [asdict(r) for r in v] for k, v in observations.items()},
            "ambiguous_observations": {k: [asdict(r) for r in v] for k, v in ambiguity.items()}}, masks


def oracle_map(ctx, protocol, tp_all, *, strict_routing=False):
    config = IncrementalLandmarkMapConfig(**protocol["landmark_map"])
    obs, ambiguous = defaultdict(list), defaultdict(list)
    for identity, rows in ctx["oracle"].items():
        level = identity.split("_")[1]
        for row in rows:
            frame = ctx["frames"][row.frame_sequence]
            if strict_routing:
                candidates = route(row, frame, level, ctx, tp_all)
            else:
                candidates = int(row.semantic_class == "SAP" or tp_all or frame["tp_eligible_for_mapping"])
            if candidates:
                (obs if candidates == 1 else ambiguous)[identity].append(row)
    result = simple_map(obs, ambiguous, config)
    score = score_map(result, obs, ctx["visible"], ctx["ref"],
                      protocol["evaluator"]["maximum_semantic_interface_distance_pixels"])
    return {"mapping": result, "score": score,
            "complete": score["complete_map"] and len(ctx["valid"]) == 102 and ctx["projection"]["passes"]}


def visualise_diagnostics(ctx, masks, records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 4, figsize=(10, 10.5), layout="constrained", squeeze=False)
    for key, axes in zip(LANDMARK_IDENTITIES, axs):
        level = key.split("_")[1]
        rows = [r for r in records if r["identity"] == key]
        wanted = []
        for predicate in (lambda x: x["truth_pixels"] > 0 and x["reason"] != "correct_interface",
                          lambda x: x["reason"] == "no_visible_target_accepted",
                          lambda x: x["reason"] == "representative_off_interface",
                          lambda x: x["reason"] == "correct_interface"):
            wanted.append(next((x for x in rows if predicate(x)), None))
        for ax, row in zip(axes, wanted):
            ax.axis("off")
            if row is None:
                continue
            i = row["frame"]
            image = ctx["images"][i]
            ax.imshow(image, cmap="gray", vmin=0, vmax=np.quantile(image, .99))
            cat = CLASSES.index(key.split("_")[-1])
            for overlay, color in ((ctx["labels"][level][i], "#77e3ff"), (masks[level][i], "#ff9b56")):
                if (overlay == cat).any():
                    ax.contour(overlay == cat, levels=[.5], colors=[color], linewidths=1)
            for r in row["false_components"]:
                ax.plot(r["column"], r["row"], "x", color="orange", markersize=5)
            ax.set_title(f"{key[6:].replace('_',' ')} | WP {row['waypoint']}\n"
                         f"{row['reason'].replace('_',' ')}", fontsize=8)
    fig.suptitle("Actor and oracle contours | orange prediction · cyan oracle")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol", "profile", "previous", "historical", "output"):
        parser.add_argument("--"+key, required=True, type=Path)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    campaign = yaml.safe_load(args.profile.read_text())
    args.output.mkdir(exist_ok=False, parents=True)
    evaluator = load_evaluator(protocol, CONVERSION)
    old = protocol["models"]["sonoanatomy"]
    e3 = json.loads((args.previous/"pilot-recovery-52713119/metrics/training_result.json").read_text())
    arms = {"original": old, "E3": dict(old, L5={"path": e3["checkpoint"], "sha256": e3["checkpoint_sha256"]})}
    if args.checkpoints:
        arms.update(json.loads(args.checkpoints.read_text()))
    loaded = {k: models_for(v) for k, v in arms.items()}
    previous = args.previous/"acquisition-52710240"
    sources = [(previous/f"E1_{seed}", "medium") for seed in campaign["development_seeds"][:2]]
    if args.development:
        sources += [(args.development/str(seed), "medium") for seed in campaign["development_seeds"][2:]]
    if not args.diagnose:
        sources += [(p, "historical") for p in sorted(args.historical.glob("session_*"))]
    report = {"cases": [], "protocol_sha256": sha256(args.protocol), "checkpoint_specs": arms}
    for session, kind in sources:
        if not (session/"actor_result.json").exists():
            report["cases"].append({"source": str(session), "kind": kind, "failed": True})
            continue
        ctx = context(session, protocol, evaluator, kind)
        seed = ctx["actor"]["seed"]
        policies = [("all_tp", True)] if kind == "medium" else [("historical_tp", False)]
        if kind == "medium":
            policies.append(("original_50_tp", False))
        entry = {"source": str(session), "seed": seed, "kind": kind,
                 "projection": ctx["projection"], "policies": {}}
        for named, all_tp in policies:
            values = {}
            for arm, models in loaded.items():
                if named == "original_50_tp" and arm != "original":
                    continue
                collect = args.diagnose and all_tp and arm == "original"
                result, masks = replay(ctx, protocol, models, tp_all=all_tp, diagnose=collect, device=args.device)
                if collect:
                    visualise_diagnostics(ctx, masks, result["diagnostics"], args.output/f"diagnosis-{seed}.png")
                values[arm] = result
            values["oracle"] = oracle_map(ctx, protocol, all_tp)
            if named == "all_tp":
                values["oracle_routed"] = oracle_map(ctx, protocol, all_tp, strict_routing=True)
            entry["policies"][named] = values
        report["cases"].append(entry)
        print("Evaluated", kind, seed, flush=True)
        write_json(args.output/"mapping.json", report)


if __name__ == "__main__":
    main()
