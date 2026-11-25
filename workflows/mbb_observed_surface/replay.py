"""Seal surface predictions before evaluating the recorded development cohort."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree

from run_simple_landmark_mapping import load_public, read_jsonl, sha256
from workflows.mbb_observed_surface.localize import assign_views, distance_both_ways, distinct_poses


MODES = {"Oracle-visible": "B0", "U-Net": "B1"}


def checked(entry):
    path = Path(entry["path"])
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"checksum drift: {path}")
    return path


def write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def predict_cohort(manifest, rules, program, output):
    actors = output / "predictions"
    actors.mkdir(parents=True)
    inputs = {}
    for session in manifest["sessions"]:
        seed = str(session["seed"])
        root = Path(session["path"])
        for name in ("public/frames.jsonl", "public/ultrasound.npz", "waypoints.json", "actor_result.json"):
            checked({"path": str(root / name), "sha256": session["files"][name]})
        records, _, valid, _, _, _ = load_public(root, rules["pixel_mm"])
        count = session["eligible_waypoint_count"]
        eligible = [i for i in valid if records[i]["waypoint_index"] < count]
        if eligible != list(range(count)) or {records[i]["stage1_session_id"] for i in eligible} != {"cf7_00"}:
            raise ValueError(f"public cohort or Stage 1 reference mismatch for {seed}")
        directory = actors / seed
        directory.mkdir()
        inputs[seed] = {}
        for mode, variant in MODES.items():
            cloud = checked(manifest["clouds"][seed][variant]["cloud.npz"])
            prediction = directory / f"{variant}.json"
            subprocess.run([sys.executable, "-m", "workflows.mbb_observed_surface.localize",
                            "--frames", str(root / "public/frames.jsonl"),
                            "--frame-count", str(count), "--rules", str(program),
                            "--cloud", str(cloud), "--output", str(prediction)], check=True)
            inputs[seed][mode] = {"path": str(prediction.relative_to(output)),
                                  "sha256": sha256(prediction), "cloud_sha256": sha256(cloud)}
    write(output / "prediction_seals.json", inputs)
    return inputs


def ras_to_actor(midpoint, affine, record, native):
    from nibabel.affines import apply_affine
    rotation = (np.asarray(record["actor_base_to_ee_rotation_columns"]) @
                np.asarray(native["human_to_ee_rotation_columns"]).T)
    origin = 1000 * (np.asarray(record["actor_base_to_ee_position_m"]) -
                     rotation @ native["human_to_ee_position_m"])
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    return (apply_affine(np.linalg.inv(affine), midpoint) * spacing) @ rotation.T + origin


def bidirectional_score(region, reference, gate):
    a, b = distance_both_ways(region, reference)
    def counts(d):
        return {"median_mm": float(np.median(d)), "p95_mm": float(np.quantile(d, .95)),
                "maximum_mm": float(d.max()), "within_technical_gate": int((d <= gate).sum()),
                "samples": len(d)}
    return {"prediction_to_reference": counts(a), "reference_to_prediction": counts(b)}


def historical_controls(frozen, manifest, seed):
    outcomes = {}
    control = frozen["four_landmark_controls"][seed]
    for mode, entry in control.items():
        old = json.loads(checked(entry).read_text())["score"]
        outcomes[mode] = {key: value["outcome"] for key, value in old["landmarks"].items()}
    old_gate = json.loads(checked(manifest["historical_acoustic_audit"]).read_text())
    record = next(row for row in old_gate["sessions"] if str(row["seed"]) == seed)
    return {"four_landmark_outcomes": outcomes,
            "old_endpoint_gate_frames": record["paired_image_supported_frames"],
            "endpoint_gate_qualifies": record["qualifies"],
            "old_target_output": "NO_TARGET_OUTPUT_FOR_THIS_SESSION"}


def surface_support(entry, frames, rules):
    with np.load(checked(entry), allow_pickle=False) as arrays:
        xyz, identities, sequences = (arrays[k] for k in ("xyz_mm", "identity", "frame_sequence"))
    distinct = distinct_poses(frames, rules)
    surfaces, support = {}, {}
    for label in ("TP", "SAP"):
        mask = identities == f"RIGHT_L5_{label}"
        views = np.unique(sequences[mask])
        surfaces[label] = xyz[mask]
        support[label] = {"points": int(mask.sum()), "source_frames": views.tolist(),
                          "distinct_frames": np.unique(assign_views(views, distinct)).tolist()}
    if all(len(points) for points in surfaces.values()):
        tp, sap = cKDTree(surfaces["TP"]), cKDTree(surfaces["SAP"])
        support["nearest_gap_mm"] = float(tp.query(surfaces["SAP"])[0].min())
        support["within_gap_pairs"] = int(tp.count_neighbors(sap, rules["max_gap_mm"]))
    return support


def draw_replay(output, manifest, sessions, references):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 14), layout="constrained")
    for row, session in enumerate(sessions):
        seed = session["seed"]
        reference = references[seed]
        for col, (mode, variant) in enumerate(MODES.items()):
            ax = fig.add_subplot(len(sessions), 2, row * 2 + col + 1, projection="3d")
            result = session["replay"][mode]["prediction"]
            cloud = manifest["clouds"][seed][variant]["cloud.npz"]["path"]
            with np.load(cloud, allow_pickle=False) as arrays:
                xyz, identity = arrays["xyz_mm"], arrays["identity"]
            for label, color in (("TP", "#13785c"), ("SAP", "#d77d19")):
                points = xyz[identity == f"RIGHT_L5_{label}"]
                ax.scatter(*points.T, s=8, color=color, alpha=.7, label=f"Observed {label}", depthshade=False)
            ax.scatter(*reference.T, marker="x", s=16, color="#992353", depthshade=False,
                       label="Historical proxy · evaluator")
            scene = result.get("observed_coverage", {})
            pair = scene.get("nearest_pair_observations")
            if pair:
                endpoints = np.array([pair[k]["points"][0]["xyz_mm"] for k in ("TP", "SAP")])
                ax.plot(*endpoints.T, color="#212121", marker="o", linewidth=1, markersize=3,
                        label="Closest observed pair · diagnostic")
            if result["candidate"]:
                ax.scatter(*np.asarray(result["candidate"]["observed_region_mm"]).T, c="#344eb8", s=18,
                           marker="^", label="Predicted surface region")
            ax.view_init(elev=27, azim=-60)
            center = reference.mean(0)
            ax.set(xlim=center[0] + np.array([-27, 27]), ylim=center[1] + np.array([-27, 27]),
                   zlim=center[2] + np.array([-27, 27]), xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)",
                   title=f"{seed} · {mode} · {result['status']}")
            if row == 0:
                ax.legend(loc="upper left", fontsize=7, frameon=False)
    fig.suptitle("Recorded right L5 surfaces in robot-base coordinates · separate evaluator overlay")
    fig.savefig(output / "surface_evidence.png", dpi=145)
    plt.close(fig)


def run(args):
    import nibabel as nib

    config_sha = sha256(args.protocol)
    if config_sha != args.protocol.with_suffix(".sha256").read_text().split()[0]:
        raise ValueError("replay protocol checksum mismatch")
    frozen = json.loads(args.protocol.read_text())
    if sha256(args.manifest) != frozen["source_manifest_sha256"]:
        raise ValueError("source manifest checksum mismatch")
    if sha256(args.rules) != frozen["actor_rules_sha256"]:
        raise ValueError("actor thresholds changed after freeze")
    manifest = json.loads(args.manifest.read_text())
    audit = checked({"path": str(args.audit), "sha256": frozen["anatomy_audit_sha256"]})
    for entry in manifest["models"].values():
        checked(entry)
    seeds = [s["seed"] for s in manifest["sessions"]]
    if seeds != frozen["development_seeds"] or set(seeds) & set(manifest["reserved_seeds"]):
        raise ValueError("development inputs differ from the frozen cohort or use a reserved seed")
    args.output.mkdir(parents=True, exist_ok=False)
    rules = json.loads(args.rules.read_text())
    seals = predict_cohort(manifest, rules, args.rules, args.output)
    midpoint_ras = json.loads(audit.read_text())["point sets"]["Historical midpoint"]["coordinates_ras_mm"]
    affine = nib.load(checked(manifest["target"]["segmentation"])).affine
    results, references = [], {}
    for session in manifest["sessions"]:
        seed = str(session["seed"])
        directory = Path(session["path"])
        frames = read_jsonl(directory / "public/frames.jsonl")[:session["eligible_waypoint_count"]]
        record = frames[0]
        key = "privileged/full_pose_replay.jsonl"
        native = read_jsonl(checked({"path": str(directory / key), "sha256": session["files"][key]}))[0]
        reference = ras_to_actor(midpoint_ras, affine, record, native)
        references[seed] = reference
        controls = historical_controls(frozen, manifest, seed)
        row = {"seed": seed, "frames": session["eligible_waypoint_count"], "historical": controls,
               "all_oracle_visible_before_actor_eligibility": surface_support(
                   manifest["clouds"][seed]["references"]["evaluator_visible.npz"], frames, rules),
               "replay": {}}
        for mode, entry in seals[seed].items():
            prediction = json.loads(checked({"path": str(args.output / entry["path"]),
                                             "sha256": entry["sha256"]}).read_text())
            comparison = None
            if prediction["candidate"]:
                region = prediction["candidate"]["observed_region_mm"]
                comparison = bidirectional_score(region, reference, frozen["legacy_diagnostic_tolerance_mm"])
            support = surface_support(manifest["clouds"][seed][MODES[mode]]["cloud.npz"], frames, rules)
            row["replay"][mode] = {"prediction": prediction, "legacy_comparison_only": comparison,
                                   "whole_surface_support": support,
                                   "reference_specific_correctness": "TARGET_LOCALIZATION_NOT_EVALUABLE"}
        results.append(row)
    aggregate = {}
    for mode in MODES:
        accepted = sum(s["replay"][mode]["prediction"]["candidate"] is not None for s in results)
        aggregate[mode] = {"candidate_sessions": accepted, "unresolved_sessions": len(results) - accepted,
                           "confirmed_correct": None, "confirmed_wrong": None,
                           "total_sessions": len(results)}
    report = {"protocol_sha256": config_sha, "reference_decision": frozen["reference_decision"],
              "seed_policy": "opened development replay, no reserved seeds",
              "sessions": results, "aggregate": aggregate, "fresh_confirmation": "NOT_RUN",
              "readiness": "FAIL", "readiness_reason": "anatomical reference remains unresolved"}
    write(args.output / "results.json", report)
    draw_replay(args.output, manifest, results, references)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "audit", "rules", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
