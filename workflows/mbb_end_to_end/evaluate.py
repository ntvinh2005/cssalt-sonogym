"""Run and seal matched surface predictions before opening the anatomical reference."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist


ARMS = {"E1": "oracle", "E2": "learned"}
ANCHORS = ("RIGHT_L4_TP", "RIGHT_L4_SAP", "RIGHT_L5_TP", "RIGHT_L5_SAP")
DISPLAY = {
    "RIGHT_L5_TP": "#147d68", "RIGHT_L5_SAP": "#d87918",
    "RIGHT_L4_TP": "#3969a5", "RIGHT_L4_SAP": "#bb5c81",
    "OTHER_BONE": "#b6b2c2", "UNCERTAIN_BONE": "#685d92",
}


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def checked(entry):
    path = Path(entry["path"])
    if digest(path) != entry["sha256"]:
        raise ValueError(f"invalid digest for {path}")
    return path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def jsonl_first(path):
    with path.open() as stream:
        return json.loads(stream.readline())


def array_file(path):
    with np.load(path, allow_pickle=False) as cloud:
        return dict(cloud)


def load_protocol(path):
    sha = digest(path)
    if path.with_suffix(".sha256").read_text().split()[0] != sha:
        raise ValueError("localization evaluation rules have changed")
    return read(path), sha


def invoke_actor(cloud, frames, output):
    subprocess.run([
        sys.executable, "-m", "workflows.mbb_end_to_end.localize",
        "--cloud", str(cloud), "--frames", str(frames), "--output", str(output),
    ], check=True)


def seal_predictions(source, cloud_dir, output, expected_seeds):
    if [s["seed"] for s in source["sessions"]] != expected_seeds:
        raise ValueError("cohort does not match frozen seed ordering")
    if set(expected_seeds) & set(source["reserved_seeds"]):
        raise ValueError("reserved-seed overlap")
    seals = {}
    for session in source["sessions"]:
        seed = str(session["seed"])
        folder = output / seed
        folder.mkdir()
        frames = checked({"path": str(Path(session["path"]) / "public/frames.jsonl"),
                          "sha256": session["files"]["public/frames.jsonl"]})
        seals[seed] = {}
        for arm, name in ARMS.items():
            cloud = cloud_dir / seed / (name + ".npz")
            values = array_file(cloud)
            sequence = values["frame_sequence"]
            if np.any((sequence < 0) | (sequence >= session["eligible_waypoint_count"])):
                raise ValueError("point outside the matched 102-waypoint cohort")
            mask = np.isin(values["identity"], ANCHORS)
            anchors = folder / f"{arm}-anchors-only.npz"
            np.savez_compressed(anchors, **{
                key: value[mask] if value.shape[:1] == mask.shape else value
                for key, value in values.items()
            })
            for variant, point_file in ((arm, cloud), (f"{arm}_anchors_only", anchors)):
                path = folder / f"{variant}.json"
                invoke_actor(point_file, frames, path)
                seals[seed][variant] = {
                    "result": {"path": str(path), "sha256": digest(path)},
                    "input_cloud": {"path": str(point_file), "sha256": digest(point_file)},
                    "public_frames_sha256": digest(frames),
                }
    write(output / "actor_prediction_seals.json", seals)
    return seals


def ras_to_base(points, affine, session):
    import nibabel as nib

    root = Path(session["path"])
    f = jsonl_first(checked({"path": str(root / "public/frames.jsonl"),
                             "sha256": session["files"]["public/frames.jsonl"]}))
    pose_file = "privileged/full_pose_replay.jsonl"
    native = jsonl_first(checked({"path": str(root / pose_file), "sha256": session["files"][pose_file]}))
    rotation = (np.asarray(f["actor_base_to_ee_rotation_columns"]) @
                np.asarray(native["human_to_ee_rotation_columns"]).T)
    translated = 1000 * (np.asarray(f["actor_base_to_ee_position_m"]) -
                         rotation @ native["human_to_ee_position_m"])
    voxels = nib.affines.apply_affine(np.linalg.inv(affine), points)
    return (voxels * nib.affines.voxel_sizes(affine)) @ rotation.T + translated


def reference_data(directory):
    path = directory / "reference.json"
    receipt = directory / "checksums.sha256"
    # Reference bytes must have been sealed before the first replay prediction.
    members = {}
    for row in receipt.read_text().splitlines():
        sha, name = row.split(maxsplit=1)
        member = Path(name.strip().lstrip("*"))
        if member.is_absolute() or ".." in member.parts:
            raise ValueError("reference receipt includes an external file")
        members[member.as_posix()] = {"path": str(directory / member), "sha256": sha}
    checked(members["reference.json"])
    info = read(path)
    if info["status"] in ("NOT_EVALUABLE", "TARGET_LOCALIZATION_NOT_EVALUABLE"):
        return info, {}, members
    regions = array_file(checked(members["region.npz"]))
    return info, {k: v for k, v in regions.items() if k == "points_ras_mm" or k.startswith("hypothesis_")}, members


def nearest(a, b):
    return cKDTree(b).query(a)[0] if len(b) else np.full(len(a), np.inf)


def stats(values, gate_mm):
    return {"sample_count": len(values), "median_mm": float(np.median(values)),
            "p95_mm": float(np.quantile(values, .95)), "maximum_mm": float(np.max(values)),
            "within_tolerance": int((values <= gate_mm + 1e-9).sum())}


def candidate_points(candidate):
    if not candidate:
        return None, 0, 0
    observed = np.asarray(candidate["observed_region_mm"], dtype=float).reshape(-1, 3)
    inferred = np.asarray(candidate.get("inferred_region_mm", []), dtype=float).reshape(-1, 3)
    points = np.unique(np.concatenate((observed, inferred)), axis=0)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("invalid candidate coordinates")
    return points, len(observed), len(inferred)


def score(candidate, references, historical, gates):
    points, observed_count, inferred_count = candidate_points(candidate)
    if points is None:
        return {"status": "UNRESOLVED", "anatomical_correct": None,
                "diameter_mm": None, "observed_samples": 0, "inferred_samples": 0,
                "reference_scores": {}, "historical_diagnostic_only": None}
    diameter = float(pdist(points).max()) if len(points) > 1 else 0.
    def compare(reference):
        distance_gate = gates["max_p95_mm"]
        a = stats(nearest(points, reference), distance_gate)
        b = stats(nearest(reference, points), distance_gate)
        return {"candidate_to_reference": a, "reference_to_candidate": b,
                "passes": max(a["p95_mm"], b["p95_mm"]) <= distance_gate and diameter <= gates["max_diameter_mm"]}
    results = {name: compare(region) for name, region in references.items()}
    old_distances = compare(historical)
    del old_distances["passes"]
    correctness = all(result["passes"] for result in results.values()) if results else None
    return {"status": "SCORED" if results else "TARGET_LOCALIZATION_NOT_EVALUABLE",
            "anatomical_correct": correctness,
            "diameter_mm": diameter, "observed_samples": observed_count, "inferred_samples": inferred_count,
            "reference_scores": results, "historical_diagnostic_only": old_distances}


def count_coverage(points, xyz, gate_mm):
    return {"covered": int((nearest(points, xyz) <= gate_mm + 1e-9).sum()), "total": len(points)}


def cortex_diagnostic(cloud, cortex, owners, gate_mm):
    categories = np.isin(cloud["identity"], ("OTHER_BONE", "UNCERTAIN_BONE"))
    levels = cloud["level_hint"]
    rows = []
    tree = cKDTree(cloud["xyz_mm"])
    for point, owner in zip(cortex, owners):
        within = np.asarray(tree.query_ball_point(point, gate_mm + 1e-9), dtype=int)
        rows.append({
            "ct_face_owner_subregion": int(owner),
            "observed_samples": len(within),
            "samples_routed_l5": int((levels[within] == "L5").sum()),
            "other_or_uncertain_samples": int(categories[within].sum()),
            "any_bone_frames": np.unique(cloud["frame_sequence"][within]).tolist(),
        })
    return {
        "sample_distances": [
            float(value) if np.isfinite(value) else None for value in nearest(cortex, cloud["xyz_mm"])
        ],
        "matched_faces": sum(r["observed_samples"] > 0 for r in rows),
        "matched_faces_routed_l5": sum(r["samples_routed_l5"] > 0 for r in rows),
        "matched_faces_with_other_or_uncertain": sum(r["other_or_uncertain_samples"] > 0 for r in rows),
        "total_faces": len(rows), "faces": rows,
    }


def visible_coverage(cloud, oracle, historical, references, rule):
    xyz, identities = cloud["xyz_mm"], cloud["identity"]
    name, count = np.unique(identities, return_counts=True)
    full = {"classes": {k: int(n) for k, n in zip(name.tolist(), count.tolist())},
            "historical_roi_diagnostic_only": {}, "anatomical_reference": {}}
    rad = rule["observed_coverage_distance_mm"]
    for radius in rule["historical_diagnostic_roi_radii_mm"]:
        seen = nearest(oracle["xyz_mm"], historical) <= radius
        cloud_local = nearest(xyz, historical) <= radius
        rows = {"all_visible": count_coverage(oracle["xyz_mm"][seen], xyz[cloud_local], rad)}
        for label in ("OTHER_BONE", "UNCERTAIN_BONE", "RIGHT_L5_TP", "RIGHT_L5_SAP"):
            relevant = seen & (oracle["identity"] == label)
            sample = cloud_local & (label == identities)
            rows[label] = {
                "same_label": count_coverage(oracle["xyz_mm"][relevant], xyz[sample], rad),
                "any_bone_label": count_coverage(oracle["xyz_mm"][relevant], xyz[cloud_local], rad),
                "observed_points": int(sample.sum()),
                "unique_frame_sequences": np.unique(cloud["frame_sequence"][sample]).tolist(),
            }
        full["historical_roi_diagnostic_only"][str(radius)] = rows
    for name, ref in references.items():
        full["anatomical_reference"][name] = count_coverage(ref, xyz, rad)
    return full


def evaluate(args):
    import nibabel as nib

    protocol, protocol_sha = load_protocol(args.protocol)
    source = read(checked(protocol["source_manifest"]))
    args.output.mkdir(parents=True, exist_ok=False)
    reference_receipt = args.reference / "checksums.sha256"
    reference_sha_before = digest(reference_receipt)
    write(args.output / "input_freeze.json",
          {"protocol_sha256": protocol_sha, "reference_receipt_sha256": reference_sha_before,
           "observable_recording_ids": protocol["development_seeds"]})
    seals = seal_predictions(source, args.reconstruction, args.output, protocol["development_seeds"])
    if digest(reference_receipt) != reference_sha_before:
        raise ValueError("reference seal changed during prediction")
    ref_info, ref_regions, ref_members = reference_data(args.reference)
    anatomical_hypotheses = array_file(checked(ref_members["anatomy_hypotheses.npz"]))
    cortical_trace = anatomical_hypotheses["diagnostic_cortex_connection_ras_mm"]
    owner_subregions = anatomical_hypotheses["diagnostic_cortex_connection_owner_labels"]
    archived = read(checked(protocol["historical_replay"]))
    history = {str(s["seed"]): s for s in archived["sessions"]}
    original = read(checked(protocol["historical_target_for_diagnostics"]))
    legacy_ras = original["point sets"]["Historical midpoint"]["coordinates_ras_mm"]
    affine = nib.load(checked(source["target"]["segmentation"])).affine
    gate = protocol["reference_technical_gate"]
    rows = []
    for session in source["sessions"]:
        seed = str(session["seed"])
        references = {name: ras_to_base(points, affine, session) for name, points in ref_regions.items()}
        trace = ras_to_base(cortical_trace, affine, session)
        legacy = ras_to_base(legacy_ras, affine, session)
        oracle = array_file(checked(seals[seed]["E1"]["input_cloud"]))
        old = history[seed]
        row = {"seed": seed, "valid_frames": session["eligible_waypoint_count"],
               "evaluator_geometry_for_review": {
                   "historical_midpoint_mm": legacy.tolist(),
                   "diagnostic_cortical_trace_mm": trace.tolist(),
                   "anatomical_regions_mm": {name: points.tolist() for name, points in references.items()},
               },
               "E0": {name: {
                   "status": arm["prediction"]["status"],
                   "reason": arm["prediction"]["reason"],
                   "nearest_tp_sap_mm": arm["prediction"]["observed_coverage"]["nearest_gap_mm"],
               } for name, arm in old["replay"].items()},
               "arms": {}}
        for arm, entry in seals[seed].items():
            prediction = read(checked(entry["result"]))
            cloud = array_file(checked(entry["input_cloud"]))
            old_trace = None
            if arm in ARMS:
                variant = "B0" if arm == "E1" else "B1"
                old_cloud = array_file(checked(source["clouds"][seed][variant]["cloud.npz"]))
                old_trace = count_coverage(trace, old_cloud["xyz_mm"], gate["observed_coverage_distance_mm"])
            row["arms"][arm] = {
                "prediction": prediction,
                "evaluation": score(prediction.get("candidate"), references, legacy, gate),
                "coverage": visible_coverage(cloud, oracle, legacy, references, gate),
                "diagnostic_cortical_trace_only": cortex_diagnostic(
                    cloud, trace, owner_subregions, gate["observed_coverage_distance_mm"],
                ),
                "original_tp_sap_trace_diagnostic": old_trace,
            }
        rows.append(row)
    aggregate = {}
    for arm in ARMS:
        results = [s["arms"][arm] for s in rows]
        candidate_count = sum(r["prediction"].get("candidate") is not None for r in results)
        correct = [r["evaluation"]["anatomical_correct"] for r in results]
        aggregate[arm] = {"candidates": candidate_count, "unresolved": len(rows) - candidate_count,
                          "correct": sum(v is True for v in correct) if ref_regions else None,
                          "wrong": sum(v is False for v in correct) if ref_regions else None}
    write(args.output / "results.json", {
        "reference": ref_info, "reference_sha256": ref_members["reference.json"]["sha256"],
        "anatomical_target_evaluable": bool(ref_regions), "technical_gate": gate,
        "aggregate": aggregate, "sessions": rows,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "reconstruction", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
