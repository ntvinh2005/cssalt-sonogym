"""Measure local observability and identifiability on frozen ultrasound reconstructions."""

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import resource
import time

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
import yaml

from run_s0010_l4_medial_branch_target import load_target_geometry
from run_simple_landmark_mapping import read_jsonl, sha256, write_json
from workflows.reconstruction_s0010.dense import IDENTITIES, distinct_pose_matrix, nearest
from workflows.mbb_target_identifiability.geometry import (
    coverage, crop_geometry, first_bone_voxels, hidden_sap_variants, leave_views, observed_coverage,
    prepare_cloud, reference_coverage, score_region, subset, transform, view_support,
)

METHODS = ("B0", "B1", "B2", "B3")


def checked(entry):
    path = Path(entry["path"])
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"checksum drift: {path}")
    return path


def load_array(entry):
    with np.load(checked(entry)) as f:
        return dict(f)


def check_manifest(path):
    if sha256(path) != path.with_suffix(".json.sha256").read_text().split()[0]:
        raise ValueError("manifest checksum mismatch")
    data = json.loads(path.read_text())
    seeds = {str(item["seed"]) for item in data["sessions"]}
    assert len(seeds) == 4 and seeds == data["clouds"].keys()
    assert seeds.isdisjoint({str(seed) for seed in data["reserved_seeds"]})
    assert data["local_evaluation"]["frozen_radii_mm"] == [5, 10, 20]
    checked(data["historical_sweep"])
    checked(data["historical_pose_core_audit"])
    checked(data["historical_acoustic_audit"])
    for model in data["models"].values():
        checked(model)
    return data


def endpoint_measurement(source, frame_rows, surface, bone_ids):
    root = Path(source["path"])
    endpoint_counts = defaultdict(dict)
    protected = np.zeros(surface.sap_core.shape, dtype=bool)
    masks = {"sap": surface.sap_core, "tp": surface.tp_core}
    seen_voxels = {k: set() for k in masks}
    names = ("privileged/raw_labels.npz", "privileged/pixel_to_volume_coordinates.npz")
    for name in (*names, "privileged/full_pose_replay.jsonl"):
        checked({"path": str(root / name), "sha256": source["files"][name]})
    with np.load(root / names[0]) as raw_file, np.load(root / names[1]) as coords_file:
        raw = raw_file["raw_labels"]
        coords = coords_file["human_img_coords"]
        for frame in frame_rows:
            i = frame["frame_sequence"]
            first = first_bone_voxels(coords[i], raw[i], bone_ids, protected.shape)
            protected[tuple(first.T)] = True
            for name, mask in masks.items():
                end = first[mask[tuple(first.T)]]
                if len(end):
                    endpoint_counts[name][i] = len(end)
                    seen_voxels[name].update(map(tuple, end.tolist()))
    return protected, {
        name: {"first_visible_frames": len(endpoint_counts[name]),
               "first_visible_pixel_hits": sum(endpoint_counts[name].values()),
               "first_visible_voxels": len(seen_voxels[name]), "core_ct_voxels": int(mask.sum()),
               "frame_sequences": sorted(endpoint_counts[name])}
        for name, mask in masks.items()
    }


def report_session(source, artifact, target, bone_ids, protocol, output):
    projection = json.loads(checked(artifact["references"]["frames.json"]).read_text())
    frame_rows = projection["frames"]
    sequences = [row["frame_sequence"] for row in frame_rows]
    if sequences != list(range(102)) or not projection["projection"]["passes"]:
        raise ValueError("unexpected frame order or calibration parity")
    replay = read_jsonl(Path(source["path"]) / "privileged/full_pose_replay.jsonl")
    seed = str(source["seed"])
    spacing = protocol["geometry"]["label_resolution_mm"]
    project = transform(frame_rows[0], replay[0], spacing)
    ct_file, sap, tp, contract = target
    world_mm = contract.midpoint_world_mm
    midpoint_index = nib.affines.apply_affine(np.linalg.inv(ct_file.affine), world_mm)
    midpoint = project(midpoint_index)
    tree = cKDTree(midpoint)
    protected, endpoint_hits = endpoint_measurement(source, frame_rows, contract, bone_ids)
    surface = {k: project(np.argwhere(m)) for k,m in {"SAP": contract.sap_core, "TP": contract.tp_core}.items()}
    boundary = load_array(artifact["references"]["evaluator_ct.npz"])
    boundary = {identity: boundary[identity] for identity in IDENTITIES}
    visible = prepare_cloud(load_array(artifact["references"]["evaluator_visible.npz"]))
    cloud_artifacts = {m: load_array(artifact[m]["cloud.npz"]) for m in METHODS}
    clouds = {m: prepare_cloud(d) for m, d in cloud_artifacts.items()}
    oracle = clouds["B0"]
    sparse = {m: prepare_cloud(load_array(item)) for m, item in artifact["sparse"].items()}
    code_for = {"SAP": IDENTITIES.index("RIGHT_L5_SAP"), "TP": IDENTITIES.index("RIGHT_L5_TP")}
    endpoints = {}
    for k, core in surface.items():
        endpoints[k] = {
            **endpoint_hits[k.lower()],
            "distance_from_core_voxels_to_all_visible_class": coverage(core, visible.xyz_mm[visible.code == code_for[k]], 1.5),
            "distance_from_core_voxels_to_eligible_visible_class": coverage(core, oracle.xyz_mm[oracle.code == code_for[k]], 1.5),
            "by_method_near_core_voxels": {
                method: coverage(core, cloud.xyz_mm[cloud.code == code_for[k]], 1.5)
                for method, cloud in clouds.items()
            },
        }
    results = {}
    gate = 1.5
    map_config = protocol["landmark_map"]
    dist_pose = distinct_pose_matrix(frame_rows, map_config)
    for m, cloud in clouds.items():
        correctness = load_array(artifact[m]["distances.npz"])
        fusion = cloud_artifacts[m] if m in ("B2", "B3") else None
        distances = tree.query(cloud.xyz_mm)[0]
        from_running = {
            "global_observed_coverage": observed_coverage(oracle, cloud, gate),
            "global_all_visible_coverage": observed_coverage(visible, cloud, gate),
            "global_full_ct_boundary_observed": reference_coverage(boundary, visible, gate),
            "local": {},
        }
        for r in (5, 10, 20):
            ref_boundary = {name: pts[tree.query(pts)[0] <= r] for name, pts in boundary.items()}
            local_eligible = subset(oracle, tree.query(oracle.xyz_mm)[0] <= r)
            local_visible = subset(visible, tree.query(visible.xyz_mm)[0] <= r)
            measured = score_region(cloud, distances <= r, local_eligible, local_visible, ref_boundary,
                                    correctness, dist_pose, fusion, gate)
            measured["full_ct_boundary_observed_all_visible"] = reference_coverage(ref_boundary, local_visible, gate)
            measured["full_ct_boundary_observed_eligible"] = reference_coverage(ref_boundary, local_eligible, gate)
            measured["observed_views"] = {identity: view_support(
                visible, np.flatnonzero((visible.code == code) & (tree.query(visible.xyz_mm)[0] <= r)),
                dist_pose, None) for code, identity in enumerate(IDENTITIES)}
            measured["leave_out"] = leave_views(frame_rows, clouds[{"B2":"B0","B3":"B1"}.get(m,m)],
                                                oracle, tree, map_config, r, gate, fuse=fusion is not None)
            from_running["local"][str(r)] = measured
        results[m] = from_running
    sparse_coverage = {
        m: {str(r): observed_coverage(subset(oracle, tree.query(oracle.xyz_mm)[0] <= r),
                                     subset(q, tree.query(q.xyz_mm)[0] <= r), gate)
            for r in (5,10,20)} for m,q in sparse.items()
    }
    np.savez_compressed(output / f"plot_{seed}.npz", midpoint=midpoint, sap_core=surface["SAP"], tp_core=surface["TP"],
                        visible_xyz=visible.xyz_mm, visible_code=visible.code,
                        **{f"{m}_{name}": value for m, cloud in (clouds | sparse).items()
                           for name, value in (("xyz",cloud.xyz_mm),("code",cloud.code))},
                        **{f"surface_{name}": pts for name,pts in boundary.items()})
    return {"seed": seed, "frames": len(frame_rows), "strict_endpoint_coverage": endpoints, "methods": results,
            "projection": projection["projection"], "sparse_local_observed_coverage":sparse_coverage}, protected


def midpoint_change(points, other, gate):
    forward = nearest(points, other)
    reverse = nearest(other, points)
    return {
        "hausdorff_mm": float(max(forward.max(), reverse.max())),
        "centroid_shift_mm": float(np.linalg.norm(points.mean(0)-other.mean(0))),
        "first_within_3_mm": int(np.count_nonzero(forward <= gate)), "first_total": len(points),
        "second_within_3_mm": int(np.count_nonzero(reverse <= gate)), "second_total": len(other),
    }


def counterfactual(target, segmentation_path, protected, output):
    img, sap, tp, reference = target
    raw = np.asanyarray(nib.load(segmentation_path).dataobj)
    right_coords = nib.affines.apply_affine(img.affine, np.column_stack(
        (np.arange(raw.shape[0]), np.zeros((raw.shape[0], 2)))))[:,0]
    right = right_coords > np.median(right_coords[np.argwhere(raw == 27)[:,0]])
    ranges, affine = crop_geometry(sap, tp, img.affine)
    region = (raw[ranges] == 27) & right[ranges[0], None, None]
    p = protected[ranges]
    base, variants = hidden_sap_variants(sap[ranges], tp[ranges], region, p, affine)
    np.testing.assert_allclose(base.midpoint_world_mm, reference.midpoint_world_mm)
    labels = {name: np.argwhere(mask) for name, mask in (("sap_original",sap[ranges]),("tp_original",tp[ranges]))}
    labels["protected_local_first_bone"] = np.argwhere(p & region)
    mids = {"reference":base.midpoint_world_mm}
    outcomes = []
    for name, change, result in variants:
        n_change = int(np.count_nonzero(change ^ sap[ranges]))
        mids[name] = result.midpoint_world_mm
        outcomes.append({"variant":name,"changed_sap_annotation_voxels":n_change,
                         "sap_mask_voxels":int(change.sum()),"first_visible_semantic_voxels_changed":0,
                         "first_visible_uncertainty_voxels_changed":0,
                         "minimum_separation_mm":result.minimum_surface_separation_mm,
                         "comparison_to_reference":midpoint_change(base.midpoint_world_mm,result.midpoint_world_mm,3.0)})
        labels[f"sap_{name}"] = np.argwhere(change)
    comparisons = [{"first":i,"second":j,**midpoint_change(mids[i],mids[j],3.0)}
                   for i,j in itertools.combinations(mids,2)]
    np.savez_compressed(output/"annotation_sensitivity_geometry.npz",
                        **{k:nib.affines.apply_affine(affine,v) for k,v in labels.items()}, **mids)
    return {
        "scope":"Sensitivity to a hidden semantic annotation frontier within identical source bone and CT. The renderer receives neither the alternative SAP annotation nor altered acoustic, CT, or pose data. This is not a physical anatomy completion or acoustic forward simulation.",
        "constraints":"One or two six-neighbor voxel steps in the apposition-facing L5 SAP frontier. Volume changes remain inside right L5 bone, disjoint from TP, never alter protected first-hit voxels or their one-voxel label uncertainty shell, and preserve six-connected component count.",
        "protection_invariant":"Every first-hit bone voxel sampled in all 408 recorded views and its six-neighbor buffer are protected. SAP class membership and exterior weak-label uncertainty are checked at each protected voxel.",
        "protected_first_hit_voxels_all_frames": int(protected.sum()),
        "protected_in_crop_right_l5":int(np.count_nonzero(p & region)),
        "variants":outcomes,"pairs":comparisons,
    }


def totals(sessions, m, key, region=None, identity=None):
    values=[]
    for session in sessions:
        outcome = session["methods"][m]
        if region is not None:
            outcome = outcome["local"][region]
        value=outcome[key]
        values.append(value["by_identity"][identity] if identity else value)
    return {name: sum(v[name] for v in values) for name in ("covered","total")}


def pooled(sessions):
    return {
        method: {
            "global":totals(sessions,method,"global_observed_coverage"),
            "global_all_visible":totals(sessions,method,"global_all_visible_coverage"),
            "local":{
                r:{name:{
                    "coverage_eligible_visible":totals(sessions,method,"coverage_eligible_visible",r,identity),
                    "coverage_all_visible":totals(sessions,method,"coverage_all_visible",r,identity),
                    "full_ct_boundary_observed_all_visible":totals(sessions,method,"full_ct_boundary_observed_all_visible",r,identity),
                    "full_ct_boundary_observed_eligible":totals(sessions,method,"full_ct_boundary_observed_eligible",r,identity),
                    "full_ct_boundary_to_reconstruction":totals(sessions,method,"full_ct_boundary_to_reconstruction",r,identity),
                } for name,identity in [(k,k) for k in IDENTITIES]+[("all",None)]}
                for r in ("5","10","20")
            }
        } for method in METHODS
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args=parser.parse_args()
    started=time.monotonic()
    data=check_manifest(args.manifest)
    protocol=yaml.safe_load(checked(data["protocol"]).read_text())
    target_files=data["target"]
    for k in ("reference_protocol","observability_protocol","reference_report"):
        checked(target_files[k])
    conversion=yaml.safe_load(checked(target_files["label_conversion"]).read_text())
    bone_ids = {int(k) for k,value in conversion.items() if int(value)==13}
    seg=checked(target_files["segmentation"])
    target=load_target_geometry(checked(target_files["prediction"]),seg)
    args.output.mkdir(parents=True,exist_ok=False)
    history=json.loads(checked(data["historical_acoustic_audit"]).read_text())
    first_hits=np.zeros(target[1].shape,dtype=bool)
    sessions=[]
    for source in data["sessions"]:
        seed=str(source["seed"])
        print(f"Analyzing {seed}",flush=True)
        output, protected = report_session(source,data["clouds"][seed],target,bone_ids,protocol,args.output)
        expected=next(row for row in history["sessions"] if str(row["seed"])==seed)
        for name in ("SAP","TP"):
            if output["strict_endpoint_coverage"][name]["first_visible_frames"] != expected[f"{name.lower()}_endpoint_first_bone_frames"]:
                raise ValueError(f"historical strict first bone mismatch {seed}/{name}")
        sessions.append(output)
        first_hits |= protected
    ambiguity=counterfactual(target,seg,first_hits,args.output)
    combined={
        "scope":"Evaluator-only local reconstruction coverage and geometric identifiability at the provisional s0010 right-L5 proxy.",
        "manifest_sha256":sha256(args.manifest),"local_radii_mm":[5,10,20],"coverage_gate_mm":1.5,
        "target_status":"TARGET_LOCALIZATION_NOT_EVALUABLE",
        "branch_verdict":"INSUFFICIENT_LOCAL_OBSERVABILITY",
        "sessions":sessions,"pooled":pooled(sessions),"hidden_annotation_sensitivity":ambiguity,
        "wall_seconds":time.monotonic()-started,"peak_process_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    }
    write_json(args.output/"summary.json",combined)
    print("Wrote local metrics and annotation sensitivity",flush=True)


if __name__=="__main__":
    main()
