"""Evaluate sealed rigid shapes against evaluator-only posterior anatomy."""

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from workflows.mbb_end_to_end.evaluate import ras_to_base
from workflows.mbb_final_registration.predict import PROTOCOL, read_protocol, result_key
from workflows.mbb_final_registration.register import ARMS
from workflows.mbb_shape_completion.score import Cortex, export_ply
from workflows.mbb_ssm.fit import digest, read, write
from workflows.mbb_ssm.geometry import Mesh, distance_metrics, export_mesh, nearest_points
from workflows.mbb_ssm.transfer import load_reference, verify


def optional_summary(values):
    values = list(values)
    return float(np.median(values)) if values and all(v is not None for v in values) else None


def surface_distance(points, surface):
    if not len(surface.faces):
        return np.full(len(points), np.inf)
    near, _ = nearest_points(points, surface)
    return np.linalg.norm(near - points, axis=1)


def useful_vertices(mesh):
    return mesh.points[np.unique(mesh.faces)]


def subset_faces(mesh, keep):
    return Mesh(mesh.points, mesh.faces[keep])


def patch_metrics(reference, predicted, sources, tolerances):
    a, b = useful_vertices(reference), useful_vertices(predicted)
    if not len(a) or not len(b):
        return None
    return bidirectional(sources, b, reference, predicted, tolerances)


def bidirectional(reference_points, inferred_points, reference, inferred, tolerance):
    x = distance_metrics(surface_distance(reference_points, inferred), tolerance)
    y = distance_metrics(surface_distance(inferred_points, reference), tolerance)
    return {"reference_to_inferred": x, "inferred_to_reference": y,
            "bidirectional_median_mm": max(x["median_mm"], y["median_mm"]),
            "bidirectional_p95_mm": max(x["p95_mm"], y["p95_mm"])}


def verify_stage(predict, inputs, phase, cfg, freeze):
    seal = read(predict / "seals.json")
    requests = read(inputs / "requests.json")
    expected = digest(freeze) if freeze is not None else None
    if (seal["stage"] != phase or seal["protocol_sha256"] != digest(PROTOCOL) or
            seal["model_sha256"] != cfg["model_sha256"] or
            seal["input_manifest_sha256"] != digest(inputs / "requests.json") or
            seal["freeze_sha256"] != expected or requests.keys() != seal["cases"].keys()):
        raise ValueError("Invalid sealed actor evaluation inputs")
    for case, row in seal["cases"].items():
        if row["input_sha256"] != requests[case]["sha256"]:
            raise ValueError("Sealed observation provenance differs")
        verify(inputs / requests[case]["file"], row["input_sha256"])
        verify(predict / f"{case}.npz", row["array_sha256"])
        verify(predict / f"{case}.json", row["outcome_sha256"])
    return seal, requests


def compare_baseline(previous, phase, case, saved, shape):
    bucket = "heldout/predictions" if phase == "development" else "transfer-predictions"
    root = previous / bucket
    row = read(root / "seals.json")["fits"][case]
    meta = read(verify(root / f"{case}.json", row["outcome"]))
    with np.load(verify(root / f"{case}.npz", row["points"]), allow_pickle=False) as legacy:
        eligible = shape is not None and "P3" in legacy
        agreement = float(np.max(np.abs(shape - legacy["P3"]))) if eligible else None
    same_residual = (abs(meta["P3"]["observed_p95_mm"] - saved["observed_p95_mm"]) <= 1e-8
                     if "observed_p95_mm" in saved else "observed_p95_mm" not in meta["P3"])
    return {"previous_status": meta["P3"]["status"], "proposal_max_abs_mm": agreement,
            "reproduced": saved["status"] == meta["P3"]["status"] and same_residual and
                          (agreement is not None and agreement <= 1e-8 or shape is None)}


def score_external(study, previous, phase, cfg, frozen_path=None):
    base = study / ("development" if phase == "development" else "external-confirmation")
    inp, reference, predictions = base / "inputs", base / "reference", base / "predictions"
    seal, requests = verify_stage(predictions, inp, phase, cfg, frozen_path)
    hashes = read(reference / "checksums.json")
    if phase == "external" and digest(reference / "checksums.json") != read(
            study / "external_reservation.json")["confirmation_reference_seal_sha256"]:
        raise ValueError("Reserved external geometry changed")
    tol = cfg["evaluation"]["tolerances_mm"]
    details, pooled = {}, defaultdict(list)
    for key, request in requests.items():
        with np.load(verify(reference / f"{key}.npz", hashes[key + ".npz"]), allow_pickle=False) as data:
            truth = Mesh(data["full_xyz"], data["full_faces"])
            hidden = data["hidden"]
        with np.load(inp / request["file"], allow_pickle=False) as input_data:
            observed = input_data["xyz_mm"]
        boundary = request["right_boundary_mm"]
        posterior_limit = np.quantile(truth.points[:, 1], .6)
        def clip(mesh):
            c = mesh.points[mesh.faces].mean(axis=1)
            return subset_faces(mesh, (c[:, 0] >= boundary) & (c[:, 1] <= posterior_limit))
        local_truth = clip(truth)
        observed_only = cKDTree(observed).query(hidden)[0]
        rows = {}
        meta = read(predictions / f"{key}.json")
        with np.load(predictions / f"{key}.npz", allow_pickle=False) as saved:
            for arm, fit in meta.items():
                surface = saved[f"{arm}_proposal"] if f"{arm}_proposal" in saved else None
                nominal = None
                if surface is not None:
                    nominal = clip(Mesh(surface, saved["faces"]))
                accepted = fit["status"] == "INFERRED"
                shape = nominal if accepted else None
                union = np.minimum(observed_only, surface_distance(hidden, shape)) if shape is not None else observed_only
                d = patch_metrics(local_truth, nominal, hidden, tol) if nominal is not None else None
                rows[arm] = {**fit, "accepted": accepted, "hidden_count": len(hidden),
                             "all_case_hidden_to_obs_or_inferred": distance_metrics(union, tol),
                             "bidirectional_local_error": d if accepted else None,
                             "rejected_proposal_local_error": d if not accepted else None}
                if phase == "development" and arm == "A0":
                    rows[arm]["baseline_reproduction"] = compare_baseline(previous, phase, key, fit, surface)
                pooled[arm].append((union, rows[arm]))
        details[key] = rows
    aggregate = {}
    for arm, entries in pooled.items():
        accepted = [v for _, v in entries if v["accepted"]]
        conditional = [v["bidirectional_local_error"] for v in accepted]
        aggregate[arm] = {
            "cases": len(entries), "accepted": len(accepted), "rejected": len(entries) - len(accepted),
            "all_case_hidden_to_obs_or_inferred": distance_metrics(np.concatenate([x for x, _ in entries]), tol),
            "accepted_bidirectional_median_mm": optional_summary(
                x["bidirectional_median_mm"] if x else None for x in conditional),
            "accepted_bidirectional_p95_mm": optional_summary(
                x["bidirectional_p95_mm"] if x else None for x in conditional),
            "runtime_seconds_median": optional_summary(v["seconds_with_subsets"] for _, v in entries)}
    summary = {"count": len(requests), "cases": details, "aggregate": aggregate,
               "reference_geometry": "partial simulated CT surfaces", "predictions_sha256": digest(predictions / "seals.json")}
    write(base / "results.json", summary)
    if phase != "development":
        return
    # Only previously opened external cases influence the frozen configuration.
    def ordering(entry):
        d = aggregate[entry]["all_case_hidden_to_obs_or_inferred"]
        return -d["coverage"]["1.5"], d["p95_mm"]
    if not all(r["A0"]["baseline_reproduction"]["reproduced"] for r in details.values()):
        raise ValueError("Historical external A0 failed reproduction")
    grid = cfg["fit"]["other_weights_grid"]
    chosen_weight = min(grid, key=lambda v: (*ordering(result_key("A2", v)), v))
    best_a2 = result_key("A2", chosen_weight)
    winner = "A1" if ordering("A1") <= ordering(best_a2) else "A2"
    write(study / "freeze.json", {"model_sha256": cfg["model_sha256"], "protocol_sha256": digest(PROTOCOL),
        "external_reservation_sha256": seal["dataset_reserve_sha256"], "selection_inputs_sha256": seal["input_manifest_sha256"],
        "development_results_sha256": digest(base / "results.json"), "selected_arm": winner,
        "A2_other_weight": chosen_weight, "tie_prefers": "A1", "comparison": "all-case coverage then hidden forward P95"})
    print("Frozen", winner, "A2 weight", chosen_weight, flush=True)


def local_ct_mesh(ct, faces, voxel_axes, include):
    """Build only the chosen native voxel-face quads, for reverse local distances."""
    f, axis = faces[include], voxel_axes[include]
    other = np.asarray([[i for i in range(3) if i != v] for v in axis])
    quad = np.repeat(f[:, None, :], 4, axis=1)
    corners = np.asarray([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    for k in range(2):
        quad[np.arange(len(f))[:, None], np.arange(4)[None], other[:, k, None]] += corners[None, :, k]
    triangles = (4 * np.arange(len(f))[:, None, None] + [[[0, 1, 2], [0, 2, 3]]]).reshape(-1, 3)
    return Mesh(ct.base(quad.reshape(-1, 3)), triangles)


def score_transfer(study, previous, shape_previous, cfg, frozen):
    inputs, predictions = previous / "transfer-inputs", study / "transfer-predictions"
    _, requests = verify_stage(predictions, inputs, "transfer", cfg, frozen)
    # None of these evaluator resources is used by the prediction process.
    trace_ras, _, volume, voxel_faces, voxel_axes, sessions, checkpoints = load_reference(shape_previous, cfg)
    for file in checkpoints.values():
        verify(Path(file["path"]), file["sha256"])
    tol = cfg["evaluation"]["tolerances_mm"]
    rad = cfg["evaluation"]["local_trace_neighborhood_mm"]
    near_trace = cfg["evaluation"]["diagnostic_tolerance_mm"]
    output = study / "transfer"
    output.mkdir(exist_ok=False)
    figures_data = output / "clouds"
    figures_data.mkdir()
    rows, poses = {}, {}
    for case, request in requests.items():
        recording = request["recording"]
        session = sessions[recording]
        ct = Cortex(voxel_faces, voxel_axes, volume.affine, session)
        trace = ras_to_base(trace_ras, volume.affine, session)
        origin = trace.mean(0)
        included_ct = np.linalg.norm(ct.points - origin, axis=1) <= rad
        cortex = local_ct_mesh(ct, voxel_faces, voxel_axes, included_ct)
        patch_points = ct.points[included_ct]
        export_mesh(figures_data / f"{case}-LOCAL_CT_EVALUATOR_ONLY.ply", cortex)
        export_ply(figures_data / f"{case}-PATH_EVALUATOR_ONLY.ply", trace, (194, 40, 54))
        with np.load(inputs / request["file"], allow_pickle=False) as obs:
            data = dict(obs)
        raw = data["xyz_mm"]
        export_ply(figures_data / f"{case}-OBSERVED.ply", raw, (35, 135, 100), data["frame_sequence"], data["confidence"])
        path_observed = cKDTree(raw).query(trace)[0]
        summary = {}
        meta = read(predictions / f"{case}.json")
        with np.load(predictions / f"{case}.npz", allow_pickle=False) as saved:
            for arm in ARMS:
                row = meta[arm]
                surface = saved[f"{arm}_proposal"] if f"{arm}_proposal" in saved else None
                admissible = row["status"] == "INFERRED"
                d, proposals = None, None
                trace_inferred = np.full(len(trace), np.inf)
                if surface is not None:
                    mesh = Mesh(surface, saved["faces"])
                    mask = np.linalg.norm(surface[mesh.faces].mean(axis=1) - origin, axis=1) <= rad
                    local = subset_faces(mesh, mask)
                    proposals = patch_metrics(cortex, local, patch_points, tol)
                    if admissible:
                        trace_inferred = surface_distance(trace, local)
                        d = proposals
                        ras = nib.affines.apply_affine(volume.affine, (surface - ct.origin) @ ct.inverse)
                        near = np.linalg.norm(ras - trace_ras.mean(0), axis=1) <= rad
                        poses[(case, arm)] = (request["perception"], recording, ras, near)
                    indices = np.unique(local.faces)
                    compressed = Mesh(surface[indices], np.searchsorted(indices, local.faces))
                    if len(compressed.faces):
                        export_mesh(figures_data / f"{case}-{arm}-{'INFERRED' if admissible else 'UNRESOLVED'}.ply", compressed)
                summary[arm] = {**row, "accepted": admissible, "local_error": d,
                    "rejected_proposal_local_error": proposals if not admissible else None,
                    "trace_covered": int((np.minimum(path_observed, trace_inferred) <= near_trace).sum()),
                    "trace_covered_by_inferred": int((trace_inferred <= near_trace).sum())}
                if arm == "A0":
                    summary[arm]["baseline_reproduction"] = compare_baseline(previous, "transfer", case, row, surface)
        np.savez_compressed(figures_data / f"{case}-layers.npz", observed=raw, labels=data["identity"],
                            ct_points=patch_points, trace=trace, basis=data["basis"])
        rows[case] = {"recording": recording, "perception": request["perception"],
                      "frames": request["frames"], "trace_samples": len(trace), "local_CT_samples": len(patch_points),
                      "observed_trace_covered": int((path_observed <= near_trace).sum()), "fits": summary}

    pairs = []
    for ((ka, arm_a), (pa, ra, a, mask_a)), ((kb, arm_b), (pb, rb, b, mask_b)) in combinations(poses.items(), 2):
        if pa == pb and arm_a == arm_b:
            near = mask_a | mask_b
            metrics = distance_metrics(np.linalg.norm(a[near] - b[near], axis=1), tol)
            pairs.append({"case_a": ka, "case_b": kb, "arm": arm_a, "perception": pa,
                          "both_102": not ("-126" in ra or "-126" in rb),
                          "shared_102_prefix": {ra, rb} == {"2026121700", "2026121700-126"},
                          "local_corresponding_vertex_p95_mm": metrics["p95_mm"]})
    agg = {}
    for arm in ARMS:
        agg[arm] = {}
        for modality in ("oracle", "learned"):
            all_cases = [row for row in rows.values() if row["perception"] == modality]
            all_fits = [row["fits"][arm] for row in all_cases]
            admitted = [row["local_error"] for row in all_fits if row["accepted"]]
            comparable = [x for x in pairs if x["arm"] == arm and x["perception"] == modality]
            agg[arm][modality] = {
                "accepted": sum(row["accepted"] for row in all_fits), "cases": len(all_cases),
                "median_observed_p95_mm": optional_summary(x.get("observed_p95_mm") for x in all_fits
                                                           if "observed_p95_mm" in x),
                "median_common_raw_observed_p95_mm": optional_summary(x["all_raw_to_mean_surface_p95_mm"]
                                                                       for x in all_fits if "all_raw_to_mean_surface_p95_mm" in x),
                "local_median_mm": optional_summary(d["bidirectional_median_mm"] if d else None for d in admitted),
                "local_bidirectional_p95_mm": optional_summary(d["bidirectional_p95_mm"] if d else None for d in admitted),
                "trace_covered": sum(x["trace_covered"] for x in all_fits),
                "trace_denominator": sum(x["trace_samples"] for x in all_cases),
                "inter_session_102_pairwise_p95_mm": optional_summary(
                    x["local_corresponding_vertex_p95_mm"] for x in comparable if x["both_102"]),
                "shared_prefix_p95_mm": optional_summary(
                    x["local_corresponding_vertex_p95_mm"] for x in comparable if x["shared_102_prefix"]),
                "runtime_seconds_median": optional_summary(x["seconds_with_subsets"] for x in all_fits)}

    frozen_cfg = read(frozen)
    selected = frozen_cfg["selected_arm"]
    baseline = agg["A0"]["learned"]
    trial = agg[selected]["learned"]
    independent_cases = read(study / "external-confirmation/results.json")
    e_config = cfg["evaluation"]
    if independent_cases["count"]:
        a = independent_cases["aggregate"][selected]["all_case_hidden_to_obs_or_inferred"]
        b = independent_cases["aggregate"]["A0"]["all_case_hidden_to_obs_or_inferred"]
        g5 = "PASS" if a["coverage"]["1.5"] > b["coverage"]["1.5"] and a["p95_mm"] < b["p95_mm"] else "FAIL"
    else:
        g5 = "NOT_EVALUABLE"
    gates = {
        "G1": "PASS" if trial["accepted"] >= e_config["minimum_unet_accepted"] else "FAIL",
        "G2": check_numeric(trial["local_bidirectional_p95_mm"], e_config["maximum_median_bidirectional_p95_mm"]),
        "G3": check_numeric(trial["inter_session_102_pairwise_p95_mm"], e_config["maximum_median_pairwise_p95_mm"]),
        "G4": "PASS" if trial["trace_covered"] > baseline["trace_covered"] else "FAIL",
        "G5": g5, "G6": "PASS",
    }
    decision = ("PASS" if set(gates.values()) == {"PASS"} else
                "PARTIAL" if (trial["accepted"] > baseline["accepted"] and gates["G4"] == "PASS") else "FAIL")
    old_stress = read(previous / "transfer/sealed_stress_scoring.json")
    old_counts = {method: sum(round(r[method]["semantic_anchors_only"]["diagnostic_path_to_stressed_model"]
                                  ["coverage"]["1.5"] * 7)
                               for key, r in old_stress["cases"].items() if key.endswith("-learned"))
                  for method in ("P3", "P4")}
    write(output / "results.json", {"cases": rows, "aggregates": agg, "pairwise": pairs,
            "selected_arm": selected, "gates": gates, "decision": decision, "target": e_config["target_status"],
            "baseline_reproduced": all(v["fits"]["A0"]["baseline_reproduction"]["reproduced"] for v in rows.values()),
            "previous_unet_anchor_only_path_coverage": old_counts,
            "checkpoint_hashes_verified": {k: s["sha256"] for k, s in checkpoints.items()},
            "prediction_seal_sha256": digest(predictions / "seals.json")})
    print("Final", decision, gates, flush=True)


def check_numeric(value, bound):
    return "PASS" if value is not None and value <= bound else "FAIL"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("development", "external", "transfer"))
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--shape-previous", type=Path)
    a = p.parse_args()
    cfg = read_protocol()
    freeze = a.study / "freeze.json"
    if a.stage == "transfer":
        score_transfer(a.study, a.previous, a.shape_previous, cfg, freeze)
    else:
        score_external(a.study, a.previous, a.stage, cfg, freeze if a.stage == "external" else None)
