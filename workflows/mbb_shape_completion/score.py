"""Score sealed completion predictions against withheld ultrasound and offline CT."""

import argparse
import csv
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from workflows.mbb_end_to_end.evaluate import ras_to_base, reference_data
from workflows.mbb_observed_surface.audit import exposed_faces
from workflows.mbb_shape_completion.fit import METHODS, mask_disk
from workflows.mbb_shape_completion.predict import digest, dump, load_cloud, read
from workflows.mbb_shape_completion.prepare import checked


def distances(source, destination):
    if not len(destination):
        return np.full(len(source), np.inf)
    return cKDTree(destination).query(source)[0]


def summarize(values, gate, outlier):
    values = np.asarray(values)
    volume = values[np.isfinite(values)]
    return {"sample_count": len(values), "finite_sample_count": len(volume),
            "median_mm": float(np.median(volume)) if len(volume) else None,
            "mean_mm": float(volume.mean()) if len(volume) else None,
            "p95_mm": float(np.quantile(volume, .95)) if len(volume) else None,
            "within_tolerance": int((values <= gate).sum()),
            "outliers": int((values > outlier).sum()), "tolerance_mm": gate}


class Cortex:
    """Evaluator-only CT voxel faces expressed in a session's actor coordinates."""

    def __init__(self, faces, axes, affine, session):
        self.faces = faces
        self.axes = axes
        self.spacing = nib.affines.voxel_sizes(affine)
        sample = nib.affines.apply_affine(affine, np.vstack((np.zeros(3), np.eye(3))))
        mapped = ras_to_base(sample, affine, session)
        self.origin = mapped[0]
        self.jacobian = mapped[1:] - mapped[0]
        self.inverse = np.linalg.inv(self.jacobian)
        self.points = self.base(faces)
        self.tree = cKDTree(faces * self.spacing)
        self.corner_radius = np.linalg.norm(self.spacing) / 2

    def base(self, voxel):
        return voxel @ self.jacobian + self.origin

    def distance(self, base):
        if not len(base):
            return np.empty(0)
        voxel = (base - self.origin) @ self.inverse
        radius, nearest = self.tree.query(voxel * self.spacing)
        delta = np.abs(voxel - self.faces[nearest])
        half = .5 * (np.arange(3) != self.axes[nearest, None])
        best = np.linalg.norm(np.maximum(delta - half, 0) * self.spacing, axis=1)
        # A closer face must have a centre within the current face-distance
        # bound plus one maximum half-face diagonal.
        neighbors = self.tree.query_ball_point(voxel * self.spacing, best + self.corner_radius + 1e-9)
        for i, adjacent in enumerate(neighbors):
            d = np.abs(voxel[i] - self.faces[adjacent])
            half = .5 * (np.arange(3) != self.axes[adjacent, None])
            best[i] = np.linalg.norm(np.maximum(d - half, 0) * self.spacing, axis=1).min()
        return best


def subset_for_patch(data, identity, center, route, radius):
    xy = data["xyz_mm"] @ route[:, :2]
    rows = ((data["identity"] == identity) & (data["level_hint"] == "L5") &
            mask_disk(xy, center, radius))
    return data["xyz_mm"][rows]


def check_seal(folder, output):
    seals = read(folder / "seals.json")
    requests = read(output / "inputs" / "requests.json")
    if set(seals) != set(requests):
        raise ValueError("actor seals and request set differ")
    for name, hash_value in seals.items():
        checked({"path": str(folder / name / "result.json"), "sha256": hash_value})
        req = requests[name]
        checked({"path": str(output / "inputs" / f"{name}.npz"), "sha256": req["sha256"]})
        result = read(folder / name / "result.json")
        for method in result["methods"].values():
            if "file" in method:
                checked({"path": str(folder / name / method["file"]), "sha256": method["sha256"]})
    return requests


def evaluate_band(truth, arrays, cfg):
    if not len(truth) or not len(arrays["band"]):
        return {"matched": 0, "covered": 0, "comparison": 0}
    chart = (truth - arrays["chart_origin_mm"]) @ arrays["uv_basis"]
    band = arrays["band"]
    xy = arrays["query_uv"][band[:, 0].astype(int)]
    displacement, nearest = cKDTree(xy).query(chart[:, :2])
    matched = displacement <= cfg["uncertainty_sample_radius_uv_mm"]
    low, high = band[nearest, 1], band[nearest, 2]
    inside = (chart[:, 2] >= low) & (chart[:, 2] <= high) & matched
    return {"matched": int(matched.sum()), "covered": int(inside.sum()), "comparison": len(truth)}


def export_ply(path, xyz, rgb, frames=None, confidence=None):
    with path.open("x") as stream:
        stream.write("ply\nformat ascii 1.0\ncomment coordinates actor robot base millimetres\n")
        stream.write(f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if frames is not None:
            stream.write("property int frame_sequence\nproperty float semantic_confidence\n")
        stream.write("end_header\n")
        colors = np.broadcast_to(np.asarray(rgb), (len(xyz), 3))
        extra = np.column_stack((frames, confidence)) if frames is not None else np.empty((len(xyz), 0))
        spec = "%.5f %.5f %.5f %d %d %d" + (" %d %.6f" if frames is not None else "")
        np.savetxt(stream, np.column_stack((xyz, colors, extra)), fmt=spec)


def score_benchmark(requests, output, cortex, protocol):
    cfg = protocol["evaluation"]
    gate, outlier = cfg["ultrasound_technical_tolerance_mm"], cfg["outlier_mm"]
    seals = read(output / "curator_receipt.json")
    cases, pooled, controls = {}, {}, {}
    tables = output / "inspect"
    tables.mkdir(exist_ok=False)
    for key, request in requests.items():
        f = output / "predictions" / key
        outcome = read(f / "result.json")
        if request["perception"] == "synthetic":
            controls[key] = {m: outcome["methods"][m]["status"] for m in METHODS}
            continue

        def truth_file(name):
            file = output / "heldout" / f"{name}.npz"
            return load_cloud(checked({"path": str(file), "sha256": seals[name]["heldout_sha256"]}))

        inp = load_cloud(output / "inputs" / f"{key}.npz")
        oracle_key = key.removesuffix(request["perception"]) + "oracle"
        identity = request["identity"]
        center, gap = request["center_uv_mm"], request["gap_mm"]
        select = (identity, center, inp["basis"])
        truth = subset_for_patch(truth_file(oracle_key), *select, gap / 2)
        own = subset_for_patch(truth_file(key), *select, gap / 2)
        observed = subset_for_patch(inp, *select, request["context_radius_mm"])
        ct = cortex[request["session"]]
        ct_truth_distance = ct.distance(truth)
        compatible = (ct_truth_distance <= cfg["ct_reference_distance_mm"]).mean()
        ct_available = compatible >= cfg["ct_reference_minimum_heldout_fraction"]
        cortex_in_patch = ct.points[mask_disk(ct.points @ inp["basis"][:, :2], center, gap / 2)]
        relevant_ct_samples = cortex_in_patch[distances(cortex_in_patch, truth) <= cfg["ct_reference_distance_mm"]]
        record = {"session": request["session"], "identity": identity, "gap_mm": gap,
                  "perception": request["perception"], "withheld_oracle_points": len(truth),
                  "withheld_own_points": len(own), "remaining_local_observations": len(observed),
                  "ct_withheld_correspondence_fraction": float(compatible),
                  "ct_correspondence_available": bool(ct_available), "methods": {}}
        inspect = {"observed": observed, "withheld_oracle": truth, "ct_reference": relevant_ct_samples}
        for method in ("none", *METHODS):
            arrays = {}
            accepted = np.empty((0, 3))
            meta = outcome["methods"].get(method, {})
            if "file" in meta:
                arrays = load_cloud(f / meta["file"])
                accepted = arrays["accepted"]
            if method == "none":
                status = "OBSERVED_ONLY"
            else:
                status = meta["status"]
            reported = accepted if method != "none" else observed
            directed = distances(truth, reported)
            reverse = distances(accepted, truth)
            union = np.concatenate((observed, accepted))
            union_error = distances(truth, union)
            own_to_res = distances(own, reported)
            band = evaluate_band(truth, arrays, cfg) if len(accepted) else {
                "matched": 0, "covered": 0, "comparison": len(truth)}
            ct_error = ct.distance(accepted) if ct_available else []
            region_error = (distances(relevant_ct_samples, accepted)
                            if ct_available and len(accepted) else np.empty(0))
            proposal = arrays.get("proposed", np.empty((0, 3)))
            report = {"status": status, "eligible_ct_local_reference_points": len(relevant_ct_samples),
                      "returned_inferred_points": len(accepted), "unaccepted_proposal_points": len(proposal) - len(accepted),
                      "runtime_seconds": meta.get("runtime_seconds"),
                      "oracle_to_result": summarize(directed, gate, outlier),
                      "inferred_to_oracle": summarize(reverse, gate, outlier),
                      "own_withheld_to_result": summarize(own_to_res, gate, outlier),
                      "oracle_to_observed_plus_inferred": summarize(union_error, gate, outlier),
                      "inferred_to_ct_exterior": summarize(ct_error, gate, outlier),
                      "ct_region_samples_to_inferred": summarize(region_error, gate, outlier),
                      "stress_range_empirical_calibration": band,
                      "stress_range": meta.get("fits", {}).get("ensemble"),
                      "fit_diagnostics": meta.get("fits", {}).get("nominal")}
            record["methods"][method] = report
            inspect[f"{method}_inferred"] = accepted
            inspect[f"{method}_proposal_unresolved"] = proposal if not len(accepted) else np.empty((0, 3))
            if arrays:
                for name, points in arrays.items():
                    if name.startswith("variant_"):
                        inspect[f"{method}_{name}"] = points[np.isfinite(points).all(axis=1)]
                nonempty = arrays["band"]
                q = arrays["query_uv"][nonempty[:, 0].astype(int)]
                inspect[f"{method}_stress_band_low"] = np.column_stack((q, nonempty[:, 1])) @ arrays["uv_basis"].T + arrays["chart_origin_mm"]
                inspect[f"{method}_stress_band_high"] = np.column_stack((q, nonempty[:, 2])) @ arrays["uv_basis"].T + arrays["chart_origin_mm"]
            pooled.setdefault((request["perception"], gap, method), []).append(
                (status, directed, reverse, union_error, ct_error, band, meta.get("fits", {}).get("ensemble")))
        cases[key] = record
        np.savez_compressed(tables / f"{key}.npz", **inspect)
    totals = []
    for (mode, gap, method), values in sorted(pooled.items()):
        accepted = [v for v in values if v[0] == "INFERRED" or method == "none"]
        count = sum(len(v[1]) for v in values)
        union = np.concatenate([v[3] for v in values])
        arrays = [np.concatenate([np.asarray(v[i]) for v in accepted]) if accepted else np.empty(0)
                  for i in (1, 2, 4)]
        directed, reverse, ct_error = (summarize(v, gate, outlier) for v in arrays)
        reports = [v[6] for v in values if v[6] is not None and v[6]["maximum_disagreement_mm"] is not None]
        band = {key: sum(v[5][key] for v in values) for key in ("matched", "covered", "comparison")}
        totals.append({"perception": mode, "gap_mm": gap, "method": method, "cases": len(values),
                       "abstentions": len(values) - len(accepted), "withheld_points": count,
                       "oracle_to_result_conditional": directed, "inferred_to_oracle_conditional": reverse,
                       "inferred_to_ct_exterior_conditional": ct_error,
                       "all_case_inferred_only_recovered": directed["within_tolerance"] if method != "none" else 0,
                       "all_case_observed_plus_inferred_recovered": int((union <= gate).sum()),
                       "stress_range": {"band_match": band, "available_cases": len(reports),
                                        "maximum_disagreement_mm": max((v["maximum_disagreement_mm"] for v in reports), default=None)}})
    dump(output / "benchmark_results.json", {"cases": cases, "aggregate": totals, "negative_controls": controls})
    return len(cases), len(controls)


def score_anatomical_gap(requests, output, cortex, protocol, reference, affine, sessions):
    cfg = protocol["evaluation"]
    gate, outlier = cfg["ultrasound_technical_tolerance_mm"], cfg["outlier_mm"]
    outcomes = {}
    layers = output / "clouds"
    layers.mkdir(exist_ok=False)
    for key, request in requests.items():
        folder = output / "predictions" / key
        pred = read(folder / "result.json")
        observed = load_cloud(output / "inputs" / f"{key}.npz")
        xyz, identity = observed["xyz_mm"], observed["identity"]
        use = (observed["level_hint"] == "L5") & np.isin(identity, ("RIGHT_L5_TP", "RIGHT_L5_SAP", "OTHER_BONE"))
        ct = cortex[request["session"]]
        trace = ras_to_base(reference["diagnostic_cortex_connection_ras_mm"], affine, sessions[request["session"]])
        original = distances(trace, xyz[use])
        center, radius = trace.mean(0), protocol["actual"]["context_radius_mm"]
        local_ct = ct.points[np.linalg.norm(ct.points - center, axis=1) <= radius]
        export_ply(layers / f"{key}-OBSERVED.ply", xyz, (37, 132, 98),
                   observed["frame_sequence"], observed["confidence"])
        export_ply(layers / f"{key}-CT_EVALUATOR_ONLY.ply", local_ct, (155, 157, 160))
        export_ply(layers / f"{key}-CT_DIAGNOSTIC_TRACE.ply", trace, (198, 40, 52))
        record = {"observed_points": len(xyz), "eligible_l5_points": int(use.sum()),
                  "observed_trace_coverage": summarize(original, gate, outlier),
                  "local_ct_face_centers": len(local_ct),
                  "observed_local_ct_coverage": summarize(distances(local_ct, xyz[use]), gate, outlier),
                  "methods": {}, "input_diagnostics": pred["input_diagnostics"]}
        arrays_for_review = {"observed": xyz, "eligible": use, "identity": identity,
                             "ct_reference": local_ct, "diagnostic_trace": trace}
        for method, meta in pred["methods"].items():
            arrays = load_cloud(folder / meta["file"]) if "file" in meta else {}
            inferred = arrays.get("accepted", np.empty((0, 3)))
            proposal = arrays.get("proposed", np.empty((0, 3)))
            union = np.concatenate((xyz[use], inferred))
            record["methods"][method] = {
                **meta, "completed_trace_coverage": summarize(distances(trace, union), gate, outlier),
                "completed_local_ct_coverage": summarize(distances(local_ct, union), gate, outlier),
                "inferred_to_l5_cortex": summarize(ct.distance(inferred), gate, outlier),
                "unaccepted_proposals_to_l5_cortex": summarize(ct.distance(proposal), gate, outlier) if not len(inferred) else None,
                "unaccepted_proposals_trace_coverage": summarize(distances(trace, proposal), gate, outlier) if len(proposal) and not len(inferred) else None,
                "target_localization_evaluability": "TARGET_LOCALIZATION_NOT_EVALUABLE",
            }
            if len(inferred):
                export_ply(layers / f"{key}-{method}-INFERRED.ply", inferred, (218, 109, 36))
            if len(proposal) and not len(inferred):
                export_ply(layers / f"{key}-{method}-UNRESOLVED.ply", proposal, (182, 54, 54))
            arrays_for_review[f"{method}_inferred"] = inferred
            arrays_for_review[f"{method}_unresolved"] = proposal if not len(inferred) else np.empty((0, 3))
            for name, points in arrays.items():
                if name.startswith("variant_"):
                    arrays_for_review[f"{method}_{name}"] = points[np.isfinite(points).all(axis=1)]
        np.savez_compressed(layers / f"{key}-layers.npz", **arrays_for_review)
        outcomes[key] = record
    dump(output / "actual_gap_results.json", outcomes)
    with (output / "actual_gap_results.csv").open("x", newline="") as stream:
        rows = csv.writer(stream)
        rows.writerow(("session", "perception", "method", "status", "observed_trace_hits", "completed_trace_hits",
                       "trace_samples", "inferred_points", "unresolved_proposal_points",
                       "cortex_median_mm", "cortex_p95_mm", "max_stress_disagreement_mm", "runtime_seconds"))
        for key, record in outcomes.items():
            session, perception = key.rsplit("-", maxsplit=1)
            for method, metrics in record["methods"].items():
                ct_result = metrics["inferred_to_l5_cortex"]
                variation = metrics.get("fits", {}).get("ensemble", {}).get("maximum_disagreement_mm")
                rows.writerow((session, perception, method, metrics["status"], record["observed_trace_coverage"]["within_tolerance"],
                               metrics["completed_trace_coverage"]["within_tolerance"],
                               metrics["completed_trace_coverage"]["sample_count"], metrics["accepted_points"],
                               metrics["proposal_points"] - metrics["accepted_points"],
                               ct_result["median_mm"], ct_result["p95_mm"], variation, metrics["runtime_seconds"]))
    return len(outcomes)



def compare_accepted_surfaces(requests, output, cortex, protocol, reference, affine, sessions):
    """Compare sealed fits across observed acquisitions in evaluator-only CT RAS."""
    cfg = protocol["evaluation"]
    gate, outlier = cfg["ultrasound_technical_tolerance_mm"], cfg["outlier_mm"]
    models, rows = {}, {}
    for key, request in requests.items():
        result_dir = output / "predictions" / key
        pred = read(result_dir / "result.json")
        session = request["session"]
        ct = cortex[session]
        trace = ras_to_base(reference["diagnostic_cortex_connection_ras_mm"], affine, sessions[session])
        for method, item in pred["methods"].items():
            if not item["accepted_points"]:
                continue
            points = load_cloud(result_dir / item["file"])["accepted"]
            in_trace = distances(trace, points)
            identity = f"{key}-{method}"
            models[identity] = nib.affines.apply_affine(affine, (points - ct.origin) @ ct.inverse)
            rows[identity] = {"inferred_points": len(points),
                              "trace_to_inferred": summarize(in_trace, gate, outlier),
                              "nearest_trace_sample_mm": float(in_trace.min()),
                              "inferred_to_trace": summarize(distances(points, trace), gate, outlier),
                              "inferred_to_external_cortex": summarize(ct.distance(points), gate, outlier)}
    pairs = []
    for (key_a, first), (key_b, second) in combinations(models.items(), 2):
        pairs.append({"first": key_a, "second": key_b,
                      "first_to_second": summarize(distances(first, second), gate, outlier),
                      "second_to_first": summarize(distances(second, first), gate, outlier),
                      "centroid_separation_mm": float(np.linalg.norm(first.mean(0) - second.mean(0)))})
    dump(output / "actual_accepted_surface_comparison.json", {
        "coordinate_system_for_comparison": "evaluator-only CT RAS millimetres",
        "trace_is_anatomical_target": False, "accepted_surfaces": rows, "pairwise": pairs,
    })

def run(protocol, output, phase, compare_only=False):
    # Verify every actor artifact before accessing CT or native pose transforms.
    inputs = check_seal(output / "predictions", output)
    sources = protocol["sources"]
    source = read(checked(sources["recording_manifest"]))
    sessions = {str(s["seed"]): s for s in source["sessions"]}
    root = Path(sources["artifact_root"])
    ref_root = checked(sources["reference_seal"]).parent
    info, _, members = reference_data(ref_root)
    if info["status"] != "NOT_EVALUABLE":
        raise ValueError("this study was frozen for the documented uncertain anatomical reference")
    seg = nib.load(checked(source["target"]["segmentation"]))
    faces, axes = exposed_faces(np.asanyarray(seg.dataobj) == 27)
    if phase == "B":
        fresh = root / "capture-52801423/session"
        pose_names = ("public/frames.jsonl", "privileged/full_pose_replay.jsonl")
        public_receipt = read(output / "source_receipts.json")["2026121700-126"]["pose_ledger"]
        checked(public_receipt)
        sessions["2026121700-126"] = {"path": str(fresh),
                                       "files": {name: digest(fresh / name) for name in pose_names}}
        if not compare_only:
            dump(output / "evaluator_privileged_input_receipt.json", sessions["2026121700-126"]["files"])
        elif sessions["2026121700-126"]["files"] != read(output / "evaluator_privileged_input_receipt.json"):
            raise ValueError("fresh evaluation pose inputs changed after scoring")
    ct = {key: Cortex(faces, axes, seg.affine, session) for key, session in sessions.items()}
    if compare_only:
        if phase != "B":
            raise ValueError("cross-session comparison requires phase B")
        diag = load_cloud(checked(members["anatomy_hypotheses.npz"]))
        compare_accepted_surfaces(inputs, output, ct, protocol, diag, seg.affine, sessions)
        return
    dump(output / "evaluation_input_seal.json",
         {"prediction_seals_sha256": digest(output / "predictions/seals.json"),
          "reference_receipt_sha256": sources["reference_seal"]["sha256"], "ct_anatomy": source["target"]["segmentation"],
          "development_phase": phase})
    if phase == "A":
        print("Scored benchmark cases and controls:", score_benchmark(inputs, output, ct, protocol))
    else:
        diag = load_cloud(checked(members["anatomy_hypotheses.npz"]))
        print("Scored actual-gap cases:", score_anatomical_gap(inputs, output, ct, protocol, diag, seg.affine, sessions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase", choices=("A", "B"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    if args.protocol.with_suffix(".sha256").read_text().split()[0] != digest(args.protocol):
        raise ValueError("invalid frozen protocol")
    run(read(args.protocol), args.output, args.phase, args.compare_only)
