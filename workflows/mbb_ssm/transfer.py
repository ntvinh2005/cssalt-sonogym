"""Replay s0010 observations and evaluate sealed fits using the offline references."""

import argparse
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from workflows.mbb_end_to_end.evaluate import ras_to_base, reference_data
from workflows.mbb_observed_surface.audit import exposed_faces
from workflows.mbb_shape_completion.score import Cortex, export_ply
from workflows.mbb_ssm.fit import LABELS, digest, read, write
from workflows.mbb_ssm.geometry import Mesh, distance_metrics, export_mesh, load_mesh, nearest_points, posterior
from workflows.mbb_ssm.prepare import make_mesh
from workflows.mbb_ssm.train import align


def verify(path, sha):
    if digest(path) != sha:
        raise ValueError(f"Input hash changed for {path.name}")
    return path


def prepare(previous, output, config):
    """Read the public normalized pose ledgers and observed cortical clouds only."""
    base = previous / "actual"
    sources = read(base / "source_receipts.json")
    old = read(base / "inputs/requests.json")
    roots = {"2026121700", "2026121702", "2026121900", "2026121901", "2026121700-126"}
    if set(sources) != roots or set(map(int, (r[:10] for r in roots))) & set(config["reserved_seeds"]):
        raise ValueError("Unexpected transfer cohort")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for case, request in old.items():
        source = verify(base / "inputs" / request["file"] if "file" in request
                        else base / "inputs" / f"{case}.npz", request["sha256"])
        with np.load(source, allow_pickle=False) as inp:
            data = {key: inp[key] for key in inp.files}
        raw_pool = (data["level_hint"] == "L5") & np.isin(data["identity"], LABELS)
        pose = data["basis"]
        actor = {"basis": np.column_stack((pose[:, 0], -pose[:, 2], pose[:, 1])), "probe_mm": data["probe_mm"]}
        for key in ("xyz_mm", "identity", "view_group", "frame_sequence", "confidence", "pixel_row_col"):
            actor[key] = data[key][raw_pool]
        path = output / f"{case}.npz"
        np.savez_compressed(path, **actor)
        manifest[case] = {"file": path.name, "sha256": digest(path), "recording": request["session"],
                          "frames": sources[request["session"]]["frame_count"],
                          "perception": request["perception"], "eligible_points": int(raw_pool.sum()),
                          "original_input_sha256": request["sha256"],
                          "original_input_cloud": str(source)}
    write(output / "requests.json", manifest)


def load_reference(previous, config, need_fresh=True):
    previous_config = previous / "checkout/experiments/unity_two_stage_transfer/s0010_mbb_shape_completion_v1.json"
    sources = read(previous_config)["sources"]
    data = read(verify(Path(sources["recording_manifest"]["path"]), sources["recording_manifest"]["sha256"]))
    before = Path(sources["artifact_root"])
    legacy_ref = verify(Path(sources["reference_seal"]["path"]), sources["reference_seal"]["sha256"])
    info, _, members = reference_data(legacy_ref.parent)
    if info["status"] != "NOT_EVALUABLE" or config["evaluation"]["target_status"] != "TARGET_LOCALIZATION_NOT_EVALUABLE":
        raise ValueError("Archived target status differs")
    p = verify(Path(members["anatomy_hypotheses.npz"]["path"]), members["anatomy_hypotheses.npz"]["sha256"])
    with np.load(p, allow_pickle=False) as v:
        trace = v["diagnostic_cortex_connection_ras_mm"]
    seg = verify(Path(data["target"]["segmentation"]["path"]), data["target"]["segmentation"]["sha256"])
    volume = nib.load(seg)
    faces, axes = exposed_faces(np.asanyarray(volume.dataobj) == 27)
    recording = {str(s["seed"]): s for s in data["sessions"]}
    if need_fresh:
        supplement = before / "capture-52801423/session"
        receipt = read(previous / "actual/evaluator_privileged_input_receipt.json")
        recording["2026121700-126"] = {"path": str(supplement), "files": receipt}
    return trace, seg, volume, faces, axes, recording, data["models"]


def scores_for_surface(surface, ct, ref_posterior, trace, local, observed, tolerances):
    surf = surface.points
    local_points = surf[np.linalg.norm(surf - trace.mean(0), axis=1) <= 9.5]
    trace_closest, _ = nearest_points(trace, surface)
    trace_distance = np.linalg.norm(trace - trace_closest, axis=1)
    prior_to_observed_trace = cKDTree(observed).query(trace)[0]
    closest, _ = nearest_points(ct.points, surface)
    error = np.linalg.norm(ct.points - closest, axis=1)
    from_inferred = ct.distance(surf)
    unseen_local = local & (cKDTree(observed).query(ct.points)[0] > 1.5)
    return {
        "trace_observed_or_inferred": distance_metrics(np.minimum(prior_to_observed_trace, trace_distance), tolerances),
        "trace_inferred_only": distance_metrics(trace_distance, tolerances),
        "trace_vertices_within_1p5mm": int((trace_distance <= 1.5).sum()),
        "local_reference_to_inferred": distance_metrics(error[local], tolerances),
        "local_inferred_to_reference": distance_metrics(ct.distance(local_points), tolerances),
        "local_previous_unobserved_to_inferred": distance_metrics(error[unseen_local], tolerances),
        "global_reference_to_inferred": distance_metrics(error, tolerances),
        "global_inferred_to_reference": distance_metrics(from_inferred, tolerances),
        "posterior_reference_to_inferred": distance_metrics(error[ref_posterior], tolerances),
        "inferred_local_points": len(local_points),
    }


def duplicate_audit(reference_path, training_cohort, training_receipt, spacing):
    """Offline screen for near-identical geometry with train members after sealing."""
    reference, quality = make_mesh(reference_path, spacing, label=27)
    centered = Mesh(reference.points - reference.points.mean(0), reference.faces)
    rows = []
    for subject in training_receipt["training_included"]:
        candidate = load_mesh(training_cohort / "meshes" / f"{subject}.npz")
        aligned = align(candidate, centered, 20)
        x, _ = nearest_points(centered.points, aligned)
        y, _ = nearest_points(aligned.points, centered)
        error = np.r_[np.linalg.norm(x - centered.points, axis=1), np.linalg.norm(y - aligned.points, axis=1)]
        d = distance_metrics(error, [1., 1.5, 3.])
        dims = np.max(np.abs(np.ptp(centered.points, axis=0) - np.ptp(aligned.points, axis=0)))
        rows.append({"subject_id": subject, **d, "extent_difference_mm": float(dims),
                     "near_identical_screen": d["median_mm"] < .35 and d["p95_mm"] < 1. and dims < 2.})
    rows.sort(key=lambda row: row["p95_mm"])
    return {"screen": rows, "transfer_mask": quality, "any_near_identical": any(v["near_identical_screen"] for v in rows)}


def evaluate(previous, inputs, predicted, output, config, cohort, prior):
    requests = read(inputs / "requests.json")
    seal = read(predicted / "seals.json")
    if seal["fits"].keys() != requests.keys() or digest(prior / "model.npz") != seal["model_sha256"]:
        raise ValueError("Transfer predictions do not match the frozen prior/cohort")
    for case, request in requests.items():
        record = seal["fits"][case]
        if record["input"] != request["sha256"]:
            raise ValueError("Input mismatch")
        verify(predicted / f"{case}.npz", record["points"])
        verify(predicted / f"{case}.json", record["outcome"])
        verify(inputs / request["file"], record["input"])
    # Evaluator-only full anatomy and native transforms are opened after every
    # ultrasound-only prediction (including stress variants) has been verified.
    trace_ras, segmentation, volume, faces, axes, sessions, model_sources = load_reference(previous, config)
    for _, entry in model_sources.items():
        verify(Path(entry["path"]), entry["sha256"])
    fit_provenance = read(prior / "model_receipt.json")
    cohort_digest = digest(cohort / "subjects.json")
    if fit_provenance["subject_split_sha256"] != cohort_digest:
        raise ValueError("Training receipt and subject split mismatch")
    audit = duplicate_audit(segmentation, cohort, fit_provenance, config["mesh_spacing_mm"])
    output.mkdir(parents=True, exist_ok=False)
    write(output / "s0010_identity_audit.json", audit)
    write(output / "scoring_receipt.json", {"ct_segmentation_sha256": digest(segmentation),
          "fit_seal_sha256": digest(predicted / "seals.json"), "prior_cohort_sha256": cohort_digest,
          "checkpoint_sources": model_sources})
    geometry = output / "clouds"
    geometry.mkdir()
    old = read(previous / "actual/actual_gap_results.json")
    tolerance = config["evaluation"]["technical_tolerances_mm"]
    results, accepted = {}, {}
    in_ras = nib.affines.apply_affine(volume.affine, faces)
    n_posterior = posterior(in_ras)
    for case, request in requests.items():
        session = sessions[request["recording"]]
        ct = Cortex(faces, axes, volume.affine, session)
        trace = ras_to_base(trace_ras, volume.affine, session)
        local = np.linalg.norm(ct.points - trace.mean(0), axis=1) <= 9.5
        with np.load(inputs / request["file"], allow_pickle=False) as input_set:
            observed = input_set["xyz_mm"]
            export_ply(geometry / f"{case}-OBSERVED.ply", observed, (37, 132, 98),
                       input_set["frame_sequence"], input_set["confidence"])
        export_ply(geometry / f"{case}-CT_EVALUATOR_ONLY.ply", ct.points[local], (128, 133, 145))
        export_ply(geometry / f"{case}-PATH_EVALUATOR_ONLY.ply", trace, (192, 36, 53))
        shape_archive = {"observed": observed, "evaluator_ct": ct.points[local], "evaluator_path": trace}
        outcome = read(predicted / f"{case}.json")
        old_case = old[case]
        rows = {method: old_case["methods"][kind]["completed_trace_coverage"]
                for method, kind in (("P1", "linear"), ("P2", "regularized"))}
        rows.update(P0=old_case["observed_trace_coverage"])
        for method in ("P3", "P4"):
            status = outcome[method]["status"]
            with np.load(predicted / f"{case}.npz", allow_pickle=False) as saved:
                if method not in saved:
                    rows[method] = {**outcome[method], "metrics": None}
                    continue
                inferred = Mesh(saved[method], saved["faces"])
                proposed = scores_for_surface(inferred, ct, n_posterior, trace, local, observed, tolerance)
                rows[method] = {**outcome[method], "metrics": proposed if status == "INFERRED" else None,
                                "rejected_proposal_diagnostics": proposed if status != "INFERRED" else None}
                export_mesh(geometry / f"{case}-{method}-{status}.ply", inferred)
                shape_archive[method] = inferred.points
                shape_archive["faces"] = inferred.faces
                for name in saved.files:
                    if name.startswith(method + "_alternative_"):
                        shape_archive[name] = saved[name]
                if status == "INFERRED":
                    ras = nib.affines.apply_affine(volume.affine, (inferred.points - ct.origin) @ ct.inverse)
                    relevant = np.linalg.norm(ras - trace_ras.mean(0), axis=1) <= 9.5
                    accepted[(case, method)] = (request["perception"], ras, relevant)
        np.savez_compressed(geometry / f"{case}-layers.npz", **shape_archive)
        results[case] = {"recording": request["recording"], "frames": request["frames"],
                         "perception": request["perception"], "trace_samples": len(trace),
                         "local_ct_face_centers": int(local.sum()), "fits": rows}
        print(case, {m: v.get("status") for m, v in rows.items() if m in ("P3", "P4")}, flush=True)
    comparisons = []
    for ((first, fm), (fp, fa, fr)), ((second, sm), (sp, sa, sr)) in combinations(accepted.items(), 2):
        if fm != sm or fp != sp:
            continue
        match = np.linalg.norm(fa - sa, axis=1)
        comparisons.append({"first": first, "second": second, "method": fm,
             "matching_vertex_distance_mm": distance_metrics(match, tolerance),
             "local_matching_vertex_distance_mm": distance_metrics(match[fr | sr], tolerance)})
    write(output / "transfer_results.json", {"cases": results, "cross_session": comparisons,
           "target_localization": config["evaluation"]["target_status"],
           "s0010_geometry_duplicate_flag": audit["any_near_identical"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score"))
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--prior", type=Path)
    a = parser.parse_args()
    protocol = read(a.protocol)
    if a.phase == "prepare":
        prepare(a.previous, a.output, protocol)
    else:
        evaluate(a.previous, a.inputs, a.predictions, a.output, protocol, a.cohort, a.prior)
