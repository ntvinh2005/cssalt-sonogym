"""Prepare geometric partial observations, select PCA rank on development and score held-out fits."""

import argparse
from collections import defaultdict
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from workflows.mbb_shape_completion.fit import grid, height_ensemble, mask_disk
from workflows.mbb_ssm.fit import Model, digest, fit_arrays, read, write
from workflows.mbb_ssm.geometry import Mesh, distance_metrics, load_mesh, nearest_points, posterior


def posterior_samples(points, step):
    """Approximate three first-visible exterior ray families, never simulate acoustics."""
    samples = []
    for degrees in (0., -25., 25.):
        xyz = Rotation.from_euler("z", degrees, degrees=True).apply(points)
        uv = np.floor(xyz[:, [0, 2]] / step).astype(int)
        order = np.argsort(xyz[:, 1], kind="stable")
        _, distinct = np.unique(uv[order], axis=0, return_index=True)
        samples.append(order[distinct])
    return points[np.unique(np.concatenate(samples))]


def keep_mask(xyz, case):
    points = xyz[:, [0, 2]]
    if case["gap"] == "broad":
        return points[:, 0] < case["right_boundary_mm"]
    return ~mask_disk(points, case["center_uv_mm"], case["gap"] / 2)


def prepare(cohort, output, split, protocol):
    """Save disjoint fitter inputs and geometry held by the evaluator."""
    manifest = read(cohort / "subjects.json")
    subjects = manifest["split"][split]
    input_dir, truth_dir = output / "inputs", output / "reference"
    input_dir.mkdir(parents=True, exist_ok=False)
    truth_dir.mkdir()
    requests = {}
    sizes = ["broad"] if split == "development" else [*protocol["evaluation"]["disk_diameters_mm"], "broad"]
    for subject in subjects:
        path = cohort / "meshes" / f"{subject}.npz"
        if digest(path) != manifest["subjects"][subject]["mesh_sha256"] or subject == protocol["transfer_subject"]:
            raise ValueError("Invalid or excluded geometry in benchmark")
        mesh = load_mesh(path)
        xyz = mesh.points
        visible = posterior_samples(xyz, protocol["mesh_spacing_mm"])
        uv = visible[:, [0, 2]]
        site = uv.min(axis=0) + .67 * np.ptp(uv, axis=0)
        center = uv[np.argmin(np.linalg.norm(uv - site, axis=1))]
        right = float(np.quantile(xyz[:, 0], .58))
        for gap in sizes:
            case = {"gap": gap, "center_uv_mm": center.tolist(), "right_boundary_mm": right}
            keep = keep_mask(visible, case)
            points = visible[keep]
            blocked = visible[~keep]
            if gap == "broad":
                bbox = posterior(xyz) & ~keep_mask(xyz, case)
                blocked = xyz[bbox]
            if not len(blocked):
                raise ValueError(f"Empty hidden region in {subject}, {gap}")
            fence = np.floor(points[:, 0] / 2.).astype(int)
            _, groups = np.unique(fence, return_inverse=True)
            probe = np.zeros((len(np.unique(groups)), 3))
            for index in np.unique(groups):
                probe[index] = np.mean(points[groups == index], axis=0) - [0., 75., 0.]
            key = f"{subject}-{gap}"
            np.savez_compressed(input_dir / f"{key}.npz", xyz_mm=points, basis=np.eye(3), probe_mm=probe,
                                identity=np.full(len(points), "OTHER_BONE"),
                                view_group=groups, confidence=np.ones(len(points)))
            np.savez_compressed(truth_dir / f"{key}.npz", hidden=blocked, full_xyz=xyz, full_faces=mesh.faces)
            requests[key] = {"subject_id": subject, "file": f"{key}.npz", "sha256": digest(input_dir / f"{key}.npz"), **case}
    write(input_dir / "requests.json", requests)
    write(truth_dir / "checksums.json", {p.name: digest(p) for p in sorted(truth_dir.glob("*.npz"))})
    print("Prepared", split, len(subjects), len(requests), flush=True)


def input_arrays(path, request):
    path = path / request["file"]
    if request["sha256"] != digest(path):
        raise ValueError("Actor input checksum differs")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def reference(path, key):
    file = path / f"{key}.npz"
    if read(path / "checksums.json")[file.name] != digest(file):
        raise ValueError("Hidden geometry checksum differs")
    with np.load(file, allow_pickle=False) as data:
        return Mesh(data["full_xyz"], data["full_faces"]), data["hidden"]


def validation(cohort, model_path, work, protocol):
    prepare(cohort, work, "development", protocol)
    fits = {}
    requests = read(work / "inputs/requests.json")
    rank_model = Model.load(model_path, max(protocol["fit"]["candidate_modes"]))
    ranks = [k for k in protocol["fit"]["candidate_modes"] if k <= len(rank_model.modes)]
    if not ranks:
        raise ValueError("Population fit has too few PCA modes")
    for rank, alpha in product(ranks, protocol["fit"]["candidate_prior_weights_mm2"]):
        model = Model(rank_model.mean, rank_model.faces, rank_model.modes[:rank])
        losses, rows = [], {}
        for case, request in requests.items():
            inp = input_arrays(work / "inputs", request)
            surface, _, meta = fit_arrays(inp["xyz_mm"], inp["basis"], inp["identity"], inp["view_group"],
                                           inp["confidence"], model, alpha, protocol["fit"])
            _, hidden = reference(work / "reference", case)
            loss = 20.
            if meta["status"] == "INFERRED":
                closest, _ = nearest_points(hidden, Mesh(surface, model.faces))
                loss = float(np.quantile(np.linalg.norm(hidden - closest, axis=1), .95))
            losses.append(loss)
            rows[case] = {"loss_mm": loss, **meta}
        key = f"{rank}-{alpha}"
        fits[key] = {"rank": rank, "prior_weight_mm2": alpha, "median_loss_mm": float(np.median(losses)),
                     "subjects_accepted": sum(v["status"] == "INFERRED" for v in rows.values()),
                     "cases": rows}
        print("Development", key, fits[key]["median_loss_mm"], fits[key]["subjects_accepted"], flush=True)
    best = min(fits.values(), key=lambda v: (v["median_loss_mm"], v["rank"], -v["prior_weight_mm2"]))
    write(work / "selection_experiment.json", fits)
    freeze = {"rank": best["rank"], "prior_weight_mm2": best["prior_weight_mm2"],
              "development_criterion_median_p95_mm": best["median_loss_mm"],
              "model_sha256": digest(model_path), "development_cases": sorted(requests),
              "cohort_sha256": digest(cohort / "subjects.json")}
    write(work / "freeze.json", freeze)
    print("FROZEN", freeze, flush=True)


def local_completion(data, case, config, radius):
    """Apply the pre-existing local completion fits in the public X/Z chart."""
    new_basis = np.eye(3)[:, [0, 2, 1]]
    if case["gap"] == "broad":
        x = case["right_boundary_mm"]
        z = data["xyz_mm"][:, 2]
        q = grid([(x + .5, x + 22.), (z.min(), z.max())], config["grid_mm"])
        center = np.array([x, np.median(z)])
        required = None
    else:
        center = np.asarray(case["center_uv_mm"])
        r = case["gap"] / 2
        q = grid([(m - r, m + r) for m in center], config["grid_mm"])
        q = q[mask_disk(q, center, r)]
        required = np.argmin(np.linalg.norm(q - center, axis=1))
    xyz = data["xyz_mm"] @ new_basis
    keep = mask_disk(xyz[:, :2], center, radius) if case["gap"] != "broad" else (
            np.abs(xyz[:, 0] - center[0]) <= radius)
    results, details = {}, {}
    for name, mode in (("P1", "linear"), ("P2", "regularized")):
        start = perf_counter()
        outcome, variants, members, _ = height_ensemble(xyz[keep], data["view_group"][keep],
                                                       data["probe_mm"] @ new_basis, q, center, radius,
                                                       config, mode, required_index=required)
        ok = np.isfinite(variants["nominal"])
        results[name] = np.column_stack((q[ok], variants["nominal"][ok])) @ new_basis if outcome == "INFERRED" else np.empty((0, 3))
        details[name] = {"status": outcome, "seconds": perf_counter() - start, "diagnostics": members}
    return results, details


def score_external(work, prediction_dir, protocol, baseline_protocol):
    reqs = read(work / "inputs/requests.json")
    seals = read(prediction_dir / "seals.json")["fits"]
    if seals.keys() != reqs.keys():
        raise ValueError("Incomplete held-out prediction seal")
    tolerances = protocol["evaluation"]["technical_tolerances_mm"]
    pooled = defaultdict(list)
    directory = work / "reconstructed"
    directory.mkdir()
    rows = {}
    for key, case in reqs.items():
        data = input_arrays(work / "inputs", case)
        observed = data["xyz_mm"]
        meta = prediction_dir / f"{key}.json"
        ct_fits = prediction_dir / f"{key}.npz"
        if (seals[key]["outcome"] != digest(meta) or seals[key]["points"] != digest(ct_fits) or
                seals[key]["input"] != case["sha256"]):
            raise ValueError("Held-out fitter output was not sealed")
        local, local_details = local_completion(data, case, baseline_protocol["fit"],
                                                 protocol["evaluation"]["local_context_mm"])
        truth, hidden = reference(work / "reference", key)
        results = {"P0": np.empty((0, 3)), **local}
        details = {"P0": {"status": "OBSERVED_ONLY", "seconds": 0.}, **local_details, **read(meta)}
        arrays = {"observed": observed, "hidden": hidden, "reference": truth.points}
        trials = {}
        with np.load(ct_fits, allow_pickle=False) as fitted:
            faces = fitted["faces"]
            for method in ("P3", "P4"):
                if case["gap"] == "broad":
                    trials[method] = {name: fitted[name] for name in fitted.files if
                                      name.startswith(method + "_alternative_") or
                                      name.startswith(method + "_stress_")}
                if details[method]["status"] == "INFERRED":
                    points = fitted[method]
                    shell = posterior_samples(points, protocol["mesh_spacing_mm"])
                    if case["gap"] == "broad":
                        shell = points[posterior(points)]
                    results[method] = shell[~keep_mask(shell, case)]
                    arrays[method + "_full"] = points
                else:
                    results[method] = np.empty((0, 3))
        rows[key] = {}
        for method, points in results.items():
            combined = np.concatenate((observed, points))
            union = cKDTree(combined).query(hidden)[0]
            forward = cKDTree(points).query(hidden)[0] if len(points) else np.empty(0)
            reverse = cKDTree(hidden).query(points)[0] if len(points) else np.empty(0)
            total_truth = truth.points
            only_posterior = posterior(total_truth)
            extra = {}
            if method + "_full" in arrays:
                fitted_mesh = Mesh(arrays[method + "_full"], faces)
                closest, _ = nearest_points(total_truth, fitted_mesh)
                error = np.linalg.norm(total_truth - closest, axis=1)
                back, _ = nearest_points(fitted_mesh.points, truth)
                global_reverse = np.linalg.norm(fitted_mesh.points - back, axis=1)
                h, _ = nearest_points(hidden, fitted_mesh)
                base_error = np.linalg.norm(hidden - h, axis=1)
                extra["hidden_to_full_fitted_surface"] = distance_metrics(base_error, tolerances)
                if trials.get(method):
                    approximate_spread = [
                        np.linalg.norm(variant - fitted_mesh.points, axis=1) for variant in trials[method].values()
                    ]
                    approximate_spread = np.max(np.stack(approximate_spread), axis=0)
                    linked = cKDTree(fitted_mesh.points).query(h)[1]
                    span = np.maximum(approximate_spread[linked], 0.)
                    extra["uncertainty_diagnostic"] = {
                        "variant_count": len(trials[method]),
                        "heldout_points": len(base_error),
                        "error_inside_nearest_vertex_variant_spread": int((base_error <= span).sum()),
                        "spread_p95_mm": float(np.quantile(span, .95)),
                        "observed_residual_p95_mm": details[method]["observed_p95_mm"],
                        "hidden_error_p95_mm": float(np.quantile(base_error, .95)),
                    }
            else:
                error = cKDTree(combined).query(total_truth)[0]
                global_reverse = cKDTree(total_truth).query(combined)[0]
            rows[key][method] = {
                **details[method], **extra, "hidden_count": len(hidden), "inferred_samples_in_mask": len(points),
                "all_hidden_to_observed_or_inferred": distance_metrics(union, tolerances),
                "hidden_to_inferred_conditional": distance_metrics(forward, tolerances),
                "inferred_to_hidden_conditional": distance_metrics(reverse, tolerances),
                "global_reference_to_result": distance_metrics(error, tolerances),
                "global_result_to_reference": distance_metrics(global_reverse, tolerances),
                "posterior_reference_to_result": distance_metrics(error[only_posterior], tolerances),
                "incorrect_inferred_gt3mm": int((reverse > 3).sum()),
            }
            pooled[(str(case["gap"]), method)].append((union, forward, reverse, error[only_posterior],
                details[method]["status"], details[method]["seconds"]))
            arrays[method] = points
        np.savez_compressed(directory / f"{key}.npz", **arrays)
    aggregate = []
    for (gap, method), values in sorted(pooled.items()):
        all_union, forward, reverse, post = [np.concatenate([v[i] for v in values]) for i in range(4)]
        aggregate.append({"gap": gap, "method": method, "cases": len(values),
                          "accepted": sum(v[4] == "INFERRED" for v in values), "abstained": sum(
                              v[4] not in ("INFERRED", "OBSERVED_ONLY") for v in values),
                          "union": distance_metrics(all_union, tolerances),
                          "hidden_to_inferred_conditional": distance_metrics(forward, tolerances),
                          "inferred_to_hidden_conditional": distance_metrics(reverse, tolerances),
                          "posterior": distance_metrics(post, tolerances),
                          "median_seconds": float(np.median([v[5] for v in values]))})
    write(work / "heldout_results.json", {"cases": rows, "aggregate": aggregate})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validate", "prepare-test", "score-test"))
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline-protocol", type=Path)
    parser.add_argument("--predictions", type=Path)
    a = parser.parse_args()
    config = read(a.protocol)
    if a.phase == "validate":
        validation(a.cohort, a.model, a.output, config)
    elif a.phase == "prepare-test":
        prepare(a.cohort, a.output, "held_out", config)
    else:
        score_external(a.output, a.predictions, config, read(a.baseline_protocol))
