"""Fit a frozen population prior to observed surface points and public axes."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from workflows.mbb_ssm.geometry import Mesh, face_vectors, posterior, rigid


LABELS = ("OTHER_BONE", "RIGHT_L5_TP", "RIGHT_L5_SAP")


@dataclass
class Model:
    mean: np.ndarray
    faces: np.ndarray
    modes: np.ndarray  # one standard deviation of displacement per coefficient

    @classmethod
    def load(cls, path, rank):
        with np.load(path, allow_pickle=False) as model:
            return cls(model["mean"], model["faces"],
                       model["modes"][:rank] * np.sqrt(model["eigenvalues"][:rank, None, None]))

    def shape(self, coefficients):
        if len(coefficients) == 0:
            return self.mean
        return self.mean + np.tensordot(coefficients, self.modes, axes=1)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def model_regions(mean):
    """Permissive positional constraints for named RIGHT semantic observations."""
    x, y, z = mean.T
    px, py, pz = (np.quantile(axis, grid) for axis, grid in
                  ((x, [.49, .62, .94]), (y, [.09, .47, .66]), (z, [.45])))
    return [posterior(mean, .72),
            (x >= px[1]) & (y >= py[0]) & (y <= py[2]),
            (px[0] <= x) & (x <= px[2]) & (y <= py[1]) & (z >= pz[0])]


def weights(groups, confidence):
    counts = np.bincount(groups, weights=confidence)
    return confidence / counts[groups] / len(np.unique(groups))


def correspondences(points, shape_world, classes, regions):
    idx = np.empty(len(points), dtype=int)
    distances = np.empty(len(points))
    for i, allowed in enumerate(regions):
        query = classes == i
        if not query.any():
            continue
        candidates = np.flatnonzero(allowed)
        r, j = cKDTree(shape_world[candidates]).query(points[query])
        idx[query] = candidates[j]
        distances[query] = r
    return distances, idx


def initial_translation(points, classes, mean, regions):
    centers = []
    for label, allowed in enumerate(regions):
        sample = points[classes == label]
        if len(sample) < 6:
            continue
        matching = mean[allowed]
        # OTHER_BONE alone has broad posterior compatibility.
        centers.append(np.median(sample, axis=0) - np.median(matching, axis=0))
    return np.median(centers, axis=0) if centers else np.zeros(3)


def _optimize(points, classes, sample_weights, model, regions, config, rotation0, offset0, prior):
    rotation, translation = rotation0, offset0
    standard = np.zeros(len(model.modes))
    bound = config["coefficient_bound_sd"]
    delta = config["outlier_huber_mm"]
    shape = model.shape(standard)
    previous = np.inf
    for iteration in range(config["iterations"]):
        residual, idx = correspondences(points, shape @ rotation + translation, classes, regions)
        robust = sample_weights * np.minimum(1., delta / np.maximum(residual, 1e-6))
        rotation, translation = rigid(shape[idx], points, robust)
        tilt = Rotation.from_matrix(rotation).as_rotvec()
        angle = np.linalg.norm(tilt)
        cap = np.deg2rad(config["max_rotation_from_public_deg"])
        if angle > cap:
            rotation = Rotation.from_rotvec(tilt * (cap / angle)).as_matrix()
            translation = np.average(points - shape[idx] @ rotation, weights=robust, axis=0)
        if len(standard):
            residual_local = (points - translation) @ rotation.T - model.mean[idx]
            design = model.modes[:, idx].reshape(len(standard), -1).T
            weighted = np.repeat(np.sqrt(robust), 3)
            equations = np.vstack((design * weighted[:, None], np.sqrt(prior) * np.eye(len(standard))))
            rhs = np.r_[residual_local.ravel() * weighted, np.zeros(len(standard))]
            standard = lsq_linear(equations, rhs, bounds=(-bound, bound), max_iter=25).x
            shape = model.shape(standard)
        metric = float(sample_weights @ np.where(residual < delta, residual ** 2,
                                                 delta * (2 * residual - delta)) + prior * (standard @ standard))
        if iteration > 6 and abs(previous - metric) < 1e-4:
            break
        previous = metric
    surface = shape @ rotation + translation
    res, _ = correspondences(points, surface, classes, regions)
    loss = sample_weights @ np.where(res <= delta, res ** 2, delta * (2 * res - delta))
    loss += prior * (standard @ standard)
    return surface, {"objective_mm2": float(loss), "observed_median_mm": float(np.median(res)),
                     "observed_p95_mm": float(np.quantile(res, .95)), "coefficients_sd": standard.tolist(),
                     "rotation_from_public_deg": float(Rotation.from_matrix(rotation).magnitude() * 180 / np.pi),
                     "rotation_local_rows": rotation.tolist(), "translation_local_mm": translation.tolist(),
                     "iterations": iteration + 1}



def shape_alternatives(detail, points, classes, w, model, regions, config, prior):
    """Perturb weakly constrained standardized mode combinations."""
    if not len(model.modes):
        return []
    rotation = np.asarray(detail["rotation_local_rows"])
    t = np.asarray(detail["translation_local_mm"])
    b = np.asarray(detail["coefficients_sd"])
    nominal = model.shape(b) @ rotation + t
    _, index = correspondences(points, nominal, classes, regions)
    design = model.modes[:, index].reshape(len(b), -1)
    hessian = (design * np.repeat(w, 3)) @ design.T
    _, directions = np.linalg.eigh(hessian)
    options = []
    for j in range(min(2, len(b))):
        for sign in (-1, 1):
            proposal = b + sign * directions[:, j]
            if np.abs(proposal).max() > config["coefficient_bound_sd"]:
                continue
            shape = model.shape(proposal)
            offset = np.average(points - shape[index] @ rotation, weights=w, axis=0)
            surface = shape @ rotation + offset
            residual, _ = correspondences(points, surface, classes, regions)
            delta = config["outlier_huber_mm"]
            loss = w @ np.where(residual < delta, residual ** 2, delta * (2 * residual - delta))
            loss += prior * (proposal @ proposal)
            row = {"objective_mm2": float(loss), "observed_p95_mm": float(np.quantile(residual, .95)),
                   "coefficients_sd": proposal.tolist(), "direction_index": j, "direction_sign": sign}
            options.append((row, surface))
    return options

def fit_arrays(xyz, basis, labels, groups, confidence, model, prior, config, *, single_start=False):
    """Return a proposed mesh, admissible alternatives and an acceptance record."""
    began = perf_counter()
    if len(xyz) < 20 or len(np.unique(groups)) < config["minimum_distinct_groups"]:
        return None, [], {"status": "INSUFFICIENT_OBSERVATIONS", "seconds": perf_counter() - began}
    if not np.isfinite(xyz).all() or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-6) or np.linalg.det(basis) < 0:
        raise ValueError("Invalid public coordinate frame")
    origin = np.median(xyz, axis=0)
    points = (xyz - origin) @ basis
    groups = np.unique(groups, return_inverse=True)[1]
    w = weights(groups, np.maximum(confidence, .02))
    class_id = np.array([LABELS.index(label) for label in labels], dtype=int)
    regions = model_regions(model.mean)
    translation = initial_translation(points, class_id, model.mean, regions)
    starts = [(0., 0.)]
    if not single_start:
        starts += [(d, 0.) for d in config["initial_y_offsets_mm"] if d]
        starts += [(0., a) for a in config["initial_z_tilts_deg"] if a]
    trials = []
    for shift, tilt in starts:
        surface, trial = _optimize(points, class_id, w, model, regions, config,
                                   Rotation.from_euler("z", tilt, degrees=True).as_matrix(),
                                   translation + [0., shift, 0.], prior)
        trial.update(initial_ap_shift_mm=shift, initial_z_tilt_deg=tilt)
        trials.append((trial, surface))
    trials.sort(key=lambda pair: pair[0]["objective_mm2"])
    detail, best = trials[0]
    unexplored = posterior(model.mean) & (cKDTree(points).query(best)[0] > 3.)
    alternative_surfaces = []
    distances = []
    perturbations = shape_alternatives(detail, points, class_id, w, model, regions, config, prior)
    for row, shape in [*trials[1:], *perturbations]:
        if (row["objective_mm2"] - detail["objective_mm2"] <= config["acceptable_alternative_objective_delta_mm2"] and
                row["observed_p95_mm"] <= config["maximum_observed_p95_mm"]):
            alternative_surfaces.append(shape @ basis.T + origin)
            if unexplored.any():
                distances.append(float(np.quantile(np.linalg.norm(best[unexplored] - shape[unexplored], axis=1), .95)))
    uncertainty = max(distances, default=0.) if unexplored.any() and distances else None
    observed = detail["observed_p95_mm"] <= config["maximum_observed_p95_mm"]
    coeff = np.asarray(detail["coefficients_sd"])
    nearly_bound = len(coeff) > 0 and np.mean(np.abs(coeff) >= config["coefficient_bound_sd"] - .01) >= .5
    in_model = (best - detail["translation_local_mm"]) @ np.asarray(detail["rotation_local_rows"]).T
    flipped = np.mean(np.einsum("ij,ij->i", face_vectors(Mesh(model.mean, model.faces)),
                                face_vectors(Mesh(in_model, model.faces))) < 0)
    if not observed:
        status = "OBSERVED_RESIDUAL_TOO_LARGE"
    elif nearly_bound or flipped > .07:
        status = "IMPLAUSIBLE_SHAPE"
    elif uncertainty is not None and uncertainty > config["maximum_alternative_p95_mm"]:
        status = "AMBIGUOUS_SURFACE"
    else:
        status = "INFERRED"
    times = perf_counter() - began
    return best @ basis.T + origin, alternative_surfaces, {
        **detail, "status": status, "view_groups": len(np.unique(groups)), "observed_points": len(points),
        "multistart_alternatives": len(alternative_surfaces), "hidden_posterior_vertices": int(unexplored.sum()),
        "hidden_posterior_disagreement_p95_mm": uncertainty, "face_reversal_fraction": float(flipped),
        "starts": [trial for trial, _ in trials],
        "prior_perturbations": [row for row, _ in perturbations], "seconds": times,
    }


def pose_perturbation(xyz, basis, groups, probe_mm, config):
    adjusted = xyz.copy()
    for i, group in enumerate(np.unique(groups)):
        take = groups == group
        axis = basis[:, i % 3]
        sign = (-1) ** i
        pivot = probe_mm[group]
        r = Rotation.from_rotvec(axis * sign * np.deg2rad(config["perturb_rotation_deg"]))
        adjusted[take] = r.apply(xyz[take] - pivot) + pivot + sign * config["perturb_translation_mm"] * axis
    return adjusted


def run(model_path, freeze_path, inputs, output, protocol, stress=False):
    frozen = read(freeze_path)
    if digest(model_path) != frozen["model_sha256"]:
        raise ValueError("Population model differs from the validation freeze")
    model = Model.load(model_path, frozen["rank"])
    mean = Model(model.mean, model.faces, model.modes[:0])
    manifest = read(inputs / "requests.json")
    output.mkdir(parents=True, exist_ok=False)
    seals = {}
    cfg = protocol["fit"]
    for case, request in manifest.items():
        source = inputs / request["file"]
        if digest(source) != request["sha256"]:
            raise ValueError("Observed input checksum mismatch")
        with np.load(source, allow_pickle=False) as inp:
            xyz, basis, label, views, confidence = (inp[n] for n in
                       ("xyz_mm", "basis", "identity", "view_group", "confidence"))
            probe = inp["probe_mm"]
        row = {}
        arrays = {"faces": model.faces}
        for method, active in (("P3", mean), ("P4", model)):
            surface, variants, detail = fit_arrays(xyz, basis, label, views, confidence,
                                                   active, frozen["prior_weight_mm2"] if method == "P4" else 0., cfg)
            row[method] = detail
            if surface is None:
                continue
            arrays[method] = surface
            for i, alt in enumerate(variants):
                arrays[f"{method}_alternative_{i}"] = alt
            if stress:
                # Real acquisition groups may contain different numbers of
                # pixels. Drop whole groups and change pose coherently by view.
                ids, counts = np.unique(views, return_counts=True)
                trial_inputs = [(f"drop_view_{g}", xyz, views != g) for g in ids[np.argsort(-counts)[:3]]]
                trial_inputs += [("pose_stress", pose_perturbation(xyz, basis, views, probe, cfg),
                                  np.ones(len(xyz), dtype=bool))]
                if (label != "OTHER_BONE").sum() >= 20:
                    trial_inputs.append(("semantic_anchors_only", xyz, label != "OTHER_BONE"))
                elif (confidence < np.median(confidence)).any():
                    trial_inputs.append(("higher_confidence", xyz, confidence >= np.median(confidence)))
                detail["stresses"] = {}
                posterior_unseen = posterior(model.mean) & (cKDTree(xyz).query(surface)[0] > 3.)
                for kind, moved, keep in trial_inputs:
                    trial, _, metrics = fit_arrays(moved[keep], basis, label[keep], views[keep], confidence[keep],
                                                    active, frozen["prior_weight_mm2"] if method == "P4" else 0.,
                                                    cfg, single_start=True)
                    if trial is not None:
                        arrays[f"{method}_stress_{kind}"] = trial
                        samples = np.linalg.norm(trial[posterior_unseen] - surface[posterior_unseen], axis=1)
                        metrics["hidden_posterior_disagreement_p95_mm"] = (
                            float(np.quantile(samples, .95)) if len(samples) else None)
                    detail["stresses"][kind] = metrics
        np.savez_compressed(output / f"{case}.npz", **arrays)
        write(output / f"{case}.json", row)
        seals[case] = {"points": digest(output / f"{case}.npz"), "outcome": digest(output / f"{case}.json"),
                       "input": request["sha256"]}
        print(case, {m: v["status"] for m, v in row.items()}, flush=True)
    write(output / "seals.json", {"fits": seals, "model_sha256": digest(model_path),
                                  "validation_freeze_sha256": digest(freeze_path),
                                  "stress_scenarios": stress})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--freeze", required=True, type=Path)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--stress", action="store_true")
    a = p.parse_args()
    run(a.model, a.freeze, a.inputs, a.output, read(a.protocol), a.stress)
