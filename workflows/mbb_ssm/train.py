"""Build a training-only mesh correspondence and a PCA population prior."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from workflows.mbb_ssm.geometry import (
    Mesh, distance_metrics, edges, export_mesh, face_vectors, load_mesh, nearest_points,
    normals, posterior, rigid, save_mesh,
)
from workflows.mbb_ssm.prepare import digest


def align(target, template, iterations):
    shifted = target.points - target.points.mean(axis=0)
    rotation = np.eye(3)
    translation = template.points.mean(axis=0)
    tree = cKDTree(template.points)
    for _ in range(iterations):
        xyz = shifted @ rotation + translation
        distance, idx = tree.query(xyz)
        keep = distance <= np.quantile(distance, .9)
        r, t = rigid(shifted[keep], template.points[idx[keep]])
        if Rotation.from_matrix(r).magnitude() > np.deg2rad(25.):
            break
        rotation, translation = r, t
    return Mesh(shifted @ rotation + translation, target.faces)


def correspondence_solver(template, weights):
    pair = edges(template.faces)
    index = np.r_[pair[:, 0], pair[:, 1]]
    neighbor = np.r_[pair[:, 1], pair[:, 0]]
    graph = sparse.coo_matrix((np.ones(len(index)), (index, neighbor)),
                              shape=(len(template.points), len(template.points))).tocsr()
    degree = np.maximum(np.asarray(graph.sum(axis=1)).ravel(), 1.)
    identity = sparse.identity(len(template.points), format="csc")
    laplacian = identity - sparse.diags(1 / degree) @ graph
    smoothness = laplacian.T @ laplacian
    return [factorized((identity + w * smoothness).tocsc()) for w in weights], pair



def guarded_update(vertices, proposal, faces, original_normals, area_fraction, rounds=16):
    """Backtrack only where a displacement would fold or collapse a face."""
    area2 = np.einsum("ij,ij->i", original_normals, original_normals)
    candidate = proposal.copy()
    for _ in range(rounds):
        current_normals = face_vectors(Mesh(candidate, faces))
        bad = np.einsum("ij,ij->i", original_normals, current_normals) < area_fraction * area2
        if not bad.any():
            return candidate
        affected = np.unique(faces[bad])
        candidate[affected] = .5 * (candidate[affected] + vertices[affected])
    # Preserve the last valid deformation if backtracking reaches the bound.
    return vertices

def deform(template, target, solvers, config):
    target_tree = cKDTree(target.points)
    target_normals = normals(target)
    v = template.points.copy()
    template_normals = face_vectors(template)
    for i in range(config["nonrigid_iterations"]):
        model_normals = normals(Mesh(v, template.faces))
        dist, candidates = target_tree.query(v, k=10)
        orientation = np.einsum("ij,ikj->ik", model_normals, target_normals[candidates])
        score = dist ** 2 + 4. * (1. - orientation) ** 2
        match = candidates[np.arange(len(v)), np.argmin(score, axis=1)]
        delta = target.points[match] - v

        # Reverse correspondences add support at observed features that otherwise
        # have few template points. The step cap limits rapid vertex collapse.
        nearby = cKDTree(v).query(target.points)[1]
        reverse = np.zeros_like(v)
        reverse_count = np.bincount(nearby, minlength=len(v))
        consistent = np.einsum("ij,ij->i", target_normals, model_normals[nearby]) > -.1
        np.add.at(reverse, nearby[consistent], target.points[consistent] - v[nearby[consistent]])
        delta += .35 * reverse / np.maximum(reverse_count[:, None], 1.)
        delta *= np.minimum(1., config["maximum_step_mm"] /
                            np.maximum(np.linalg.norm(delta, axis=1), 1e-6))[:, None]
        step = min(len(solvers) - 1, i * len(solvers) // config["nonrigid_iterations"])
        displacement = v + delta - template.points
        after = template.points + np.column_stack([solvers[step](column) for column in displacement.T])
        step_vertex = .2 * v + .8 * after
        if config.get("triangle_area_fraction") is None:
            v = step_vertex
        else:
            v = guarded_update(v, step_vertex, template.faces, template_normals,
                               config["triangle_area_fraction"])
    return Mesh(v, template.faces)


def shape_quality(template, candidate, target, pair, config):
    a, _ = nearest_points(candidate.points, target)
    b, _ = nearest_points(target.points, candidate)
    forward = np.linalg.norm(candidate.points - a, axis=1)
    reverse = np.linalg.norm(target.points - b, axis=1)
    back = posterior(target.points)
    front = posterior(template.points)
    ratio = (np.linalg.norm(np.diff(candidate.points[pair], axis=1)[:, 0], axis=1) /
             np.maximum(np.linalg.norm(np.diff(template.points[pair], axis=1)[:, 0], axis=1), 1e-7))
    flipped = np.einsum("ij,ij->i", face_vectors(template), face_vectors(candidate)) < 0.
    global_metric = distance_metrics(np.r_[forward, reverse], [1., 1.5, 3.])
    posterior_metric = distance_metrics(np.r_[forward[front], reverse[back]], [1., 1.5, 3.])
    distorted = float(np.mean((ratio < .38) | (ratio > 2.65)))
    flip_fraction = float(np.mean(flipped))
    accepted = (global_metric["p95_mm"] <= config["global_p95_gate_mm"] and
                posterior_metric["p95_mm"] <= config["posterior_p95_gate_mm"] and
                distorted <= config["maximum_distorted_edge_fraction"] and
                flip_fraction <= config["maximum_flipped_face_fraction"])
    return {"accepted": accepted, "global": global_metric, "posterior": posterior_metric,
            "edge_ratio_p05": float(np.quantile(ratio, .05)), "edge_ratio_p95": float(np.quantile(ratio, .95)),
            "extreme_edge_fraction": distorted, "face_orientation_reversal_fraction": flip_fraction}


def pca(registered):
    matrix = np.stack(registered).reshape(len(registered), -1)
    center = matrix.mean(axis=0)
    _, singular, v = np.linalg.svd(matrix - center, full_matrices=False)
    var = singular ** 2 / (len(matrix) - 1)
    keep = var > var[0] * 1e-8
    return center.reshape(-1, 3), v[keep].reshape(-1, matrix.shape[1] // 3, 3), var[keep]


def run(cohort, output, protocol):
    cohort_manifest = cohort / "subjects.json"
    source = json.loads(cohort_manifest.read_text())
    groups = source["split"]
    if (sum(map(len, groups.values())) != len(set().union(*map(set, groups.values()))) or
            protocol["transfer_subject"] in set().union(*map(set, groups.values()))):
        raise ValueError("Statistical prior split contains overlap or transfer subject")
    config = protocol["correspondence"]
    output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    meshes = {}
    for subject in groups["train"]:
        path = cohort / "meshes" / f"{subject}.npz"
        if digest(path) != source["subjects"][subject]["mesh_sha256"]:
            raise ValueError("Training mesh digest mismatch")
        mesh = load_mesh(path)
        meshes[subject] = Mesh(mesh.points - mesh.points.mean(axis=0), mesh.faces)
    # Favor the median physical extents. Dimensions are not rescaled.
    dimensions = np.asarray([np.ptp(m.points, axis=0) for m in meshes.values()])
    median = np.median(dimensions, axis=0)
    template_subject = list(meshes)[np.argmin(np.linalg.norm((dimensions - median) / median, axis=1))]
    template = meshes[template_subject]
    solvers, pair = correspondence_solver(template, config["diffusion_weights"])
    registered, kept, quality = [], [], {}
    registered_dir = output / "correspondences"
    registered_dir.mkdir()
    for subject, mesh in meshes.items():
        t0 = perf_counter()
        aligned = align(mesh, template, config["rigid_iterations"])
        fit = template if subject == template_subject else deform(template, aligned, solvers, config)
        report = shape_quality(template, fit, aligned, pair, config)
        quality[subject] = {**report, "seconds": perf_counter() - t0}
        if report["accepted"]:
            registered.append(fit.points)
            kept.append(subject)
            save_mesh(registered_dir / f"{subject}.npz", fit)
        print(subject, report["accepted"],
              report["posterior"]["p95_mm"], report["face_orientation_reversal_fraction"], flush=True)
    receipt = {"template_subject": template_subject, "template_vertices": len(template.points),
               "training_candidates": len(meshes), "training_included": kept, "quality": quality,
               "subject_split_sha256": digest(cohort_manifest), "seconds": perf_counter() - start}
    (output / "correspondence.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if len(kept) < 8:
        raise ValueError(f"Only {len(kept)} training correspondences met the fixed quality limits")
    mean, modes, eigenvalues = pca(registered)
    model_path = output / "model.npz"
    np.savez_compressed(model_path, mean=mean, faces=template.faces, modes=modes, eigenvalues=eigenvalues)
    export_mesh(output / "mean.ply", Mesh(mean, template.faces))
    for index in range(min(3, len(modes))):
        for sign, label in ((-1, "minus"), (1, "plus")):
            shape = mean + sign * 2 * (eigenvalues[index] ** .5) * modes[index]
            export_mesh(output / f"mode{index + 1}-{label}2sd.ply", Mesh(shape, template.faces))
    receipt.update(model_sha256=digest(model_path), variance_mm2=eigenvalues.tolist(),
                   cumulative_explained_variance=np.cumsum(eigenvalues / eigenvalues.sum()).tolist(),
                   seconds=perf_counter() - start)
    (output / "model_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("PCA", len(kept), len(modes), receipt["seconds"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", required=True, type=Path)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    run(a.cohort, a.output, json.loads(a.protocol.read_text()))
