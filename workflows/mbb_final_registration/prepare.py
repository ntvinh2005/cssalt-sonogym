"""Reserve additional local anatomies and reproduce the external broad mask."""

import argparse
from pathlib import Path
import shutil

import numpy as np

from workflows.mbb_ssm.benchmark import posterior_samples
from workflows.mbb_ssm.fit import digest, read, write
from workflows.mbb_ssm.geometry import distance_metrics, load_mesh, nearest_points, posterior
from workflows.mbb_ssm.prepare import make_mesh
from workflows.mbb_ssm.train import align


def new_synthetic_case(mesh, subject, output, protocol):
    # Same posterior broad mask and ray families as the previous campaign.
    spacing = protocol["mesh_spacing_mm"]
    xyz = mesh.points
    boundary = float(np.quantile(xyz[:, 0], .58))
    visible = posterior_samples(xyz, spacing)
    points = visible[visible[:, 0] < boundary]
    hidden = xyz[posterior(xyz) & (xyz[:, 0] >= boundary)]
    _, groups = np.unique(np.floor(points[:, 0] / 2.).astype(int), return_inverse=True)
    probe = np.asarray([points[groups == i].mean(axis=0) - [0., 75., 0.]
                        for i in np.unique(groups)])
    name = f"{subject}-broad"
    actor = output / "inputs" / f"{name}.npz"
    truth = output / "reference" / f"{name}.npz"
    np.savez_compressed(actor, xyz_mm=points, basis=np.eye(3), probe_mm=probe,
                        identity=np.full(len(points), "OTHER_BONE"), view_group=groups,
                        confidence=np.ones(len(points)))
    np.savez_compressed(truth, hidden=hidden, full_xyz=xyz, full_faces=mesh.faces)
    return name, {"subject_id": subject, "file": actor.name, "sha256": digest(actor),
                  "gap": "broad", "right_boundary_mm": boundary}, (truth.name, digest(truth))


def duplicate_screen(mesh, quality, cohort, previous):
    result = []
    extent = np.asarray(quality["extent_ras_mm"])
    for subject, row in cohort.items():
        if not row["accepted"] or np.max(np.abs(extent - row["extent_ras_mm"])) > 4.:
            continue
        path = previous / "cohort-expanded/meshes" / f"{subject}.npz"
        if digest(path) != row["mesh_sha256"]:
            raise ValueError("Historical anatomy digest mismatch")
        old = load_mesh(path)
        fit = align(old, mesh, 20)
        a, _ = nearest_points(mesh.points, fit)
        b, _ = nearest_points(fit.points, mesh)
        distances = np.r_[np.linalg.norm(a - mesh.points, axis=1), np.linalg.norm(b - fit.points, axis=1)]
        metric = distance_metrics(distances, [1., 1.5, 2.])
        result.append({"previous_id": subject, "median_mm": metric["median_mm"], "p95_mm": metric["p95_mm"]})
    return sorted(result, key=lambda row: row["p95_mm"])


def folder(path):
    (path / "inputs").mkdir(parents=True, exist_ok=False)
    (path / "reference").mkdir()


def prepare(previous, runtime, output, protocol, prior_protocol):
    cfg = protocol
    cohort_path = previous / "cohort-expanded/subjects.json"
    if digest(cohort_path) != cfg["source_cohort_sha256"]:
        raise ValueError("Wrong historical shape cohort")
    cohort = read(cohort_path)
    prior = read(previous / "prior-guarded/model_receipt.json")
    if prior["template_subject"] not in cohort["split"]["train"]:
        raise ValueError("Historical prior receipt mismatch")
    reserve = set(cfg["external_reserve_candidates"])
    if (not reserve.issubset(cohort["excluded"]) or reserve.intersection(cohort["subjects"]) or
            reserve.intersection(cfg["external_reserve_excluded_configured_simulator_ids"]) or
            reserve_in_raw_sources(reserve, previous)):
        raise ValueError("Local reserve overlaps historical source, template or configured simulator IDs")

    inherited = previous / "heldout"
    dev = output / "development"
    folder(dev)
    old_requests = read(inherited / "inputs/requests.json")
    old_truth = read(inherited / "reference/checksums.json")
    dev_requests, dev_checks = {}, {}
    for name, request in old_requests.items():
        if request["gap"] != "broad":
            continue
        actor = inherited / "inputs" / request["file"]
        truth = inherited / "reference" / f"{name}.npz"
        if digest(actor) != request["sha256"] or digest(truth) != old_truth[truth.name]:
            raise ValueError("Historical development cloud digest mismatch")
        shutil.copyfile(actor, dev / "inputs" / actor.name)
        shutil.copyfile(truth, dev / "reference" / truth.name)
        dev_requests[name], dev_checks[truth.name] = request, old_truth[truth.name]
    if {r["subject_id"] for r in dev_requests.values()} != set(cohort["split"]["held_out"]):
        raise ValueError("Unexpected development set")
    write(dev / "inputs/requests.json", dev_requests)
    write(dev / "reference/checksums.json", dev_checks)

    final = output / "external-confirmation"
    folder(final)
    requests, checksums, audit = {}, {}, {}
    for subject in cfg["external_reserve_candidates"]:
        path = runtime / "assets/data/HumanModels/selected_dataset_stl" / subject / "combined_label_map.nii.gz"
        try:
            mesh, quality = make_mesh(path, prior_protocol["mesh_spacing_mm"], label=27)
        except (ValueError, FileNotFoundError) as exc:
            audit[subject] = {"eligible": False, "reason": str(exc)}
            continue
        similar = duplicate_screen(mesh, quality, cohort["subjects"], previous)
        repeats = any(row["median_mm"] < .35 and row["p95_mm"] < 1. for row in similar)
        audit[subject] = {"eligible": not repeats, "sha256": digest(path), "source": str(path),
                          "quality": quality, "cross_cohort_geometry_screen": similar}
        if repeats:
            continue
        name, record, (filename, hash_value) = new_synthetic_case(mesh, subject, final, prior_protocol)
        requests[name], checksums[filename] = record, hash_value
    write(final / "inputs/requests.json", requests)
    write(final / "reference/checksums.json", checksums)
    write(output / "external_reservation.json", {"development_count": len(dev_requests),
          "confirmation_count": len(requests), "cases": audit,
          "confirmation_ids": [r["subject_id"] for r in requests.values()],
          "confirmation_input_seal_sha256": digest(final / "inputs/requests.json"),
          "confirmation_reference_seal_sha256": digest(final / "reference/checksums.json"),
          "source_limit": "Verification by documented source IDs and morphology, no cross-visit patient key."})
    print("Reserved", len(requests), "confirmation anatomies and", len(dev_requests), "development cases", flush=True)


def reserve_in_raw_sources(reserve, previous):
    source = read(previous / "external-ct-expanded/acquisition.json")
    return bool(reserve.intersection(row["subject_id"] for row in source["entries"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    a = p.parse_args()
    p0 = read(Path("experiments/unity_two_stage_transfer/mbb_ssm_v1.json"))
    prepare(a.previous, a.runtime, a.output, read(a.protocol), p0)
