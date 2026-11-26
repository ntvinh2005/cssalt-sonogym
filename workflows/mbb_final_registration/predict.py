"""Generate and seal fixed-mean registration proposals without evaluator geometry."""

import argparse
from pathlib import Path

import numpy as np

from workflows.mbb_final_registration.register import ARMS, register
from workflows.mbb_ssm.fit import Model, digest, read, write
from workflows.mbb_ssm.geometry import Mesh, nearest_points


PROTOCOL = Path("experiments/unity_two_stage_transfer/mbb_final_registration_v1.json")
PREVIOUS_PROTOCOL = Path("experiments/unity_two_stage_transfer/mbb_ssm_v1.json")


def read_protocol():
    if digest(PROTOCOL) != PROTOCOL.with_suffix(".sha256").read_text().split()[0]:
        raise ValueError("Frozen protocol digest mismatch")
    return read(PROTOCOL)


def result_key(arm, weight):
    return f"A2-weight-{weight:g}" if arm == "A2" else arm


def run(previous, inputs, output, stage, freeze_path=None):
    cfg, original = read_protocol(), read(PREVIOUS_PROTOCOL)["fit"]
    model_path = previous / "prior-guarded/model.npz"
    old_freeze = previous / "development/freeze.json"
    if digest(model_path) != cfg["model_sha256"] or digest(old_freeze) != cfg["old_fit_freeze_sha256"]:
        raise ValueError("Historical fixed-mean source or freeze changed")
    with np.load(model_path, allow_pickle=False) as prior:
        mean, faces = prior["mean"], prior["faces"]
    model = Model(mean, faces, np.empty((0, len(mean), 3)))
    if len(model.modes):
        raise ValueError("Rigid registration requires zero shape coefficients")
    reserve_path = (output.parent.parent if stage in ("development", "external") else output.parent) / "external_reservation.json"
    if not reserve_path.is_file():
        raise ValueError("The external candidates must be reserved before tuning")
    input_receipt = inputs / "requests.json"
    requests = read(input_receipt)
    if stage == "development":
        variants = [("A0", 0.), ("A1", 0.), ("A3", 0.), *[("A2", w) for w in cfg["fit"]["other_weights_grid"]]]
        freeze_sha = None
    else:
        frozen = read(freeze_path)
        if (frozen["protocol_sha256"] != digest(PROTOCOL) or frozen["model_sha256"] != digest(model_path) or
                frozen["external_reservation_sha256"] != digest(reserve_path)):
            raise ValueError("Parameter selection and provenance changed")
        if stage == "external" and digest(input_receipt) != read(reserve_path)["confirmation_input_seal_sha256"]:
            raise ValueError("Reserved independent inputs changed")
        variants = [(arm, frozen["A2_other_weight"] if arm == "A2" else 0.) for arm in ARMS]
        freeze_sha = digest(freeze_path)
    if stage == "transfer":
        reserved = set(cfg["reserved_simulator_seeds"])
        if len(requests) != 10 or {int(x["recording"][:10]) for x in requests.values()} & reserved:
            raise ValueError("Unexpected ultrasound cohort")
    output.mkdir(parents=True, exist_ok=False)
    seals = {}
    for case, request in requests.items():
        path = inputs / request["file"]
        if digest(path) != request["sha256"]:
            raise ValueError("Actor input hash differs")
        with np.load(path, allow_pickle=False) as stored:
            data = {k: stored[k] for k in ("xyz_mm", "basis", "identity", "confidence", "view_group")}
        solutions = {}
        arrays = {"faces": model.faces}
        for arm, weight in variants:
            key = result_key(arm, weight) if stage == "development" else arm
            fit, info = register(data, model, arm, weight, cfg, original, stage != "transfer")
            if f"{arm}_proposal" in fit:
                shape = fit[f"{arm}_proposal"]
                nearest, _ = nearest_points(data["xyz_mm"], Mesh(shape, model.faces))
                residual = np.linalg.norm(nearest - data["xyz_mm"], axis=1)
                info["all_raw_to_mean_surface_median_mm"] = float(np.median(residual))
                info["all_raw_to_mean_surface_p95_mm"] = float(np.quantile(residual, .95))
            solutions[key] = info
            for name, value in fit.items():
                arrays[key + name.removeprefix(arm)] = value
        np.savez_compressed(output / f"{case}.npz", **arrays)
        write(output / f"{case}.json", solutions)
        seals[case] = {"input_sha256": request["sha256"], "array_sha256": digest(output / f"{case}.npz"),
                       "outcome_sha256": digest(output / f"{case}.json")}
        print(case, {m: s["status"] for m, s in solutions.items()}, flush=True)
    write(output / "seals.json", {"stage": stage, "input_manifest_sha256": digest(input_receipt),
          "dataset_reserve_sha256": digest(reserve_path), "model_sha256": digest(model_path),
          "protocol_sha256": digest(PROTOCOL), "freeze_sha256": freeze_sha, "cases": seals})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("development", "external", "transfer"), required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--freeze", type=Path)
    a = p.parse_args()
    run(a.previous, a.inputs, a.output, a.stage, a.freeze)
