"""Generate scientific figures and portable meshes from the sealed comparisons."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import cKDTree

from workflows.mbb_ssm.fit import read, write
from workflows.mbb_shape_completion.score import export_ply
from workflows.mbb_ssm.geometry import Mesh, export_mesh, face_vectors, load_mesh


OBSERVED = "#168468"
POPULATION = "#8463ac"
INFERRED = "#e5772c"
REFERENCE = "#9da3ad"
ERROR = "#c33c43"
METHOD = {"P0": "Observed only", "P1": "Linear", "P2": "Regularized",
          "P3": "Rigid mean", "P4": "PCA prior"}


def cloud(ax, points, color, alpha=1., size=4, marker="o"):
    if len(points):
        ax.scatter(*points.T, c=color, s=size, alpha=alpha, marker=marker, depthshade=False)


def setup_view(ax, title):
    ax.set_title(title, fontsize=10, pad=4)
    ax.view_init(elev=24, azim=-79)
    ax.set_box_aspect((1., .85, .65))
    ax.set_axis_off()


def legend(fig, entries, count=4):
    items = [Line2D([], [], marker="o", color=color, markersize=5, linestyle="none", label=label)
             for color, label in entries]
    fig.legend(handles=items, loc="outside lower center", ncol=count, frameon=False)


def lit_mesh(ax, mesh, color):
    vector = face_vectors(mesh)
    vector /= np.maximum(np.linalg.norm(vector, axis=1)[:, None], 1e-10)
    lamp = np.asarray([-.32, -.46, .82])
    shade = .49 + .48 * np.clip(vector @ lamp, 0, 1)
    rgb = shade[:, None] * np.asarray(to_rgb(color))
    patches = Poly3DCollection(mesh.points[mesh.faces], facecolors=rgb,
                                linewidths=0, antialiased=False)
    ax.add_collection3d(patches)


def training(study, output):
    prior = study / "prior-guarded"
    receipt = read(prior / "model_receipt.json")
    frozen = read(study / "development/freeze.json")
    with np.load(prior / "model.npz", allow_pickle=False) as data:
        mean, modes, variances, faces = (data[k] for k in ("mean", "modes", "eigenvalues", "faces"))
    fig = plt.figure(figsize=(11, 8), layout="constrained")
    for index, name in enumerate(receipt["training_included"][:4]):
        mesh = load_mesh(prior / "correspondences" / f"{name}.npz")
        ax = fig.add_subplot(3, 4, index + 1, projection="3d")
        lit_mesh(ax, mesh, REFERENCE)
        setup_view(ax, f"Training anatomy {index + 1}")
        export_mesh(output / f"training-{index + 1}.ply", mesh)
    for index, m in enumerate(modes[:3]):
        for col, sign in enumerate((-1, 1)):
            v = mean + sign * 2 * (variances[index] ** .5) * m
            ax = fig.add_subplot(3, 4, 5 + 2 * index + col, projection="3d")
            lit_mesh(ax, Mesh(v, faces), POPULATION)
            setup_view(ax, f"Mode {index + 1} · {sign * 2:+d} SD")
    mean_ax = fig.add_subplot(3, 4, 11, projection="3d")
    lit_mesh(mean_ax, Mesh(mean, faces), POPULATION)
    setup_view(mean_ax, "Population mean")
    for view in fig.axes:
        view.view_init(elev=36, azim=-65)
        view.set(xlim=(-51, 51), ylim=(-48, 48), zlim=(-31, 31))
    ax = fig.add_subplot(3, 4, 12)
    values = np.array(receipt["cumulative_explained_variance"]) * 100
    ax.plot(np.arange(1, len(values) + 1), values, color=POPULATION, lw=1.7)
    ax.axvline(frozen["rank"], linestyle="--", color="#333333", linewidth=1.1)
    ax.set(xlabel="Modes", ylabel="Variance retained (%)", ylim=(0, 103))
    fig.suptitle(f"L5 population correspondence · {len(receipt['training_included'])} training shapes")
    fig.savefig(output / "population-modes.png", dpi=155)
    plt.close(fig)


def benchmark(study, output):
    root = study / "heldout"
    results = read(root / "heldout_results.json")
    sizes = ["2", "5", "9", "broad"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    for method, label in METHOD.items():
        rows = {str(r["gap"]).replace(".0", ""): r for r in results["aggregate"] if r["method"] == method}
        color = dict(P0=OBSERVED, P1="#4c78ae", P2="#6e8797", P3=POPULATION, P4=INFERRED)[method]
        for ax, sequence in zip(axes, (
                [rows[g]["union"]["coverage"]["1.5"] * 100 for g in sizes],
                [rows[g]["union"]["p95_mm"] for g in sizes],
                [rows[g]["accepted"] for g in sizes])):
            if method == "P0" and ax is axes[2]:
                continue
            ax.plot(range(4), sequence, "o-", color=color, label=label)
    for ax, title in zip(axes, ("Hidden coverage within 1.5 mm (%)", "Hidden union P95 error (mm)", "Cases with inferred geometry")):
        ax.set_xticks(range(4), ["2 mm", "5 mm", "9 mm", "Broad posterior"])
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.22)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("External held-out anatomy · geometric CT observations")
    fig.savefig(output / "heldout-metrics.png", dpi=165)
    plt.close(fig)

    broad = [name for name in results["cases"] if name.endswith("-broad")]
    accepted = [name for name in broad if results["cases"][name]["P4"]["status"] == "INFERRED"]
    ranked = sorted(accepted, key=lambda name: results["cases"][name]["P4"]["all_hidden_to_observed_or_inferred"]["p95_mm"])
    if ranked:
        broad_case = ranked[len(ranked) // 2]
    else:
        broad_case = sorted(broad)[0]
    site = broad_case.removesuffix("-broad") + "-5"
    fig = plt.figure(figsize=(14, 8.5), layout="constrained")
    for row, key in enumerate((site, broad_case)):
        with np.load(root / "reconstructed" / f"{key}.npz", allow_pickle=False) as inp:
            data = dict(inp)
        with np.load(root / "predictions" / f"{key}.npz", allow_pickle=False) as inp:
            prediction = dict(inp)
        with np.load(root / "reference" / f"{key}.npz", allow_pickle=False) as evaluator:
            export_mesh(output / f"example-{row}-EVALUATOR_ONLY.ply",
                        Mesh(evaluator["full_xyz"], evaluator["full_faces"]))
        export_ply(output / f"example-{row}-OBSERVED.ply", data["observed"], (22, 132, 104))
        export_ply(output / f"example-{row}-WITHHELD.ply", data["hidden"], (157, 163, 173))
        prefix = "5 mm" if row == 0 else "Broad posterior"
        origin = data["reference"].mean(axis=0)
        for col, method in enumerate(METHOD):
            ax = fig.add_subplot(2, 5, row * 5 + col + 1, projection="3d")
            cloud(ax, data["reference"] - origin, REFERENCE, .08, 1.5)
            cloud(ax, data["hidden"] - origin, REFERENCE, .48, 2.)
            cloud(ax, data["observed"] - origin, OBSERVED, .8, 2.)
            cloud(ax, data[method] - origin, INFERRED if method != "P3" else POPULATION, .8, 3.)
            if method in prediction and not len(data[method]) and method in ("P3", "P4"):
                cloud(ax, prediction[method][::5] - origin, ERROR, .16, 1.8)
            if method == "P4":
                candidates = [name for name in prediction if name.startswith("P4_alternative_")]
                if candidates:
                    alt = max(candidates, key=lambda name: np.mean(np.linalg.norm(prediction[name] - prediction["P4"], axis=1)))
                    cloud(ax, prediction[alt][::3] - origin, "#e6ae72", .36, 2.)
                    export_mesh(output / f"example-{row}-alternative.ply",
                                Mesh(prediction[alt], prediction["faces"]))
            record = results["cases"][key][method]
            title = record["status"].replace("_", " ").capitalize()
            setup_view(ax, f"{prefix} · {METHOD[method]}\n{title}")
            if method + "_full" in data:
                export_mesh(output / f"example-{row}-{method}.ply", Mesh(data[method + "_full"], prediction["faces"]))
    legend(fig, [(OBSERVED, "Observed geometric samples"), (REFERENCE, "Evaluator-only withheld anatomy"),
                 (POPULATION, "Rigid mean"), (INFERRED, "Inferred samples"),
                 ("#e6ae72", "Alternative PCA surface"), (ERROR, "Rejected fit")], 3)
    fig.suptitle("Same partial input across all methods")
    fig.savefig(output / "heldout-reconstruction.png", dpi=155)
    plt.close(fig)

    with np.load(root / "reconstructed" / f"{broad_case}.npz", allow_pickle=False) as cloud_data:
        data = dict(cloud_data)
    origin = data["reference"].mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, method in zip(axes, ("P3", "P4")):
        novel = data[method]
        hidden = data["hidden"] - origin
        if len(novel):
            error = cKDTree(novel).query(data["hidden"])[0]
            color = ax.scatter(hidden[:, 0], hidden[:, 2], c=error, s=8, cmap="YlOrRd", vmin=0, vmax=8.)
            fig.colorbar(color, ax=ax, label="Nearest inferred sample (mm)")
        else:
            ax.scatter(hidden[:, 0], hidden[:, 2], c=REFERENCE, s=8)
            ax.text(.05, .9, "Fit abstained", transform=ax.transAxes, color=ERROR)
        ax.set(xlabel="Right (mm)", ylabel="Superior (mm)", title=METHOD[method], aspect="equal")
    fig.suptitle("Held-out broad posterior surface error")
    fig.savefig(output / "heldout-error.png", dpi=155)
    plt.close(fig)


def transfer(study, output):
    root = study / "transfer"
    summary = read(root / "transfer_results.json")
    fig = plt.figure(figsize=(15.2, 8.2), layout="constrained")
    for i, (case, details) in enumerate(summary["cases"].items()):
        with np.load(root / "clouds" / f"{case}-layers.npz", allow_pickle=False) as v:
            data = dict(v)
        with np.load(study / "transfer-inputs" / f"{case}.npz", allow_pickle=False) as inp:
            basis = inp["basis"]
        origin = data["evaluator_path"].mean(axis=0)
        ax = fig.add_subplot(2, 5, (0 if details["perception"] == "oracle" else 5) + i // 2 + 1, projection="3d")
        size = np.linalg.norm(data["observed"] - origin, axis=1) <= 14.
        cloud(ax, (data["observed"][size] - origin) @ basis, OBSERVED, .8, 5.)
        cloud(ax, (data["evaluator_ct"] - origin) @ basis, REFERENCE, .4, 5.)
        for method, color in (("P3", POPULATION), ("P4", INFERRED)):
            if method in data:
                select = np.linalg.norm(data[method] - origin, axis=1) <= 11.
                status = details["fits"][method]["status"]
                cloud(ax, (data[method][select] - origin) @ basis,
                      color if status == "INFERRED" else ERROR, .5 if status == "INFERRED" else .2, 5.)
        cloud(ax, (data["evaluator_path"] - origin) @ basis, ERROR, 1., 25., "x")
        caption = []
        for m in ("P3", "P4"):
            info = details["fits"][m]
            caption.append(f"{m}: {info['metrics']['trace_vertices_within_1p5mm']}/7" if info["metrics"] else f"{m}: abstains")
        perception = "U-Net" if details["perception"] == "learned" else "Oracle"
        setup_view(ax, f"{details['recording'].split('-')[0]} · {details['frames']} frames\n"
                    f"{perception} · {'  '.join(caption)}")
        ax.set(xlim=(-12, 12), ylim=(-12, 12), zlim=(-12, 12))
    legend(fig, [(OBSERVED, "Observed ultrasound"), (REFERENCE, "CT · evaluator only"),
                 (POPULATION, "Inferred rigid mean"), (INFERRED, "Inferred PCA surface"),
                 (ERROR, "CT path or rejected fit")], 5)
    fig.suptitle("s0010 transfer · L5 cortical-path diagnostics")
    fig.savefig(output / "s0010-transfer.png", dpi=160)
    plt.close(fig)


def evaluate_shape_envelope(study, output):
    """Describe external size extrapolation and empirical stress coverage."""
    cohort = read(study / "cohort-expanded/subjects.json")
    model = read(study / "prior-guarded/model_receipt.json")
    subjects = model["training_included"]
    training = [cohort["subjects"][s]["extent_ras_mm"] for s in subjects]
    lo, hi = np.min(training, axis=0), np.max(training, axis=0)
    broad = {key.removesuffix("-broad"): value for key, value in
             read(study / "heldout/heldout_results.json")["cases"].items() if key.endswith("-broad")}
    result = {}
    for key in cohort["split"]["held_out"]:
        extent = np.asarray(cohort["subjects"][key]["extent_ras_mm"])
        outside = np.maximum(np.maximum(extent - hi, lo - extent), 0.)
        methods = {}
        for method in ("P3", "P4"):
            metrics = broad[key][method]
            methods[method] = {"status": metrics["status"], "observed_p95_mm": metrics["observed_p95_mm"],
                               "hidden_p95_mm": metrics["all_hidden_to_observed_or_inferred"]["p95_mm"],
                               "stress_comparison": metrics.get("uncertainty_diagnostic")}
        result[key] = {"out_of_training_size_envelope": bool(outside.max() > 0.),
                       "extent_outside_train_max_mm": float(outside.max()), "methods": methods}
    write(output / "heldout_size_and_uncertainty.json", {
        "training_extent_RAS_min_mm": lo.tolist(), "training_extent_RAS_max_mm": hi.tolist(),
        "subjects": result})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    training(a.study, a.output)
    benchmark(a.study, a.output)
    transfer(a.study, a.output)
    evaluate_shape_envelope(a.study, a.output)
