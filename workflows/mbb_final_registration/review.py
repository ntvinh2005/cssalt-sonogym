"""Render four diagnostic figures and export compact measured tables."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from workflows.mbb_final_registration.evaluate import surface_distance, subset_faces
from workflows.mbb_final_registration.register import SAP, TP
from workflows.mbb_ssm.fit import read
from workflows.mbb_ssm.geometry import Mesh


GREEN = "#218260"
ORANGE = "#db702a"
GRAY = "#9da2ad"
RED = "#b52e44"


def cloud(ax, xyz, color, *, marker="o", size=7, alpha=1.):
    if len(xyz):
        ax.scatter(*xyz.T, c=color, marker=marker, s=size, depthshade=False, alpha=alpha)


def axes3(ax, order, extent=14):
    ax.set_title(order, fontsize=11)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=27, azim=-69)


def legend(fig, entries):
    fig.legend(handles=[Line2D([], [], color=color, marker=marker, linestyle="none", label=label)
                        for color, marker, label in entries], ncol=len(entries),
               loc="outside lower center", frameon=False)


def load(path):
    with np.load(path, allow_pickle=False) as inp:
        return {key: inp[key] for key in inp.files}


def primary_view(study, output, info):
    selected = info["selected_arm"]
    outcomes = info["cases"]
    cases = []
    for modality in ("oracle", "learned"):
        visible = [k for k, r in outcomes.items() if r["perception"] == modality]
        choices = [k for k in visible if f"{selected}_observed_anchors" in load(
            study / "transfer-predictions" / f"{k}.npz")]
        cases.append(choices[0] if choices else visible[0])
    fig = plt.figure(figsize=(10, 5.3), layout="constrained")
    for i, key in enumerate(cases):
        d = load(study / "transfer/clouds" / f"{key}-layers.npz")
        fit = load(study / "transfer-predictions" / f"{key}.npz")
        mask = (d["labels"] == TP) | (d["labels"] == SAP)
        obs = d["observed"][mask]
        center, frame = np.median(obs, axis=0), d["basis"]
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        for label, mark in ((TP, "^"), (SAP, "s")):
            cloud(ax, (d["observed"][d["labels"] == label] - center) @ frame, GREEN, marker=mark)
        if f"{selected}_observed_anchors" in fit:
            cloud(ax, (fit[f"{selected}_observed_anchors"] - center) @ frame,
                  GREEN, marker="X", size=80)
            guesses = (fit[f"{selected}_initial_anchor_guesses"] - center) @ frame
            for pair in guesses:
                ax.plot(*pair.T, color=ORANGE, marker="o", alpha=.65)
        else:
            ax.text2D(.06, .85, "No supported anchor pair", color=RED, transform=ax.transAxes)
        axes3(ax, f"{key}\n{selected} anchor initialization", extent=19)
    legend(fig, [(GREEN, "^", "Observed TP"), (GREEN, "s", "Observed SAP"),
                 (GREEN, "X", "Robust anchor centers"), (ORANGE, "o", "Initial model anchor pairs")])
    fig.suptitle("Anatomical initialization from observed interfaces")
    fig.savefig(output / "01-anatomical-initialization.png", dpi=155)
    plt.close(fig)
    return cases


def original_and_frozen(study, output, info, cases):
    selected = info["selected_arm"]
    fig = plt.figure(figsize=(10.5, 7.7), layout="constrained")
    err_fig, err_axes = plt.subplots(2, 2, figsize=(9.3, 7.), sharex=True, sharey=True, layout="constrained")
    mappable = None
    for row, key in enumerate(cases):
        d = load(study / "transfer/clouds" / f"{key}-layers.npz")
        fit = load(study / "transfer-predictions" / f"{key}.npz")
        center, frame = d["trace"].mean(axis=0), d["basis"]
        c = (d["ct_points"] - center) @ frame
        obs = d["observed"][np.linalg.norm(d["observed"] - center, axis=1) <= 14.]
        for col, arm in enumerate(("A0", selected)):
            ax = fig.add_subplot(2, 2, 2 * row + col + 1, projection="3d")
            cloud(ax, (obs - center) @ frame, GREEN, size=5, alpha=.8)
            cloud(ax, c, GRAY, size=5, alpha=.43)
            cloud(ax, (d["trace"] - center) @ frame, RED, marker="x", size=25)
            reason = info["cases"][key]["fits"][arm]["status"]
            error_ax = err_axes[row, col]
            if f"{arm}_proposal" in fit:
                points = fit[f"{arm}_proposal"]
                visible = np.linalg.norm(points - center, axis=1) <= 9.5
                cloud(ax, (points[visible] - center) @ frame, ORANGE if reason == "INFERRED" else RED,
                      size=6, alpha=.63 if reason == "INFERRED" else .24)
                mesh = Mesh(points, fit["faces"])
                keep = np.linalg.norm(points[fit["faces"]].mean(axis=1) - center, axis=1) <= 9.5
                error = surface_distance(d["ct_points"], subset_faces(mesh, keep))
                if keep.any():
                    mappable = error_ax.scatter(c[:, 0], c[:, 2], c=error, cmap="Reds", vmin=0, vmax=7, s=24)
            if not error_ax.collections:
                error_ax.scatter(c[:, 0], c[:, 2], c=GRAY, s=20)
                error_ax.text(.5, .08, "No local proposal", color=RED, transform=error_ax.transAxes, ha="center")
            disposition = "accepted" if reason == "INFERRED" else "unresolved"
            axes3(ax, f"{key}\n{arm} · {disposition}")
            error_ax.set(title=f"{key} · {arm}, {disposition}", aspect="equal",
                         xlabel="Right (mm)", ylabel="Superior (mm)")
    legend(fig, [(GREEN, "o", "Observed ultrasound"), (GRAY, "o", "Evaluator CT"),
                (ORANGE, "o", "Inferred shape"), (RED, "x", "Diagnostic trace or rejected proposal")])
    fig.suptitle(f"Original rigid fit and frozen {selected}")
    fig.savefig(output / "02-original-and-constrained.png", dpi=155)
    plt.close(fig)
    if mappable is not None:
        err_fig.colorbar(mappable, ax=err_axes, location="right", label="CT to local proposed surface (mm)")
    err_fig.suptitle("Local geometry against the evaluator-only CT patch")
    err_fig.savefig(output / "03-local-inferred-surface-error.png", dpi=155)
    plt.close(err_fig)


def ambiguity(study, output, info):
    selected = info["selected_arm"]
    fig = plt.figure(figsize=(11, 5.6), layout="constrained")
    ax = fig.add_subplot(1, 2, 1)
    plot = []
    for m in ("oracle", "learned"):
        for arm in ("A0", selected):
            for shared in (False, True):
                if shared:
                    rows = [r for r in info["pairwise"] if r["shared_102_prefix"]]
                else:
                    rows = [r for r in info["pairwise"] if r["both_102"]]
                values = [r["local_corresponding_vertex_p95_mm"] for r in rows if r["arm"] == arm and
                          r["perception"] == m and r["local_corresponding_vertex_p95_mm"] is not None]
                plot.append((f"{m.title()} {arm}\n{'shared prefix' if shared else 'distinct 102'}", values, shared))
    for i, (_, points, shared) in enumerate(plot):
        if points:
            ax.scatter(np.full(len(points), i), points, c=ORANGE, marker="s" if shared else "o")
        else:
            ax.text(i, .25, "N/A", color=GRAY, ha="center", fontsize=8, rotation=90)
    ax.axhline(2., color=RED, linestyle="--", lw=1.)
    ax.set(xticks=range(len(plot)), xticklabels=[x[0] for x in plot],
           ylabel="Local corresponding-vertex P95 (mm)", title="Accepted fits across sessions")
    ax.tick_params(axis="x", labelsize=7, rotation=20)

    trials = []
    for key, case in info["cases"].items():
        score = case["fits"][selected]
        file = study / "transfer-predictions" / f"{key}.npz"
        data = load(file)
        if f"{selected}_proposal" in data:
            center = data[f"{selected}_proposal"]
            for name, surface in data.items():
                if name.startswith(f"{selected}_alternative_") or name.startswith(f"{selected}_subset_"):
                    magnitude = float(np.quantile(np.linalg.norm(center - surface, axis=1), .95))
                    trials.append((magnitude, key, name, center, surface, score["status"]))
    other = fig.add_subplot(1, 2, 2, projection="3d")
    if trials:
        magnitude, key, name, best, alternate, status = max(trials, key=lambda r: r[0])
        d = load(study / "transfer/clouds" / f"{key}-layers.npz")
        origin, frame = d["trace"].mean(0), d["basis"]
        for xyz, color, marker in ((d["observed"], GREEN, "o"), (d["ct_points"], GRAY, "o"),
                                   (best, ORANGE, "o"), (alternate, RED, ".")):
            mask = np.linalg.norm(xyz - origin, axis=1) <= 14
            cloud(other, (xyz[mask] - origin) @ frame, color, marker=marker, size=5, alpha=.54)
        short = "view-subset refit" if "_subset_" in name else "similar-objective alternative"
        axes3(other, f"{key} · {selected}\n{short} (nominal {'accepted' if status == 'INFERRED' else 'unresolved'})")
    else:
        axes3(other, "No selected-arm comparison mesh")
    legend(fig, [(GREEN, "o", "Observed geometry"), (GRAY, "o", "Evaluator CT"),
                 (ORANGE, "o", "Nominal proposed geometry"), (RED, ".", "Alternative geometry")])
    fig.suptitle("Registration disagreement under repeated views and within a scan")
    fig.savefig(output / "04-consistency-and-ambiguity.png", dpi=155)
    plt.close(fig)


def write_tables(study, info):
    with (study / "transfer/per-session.csv").open("w", newline="") as file:
        csv_rows = csv.writer(file)
        csv_rows.writerow(("recording", "perception", "arm", "status", "observed_P95_mm",
                           "all_raw_observed_P95_mm", "local_median_mm", "local_bidir_P95_mm",
                           "rejected_proposal_bidir_P95_mm", "trace_hits", "trace_total", "seconds_with_subsets"))
        for row in info["cases"].values():
            for arm, f in row["fits"].items():
                error = f["local_error"] or {}
                reject = f["rejected_proposal_local_error"] or {}
                csv_rows.writerow((row["recording"], row["perception"], arm, f["status"], f.get("observed_p95_mm"),
                     f.get("all_raw_to_mean_surface_p95_mm"), error.get("bidirectional_median_mm"),
                     error.get("bidirectional_p95_mm"), reject.get("bidirectional_p95_mm"),
                     f["trace_covered"], row["trace_samples"], f["seconds_with_subsets"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    result = read(a.study / "transfer/results.json")
    a.output.mkdir(parents=True, exist_ok=False)
    cases = primary_view(a.study, a.output, result)
    original_and_frozen(a.study, a.output, result, cases)
    ambiguity(a.study, a.output, result)
    write_tables(a.study, result)
