"""Create static review figures and layered PLY exports from scored replays."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from workflows.mbb_shape_completion.geometry import describe
from workflows.mbb_shape_completion.predict import digest, dump, load_cloud, read
from workflows.mbb_shape_completion.score import distances, export_ply


OBSERVED = "#258462"
INFERRED = "#da6d24"
REFERENCE = "#a6a8ae"
ERROR = "#b63636"
METHOD_LABELS = {"none": "Observed only", "linear": "Linear interpolation",
                 "regularized": "Regularized thin-plate fit"}
ARM_LABELS = {"oracle": "Oracle-visible", "learned": "U-Net"}


def benchmark_summary(root, output):
    runs = read(root / "benchmark_results.json")
    rows = runs["aggregate"]
    fig, panels = plt.subplots(2, 3, figsize=(13, 7), sharex=True, layout="constrained")
    colors = {"none": OBSERVED, "linear": "#744c9c", "regularized": INFERRED}
    for mode, row in zip(ARM_LABELS, panels):
        for method in METHOD_LABELS:
            series = sorted([v for v in rows if v["perception"] == mode and v["method"] == method],
                            key=lambda v: v["gap_mm"])
            gaps = [v["gap_mm"] for v in series]
            coverage = [v["all_case_observed_plus_inferred_recovered"] / v["withheld_points"] * 100
                        for v in series]
            error = [v["oracle_to_result_conditional"]["median_mm"] for v in series]
            abstentions = [v["abstentions"] for v in series]
            for col, (ax, values) in enumerate(zip(row, (coverage, error, abstentions))):
                if method == "none" and col == 2:
                    continue
                ax.plot(gaps, values, "o-", color=colors[method], label=METHOD_LABELS[method], linewidth=1.7)
        row[0].set(ylabel=f"{ARM_LABELS[mode]}\nWithheld samples recovered (%)", ylim=(0, 104))
        row[1].set(ylabel="Conditional median distance (mm)")
        row[2].set(ylabel="Abstained sites", ylim=(-.4, max(v["cases"] for v in rows) + .4))
        for ax in row:
            ax.set_xticks(sorted({v["gap_mm"] for v in rows}))
            ax.grid(alpha=.2)
    for ax, title in zip(panels[0], ("All-case coverage within 1.5 mm",
                                     "Successful estimates only", "Abstention by gap size")):
        ax.set_title(title)
    for ax in panels[-1]:
        ax.set_xlabel("Masked disk diameter (mm)")
    handles, labels = panels[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", frameon=False, ncol=3)
    fig.suptitle("Recovery of withheld ultrasound-supported cortical surfaces · s0010")
    fig.savefig(output / "benchmark-summary.png", dpi=160)
    plt.close(fig)
    return runs


def select_patch(results, mode, gap):
    rows = [(name, record) for name, record in results["cases"].items()
            if record["perception"] == mode and record["gap_mm"] == gap]
    admitted = [(name, record["methods"]["regularized"]["inferred_to_oracle"]["median_mm"])
                for name, record in rows if record["methods"]["regularized"]["status"] == "INFERRED"]
    admitted = [(name, distance) for name, distance in admitted if distance is not None]
    if not admitted:
        return sorted(rows)[0][0]
    median = np.median([distance for _, distance in admitted])
    return min(admitted, key=lambda row: (abs(row[1] - median), row[0]))[0]


def plot_geometry(ax, points, origin, basis, color, size, **kwargs):
    coords = (points - origin) @ basis
    ax.scatter(*coords.T, s=size, c=color, depthshade=False, **kwargs)


def plot_stress_alternatives(ax, arrays, method, origin, basis):
    nominal = arrays.get(f"{method}_variant_nominal", np.empty((0, 3)))
    if not len(nominal):
        return
    prefix = f"{method}_variant_"
    options = [name for name, points in arrays.items()
               if name.startswith(prefix) and len(points) and not name.endswith("_nominal")]
    if not options:
        return
    candidate = max(options, key=lambda name: np.quantile(distances(arrays[name], nominal), .9))
    plot_geometry(ax, arrays[candidate], origin, basis, "#f7b77b", 9, alpha=.7)


def benchmark_examples(root, output, results):
    inspect = root / "inspect"
    exports = output / "controlled-examples"
    exports.mkdir(exist_ok=False)
    sizes = sorted({v["gap_mm"] for v in results["aggregate"]})
    for mode in ARM_LABELS:
        fig = plt.figure(figsize=(13, 11), layout="constrained")
        for i, gap in enumerate(sizes):
            key = select_patch(results, mode, gap)
            ref = results["cases"][key]
            arrays = load_cloud(inspect / f"{key}.npz")
            base = load_cloud(root / "inputs" / f"{key}.npz")["basis"]
            report = read(root / "inputs" / "requests.json")[key]
            origin = np.r_[report["center_uv_mm"], arrays["withheld_oracle"].mean(0) @ base[:, 2]] @ base.T
            tagged = [("OBSERVED", arrays["observed"], (37, 132, 98)),
                      ("WITHHELD_ULTRASOUND", arrays["withheld_oracle"], (153, 158, 164)),
                      ("CT_EVALUATOR_ONLY", arrays["ct_reference"], (187, 189, 195))]
            for j, method in enumerate(("linear", "regularized")):
                admitted = arrays[f"{method}_inferred"]
                rejected = arrays[f"{method}_proposal_unresolved"]
                tagged.extend([(f"{method}-INFERRED", admitted, (218, 109, 36)),
                               (f"{method}-UNRESOLVED", rejected, (182, 54, 54))])
                ax = fig.add_subplot(len(sizes), 3, 3 * i + j + 1, projection="3d")
                plot_geometry(ax, arrays["observed"], origin, base, OBSERVED, 8, alpha=.65)
                plot_geometry(ax, arrays["ct_reference"], origin, base, REFERENCE, 10, alpha=.52)
                plot_geometry(ax, arrays["withheld_oracle"], origin, base, "#505461", 12, alpha=.7, marker="x")
                plot_stress_alternatives(ax, arrays, method, origin, base)
                plot_geometry(ax, admitted, origin, base, INFERRED, 10)
                plot_geometry(ax, rejected, origin, base, ERROR, 8, alpha=.65)
                short_status = ref["methods"][method]["status"].replace("_", " ").title()
                ax.set(title=f"{gap:g} mm · {METHOD_LABELS[method]}\n{short_status}",
                       xlim=(-7, 7), ylim=(-7, 7), zlim=(-6, 6), zlabel="Depth (mm)")
                ax.view_init(28, -62)
                ax.set_box_aspect((1, 1, .75))
                if i == len(sizes) - 1:
                    ax.set(xlabel="Right (mm)", ylabel="Superior (mm)")
            ax = fig.add_subplot(len(sizes), 3, 3 * i + 3)
            actual = arrays["withheld_oracle"]
            local = (actual - origin) @ base
            inferred = arrays["regularized_inferred"]
            if len(inferred):
                error = distances(actual, inferred)
                image = ax.scatter(local[:, 0], local[:, 1], s=35, c=error, vmin=0, vmax=3, cmap="YlOrRd")
                fig.colorbar(image, ax=ax, shrink=.7, label="Distance (mm)")
            else:
                ax.scatter(local[:, 0], local[:, 1], s=24, c=REFERENCE, marker="x")
                ax.text(.5, .11, "No accepted fit", transform=ax.transAxes, ha="center", color=ERROR)
            ax.add_patch(plt.Circle((0, 0), gap / 2, fill=False, linestyle=":", edgecolor=ERROR))
            ax.set(title=f"Withheld-sample error\n{ref['session']} · {ref['identity'].replace('RIGHT_L5_', 'L5 ')}",
                   xlim=(-5, 5), ylim=(-5, 5), aspect="equal", xlabel="Right (mm)", ylabel="Superior (mm)")
            for layer, geometry, color in tagged:
                if len(geometry):
                    export_ply(exports / f"{key}-{layer}.ply", geometry, color)
        keys = [Line2D([], [], marker=marker, linestyle="none", color=color, label=label)
                for marker, color, label in (("o", OBSERVED, "Input observations"),
                                             ("o", INFERRED, "Accepted inferred samples"),
                                             ("o", "#f7b77b", "Alternative fit"),
                                             ("x", "#505461", "Withheld ultrasound"),
                                             ("o", REFERENCE, "CT · evaluator only"),
                                             ("o", ERROR, "Unresolved proposal"))]
        fig.legend(handles=keys, loc="outside lower center", frameon=False, ncol=3)
        fig.suptitle(f"{ARM_LABELS[mode]} · local cortical shape completion")
        fig.savefig(output / f"controlled-patches-{mode}.png", dpi=160)
        plt.close(fig)


def actual_gap(root, output):
    records = read(root / "actual_gap_results.json")
    sessions = list(dict.fromkeys(name.rsplit("-", 1)[0] for name in records))
    fig = plt.figure(figsize=(17, 8.5))
    fig.subplots_adjust(left=.025, right=.98, bottom=.125, top=.875, wspace=.1, hspace=.32)
    for j, mode in enumerate(ARM_LABELS):
        for i, session in enumerate(sessions):
            key = f"{session}-{mode}"
            record = records[key]
            data = load_cloud(root / "clouds" / f"{key}-layers.npz")
            chart = load_cloud(root / "inputs" / f"{key}.npz")["basis"]
            trace = data["diagnostic_trace"]
            origin = trace.mean(0)
            ax = fig.add_subplot(2, len(sessions), j * len(sessions) + i + 1, projection="3d")
            plot_geometry(ax, data["ct_reference"], origin, chart, REFERENCE, 4, alpha=.32)
            near = data["eligible"] & (np.linalg.norm(data["observed"] - origin, axis=1) <= 13)
            points, identities = data["observed"][near], data["identity"][near]
            for semantic, marker in (("RIGHT_L5_TP", "^"), ("RIGHT_L5_SAP", "s"), ("OTHER_BONE", ".")):
                plot_geometry(ax, points[identities == semantic], origin, chart, OBSERVED, 8, marker=marker, alpha=.7)
            plot_geometry(ax, trace, origin, chart, ERROR, 28, marker="x")
            for m in ("linear", "regularized"):
                plot_geometry(ax, data[f"{m}_inferred"], origin, chart, INFERRED, 9)
                plot_geometry(ax, data[f"{m}_unresolved"], origin, chart, ERROR, 5, alpha=.4)
            hits = record["observed_trace_coverage"]["within_tolerance"]
            ways = 126 if session.endswith("-126") else 102
            n_fits = sum(m["status"] == "INFERRED" for m in record["methods"].values())
            mode_name = "Oracle" if mode == "oracle" else ARM_LABELS[mode]
            ax.set_title(f"{session.split('-')[0]} · {ways} frames\n"
                         f"{mode_name} · {hits}/7 trace · {n_fits}/2 fits", fontsize=10, pad=3)
            ax.set(xlim=(-11, 11), ylim=(-11, 11), zlim=(-11, 11),
                   xticks=(-10, 0, 10), yticks=(-10, 0, 10), zticks=(-10, 0, 10))
            ax.tick_params(labelsize=9, pad=1)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(23, -58)
    items = [(OBSERVED, "^", "Observed TP"), (OBSERVED, "s", "Observed SAP"),
             (OBSERVED, ".", "Observed other bone"), (INFERRED, "o", "Accepted inferred samples"),
             (REFERENCE, ".", "CT · evaluator only"), (ERROR, "x", "Diagnostic CT trace"),
             (ERROR, ".", "Unresolved proposal")]
    fig.legend(handles=[Line2D([], [], marker=marker, color=color, linestyle="none", label=label)
                        for color, marker, label in items], loc="lower center", bbox_to_anchor=(.5, .026),
               ncol=4, frameon=False, fontsize=10)
    fig.suptitle("Actual right L5 SAP–TP cortical gap · no evaluable anatomical target", y=.965, fontsize=17)
    fig.text(.5, .919, "Right, superior and depth in local millimetres. Trace counts use a 1.5 mm tolerance.",
             ha="center", fontsize=11)
    fig.savefig(output / "actual-gap.png", dpi=160)
    plt.close(fig)


def ancillary_geometry(root, output, protocol):
    records = read(root / "actual_gap_results.json")
    receipts = read(root / "source_receipts.json")
    result = {}
    for name, row in records.items():
        flanks = row["input_diagnostics"].get("flank_points_mm")
        if not flanks:
            result[name] = {"status": "NO_SEMANTIC_FLANKS"}
            continue
        inp = load_cloud(root / "inputs" / f"{name}.npz")
        xyz = inp["xyz_mm"]
        result[name] = {}
        for identity, anchor in zip(("RIGHT_L5_TP", "RIGHT_L5_SAP"), np.asarray(flanks)):
            within = np.linalg.norm(xyz - anchor, axis=1) <= protocol["actual"]["anchor_radius_mm"]
            keep = (inp["level_hint"] == "L5") & (inp["identity"] == identity) & within
            result[name][identity] = describe(xyz[keep], inp["view_group"][keep], anchor,
                                               inp["probe_mm"], protocol["fit"])

    import json

    def public_rows(session):
        ledger = receipts[session]["pose_ledger"]
        path = Path(ledger["path"])
        if digest(path) != ledger["sha256"]:
            raise ValueError("modified executed pose ledger")
        return [json.loads(line) for line in path.read_text().splitlines()][:receipts[session]["frame_count"]]

    previous, fresh = public_rows("2026121700"), public_rows("2026121700-126")
    pairs = list(zip(previous, fresh))
    hash_matches = sum(a["frame_sha256"] == b["frame_sha256"] for a, b in pairs)
    pose_keys = ("actor_base_to_ee_position_m", "actor_base_to_ee_rotation_columns")
    pose_matches = sum(all(a[key] == b[key] for key in pose_keys) for a, b in pairs)
    dump(output / "observed_flank_geometry.json", result)
    dump(output / "paired_acquisition_check.json",
         {"matched_prefix_frames": len(pairs), "identical_image_sha256": hash_matches,
          "identical_public_executed_poses": pose_matches, "additional_oblique_frames": len(fresh) - len(previous)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = benchmark_summary(args.benchmark, args.output)
    benchmark_examples(args.benchmark, args.output, results)
    actual_gap(args.actual, args.output)
    ancillary_geometry(args.actual, args.output, read(args.protocol))
