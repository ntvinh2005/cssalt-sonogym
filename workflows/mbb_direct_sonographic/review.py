"""Render the seven scientific review figures from sealed replay outputs."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import math
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial.distance import cdist

from .localize import SAP, TP
from .prepare import read_json, verify, write_json
from .replay import ARMS

COLORS = {TP: "#4ce697", SAP: "#80c7ff", "echo": "#f4e268", "candidate": "#ff9c4e"}
ARM_LABELS = {"A0": "A0  Oracle · all views", "A1": "A1  Oracle · shortlisted",
              "A2": "A2  U-Net · same shortlist"}


def save(fig, folder, name):
    fig.savefig(folder / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ultrasound(ax, image, evidence, spacing, label, overlay=True, zoom=False):
    height, width = image.shape
    low, high = np.percentile(image, [1, 99.5])
    ax.imshow(image, cmap="gray", vmin=low, vmax=max(high, low + 1e-6), interpolation="nearest",
              extent=(-.5 * spacing, (width - .5) * spacing, (height - .5) * spacing, -.5 * spacing))
    if overlay:
        groups = [(evidence["profile_pixels"], "echo", 5, .65),
                  (evidence["semantic_pixels"][TP], TP, 13, .85),
                  (evidence["semantic_pixels"][SAP], SAP, 13, .85),
                  (evidence["candidate_pixels"], "candidate", 30, 1.)]
        for points, name, size, alpha in groups:
            if not points:
                continue
            positions = np.asarray(points)
            ax.scatter(positions[:, 1] * spacing, positions[:, 0] * spacing, s=size,
                       color=COLORS[name], marker="o", linewidths=0, alpha=alpha)
        for identity in (TP, SAP):
            points = evidence["semantic_pixels"][identity]
            if points:
                row, col = np.median(points, axis=0) * spacing
                ax.annotate(identity.rsplit("_", 1)[-1], (col, row), xytext=(2, -6),
                            textcoords="offset points", color=COLORS[identity],
                            fontsize=8, weight="bold")
    if zoom:
        points = evidence["semantic_pixels"][TP] + evidence["semantic_pixels"][SAP]
        if points:
            positions = np.asarray(points) * spacing
            ax.set_xlim(max(0, positions[:, 1].min() - 8), min(width * spacing, positions[:, 1].max() + 8))
            ax.set_ylim(min(height * spacing, positions[:, 0].max() + 7), max(0, positions[:, 0].min() - 9))
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + .06 * (x1 - x0)
    y = y0 - .1 * (y0 - y1)
    ax.plot([x, x + 5], [y, y], color="white", lw=2)
    ax.text(x, y - .025 * (y0 - y1), "5 mm", color="white", fontsize=7)
    identifier = f"{evidence['session']} · frame {evidence['sequence']:03d} · {evidence['planned_view']}"
    outcome = evidence["status"].replace("_", " ").capitalize()
    ax.set_title(f"{label}\n{identifier}\n{fill(outcome, 36)}", fontsize=8.5)
    ax.set_xlabel("Image columns (mm)", fontsize=8)
    ax.set_ylabel("Image depth (mm)", fontsize=8)
    ax.tick_params(labelsize=7)


def legend(fig):
    captions = {TP: "Right L5 TP evidence", SAP: "Right L5 SAP evidence",
                "echo": "Measured echo profile", "candidate": "Direct image candidate"}
    items = [Line2D([], [], linestyle="", marker="o", color=COLORS[name], label=caption)
             for name, caption in captions.items()]
    fig.legend(handles=items, loc="outside lower center", ncol=4, frameon=False, fontsize=8)


def quality(row):
    return (row["direct_candidate"], row["continuous_pairs"] > 0, row["ordered_pairs"] > 0,
            row["view_quality"], min(row["tp_columns"], row["sap_columns"]))


def workflow(folder):
    fig, ax = plt.subplots(figsize=(12, 2.5), layout="constrained")
    ax.axis("off")
    titles = (
        ("Stage 1", "Reuse cf7_00 reference"),
        ("Longitudinal context", "Image-supported L5 TP"),
        ("Transverse search", "Rank recorded views with U-Net\nand measured B-mode"),
        ("Direct image region", "Visible TP and SAP on one\nmeasured cortical profile"),
        ("Calibrated verification", "Compare distinct poses\nin robot-base mm"),
    )
    for i, (title, detail) in enumerate(titles):
        x = .10 + i * .20
        ax.text(x, .70, title, transform=ax.transAxes, ha="center", weight="bold", fontsize=10)
        ax.text(x, .49, detail, transform=ax.transAxes, ha="center", fontsize=9)
        if i < len(titles) - 1:
            ax.annotate("", (x + .128, .69), (x + .074, .69), xycoords=ax.transAxes,
                        arrowprops={"arrowstyle": "->", "color": "#466279"})
    ax.text(.5, .15, "Return only an image-supported bony region. Abstain if evidence is missing or disagrees.",
            transform=ax.transAxes, ha="center", fontsize=10)
    fig.suptitle("Figure 1. Physician-inspired sonographic localization in simulation", weight="bold")
    save(fig, folder, "01_workflow.png")


def render(inputs, replay, config, folder):
    seal = read_json(replay / "prediction_seal.json")
    verify(config, seal["configuration_sha256"])
    for name, sha in seal["output_sha256"].items():
        verify(replay / name, sha)
    verify(inputs / "manifest.json", seal["public_input_manifest_sha256"])
    manifest = read_json(inputs / "manifest.json")
    rules = read_json(config)
    audits = read_json(replay / "frame_audits.json")
    predictions = read_json(replay / "predictions.json")
    images = {}
    for identifier, data in manifest["recordings"].items():
        image_path = inputs / identifier / "bmode.npz"
        verify(image_path, data["staged_sha256"]["bmode.npz"])
        with np.load(image_path, allow_pickle=False) as raw:
            images[identifier] = raw["ultrasound"]
    folder.mkdir(parents=True, exist_ok=False)
    spacing = rules["pixel_mm"]
    write_json(folder.parent / "observability_summary.json", summarize(audits, predictions, rules))

    def show(ax, row, label, overlay=True, zoom=False):
        ultrasound(ax, images[row["session"]][row["sequence"]], row, spacing, label, overlay, zoom)

    workflow(folder)
    for number, family in ((2, "longitudinal"), (3, "transverse")):
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 8), layout="constrained")
        for k, perception in enumerate(("oracle", "learned")):
            rows = [r for a in audits.values() for r in a["frames"][perception]
                    if r["view"] == family]
            show(axes[k, 0], max(rows, key=quality), f"{perception.title()} · highest image evidence")
            show(axes[k, 1], min(rows, key=quality), f"{perception.title()} · lowest image evidence")
        fig.suptitle(f"Figure {number}. {family.capitalize()} views on original B-mode", weight="bold")
        legend(fig)
        save(fig, folder, f"0{number}_{family}.png")

    paired = [(a["frames"]["oracle"][i], a["frames"]["learned"][i])
              for a in audits.values() for i in a["shared_actor_shortlist"]
              if a["frames"]["oracle"][i]["view"] == "transverse"]
    best_oracle = max(paired, key=lambda p: quality(p[0]))
    best_learned = max(paired, key=lambda p: quality(p[1]))
    if best_oracle[0]["frame_sha256"] == best_learned[0]["frame_sha256"]:
        best_learned = max((p for p in paired if p[0]["frame_sha256"] != best_oracle[0]["frame_sha256"]),
                           key=lambda p: quality(p[1]))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    for k, (oracle, learned) in enumerate((best_oracle, best_learned)):
        show(axes[k, 0], oracle, "Matched raw B-mode · Oracle decision below", overlay=False)
        show(axes[k, 1], oracle, "Oracle-visible diagnostic")
        show(axes[k, 2], learned, "U-Net · same image and executed pose")
    fig.suptitle("Figure 4. Matched B-mode and semantic evidence on the shared shortlist", weight="bold")
    legend(fig)
    save(fig, folder, "04_matched_bmode_semantics.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    for ax, perception in zip(axes, ("oracle", "learned")):
        rows = [r for a in audits.values() for r in a["frames"][perception] if r["view"] == "transverse"]
        best = max(rows, key=quality)
        label = "Directly supported image region" if best["direct_candidate"] else "Closest evidence · no candidate"
        show(ax, best, f"{perception.title()} · {label}", zoom=True)
    fig.suptitle("Figure 5. Image-space localization and cortical support", weight="bold")
    legend(fig)
    save(fig, folder, "05_image_region.png")

    fig = plt.figure(figsize=(13.2, 5), layout="constrained")
    all_points = [
        point for sessions in predictions.values() for result in sessions.values()
        for region in result["regions"].values() for point in region["candidate_xyz_mm"]
    ]
    if all_points:
        coords = np.asarray(all_points)
        mid = (coords.min(0) + coords.max(0)) / 2
        radius = max(2.0, np.ptp(coords, axis=0).max() / 2 + .5)
    for j, (arm, name) in enumerate(ARM_LABELS.items(), start=1):
        ax = fig.add_subplot(1, 3, j, projection="3d")
        items = list(predictions[arm].values())
        projected = sum(len(p["regions"]) for p in items)
        accepted = sum(p["accepted"] for p in items)
        for k, (sid, result) in enumerate(predictions[arm].items()):
            color = plt.get_cmap("tab10")(k)
            for index, (sequence, region) in enumerate(result["regions"].items()):
                xyz = np.asarray(region["candidate_xyz_mm"])
                ax.scatter(*xyz.T, marker="o" if result["accepted"] else "x",
                           s=22, color=color, label=sid if index == 0 else None)
        if projected:
            ax.legend(fontsize=7, loc="upper right")
            for method, center in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
                method(center - radius, center + radius)
        else:
            ax.text2D(.5, .55, "No supported image region\nto project", transform=ax.transAxes,
                      ha="center", color="#8b2632", bbox={"facecolor": "white", "edgecolor": "none"})
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        ax.set_xlabel("Robot x (mm)", fontsize=7)
        ax.set_ylabel("Robot y (mm)", fontsize=7)
        ax.set_zlabel("Robot z (mm)", fontsize=7)
        ax.view_init(22, -57)
        ax.set_title(f"{name}\n{projected} projected views · {accepted}/5 accepted sessions", fontsize=9)
    fig.suptitle("Figure 6. Projected bony regions · × indicates an unresolved session", weight="bold")
    save(fig, folder, "06_projected_regions.png")

    failures = [r for a in audits.values() for r in a["frames"]["oracle"]
                if r["view"] == "transverse" and not r["direct_candidate"]]
    reasons = {
        "L5_SAP_NOT_SUPPORTED": "TP evidence without a supported SAP ridge",
        "L5_TP_NOT_SUPPORTED": "SAP evidence without a supported TP ridge",
        "NO_COMPATIBLE_ORDERED_PROFILE": "Both interfaces appear without one compatible local profile",
        "DISCONTINUOUS_OR_CONFLICTING_PROFILE": "Interrupted or competing signal within the profile",
        "NO_LOCAL_CORTICAL_TURN": "Continuous signal without a sufficiently distinct turn",
        "AMBIGUOUS_IMAGE_REGIONS": "Separated image interpretations",
        "NEITHER_L5_INTERFACE_SUPPORTED": "Neither named interface has sufficient support",
    }
    examples = [
        (max((r for r in failures if r["status"] == status), key=quality), explanation)
        for status, explanation in reasons.items() if any(r["status"] == status for r in failures)
    ]
    examples = examples[:4]
    if not examples:
        examples = [(best_oracle[0], "Review the actual transverse evidence for eligibility")]
    fig, axes = plt.subplots(math.ceil(len(examples) / 2), 2, figsize=(10, 4.5 * math.ceil(len(examples) / 2)),
                             layout="constrained", squeeze=False)
    for ax, (row, description) in zip(axes.flat, examples):
        show(ax, row, fill(description, 43), zoom=True)
    for ax in list(axes.flat)[len(examples):]:
        ax.axis("off")
    fig.suptitle("Figure 7. B-mode evidence for representative rejections", weight="bold")
    legend(fig)
    save(fig, folder, "07_representative_failures.png")


def summarize(audits, predictions, rules):
    """Describe the measured failure stages without re-running or selecting an arm."""
    from .replay import compare_sessions

    def counters(rows):
        transverse = [row for row in rows if row["view"] == "transverse"]
        minimum = rules["anchor_span_px"]
        result = {
            "frames": len(rows),
            "longitudinal": sum(row["view"] == "longitudinal" for row in rows),
            "transverse_compatible": len(transverse),
            "longitudinal_context": sum(row["status"] == "LONGITUDINAL_CONTEXT" for row in rows),
            "longitudinal_both_l4_l5_tp_context": sum(
                row["status"] == "LONGITUDINAL_CONTEXT" and row["l4_tp_columns"] >= minimum for row in rows
            ),
            "transverse_named_tp": sum(row["n_tp_pixels"] > 0 for row in transverse),
            "transverse_named_sap": sum(row["n_sap_pixels"] > 0 for row in transverse),
            "transverse_named_tp_and_sap": sum(
                row["n_tp_pixels"] > 0 and row["n_sap_pixels"] > 0 for row in transverse
            ),
            "transverse_bmode_tp": sum(row["tp_columns"] >= minimum for row in transverse),
            "transverse_bmode_sap": sum(row["sap_columns"] >= minimum for row in transverse),
            "transverse_bmode_both": sum(
                min(row["tp_columns"], row["sap_columns"]) >= minimum for row in transverse
            ),
            "transverse_same_profile": sum(row["same_profile_pairs"] > 0 for row in transverse),
            "transverse_ordered": sum(row["ordered_pairs"] > 0 for row in transverse),
            "transverse_continuous": sum(row["continuous_pairs"] > 0 for row in transverse),
            "transverse_direct": sum(row["direct_candidate"] for row in transverse),
        }
        result["transverse_reasons"] = dict(Counter(row["status"] for row in transverse))
        return result

    def summarize_overlaps(items):
        overlaps = []
        for a, b in combinations(items, 2):
            distances = cdist(items[a]["candidate_xyz_mm"], items[b]["candidate_xyz_mm"])
            overlaps.append({
                "a": a, "b": b,
                "a_fraction_within_one_pixel_of_b": float(np.mean(distances.min(axis=1) <= rules["pixel_mm"])),
                "b_fraction_within_one_pixel_of_a": float(np.mean(distances.min(axis=0) <= rules["pixel_mm"])),
            })
        return overlaps

    left, right, stop = rules["shared_prefix"]
    extension = audits[right]["frames"]
    pairs = compare_sessions(predictions, rules)
    matched = [
        (a["frames"]["oracle"][i], a["frames"]["learned"][i])
        for a in audits.values() for i in a["shared_actor_shortlist"]
        if a["frames"]["oracle"][i]["view"] == "transverse"
    ]
    minimum = rules["anchor_span_px"]
    matches = Counter(
        f"oracle_{int(min(o['tp_columns'], o['sap_columns']) >= minimum)}_"
        f"learned_{int(min(l['tp_columns'], l['sap_columns']) >= minimum)}"
        for o, l in matched
    )
    result = {
        "matched_transverse_shortlist_paired_bmode_tp_and_sap": dict(matches),
        "all_frame_diagnostics": {
            key: counters([row for a in audits.values() for row in a["frames"][key]])
            for key in ("oracle", "learned")
        },
        "recorded_frame_rows": sum(len(a["public_poses"]) for a in audits.values()),
        "distinct_bmode_hashes": len({frame["frame_sha256"] for a in audits.values()
                                     for frame in a["public_poses"]}),
        "extended_acquisition_only": {key: counters(rows[stop:]) for key, rows in extension.items()},
        "extended_prefix_status_agreement": {
            key: sum(a["status"] == b["status"] for a, b in zip(rows[:stop], audits[left]["frames"][key]))
            for key, rows in extension.items()
        },
        "arms": {}, "candidate_point_overlap_tolerance_mm": rules["pixel_mm"],
    }
    for arm, perception in ARMS.items():
        items = predictions[arm]
        selected = [
            audits[sid]["frames"][perception][i] for sid, session in items.items()
            for i in session["selected_frames"]
        ]
        distinct_102 = [pair["bidirectional_p95_mm"] for pair in pairs
                        if pair["arm"] == arm and pair["comparison"] == "distinct_102"
                        and pair["both_accepted"]]
        result["arms"][arm] = {
            "visible_evidence": counters(selected),
            "accepted_sessions": sum(item["accepted"] for item in items.values()),
            "distinct_102_pair_count": len(distinct_102),
            "inter_session_median_bidirectional_p95_mm": float(np.median(distinct_102)) if distinct_102 else None,
            "within_session_region_overlap": {
                sid: summarize_overlaps(item["regions"]) for sid, item in items.items()
            },
        }
    return result


def reference_audit(replay, audit_path, output):
    seal = read_json(replay / "prediction_seal.json")
    verify(replay / "predictions.json", seal["output_sha256"]["predictions.json"])
    sha = verify(audit_path, next(
        line.split()[0] for line in (audit_path.parent / "checksums.sha256").read_text().splitlines()
        if line.split()[-1] == audit_path.name))
    audit = read_json(audit_path)
    if audit["accepted_region"] is not None or audit["status"] != "NOT_EVALUABLE":
        raise ValueError("this protocol has no approved independent reference or scoring rule")
    write_json(output, {
        "status": "TARGET_LOCALIZATION_NOT_EVALUABLE",
        "independent_mbb_reference_available": False, "anatomical_proxy_error_mm": None,
        "audited_reference_sha256": sha, "audited_reference_path": str(audit_path),
        "reason": audit["reason"], "geometry_loaded": False,
        "prediction_seal_verified": True,
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--inputs", required=True, type=Path)
    cli.add_argument("--replay", required=True, type=Path)
    cli.add_argument("--config", required=True, type=Path)
    cli.add_argument("--figures", required=True, type=Path)
    cli.add_argument("--reference-audit", required=True, type=Path)
    args = cli.parse_args()
    render(args.inputs, args.replay, args.config, args.figures)
    reference_audit(args.replay, args.reference_audit, args.figures.parent / "reference_evaluation.json")
