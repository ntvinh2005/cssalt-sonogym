"""Render recorded B-mode, candidate peaks and sealed echo paths for review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from itertools import groupby

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from workflows.mbb_direct_sonographic.localize import SAP, TP
from workflows.mbb_direct_sonographic.prepare import read_json, verify

ARM_TITLES = {"A0": "A0 Fixed window", "A1": "A1 Adaptive · 0 mm", "A2_2mm": "A2 Adaptive · 2 mm"}


def scatter(ax, pixels, color, size, marker="o"):
    if len(pixels):
        xy = .5 * np.asarray(pixels)
        ax.scatter(xy[:, 1], xy[:, 0], color=color, s=size, marker=marker, linewidths=.6 if marker == "x" else 0)


def bmode(ax, image, bounds):
    lo, hi = np.percentile(image, [1., 99.5])
    height, width = image.shape
    ax.imshow(image, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest",
              extent=(-.25, (width - .5) / 2, (height - .5) / 2, -.25))
    left, right, top, bottom = bounds
    ax.set(xlim=(left, right), ylim=(bottom, top), xlabel="Image columns (mm)", ylabel="Depth (mm)")
    ax.tick_params(labelsize=7)
    x = left + .07 * (right - left)
    y = bottom - .08 * (bottom - top)
    ax.plot([x, x + 5], [y, y], color="white", lw=2)
    ax.text(x, y - .03 * (bottom - top), "5 mm", color="white", fontsize=7)


def draw_brackets(ax, rows, arm):
    for b in rows:
        cols = b["columns"]
        samples = [[candidate[0], col] for col, candidates in zip(cols, b["qualified_candidates"])
                   for candidate in candidates]
        scatter(ax, samples, "#9fa8b7", 9)
        prefix, suffix = b["prefix_rows"], b["suffix_rows"] or [None] * len(cols)
        once = [r if r is not None else s for r, s in zip(prefix, suffix)]
        if arm == "A0":
            scatter(ax, [[row, col] for col, row, supported in zip(cols, b["baseline_rows"], b["baseline_observed"])
                         if not supported], "#f68ebd", 12, "x")
        scatter(ax, [[row, col] for col, row in zip(cols, prefix) if row is not None], "#ffe06a", 19)
        scatter(ax, [[row, col] for col, row in zip(cols, suffix) if row is not None], "#50d3ed", 16)
        if b["alternate_rows"]:
            alternate = [[row, col] for col, row in zip(cols, b["alternate_rows"]) if row is not None]
            scatter(ax, alternate, "#bb87fb" if b["status"] == "AMBIGUOUS" else "#b491a9", 22, "x")
        for missing, group in groupby(zip(cols, once), key=lambda col_row: col_row[1] is None):
            if missing:
                numbers = [pair[0] for pair in group]
                ax.axvspan((min(numbers) - .5) * .5, (max(numbers) + .5) * .5, ymin=.92, ymax=.99,
                           color="#ce3f68", alpha=.85)


def render_case(image, old, paths, estimates, title, output):
    pixels = []
    for perception in ("oracle", "learned"):
        for identity in (TP, SAP):
            pixels.extend(old[perception]["semantic_pixels"][identity])
    if pixels:
        low, high = np.asarray(pixels).min(0) * .5, np.asarray(pixels).max(0) * .5
        bounds = (max(0., low[1] - 8), min(75., high[1] + 8),
                  max(0., low[0] - 9), min(100., high[0] + 9))
    else:
        bounds = (12, 66, 35, 85)
    fig, axes = plt.subplots(2, 4, figsize=(15.4, 7.6), layout="constrained")
    for r, perception in enumerate(("oracle", "learned")):
        bmode(axes[r, 0], image, bounds)
        axes[r, 0].set_title(f"{perception.title()} · raw B-mode", fontsize=10)
        for col, arm in enumerate(ARM_TITLES, 1):
            ax = axes[r, col]
            bmode(ax, image, bounds)
            for identity, color in ((TP, "#49e394"), (SAP, "#70bcff")):
                scatter(ax, old[perception]["semantic_pixels"][identity], color, 12)
            active = paths.get((perception, arm), [])
            if active:
                draw_brackets(ax, active, arm)
            else:
                row = old[perception]
                scatter(ax, row["profile_pixels"], "#ffe06a", 8)
            result = estimates[(perception, arm)]
            candidate = result["candidate_pixels"]
            if candidate:
                scatter(ax, json.loads(candidate), "#ff953e", 38, "s")
            diagnostic = result["status"].replace("_", " ").title()
            if len(diagnostic) > 30:
                split = diagnostic.rfind(" ", 0, 28)
                diagnostic = diagnostic[:split] + "\n" + diagnostic[split + 1:]
            counts = ", ".join(f"{b['supported']}/{b['total']}" for b in active)
            count_text = f"\nObserved columns {counts}" if counts else ""
            ax.set_title(f"{ARM_TITLES[arm]}\n{diagnostic}{count_text}", fontsize=9)
    fig.suptitle(title + "\nGreen TP · blue SAP · gray qualifying peaks · yellow path · cyan separate end · red missing · orange region",
                 fontsize=13, weight="bold")
    fig.savefig(output, dpi=175, facecolor="white")
    plt.close(fig)


def review(result, prior, config, output):
    source = prior / "frozen-run"
    seal = read_json(result / "actor_seal.json")
    for name, checksum in seal["output_sha256"].items():
        verify(result / name, checksum)
    verify(config, seal["adaptive_config_sha256"])
    config = read_json(config)
    old = read_json(source / "replay/frame_audits.json")
    with (result / "frames.csv").open() as stream:
        estimates = {(item["session"], int(item["sequence"]), item["perception"], item["arm"]): item
                     for item in csv.DictReader(stream)}
    brackets = read_json(result / "brackets.json")
    records = [(f"R{i:02d}", session, n) for i, (session, n) in enumerate(config["review_records"], 1)]
    learned_only = sorted({
        (b["session"], b["sequence"]) for b in brackets if b["perception"] == "learned"
        and estimates[(b["session"], b["sequence"], "oracle", "A0")]["candidate_brackets"] == "0"
    })
    records.extend((f"Learned-only {i}", s, n) for i, (s, n) in enumerate(learned_only, 1))
    output.mkdir(parents=True, exist_ok=False)
    images = {}
    for ident, session, n in records:
        if session not in images:
            with np.load(source / "inputs" / session / "bmode.npz", allow_pickle=False) as data:
                images[session] = data["ultrasound"]
        current = {perception: old[session]["frames"][perception][n] for perception in ("oracle", "learned")}
        paths = {(perception, arm): [b for b in brackets if b["session"] == session and b["sequence"] == n
                                    and b["perception"] == perception and b["arm"] == arm]
                 for perception in current for arm in ARM_TITLES}
        decision = {(perception, arm): estimates[(session, n, perception, arm)]
                    for perception in current for arm in ARM_TITLES}
        render_case(images[session][n], current, paths, decision, f"{ident} · {session} · frame {n:03d}",
                    output / f"{ident.replace(' ', '_')}_{session}_{n:03d}.png")
    if (result / "bracket_evaluation.csv").exists():
        with (result / "bracket_evaluation.csv").open() as stream:
            crossings = [r for r in csv.DictReader(stream) if r["joined_path_crosses_competing_vertebra"] == "True"]
    else:
        crossings = []
    examples = []
    if crossings:
        c = crossings[0]
        examples.append((c["session"], int(c["sequence"]), c["perception"], c["arm"], "Competing first-visible vertebra"))
    ambiguity = next((b for b in brackets if b["status"] == "AMBIGUOUS"), None)
    if ambiguity:
        examples.append((ambiguity["session"], ambiguity["sequence"], ambiguity["perception"], ambiguity["arm"],
                         "Two comparable paths"))
    if not examples:
        examples = [(s, n, "learned", "A2_2mm", "Learned anchors without a compatible Oracle bracket")
                    for s, n in learned_only]
    if examples:
        fig, axes = plt.subplots(1, len(examples), figsize=(6 * len(examples), 5), layout="constrained", squeeze=False)
        for ax, (s, n, perception, arm, title) in zip(axes.ravel(), examples):
            bmode(ax, images[s][n], (20, 64, 46, 76))
            match = [b for b in brackets if (b["session"], b["sequence"], b["perception"], b["arm"]) == (s, n, perception, arm)]
            draw_brackets(ax, match, arm)
            ax.set_title(f"{title}\n{s} · frame {n:03d} · {arm}", fontsize=10)
        fig.suptitle("Examples requiring anatomical review", weight="bold")
        fig.savefig(output / "tracking_risks.png", dpi=175, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("result", "prior", "config", "output"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    review(*vars(cli.parse_args()).values())
