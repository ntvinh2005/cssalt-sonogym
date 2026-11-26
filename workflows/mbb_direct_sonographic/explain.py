"""Attribute failed continuity checks using the original sealed frame profiles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .localize import (
    COMPETING, SAP, TP, acquisition_views, contiguous_runs, observed_profiles, profile_evidence,
)
from .prepare import read_json, verify
from .replay import load_annotations


def ordered_bracket_failures(profile, annotations, sign, rules):
    pixels, supported, _ = profile
    labels, _ = profile_evidence(pixels, supported, annotations, rules)
    if sign == 1:
        pixels, supported = pixels[::-1], supported[::-1]
        labels = {name: flags[::-1] for name, flags in labels.items()}
    tp = contiguous_runs(labels[TP], rules["anchor_span_px"])
    sap = contiguous_runs(labels[SAP], rules["anchor_span_px"])
    competing = np.logical_or.reduce([labels[name] for name in COMPETING])
    height = pixels[:, 0].astype(float)
    flank = rules["turn_window_px"]
    spacing = rules["pixel_mm"]
    rows = []
    for a in tp:
        for b in sap:
            start, stop = int(a[-1]), int(b[0])
            lo, hi = start - flank + 1, stop + flank
            rise = float((np.median(height[a[-flank:]]) - np.median(height[b[:flank]])) * spacing)
            span = (stop - start) * spacing
            if (stop <= start or lo < 0 or hi > len(height)
                    or span > rules["max_anchor_span_mm"] or rise < rules["min_anchor_depth_rise_mm"]):
                continue
            bridge = slice(lo, hi)
            zones = contiguous_runs(~supported[bridge], 1)
            missing = int((~supported[bridge]).sum())
            competing_columns = int(competing[bridge].sum())
            jump = float(np.max(np.abs(np.diff(height[bridge]))))
            rows.append({
                "tp_anchor_col": int(pixels[start, 1]), "sap_anchor_col": int(pixels[stop, 1]),
                "horizontal_anchor_span_mm": span, "anchor_rise_mm": rise,
                "supported_columns": int(supported[bridge].sum()), "total_columns": hi - lo,
                "unsupported_columns": missing,
                "longest_unsupported_run_mm": max((len(z) for z in zones), default=0) * spacing,
                "competing_l4_columns": competing_columns, "max_column_jump_px": jump,
                "passes": missing == 0 and competing_columns == 0 and jump <= rules["max_rise_per_column_px"],
            })
    return rows


def explain(inputs, replay, config, output):
    seal = read_json(replay / "prediction_seal.json")
    verify(config, seal["configuration_sha256"])
    verify(replay / "frame_audits.json", seal["output_sha256"]["frame_audits.json"])
    verify(inputs / "manifest.json", seal["public_input_manifest_sha256"])
    audits = read_json(replay / "frame_audits.json")
    rules = read_json(config)
    manifest = read_json(inputs / "manifest.json")
    rows = []
    for session, data in audits.items():
        source = inputs / session
        frames = data["public_poses"]
        views, _ = acquisition_views(frames, rules)
        assets = manifest["recordings"][session]["staged_sha256"]
        with np.load(source / "bmode.npz", allow_pickle=False) as raw:
            verify(source / "bmode.npz", assets["bmode.npz"])
            images = raw["ultrasound"]
        for perception, evidence in data["frames"].items():
            annotation_file = source / f"{perception}_pixels.npz"
            verify(annotation_file, assets[annotation_file.name])
            annotations = load_annotations(annotation_file, len(frames))
            for i, frame in enumerate(evidence):
                if not frame["ordered_pairs"]:
                    continue
                result = []
                for index, profile in enumerate(observed_profiles(images[i], rules)):
                    result.extend({
                        "session": session, "perception": perception, "frame_sequence": i,
                        "frame_sha256": frame["frame_sha256"], "profile": index, **value,
                    } for value in ordered_bracket_failures(profile, annotations[i], views[i][1], rules))
                if (len(result) != frame["ordered_pairs"]
                        or sum(item["passes"] for item in result) != frame["continuous_pairs"]):
                    raise ValueError("failure audit disagrees with the frozen replay")
                rows.extend(result)
    if not rows:
        raise ValueError("no ordered profiles require continuity attribution")
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--inputs", required=True, type=Path)
    cli.add_argument("--replay", required=True, type=Path)
    cli.add_argument("--config", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    explain(args.inputs, args.replay, args.config, args.output)
