"""Replay three sonographic arms on synchronized B-mode and interface pixels."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np

from .localize import (
    COMPETING, SAP, TP, acquisition_views, audit_frame, observed_profiles,
    region_distance, search_quality, select_views, verify_session,
)
from .prepare import PIXEL_FIELDS, POSE_FIELDS, digest, read_json, verify, write_json

NAMES = (TP, SAP, *COMPETING, "OTHER_BONE", "UNCERTAIN_BONE")
ARMS = {"A0": "oracle", "A1": "oracle", "A2": "learned"}


def load_inputs(root, session, expected):
    source = root / session
    for name, sha in expected["staged_sha256"].items():
        verify(source / name, sha)
    frames = read_json(source / "public_poses.json")
    if not all(set(frame) == set(POSE_FIELDS) for frame in frames):
        raise ValueError("unexpected public pose fields")
    with np.load(source / "bmode.npz", allow_pickle=False) as data:
        if data.files != ["ultrasound"]:
            raise ValueError("unexpected B-mode fields")
        images = data["ultrasound"]
    return frames, images


def load_annotations(path, count):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(PIXEL_FIELDS):
            raise ValueError("only visible 2D evidence may enter the estimator")
        sequences, pixels, identities, confidence = (data[key] for key in PIXEL_FIELDS)
    return [
        {name: (pixels[mask], confidence[mask])
         for name in NAMES
         for mask in [(sequences == i) & (identities == name)]}
        for i in range(count)
    ]


def write_table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_sessions(results, rules):
    left, right, _ = rules["shared_prefix"]
    pairs = []
    for arm, values in results.items():
        for (a, former), (b, later) in combinations(values.items(), 2):
            shared = {a, b} == {left, right}
            extended = right in (a, b)
            if former["accepted"] and later["accepted"]:
                distance = region_distance(former["region_xyz_mm"], later["region_xyz_mm"])
            else:
                distance = None
            pairs.append({
                "arm": arm, "session_a": a, "session_b": b,
                "comparison": "shared_prefix" if shared else "extended_vs_other" if extended else "distinct_102",
                "bidirectional_p95_mm": distance,
                "both_accepted": former["accepted"] and later["accepted"],
            })
    return pairs


def run(inputs: Path, config: Path, output: Path) -> None:
    started = perf_counter()
    manifest = read_json(inputs / "manifest.json")
    verify(config, manifest["configuration_sha256"])
    rules = read_json(config)
    output.mkdir(parents=True, exist_ok=False)
    frame_table, session_table = [], []
    results = {arm: {} for arm in ARMS}
    all_audits = {}
    for session in rules["recordings"]:
        count_source = manifest["recordings"][session]
        frames, images = load_inputs(inputs, session, count_source)
        views, axis_residual = acquisition_views(frames, rules)
        start_tracking = perf_counter()
        profiles = [observed_profiles(image, rules) for image in images]
        tracking_runtime = perf_counter() - start_tracking
        evidence = {}
        # The actor-only shortlist and A2 outputs are built before Oracle pixels are loaded.
        selected, rankings = None, None
        for perception in ("learned", "oracle"):
            pixel_path = inputs / session / f"{perception}_pixels.npz"
            annotations = load_annotations(pixel_path, len(frames))
            rows = []
            for i, ((family, sign), samples) in enumerate(zip(views, annotations)):
                row = audit_frame(profiles[i], samples, family, sign, rules)
                row.update(
                    session=session, sequence=i, frame_sha256=frames[i]["frame_sha256"],
                    planned_view=frames[i]["phase"], view_quality=search_quality(row, rules),
                    semantic_pixels={name: values[0].tolist() for name, values in samples.items()},
                    semantic_confidence={name: values[1].tolist() for name, values in samples.items()},
                )
                rows.append(row)
            evidence[perception] = rows
            if perception == "learned":
                start_selection = perf_counter()
                selected, rankings = select_views(frames, rows, rules)
                selection_runtime = perf_counter() - start_selection
                start_projection = perf_counter()
                prediction = verify_session(frames, rows, selected, rules)
                prediction["projection_and_verification_seconds"] = perf_counter() - start_projection
                prediction["selected_frames"] = selected
                results["A2"][session] = prediction
        all_audits[session] = {
            "public_axis_residual_index": axis_residual, "public_poses": frames, "frames": evidence,
            "shared_actor_shortlist": selected, "learned_neighbor_adjusted_quality": rankings,
            "tracking_time_seconds": tracking_runtime, "selection_time_seconds": selection_runtime,
        }
        selection = {"A0": list(range(len(frames))), "A1": selected, "A2": selected}
        for arm, perception in ARMS.items():
            rows = evidence[perception]
            selected_indices = selection[arm]
            if arm != "A2":
                start_projection = perf_counter()
                prediction = verify_session(frames, rows, selected_indices, rules)
                prediction["projection_and_verification_seconds"] = perf_counter() - start_projection
                prediction["selected_frames"] = selected_indices
                results[arm][session] = prediction
            prediction = results[arm][session]
            statuses = Counter(rows[i]["status"] for i in selected_indices)
            n_long = sum(rows[i]["view"] == "longitudinal" for i in selected_indices)
            n_trans = sum(rows[i]["view"] == "transverse" for i in selected_indices)
            detected = statuses["DIRECT_IMAGE_CANDIDATE"]
            context = statuses["LONGITUDINAL_CONTEXT"]
            session_table.append({
                "session": session, "arm": arm, "perception": perception, "eligible_frames": len(frames),
                "selected_frames": len(selected_indices), "selected_longitudinal": n_long,
                "usable_longitudinal": context, "selected_transverse": n_trans,
                "usable_transverse": detected, "localized_2d": detected,
                "selected_rejected": len(selected_indices) - context - detected,
                "not_selected": len(frames) - len(selected_indices),
                "accepted": prediction["accepted"], "status": prediction["status"],
                "transverse_support_frames": len(prediction["support_frames"]),
                "region_diameter_mm": prediction["diameter_mm"],
                "within_session_max_agreement_p95_mm": prediction["max_disagreement_mm"],
                "bmode_track_seconds": tracking_runtime,
                "frame_audit_seconds": sum(row["runtime_seconds"] for row in rows),
                "selection_seconds": selection_runtime if arm != "A0" else 0.,
                "projection_and_verification_seconds": prediction["projection_and_verification_seconds"],
            })
        shortlisted = set(selected)
        for perception, rows in evidence.items():
            for row in rows:
                visible = {key: value for key, value in row.items()
                           if key not in ("semantic_pixels", "semantic_confidence", "profile_pixels",
                                          "candidate_pixels", "support_pixels", "observation_ambiguity")}
                frame_table.append({
                    **visible, "shared_search_selected": row["sequence"] in shortlisted,
                    "actor_neighbor_adjusted_quality": rankings[row["sequence"]],
                })
    # No evaluator or subject-specific 3D geometry is loaded by this module.
    write_json(output / "predictions.json", results)
    write_json(output / "frame_audits.json", all_audits)
    write_table(output / "frames.csv", frame_table)
    write_table(output / "sessions.csv", session_table)
    write_table(output / "inter_session.csv", compare_sessions(results, rules))
    write_json(output / "runtime.json", {"end_to_end_replay_seconds": perf_counter() - started})
    sealed = ("predictions.json", "frame_audits.json", "frames.csv", "sessions.csv", "inter_session.csv")
    write_json(output / "prediction_seal.json", {
        "configuration_sha256": digest(config),
        "public_input_manifest_sha256": digest(inputs / "manifest.json"),
        "source_sha256": {name: digest(Path(__file__).with_name(name))
                          for name in ("localize.py", "prepare.py", "replay.py")},
        "output_sha256": {name: digest(output / name) for name in sealed},
        "independent_reference_opened": False,
    })
    for arm, cohorts in results.items():
        print(arm, sum(p["accepted"] for p in cohorts.values()), "/", len(cohorts), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--inputs", required=True, type=Path)
    cli.add_argument("--config", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    run(args.inputs, args.config, args.output)
