"""Replay fixed-window and adaptive tracking on the same sealed recordings."""

from __future__ import annotations

import argparse
from itertools import groupby
from pathlib import Path
from time import perf_counter

import numpy as np

from workflows.mbb_direct_sonographic.localize import acquisition_views, audit_frame, observed_profiles, verify_session
from workflows.mbb_direct_sonographic.prepare import digest, read_json, verify, write_json
from workflows.mbb_direct_sonographic.replay import load_annotations, load_inputs, write_table
from two_stage_navigation.stage1_interfaces import track_bright_interfaces

from .tracker import adaptive_frame, anchor_brackets, candidates_for, echo_fields, track_path


def longest_gap(observed):
    return max((len(list(part)) for supported, part in groupby(observed) if not supported), default=0)


def bracket_record(session, frame, perception, arm, bracket_id, bracket, candidates, path, rules):
    original = bracket["pixels"][:, 0].tolist()
    baseline = bracket["baseline_support"].tolist()
    prefix = path["rows"]
    suffix = path.get("suffix_rows") or [None] * len(prefix)
    selected = [left if left is not None else right for left, right in zip(prefix, suffix)]
    observed = [value is not None for value in selected]
    steps = [abs(a - b) for a, b in zip(prefix, prefix[1:]) if a is not None and b is not None]
    steps += [abs(a - b) for a, b in zip(suffix, suffix[1:]) if a is not None and b is not None]
    return {
        "session": session, "sequence": frame["frame_sequence"], "frame_sha256": frame["frame_sha256"],
        "perception": perception, "arm": arm, "bracket": bracket_id, "profile": bracket["profile"],
        "columns": bracket["pixels"][:, 1].tolist(), "baseline_rows": original, "baseline_observed": baseline,
        "anchor_tp_index": bracket["start"], "anchor_sap_index": bracket["stop"],
        "status": path["status"], "prefix_rows": prefix, "suffix_rows": path.get("suffix_rows"),
        "alternate_rows": path.get("alternative_rows"), "cost": path.get("cost"),
        "alternate_cost": path.get("alternative_cost"), "qualified_candidates": candidates,
        "supported": sum(observed), "total": len(observed),
        "unsupported_run_mm": longest_gap(observed) * rules["pixel_mm"],
        "new_echoes": sum(now and not before for now, before in zip(observed, baseline)),
        "max_adjacent_depth_step_mm": max(steps, default=0) * rules["pixel_mm"],
    }


def run(source, frozen_config, adaptive_config, review_zip, destination):
    started = perf_counter()
    rules = read_json(frozen_config)
    config = read_json(adaptive_config)
    seal = read_json(source / "replay/prediction_seal.json")
    verify(frozen_config, config["baseline_configuration_sha256"])
    verify(frozen_config, seal["configuration_sha256"])
    for name, expected in seal["output_sha256"].items():
        verify(source / "replay" / name, expected)
    manifest_file = source / "inputs/manifest.json"
    verify(manifest_file, seal["public_input_manifest_sha256"])
    manifest = read_json(manifest_file)
    previous = read_json(source / "replay/frame_audits.json")
    previous_sessions = read_json(source / "replay/predictions.json")
    arms = {"A0": 0, "A1": 0, **{f"A2_{g:g}mm": round(g / rules["pixel_mm"])
                                  for g in config["gap_diagnostics_mm"]}}
    destination.mkdir(parents=True, exist_ok=False)
    frame_rows, bracket_rows, summaries = [], [], []
    session_regions = {}
    timing = {p: dict.fromkeys(arms, 0.) for p in ("oracle", "learned")}
    shared_tracking_seconds = 0.
    comparisons = 0
    for session, recording in manifest["recordings"].items():
        frames, images = load_inputs(source / "inputs", session, recording)
        views, _ = acquisition_views(frames, rules)
        mark = perf_counter()
        profiles = [observed_profiles(image, rules) for image in images]
        shared_tracking_seconds += perf_counter() - mark
        shortlist = previous[session]["shared_actor_shortlist"]
        for perception in ("learned", "oracle"):
            annotations = load_annotations(source / "inputs" / session / f"{perception}_pixels.npz", len(frames))
            evidence = {arm: [] for arm in arms}
            for i, ((family, sign), named) in enumerate(zip(views, annotations)):
                base = audit_frame(profiles[i], named, family, sign, rules)
                archived = previous[session]["frames"][perception][i]
                assert {k: v for k, v in base.items() if k != "runtime_seconds"} == {
                    k: archived[k] for k in base if k != "runtime_seconds"
                }, (session, perception, i)
                comparisons += 1
                timing[perception]["A0"] += base["runtime_seconds"]
                brackets = list(anchor_brackets(profiles[i], named, sign, rules)) if family == "transverse" else []
                assert len(brackets) == base["ordered_pairs"], (session, perception, i)
                field = echo_fields(images[i], rules) if brackets else None
                seeds = track_bright_interfaces(images[i]) if brackets else []
                candidates = [candidates_for(b, seeds[b["profile"]], field, named, rules, config) for b in brackets]
                for arm, allowance in arms.items():
                    start_arm = perf_counter()
                    paths = []
                    for j, b in enumerate(brackets):
                        if arm == "A0":
                            values = [int(row) if good else None for (row, _), good in zip(b["pixels"], b["baseline_support"])]
                            continuous = (all(b["baseline_support"]) and not b["competing_on_original"]
                                          and np.abs(np.diff(b["pixels"][:, 0])).max() <= rules["max_rise_per_column_px"])
                            result = {"status": "JOINED" if continuous else "UNSUPPORTED", "rows": values}
                        elif b["competing_on_original"]:
                            result = {"status": "COMPETING", "rows": [None] * len(b["pixels"])}
                        else:
                            pinned = [c <= b["start"] or c >= b["stop"] for c in range(len(b["pixels"]))]
                            result = track_path(candidates[j][1], allowance, pinned,
                                                rules["max_rise_per_column_px"], config, rules["pixel_mm"])
                        paths.append(result)
                        bracket_rows.append(bracket_record(session, frames[i], perception, arm, j, b,
                                                           candidates[j][0], result, rules))
                    estimate = base if arm == "A0" else adaptive_frame(base, brackets, paths, rules)
                    if arm != "A0":
                        timing[perception][arm] += perf_counter() - start_arm
                    evidence[arm].append(estimate)
                    frame_rows.append({
                        "session": session, "sequence": i, "frame_sha256": frames[i]["frame_sha256"],
                        "perception": perception, "arm": arm, "view": family, "shortlist": i in shortlist,
                        "tp_supported_columns": base["tp_columns"], "sap_supported_columns": base["sap_columns"],
                        "candidate_brackets": len(brackets), "status": estimate["status"],
                        "localized": estimate["direct_candidate"], "candidate_pixels": estimate["candidate_pixels"],
                    })
            for arm, estimates in evidence.items():
                for scope, indices in (("all", list(range(len(frames)))), ("matched_shortlist", shortlist)):
                    prediction = verify_session(frames, estimates, indices, rules)
                    if arm == "A0":
                        previous_arm = ("A0" if scope == "all" else "A1") if perception == "oracle" else (
                            "A2" if scope == "matched_shortlist" else None)
                        if previous_arm:
                            old = previous_sessions[previous_arm][session]
                            assert prediction == {k: old[k] for k in prediction}, (session, perception, scope)
                    key = f"{session}/{perception}/{arm}/{scope}"
                    session_regions[key] = prediction
                    distances = [p["p95_mm"] for p in prediction["pairs"]]
                    summaries.append({
                        "session": session, "perception": perception, "arm": arm, "scope": scope,
                        "eligible_rows": len(indices), "localized_2d": len(prediction["candidate_frames"]),
                        "localized_3d_projections": len(prediction["regions"]), "accepted": prediction["accepted"],
                        "status": prediction["status"], "inter_view_pairs": len(distances),
                        "median_inter_view_p95_mm": float(np.median(distances)) if distances else None,
                        "max_inter_view_p95_mm": max(distances, default=None),
                    })
    write_table(destination / "frames.csv", frame_rows)
    write_table(destination / "sessions.csv", summaries)
    write_json(destination / "brackets.json", bracket_rows)
    write_json(destination / "predictions.json", session_regions)
    write_json(destination / "runtime.json", {"seconds": perf_counter() - started,
        "shared_original_profiles_seconds": shared_tracking_seconds, "per_arm_seconds": timing,
        "baseline_frame_perception_decisions_reproduced": comparisons,
        "baseline_historical_session_records_reproduced": 15})
    write_json(destination / "actor_seal.json", {
        "source_prediction_seal_sha256": digest(source / "replay/prediction_seal.json"),
        "visual_review_zip_sha256": digest(review_zip),
        "frozen_baseline_config_sha256": digest(frozen_config), "adaptive_config_sha256": digest(adaptive_config),
        "output_sha256": {p.name: digest(p) for p in sorted(destination.iterdir())},
        "evaluator_files_opened": False,
    })
    print(f"Reproduced {comparisons} baseline frame/perception decisions and 15 historical arm-session records.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "frozen-config", "adaptive-config", "review-zip", "destination"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    run(*vars(cli.parse_args()).values())
