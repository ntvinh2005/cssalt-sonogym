"""Summarize record-level and deduplicated replay and evaluator evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np

from workflows.mbb_direct_sonographic.prepare import read_json, verify, write_json
from workflows.mbb_direct_sonographic.replay import write_table

ARMS = ("A0", "A1", "A2_1mm", "A2_2mm")


def read_table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def summarize(result):
    seal = read_json(result / "actor_seal.json")
    for name, checksum in seal["output_sha256"].items():
        verify(result / name, checksum)
    brackets = read_json(result / "brackets.json")
    frame_data = read_table(result / "frames.csv")
    sessions = read_table(result / "sessions.csv")
    evaluator = read_table(result / "bracket_evaluation.csv")
    points = read_table(result / "echo_evaluation.csv")
    controls = read_table(result / "natural_negative_controls.csv")
    runtime = read_json(result / "runtime.json")
    metrics, diagnostics = [], []
    for perception in ("oracle", "learned"):
        for arm in ARMS:
            def relevant(rows):
                return [r for r in rows if (r["perception"], r["arm"]) == (perception, arm)]

            rows = relevant(brackets)
            unique = {(r["frame_sha256"], r["profile"], tuple(r["columns"])): r for r in rows}
            joined = [r for r in unique.values() if r["status"] == "JOINED"]
            assessed = relevant(evaluator)
            wrong = {(r["frame_sha256"], r["bracket"]) for r in assessed
                     if r["joined_path_crosses_competing_vertebra"] == "True"}
            observed = [p for p in relevant(points) if p["newly_recovered"] == "True"]
            columns = {(p["frame_sha256"], p["column_px"]) for p in observed}
            echoes = {(p["frame_sha256"], p["column_px"], p["selected_row_px"]): p for p in observed}
            identities = Counter(p["evaluator_identity"] for p in echoes.values())
            full_echoes = sum(p["path_status"] == "JOINED" for p in echoes.values())
            projected = [r for r in relevant(frame_data) if r["localized"] == "True"]
            shared = [s for s in relevant(sessions) if s["scope"] == "matched_shortlist"]
            differences = [float(s["max_inter_view_p95_mm"]) for s in shared if s["max_inter_view_p95_mm"]]
            rejected_control = relevant(controls)
            control_unique = {r["frame_sha256"]: r for r in rejected_control}
            positive_control = {r["frame_sha256"] for r in rejected_control if r["candidate"] == "True"}
            metrics.append({
                "perception": perception, "arm": arm,
                "brackets_records": len(rows), "brackets_unique": len(unique),
                "joined_profiles_records": sum(b["status"] == "JOINED" for b in rows),
                "joined_profiles_unique": len(joined), "joined_images_unique": len({r["frame_sha256"] for r in joined}),
                "ambiguous_profiles_unique": sum(r["status"] == "AMBIGUOUS" for r in unique.values()),
                "supported_columns_record_numerator": sum(b["supported"] for b in rows),
                "supported_columns_record_denominator": sum(b["total"] for b in rows),
                "supported_columns_unique_numerator": sum(b["supported"] for b in unique.values()),
                "supported_columns_unique_denominator": sum(b["total"] for b in unique.values()),
                "longest_unsupported_run_mm": max((r["unsupported_run_mm"] for r in unique.values()), default=None),
                "new_echo_columns_records": sum(r["new_echoes"] for r in rows),
                "new_echo_columns_unique": len(columns), "new_echo_points_unique": len(echoes),
                "new_echo_points_on_joined_paths_unique": full_echoes,
                "new_named_l5_tp_or_sap_unique": identities["RIGHT_L5_TP"] + identities["RIGHT_L5_SAP"],
                "new_other_l5_bone_unique": identities["OTHER_L5_BONE"],
                "new_l5_subregion_uncertain_unique": identities["L5_FIRST_VISIBLE_SUBREGION_UNCERTAIN"],
                "new_competing_vertebra_unique": identities["OTHER_VERTEBRA"],
                "new_nonbone_boundary_uncertain_unique": identities["NONBONE_LABEL_BOUNDARY_UNCERTAIN"],
                "new_unmatched_unique": identities["UNMATCHED"],
                "joined_competing_vertebra_paths_unique": len(wrong),
                "negative_controls_unique": len(control_unique), "negative_controls_with_localization": len(positive_control),
                "localized_2d_records": len(projected), "localized_2d_unique": len({r["frame_sha256"] for r in projected}),
                "projected_3d_matched_shortlist": sum(int(s["localized_3d_projections"]) for s in shared),
                "accepted_sessions_matched_shortlist": sum(s["accepted"] == "True" for s in shared),
                "total_sessions_matched_shortlist": len(shared),
                "max_inter_view_p95_mm": max(differences, default=None),
                "median_of_available_session_max_p95_mm": float(np.median(differences)) if differences else None,
                "arm_measured_seconds": runtime["per_arm_seconds"][perception][arm],
            })
            for category in sorted({r["control"] for r in rejected_control}):
                cases = [r for r in rejected_control if r["control"] == category]
                diagnostics.append({
                    "perception": perception, "arm": arm, "control": category, "records": len(cases),
                    "unique_images": len({r["frame_sha256"] for r in cases}),
                    "localized_records": sum(r["candidate"] == "True" for r in cases),
                    "localized_distinct_images": len({r["frame_sha256"] for r in cases if r["candidate"] == "True"}),
                })
    write_table(result / "metrics.csv", metrics)
    write_table(result / "control_summary.csv", diagnostics)
    write_json(result / "metrics.json", metrics)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--result", type=Path, required=True)
    summarize(cli.parse_args().result)
