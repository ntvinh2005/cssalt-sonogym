"""Evaluate sealed echo paths against synchronized first-visible simulator labels."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np
import yaml

from two_stage_navigation.sonoanatomy_labels import LEVEL_SOURCE_LABEL, first_bone_surface
from workflows.mbb_direct_sonographic.prepare import digest, read_json, verify, write_json
from workflows.mbb_direct_sonographic.replay import write_table

LABEL_CONVERSION = Path("source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml")
NAMED_L5 = {"RIGHT_L5_TP", "RIGHT_L5_SAP"}


def classify(pixel, raw, first, samples, identities, hints, bone_ids, tolerance):
    row, col = pixel
    first_rows = np.flatnonzero(first[:, col])
    raw_id = int(raw[row, col])
    if not len(first_rows) or abs(int(first_rows[0]) - row) > tolerance:
        neighbors = raw[max(0, row - 2):row + 3, col]
        at_tissue_boundary = raw_id not in bone_ids and np.any(neighbors != raw_id)
        return "NONBONE_LABEL_BOUNDARY_UNCERTAIN" if at_tissue_boundary else "UNMATCHED", raw_id, None
    front_row = int(first_rows[0])
    front_id = int(raw[front_row, col])
    first_distance = abs(front_row - row)
    if front_id != LEVEL_SOURCE_LABEL["L5"]:
        return "OTHER_VERTEBRA" if front_id in bone_ids else "UNMATCHED", front_id, first_distance
    if len(samples):
        # Match only the same acoustic ray and the same first-visible bone tier.
        matched = (samples[:, 1] == col) & (np.abs(samples[:, 0] - row) <= tolerance) & (hints == "L5")
        names = set(identities[matched].tolist())
        known = names & NAMED_L5
        if len(known) == 1:
            return known.pop(), front_id, first_distance
        if names == {"OTHER_BONE"}:
            return "OTHER_L5_BONE", front_id, first_distance
    return "L5_FIRST_VISIBLE_SUBREGION_UNCERTAIN", front_id, first_distance


def load_evaluator_record(record, poses):
    receipt_path = Path(record["source_receipt"])
    verify(receipt_path, record["source_receipt_sha256"])
    receipt = read_json(receipt_path)
    raw_path = Path(receipt["actor_pose_source"]["path"]).parent.parent / "privileged/raw_labels.npz"
    ledger = raw_path.with_name("full_pose_replay.jsonl")
    import json
    privileged_poses = [json.loads(line) for line in ledger.read_text().splitlines()]
    if any(poses[i][key] != privileged_poses[i][key]
           for i in range(len(poses)) for key in ("frame_sequence", "frame_sha256", "acquisition_id")):
        raise ValueError("evaluator labels are not synchronized with the sealed acquisition")
    with np.load(raw_path, allow_pickle=False) as data:
        raw = data["raw_labels"]
    if raw.shape[1:] != (200, 150) or len(raw) != len(privileged_poses):
        raise ValueError("evaluator label image dimensions differ from the capture")
    source = Path(receipt["oracle"]["output"]["npz"]["path"])
    verify(source, record["source_oracle_visible_pixel_cache_sha256"])
    with np.load(source, allow_pickle=False) as data:
        cloud = [data[name] for name in ("frame_sequence", "pixel_row_col", "identity", "level_hint")]
    return raw, cloud, {"raw_label_sha256": digest(raw_path), "synchronized_pose_ledger_sha256": digest(ledger),
                        "visible_pixel_labels_sha256": digest(source)}


def evaluate(result, source, adaptive_config):
    seal = read_json(result / "actor_seal.json")
    for name, checksum in seal["output_sha256"].items():
        verify(result / name, checksum)
    verify(adaptive_config, seal["adaptive_config_sha256"])
    config = read_json(adaptive_config)
    previous = read_json(source / "frozen-run/replay/frame_audits.json")
    manifest = read_json(source / "frozen-run/inputs/manifest.json")
    frozen = read_json(source / "frozen-run/reference_evaluation.json")
    verify(Path(frozen["audited_reference_path"]), frozen["audited_reference_sha256"])
    bone_ids = {int(label) for label, code in yaml.safe_load(LABEL_CONVERSION.read_text()).items()
                if int(code) == 13}
    evidence = read_json(result / "brackets.json")
    point_audit, bracket_audit, provenance = [], [], {}
    tolerance = config["evaluation_pixel_tolerance_mm"] / .5
    for session in previous:
        raw, (sequences, pixels, identities, hints), provenance[session] = load_evaluator_record(
            manifest["recordings"][session], previous[session]["public_poses"],
        )
        for b in (item for item in evidence if item["session"] == session):
            frame = b["sequence"]
            first = first_bone_surface(raw[frame], bone_ids)
            mask = sequences == frame
            classified = []
            for j, col in enumerate(b["columns"]):
                prefix = b["prefix_rows"][j]
                suffix = b["suffix_rows"][j] if b["suffix_rows"] is not None else None
                selected = prefix if prefix is not None else suffix
                label, raw_id, dist = ("UNSUPPORTED", None, None) if selected is None else classify(
                    (selected, col), raw[frame], first, pixels[mask], identities[mask], hints[mask], bone_ids, tolerance,
                )
                point = {
                    **{k: b[k] for k in ("session", "sequence", "frame_sha256", "perception", "arm", "bracket")},
                    "column_px": col, "original_row_px": b["baseline_rows"][j],
                    "original_supported": b["baseline_observed"][j], "selected_row_px": selected,
                    "echo_status": "OBSERVED" if selected is not None else "UNSUPPORTED", "path_status": b["status"],
                    "newly_recovered": selected is not None and not b["baseline_observed"][j],
                    "evaluator_identity": label, "simulator_label_id": raw_id,
                    "same_ray_first_bone_distance_mm": dist * .5 if dist is not None else None,
                }
                point_audit.append(point)
                classified.append(point)
            categories = Counter(p["evaluator_identity"] for p in classified if p["newly_recovered"])
            all_categories = Counter(p["evaluator_identity"] for p in classified)
            bracket_audit.append({
                **{k: b[k] for k in ("session", "sequence", "frame_sha256", "perception", "arm", "bracket", "status",
                                   "supported", "total", "unsupported_run_mm", "new_echoes", "max_adjacent_depth_step_mm")},
                "new_right_l5_tp_or_sap": sum(categories[name] for name in NAMED_L5),
                "new_other_l5_bone": categories["OTHER_L5_BONE"],
                "new_l5_subregion_uncertain": categories["L5_FIRST_VISIBLE_SUBREGION_UNCERTAIN"],
                "new_competing_vertebra": categories["OTHER_VERTEBRA"],
                "new_nonbone_boundary_uncertain": categories["NONBONE_LABEL_BOUNDARY_UNCERTAIN"],
                "new_unmatched": categories["UNMATCHED"],
                "joined_path_crosses_competing_vertebra": b["status"] == "JOINED" and all_categories["OTHER_VERTEBRA"] > 0,
            })
    write_table(result / "echo_evaluation.csv", point_audit)
    write_table(result / "bracket_evaluation.csv", bracket_audit)
    with (result / "frames.csv").open() as stream:
        frames = list(csv.DictReader(stream))
    controls = []
    for r in frames:
        if r["view"] != "transverse":
            continue
        oracle = previous[r["session"]]["frames"]["oracle"][int(r["sequence"])]
        tp, sap = oracle["tp_columns"] >= 3, oracle["sap_columns"] >= 3
        categories = []
        if tp != sap:
            categories.append("oracle_TP_without_SAP" if tp else "oracle_SAP_without_TP")
        if not (tp and sap) and max(oracle["n_l4_tp_pixels"], oracle["n_l4_sap_pixels"]) >= 3:
            categories.append("visible_L4_evidence_without_joint_L5_support")
        if not tp and not sap:
            categories.append("neither_oracle_L5_anchor")
        for category in categories:
            controls.append({
                **{k: r[k] for k in ("session", "sequence", "frame_sha256", "perception", "arm")},
                "control": category, "candidate": r["localized"] == "True",
            })
    write_table(result / "natural_negative_controls.csv", controls)
    write_json(result / "evaluation_provenance.json", {
        "actor_seal_sha256": digest(result / "actor_seal.json"), "labels": provenance,
        "label_conversion_sha256": digest(LABEL_CONVERSION),
        "first_visible_depth_tolerance_mm": config["evaluation_pixel_tolerance_mm"],
        "independent_target_reference": frozen["status"],
        "limitation": "Proxy same-ray identity. Nonbone boundaries and unmatched peaks are uncertain. No CT repair.",
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("result", "source", "adaptive-config"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    evaluate(*vars(cli.parse_args()).values())
