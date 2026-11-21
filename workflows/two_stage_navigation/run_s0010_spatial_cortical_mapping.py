#!/usr/bin/env python3
"""Evaluate actor-side 2.5D cortical mapping on the historical integrated run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from two_stage_navigation.tp_surface_core_dataset import PatchProposal
from two_stage_navigation.ultrasound_multiview import interface_descriptors
from two_stage_navigation.ultrasound_plane_geometry import (
    plane_from_full_pose,
    project_points_to_pixels,
)
from two_stage_navigation.ultrasound_spatial_map import (
    CorticalMapPoint,
    SpatialAssociation,
    SpatialMapConfig,
    associate_spatial_points,
    cortical_map_points,
    frame_support_counts,
)

from evaluate_stage2_tp_active import CANDIDATES, locate_geometry, proxy_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def map_config(protocol: dict[str, Any], hypothesis: dict[str, Any]) -> SpatialMapConfig:
    shared = protocol["shared_map"]
    return SpatialMapConfig(
        pixel_resolution_mm=float(shared["pixel_resolution_mm"]),
        label_resolution_mm=float(shared["label_resolution_mm"]),
        sample_stride_pixels=int(shared["sample_stride_pixels"]),
        association_radius_mm=float(hypothesis["association_radius_mm"]),
        minimum_view_translation_mm=float(shared["minimum_view_translation_mm"]),
        minimum_view_angle_deg=float(shared["minimum_view_angle_deg"]),
        include_depth=bool(hypothesis["include_depth"]),
    )


def appearance_associations(
    points: list[CorticalMapPoint], poses: dict[int, np.ndarray], config: SpatialMapConfig
) -> tuple[SpatialAssociation, ...]:
    """Mutual-nearest row/confidence matching without motion compensation."""

    grouped: dict[int, list[int]] = {}
    for index, point in enumerate(points):
        grouped.setdefault(point.frame_sequence, []).append(index)
    result: list[SpatialAssociation] = []
    frames = sorted(grouped)
    for offset, left_frame in enumerate(frames):
        for right_frame in frames[offset + 1 :]:
            # Reuse the map's distinct-view contract but omit its coordinates.
            left_pose = np.asarray(poses[left_frame])
            right_pose = np.asarray(poses[right_frame])
            translation = np.linalg.norm(left_pose[:2] - right_pose[:2]) * config.label_resolution_mm
            angle = abs(np.degrees((left_pose[2] - right_pose[2] + np.pi) % (2 * np.pi) - np.pi))
            if translation < config.minimum_view_translation_mm and angle < config.minimum_view_angle_deg:
                continue
            li = np.asarray(grouped[left_frame], dtype=np.int64)
            ri = np.asarray(grouped[right_frame], dtype=np.int64)
            left = np.asarray(
                [[points[i].row / 200.0, points[i].confidence / 2.0] for i in li]
            )
            right = np.asarray(
                [[points[i].row / 200.0, points[i].confidence / 2.0] for i in ri]
            )
            distance = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
            lb = np.argmin(distance, axis=1)
            rb = np.argmin(distance, axis=0)
            for a, b in enumerate(lb):
                if rb[b] == a and distance[a, b] <= 0.08:
                    result.append(SpatialAssociation(int(li[a]), int(ri[b]), float(distance[a, b])))
    return tuple(result)


def evaluator_distances(
    points: list[CorticalMapPoint],
    edges: tuple[SpatialAssociation, ...],
    native_coordinates: np.ndarray,
    label_resolution_mm: float,
) -> np.ndarray:
    physical = []
    for point in points:
        flat = point.column * 200 + point.row
        physical.append(native_coordinates[point.frame_sequence, flat] * label_resolution_mm)
    physical_array = np.asarray(physical, dtype=np.float64)
    return np.asarray(
        [
            np.linalg.norm(physical_array[edge.left_index] - physical_array[edge.right_index])
            for edge in edges
        ],
        dtype=np.float64,
    )


def metrics(distances: np.ndarray) -> dict[str, Any]:
    if not len(distances):
        return {
            "association_count": 0,
            "median_native_separation_mm": None,
            "p90_native_separation_mm": None,
            "false_association_fraction_over_10mm": None,
        }
    return {
        "association_count": int(len(distances)),
        "median_native_separation_mm": float(np.median(distances)),
        "p90_native_separation_mm": float(np.quantile(distances, 0.90)),
        "false_association_fraction_over_10mm": float(np.mean(distances > 10.0)),
    }


def gate(record: dict[str, Any], shuffled: dict[str, Any], gates: dict[str, Any]) -> bool:
    if not record["association_count"] or not shuffled["association_count"]:
        return False
    ratio = record["median_native_separation_mm"] / max(
        shuffled["median_native_separation_mm"], 1e-9
    )
    record["correct_to_shuffled_median_ratio"] = ratio
    return bool(
        record["association_count"] >= gates["minimum_association_count_per_target_session"]
        and record["median_native_separation_mm"] <= gates["maximum_median_native_separation_mm"]
        and record["p90_native_separation_mm"] <= gates["maximum_p90_native_separation_mm"]
        and record["false_association_fraction_over_10mm"]
        <= gates["maximum_false_association_fraction_over_10mm"]
        and ratio <= gates["maximum_correct_to_shuffled_median_ratio"]
    )


def proxy_states_for_all_frames(
    *,
    frames: list[dict[str, Any]],
    poses: list[dict[str, Any]],
    local: np.ndarray,
    geometries: dict[str, dict[str, Any]],
    slab_tolerance_mm: float,
) -> dict[int, str]:
    result: dict[int, str] = {}
    for sequence, (frame, pose) in enumerate(zip(frames, poses)):
        if frame["proposal_row"] is None or frame["proposal_column"] is None:
            result[sequence] = "NO_PROPOSAL"
            continue
        plane = plane_from_full_pose(
            end_effector_position_m=np.asarray(pose["human_to_ee_position_m"]),
            rotation_columns=np.asarray(pose["human_to_ee_rotation_columns"]),
            image_origin_height_m=float(pose["image_origin_height_m"]),
            local_coordinates_m=local,
            label_resolution_m=float(pose["label_resolution_m"]),
        )
        projections: dict[str, dict[str, np.ndarray]] = {}
        for name, geometry in geometries.items():
            core = project_points_to_pixels(
                points_index=geometry["vertices"][geometry["core"]],
                image_origin_index=plane.origin_index,
                rotation_columns=plane.rotation_columns,
                width=150,
                depth=200,
                image_resolution_m=float(pose["image_resolution_m"]),
                label_resolution_m=float(pose["label_resolution_m"]),
                plane_tolerance_mm=slab_tolerance_mm,
            )
            uncertain = project_points_to_pixels(
                points_index=geometry["vertices"][geometry["uncertain"]],
                image_origin_index=plane.origin_index,
                rotation_columns=plane.rotation_columns,
                width=150,
                depth=200,
                image_resolution_m=float(pose["image_resolution_m"]),
                label_resolution_m=float(pose["label_resolution_m"]),
                plane_tolerance_mm=slab_tolerance_mm,
            )
            projections[name] = {
                "core_rows": core["rows"][core["inside"]],
                "core_columns": core["columns"][core["inside"]],
                "uncertain_rows": uncertain["rows"][uncertain["inside"]],
                "uncertain_columns": uncertain["columns"][uncertain["inside"]],
            }
        proposal = PatchProposal(
            str(frame["proposal_track_id"]),
            float(frame["proposal_row"]),
            float(frame["proposal_column"]),
            np.empty(0),
        )
        result[sequence] = proxy_state(str(frame["target_name"]), proposal, projections)[0]
    return result


def main() -> int:
    args = parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    expected = args.protocol.with_suffix(".sha256").read_text().split()[0]
    if sha256(args.protocol) != expected:
        raise RuntimeError("protocol seal mismatch")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    gates = protocol["phase_a_gates"]
    source_protocol = yaml.safe_load((args.run_root / "context/s0010_mbb_single_anatomy_poc_protocol_v3.yaml").read_text())
    geometry_root = Path(source_protocol["evaluator"]["geometry_root"])
    geometries: dict[str, dict[str, Any]] = {}
    for name, candidate in CANDIDATES.items():
        path = locate_geometry(geometry_root, candidate)
        archive = np.load(path)
        geometries[name] = {
            "vertices": archive["vertices_mesh_index"].astype(np.float64),
            "core": archive["proposed_revised_vertex_mask"].astype(bool),
            "uncertain": archive["boundary_uncertainty_vertex_mask"].astype(bool),
        }

    def load_session(session_id: str) -> dict[str, Any]:
        root = args.run_root / "stage2" / session_id
        frames = read_jsonl(root / "public/frames.jsonl")
        poses = read_jsonl(root / "privileged/full_pose_replay.jsonl")
        images = np.load(root / "public/ultrasound.npz")["ultrasound"]
        coordinates_archive = np.load(root / "privileged/pixel_to_volume_coordinates.npz")
        coordinates = coordinates_archive["human_img_coords"]
        local = coordinates_archive["native_local_image_coordinates_m"].astype(np.float64)
        if not len(frames) == len(poses) == len(images) == len(coordinates):
            raise RuntimeError(f"incomplete frame join: {session_id}")
        states = proxy_states_for_all_frames(
            frames=frames,
            poses=poses,
            local=local,
            geometries=geometries,
            slab_tolerance_mm=float(source_protocol["evaluator"]["surface_slab_tolerance_mm"]),
        )
        return {
            "frames": frames,
            "images": images,
            "coordinates": coordinates,
            "states": states,
        }

    selection_sessions = list(protocol["splits"]["method_selection_sessions"])
    final_sessions = list(protocol["splits"]["final_comparison_sessions"])
    session_data = {session_id: load_session(session_id) for session_id in selection_sessions}
    records: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    rng = np.random.default_rng(int(protocol["controls"]["shuffled_motion_seed"]))

    def evaluate_session(
        hypothesis: dict[str, Any], session_id: str, data: dict[str, Any], split: str
    ) -> None:
        config = map_config(protocol, hypothesis)
        for target in CANDIDATES:
            selected = [
                index for index, frame in enumerate(data["frames"])
                if frame["target_name"] == target
            ]
            poses = {
                int(data["frames"][index]["frame_sequence"]): np.asarray(
                    data["frames"][index]["executed_relative_pose_index"], dtype=np.float64
                )
                for index in selected
            }
            points: list[CorticalMapPoint] = []
            for index in selected:
                frame = data["frames"][index]
                points.extend(
                    cortical_map_points(
                        data["images"][index],
                        executed_pose_index=poses[int(frame["frame_sequence"])],
                        frame_sequence=int(frame["frame_sequence"]),
                        frame_sha256=str(frame["frame_sha256"]),
                        config=config,
                    )
                )
            edges = associate_spatial_points(points, poses_by_frame=poses, config=config)
            distance = evaluator_distances(
                points, edges, data["coordinates"], config.label_resolution_mm
            )
            shuffled_values = list(poses.values())
            rng.shuffle(shuffled_values)
            shuffled_lookup = {
                key: value for key, value in zip(sorted(poses), shuffled_values)
            }
            shuffled_points: list[CorticalMapPoint] = []
            for index in selected:
                frame = data["frames"][index]
                sequence = int(frame["frame_sequence"])
                shuffled_points.extend(
                    cortical_map_points(
                        data["images"][index],
                        executed_pose_index=shuffled_lookup[sequence],
                        frame_sequence=sequence,
                        frame_sha256=str(frame["frame_sha256"]),
                        config=config,
                    )
                )
            shuffled_edges = associate_spatial_points(
                shuffled_points, poses_by_frame=shuffled_lookup, config=config
            )
            shuffled_distance = evaluator_distances(
                shuffled_points,
                shuffled_edges,
                data["coordinates"],
                config.label_resolution_mm,
            )
            appearance_edges = appearance_associations(points, poses, config)
            appearance_distance = evaluator_distances(
                points, appearance_edges, data["coordinates"], config.label_resolution_mm
            )
            record = {
                "hypothesis": hypothesis["name"],
                "session_id": session_id,
                "target": target,
                "split": split,
                **metrics(distance),
                "shuffled_motion": metrics(shuffled_distance),
                "appearance_only": metrics(appearance_distance),
            }
            record["gate_pass"] = gate(record, record["shuffled_motion"], gates)
            records.append(record)
            cache[(hypothesis["name"], session_id, target)] = {
                "points": points,
                "edges": edges,
                "supports": frame_support_counts(points, edges),
                "poses": poses,
            }

    for hypothesis in protocol["hypotheses"]:
        for session_id in selection_sessions:
            evaluate_session(hypothesis, session_id, session_data[session_id], "selection")

    selection_records = [row for row in records if row["split"] == "selection"]
    passing_hypotheses = []
    for hypothesis in protocol["hypotheses"]:
        rows = [row for row in selection_records if row["hypothesis"] == hypothesis["name"]]
        if rows and all(row["gate_pass"] for row in rows):
            passing_hypotheses.append(
                (float(np.mean([row["median_native_separation_mm"] for row in rows])), hypothesis["name"])
            )
    chosen = min(passing_hypotheses)[1] if passing_hypotheses else None
    if chosen is not None:
        chosen_hypothesis = next(
            row for row in protocol["hypotheses"] if row["name"] == chosen
        )
        for session_id in final_sessions:
            session_data[session_id] = load_session(session_id)
            evaluate_session(
                chosen_hypothesis, session_id, session_data[session_id], "final_comparison"
            )
    final_rows = [row for row in records if row["split"] == "final_comparison"]
    phase_a_pass = bool(chosen and final_rows and all(row["gate_pass"] for row in final_rows))

    localizer_rows: list[dict[str, Any]] = []
    readiness: dict[str, bool] = {target: False for target in CANDIDATES}
    if phase_a_pass and chosen is not None:
        settings = protocol["phase_b_localizer"]
        for session_id in protocol["splits"]["final_comparison_sessions"]:
            data = session_data[session_id]
            for target in CANDIDATES:
                cache_row = cache[(chosen, session_id, target)]
                supports = cache_row["supports"]
                frame_rows = [
                    frame for frame in data["frames"] if frame["target_name"] == target
                ]
                eligible = [
                    frame for frame in frame_rows
                    if supports.get(int(frame["frame_sequence"]), 0)
                    >= int(settings["minimum_spatial_supporting_frames"])
                    and float(frame["target_probability"])
                    >= float(settings["minimum_target_probability"])
                ]
                ordered = sorted(
                    eligible,
                    key=lambda frame: (
                        -supports[int(frame["frame_sequence"])],
                        -float(frame["target_probability"]),
                        int(frame["frame_sequence"]),
                    ),
                )
                top = ordered[: int(settings["top_k"])]
                record = {
                    "session_id": session_id,
                    "target": target,
                    "eligible_frames": len(eligible),
                    "selected_frame_sequence": None if not top else int(top[0]["frame_sequence"]),
                    "selected_state": "ABSTAIN" if not top else data["states"][int(top[0]["frame_sequence"])],
                    "top3_states": [data["states"][int(frame["frame_sequence"])] for frame in top],
                    "top3_sequences": [int(frame["frame_sequence"]) for frame in top],
                }
                localizer_rows.append(record)
        for target in CANDIDATES:
            rows = [row for row in localizer_rows if row["target"] == target]
            readiness[target] = bool(
                len(rows) == 2
                and all(row["selected_state"] == "PROXY_POSITIVE" for row in rows)
                and all("PROXY_POSITIVE" in row["top3_states"] for row in rows)
            )

    result = {
        "schema": "s0010-spatial-cortical-mapping-result-v1",
        "source_commit": args.source_commit,
        "protocol_sha256": expected,
        "source_run": str(args.run_root.resolve()),
        "scientific_evidence": "historically_opened_development_replay",
        "selected_hypothesis": chosen,
        "phase_a_verdict": "SPATIAL_MAPPING_FEASIBLE" if phase_a_pass else "SPATIAL_MAPPING_NOT_SUPPORTED",
        "phase_b_verdict": "LOCALIZER_READY_FOR_CLOSED_LOOP" if any(readiness.values()) else "LOCALIZER_NOT_READY",
        "target_readiness": readiness,
        "mapping_records": records,
        "localizer_records": localizer_rows,
        "gpu_eligible": bool(phase_a_pass and any(readiness.values())),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    flat_records = []
    for row in records:
        flat_records.append(
            {
                key: value for key, value in row.items()
                if key not in {"shuffled_motion", "appearance_only"}
            }
            | {
                "shuffled_median_mm": row["shuffled_motion"]["median_native_separation_mm"],
                "appearance_median_mm": row["appearance_only"]["median_native_separation_mm"],
            }
        )
    write_csv(output / "mapping_metrics.csv", flat_records)
    if localizer_rows:
        write_csv(output / "localizer_metrics.csv", localizer_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    names = [row["hypothesis"] for row in selection_records]
    axes[0].scatter(range(len(selection_records)), [row["median_native_separation_mm"] for row in selection_records], label="correct motion")
    axes[0].scatter(range(len(selection_records)), [row["shuffled_motion"]["median_native_separation_mm"] for row in selection_records], label="shuffled motion")
    axes[0].axhline(float(gates["maximum_median_native_separation_mm"]), color="red", linestyle="--")
    axes[0].set_title("Selection sessions: native separation")
    axes[0].set_ylabel("median evaluator separation (mm)")
    axes[0].set_xticks(range(len(names)), names, rotation=75, fontsize=6)
    axes[0].legend()
    axes[1].axis("off")
    axes[1].text(
        0.02,
        0.95,
        f"Phase A: {result['phase_a_verdict']}\nSelected: {chosen}\nPhase B: {result['phase_b_verdict']}\nL4 ready: {readiness['L4_RIGHT_TP_SURFACE_CORE']}\nL5 ready: {readiness['L5_RIGHT_TP_SURFACE_CORE']}",
        va="top",
        family="monospace",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output / "spatial_mapping_summary.png")
    plt.close(fig)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
