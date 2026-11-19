#!/usr/bin/env python3
"""Privileged evaluator for the actor-sealed L4/L5 TP acquisition run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import yaml

from two_stage_navigation.stage1_features import (
    UNITY_REFERENCE_PATCH_SPEC,
)
from two_stage_navigation.stage2_mbb_landmarks import (
    FacetLevel,
    LandmarkReadiness,
    LandmarkType,
    Laterality,
    evaluate_landmark_evidence,
)
from two_stage_navigation.tp_surface_core_dataset import (
    L4_CLASS,
    L5_CLASS,
    PatchProposal,
    count_points_in_patch,
    projected_pixel_diagnostics,
)
from two_stage_navigation.ultrasound_plane_geometry import (
    plane_from_full_pose,
    project_points_to_pixels,
)


PROTOCOL_IDS = {
    "sonogym-s0010-mbb-single-anatomy-poc-v1",
    "sonogym-s0010-mbb-single-anatomy-poc-v2",
}
CANDIDATES = {
    L4_CLASS: "L4_RIGHT_transverse_process",
    L5_CLASS: "L5_RIGHT_transverse_process",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def locate_geometry(root: Path, candidate: str) -> Path:
    matches = list(root.rglob(f"{candidate}_revision.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one geometry for {candidate}, found {len(matches)}")
    return matches[0]


def proxy_state(
    target: str,
    proposal: PatchProposal,
    projections: dict[str, dict[str, np.ndarray]],
) -> tuple[str, dict[str, Any]]:
    counts = {
        name: count_points_in_patch(
            proposal,
            rows=value["core_rows"],
            columns=value["core_columns"],
            patch_spec=UNITY_REFERENCE_PATCH_SPEC,
        )
        for name, value in projections.items()
    }
    uncertain = {
        name: count_points_in_patch(
            proposal,
            rows=value["uncertain_rows"],
            columns=value["uncertain_columns"],
            patch_spec=UNITY_REFERENCE_PATCH_SPEC,
        )
        for name, value in projections.items()
    }
    active = projections[target]
    diagnostics = projected_pixel_diagnostics(
        proposal,
        core_rows=active["core_rows"],
        core_columns=active["core_columns"],
        uncertain_rows=active["uncertain_rows"],
        uncertain_columns=active["uncertain_columns"],
        patch_spec=UNITY_REFERENCE_PATCH_SPEC,
    )
    other = L5_CLASS if target == L4_CLASS else L4_CLASS
    if any(value > 0 for value in uncertain.values()) or (counts[target] > 0 and counts[other] > 0):
        state = "PROXY_AMBIGUOUS"
    elif counts[target] > 0 and bool(diagnostics["adequate_projected_support"]):
        state = "PROXY_POSITIVE"
    elif counts[target] > 0:
        state = "PROXY_AMBIGUOUS"
    else:
        state = "PROXY_NEGATIVE"
    return state, {"core_counts": counts, "uncertain_counts": uncertain, **diagnostics}


def save_overlay(
    path: Path,
    image: np.ndarray,
    row: dict[str, Any],
    projections: dict[str, dict[str, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 5), dpi=140)
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Actor-visible clean ultrasound")
    axes[1].imshow(image, cmap="gray")
    for name, color in ((L4_CLASS, "#ff8c00"), (L5_CLASS, "#00b5ff")):
        value = projections[name]
        axes[1].scatter(value["core_columns"], value["core_rows"], s=4, c=color, label=name[:2])
        axes[1].scatter(
            value["uncertain_columns"], value["uncertain_rows"],
            s=4, c="#ff00ff", alpha=0.45,
        )
    column_start = int(round(float(row["proposal_column"]) - 24))
    row_start = int(round(float(row["proposal_row"]) - 34))
    axes[1].add_patch(plt.Rectangle((column_start, row_start), 48, 80, fill=False, edgecolor="lime", linewidth=1.5))
    axes[1].set_title("Evaluator-only geometry overlay")
    for axis in axes:
        axis.axis("off")
    axes[1].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_search_video(path: Path, images: np.ndarray, frames: list[dict[str, Any]]) -> None:
    """Render actor-visible search evidence without evaluator geometry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (600, 800))
    if not writer.isOpened():
        raise RuntimeError("cannot create ultrasound search video")
    try:
        for image, row in zip(images, frames):
            normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            panel = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
            panel = cv2.resize(panel, (600, 800), interpolation=cv2.INTER_NEAREST)
            if row["proposal_row"] is not None:
                x0 = int(round((float(row["proposal_column"]) - 24) * 4))
                y0 = int(round((float(row["proposal_row"]) - 34) * 4))
                cv2.rectangle(panel, (x0, y0), (x0 + 48 * 4, y0 + 80 * 4), (0, 255, 0), 2)
            cv2.rectangle(panel, (0, 0), (600, 58), (10, 10, 10), -1)
            cv2.putText(
                panel,
                f"{row['target_name'][:2]} | {row['phase'].split(':')[-1]} | p={row['target_probability']:.3f}",
                (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA,
            )
            writer.write(panel)
    finally:
        writer.release()


def save_trajectory(path: Path, frames: list[dict[str, Any]], actor: dict[str, Any]) -> None:
    selected = {
        int(row["selected_frame_sequence"]): row["target_name"]
        for row in actor["targets"]
        if row["selected_frame_sequence"] is not None
    }
    fig, axis = plt.subplots(figsize=(8, 6), dpi=150)
    for target, color in ((L4_CLASS, "#ff8c00"), (L5_CLASS, "#00b5ff")):
        rows = [row for row in frames if row["target_name"] == target]
        x = [row["executed_relative_pose_index"][0] for row in rows]
        z = [row["executed_relative_pose_index"][1] for row in rows]
        axis.plot(x, z, "o-", color=color, alpha=0.55, markersize=3, label=target[:2])
    for sequence, target in selected.items():
        pose = frames[sequence]["executed_relative_pose_index"]
        axis.scatter([pose[0]], [pose[1]], s=120, marker="*", color="red", zorder=5)
        axis.annotate(f"selected {target[:2]}", (pose[0], pose[1]))
    axis.set_xlabel("Probe x (simulator index)")
    axis.set_ylabel("Probe z (simulator index)")
    axis.set_title("Executed coarse/fine/micro and confirmation trajectory")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol.get("protocol_id") not in PROTOCOL_IDS:
        raise RuntimeError("unexpected protocol")
    expected_protocol = args.protocol.with_suffix(".sha256").read_text().split()[0]
    if sha256(args.protocol) != expected_protocol:
        raise RuntimeError("protocol seal mismatch")
    run = args.run_dir.resolve()
    actor_path = run / "results/actor_result.json"
    expected_actor = actor_path.with_suffix(".json.sha256").read_text().split()[0]
    if sha256(actor_path) != expected_actor:
        raise RuntimeError("actor result seal mismatch")
    actor = json.loads(actor_path.read_text())
    frames = read_jsonl(run / "public/frames.jsonl")
    poses = read_jsonl(run / "privileged/full_pose_replay.jsonl")
    images = np.load(run / "public/ultrasound.npz")["ultrasound"]
    coordinate_archive = np.load(run / "privileged/pixel_to_volume_coordinates.npz")
    coordinates = coordinate_archive["human_img_coords"]
    local = coordinate_archive["native_local_image_coordinates_m"].astype(np.float64)
    if not (len(frames) == len(poses) == len(images) == len(coordinates)):
        raise RuntimeError("actor/evaluator frame join is incomplete")
    for index, (frame, pose) in enumerate(zip(frames, poses)):
        if frame["frame_sequence"] != index or pose["frame_sequence"] != index:
            raise RuntimeError("frame sequence mismatch")
        if frame["frame_sha256"] != pose["frame_sha256"]:
            raise RuntimeError("frame hash mismatch")

    geometries: dict[str, dict[str, Any]] = {}
    for class_name, candidate in CANDIDATES.items():
        path = locate_geometry(args.geometry_root.resolve(), candidate)
        archive = np.load(path)
        geometries[class_name] = {
            "path": str(path),
            "sha256": sha256(path),
            "vertices": archive["vertices_mesh_index"].astype(np.float64),
            "core": archive["proposed_revised_vertex_mask"].astype(bool),
            "uncertain": archive["boundary_uncertainty_vertex_mask"].astype(bool),
        }

    frame_evaluation: dict[int, dict[str, Any]] = {}
    visual_dir = run / "visual_evidence"
    inspected = {
        int(target["selected_frame_sequence"])
        for target in actor["targets"]
        if target["selected_frame_sequence"] is not None
    }
    inspected.update(
        int(sequence)
        for target in actor["targets"]
        for sequence in target["fresh_confirmation_frame_sequences"]
    )
    for sequence in sorted(inspected):
        frame = frames[sequence]
        pose = poses[sequence]
        plane = plane_from_full_pose(
            end_effector_position_m=np.asarray(pose["human_to_ee_position_m"]),
            rotation_columns=np.asarray(pose["human_to_ee_rotation_columns"]),
            image_origin_height_m=float(pose["image_origin_height_m"]),
            local_coordinates_m=local,
            label_resolution_m=float(pose["label_resolution_m"]),
        )
        projections: dict[str, dict[str, np.ndarray]] = {}
        for class_name, geometry in geometries.items():
            projected = project_points_to_pixels(
                points_index=geometry["vertices"][geometry["core"]],
                image_origin_index=plane.origin_index,
                rotation_columns=plane.rotation_columns,
                width=150, depth=200,
                image_resolution_m=float(pose["image_resolution_m"]),
                label_resolution_m=float(pose["label_resolution_m"]),
                plane_tolerance_mm=float(protocol["evaluator"]["surface_slab_tolerance_mm"]),
            )
            uncertain = project_points_to_pixels(
                points_index=geometry["vertices"][geometry["uncertain"]],
                image_origin_index=plane.origin_index,
                rotation_columns=plane.rotation_columns,
                width=150, depth=200,
                image_resolution_m=float(pose["image_resolution_m"]),
                label_resolution_m=float(pose["label_resolution_m"]),
                plane_tolerance_mm=float(protocol["evaluator"]["surface_slab_tolerance_mm"]),
            )
            projections[class_name] = {
                "core_rows": projected["rows"][projected["inside"]],
                "core_columns": projected["columns"][projected["inside"]],
                "uncertain_rows": uncertain["rows"][uncertain["inside"]],
                "uncertain_columns": uncertain["columns"][uncertain["inside"]],
            }
        proposal = PatchProposal(
            str(frame["proposal_track_id"]),
            float(frame["proposal_row"]),
            float(frame["proposal_column"]),
            np.empty(0),
        )
        state, diagnostics = proxy_state(str(frame["target_name"]), proposal, projections)
        row_index = int(np.clip(round(float(frame["proposal_row"])), 0, 199))
        column_index = int(np.clip(round(float(frame["proposal_column"])), 0, 149))
        # Native coordinates are flattened from (width, depth), while actor images
        # are displayed as (depth, width).
        estimated_index = coordinates[sequence, column_index * 200 + row_index]
        geometry = geometries[str(frame["target_name"])]
        core = geometry["vertices"][geometry["core"]]
        distance_mm = float(np.min(np.linalg.norm(core - estimated_index, axis=1)) * 1.5)
        frame_evaluation[sequence] = {
            "frame_sequence": sequence,
            "frame_sha256": frame["frame_sha256"],
            "target_name": frame["target_name"],
            "phase": frame["phase"],
            "proxy_state": state,
            "interface_point_to_core_error_mm": distance_mm,
            "target_probability": frame["target_probability"],
            "diagnostics": diagnostics,
        }
        save_overlay(
            visual_dir / f"frame_{sequence:03d}_{str(frame['target_name'])[:2]}.png",
            images[sequence], frame, projections,
        )

    save_search_video(run / "video/ultrasound_search.mp4", images, frames)
    save_trajectory(run / "visual_evidence/executed_trajectory.png", frames, actor)

    target_reports = []
    for target in actor["targets"]:
        selected_sequence = target["selected_frame_sequence"]
        selected = None if selected_sequence is None else frame_evaluation[int(selected_sequence)]
        confirmations = [frame_evaluation[int(value)] for value in target["fresh_confirmation_frame_sequences"]]
        errors = [] if selected is None else [selected["interface_point_to_core_error_mm"]]
        errors.extend(row["interface_point_to_core_error_mm"] for row in confirmations)
        proxy_pass = bool(
            target["decision"] == "ACCEPT"
            and selected is not None
            and selected["proxy_state"] == "PROXY_POSITIVE"
            and len(confirmations) == int(protocol["search"]["required_fresh_confirmation_count"])
            and all(row["proxy_state"] == "PROXY_POSITIVE" for row in confirmations)
            and errors
            and float(np.mean(errors))
            <= float(protocol["evaluator"]["maximum_tp_interface_localization_error_mm"])
        )
        target_reports.append(
            {
                "target_name": target["target_name"],
                "actor_decision": target["decision"],
                "selected": selected,
                "fresh_confirmations": confirmations,
                "proxy_target_view_success": proxy_pass,
                "localization_error_mm": None if not errors else float(np.mean(errors)),
                "maximum_localization_error_mm": None if not errors else float(np.max(errors)),
            }
        )

    readiness = {
        (level, LandmarkType.SUPERIOR_ARTICULAR_PROCESS, Laterality.RIGHT): LandmarkReadiness.UNRESOLVED
        for level in ("L4", "L5")
    }
    readiness.update(
        {
            (level, LandmarkType.TRANSVERSE_PROCESS, Laterality.RIGHT): LandmarkReadiness.PROVISIONAL
            for level in ("L4", "L5")
        }
    )
    mbb = evaluate_landmark_evidence(
        facet_level=FacetLevel.L4_L5,
        laterality=Laterality.RIGHT,
        hypotheses=(),
        ground_truth_readiness=readiness,
        observability_validated=False,
    )
    report = {
        "schema": "sonogym-s0010-mbb-poc-evaluation-v1",
        "protocol_sha256": expected_protocol,
        "actor_result_sha256": expected_actor,
        "session_id": actor["session_id"],
        "stage1_session_id": actor["stage1_session_id"],
        "actor_outcome": actor["outcome"],
        "target_reports": target_reports,
        "both_tp_target_views_successful": all(row["proxy_target_view_success"] for row in target_reports),
        "mbb_contract_status": mbb.status.value,
        "mbb_contract_failure_reasons": list(mbb.failure_reasons),
        "target_region_localized": mbb.target_region_localized,
        "simulated_intervention_executed": False,
        "simulated_intervention_reason": "no validated MBB target region or needle/injection simulator exists",
        "milestone_verdict": (
            "PARTIAL_PASS"
            if all(row["proxy_target_view_success"] for row in target_reports)
            else "BLOCKED"
        ),
        "geometry": {
            name: {"path": row["path"], "sha256": row["sha256"]}
            for name, row in geometries.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    args.output.with_suffix(".json.sha256").write_text(
        f"{sha256(args.output)}  {args.output.name}\n"
    )
    print(json.dumps({"milestone_verdict": report["milestone_verdict"], "targets": len(target_reports)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

