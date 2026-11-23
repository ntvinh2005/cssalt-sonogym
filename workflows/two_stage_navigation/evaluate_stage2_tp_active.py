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
import nibabel as nib
import numpy as np
from scipy import ndimage
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
from two_stage_navigation.stage2_tp_active import (
    POC_PROTOCOL_IDS,
    proxy_confirmation_requirement_met,
)
from two_stage_navigation.sonoanatomy_labels import (
    SAP_INTERFACE,
    TP_INTERFACE,
    right_level_masks,
    visible_interface_labels,
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


def bone_source_ids(path: Path) -> set[int]:
    mapping = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {int(source) for source, acoustic in mapping.items() if int(acoustic) == 13}


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


def semantic_pair_identity(
    sonoanatomy: dict[str, Any],
    labels_by_frame: dict[int, np.ndarray],
    *,
    maximum_distance_pixels: float,
) -> dict[str, Any]:
    """Evaluate the actor-selected clusters without exposing labels to the actor."""

    trajectory_id = sonoanatomy.get("selected_trajectory_id")
    pair_decision = sonoanatomy.get("selected_pair_decision") or {}
    selected = pair_decision.get("selected")
    if trajectory_id is None or selected is None:
        return {"accepted": False, "pair_correct": False, "structures": {}}
    trajectory = next(
        row
        for row in sonoanatomy["trajectory_decisions"]
        if row["trajectory_id"] == trajectory_id
    )
    observations = trajectory["semantic_observations"]
    structures = {}
    for name, class_id, cluster_key in (
        ("TP", int(TP_INTERFACE), "tp_cluster_index"),
        ("SAP", int(SAP_INTERFACE), "sap_cluster_index"),
    ):
        cluster = trajectory["semantic_clusters"][int(selected[cluster_key])]
        distances = []
        correct = []
        for observation_index in cluster["observation_indices"]:
            observation = observations[int(observation_index)]
            label = labels_by_frame[int(observation["frame_sequence"])]
            target = label == class_id
            if np.any(target):
                distance = ndimage.distance_transform_edt(~target)
                row = int(np.clip(round(observation["row"]), 0, target.shape[0] - 1))
                column = int(
                    np.clip(round(observation["column"]), 0, target.shape[1] - 1)
                )
                value = float(distance[row, column])
            else:
                value = None
            distances.append(value)
            correct.append(value is not None and value <= maximum_distance_pixels)
        structures[name] = {
            "correct_observation_count": int(sum(correct)),
            "observation_count": len(correct),
            "majority_correct": bool(sum(correct) > len(correct) / 2.0),
            "distances_pixels": distances,
        }
    return {
        "accepted": True,
        "pair_correct": bool(
            structures["TP"]["majority_correct"]
            and structures["SAP"]["majority_correct"]
        ),
        "selected_trajectory_id": trajectory_id,
        "maximum_distance_pixels": maximum_distance_pixels,
        "structures": structures,
    }


def binary_pair_metrics(truth: list[bool], prediction: list[bool]) -> dict[str, Any]:
    tp = sum(left and right for left, right in zip(truth, prediction))
    fp = sum(not left and right for left, right in zip(truth, prediction))
    fn = sum(left and not right for left, right in zip(truth, prediction))
    tn = sum(not left and not right for left, right in zip(truth, prediction))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accepted_count": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "clear_negative_false_accept_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def main() -> int:
    args = parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol.get("protocol_id") not in POC_PROTOCOL_IDS:
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
    raw_labels = np.load(run / "privileged/raw_labels.npz")["raw_labels"]
    coordinate_archive = np.load(run / "privileged/pixel_to_volume_coordinates.npz")
    coordinates = coordinate_archive["human_img_coords"]
    local = coordinate_archive["native_local_image_coordinates_m"].astype(np.float64)
    if not (
        len(frames)
        == len(poses)
        == len(images)
        == len(raw_labels)
        == len(coordinates)
    ):
        raise RuntimeError("actor/evaluator frame join is incomplete")
    for index, (frame, pose) in enumerate(zip(frames, poses)):
        if frame["frame_sequence"] != index or pose["frame_sequence"] != index:
            raise RuntimeError("frame sequence mismatch")
        if frame["frame_sha256"] != pose["frame_sha256"]:
            raise RuntimeError("frame hash mismatch")

    semantic_labels: dict[int, np.ndarray] = {}
    if any("sonoanatomy" in target for target in actor["targets"]):
        prediction_path = Path(protocol["evaluator"]["subregion_prediction_path"])
        segmentation_path = Path(protocol["evaluator"]["source_segmentation_path"])
        conversion_path = Path(protocol["evaluator"]["label_conversion_path"])
        prediction_image = nib.load(str(prediction_path))
        segmentation_image = nib.load(str(segmentation_path))
        if prediction_image.shape != segmentation_image.shape or not np.allclose(
            prediction_image.affine, segmentation_image.affine
        ):
            raise RuntimeError("semantic evaluator volumes use incompatible grids")
        prediction_volume = np.asarray(prediction_image.dataobj, dtype=np.uint8)
        segmentation_volume = np.asarray(segmentation_image.dataobj, dtype=np.uint8)
        masks = {
            level: right_level_masks(
                prediction_volume,
                segmentation_volume,
                prediction_image.affine,
                level,
            )
            for level in ("L4", "L5")
        }
        bone_ids = bone_source_ids(conversion_path)
        for sequence, frame in enumerate(frames):
            if "tp_predicted" not in frame:
                continue
            level = str(frame["target_name"])[:2]
            semantic_labels[sequence] = visible_interface_labels(
                image=images[sequence],
                raw_labels=raw_labels[sequence],
                coordinate_map_index=coordinates[sequence],
                masks=masks[level],
                bone_source_ids=bone_ids,
            ).labels

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
    required_confirmation_count = int(protocol["search"]["required_fresh_confirmation_count"])
    minimum_positive_confirmation_count = int(
        protocol["evaluator"].get(
            "minimum_proxy_positive_fresh_confirmations",
            required_confirmation_count,
        )
    )
    if not 1 <= minimum_positive_confirmation_count <= required_confirmation_count:
        raise RuntimeError("invalid proxy-positive fresh-confirmation requirement")
    for target in actor["targets"]:
        selected_sequence = target["selected_frame_sequence"]
        selected = None if selected_sequence is None else frame_evaluation[int(selected_sequence)]
        confirmations = [frame_evaluation[int(value)] for value in target["fresh_confirmation_frame_sequences"]]
        errors = [] if selected is None else [selected["interface_point_to_core_error_mm"]]
        errors.extend(row["interface_point_to_core_error_mm"] for row in confirmations)
        positive_confirmation_count = sum(
            row["proxy_state"] == "PROXY_POSITIVE" for row in confirmations
        )
        proxy_pass = bool(
            target["decision"] == "ACCEPT"
            and selected is not None
            and selected["proxy_state"] == "PROXY_POSITIVE"
            and proxy_confirmation_requirement_met(
                [row["proxy_state"] for row in confirmations],
                required_count=required_confirmation_count,
                minimum_positive_count=minimum_positive_confirmation_count,
            )
            and errors
            and float(np.mean(errors))
            <= float(protocol["evaluator"]["maximum_tp_interface_localization_error_mm"])
        )
        semantic_identity = None
        semantic_confirmation_rows = []
        semantic_frame_metrics = None
        semantic_pair_success = None
        if "sonoanatomy" in target:
            sonoanatomy = target["sonoanatomy"]
            semantic_identity = semantic_pair_identity(
                sonoanatomy,
                semantic_labels,
                maximum_distance_pixels=float(
                    protocol["evaluator"][
                        "maximum_semantic_interface_distance_pixels"
                    ]
                ),
            )
            for confirmation in sonoanatomy.get("fresh_confirmations", []):
                sequence = int(confirmation["frame_sequence"])
                label = semantic_labels[sequence]
                semantic_confirmation_rows.append(
                    {
                        "frame_sequence": sequence,
                        "frame_sha256": confirmation["frame_sha256"],
                        "tp_supported": bool(np.any(label == TP_INTERFACE)),
                        "sap_supported": bool(np.any(label == SAP_INTERFACE)),
                    }
                )
            current_semantic_frames = [
                row
                for row in frames
                if row["target_name"] == target["target_name"]
                and "tp_predicted" in row
            ]
            semantic_frame_metrics = binary_pair_metrics(
                [
                    bool(
                        np.any(semantic_labels[int(row["frame_sequence"])] == TP_INTERFACE)
                        and np.any(
                            semantic_labels[int(row["frame_sequence"])] == SAP_INTERFACE
                        )
                    )
                    for row in current_semantic_frames
                ],
                [
                    bool(row["tp_predicted"] and row["sap_predicted"])
                    for row in current_semantic_frames
                ],
            )
            semantic_pair_success = bool(
                target["decision"] == "ACCEPT"
                and semantic_identity["pair_correct"]
                and len(semantic_confirmation_rows)
                == len(protocol["sonoanatomy_scan"]["confirmation_dither_offsets"])
                and sum(
                    row["tp_supported"] and row["sap_supported"]
                    for row in semantic_confirmation_rows
                )
                >= int(
                    protocol["evaluator"][
                        "minimum_proxy_positive_semantic_confirmations"
                    ]
                )
            )
        integrated_success = bool(
            proxy_pass
            and (semantic_pair_success is None or semantic_pair_success)
        )
        target_reports.append(
            {
                "target_name": target["target_name"],
                "actor_decision": target["decision"],
                "selected": selected,
                "fresh_confirmations": confirmations,
                "positive_fresh_confirmation_count": positive_confirmation_count,
                "required_fresh_confirmation_count": required_confirmation_count,
                "minimum_proxy_positive_fresh_confirmations": minimum_positive_confirmation_count,
                "proxy_target_view_success": proxy_pass,
                "semantic_pair_identity": semantic_identity,
                "semantic_fresh_confirmations": semantic_confirmation_rows,
                "semantic_positive_fresh_confirmation_count": sum(
                    row["tp_supported"] and row["sap_supported"]
                    for row in semantic_confirmation_rows
                ),
                "minimum_proxy_positive_semantic_confirmations": int(
                    protocol["evaluator"].get(
                        "minimum_proxy_positive_semantic_confirmations",
                        len(semantic_confirmation_rows),
                    )
                ),
                "semantic_frame_metrics": semantic_frame_metrics,
                "semantic_pair_success": semantic_pair_success,
                "integrated_target_success": integrated_success,
                "localization_error_mm": None if not errors else float(np.mean(errors)),
                "maximum_localization_error_mm": None if not errors else float(np.max(errors)),
            }
        )

    successful_levels = {
        str(row["target_name"])[:2]
        for row in target_reports
        if row["semantic_pair_success"] is True
    }
    readiness = {
        (level, LandmarkType.SUPERIOR_ARTICULAR_PROCESS, Laterality.RIGHT): (
            LandmarkReadiness.PROVISIONAL
            if level in successful_levels
            else LandmarkReadiness.UNRESOLVED
        )
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
        observability_validated=all(level in successful_levels for level in ("L4", "L5")),
    )
    all_integrated_targets = all(
        row["integrated_target_success"] for row in target_reports
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
        "both_sap_tp_pairs_successful": all(
            row["semantic_pair_success"] is True for row in target_reports
        ),
        "integrated_stage2_session_success": all_integrated_targets,
        "mbb_contract_status": mbb.status.value,
        "mbb_contract_failure_reasons": list(mbb.failure_reasons),
        "target_region_localized": mbb.target_region_localized,
        "simulated_intervention_executed": False,
        "simulated_intervention_reason": "no validated MBB target region or needle/injection simulator exists",
        "milestone_verdict": (
            "STAGE2_SESSION_PASS" if all_integrated_targets else "BLOCKED"
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

