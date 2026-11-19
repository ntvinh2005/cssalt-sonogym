#!/usr/bin/env python3
"""Audit frozen L4 whole-image proposal errors without fitting or changing thresholds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = ROOT / "source/spinal_surgery/spinal_surgery/evaluation"
sys.path.insert(0, str(EVALUATION_ROOT))
from two_stage_navigation.stage1_features import (  # noqa: E402
    UNITY_REFERENCE_PATCH_SPEC,
    extract_interface_patch,
    extract_patch_feature,
    normalized_patch_intensity,
)
from two_stage_navigation.ultrasound_plane_geometry import (  # noqa: E402
    plane_from_full_pose,
    project_points_to_pixels,
)


CLASS_NAMES = {
    0: "BACKGROUND_OR_OTHER_STRUCTURE",
    1: "L4_RIGHT_TP_SURFACE_CORE",
    2: "L5_RIGHT_TP_SURFACE_CORE",
}
EXPECTED_CHECKPOINT_SHA256 = "338351fdee5a8444dee5a6ec250d828b8f1eedbc3f76e79495aa1aafbdeb2f8b"
EXPECTED_L4 = {"tp": 12, "fp": 14, "fn": 5}
EXPECTED_L5 = {"tp": 16, "fp": 4, "fn": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-slab-tolerance-mm", type=float, default=1.5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def zip_member(archive: zipfile.ZipFile, basename: str) -> bytes:
    matches = [name for name in archive.namelist() if Path(name).name == basename]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {basename} in source ZIP")
    return archive.read(matches[0])


def load_geometry(source_zip: Path) -> dict[int, dict[str, np.ndarray]]:
    names = {
        1: "L4_RIGHT_transverse_process_revision.npz",
        2: "L5_RIGHT_transverse_process_revision.npz",
    }
    result: dict[int, dict[str, np.ndarray]] = {}
    with zipfile.ZipFile(source_zip) as archive:
        for class_id, name in names.items():
            with np.load(io.BytesIO(zip_member(archive, name))) as values:
                result[class_id] = {
                    "vertices": values["vertices_mesh_index"].astype(np.float64),
                    "core": values["proposed_revised_vertex_mask"].astype(bool),
                    "uncertain": values["boundary_uncertainty_vertex_mask"].astype(bool),
                }
    return result


def apply_thresholds(
    probabilities: np.ndarray, classes: np.ndarray, thresholds: dict[int, float]
) -> np.ndarray:
    predictions = np.zeros(len(probabilities), dtype=np.int64)
    for index, values in enumerate(probabilities):
        column = int(np.argmax(values))
        class_id = int(classes[column])
        if class_id in thresholds and values[column] >= thresholds[class_id]:
            predictions[index] = class_id
    return predictions


def binary_counts(
    truth_by_frame: dict[str, set[int]], predicted_by_frame: dict[str, set[int]], class_id: int
) -> dict[str, int]:
    frames = sorted(set(truth_by_frame) | set(predicted_by_frame))
    return {
        "tp": sum(class_id in truth_by_frame[frame] and class_id in predicted_by_frame[frame] for frame in frames),
        "fp": sum(class_id not in truth_by_frame[frame] and class_id in predicted_by_frame[frame] for frame in frames),
        "fn": sum(class_id in truth_by_frame[frame] and class_id not in predicted_by_frame[frame] for frame in frames),
        "tn": sum(class_id not in truth_by_frame[frame] and class_id not in predicted_by_frame[frame] for frame in frames),
    }


def classify_false_positive(rows: list[dict[str, Any]]) -> tuple[str, list[str], str]:
    accepted = [row for row in rows if int(row["predicted_class_id"]) == 1]
    competing = sum(int(row["l5_core_points_in_patch"]) > 0 for row in accepted)
    uncertain = sum(
        int(row["l4_uncertain_points_in_patch"]) > 0
        or int(row["l5_uncertain_points_in_patch"]) > 0
        for row in accepted
    )
    accepted_l4_overlap = sum(int(row["l4_core_points_in_patch"]) > 0 for row in accepted)
    frame_l4_overlap = any(int(row["l4_core_points_in_patch"]) > 0 for row in rows)
    clean_proxy_negative = sum(
        int(row["l4_core_points_in_patch"]) == 0
        and int(row["l5_core_points_in_patch"]) == 0
        and int(row["l4_uncertain_points_in_patch"]) == 0
        and int(row["l5_uncertain_points_in_patch"]) == 0
        for row in accepted
    )
    evidence: list[str] = []
    if competing:
        evidence.append(f"{competing} accepted L4 proposal(s) overlap the competing L5 proxy core")
    if uncertain:
        evidence.append(f"{uncertain} accepted L4 proposal(s) overlap a declared uncertainty region")
    if frame_l4_overlap:
        evidence.append("the frame contains projected L4 core evidence but no proposal passes every proxy gate")
    if len(accepted) > 1:
        evidence.append(f"{len(accepted)} proposals independently trigger the image-level L4 alarm")
    if clean_proxy_negative:
        evidence.append(f"{clean_proxy_negative} accepted proposal(s) have no projected core or uncertainty overlap")

    if competing:
        primary = "an accepted proposal overlaps the competing L5 proxy core"
        direction = "test a frozen-checkpoint L4-versus-L5 score-margin rule on calibration images"
    elif frame_l4_overlap and accepted_l4_overlap:
        primary = "the proxy label excludes projected L4 evidence admitted by the classifier"
        direction = "audit the projected-support and uncertainty exclusion for this exact proposal before changing the model"
    elif len(accepted) > 1:
        primary = "multiple accepted proposals amplify the image-level false alarm"
        direction = "test a calibration-selected image-level aggregation rule with the checkpoint frozen"
    else:
        primary = "the model assigns L4 morphology to a proxy-negative bone-interface proposal"
        direction = "test a calibration-selected L4 score margin against background on the frozen checkpoint"
    return primary, evidence, direction


def display_image(path: Path, image: np.ndarray, title: str) -> None:
    fig, axis = plt.subplots(figsize=(4.5, 6), dpi=150)
    axis.imshow(image, cmap="gray", vmin=float(np.min(image)), vmax=float(np.max(image)))
    axis.set_title(title, fontsize=9)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def frame_projection(
    row: dict[str, Any],
    geometry: dict[int, dict[str, np.ndarray]],
    tolerance_mm: float,
    pose_cache: dict[Path, list[dict[str, Any]]],
    local_cache: dict[Path, np.ndarray],
) -> dict[int, dict[str, np.ndarray]]:
    archive = Path(row["image_archive"])
    session_dir = archive.parent.parent
    pose_path = session_dir / "privileged/full_pose_replay.jsonl"
    coordinate_path = session_dir / "privileged/pixel_to_volume_coordinates.npz"
    if pose_path not in pose_cache:
        pose_cache[pose_path] = read_jsonl(pose_path)
    if coordinate_path not in local_cache:
        local_cache[coordinate_path] = np.load(coordinate_path)["native_local_image_coordinates_m"]
    pose = pose_cache[pose_path][int(row["image_archive_index"])]
    if pose["acquisition_id"] != row["acquisition_id"]:
        raise RuntimeError("pose and image acquisition identifiers differ")
    plane = plane_from_full_pose(
        end_effector_position_m=np.asarray(pose["human_to_ee_position_m"]),
        rotation_columns=np.asarray(pose["human_to_ee_rotation_columns"]),
        image_origin_height_m=float(pose["image_origin_height_m"]),
        local_coordinates_m=local_cache[coordinate_path],
        label_resolution_m=float(pose["label_resolution_m"]),
    )
    projected: dict[int, dict[str, np.ndarray]] = {}
    for class_id, values in geometry.items():
        projected[class_id] = {}
        for region in ("core", "uncertain"):
            points = project_points_to_pixels(
                points_index=values["vertices"][values[region]],
                image_origin_index=plane.origin_index,
                rotation_columns=plane.rotation_columns,
                width=150,
                depth=200,
                image_resolution_m=float(pose["image_resolution_m"]),
                label_resolution_m=float(pose["label_resolution_m"]),
                plane_tolerance_mm=tolerance_mm,
            )
            inside = points["inside"]
            projected[class_id][f"{region}_rows"] = points["rows"][inside]
            projected[class_id][f"{region}_columns"] = points["columns"][inside]
    return projected


def export_context(
    path: Path,
    image: np.ndarray,
    rows: list[dict[str, Any]],
    failure_id: str,
) -> None:
    fig, axis = plt.subplots(figsize=(5, 6), dpi=150)
    axis.imshow(image, cmap="gray", vmin=float(np.min(image)), vmax=float(np.max(image)))
    for row in rows:
        accepted = int(row["predicted_class_id"]) == 1
        color = "red" if accepted else "white"
        alpha = 0.95 if accepted else 0.25
        rectangle = plt.Rectangle(
            (float(row["patch_column_start"]), float(row["patch_row_start"])),
            float(row["patch_width"]),
            float(row["patch_depth"]),
            fill=False,
            edgecolor=color,
            linewidth=2.0 if accepted else 0.6,
            alpha=alpha,
        )
        axis.add_patch(rectangle)
        if accepted:
            axis.text(
                float(row["patch_column_start"]),
                float(row["patch_row_start"]) - 2,
                f"pL4={float(row['probability_l4']):.3f}",
                color="red",
                fontsize=6,
                backgroundcolor="black",
            )
    axis.set_title(f"{failure_id}: actor-visible proposals\nred = accepted L4", fontsize=9)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def export_overlay(
    path: Path,
    image: np.ndarray,
    rows: list[dict[str, Any]],
    projected: dict[int, dict[str, np.ndarray]],
    failure_id: str,
) -> None:
    fig, axis = plt.subplots(figsize=(5, 6), dpi=150)
    axis.imshow(image, cmap="gray", vmin=float(np.min(image)), vmax=float(np.max(image)))
    colors = {1: "orange", 2: "magenta"}
    for class_id in (1, 2):
        values = projected[class_id]
        axis.scatter(
            values["core_columns"],
            values["core_rows"],
            s=5,
            c=colors[class_id],
            label=f"{CLASS_NAMES[class_id]} core",
        )
        axis.scatter(
            values["uncertain_columns"],
            values["uncertain_rows"],
            s=5,
            c="cyan",
            marker="x",
            label=f"{CLASS_NAMES[class_id]} uncertainty",
        )
    for row in rows:
        if int(row["predicted_class_id"]) != 1:
            continue
        axis.add_patch(
            plt.Rectangle(
                (float(row["patch_column_start"]), float(row["patch_row_start"])),
                float(row["patch_width"]),
                float(row["patch_depth"]),
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
        )
    axis.set_title(f"{failure_id}: evaluator-only proxy overlay", fontsize=9)
    axis.legend(fontsize=5, loc="lower right")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def export_proposal_crop(
    path: Path, image: np.ndarray, row: dict[str, Any], failure_id: str
) -> None:
    patch = extract_interface_patch(
        image,
        float(row["patch_row"]),
        float(row["patch_column"]),
        UNITY_REFERENCE_PATCH_SPEC,
    )
    if patch is None:
        raise RuntimeError("audited proposal crop is outside the image")
    normalized = normalized_patch_intensity(patch, UNITY_REFERENCE_PATCH_SPEC)
    fig, axes = plt.subplots(1, 2, figsize=(6, 4), dpi=150)
    axes[0].imshow(patch, cmap="gray")
    axes[0].set_title("raw 48x80")
    axes[1].imshow(normalized, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("model intensity 16x16")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(
        f"{failure_id} {row['sample_id']}\np=[{float(row['probability_background']):.3f}, "
        f"{float(row['probability_l4']):.3f}, {float(row['probability_l5']):.3f}]",
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    training_dir = args.training_dir.resolve()
    output = args.output_dir.resolve()
    assets = output / "assets"
    crops = assets / "proposal_crops"
    assets.mkdir(parents=True, exist_ok=True)
    crops.mkdir(parents=True, exist_ok=True)

    checkpoint = training_dir / "tp_surface_core_simulator_proxy_extratrees.joblib"
    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("frozen checkpoint checksum changed")
    baseline = json.loads((training_dir / "result.json").read_text(encoding="utf-8"))
    if baseline["status"] != "EXPLORATORY_PROXY_TRAINING_COMPLETED":
        raise RuntimeError("unexpected frozen baseline status")
    bundle = joblib.load(checkpoint)
    estimator = bundle["estimator"]
    thresholds = {int(key): float(value) for key, value in bundle["thresholds"].items()}

    source_rows = read_csv(dataset_dir / "candidate_sample_manifest.csv")
    selected_rows = read_csv(training_dir / "proxy_sample_manifest.csv")
    selected_truth: dict[str, set[int]] = defaultdict(set)
    for row in selected_rows:
        if row["split"] == "development_evaluation":
            selected_truth[row["frame_sha256"]].add(int(row["proxy_class_id"]))
    development = [row for row in source_rows if row["split"] == "development_evaluation"]
    frame_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    archive_cache: dict[Path, np.ndarray] = {}
    image_cache: dict[str, np.ndarray] = {}
    for row in development:
        archive = Path(row["image_archive"])
        if archive not in archive_cache:
            archive_cache[archive] = np.load(archive)["ultrasound"]
        image = np.asarray(archive_cache[archive][int(row["image_archive_index"])])
        image_cache[row["frame_sha256"]] = image
        feature = extract_patch_feature(
            image,
            float(row["patch_row"]),
            float(row["patch_column"]),
            UNITY_REFERENCE_PATCH_SPEC,
        )
        if feature is None or not np.all(np.isfinite(feature)):
            raise RuntimeError(f"invalid feature for {row['sample_id']}")
        frame_rows[row["frame_sha256"]].append({**row, "feature": feature})

    predicted: dict[str, set[int]] = defaultdict(set)
    scored_rows: dict[str, list[dict[str, Any]]] = {}
    for frame_hash, rows in frame_rows.items():
        probabilities = estimator.predict_proba(np.stack([row["feature"] for row in rows]))
        predictions = apply_thresholds(probabilities, estimator.classes_, thresholds)
        scored: list[dict[str, Any]] = []
        for row, values, prediction in zip(rows, probabilities, predictions):
            item = {key: value for key, value in row.items() if key != "feature"}
            item.update(
                predicted_class_id=int(prediction),
                probability_background=float(values[int(np.flatnonzero(estimator.classes_ == 0)[0])]),
                probability_l4=float(values[int(np.flatnonzero(estimator.classes_ == 1)[0])]),
                probability_l5=float(values[int(np.flatnonzero(estimator.classes_ == 2)[0])]),
            )
            scored.append(item)
            if int(prediction) != 0:
                predicted[frame_hash].add(int(prediction))
        scored_rows[frame_hash] = scored

    truth = {frame: values - {0} for frame, values in selected_truth.items()}
    for frame in frame_rows:
        truth.setdefault(frame, set())
        predicted.setdefault(frame, set())
    l4_counts = binary_counts(truth, predicted, 1)
    l5_counts = binary_counts(truth, predicted, 2)
    if {key: l4_counts[key] for key in EXPECTED_L4} != EXPECTED_L4:
        raise RuntimeError(f"L4 baseline mismatch: {l4_counts}")
    if {key: l5_counts[key] for key in EXPECTED_L5} != EXPECTED_L5:
        raise RuntimeError(f"L5 baseline mismatch: {l5_counts}")

    fp_frames = sorted(frame for frame in frame_rows if 1 not in truth[frame] and 1 in predicted[frame])
    fn_frames = sorted(frame for frame in frame_rows if 1 in truth[frame] and 1 not in predicted[frame])
    geometry = load_geometry(args.source_zip.resolve())
    pose_cache: dict[Path, list[dict[str, Any]]] = {}
    local_cache: dict[Path, np.ndarray] = {}
    proposal_records: list[dict[str, Any]] = []
    fp_records: list[dict[str, Any]] = []
    fn_records: list[dict[str, Any]] = []
    contact: list[tuple[str, Path]] = []

    for kind, frames in (("FP", fp_frames), ("FN", fn_frames)):
        for index, frame_hash in enumerate(frames, 1):
            failure_id = f"L4_{kind}_{index:02d}"
            rows = scored_rows[frame_hash]
            image = image_cache[frame_hash]
            projected = frame_projection(
                rows[0], geometry, args.surface_slab_tolerance_mm, pose_cache, local_cache
            )
            clean_path = assets / f"{failure_id}_clean.png"
            context_path = assets / f"{failure_id}_context.png"
            overlay_path = assets / f"{failure_id}_proxy_overlay.png"
            display_image(clean_path, image, f"{failure_id}: clean actor-visible ultrasound")
            export_context(context_path, image, rows, failure_id)
            export_overlay(overlay_path, image, rows, projected, failure_id)
            contact.append((failure_id, context_path))

            accepted = [row for row in rows if int(row["predicted_class_id"]) == 1]
            max_row = max(rows, key=lambda row: float(row["probability_l4"]))
            crop_rows = accepted if accepted else [max_row]
            for crop_index, row in enumerate(crop_rows, 1):
                crop_path = crops / f"{failure_id}_proposal_{crop_index:02d}.png"
                export_proposal_crop(crop_path, image, row, failure_id)
                row["crop_asset"] = str(crop_path.relative_to(output))
            for row in rows:
                proposal_records.append(
                    {
                        "failure_id": failure_id,
                        "failure_kind": kind,
                        "session_id": row["session_id"],
                        "frame_sequence": row["frame_sequence"],
                        "frame_sha256": frame_hash,
                        "sample_id": row["sample_id"],
                        "track_id": row["track_id"],
                        "patch_row": row["patch_row"],
                        "patch_column": row["patch_column"],
                        "probability_background": row["probability_background"],
                        "probability_l4": row["probability_l4"],
                        "probability_l5": row["probability_l5"],
                        "predicted_class_id": row["predicted_class_id"],
                        "l4_core_points_in_patch": row["l4_core_points_in_patch"],
                        "l5_core_points_in_patch": row["l5_core_points_in_patch"],
                        "l4_uncertain_points_in_patch": row["l4_uncertain_points_in_patch"],
                        "l5_uncertain_points_in_patch": row["l5_uncertain_points_in_patch"],
                        "adequate_projected_support": row["adequate_projected_support"],
                        "crop_asset": row.get("crop_asset", ""),
                    }
                )

            common = {
                "failure_id": failure_id,
                "session_id": rows[0]["session_id"],
                "frame_sequence": rows[0]["frame_sequence"],
                "frame_sha256": frame_hash,
                "proposal_count": len(rows),
                "accepted_l4_proposal_count": len(accepted),
                "maximum_l4_probability": float(max_row["probability_l4"]),
                "maximum_l4_sample_id": max_row["sample_id"],
                "projected_l4_core_pixel_count": len(
                    set(zip(projected[1]["core_rows"], projected[1]["core_columns"]))
                ),
                "projected_l5_core_pixel_count": len(
                    set(zip(projected[2]["core_rows"], projected[2]["core_columns"]))
                ),
                "clean_asset": str(clean_path.relative_to(output)),
                "context_asset": str(context_path.relative_to(output)),
                "proxy_overlay_asset": str(overlay_path.relative_to(output)),
            }
            if kind == "FP":
                primary, evidence, direction = classify_false_positive(rows)
                fp_records.append(
                    {
                        **common,
                        "accepted_l4_sample_ids": ";".join(row["sample_id"] for row in accepted),
                        "primary_evidence_classification": primary,
                        "supporting_evidence": "; ".join(evidence),
                        "corresponding_test_direction": direction,
                    }
                )
            else:
                selected_l4 = [
                    row for row in rows if row["sample_id"] in {
                        selected["sample_id"]
                        for selected in selected_rows
                        if selected["frame_sha256"] == frame_hash
                        and int(selected["proxy_class_id"]) == 1
                    }
                ]
                truth_row = selected_l4[0] if selected_l4 else max_row
                fn_records.append(
                    {
                        **common,
                        "selected_proxy_sample_id": truth_row["sample_id"],
                        "selected_proxy_l4_probability": truth_row["probability_l4"],
                        "selected_proxy_background_probability": truth_row["probability_background"],
                        "evidence": "no actor-visible proposal passes the frozen L4 decision rule",
                        "corresponding_test_direction": "measure a calibration-selected L4 score margin with the checkpoint frozen",
                    }
                )

    write_csv(output / "l4_false_positive_audit.csv", fp_records)
    write_csv(output / "l4_false_negative_audit.csv", fn_records)
    write_csv(output / "l4_failure_proposal_scores.csv", proposal_records)

    session_rows: list[dict[str, Any]] = []
    sessions = sorted({rows[0]["session_id"] for rows in frame_rows.values()})
    for session in sessions:
        frames = [frame for frame, rows in frame_rows.items() if rows[0]["session_id"] == session]
        for class_id in (1, 2):
            local_truth = {frame: truth[frame] for frame in frames}
            local_predicted = {frame: predicted[frame] for frame in frames}
            values = binary_counts(local_truth, local_predicted, class_id)
            session_rows.append(
                {
                    "session_id": session,
                    "class_name": CLASS_NAMES[class_id],
                    "image_count": len(frames),
                    **values,
                    "precision": None if values["tp"] + values["fp"] == 0 else values["tp"] / (values["tp"] + values["fp"]),
                    "recall": None if values["tp"] + values["fn"] == 0 else values["tp"] / (values["tp"] + values["fn"]),
                }
            )
    write_csv(output / "session_l4_l5_comparison.csv", session_rows)

    fig, axes = plt.subplots(5, 4, figsize=(14, 17), dpi=120)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (failure_id, path) in zip(axes.flat, contact):
        axis.imshow(plt.imread(path))
        axis.set_title(failure_id, fontsize=8)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output / "l4_failure_contact_sheet.png")
    plt.close(fig)

    cause_counts = Counter(row["primary_evidence_classification"] for row in fp_records)
    result = {
        "schema": "sonogym-tp-surface-core-proxy-l4-failure-audit-v1",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "fit_called": False,
        "thresholds_unchanged": {str(key): value for key, value in thresholds.items()},
        "baseline_reproduced": {"L4": l4_counts, "L5": l5_counts},
        "l4_false_positive_count": len(fp_records),
        "l4_false_negative_count": len(fn_records),
        "l4_false_positive_primary_classifications": dict(cause_counts),
        "source_image_count": len(frame_rows),
        "surface_slab_tolerance_mm": args.surface_slab_tolerance_mm,
        "status": "ACTIVE_ACQUISITION_INTEGRATION_HOLD",
        "limitations": [
            "privileged projection is used only after frozen actor predictions",
            "projection overlap does not prove acoustic or anatomical identity",
            "opened s0010 development data cannot serve as independent confirmation",
        ],
    }
    (output / "audit_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
