#!/usr/bin/env python3
"""Prepare auditable RIGHT L4/L5 TP surface-core review samples from verified captures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

EVALUATION_ROOT = (
    Path(__file__).resolve().parents[2]
    / "source/spinal_surgery/spinal_surgery/evaluation"
)
sys.path.insert(0, str(EVALUATION_ROOT))
from two_stage_navigation.stage1_features import (  # noqa: E402
    UNITY_REFERENCE_PATCH_SPEC,
    extract_interface_patch,
    normalized_patch_intensity,
    patch_feature,
)
from two_stage_navigation.tp_surface_core_dataset import (  # noqa: E402
    BACKGROUND_CLASS,
    L4_CLASS,
    L5_CLASS,
    actor_visible_patch_proposals,
    count_points_in_patch,
    projected_pixel_diagnostics,
    proposed_review_target,
)

GEOMETRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/ultrasound_plane_geometry.py"
)
GEOMETRY_SPEC = importlib.util.spec_from_file_location("tp_surface_core_geometry", GEOMETRY_PATH)
if GEOMETRY_SPEC is None or GEOMETRY_SPEC.loader is None:
    raise RuntimeError("cannot load CPU ultrasound-plane geometry")
GEOMETRY = importlib.util.module_from_spec(GEOMETRY_SPEC)
sys.modules[GEOMETRY_SPEC.name] = GEOMETRY
GEOMETRY_SPEC.loader.exec_module(GEOMETRY)
plane_from_full_pose = GEOMETRY.plane_from_full_pose
project_points_to_pixels = GEOMETRY.project_points_to_pixels


ACTIVE_CANDIDATES = (
    "L4_RIGHT_transverse_process",
    "L5_RIGHT_transverse_process",
)
CANDIDATE_TO_CLASS = {
    "L4_RIGHT_transverse_process": L4_CLASS,
    "L5_RIGHT_transverse_process": L5_CLASS,
}
REVIEW_CLASSES = (BACKGROUND_CLASS, L4_CLASS, L5_CLASS)
REVIEW_MINIMUM_UNIQUE_IMAGES = {
    "train": 40,
    "calibration": 10,
    "development_evaluation": 10,
}
REVIEW_RESERVE_IMAGES = {
    "train": 12,
    "calibration": 5,
    "development_evaluation": 5,
}
PREPROCESSING_SPEC = {
    "patch_width_columns": 48,
    "patch_depth_rows": 80,
    "interface_column_offset": 24,
    "interface_row_offset": 34,
    "feature_size": [16, 16],
    "normalization_quantiles": [0.05, 0.95],
    "feature_layout": "intensity_16x16,gradient_x_16x16,gradient_y_16x16,summary_6",
}
SELECTION_SETTINGS = {
    "core_support_units": "unique projected image pixels and occupied 16x16 feature cells",
    "minimum_core_feature_cells": 2,
    "minimum_interior_core_feature_cells": 1,
    "maximum_nearest_core_distance_feature_cells": 3.0,
    "minimum_near_interface_core_feature_cells": 1,
    "feature_cell_size_source_pixels": [5.0, 3.0],
    "background_order": "normalized gradient p90, normalized intensity std, session/column diversity",
    "positive_order": "interior cells, near-interface cells, occupied cells, distance, border fraction",
    "rationale": (
        "Settings are fixed from the 48x80 to 16x16 preprocessing geometry: a target must occupy "
        "more than one feature cell, include a non-border cell, intersect the interface-row "
        "neighborhood, and lie within three feature cells of the actor-visible proposal centre."
    ),
    "development_tuning": False,
}


def canonical_json_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


PREPROCESSING_SPEC_SHA256 = canonical_json_sha256(PREPROCESSING_SPEC)
SELECTION_SETTINGS_SHA256 = canonical_json_sha256(SELECTION_SETTINGS)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def review_support_is_adequate(row: dict[str, Any], class_name: str) -> bool:
    if not row_is_review_candidate(row, class_name):
        return False
    if class_name == BACKGROUND_CLASS:
        return True
    return bool(row["adequate_projected_support"])


def candidate_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    if row["candidate_id"] == BACKGROUND_CLASS:
        return (
            -float(row["patch_gradient_p90"]),
            -float(row["patch_normalized_std"]),
            str(row["sample_id"]),
        )
    return (
        -int(row["interior_core_feature_cells"]),
        -int(row["near_interface_core_feature_cells"]),
        -int(row["core_feature_cells"]),
        float(row["nearest_core_distance_feature_cells"]),
        float(row["border_core_feature_cell_fraction"]),
        -int(row["unique_core_pixels"]),
        str(row["sample_id"]),
    )


def best_candidate_per_source_image(
    rows: list[dict[str, Any]], class_name: str, split: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["split"] != split or not review_support_is_adequate(row, class_name):
            continue
        grouped.setdefault(str(row["frame_sha256"]), []).append(row)
    return [min(candidates, key=candidate_rank) for candidates in grouped.values()]


def diversified_selection(
    candidates: list[dict[str, Any]], limit: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    session_counts: dict[str, int] = {}
    column_counts: dict[int, int] = {}
    while remaining and len(selected) < limit:
        chosen = min(
            remaining,
            key=lambda row: (
                session_counts.get(str(row["session_id"]), 0),
                column_counts.get(int(float(row["patch_column"])) // 12, 0),
                candidate_rank(row),
            ),
        )
        remaining.remove(chosen)
        selected.append(chosen)
        session = str(chosen["session_id"])
        column_bin = int(float(chosen["patch_column"])) // 12
        session_counts[session] = session_counts.get(session, 0) + 1
        column_counts[column_bin] = column_counts.get(column_bin, 0) + 1
    return selected, remaining


def row_is_review_candidate(row: dict[str, Any], class_name: str) -> bool:
    """Return whether a row can be reviewed for the declared class."""
    if row["candidate_id"] != class_name:
        return False
    if class_name == BACKGROUND_CLASS:
        return row["review_target"] == "REVIEW_HARD_NEGATIVE"
    return row["review_target"] == "REVIEW_POSITIVE"


def candidate_unique_image_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count independent source images, not correlated patches."""
    return {
        class_name: {
            split: len(
                {
                    row["frame_sha256"]
                    for row in rows
                    if row["split"] == split and review_support_is_adequate(row, class_name)
                }
            )
            for split in REVIEW_MINIMUM_UNIQUE_IMAGES
        }
        for class_name in REVIEW_CLASSES
    }


def candidate_coverage_deficits(
    counts: dict[str, dict[str, int]],
) -> list[str]:
    """Describe pre-review source-image shortfalls without promoting labels."""
    return [
        f"{class_name} {split}: {counts[class_name][split]} unique candidate images < {minimum}"
        for class_name in REVIEW_CLASSES
        for split, minimum in REVIEW_MINIMUM_UNIQUE_IMAGES.items()
        if counts[class_name][split] < minimum
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def locate_geometry(root: Path, candidate: str) -> Path:
    matches = list(root.rglob(f"{candidate}_revision.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one editable geometry for {candidate}, found {len(matches)}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, action="append", required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-dataset-dir", type=Path)
    return parser.parse_args()


def load_source_image(row: dict[str, Any], cache: dict[Path, np.ndarray]) -> np.ndarray:
    path = Path(str(row["image_archive"]))
    if path not in cache:
        cache[path] = np.load(path)["ultrasound"]
    return cache[path][int(row["image_archive_index"])]


def save_clean_grayscale(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.asarray(array), cmap="gray")


def crop_projection(
    row: dict[str, Any], projection: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    row_start = int(
        row.get(
            "patch_row_start",
            round(float(row["patch_row"]) - UNITY_REFERENCE_PATCH_SPEC.interface_offset_pixels),
        )
    )
    column_start = int(
        row.get(
            "patch_column_start",
            round(float(row["patch_column"]) - UNITY_REFERENCE_PATCH_SPEC.width_pixels / 2.0),
        )
    )
    inside = (
        (projection["rows"] >= row_start)
        & (projection["rows"] < row_start + int(row["patch_depth"]))
        & (projection["columns"] >= column_start)
        & (projection["columns"] < column_start + int(row["patch_width"]))
    )
    return (
        projection["rows"][inside] - row_start,
        projection["columns"][inside] - column_start,
    )


def export_review_sample(
    *,
    output: Path,
    row: dict[str, Any],
    image: np.ndarray,
    projections: dict[str, dict[str, np.ndarray]],
    npz_index: int,
) -> dict[str, Any]:
    raw_patch = extract_interface_patch(
        image,
        float(row["patch_row"]),
        float(row["patch_column"]),
        UNITY_REFERENCE_PATCH_SPEC,
    )
    if raw_patch is None:
        raise RuntimeError(f"selected patch is outside the source image: {row['sample_id']}")
    intensity = normalized_patch_intensity(raw_patch, UNITY_REFERENCE_PATCH_SPEC)
    feature = patch_feature(raw_patch, UNITY_REFERENCE_PATCH_SPEC)
    if array_sha256(raw_patch) != row["raw_crop_sha256"]:
        raise RuntimeError(f"raw crop identity changed: {row['sample_id']}")
    if array_sha256(intensity) != row["preprocessed_intensity_sha256"]:
        raise RuntimeError(f"preprocessed intensity identity changed: {row['sample_id']}")
    if array_sha256(feature) != row["feature_sha256"]:
        raise RuntimeError(f"feature identity changed: {row['sample_id']}")

    asset_name = str(row["sample_id"]).replace(":", "_")
    role = str(row["review_selection"]).lower()
    relative_root = Path("review") / role / asset_name
    clean_path = output / relative_root / "clean_full.png"
    raw_path = output / relative_root / "raw_patch.png"
    intensity_path = output / relative_root / "intensity_16.png"
    context_path = output / relative_root / "context.png"
    overlay_path = output / relative_root / "overlay_patch.png"
    sheet_path = output / relative_root / "review_sheet.png"
    save_clean_grayscale(clean_path, image)
    save_clean_grayscale(raw_path, raw_patch)
    save_clean_grayscale(intensity_path, intensity)

    fig, axis = plt.subplots(figsize=(5, 6), dpi=160)
    axis.imshow(image, cmap="gray")
    rectangle = plt.Rectangle(
        (int(row["patch_column_start"]), int(row["patch_row_start"])),
        int(row["patch_width"]),
        int(row["patch_depth"]),
        fill=False,
        edgecolor="lime",
        linewidth=1.5,
    )
    axis.add_patch(rectangle)
    axis.set_title(f"{row['session_id']} frame {row['frame_sequence']} | patch boundary")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(context_path)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(4, 6), dpi=160)
    axis.imshow(raw_patch, cmap="gray")
    for class_name, projection in projections.items():
        core_rows, core_columns = crop_projection(row, projection)
        uncertain_rows, uncertain_columns = crop_projection(
            row,
            {
                "rows": projection["uncertain_rows"],
                "columns": projection["uncertain_columns"],
            },
        )
        axis.scatter(core_columns, core_rows, s=5, label=f"{class_name} core")
        axis.scatter(uncertain_columns, uncertain_rows, s=4, c="cyan", label="uncertain")
    axis.set_title("Reviewer-only projected geometry")
    axis.set_xlim(-0.5, int(row["patch_width"]) - 0.5)
    axis.set_ylim(int(row["patch_depth"]) - 0.5, -0.5)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(overlay_path)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=150)
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Clean full ultrasound")
    axes[0, 1].imshow(image, cmap="gray")
    axes[0, 1].add_patch(
        plt.Rectangle(
            (int(row["patch_column_start"]), int(row["patch_row_start"])),
            int(row["patch_width"]),
            int(row["patch_depth"]),
            fill=False,
            edgecolor="lime",
        )
    )
    axes[0, 1].set_title("Context only")
    axes[0, 2].imshow(raw_patch, cmap="gray")
    axes[0, 2].set_title("Exact 48 x 80 raw patch")
    axes[1, 0].imshow(intensity, cmap="gray")
    axes[1, 0].set_title("Exact normalized 16 x 16 intensity")
    overlay_image = plt.imread(overlay_path)
    axes[1, 1].imshow(overlay_image)
    axes[1, 1].set_title("Separate geometry overlay")
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0,
        1.0,
        "\n".join(
            (
                str(row["sample_id"]),
                f"selection={row['review_selection']}",
                f"core pixels={row['unique_core_pixels']}",
                f"feature cells={row['core_feature_cells']}",
                f"interior cells={row['interior_core_feature_cells']}",
                f"nearest={float(row['nearest_core_distance_feature_cells']):.3f} cells",
                f"border fraction={float(row['border_core_feature_cell_fraction']):.3f}",
                f"raw={row['raw_crop_sha256'][:16]}...",
                f"feature={row['feature_sha256'][:16]}...",
            )
        ),
        va="top",
        family="monospace",
        fontsize=8,
    )
    for axis in axes.flat[:5]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(sheet_path)
    plt.close(fig)
    row["review_asset"] = str(sheet_path.relative_to(output))
    row["clean_full_asset"] = str(clean_path.relative_to(output))
    row["context_asset"] = str(context_path.relative_to(output))
    row["raw_patch_asset"] = str(raw_path.relative_to(output))
    row["preprocessed_intensity_asset"] = str(intensity_path.relative_to(output))
    row["overlay_asset"] = str(overlay_path.relative_to(output))
    row["selected_array_index"] = npz_index
    return {
        "sample_id": row["sample_id"],
        "array_index": npz_index,
        "raw_patch_sha256": row["raw_crop_sha256"],
        "preprocessed_intensity_sha256": row["preprocessed_intensity_sha256"],
        "feature_sha256": row["feature_sha256"],
        "review_selection": row["review_selection"],
    }


def draw_crop_overlay(
    axis: Any,
    *,
    row: dict[str, Any],
    image: np.ndarray,
    projections: dict[str, dict[str, np.ndarray]],
) -> None:
    patch = extract_interface_patch(
        image,
        float(row["patch_row"]),
        float(row["patch_column"]),
        UNITY_REFERENCE_PATCH_SPEC,
    )
    if patch is None:
        raise RuntimeError(f"comparison patch is invalid: {row['sample_id']}")
    axis.imshow(patch, cmap="gray")
    for projection in projections.values():
        core_rows, core_columns = crop_projection(row, projection)
        uncertain_rows, uncertain_columns = crop_projection(
            row,
            {
                "rows": projection["uncertain_rows"],
                "columns": projection["uncertain_columns"],
            },
        )
        axis.scatter(core_columns, core_rows, s=5)
        axis.scatter(uncertain_columns, uncertain_rows, s=4, c="cyan")
    axis.axis("off")


def export_before_after_comparisons(
    *,
    output: Path,
    prior_dataset: Path,
    current_rows: list[dict[str, Any]],
    primary_rows: list[dict[str, Any]],
    frame_contexts: dict[tuple[str, int], dict[str, Any]],
) -> None:
    prior_rows = read_csv(prior_dataset / "candidate_sample_manifest.csv")
    prior_by_id = {row["sample_id"]: row for row in prior_rows}
    current_by_frame_class: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in current_rows:
        if row_is_review_candidate(row, str(row["candidate_id"])):
            current_by_frame_class.setdefault(
                (str(row["session_id"]), int(row["frame_sequence"]), str(row["candidate_id"])),
                [],
            ).append(row)
    comparisons = (
        ("L4-06", "s0010tpscv6_train_00:12:H0_BASELINE:2"),
        ("L5-52", "s0010tpscv6_dev_00:1:H0_BASELINE:0"),
    )
    comparison_dir = output / "review" / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[Path, np.ndarray] = {}
    for label, sample_id in comparisons:
        old = prior_by_id.get(sample_id)
        if old is None:
            raise RuntimeError(f"prior comparison sample is missing: {sample_id}")
        key = (str(old["session_id"]), int(old["frame_sequence"]), str(old["candidate_id"]))
        candidates = current_by_frame_class.get(key, [])
        if not candidates:
            raise RuntimeError(f"no current comparison candidates for {sample_id}")
        new = min(candidates, key=candidate_rank)
        context = frame_contexts[(key[0], key[1])]
        old_image = load_source_image(old, cache)
        new_image = np.asarray(context["image"])
        fig, axes = plt.subplots(2, 2, figsize=(8, 10), dpi=160)
        old_patch = extract_interface_patch(
            old_image,
            float(old["patch_row"]),
            float(old["patch_column"]),
            UNITY_REFERENCE_PATCH_SPEC,
        )
        new_patch = extract_interface_patch(
            new_image,
            float(new["patch_row"]),
            float(new["patch_column"]),
            UNITY_REFERENCE_PATCH_SPEC,
        )
        if old_patch is None or new_patch is None:
            raise RuntimeError("comparison source patch is invalid")
        axes[0, 0].imshow(old_patch, cmap="gray")
        axes[0, 0].set_title(f"Before raw patch\n{old['sample_id']}")
        axes[0, 1].imshow(new_patch, cmap="gray")
        axes[0, 1].set_title(f"After raw patch\n{new['sample_id']}")
        draw_crop_overlay(
            axes[1, 0], row=old, image=old_image, projections=context["projections"]
        )
        axes[1, 0].set_title("Before geometry overlay")
        draw_crop_overlay(
            axes[1, 1], row=new, image=new_image, projections=context["projections"]
        )
        axes[1, 1].set_title(
            "After geometry overlay\n"
            f"unique pixels={new['unique_core_pixels']}, cells={new['core_feature_cells']}"
        )
        for axis in axes.flat[:2]:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(comparison_dir / f"{label}_before_after.png")
        plt.close(fig)

    old_background = [
        row
        for row in prior_rows
        if row.get("review_asset")
        and row["candidate_id"] == BACKGROUND_CLASS
        and row["split"] == "train"
    ][:6]
    new_background = [
        row
        for row in primary_rows
        if row["candidate_id"] == BACKGROUND_CLASS and row["split"] == "train"
    ][:6]
    if len(old_background) == 6 and len(new_background) == 6:
        fig, axes = plt.subplots(2, 6, figsize=(15, 6), dpi=150)
        for column, row in enumerate(old_background):
            image = load_source_image(row, cache)
            patch = extract_interface_patch(
                image,
                float(row["patch_row"]),
                float(row["patch_column"]),
                UNITY_REFERENCE_PATCH_SPEC,
            )
            axes[0, column].imshow(patch, cmap="gray")
            axes[0, column].set_title(f"Before\n{row['session_id']} c={row['patch_column']}", fontsize=7)
        for column, row in enumerate(new_background):
            context = frame_contexts[(str(row["session_id"]), int(row["frame_sequence"]))]
            patch = extract_interface_patch(
                context["image"],
                float(row["patch_row"]),
                float(row["patch_column"]),
                UNITY_REFERENCE_PATCH_SPEC,
            )
            axes[1, column].imshow(patch, cmap="gray")
            axes[1, column].set_title(f"After\n{row['session_id']} c={row['patch_column']}", fontsize=7)
        for axis in axes.flat:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(comparison_dir / "background_before_after.png")
        plt.close(fig)


def main() -> int:
    args = parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if not protocol["protocol_id"].startswith("sonogym-s0010-tp-surface-core-dataset-v"):
        raise RuntimeError("unexpected protocol")
    capture_roots = [path.resolve() for path in args.capture_root]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    geometries: dict[str, dict[str, Any]] = {}
    for candidate in ACTIVE_CANDIDATES:
        path = locate_geometry(args.geometry_root.resolve(), candidate)
        archive = np.load(path)
        vertices = archive["vertices_mesh_index"].astype(np.float64)
        core = archive["proposed_revised_vertex_mask"].astype(bool)
        uncertain = archive["boundary_uncertainty_vertex_mask"].astype(bool)
        if not core.any():
            raise RuntimeError(f"{candidate} has no proposed surface-core geometry")
        geometries[candidate] = {
            "path": path,
            "sha256": sha256(path),
            "vertices": vertices,
            "core": core,
            "uncertain": uncertain,
        }

    rows: list[dict[str, Any]] = []
    missing_sessions: list[str] = []
    processed_sessions: list[str] = []
    frame_contexts: dict[tuple[str, int], dict[str, Any]] = {}
    for session in protocol["sessions"]:
        matching_session_dirs = [
            root / "sessions" / session["session_id"]
            for root in capture_roots
            if (root / "sessions" / session["session_id"]).is_dir()
        ]
        # A later capture root may preserve an earlier verified session through a
        # symlink (V9 retains the V8 smoke session).  Treat references to the
        # same canonical directory as one source, while continuing to fail on
        # genuinely different directories with the same session identifier.
        matching_session_dirs = list(
            dict.fromkeys(path.resolve() for path in matching_session_dirs)
        )
        if not matching_session_dirs:
            missing_sessions.append(session["session_id"])
            continue
        if len(matching_session_dirs) != 1:
            raise RuntimeError(f"session appears in multiple capture roots: {session['session_id']}")
        session_dir = matching_session_dirs[0]
        processed_sessions.append(session["session_id"])
        report = json.loads((session_dir / "verification_report.json").read_text())
        if report["verdict"] != "PROJECTION_PARITY_PASS":
            raise RuntimeError(f"unverified session: {session['session_id']}")
        public = read_jsonl(session_dir / "public/frames.jsonl")
        poses = read_jsonl(session_dir / "privileged/full_pose_replay.jsonl")
        images = np.load(session_dir / "public/ultrasound.npz")["ultrasound"]
        if not (len(public) == len(poses) == len(images)):
            raise RuntimeError("capture records are misaligned")
        for index, (actor, pose) in enumerate(zip(public, poses)):
            if actor["acquisition_id"] != pose["acquisition_id"]:
                raise RuntimeError("acquisition identity mismatch")
            local = np.load(session_dir / "privileged/pixel_to_volume_coordinates.npz")[
                "native_local_image_coordinates_m"
            ]
            plane = plane_from_full_pose(
                end_effector_position_m=np.asarray(pose["human_to_ee_position_m"]),
                rotation_columns=np.asarray(pose["human_to_ee_rotation_columns"]),
                image_origin_height_m=float(pose["image_origin_height_m"]),
                local_coordinates_m=local,
                label_resolution_m=float(pose["label_resolution_m"]),
            )
            projections: dict[str, dict[str, np.ndarray]] = {}
            for candidate in ACTIVE_CANDIDATES:
                geometry = geometries[candidate]
                projected = project_points_to_pixels(
                    points_index=geometry["vertices"][geometry["core"]],
                    image_origin_index=plane.origin_index,
                    rotation_columns=plane.rotation_columns,
                    width=150,
                    depth=200,
                    image_resolution_m=float(pose["image_resolution_m"]),
                    label_resolution_m=float(pose["label_resolution_m"]),
                    plane_tolerance_mm=float(protocol["label_preparation"]["surface_slab_tolerance_mm"]),
                )
                uncertain_projection = project_points_to_pixels(
                    points_index=geometry["vertices"][geometry["uncertain"]],
                    image_origin_index=plane.origin_index,
                    rotation_columns=plane.rotation_columns,
                    width=150,
                    depth=200,
                    image_resolution_m=float(pose["image_resolution_m"]),
                    label_resolution_m=float(pose["label_resolution_m"]),
                    plane_tolerance_mm=float(protocol["label_preparation"]["surface_slab_tolerance_mm"]),
                )
                projections[CANDIDATE_TO_CLASS[candidate]] = {
                    "rows": projected["rows"][projected["inside"]],
                    "columns": projected["columns"][projected["inside"]],
                    "uncertain_rows": uncertain_projection["rows"][uncertain_projection["inside"]],
                    "uncertain_columns": uncertain_projection["columns"][uncertain_projection["inside"]],
                }
            frame_contexts[(session["session_id"], index)] = {
                "image": images[index],
                "projections": projections,
            }
            proposals = actor_visible_patch_proposals(
                images[index],
                patch_spec=UNITY_REFERENCE_PATCH_SPEC,
                column_stride_pixels=int(
                    protocol["label_preparation"].get("proposal_column_stride_pixels", 12)
                ),
            )
            for proposal_index, proposal in enumerate(proposals):
                core_counts = {
                    name: count_points_in_patch(
                        proposal,
                        rows=value["rows"],
                        columns=value["columns"],
                        patch_spec=UNITY_REFERENCE_PATCH_SPEC,
                    )
                    for name, value in projections.items()
                }
                uncertain_counts = {
                    name: count_points_in_patch(
                        proposal,
                        rows=value["uncertain_rows"],
                        columns=value["uncertain_columns"],
                        patch_spec=UNITY_REFERENCE_PATCH_SPEC,
                    )
                    for name, value in projections.items()
                }
                review_target, class_id, target_reason = proposed_review_target(
                    core_counts, uncertain_counts
                )
                candidate_id = next(
                    (name for name, value in core_counts.items() if value > 0),
                    "BACKGROUND_OR_OTHER_STRUCTURE",
                )
                active_projection = projections.get(
                    candidate_id,
                    {
                        "rows": np.empty(0, dtype=int),
                        "columns": np.empty(0, dtype=int),
                        "uncertain_rows": np.empty(0, dtype=int),
                        "uncertain_columns": np.empty(0, dtype=int),
                    },
                )
                diagnostics = projected_pixel_diagnostics(
                    proposal,
                    core_rows=active_projection["rows"],
                    core_columns=active_projection["columns"],
                    uncertain_rows=active_projection["uncertain_rows"],
                    uncertain_columns=active_projection["uncertain_columns"],
                    patch_spec=UNITY_REFERENCE_PATCH_SPEC,
                )
                raw_patch = extract_interface_patch(
                    images[index], proposal.row, proposal.column, UNITY_REFERENCE_PATCH_SPEC
                )
                if raw_patch is None:
                    raise AssertionError("actor-visible proposal produced an invalid source patch")
                intensity = normalized_patch_intensity(raw_patch, UNITY_REFERENCE_PATCH_SPEC)
                feature = patch_feature(raw_patch, UNITY_REFERENCE_PATCH_SPEC)
                if not np.array_equal(feature, proposal.feature):
                    raise AssertionError("proposal feature differs from production patch feature")
                record = {
                    "sample_id": f"{session['session_id']}:{index}:{proposal.track_id}:{proposal_index}",
                    "session_id": session["session_id"],
                    "split": session["split"],
                    "frame_sequence": index,
                    "acquisition_id": actor["acquisition_id"],
                    "frame_sha256": actor["frame_sha256"],
                    "image_archive": str((session_dir / "public/ultrasound.npz").resolve()),
                    "image_archive_index": index,
                    "proposal_source": "ACTOR_VISIBLE_BRIGHT_INTERFACE",
                    "track_id": proposal.track_id,
                    "candidate_id": candidate_id,
                    "class_id": -1 if class_id is None else class_id,
                    "geometry_version_sha256": json.dumps(
                        {name: geometries[candidate]["sha256"] for candidate, name in CANDIDATE_TO_CLASS.items()},
                        sort_keys=True,
                    ),
                    "projection_status": "VERIFIED",
                    "candidate_intersection": review_target,
                    "l4_core_points_in_patch": core_counts[L4_CLASS],
                    "l5_core_points_in_patch": core_counts[L5_CLASS],
                    "l4_uncertain_points_in_patch": uncertain_counts[L4_CLASS],
                    "l5_uncertain_points_in_patch": uncertain_counts[L5_CLASS],
                    "patch_row": proposal.row,
                    "patch_column": proposal.column,
                    "patch_width": 48,
                    "patch_depth": 80,
                    **diagnostics,
                    "patch_normalized_std": float(feature[-5]),
                    "patch_gradient_mean": float(feature[-2]),
                    "patch_gradient_p90": float(feature[-1]),
                    "raw_crop_sha256": array_sha256(raw_patch),
                    "preprocessed_intensity_sha256": array_sha256(intensity),
                    "feature_sha256": array_sha256(feature),
                    "preprocessing_spec_sha256": PREPROCESSING_SPEC_SHA256,
                    "selection_settings_sha256": SELECTION_SETTINGS_SHA256,
                    "acoustic_visibility": "NOT_REVIEWED",
                    "semantic_identity": "NOT_REVIEWED",
                    "anatomical_approval": "PENDING",
                    "training_label": "EXCLUDED",
                    "training_eligible": False,
                    "review_target": review_target,
                    "review_target_reason": target_reason,
                    "exclusion_reason": "anatomical and patch-semantic review pending",
                    "review_asset": "",
                    "review_selection": "NONE",
                    "selection_reason": "",
                    "clean_full_asset": "",
                    "context_asset": "",
                    "raw_patch_asset": "",
                    "preprocessed_intensity_asset": "",
                    "overlay_asset": "",
                    "selected_array_index": -1,
                }
                rows.append(record)

    primary_rows: list[dict[str, Any]] = []
    reserve_rows: list[dict[str, Any]] = []
    best_rows_by_image: dict[tuple[str, str, str], dict[str, Any]] = {}
    for class_name in REVIEW_CLASSES:
        for split, minimum in REVIEW_MINIMUM_UNIQUE_IMAGES.items():
            candidates = best_candidate_per_source_image(rows, class_name, split)
            for row in candidates:
                best_rows_by_image[(class_name, split, str(row["frame_sha256"]))] = row
            selected, remaining = diversified_selection(candidates, minimum)
            reserves, _ = diversified_selection(remaining, REVIEW_RESERVE_IMAGES[split])
            for rank, row in enumerate(selected):
                row["review_selection"] = "PRIMARY"
                row["selection_reason"] = (
                    f"best supported proposal for source image; diversified rank {rank}; "
                    f"settings {SELECTION_SETTINGS_SHA256}"
                )
            for rank, row in enumerate(reserves):
                row["review_selection"] = "RESERVE"
                row["selection_reason"] = (
                    f"reserve proposal after primary quota; diversified rank {rank}; "
                    f"settings {SELECTION_SETTINGS_SHA256}"
                )
            primary_rows.extend(selected)
            reserve_rows.extend(reserves)

    selected_rows = primary_rows + reserve_rows
    raw_patches: list[np.ndarray] = []
    intensities: list[np.ndarray] = []
    features: list[np.ndarray] = []
    selected_index_rows: list[dict[str, Any]] = []
    for npz_index, row in enumerate(selected_rows):
        context = frame_contexts[(str(row["session_id"]), int(row["frame_sequence"]))]
        image = np.asarray(context["image"])
        index_row = export_review_sample(
            output=output,
            row=row,
            image=image,
            projections=context["projections"],
            npz_index=npz_index,
        )
        raw_patch = extract_interface_patch(
            image,
            float(row["patch_row"]),
            float(row["patch_column"]),
            UNITY_REFERENCE_PATCH_SPEC,
        )
        if raw_patch is None:
            raise AssertionError("selected patch became invalid during array export")
        intensity = normalized_patch_intensity(raw_patch, UNITY_REFERENCE_PATCH_SPEC)
        feature = patch_feature(raw_patch, UNITY_REFERENCE_PATCH_SPEC)
        raw_patches.append(raw_patch)
        intensities.append(intensity)
        features.append(feature)
        index_row.update(
            session_id=row["session_id"],
            split=row["split"],
            frame_sha256=row["frame_sha256"],
            source_array_index=row["image_archive_index"],
            track_id=row["track_id"],
            patch_row=row["patch_row"],
            patch_column=row["patch_column"],
            patch_row_start=row["patch_row_start"],
            patch_column_start=row["patch_column_start"],
            geometry_version_sha256=row["geometry_version_sha256"],
            preprocessing_spec_sha256=PREPROCESSING_SPEC_SHA256,
        )
        selected_index_rows.append(index_row)
    if selected_rows:
        np.savez_compressed(
            output / "selected_patch_arrays.npz",
            raw_patches=np.stack(raw_patches),
            normalized_intensity_16=np.stack(intensities),
            feature_vectors=np.stack(features),
        )
    write_csv(output / "selected_patch_index.csv", selected_index_rows)
    write_csv(output / "reserve_candidate_manifest.csv", reserve_rows)
    (output / "preprocessing_spec.json").write_text(
        json.dumps(
            {"sha256": PREPROCESSING_SPEC_SHA256, "specification": PREPROCESSING_SPEC},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "selection_settings.json").write_text(
        json.dumps(
            {"sha256": SELECTION_SETTINGS_SHA256, "settings": SELECTION_SETTINGS},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.prior_dataset_dir is not None:
        export_before_after_comparisons(
            output=output,
            prior_dataset=args.prior_dataset_dir.resolve(),
            current_rows=rows,
            primary_rows=primary_rows,
            frame_contexts=frame_contexts,
        )

    write_csv(output / "candidate_sample_manifest.csv", rows)
    split_rows = [
        {
            "session_id": session["session_id"],
            "role": session["role"],
            "split": session["split"],
            "prior_inspection": "new_opened_development_capture",
        }
        for session in protocol["sessions"]
        if session["role"] != "smoke"
    ]
    write_csv(output / "session_split_manifest.csv", split_rows)
    unique_image_counts = candidate_unique_image_counts(rows)
    geometric_candidate_counts = {
        class_name: {
            split: len(
                {
                    row["frame_sha256"]
                    for row in rows
                    if row["split"] == split and row_is_review_candidate(row, class_name)
                }
            )
            for split in REVIEW_MINIMUM_UNIQUE_IMAGES
        }
        for class_name in REVIEW_CLASSES
    }
    coverage_deficits = candidate_coverage_deficits(unique_image_counts)
    summary = {
        "schema": "sonogym-tp-surface-core-dataset-preparation-v1",
        "active_classes": list(ACTIVE_CANDIDATES),
        "task_name": "TP_SURFACE_CORE detection",
        "rows": len(rows),
        "verified_intersections": sum(row["review_target"] == "REVIEW_POSITIVE" for row in rows),
        "training_eligible_rows": 0,
        "anatomical_approval": "PENDING",
        "processed_sessions": processed_sessions,
        "missing_sessions": missing_sessions,
        "geometric_candidate_unique_image_counts": geometric_candidate_counts,
        "candidate_unique_image_counts": unique_image_counts,
        "candidate_coverage_deficits": coverage_deficits,
        "review_asset_count": sum(bool(row["review_asset"]) for row in rows),
        "primary_review_count": len(primary_rows),
        "reserve_review_count": len(reserve_rows),
        "selection_settings_sha256": SELECTION_SETTINGS_SHA256,
        "preprocessing_spec_sha256": PREPROCESSING_SPEC_SHA256,
        "status": (
            "DATASET_INCOMPLETE"
            if missing_sessions or coverage_deficits
            else "ENGINEERING_READY_LABEL_REVIEW_PENDING"
        ),
        "counts": {
            CANDIDATE_TO_CLASS[candidate]: {
                split: sum(
                    row["candidate_id"] == CANDIDATE_TO_CLASS[candidate]
                    and row["split"] == split
                    and row["review_target"] == "REVIEW_POSITIVE"
                    for row in rows
                )
                for split in ("train", "calibration", "development_evaluation")
            }
            for candidate in ACTIVE_CANDIDATES
        },
    }
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "label_schema.json").write_text(
        json.dumps(
            {
                "task": "TP_SURFACE_CORE detection",
                "classes": ["BACKGROUND_OR_OTHER_STRUCTURE", L4_CLASS, L5_CLASS],
                "excluded": ["SAP", "whole_TP", "MBB_target"],
                "supervision": "48x80 interface-centred patch classification",
                "feature_mode": "16x16 normalized intensity and gradients",
                "preprocessing_spec_sha256": PREPROCESSING_SPEC_SHA256,
                "selection_settings_sha256": SELECTION_SETTINGS_SHA256,
                "inference_proposals": "actor-visible bright-interface tracks and fixed-stride patches",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
