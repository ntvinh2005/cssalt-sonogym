#!/usr/bin/env python3
"""Audit image-only cortical tracks against matched and mismatched SAP surfaces."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

import correct_s0010_sap_observability_v1 as correction
import run_s0010_sap_observability as obs

INTERFACE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/stage1_interfaces.py"
)
INTERFACE_SPEC = importlib.util.spec_from_file_location(
    "sap_correspondence_interfaces", INTERFACE_PATH
)
if INTERFACE_SPEC is None or INTERFACE_SPEC.loader is None:
    raise RuntimeError("cannot load image-only interface tracker")
INTERFACES = importlib.util.module_from_spec(INTERFACE_SPEC)
sys.modules[INTERFACE_SPEC.name] = INTERFACES
INTERFACE_SPEC.loader.exec_module(INTERFACES)
InterfaceTrack = INTERFACES.InterfaceTrack
track_bright_interfaces = INTERFACES.track_bright_interfaces


ALIGNMENT_TOLERANCE_PIXELS = 4.0
MINIMUM_SURFACE_COLUMNS = 5
MINIMUM_ALIGNED_FRACTION = 0.40
MINIMUM_ALIGNED_RUN_COLUMNS = 5
MAXIMUM_MEDIAN_DISTANCE_PIXELS = 4.0
MINIMUM_LOCAL_GRADIENT = 0.10
MINIMUM_LOCAL_CONTRAST = 0.02
MINIMUM_POSTERIOR_SHADOW = 0.02


@dataclass(frozen=True)
class Alignment:
    track_id: str
    surface_columns: int
    aligned_columns: int
    aligned_fraction: float
    longest_aligned_run_columns: int
    median_distance_pixels: float
    local_gradient_median: float
    local_contrast_median: float
    posterior_shadow_median: float
    maximum_track_jump_pixels: float
    passes: bool


def normalized_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    low, high = np.quantile(array, (0.05, 0.95))
    return np.clip((array - low) / max(float(high - low), 1e-6), 0.0, 1.0)


def longest_consecutive(columns: np.ndarray) -> int:
    if not len(columns):
        return 0
    ordered = np.unique(np.asarray(columns, dtype=np.int64))
    runs = np.split(ordered, np.where(np.diff(ordered) != 1)[0] + 1)
    return max((len(run) for run in runs), default=0)


def track_alignment(
    image: np.ndarray,
    track: InterfaceTrack,
    surface_mask: np.ndarray,
) -> Alignment:
    normalized = normalized_image(image)
    gradient = np.hypot(
        np.gradient(normalized, axis=0), np.gradient(normalized, axis=1)
    )
    surface_rows, surface_columns = np.where(surface_mask)
    unique_columns = np.unique(surface_columns)
    distances: list[float] = []
    track_rows: list[int] = []
    used_columns: list[int] = []
    for column in unique_columns:
        if column < track.columns[0] or column > track.columns[-1]:
            continue
        row = int(round(track.row_at(float(column))))
        candidates = surface_rows[surface_columns == column]
        if not len(candidates):
            continue
        distances.append(float(np.min(np.abs(candidates - row))))
        track_rows.append(row)
        used_columns.append(int(column))
    if not distances:
        return Alignment(
            track.track_id,
            0,
            0,
            0.0,
            0,
            float("inf"),
            0.0,
            0.0,
            0.0,
            float(track.maximum_jump_pixels),
            False,
        )
    distance_array = np.asarray(distances)
    used_columns_array = np.asarray(used_columns, dtype=np.int64)
    rows = np.asarray(track_rows, dtype=np.int64)
    aligned = distance_array <= ALIGNMENT_TOLERANCE_PIXELS
    aligned_rows = rows[aligned]
    aligned_columns = used_columns_array[aligned]
    local_gradient = gradient[aligned_rows, aligned_columns] if len(aligned_rows) else np.asarray([])
    local_contrast: list[float] = []
    posterior_shadow: list[float] = []
    for row, column in zip(aligned_rows, aligned_columns):
        above = normalized[max(0, row - 3) : row + 1, column]
        below = normalized[row + 2 : min(normalized.shape[0], row + 10), column]
        if len(above) and len(below):
            local_contrast.append(float(np.mean(above) - np.mean(below)))
            posterior_shadow.append(float(normalized[row, column] - np.mean(below)))
    aligned_fraction = float(np.mean(aligned))
    longest_run = longest_consecutive(aligned_columns)
    gradient_median = float(np.median(local_gradient)) if len(local_gradient) else 0.0
    contrast_median = float(np.median(local_contrast)) if local_contrast else 0.0
    shadow_median = float(np.median(posterior_shadow)) if posterior_shadow else 0.0
    median_distance = float(np.median(distance_array))
    passes = bool(
        len(used_columns_array) >= MINIMUM_SURFACE_COLUMNS
        and aligned_fraction >= MINIMUM_ALIGNED_FRACTION
        and longest_run >= MINIMUM_ALIGNED_RUN_COLUMNS
        and median_distance <= MAXIMUM_MEDIAN_DISTANCE_PIXELS
        and gradient_median >= MINIMUM_LOCAL_GRADIENT
        and contrast_median >= MINIMUM_LOCAL_CONTRAST
        and shadow_median >= MINIMUM_POSTERIOR_SHADOW
        and track.maximum_jump_pixels <= 4.0
    )
    return Alignment(
        track.track_id,
        len(used_columns_array),
        int(np.count_nonzero(aligned)),
        aligned_fraction,
        longest_run,
        median_distance,
        gradient_median,
        contrast_median,
        shadow_median,
        float(track.maximum_jump_pixels),
        passes,
    )


def best_alignment(image: np.ndarray, surface_mask: np.ndarray) -> Alignment:
    tracks = track_bright_interfaces(image)
    if not tracks:
        return Alignment("NONE", 0, 0, 0.0, 0, float("inf"), 0.0, 0.0, 0.0, 0.0, False)
    rows = [track_alignment(image, track, surface_mask) for track in tracks]
    return max(
        rows,
        key=lambda row: (
            row.passes,
            row.longest_aligned_run_columns,
            row.aligned_fraction,
            -row.median_distance_pixels,
        ),
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    prediction_image = nib.load(args.prediction)
    segmentation_image = nib.load(args.segmentation)
    segmentation = np.asarray(segmentation_image.dataobj)
    masks = obs.pseudo_sap_masks(
        np.asarray(prediction_image.dataobj), segmentation, prediction_image.affine
    )
    bone_ids = correction.bone_source_ids(args.label_conversion)
    manifest = {
        (row["session_id"], int(row["frame_sequence"])): row
        for row in read_csv(args.review_root / "projection_manifest.csv")
    }
    frames: list[dict[str, Any]] = []
    surfaces: dict[tuple[str, int, str], np.ndarray] = {}
    images: dict[tuple[str, int], np.ndarray] = {}
    for session_dir in sorted((args.run_root / "sessions").iterdir()):
        if not session_dir.is_dir():
            continue
        ultrasound = np.load(session_dir / "public/ultrasound.npz")["ultrasound"]
        coordinates = np.load(
            session_dir / "privileged/pixel_to_volume_coordinates.npz"
        )["human_img_coords"]
        raw_labels = np.load(session_dir / "privileged/raw_labels.npz")["raw_labels"]
        public = [json.loads(line) for line in (session_dir / "public/frames.jsonl").read_text().splitlines()]
        for index, image in enumerate(ultrasound):
            key = (session_dir.name, index)
            images[key] = image
            for level in ("L4", "L5"):
                diagnostics = obs.projection_diagnostics(coordinates[index], masks[level])
                surface = correction.first_bone_surface(
                    raw_labels=raw_labels[index],
                    corrected_projection=diagnostics["projected"],
                    bone_ids=bone_ids,
                )
                surfaces[(session_dir.name, index, level)] = surface[
                    "visible_pseudo_surface"
                ]
            matched_level = manifest[key]["level"]
            frames.append(
                {
                    "session_id": session_dir.name,
                    "frame_sequence": index,
                    "frame_sha256": public[index]["frame_sha256"],
                    "matched_level": matched_level,
                }
            )

    comparisons: list[dict[str, Any]] = []
    for frame in frames:
        key = (frame["session_id"], frame["frame_sequence"])
        image = images[key]
        level = frame["matched_level"]
        matched = best_alignment(image, surfaces[(*key, level)])
        comparisons.append({**frame, "comparison": "MATCHED", "projection_source": f"{key[0]}:{key[1]}:{level}", **asdict(matched)})
        other_level = "L5" if level == "L4" else "L4"
        other = best_alignment(image, surfaces[(*key, other_level)])
        comparisons.append({**frame, "comparison": "OTHER_LEVEL", "projection_source": f"{key[0]}:{key[1]}:{other_level}", **asdict(other)})
        candidates = [
            candidate
            for candidate in frames
            if candidate["matched_level"] == level
            and candidate["session_id"] != frame["session_id"]
        ]
        for candidate in candidates[:5]:
            source = (candidate["session_id"], candidate["frame_sequence"], level)
            mismatch = best_alignment(image, surfaces[source])
            comparisons.append({**frame, "comparison": "DISTINCT_POSE", "projection_source": f"{source[0]}:{source[1]}:{level}", **asdict(mismatch)})

    write_csv(args.output_dir / "comparisons.csv", comparisons)
    summary: dict[str, Any] = {"criteria": {
        "alignment_tolerance_pixels": ALIGNMENT_TOLERANCE_PIXELS,
        "minimum_surface_columns": MINIMUM_SURFACE_COLUMNS,
        "minimum_aligned_fraction": MINIMUM_ALIGNED_FRACTION,
        "minimum_aligned_run_columns": MINIMUM_ALIGNED_RUN_COLUMNS,
        "maximum_median_distance_pixels": MAXIMUM_MEDIAN_DISTANCE_PIXELS,
        "minimum_local_gradient": MINIMUM_LOCAL_GRADIENT,
        "minimum_local_contrast": MINIMUM_LOCAL_CONTRAST,
        "minimum_posterior_shadow": MINIMUM_POSTERIOR_SHADOW,
    }, "levels": {}}
    for level in ("L4", "L5"):
        level_rows = [row for row in comparisons if row["matched_level"] == level]
        matched_rows = [row for row in level_rows if row["comparison"] == "MATCHED"]
        mismatch_rows = [row for row in level_rows if row["comparison"] != "MATCHED"]
        matched_pass = sum(bool(row["passes"]) for row in matched_rows)
        mismatch_pass = sum(bool(row["passes"]) for row in mismatch_rows)
        ranking_wins = 0
        for frame in [row for row in frames if row["matched_level"] == level]:
            group = [row for row in level_rows if row["session_id"] == frame["session_id"] and row["frame_sequence"] == frame["frame_sequence"]]
            match = next(row for row in group if row["comparison"] == "MATCHED")
            controls = [row for row in group if row["comparison"] != "MATCHED"]
            ranking_wins += int(all(match["longest_aligned_run_columns"] > row["longest_aligned_run_columns"] for row in controls))
        summary["levels"][level] = {
            "matched_pass": matched_pass,
            "matched_total": len(matched_rows),
            "mismatch_pass": mismatch_pass,
            "mismatch_total": len(mismatch_rows),
            "ranking_wins": ranking_wins,
            "ranking_total": len(matched_rows),
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
