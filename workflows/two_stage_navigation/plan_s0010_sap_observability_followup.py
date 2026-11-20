#!/usr/bin/env python3
"""Select fresh-capture waypoints from corrected archived SAP surface evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

import correct_s0010_sap_observability_v1 as correction
import run_s0010_sap_observability as obs


MINIMUM_VISIBLE_SURFACE_PIXELS = 5
MINIMUM_TRANSLATION_SEPARATION_INDEX = 3.0
MINIMUM_ANGLE_SEPARATION_RAD = np.deg2rad(2.4)
VIEWS_PER_LEVEL_PER_BANK = 3


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def normalized_gradient(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    lo, hi = np.percentile(array, (0.5, 99.5))
    normalized = np.clip((array - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return np.hypot(ndimage.sobel(normalized, axis=0), ndimage.sobel(normalized, axis=1))


def physically_distinct(a: np.ndarray, b: np.ndarray) -> bool:
    translation = float(np.linalg.norm(a[:2] - b[:2]))
    angle = float(abs(np.arctan2(np.sin(a[2] - b[2]), np.cos(a[2] - b[2]))))
    return bool(
        translation > MINIMUM_TRANSLATION_SEPARATION_INDEX
        or angle > MINIMUM_ANGLE_SEPARATION_RAD
    )


def select_diverse(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["surface_gradient_median"]),
            -int(row["visible_pseudo_surface_pixel_count"]),
            -int(row["core_margin_pixels"]),
            str(row["session_id"]),
            int(row["frame_sequence"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    for row in ordered:
        pose = np.asarray(row["requested_relative_pose_index"], dtype=np.float64)
        if all(
            physically_distinct(
                pose,
                np.asarray(existing["requested_relative_pose_index"], dtype=np.float64),
            )
            for existing in selected
        ):
            selected.append(row)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"only {len(selected)} physically distinct candidates for {count} requested")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    serial = []
    for row in rows:
        serial.append(
            {
                key: json.dumps(value) if isinstance(value, list) else value
                for key, value in row.items()
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serial[0]))
        writer.writeheader()
        writer.writerows(serial)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=False)
    prediction_image = nib.load(args.prediction)
    segmentation_image = nib.load(args.segmentation)
    segmentation = np.asarray(segmentation_image.dataobj)
    masks = obs.pseudo_sap_masks(
        np.asarray(prediction_image.dataobj), segmentation, prediction_image.affine
    )
    bone_ids = correction.bone_source_ids(args.label_conversion)
    rows: list[dict[str, Any]] = []
    images: dict[tuple[str, int], np.ndarray] = {}
    for session_dir in sorted(args.reference_root.iterdir()):
        if not session_dir.is_dir():
            continue
        frames = read_jsonl(session_dir / "public/frames.jsonl")
        ultrasound = np.load(session_dir / "public/ultrasound.npz")["ultrasound"]
        coordinates = np.load(
            session_dir / "privileged/pixel_to_volume_coordinates.npz"
        )["human_img_coords"]
        raw_labels = np.load(session_dir / "privileged/raw_labels.npz")["raw_labels"]
        for index, frame in enumerate(frames):
            images[(session_dir.name, index)] = ultrasound[index]
            gradient = normalized_gradient(ultrasound[index])
            for level in ("L4", "L5"):
                diagnostics = obs.projection_diagnostics(coordinates[index], masks[level])
                surface = correction.first_bone_surface(
                    raw_labels=raw_labels[index],
                    corrected_projection=diagnostics["projected"],
                    bone_ids=bone_ids,
                )
                surface_values = gradient[surface["visible_pseudo_surface"]]
                rows.append(
                    {
                        "session_id": session_dir.name,
                        "frame_sequence": index,
                        "frame_sha256": frame["frame_sha256"],
                        "level": level,
                        "phase": frame["phase"],
                        "target_name": frame["target_name"],
                        "requested_relative_pose_index": frame[
                            "requested_relative_pose_index"
                        ],
                        "executed_relative_pose_index": frame[
                            "executed_relative_pose_index"
                        ],
                        "core_pixel_count": diagnostics["core_pixel_count"],
                        "core_margin_pixels": diagnostics["core_margin_pixels"],
                        "projection_gate_passed": diagnostics[
                            "projection_gate_passed"
                        ],
                        "visible_pseudo_surface_pixel_count": surface[
                            "visible_pseudo_surface_pixel_count"
                        ],
                        "occluded_pseudo_column_count": surface[
                            "occluded_pseudo_column_count"
                        ],
                        "surface_gradient_median": (
                            float(np.median(surface_values)) if len(surface_values) else 0.0
                        ),
                    }
                )

    eligible = [
        row
        for row in rows
        if bool(row["projection_gate_passed"])
        and int(row["visible_pseudo_surface_pixel_count"])
        >= MINIMUM_VISIBLE_SURFACE_PIXELS
    ]
    selected: dict[str, list[dict[str, Any]]] = {}
    for level in ("L4", "L5"):
        selected[level] = select_diverse(
            [row for row in eligible if row["level"] == level],
            2 * VIEWS_PER_LEVEL_PER_BANK,
        )

    plan = {
        "schema": "sonogym-s0010-sap-observability-followup-plan-v1",
        "source_reference_root": str(args.reference_root),
        "prediction_sha256": obs.sha256(args.prediction),
        "segmentation_sha256": obs.sha256(args.segmentation),
        "selection": {
            "minimum_visible_surface_pixels": MINIMUM_VISIBLE_SURFACE_PIXELS,
            "minimum_core_pixels": obs.MINIMUM_CORE_PIXELS,
            "minimum_core_margin_pixels": obs.MINIMUM_CORE_MARGIN_PIXELS,
            "minimum_translation_separation_index": MINIMUM_TRANSLATION_SEPARATION_INDEX,
            "minimum_angle_separation_rad": MINIMUM_ANGLE_SEPARATION_RAD,
            "ranking": [
                "surface_gradient_median_descending",
                "visible_pseudo_surface_pixel_count_descending",
                "core_margin_pixels_descending",
            ],
        },
        "candidate_counts": {
            level: sum(row["level"] == level for row in eligible)
            for level in ("L4", "L5")
        },
        "banks": {
            "bank_a": [selected["L4"][0], selected["L4"][2], selected["L4"][4], selected["L5"][0], selected["L5"][2], selected["L5"][4]],
            "bank_b": [selected["L4"][1], selected["L4"][3], selected["L4"][5], selected["L5"][1], selected["L5"][3], selected["L5"][5]],
        },
    }
    obs.write_json(args.output_root / "waypoint_plan.json", plan)
    write_csv(args.output_root / "all_candidates.csv", rows)
    write_csv(args.output_root / "eligible_candidates.csv", eligible)

    selected_rows = plan["banks"]["bank_a"] + plan["banks"]["bank_b"]
    fig, axes = plt.subplots(3, 4, figsize=(12, 10), constrained_layout=True)
    for ax, row in zip(axes.flat, selected_rows):
        image = images[(row["session_id"], int(row["frame_sequence"]))]
        ax.imshow(image, cmap="gray", origin="upper")
        ax.set_title(
            f"{row['level']} {row['session_id']}:{row['frame_sequence']}\n"
            f"grad={row['surface_gradient_median']:.3f} surface={row['visible_pseudo_surface_pixel_count']} margin={row['core_margin_pixels']}"
        )
        ax.axis("off")
    fig.savefig(args.output_root / "selected_clean_views.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"eligible": plan["candidate_counts"], "selected": 12}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
