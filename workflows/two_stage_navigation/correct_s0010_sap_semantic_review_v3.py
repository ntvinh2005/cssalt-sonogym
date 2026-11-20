#!/usr/bin/env python3
"""Build the compact CPU-only correction to the s0010 SAP V3 review evidence."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "workflows/two_stage_navigation/build_s0010_sap_semantic_review_v3.py"
SPEC = importlib.util.spec_from_file_location("sap_semantic_review_v3_builder", BUILDER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {BUILDER_PATH}")
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)

PACKAGE_NAME = "s0010_right_l4l5_sap_semantic_review_v3_correction"
COLORS = {"source": "#ffcc00", "sap": "#00bcd4", "iap": "#d627a8", "point": "#e41a1c"}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_targeted_sheet(
    *,
    ct: np.ndarray,
    segmentation: np.ndarray,
    prediction: np.ndarray,
    level: str,
    rows: list[dict[str, Any]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        len(rows), 3, figsize=(12, max(4.0, 2.8 * len(rows))), dpi=150, squeeze=False
    )
    axis_specs = ((2, "axial"), (0, "sagittal"), (1, "coronal"))
    source_label = BUILDER.LABEL_VALUE[level]
    for row_index, record in enumerate(rows):
        point = np.asarray(
            [record["x_index"], record["y_index"], record["z_index"]], dtype=float
        )
        voxel = np.rint(point).astype(int)
        for column, (fixed_axis, plane_name) in enumerate(axis_specs):
            axis = axes[row_index, column]
            coordinate = int(voxel[fixed_axis])
            image = BUILDER._slice(ct, fixed_axis, coordinate)
            source = BUILDER._slice(segmentation == source_label, fixed_axis, coordinate)
            sap = BUILDER._slice(prediction == 6, fixed_axis, coordinate)
            iap = BUILDER._slice(prediction == 7, fixed_axis, coordinate)
            axis.imshow(np.clip(image, -300, 1200), cmap="gray", origin="lower")
            for mask, color in ((source, COLORS["source"]), (sap, COLORS["sap"]), (iap, COLORS["iap"])):
                if np.any(mask):
                    axis.contour(mask, [0.5], colors=[color], linewidths=0.9)
            horizontal, vertical = ((1, 2), (0, 2), (0, 1))[fixed_axis]
            axis.scatter(
                point[horizontal], point[vertical], marker="x", s=55,
                linewidths=1.5, color=COLORS["point"],
            )
            axis.text(
                point[horizontal] + 1, point[vertical] + 1, str(record["callout_id"]),
                color=COLORS["point"], fontsize=8, fontweight="bold",
            )
            radius = 18
            axis.set_xlim(point[horizontal] - radius, point[horizontal] + radius)
            axis.set_ylim(point[vertical] - radius, point[vertical] + radius)
            if row_index == 0:
                axis.set_title(plane_name)
            axis.set_axis_off()
        axes[row_index, 0].text(
            -0.03, 0.5,
            f"#{record['callout_id']}  d={float(record['nearest_model_sap_surface_distance_mm']):.3f} mm\n"
            f"sample={record['sampled_model_structure']}\n{record['review_reason']}",
            transform=axes[row_index, 0].transAxes, ha="right", va="center", fontsize=7,
        )
    figure.suptitle(
        f"RIGHT {level} SAP targeted correction — yellow=source {source_label}; "
        "cyan=model SAP; magenta=model IAP; red=V2 point",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.08, 0.01, 1.0, 0.99))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def build(args: argparse.Namespace) -> None:
    package = args.output_root.resolve() / PACKAGE_NAME
    if package.exists() and any(package.iterdir()):
        raise RuntimeError(f"refusing nonempty package directory: {package}")
    (package / "figures").mkdir(parents=True, exist_ok=True)
    ct_image = nib.load(args.ct)
    ct = np.asanyarray(ct_image.dataobj)
    prediction_image, prediction, segmentation, prediction_report = BUILDER.validate_prediction(
        args.prediction, args.ct, args.segmentation
    )
    label_conversion = BUILDER.load_label_conversion(args.label_conversion)
    consistency = BUILDER.anatomical_consistency_report(
        prediction, segmentation, label_conversion
    )
    level_masks, _ = BUILDER.split_level_and_laterality(
        prediction, segmentation, prediction_image.affine
    )
    all_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    callout_id = 1
    for level in ("L4", "L5"):
        v2 = BUILDER.load_v2_core(args.v2_package, level)
        report, masks = BUILDER.compare_v2_model(
            level=level,
            model_mask=level_masks[level]["superior_articular_process"],
            prediction=prediction,
            v2=v2,
            affine=prediction_image.affine,
        )
        world = nib.affines.apply_affine(prediction_image.affine, v2["core_index"])
        level_review: list[dict[str, Any]] = []
        for index, (point_index, point_world) in enumerate(zip(v2["core_index"], world)):
            distance = float(masks["v2_distances_mm"][index])
            sampled_label = int(masks["v2_sampled_model_labels"][index])
            proximal = bool(masks["v2_agreement"][index])
            reasons: list[str] = []
            if not proximal:
                reasons.append("V2_DISTANCE_OUTLIER_REVIEW")
            if sampled_label == 7:
                reasons.append("SAMPLED_IAP_CONFLICT_REVIEW")
            record = {
                "structure_id": f"RIGHT_{level}_SAP",
                "v2_point_index": index,
                "x_index": float(point_index[0]),
                "y_index": float(point_index[1]),
                "z_index": float(point_index[2]),
                "x_world_mm": float(point_world[0]),
                "y_world_mm": float(point_world[1]),
                "z_world_mm": float(point_world[2]),
                "nearest_model_sap_surface_distance_mm": distance,
                "distance_within_1p5_mm": proximal,
                "sampled_model_label": sampled_label,
                "sampled_model_structure": BUILDER.MODEL_LABEL.get(
                    sampled_label, "outside_or_unknown"
                ),
                "semantic_category": str(masks["v2_categories"][index]),
                "review_reason": ";".join(reasons) if reasons else "NONE",
                "callout_id": "",
            }
            if reasons:
                record["callout_id"] = callout_id
                callout_id += 1
                level_review.append(record)
                review_rows.append(record)
            all_rows.append(record)
        summaries[level] = {
            **report,
            "targeted_review_point_count": len(level_review),
            "distance_outlier_count": sum(
                "V2_DISTANCE_OUTLIER_REVIEW" in row["review_reason"] for row in level_review
            ),
            "sampled_iap_conflict_count": sum(
                "SAMPLED_IAP_CONFLICT_REVIEW" in row["review_reason"] for row in level_review
            ),
        }
        render_targeted_sheet(
            ct=ct, segmentation=segmentation, prediction=prediction, level=level,
            rows=level_review, output=package / f"figures/RIGHT_{level}_SAP_targeted_correction.png",
        )
    write_csv(package / "01_ALL_V2_POINT_COMPARISON.csv", all_rows)
    write_csv(package / "02_TARGETED_REVIEW_POINTS.csv", review_rows)
    write_json(package / "03_ANATOMICAL_CONSISTENCY.json", consistency)
    write_json(package / "04_CORRECTION_SUMMARY.json", summaries)
    source_revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    (package / "00_README.md").write_text(
        f"""# s0010 SAP V3 targeted correction

This CPU-only package corrects two evaluation defects in the frozen V3 review evidence. It reuses the raw prediction unchanged and does not promote any proposal to semantic ground truth.

- Source revision: `{source_revision}`
- Raw prediction SHA-256: `{BUILDER.sha256(args.prediction)}`
- Original V3 ZIP SHA-256: `{BUILDER.sha256(args.original_v3_zip)}`
- Vertebral consistency uses explicit source labels `25-31`, not `segmentation > 0`.
- Acoustic-bone consistency uses only raw source labels mapped to renderer class `13` by `{args.label_conversion}`.
- Distance-to-SAP and sampled semantic class are reported independently.
- Every V2 point farther than 1.5 mm or sampled as model IAP has a numbered CT MPR callout.

The original model masks, checkpoint, inference command and V3 ZIP remain frozen. Background samples near a surface are uncertainty evidence, not automatic SAP errors. The figures support review only; they do not establish ultrasound visibility, MBB targets or intervention readiness.
""",
        encoding="utf-8",
    )
    provenance = {
        "schema": "sonogym-s0010-sap-semantic-review-v3-targeted-correction",
        "source_revision": source_revision,
        "raw_prediction": {"path": str(args.prediction), "sha256": BUILDER.sha256(args.prediction)},
        "ct": {"path": str(args.ct), "sha256": BUILDER.sha256(args.ct)},
        "segmentation": {"path": str(args.segmentation), "sha256": BUILDER.sha256(args.segmentation)},
        "label_conversion": {"path": str(args.label_conversion), "sha256": BUILDER.sha256(args.label_conversion)},
        "original_v3_zip": {"path": str(args.original_v3_zip), "sha256": BUILDER.sha256(args.original_v3_zip)},
        "v2_package": str(args.v2_package),
        "model_masks_modified": False,
        "semantic_ground_truth_accepted": False,
        "ultrasound_observability_confirmed": False,
    }
    write_json(package / "05_PROVENANCE.json", provenance)
    BUILDER.write_checksums(package)
    zip_path = args.output_root.resolve() / f"{PACKAGE_NAME}.zip"
    BUILDER.create_zip(package, zip_path)
    result = {
        "schema": "sonogym-s0010-sap-semantic-review-v3-targeted-correction-result",
        "package": str(package),
        "zip_path": str(zip_path),
        "zip_sha256": BUILDER.sha256(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "anatomical_consistency": consistency,
        "summaries": summaries,
        "targeted_review_point_count": len(review_rows),
    }
    write_json(args.output_root.resolve() / "correction_result.json", result)
    print(json.dumps(result, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--v2-package", type=Path, required=True)
    parser.add_argument("--original-v3-zip", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
