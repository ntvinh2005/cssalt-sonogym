#!/usr/bin/env python3
"""Build the compact final SAP validation/training review ZIP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Rectangle

import correct_s0010_sap_observability_v1 as correction
import run_s0010_sap_observability as observability

LEVELS = ("L4", "L5")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selected_examples(rows: list[dict[str, str]], level: str) -> list[tuple[str, dict[str, str]]]:
    level_rows = [row for row in rows if row["conditioned_level"] == level]
    positive = [row for row in level_rows if row["provisional_label"] == "SAP_RELATED_CORTICAL_INTERFACE"]
    negative = [row for row in level_rows if row["provisional_label"] == "OTHER_OR_BACKGROUND"]
    ambiguous = [row for row in level_rows if row["provisional_label"] == "BOUNDARY_OR_IDENTITY_AMBIGUOUS"]
    result: list[tuple[str, dict[str, str]]] = []
    if positive:
        result.append(("positive", max(positive, key=lambda row: int(row["surface_pixels_in_patch"]))))
    if negative:
        result.append(("negative", max(negative, key=lambda row: float(row["feature_gradient_p90"]))))
    if ambiguous:
        result.append(("ambiguous", max(ambiguous, key=lambda row: int(row["pseudo_pixels_in_patch"]))))
    return result


def render_example(
    *,
    row: dict[str, str],
    kind: str,
    level: str,
    masks: dict[str, dict[str, np.ndarray]],
    bone_ids: set[int],
    output: Path,
) -> dict[str, Any]:
    archive = Path(row["image_archive"])
    session = archive.parent.parent
    index = int(row["image_archive_index"])
    image = np.load(archive)["ultrasound"][index]
    coordinates = np.load(session / "privileged/pixel_to_volume_coordinates.npz")[
        "human_img_coords"
    ][index]
    raw_labels = np.load(session / "privileged/raw_labels.npz")["raw_labels"][index]
    projection = observability.projection_diagnostics(coordinates, masks[level])["projected"]
    surface = correction.first_bone_surface(
        raw_labels=raw_labels, corrected_projection=projection, bone_ids=bone_ids
    )["visible_pseudo_surface"]
    stem = f"{level}_{kind}_{row['session_id']}_{int(row['frame_sequence']):03d}"
    clean = output / f"{stem}_clean.png"
    overlay = output / f"{stem}_privileged_overlay.png"
    crop = output / f"{stem}_raw_patch.png"
    plt.imsave(clean, image, cmap="gray")
    fig, axis = plt.subplots(figsize=(6, 8), dpi=140)
    axis.imshow(image, cmap="gray")
    pseudo_rows, pseudo_columns = np.where(projection["pseudo"])
    surface_rows, surface_columns = np.where(surface)
    axis.scatter(pseudo_columns, pseudo_rows, s=4, c="#00bcd4", alpha=0.55, label="pseudo-SAP projection")
    axis.scatter(surface_columns, surface_rows, s=12, c="#ffd54f", label="first-hit pseudo-SAP")
    rectangle = Rectangle(
        (int(row["patch_column_start"]), int(row["patch_row_start"])),
        int(row["patch_column_end"]) - int(row["patch_column_start"]),
        int(row["patch_row_end"]) - int(row["patch_row_start"]),
        fill=False,
        edgecolor="#ff5252",
        linewidth=1.5,
        label="48x80 actor patch",
    )
    axis.add_patch(rectangle)
    axis.set_title(f"{level} {kind}; privileged evaluator overlay")
    axis.legend(loc="upper left", fontsize=7)
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(overlay)
    plt.close(fig)
    raw_patch = image[
        int(row["patch_row_start"]) : int(row["patch_row_end"]),
        int(row["patch_column_start"]) : int(row["patch_column_end"]),
    ]
    plt.imsave(crop, raw_patch, cmap="gray")
    return {
        "level": level,
        "kind": kind,
        "sample_id": row["sample_id"],
        "frame_sha256": row["frame_sha256"],
        "clean": str(clean.relative_to(output.parent)),
        "overlay": str(overlay.relative_to(output.parent)),
        "raw_patch": str(crop.relative_to(output.parent)),
    }


def plot_correspondence(paths: list[Path], output: Path) -> None:
    rows = [row for path in paths if path.exists() for row in read_csv(path)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150, sharey=True)
    for axis, level in zip(axes, LEVELS):
        level_rows = [row for row in rows if row["level"] == level]
        matched = [int(row["longest_aligned_run_columns"]) for row in level_rows if row["comparison"] == "MATCHED"]
        mismatched = [int(row["longest_aligned_run_columns"]) for row in level_rows if row["comparison"] != "MATCHED"]
        axis.hist(mismatched, bins=20, alpha=0.65, label="mismatched controls")
        axis.hist(matched, bins=20, alpha=0.65, label="matched")
        axis.set_title(level)
        axis.set_xlabel("longest aligned run (columns)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("comparisons")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-dataset", type=Path, required=True)
    parser.add_argument("--followup-dataset", type=Path, required=True)
    parser.add_argument("--archive-dataset", type=Path)
    parser.add_argument("--merged-dataset", type=Path, required=True)
    parser.add_argument("--evaluation-dataset", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--correspondence-validation", type=Path)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    package = args.output_dir.resolve() / "s0010_right_l4l5_sap_validation_training_review"
    if package.exists():
        shutil.rmtree(package)
    examples_dir = package / "figures/examples"
    figures_dir = package / "figures"
    examples_dir.mkdir(parents=True)
    readiness = json.loads((args.merged_dataset / "readiness_report.json").read_text())
    training_result = (
        json.loads((args.training_dir / "training_result.json").read_text())
        if (args.training_dir / "training_result.json").exists()
        else {"status": "NOT_RUN"}
    )
    evaluation = (
        json.loads(args.evaluation_report.read_text())
        if args.evaluation_report.exists()
        else {"overall_status": "NOT_RUN"}
    )
    prediction_image = nib.load(args.prediction)
    segmentation_image = nib.load(args.segmentation)
    masks = observability.pseudo_sap_masks(
        np.asarray(prediction_image.dataobj),
        np.asarray(segmentation_image.dataobj),
        prediction_image.affine,
    )
    bone_ids = correction.bone_source_ids(args.label_conversion)
    proposal_rows = read_csv(args.merged_dataset / "proposal_manifest.csv")
    example_index = []
    for level in LEVELS:
        for kind, row in selected_examples(proposal_rows, level):
            example_index.append(
                render_example(
                    row=row,
                    kind=kind,
                    level=level,
                    masks=masks,
                    bone_ids=bone_ids,
                    output=examples_dir,
                )
            )
    write_json(package / "01_EXAMPLE_INDEX.json", example_index)
    shutil.copy2(args.merged_dataset / "readiness_report.json", package / "02_DATASET_READINESS.json")
    shutil.copy2(args.first_dataset / "readiness_report.json", package / "02A_INITIAL_DATASET_READINESS.json")
    shutil.copy2(args.followup_dataset / "readiness_report.json", package / "02B_FOLLOWUP_DATASET_READINESS.json")
    if args.archive_dataset is not None:
        shutil.copy2(
            args.archive_dataset / "readiness_report.json",
            package / "02C_ARCHIVE_DATASET_READINESS.json",
        )
    shutil.copy2(args.merged_dataset / "sample_manifest.csv", package / "03_SAMPLE_MANIFEST.csv")
    if (args.training_dir / "training_result.json").exists():
        shutil.copy2(args.training_dir / "training_result.json", package / "04_TRAINING_RESULT.json")
    if args.evaluation_report.exists():
        shutil.copy2(args.evaluation_report, package / "05_HELDOUT_EVALUATION.json")
    if args.correspondence_validation is not None:
        shutil.copy2(
            args.correspondence_validation,
            package / "05A_CORRESPONDENCE_VALIDATION.json",
        )
    checkpoints = sorted(args.training_dir.glob("s0010_*_sap_classifier_v1.joblib"))
    for checkpoint in checkpoints:
        shutil.copy2(checkpoint, package / checkpoint.name)
    plot_correspondence(
        [
            args.first_dataset / "correspondence_controls.csv",
            args.followup_dataset / "correspondence_controls.csv",
            *(
                [args.archive_dataset / "correspondence_controls.csv"]
                if args.archive_dataset is not None
                else []
            ),
        ],
        figures_dir / "matched_mismatched_alignment.png",
    )
    lines = [
        "# s0010 RIGHT L4/L5 SAP validation and training review",
        "",
        "This package reports simulator-proxy ultrasound recognition on fixed anatomy s0010. It is not clinical ground truth.",
        "",
        f"Source commit: `{args.source_commit}`",
        f"Training status: `{training_result.get('status', 'NOT_RUN')}`",
        f"Held-out status: `{evaluation.get('overall_status', 'NOT_RUN')}`",
        "",
        "## Per-level annotation",
        "",
    ]
    for level in LEVELS:
        value = readiness["readiness"][level]
        lines.extend(
            [
                f"- {level}: `{value['annotation_status']}`; dataset `{value['dataset_status']}`; blockers: {value['deficits'] or 'none'}.",
            ]
        )
    if readiness.get("incomplete_sessions"):
        lines.extend(["", "## Incomplete capture sessions", ""])
        for item in readiness["incomplete_sessions"]:
            lines.append(
                f"- `{item['session_id']}` ({item['split']}): {item['reason']}"
            )
    lines.extend(
        [
            "",
            "Clean images contain no overlays. Files marked `privileged_overlay` are evaluator-only and must not be used as classifier inputs.",
        ]
    )
    (package / "00_EXECUTIVE_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    provenance = {
        "source_commit": args.source_commit,
        "prediction_sha256": file_sha256(args.prediction),
        "segmentation_sha256": file_sha256(args.segmentation),
        "first_dataset_readiness_sha256": file_sha256(args.first_dataset / "readiness_report.json"),
        "followup_dataset_readiness_sha256": file_sha256(args.followup_dataset / "readiness_report.json"),
        "merged_dataset_readiness_sha256": file_sha256(args.merged_dataset / "readiness_report.json"),
        "archive_dataset_readiness_sha256": (
            file_sha256(args.archive_dataset / "readiness_report.json")
            if args.archive_dataset is not None
            else None
        ),
        "correspondence_validation_sha256": (
            file_sha256(args.correspondence_validation)
            if args.correspondence_validation is not None
            else None
        ),
        "checkpoint_sha256": {
            checkpoint.name: file_sha256(checkpoint) for checkpoint in checkpoints
        },
        "evaluation_dataset_present": args.evaluation_dataset.exists(),
    }
    write_json(package / "06_PROVENANCE.json", provenance)
    checksum_lines = []
    for path in sorted(package.rglob("*")):
        if path.is_file():
            checksum_lines.append(f"{file_sha256(path)}  {path.relative_to(package)}")
    (package / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    zip_path = args.output_dir.resolve() / "s0010_right_l4l5_sap_validation_training_review.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(Path(package.name) / path.relative_to(package)))
    print(json.dumps({"zip": str(zip_path), "sha256": file_sha256(zip_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
