#!/usr/bin/env python3
"""Correct V1 SAP rasterization and audit the saved frames without recapture."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import yaml

import run_s0010_sap_observability as obs


SURFACE_ALIGNMENT_TOLERANCE_PIXELS = 4.0
TARGET_ROLE = "PSEUDO_SAP_TARGET"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bone_source_ids(path: Path) -> set[int]:
    mapping = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {int(source) for source, acoustic in mapping.items() if int(acoustic) == 13}


def parse_points(value: str) -> np.ndarray:
    points = []
    for token in value.split(";"):
        row, column = token.split(":")
        points.append((int(row), int(column)))
    array = np.asarray(points, dtype=np.int64)
    if np.any(array[:, 0] < 0) or np.any(array[:, 0] >= obs.IMAGE_DEPTH):
        raise ValueError("polyline row is outside image")
    if np.any(array[:, 1] < 0) or np.any(array[:, 1] >= obs.IMAGE_WIDTH):
        raise ValueError("polyline column is outside image")
    return array


def rasterize_polyline(points: np.ndarray) -> np.ndarray:
    pixels: list[np.ndarray] = []
    for start, end in zip(points[:-1], points[1:]):
        steps = int(np.max(np.abs(end - start))) + 1
        pixels.append(
            np.rint(np.linspace(start, end, steps)).astype(np.int64)
        )
    if not pixels:
        pixels = [points]
    return np.unique(np.concatenate(pixels, axis=0), axis=0)


def minimum_distance(points: np.ndarray, mask: np.ndarray) -> float | None:
    targets = np.argwhere(mask)
    if not len(points) or not len(targets):
        return None
    squared = np.sum(
        (points[:, None, :].astype(np.float64) - targets[None, :, :]) ** 2,
        axis=2,
    )
    return float(np.sqrt(np.min(squared)))


def first_bone_surface(
    *,
    raw_labels: np.ndarray,
    corrected_projection: dict[str, np.ndarray],
    bone_ids: set[int],
) -> dict[str, Any]:
    if raw_labels.shape != obs.IMAGE_SHAPE:
        raise ValueError(f"unexpected raw label shape: {raw_labels.shape}")
    bone = np.isin(raw_labels, list(bone_ids))
    first_bone = np.zeros_like(bone)
    for column in range(obs.IMAGE_WIDTH):
        rows = np.flatnonzero(bone[:, column])
        if len(rows):
            first_bone[int(rows[0]), column] = True

    pseudo = corrected_projection["pseudo"]
    first_pseudo = np.zeros_like(pseudo)
    occluded_pseudo = np.zeros_like(pseudo)
    pseudo_columns = 0
    occluded_columns = 0
    for column in range(obs.IMAGE_WIDTH):
        pseudo_rows = np.flatnonzero(pseudo[:, column])
        if not len(pseudo_rows):
            continue
        pseudo_columns += 1
        first_pseudo_row = int(pseudo_rows[0])
        first_pseudo[first_pseudo_row, column] = True
        bone_rows = np.flatnonzero(first_bone[:, column])
        if len(bone_rows) and int(bone_rows[0]) < first_pseudo_row:
            occluded_columns += 1
            occluded_pseudo[first_pseudo_row, column] = True

    visible_pseudo_surface = first_bone & pseudo
    return {
        "first_bone": first_bone,
        "first_pseudo": first_pseudo,
        "visible_pseudo_surface": visible_pseudo_surface,
        "occluded_pseudo": occluded_pseudo,
        "first_bone_pixel_count": int(np.count_nonzero(first_bone)),
        "pseudo_column_count": pseudo_columns,
        "visible_pseudo_surface_pixel_count": int(
            np.count_nonzero(visible_pseudo_surface)
        ),
        "occluded_pseudo_column_count": occluded_columns,
    }


def frame_status(
    *,
    role: str,
    clean_decision: str,
    visible_pseudo_surface_count: int,
    occluded_columns: int,
    minimum_polyline_distance: float | None,
) -> str:
    if role != TARGET_ROLE:
        return "CONTEXT_CONTROL"
    if (
        visible_pseudo_surface_count > 0
        and minimum_polyline_distance is not None
        and minimum_polyline_distance <= SURFACE_ALIGNMENT_TOLERANCE_PIXELS
    ):
        return "SAP_SURFACE_CANDIDATE_VISIBLE"
    if visible_pseudo_surface_count == 0 and occluded_columns > 0:
        return "OCCLUDED_BY_OTHER_BONE"
    if clean_decision == "NOT_VISIBLE":
        return "NO_VISIBLE_INTERFACE"
    return "UNRESOLVED"


def summarize_structure(targets: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [row["corrected_surface_status"] for row in targets]
    qualifying = [
        row
        for row in targets
        if row["corrected_surface_status"] == "SAP_SURFACE_CANDIDATE_VISIBLE"
        and bool(row["corrected_projection_gate_passed"])
    ]
    qualifying_sessions = {row["session_id"] for row in qualifying}
    if len(qualifying) >= 2 and len(qualifying_sessions) >= 2:
        verdict = "SAP_SURFACE_VISIBLE_PROVISIONAL"
    elif statuses and all(status == "OCCLUDED_BY_OTHER_BONE" for status in statuses):
        verdict = "OCCLUDED_BY_OTHER_BONE"
    elif statuses and all(status == "NO_VISIBLE_INTERFACE" for status in statuses):
        verdict = "NO_VISIBLE_INTERFACE"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "target_frame_count": len(targets),
        "corrected_projection_gate_passed_count": sum(
            bool(row["corrected_projection_gate_passed"]) for row in targets
        ),
        "qualifying_visible_frame_count": len(qualifying),
        "qualifying_visible_session_count": len(qualifying_sessions),
        "status_counts": {
            status: statuses.count(status)
            for status in (
                "SAP_SURFACE_CANDIDATE_VISIBLE",
                "OCCLUDED_BY_OTHER_BONE",
                "NO_VISIBLE_INTERFACE",
                "UNRESOLVED",
            )
        },
        "verdict": verdict,
    }


def overlay(ax: Any, masks: dict[str, np.ndarray], *, title: str) -> None:
    colors = {
        "uncertainty": "#ff9f1c",
        "pseudo": "#2ec4b6",
        "core": "#e71d36",
    }
    for name in ("uncertainty", "pseudo", "core"):
        rows, columns = np.where(masks[name])
        ax.scatter(columns, rows, s=3, c=colors[name], label=name)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=6)


def save_comparison(
    path: Path,
    image: np.ndarray,
    legacy: dict[str, np.ndarray],
    corrected: dict[str, np.ndarray],
    surface: dict[str, Any],
    polylines: list[np.ndarray],
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), constrained_layout=True)
    for ax in axes:
        ax.imshow(image, cmap="gray", origin="upper")
        ax.set_xlim(-0.5, obs.IMAGE_WIDTH - 0.5)
        ax.set_ylim(obs.IMAGE_DEPTH - 0.5, -0.5)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
    axes[0].set_title("clean ultrasound")
    overlay(axes[1], legacy, title="invalid V1 rasterization")
    overlay(axes[2], corrected, title="corrected native rasterization")
    first_rows, first_columns = np.where(surface["first_bone"])
    axes[3].scatter(first_columns, first_rows, s=4, c="#00b4d8", label="first bone")
    visible_rows, visible_columns = np.where(surface["visible_pseudo_surface"])
    axes[3].scatter(
        visible_columns,
        visible_rows,
        s=12,
        c="#f72585",
        label="first bone is pseudo-SAP",
    )
    occ_rows, occ_columns = np.where(surface["occluded_pseudo"])
    axes[3].scatter(
        occ_columns,
        occ_rows,
        s=8,
        c="#ffd166",
        label="pseudo-SAP behind first bone",
    )
    for index, points in enumerate(polylines):
        axes[3].plot(
            points[:, 1],
            points[:, 0],
            color="#80ff72",
            linewidth=2,
            label="clean-image polyline" if index == 0 else None,
        )
    axes[3].set_title("first-bone audit and clean review")
    axes[3].legend(loc="upper right", fontsize=6)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_binary_masks(
    path: Path,
    corrected: dict[str, np.ndarray],
    surface: dict[str, Any],
    title: str,
) -> None:
    names = (
        "pseudo",
        "core",
        "uncertainty",
        "first_bone",
        "visible_pseudo_surface",
        "occluded_pseudo",
    )
    fig, axes = plt.subplots(2, 3, figsize=(10, 9), constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        source = corrected[name] if name in corrected else surface[name]
        ax.imshow(source, cmap="gray", origin="upper", interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_review_documents(output: Path, result: dict[str, Any]) -> None:
    lines = [
        "# s0010 SAP observability V1 correction",
        "",
        "The V1/V1R1 spatial evaluation is invalidated because its flattened native ",
        "coordinate map was reshaped directly as depth-by-width instead of native ",
        "width-by-depth followed by display transpose. The saved acquisitions are ",
        "reused; no inference, capture or training was run.",
        "",
        "## Corrected result",
        "",
    ]
    for level in ("L4", "L5"):
        structure = result["structures"][level]
        lines.append(
            f"- RIGHT {level} SAP: `{structure['verdict']}`; "
            f"statuses `{json.dumps(structure['status_counts'], sort_keys=True)}`."
        )
    lines.extend(
        [
            "",
            "These are simulator-only provisional surface observations. They do not ",
            "approve SAP ground truth, define an MBB target or authorize Stage 3.",
            "",
            "## Package map",
            "",
            "- `01_CORRECTED_PROJECTION_METRICS.csv`: old versus corrected positions and gates.",
            "- `02_CORTICAL_POLYLINES.csv`: clean-image development annotations locked before corrected overlays.",
            "- `03_CORRECTED_BINARY_MASKS.npz`: corrected pseudo/core/uncertainty and first-bone masks.",
            "- `figures/frame_comparisons/`: clean, invalid, corrected and first-bone panels.",
            "- `verification_arrays/`: ultrasound, raw labels, native coordinate maps, poses and frame records.",
            "",
        ]
    )
    (output / "00_EXECUTIVE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    for level, number in (("L4", "06"), ("L5", "07")):
        structure = result["structures"][level]
        decision = [
            f"# RIGHT {level} SAP corrected decision sheet",
            "",
            f"**Verdict: `{structure['verdict']}`.**",
            "",
            "## Evidence",
            "",
            f"- Target frames: {structure['target_frame_count']}.",
            f"- Corrected projection-gated frames: {structure['corrected_projection_gate_passed_count']}.",
            f"- Per-frame statuses: `{json.dumps(structure['status_counts'], sort_keys=True)}`.",
            "- Exact source arrays are included under `verification_arrays/`.",
            "",
            "## Review limits",
            "",
            "- The CT SAP mask is a pretrained-model proposal, not verified anatomy.",
            "- Clean-image polylines are opened-development notes, not independent semantic labels.",
            "- A first raw-label bone hit is privileged geometry evidence, not proof of a recognizable SAP echo.",
            "",
            "## Requested external decision",
            "",
            "Inspect the corrected frame comparisons and decide whether any first-hit ",
            "pseudo-SAP surface follows the clean cortical interface, is occluded by ",
            "another bone surface, or remains unresolved.",
            "",
        ]
        (output / f"{number}_RIGHT_{level}_SAP_DECISION_SHEET.md").write_text(
            "\n".join(decision), encoding="utf-8"
        )


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root
    output.mkdir(parents=True, exist_ok=False)
    prediction_image = nib.load(args.prediction)
    segmentation_image = nib.load(args.segmentation)
    prediction = np.asarray(prediction_image.dataobj)
    segmentation = np.asarray(segmentation_image.dataobj)
    masks = obs.pseudo_sap_masks(prediction, segmentation, prediction_image.affine)
    bone_ids = bone_source_ids(args.label_conversion)

    old_review = args.review_root
    old_rows = read_csv(old_review / "projection_manifest.csv")
    row_by_frame = {
        (row["session_id"], int(row["frame_sequence"])): row for row in old_rows
    }
    decision_path = args.decisions or old_review / "blind_decisions.csv"
    clean_decisions = {row["review_id"]: row for row in read_csv(decision_path)}
    polyline_rows = read_csv(args.polylines)
    polylines_by_review: dict[str, list[np.ndarray]] = {}
    for row in polyline_rows:
        polylines_by_review.setdefault(row["review_id"], []).append(
            rasterize_polyline(parse_points(row["polyline_points_rc"]))
        )

    metrics: list[dict[str, Any]] = []
    mask_archive: dict[str, np.ndarray] = {}
    raw_replay_passed = 0
    for session_dir in sorted((args.run_root / "sessions").iterdir()):
        if not session_dir.is_dir():
            continue
        session_id = session_dir.name
        ultrasound = np.load(session_dir / "public/ultrasound.npz")["ultrasound"]
        coordinates = np.load(
            session_dir / "privileged/pixel_to_volume_coordinates.npz"
        )["human_img_coords"]
        raw_labels = np.load(session_dir / "privileged/raw_labels.npz")["raw_labels"]
        for frame_sequence in range(len(ultrasound)):
            old = row_by_frame[(session_id, frame_sequence)]
            level = old["level"]
            legacy = obs.projection_diagnostics(
                coordinates[frame_sequence],
                masks[level],
                projector=obs.project_mask_legacy_v1,
            )
            corrected = obs.projection_diagnostics(
                coordinates[frame_sequence], masks[level]
            )
            sampled_raw = segmentation[
                np.trunc(coordinates[frame_sequence, :, 0]).astype(np.int64),
                np.trunc(coordinates[frame_sequence, :, 1]).astype(np.int64),
                np.trunc(coordinates[frame_sequence, :, 2]).astype(np.int64),
            ].reshape(obs.IMAGE_WIDTH, obs.IMAGE_DEPTH).T
            raw_equal = bool(np.array_equal(sampled_raw, raw_labels[frame_sequence]))
            if not raw_equal:
                raise RuntimeError(f"raw-label replay mismatch: {session_id}:{frame_sequence}")
            raw_replay_passed += 1
            surface = first_bone_surface(
                raw_labels=raw_labels[frame_sequence],
                corrected_projection=corrected["projected"],
                bone_ids=bone_ids,
            )
            review_id = old["review_id"]
            frame_polylines = polylines_by_review[review_id]
            all_polyline_points = np.unique(
                np.concatenate(frame_polylines, axis=0), axis=0
            )
            distance = minimum_distance(
                all_polyline_points, surface["visible_pseudo_surface"]
            )
            clean_decision = clean_decisions[review_id]["decision"]
            status = frame_status(
                role=old["role"],
                clean_decision=clean_decision,
                visible_pseudo_surface_count=surface[
                    "visible_pseudo_surface_pixel_count"
                ],
                occluded_columns=surface["occluded_pseudo_column_count"],
                minimum_polyline_distance=distance,
            )
            key = f"{session_id}_frame_{frame_sequence:02d}_{level}"
            save_comparison(
                output / "figures/frame_comparisons" / f"{key}.png",
                ultrasound[frame_sequence],
                legacy["projected"],
                corrected["projected"],
                surface,
                frame_polylines,
                f"{session_id} frame {frame_sequence} {level} {old['role']}",
            )
            save_binary_masks(
                output / "figures/binary_masks" / f"{key}.png",
                corrected["projected"],
                surface,
                f"{session_id} frame {frame_sequence} {level}",
            )
            clean_destination = output / "clean_ultrasound" / f"{key}.png"
            clean_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_review / old["clean_image"], clean_destination)
            for name in ("pseudo", "core", "uncertainty"):
                mask_archive[f"{key}_{name}"] = corrected["projected"][name]
            for name in (
                "first_bone",
                "first_pseudo",
                "visible_pseudo_surface",
                "occluded_pseudo",
            ):
                mask_archive[f"{key}_{name}"] = surface[name]
            metrics.append(
                {
                    "session_id": session_id,
                    "frame_sequence": frame_sequence,
                    "review_id": review_id,
                    "level": level,
                    "role": old["role"],
                    "clean_image_decision": clean_decision,
                    "legacy_pseudo_pixel_count": legacy["pseudo_pixel_count"],
                    "corrected_pseudo_pixel_count": corrected["pseudo_pixel_count"],
                    "legacy_core_pixel_count": legacy["core_pixel_count"],
                    "corrected_core_pixel_count": corrected["core_pixel_count"],
                    "legacy_core_centroid_row": legacy["core_centroid_row"],
                    "legacy_core_centroid_column": legacy["core_centroid_column"],
                    "corrected_core_centroid_row": corrected["core_centroid_row"],
                    "corrected_core_centroid_column": corrected[
                        "core_centroid_column"
                    ],
                    "legacy_core_margin_pixels": legacy["core_margin_pixels"],
                    "corrected_core_margin_pixels": corrected[
                        "core_margin_pixels"
                    ],
                    "legacy_projection_gate_passed": legacy[
                        "projection_gate_passed"
                    ],
                    "corrected_projection_gate_passed": corrected[
                        "projection_gate_passed"
                    ],
                    "first_bone_pixel_count": surface["first_bone_pixel_count"],
                    "pseudo_column_count": surface["pseudo_column_count"],
                    "visible_pseudo_surface_pixel_count": surface[
                        "visible_pseudo_surface_pixel_count"
                    ],
                    "occluded_pseudo_column_count": surface[
                        "occluded_pseudo_column_count"
                    ],
                    "minimum_polyline_distance_to_visible_pseudo_surface_pixels": (
                        "" if distance is None else distance
                    ),
                    "corrected_surface_status": status,
                    "raw_label_replay_exact": raw_equal,
                }
            )

        verification = output / "verification_arrays" / session_id
        verification.mkdir(parents=True)
        for relative in (
            "public/ultrasound.npz",
            "public/frames.jsonl",
            "privileged/raw_labels.npz",
            "privileged/pixel_to_volume_coordinates.npz",
            "privileged/full_pose_replay.jsonl",
        ):
            source = session_dir / relative
            destination = verification / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    write_csv(output / "01_CORRECTED_PROJECTION_METRICS.csv", metrics)
    write_csv(output / "02_CORTICAL_POLYLINES.csv", polyline_rows)
    np.savez_compressed(output / "03_CORRECTED_BINARY_MASKS.npz", **mask_archive)

    structures: dict[str, dict[str, Any]] = {}
    for level in ("L4", "L5"):
        targets = [
            row
            for row in metrics
            if row["level"] == level and row["role"] == TARGET_ROLE
        ]
        structures[level] = summarize_structure(targets)

    result = {
        "schema": "sonogym-s0010-sap-observability-correction-v1",
        "historical_spatial_evaluation": "INVALIDATED_AXIS_ORDERING",
        "source_job": args.source_job,
        "source_run_root": str(args.run_root),
        "source_prediction_sha256": obs.sha256(args.prediction),
        "source_segmentation_sha256": obs.sha256(args.segmentation),
        "label_conversion_sha256": obs.sha256(args.label_conversion),
        "cortical_polylines_sha256": obs.sha256(args.polylines),
        "raw_label_replay_passed_frames": raw_replay_passed,
        "frame_count": len(metrics),
        "surface_alignment_tolerance_pixels": SURFACE_ALIGNMENT_TOLERANCE_PIXELS,
        "structures": structures,
    }
    obs.write_json(output / "04_CORRECTION_RESULT.json", result)
    provenance = {
        **result,
        "script": str(Path(__file__).resolve()),
        "prediction_path": str(args.prediction),
        "segmentation_path": str(args.segmentation),
        "label_conversion_path": str(args.label_conversion),
    }
    obs.write_json(output / "05_PROVENANCE.json", provenance)
    write_review_documents(output, result)
    source_root = Path(__file__).resolve().parents[2]
    source_output = output / "source"
    source_output.mkdir()
    for source in (
        Path(__file__).resolve(),
        Path(obs.__file__).resolve(),
        source_root / "tests/test_s0010_sap_observability.py",
        source_root / "tests/test_s0010_sap_observability_correction.py",
    ):
        shutil.copy2(source, source_output / source.name)
    for test_log in args.test_log:
        test_log_output = output / "test_logs" / test_log.name
        test_log_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(test_log, test_log_output)
    return result


def package(output_root: Path, zip_path: Path) -> str:
    checksum_path = output_root / "checksums.sha256"
    files = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(
            f"{obs.sha256(path)}  {path.relative_to(output_root)}\n" for path in files
        ),
        encoding="utf-8",
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_root.parent))
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"ZIP integrity failure: {bad}")
    return obs.sha256(zip_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--polylines", type=Path, required=True)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--source-job", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--test-log", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build(args)
    zip_sha256 = package(args.output_root, args.zip)
    print(json.dumps({"result": result, "zip": str(args.zip), "zip_sha256": zip_sha256}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
