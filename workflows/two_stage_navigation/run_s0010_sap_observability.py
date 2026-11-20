#!/usr/bin/env python3
"""Plan and package a fixed-s0010 pseudo-SAP ultrasound observability pilot.

The pretrained CT prediction and exact pixel-to-volume coordinates are
privileged offline evidence.  Clean ultrasound remains the only input to the
phase-separated image review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import yaml
from scipy import ndimage


LEVEL_LABEL = {"L4": 28, "L5": 27}
SAP_LABEL = 6
IMAGE_DEPTH = 200
IMAGE_WIDTH = 150
IMAGE_SHAPE = (IMAGE_DEPTH, IMAGE_WIDTH)
STRUCTURE = ndimage.generate_binary_structure(3, 1)
MINIMUM_CORE_PIXELS = 20
MINIMUM_CORE_MARGIN_PIXELS = 4
INTERFACE_ALIGNMENT_TOLERANCE_PIXELS = 4.0
ALLOWED_DECISIONS = {"VISIBLE", "AMBIGUOUS", "NOT_VISIBLE"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def pseudo_sap_masks(
    prediction: np.ndarray, segmentation: np.ndarray, affine: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    """Return frozen right-sided pseudo-SAP, core, and one-voxel uncertainty."""
    if prediction.shape != segmentation.shape:
        raise ValueError("prediction and segmentation shapes differ")
    x_world = nib.affines.apply_affine(
        affine,
        np.column_stack(
            (
                np.arange(prediction.shape[0]),
                np.zeros(prediction.shape[0]),
                np.zeros(prediction.shape[0]),
            )
        ),
    )[:, 0]
    result: dict[str, dict[str, np.ndarray]] = {}
    for level, source_label in LEVEL_LABEL.items():
        level_points = np.argwhere(segmentation == source_label)
        if not len(level_points):
            raise ValueError(f"source segmentation has no {level}")
        midline = float(np.median(x_world[level_points[:, 0]]))
        right_half = x_world[:, None, None] > midline
        pseudo = (prediction == SAP_LABEL) & (segmentation == source_label) & right_half
        core = ndimage.binary_erosion(pseudo, structure=STRUCTURE, iterations=1)
        uncertainty = ndimage.binary_dilation(
            pseudo, structure=STRUCTURE, iterations=1
        ) & ~core
        if not np.any(pseudo) or not np.any(core):
            raise ValueError(f"{level} pseudo-SAP or one-voxel core is empty")
        result[level] = {
            "pseudo": pseudo,
            "core": core,
            "uncertainty": uncertainty,
            "midline_world_x_mm": np.asarray(midline),
        }
    return result


def project_mask(
    coordinates: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinate map must be N x 3")
    indices = np.trunc(coords).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < np.asarray(mask.shape)), axis=1)
    hits = np.zeros(len(indices), dtype=bool)
    valid_indices = indices[valid]
    hits[valid] = mask[
        valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
    ]
    # The native slicer flattens (width, depth, elevation), then the actor image
    # transposes width/depth. A direct (depth, width) reshape moves pixels.
    image_hits = hits.reshape(IMAGE_WIDTH, IMAGE_DEPTH).T
    source_hits = mask[
        valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
    ]
    source_voxels = np.unique(valid_indices[source_hits], axis=0)
    return image_hits, source_voxels


def project_mask_legacy_v1(
    coordinates: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the invalid V1/V1R1 rasterization for explicit comparison."""
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinate map must be N x 3")
    indices = np.trunc(coords).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < np.asarray(mask.shape)), axis=1)
    hits = np.zeros(len(indices), dtype=bool)
    valid_indices = indices[valid]
    hits[valid] = mask[
        valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
    ]
    image_hits = hits.reshape(IMAGE_SHAPE)
    source_hits = mask[
        valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
    ]
    source_voxels = np.unique(valid_indices[source_hits], axis=0)
    return image_hits, source_voxels


def projection_diagnostics(
    coordinates: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    projector: Any = project_mask,
) -> dict[str, Any]:
    projected: dict[str, np.ndarray] = {}
    source_counts: dict[str, int] = {}
    for name in ("pseudo", "core", "uncertainty"):
        projected[name], source = projector(coordinates, masks[name])
        source_counts[name] = int(len(source))
    rows, columns = np.where(projected["core"])
    if len(rows):
        margin = int(
            min(
                rows.min(),
                IMAGE_SHAPE[0] - 1 - rows.max(),
                columns.min(),
                IMAGE_SHAPE[1] - 1 - columns.max(),
            )
        )
        centroid_row = float(np.mean(rows))
        centroid_column = float(np.mean(columns))
    else:
        margin = -1
        centroid_row = centroid_column = -1.0
    return {
        "projected": projected,
        "pseudo_pixel_count": int(np.count_nonzero(projected["pseudo"])),
        "core_pixel_count": int(np.count_nonzero(projected["core"])),
        "uncertainty_pixel_count": int(np.count_nonzero(projected["uncertainty"])),
        "pseudo_source_voxel_count": source_counts["pseudo"],
        "core_source_voxel_count": source_counts["core"],
        "uncertainty_source_voxel_count": source_counts["uncertainty"],
        "core_margin_pixels": margin,
        "core_centroid_row": centroid_row,
        "core_centroid_column": centroid_column,
        "projection_gate_passed": bool(
            len(rows) >= MINIMUM_CORE_PIXELS
            and margin >= MINIMUM_CORE_MARGIN_PIXELS
        ),
    }


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    seal = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if sha256(path) != seal:
        raise ValueError("protocol seal mismatch")
    if protocol["protocol_id"] not in {
        "sonogym-s0010-sap-observability-v1",
        "sonogym-s0010-sap-observability-v1r1",
        "sonogym-s0010-sap-observability-v2",
        "sonogym-s0010-sap-observability-v2r1",
    }:
        raise ValueError("unexpected protocol")
    return protocol


def validate_frozen_plan(
    *, protocol: dict[str, Any], reference_root: Path, masks: dict[str, dict[str, np.ndarray]]
) -> dict[str, Any]:
    references: dict[tuple[str, int], tuple[dict[str, Any], np.ndarray]] = {}
    for session_dir in sorted(reference_root.iterdir()):
        if not session_dir.is_dir():
            continue
        frames = read_jsonl(session_dir / "public/frames.jsonl")
        coordinates = np.load(
            session_dir / "privileged/pixel_to_volume_coordinates.npz"
        )["human_img_coords"]
        for index, frame in enumerate(frames):
            references[(session_dir.name, index)] = (frame, coordinates[index])
    rows: list[dict[str, Any]] = []
    for bank_id, specs in protocol["observability_waypoints"].items():
        poses = protocol["sampling"]["absolute_waypoint_banks_index"][bank_id]
        if len(poses) != len(specs):
            raise ValueError(f"bank metadata mismatch: {bank_id}")
        for frame_index, (pose, spec) in enumerate(zip(poses, specs)):
            frame, coordinates = references[
                (str(spec["anchor_session_id"]), int(spec["anchor_frame_sequence"]))
            ]
            if not np.allclose(
                np.asarray(frame["requested_relative_pose_index"], dtype=np.float64),
                np.asarray(pose, dtype=np.float64),
                atol=1e-6,
            ):
                raise ValueError(f"frozen pose differs from anchor: {bank_id}:{frame_index}")
            level = str(spec["level"])
            diagnostics = projection_diagnostics(coordinates, masks[level])
            expected_positive = spec["role"] == "PSEUDO_SAP_TARGET"
            role_passed = bool(
                diagnostics["projection_gate_passed"]
                if expected_positive
                else diagnostics["pseudo_pixel_count"] == 0
            )
            rows.append(
                {
                    "bank_id": bank_id,
                    "frame_sequence": frame_index,
                    "level": level,
                    "role": spec["role"],
                    "anchor_session_id": spec["anchor_session_id"],
                    "anchor_frame_sequence": spec["anchor_frame_sequence"],
                    "anchor_frame_sha256": frame["frame_sha256"],
                    **{key: value for key, value in diagnostics.items() if key != "projected"},
                    "role_gate_passed": role_passed,
                }
            )
    if not all(row["role_gate_passed"] for row in rows):
        raise ValueError("at least one frozen waypoint fails its declared projection role")
    return {
        "schema": "sonogym-s0010-sap-observability-plan-v1",
        "status": "CPU_PREFLIGHT_PASS",
        "minimum_core_pixels": MINIMUM_CORE_PIXELS,
        "minimum_core_margin_pixels": MINIMUM_CORE_MARGIN_PIXELS,
        "interface_alignment_tolerance_pixels": INTERFACE_ALIGNMENT_TOLERANCE_PIXELS,
        "waypoints": rows,
    }


def save_clean_png(path: Path, image: np.ndarray) -> None:
    array = np.asarray(image, dtype=np.float64)
    lo, hi = np.percentile(array, (0.5, 99.5))
    normalized = np.clip((array - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    plt.imsave(path, normalized, cmap="gray", vmin=0.0, vmax=1.0)


def save_overlay_png(
    path: Path,
    image: np.ndarray,
    projected: dict[str, np.ndarray],
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 7), constrained_layout=True)
    axis.imshow(image, cmap="gray", origin="upper")
    for name, color, size in (
        ("uncertainty", "#ff9f1c", 6),
        ("pseudo", "#2ec4b6", 8),
        ("core", "#e71d36", 12),
    ):
        rows, columns = np.where(projected[name])
        axis.scatter(columns, rows, s=size, c=color, label=name, alpha=0.65)
    axis.set_title(title)
    axis.set_xlim(0, IMAGE_SHAPE[1] - 1)
    axis.set_ylim(IMAGE_SHAPE[0] - 1, 0)
    axis.legend(loc="upper right", fontsize=8)
    axis.set_xlabel("image column")
    axis.set_ylabel("image row")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def prepare_review(
    *, protocol_path: Path, run_root: Path, prediction_path: Path,
    segmentation_path: Path, output_root: Path
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    prediction_image = nib.load(prediction_path)
    segmentation_image = nib.load(segmentation_path)
    if not np.array_equal(prediction_image.affine, segmentation_image.affine):
        raise ValueError("prediction and segmentation affines differ")
    prediction = np.asarray(prediction_image.dataobj)
    segmentation = np.asarray(segmentation_image.dataobj)
    masks = pseudo_sap_masks(prediction, segmentation, prediction_image.affine)
    output_root.mkdir(parents=True, exist_ok=False)
    blind_dir = output_root / "blind_clean_images"
    overlay_dir = output_root / "privileged_overlays"
    blind_dir.mkdir()
    overlay_dir.mkdir()
    records: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for session in protocol["sessions"]:
        session_id = str(session["session_id"])
        session_dir = run_root / "sessions" / session_id
        frames = read_jsonl(session_dir / "public/frames.jsonl")
        images = np.load(session_dir / "public/ultrasound.npz")["ultrasound"]
        coords = np.load(
            session_dir / "privileged/pixel_to_volume_coordinates.npz"
        )["human_img_coords"]
        specs = protocol["observability_waypoints"][
            str(session["absolute_waypoint_bank_id"])
        ]
        for index, (frame, image, coordinate_map, spec) in enumerate(
            zip(frames, images, coords, specs)
        ):
            token = hashlib.sha256(
                f"{protocol['protocol_id']}:{frame['acquisition_id']}".encode("utf-8")
            ).hexdigest()[:12]
            review_id = f"R_{token}"
            level = str(spec["level"])
            diagnostics = projection_diagnostics(coordinate_map, masks[level])
            clean_name = f"{review_id}.png"
            overlay_name = f"{session_id}_frame_{index:02d}_{level}.png"
            save_clean_png(blind_dir / clean_name, image)
            save_overlay_png(
                overlay_dir / overlay_name,
                image,
                diagnostics.pop("projected"),
                f"{session_id} frame {index} {level} {spec['role']}",
            )
            row = {
                "review_id": review_id,
                "session_id": session_id,
                "frame_sequence": index,
                "split": session["split"],
                "level": level,
                "role": spec["role"],
                "acquisition_id": frame["acquisition_id"],
                "producer_sequence": frame["producer_sequence"],
                "frame_sha256": frame["frame_sha256"],
                "clean_image": f"blind_clean_images/{clean_name}",
                "privileged_overlay": f"privileged_overlays/{overlay_name}",
                **diagnostics,
            }
            records.append(row)
            private_rows.append(
                {
                    "review_id": review_id,
                    "clean_image": row["clean_image"],
                    "source_array_index": index,
                    "source_ultrasound_archive": str(
                        session_dir / "public/ultrasound.npz"
                    ),
                    "source_frame_sha256": frame["frame_sha256"],
                }
            )
    records.sort(key=lambda row: (row["session_id"], row["frame_sequence"]))
    private_rows.sort(key=lambda row: row["review_id"])
    write_csv(output_root / "projection_manifest.csv", records)
    write_json(output_root / "blind_key.json", private_rows)
    write_csv(
        output_root / "blind_decisions.csv",
        [
            {
                "review_id": row["review_id"],
                "decision": "",
                "interface_points_rc": "",
                "notes": "",
            }
            for row in private_rows
        ],
    )
    (output_root / "00_BLIND_REVIEW_INSTRUCTIONS.md").write_text(
        "# Clean-image review\n\n"
        "Inspect only `blind_clean_images/` before filling `blind_decisions.csv`. "
        "Do not open `blind_key.json`, `projection_manifest.csv`, or "
        "`privileged_overlays/` until the decision file has been hashed.\n\n"
        "For every image, record `VISIBLE`, `AMBIGUOUS`, or `NOT_VISIBLE` for a "
        "coherent cortical interface with an acoustic shadow. Record one or more "
        "candidate interface points as `row:column;row:column`. This phase does "
        "not establish SAP identity.\n",
        encoding="utf-8",
    )
    provenance = {
        "schema": "sonogym-s0010-sap-observability-review-preparation-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(protocol_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256(prediction_path),
        "segmentation_path": str(segmentation_path),
        "segmentation_sha256": sha256(segmentation_path),
        "frame_count": len(records),
        "blind_review_is_phase_separated_not_independent": True,
    }
    write_json(output_root / "provenance.json", provenance)
    return provenance


def parse_points(value: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for token in value.split(";"):
        if not token.strip():
            continue
        row, column = token.split(":", 1)
        points.append((float(row), float(column)))
    return points


def finalize_review(review_root: Path, output: Path) -> dict[str, Any]:
    with (review_root / "projection_manifest.csv").open(newline="", encoding="utf-8") as handle:
        projection_rows = list(csv.DictReader(handle))
    with (review_root / "blind_decisions.csv").open(newline="", encoding="utf-8") as handle:
        decisions = {row["review_id"]: row for row in csv.DictReader(handle)}
    decision_sha = sha256(review_root / "blind_decisions.csv")
    joined: list[dict[str, Any]] = []
    for row in projection_rows:
        decision = decisions.get(row["review_id"])
        if decision is None or decision["decision"] not in ALLOWED_DECISIONS:
            raise ValueError(f"missing or invalid blind decision: {row['review_id']}")
        points = parse_points(decision["interface_points_rc"])
        centroid = np.asarray(
            [float(row["core_centroid_row"]), float(row["core_centroid_column"])]
        )
        aligned = bool(
            points
            and float(
                min(np.linalg.norm(np.asarray(point) - centroid) for point in points)
            )
            <= INTERFACE_ALIGNMENT_TOLERANCE_PIXELS
        )
        joined.append(
            {
                **row,
                "clean_image_decision": decision["decision"],
                "interface_points_rc": decision["interface_points_rc"],
                "review_notes": decision["notes"],
                "interface_point_aligned_with_core_centroid": aligned,
            }
        )
    structures: dict[str, Any] = {}
    for level in ("L4", "L5"):
        positives = [
            row
            for row in joined
            if row["level"] == level and row["role"] == "PSEUDO_SAP_TARGET"
        ]
        sessions = sorted({row["session_id"] for row in positives})
        qualifying = [
            row
            for row in positives
            if row["clean_image_decision"] == "VISIBLE"
            and row["interface_point_aligned_with_core_centroid"]
            and row["projection_gate_passed"].lower() == "true"
        ]
        qualifying_sessions = {row["session_id"] for row in qualifying}
        if len(qualifying) >= 2 and qualifying_sessions == set(sessions) and len(sessions) >= 2:
            verdict = "VISIBLE"
        elif any(
            row["clean_image_decision"] in {"VISIBLE", "AMBIGUOUS"}
            for row in positives
        ):
            verdict = "AMBIGUOUS"
        else:
            verdict = "NOT_VISIBLE"
        structures[level] = {
            "verdict": verdict,
            "positive_view_count": len(positives),
            "qualifying_view_count": len(qualifying),
            "positive_session_count": len(sessions),
            "qualifying_session_count": len(qualifying_sessions),
            "visible_count": sum(row["clean_image_decision"] == "VISIBLE" for row in positives),
            "ambiguous_count": sum(row["clean_image_decision"] == "AMBIGUOUS" for row in positives),
            "not_visible_count": sum(row["clean_image_decision"] == "NOT_VISIBLE" for row in positives),
        }
    result = {
        "schema": "sonogym-s0010-sap-observability-result-v1",
        "blind_decisions_sha256": decision_sha,
        "review_design": "single_agent_phase_separated_not_independent",
        "interface_alignment_tolerance_pixels": INTERFACE_ALIGNMENT_TOLERANCE_PIXELS,
        "structures": structures,
        "rows": joined,
    }
    write_json(output, result)
    write_csv(output.with_suffix(".csv"), joined)
    return result


def package_review(review_root: Path, zip_path: Path) -> None:
    checksums = review_root / "checksums.sha256"
    # The checksum manifest cannot contain a checksum of its previous contents.
    # Excluding it also makes repeated packaging deterministic and verifiable.
    files = sorted(
        path
        for path in review_root.rglob("*")
        if path.is_file() and path != checksums
    )
    checksums.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(review_root)}\n" for path in files),
        encoding="utf-8",
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(review_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(review_root.parent))
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"ZIP integrity failure: {bad}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--protocol", type=Path, required=True)
    plan.add_argument("--reference-root", type=Path, required=True)
    plan.add_argument("--prediction", type=Path, required=True)
    plan.add_argument("--segmentation", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--prediction", type=Path, required=True)
    prepare.add_argument("--segmentation", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--review-root", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    package = subparsers.add_parser("package")
    package.add_argument("--review-root", type=Path, required=True)
    package.add_argument("--zip", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "plan":
        protocol = load_protocol(args.protocol)
        prediction_image = nib.load(args.prediction)
        segmentation_image = nib.load(args.segmentation)
        masks = pseudo_sap_masks(
            np.asarray(prediction_image.dataobj),
            np.asarray(segmentation_image.dataobj),
            prediction_image.affine,
        )
        plan = validate_frozen_plan(
            protocol=protocol, reference_root=args.reference_root, masks=masks
        )
        plan["prediction_sha256"] = sha256(args.prediction)
        plan["segmentation_sha256"] = sha256(args.segmentation)
        write_json(args.output, plan)
        print(json.dumps({"status": plan["status"], "waypoints": len(plan["waypoints"])}))
    elif args.command == "prepare":
        result = prepare_review(
            protocol_path=args.protocol,
            run_root=args.run_root,
            prediction_path=args.prediction,
            segmentation_path=args.segmentation,
            output_root=args.output_root,
        )
        print(json.dumps(result, sort_keys=True))
    elif args.command == "finalize":
        result = finalize_review(args.review_root, args.output)
        print(json.dumps(result["structures"], sort_keys=True))
    else:
        package_review(args.review_root, args.zip)
        print(json.dumps({"zip": str(args.zip), "sha256": sha256(args.zip)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
