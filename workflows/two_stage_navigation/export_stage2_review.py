#!/usr/bin/env python3
"""Export a small, deterministic Stage 2 image-review package.

Clean ultrasound images are actor-visible. Colored raw-label overlays are privileged,
evaluator-only visualizations and are stored in a separate directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw


MAJOR_LABELS = {
    25: (220, 40, 40),
    26: (255, 128, 0),
    27: (255, 210, 0),
    28: (30, 220, 80),
    29: (0, 210, 220),
    30: (50, 110, 255),
    31: (210, 50, 230),
}
SPLIT_ORDER = ("train", "calibration", "development_evaluation")


def record_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["session_id"]), int(record["frame_sequence"])


def _ordered(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=record_key)


def select_review_records(
    records: list[dict[str, Any]], maximum_items: int = 18
) -> list[dict[str, Any]]:
    """Select representative frames by frozen, auditable rules."""
    if maximum_items <= 0:
        raise ValueError("maximum_items must be positive")
    if not records:
        raise ValueError("records must not be empty")

    selected: dict[tuple[str, int], dict[str, Any]] = {}
    reasons: dict[tuple[str, int], list[str]] = defaultdict(list)

    def add(reason: str, candidates: list[dict[str, Any]], score: Callable[[dict[str, Any]], Any], reverse: bool = False) -> None:
        if not candidates:
            return
        candidate = sorted(_ordered(candidates), key=score, reverse=reverse)[0]
        key = record_key(candidate)
        selected[key] = candidate
        reasons[key].append(reason)

    for split in SPLIT_ORDER:
        split_rows = [row for row in records if row["split"] == split]
        valid = [row for row in split_rows if int(row["validity_label"]) == 1]
        invalid = [row for row in split_rows if int(row["validity_label"]) == 0]
        add(f"highest_quality_valid_{split}", valid, lambda row: float(row["quality_label"]), True)
        add(f"highest_quality_invalid_{split}", invalid, lambda row: float(row["quality_label"]), True)

    valid_all = [row for row in records if int(row["validity_label"]) == 1]
    invalid_all = [row for row in records if int(row["validity_label"]) == 0]
    technical = [row for row in records if bool(row["technical_image_valid"])]
    add("lowest_quality_valid", valid_all, lambda row: float(row["quality_label"]))
    add(
        "closest_validity_boundary",
        valid_all,
        lambda row: min(
            10.0 - float(row["position_error_mm"]),
            10.0 - float(row["orientation_error_deg"]),
            100.0 * (float(row["l4_visible_fraction"]) - 0.05),
        ),
    )
    add("highest_l4_fraction_invalid", invalid_all, lambda row: float(row["l4_visible_fraction"]), True)
    add(
        "highest_adjacent_level_negative",
        invalid_all,
        lambda row: sum(
            int(row["raw_label_pixel_counts"].get(str(label), 0))
            for label in (26, 27, 29, 30)
        ),
        True,
    )
    add("lowest_l4_fraction_technical", technical, lambda row: float(row["l4_visible_fraction"]))
    add("largest_position_error", records, lambda row: float(row["position_error_mm"]), True)
    add("largest_orientation_error", records, lambda row: float(row["orientation_error_deg"]), True)

    for session_id in sorted({str(row["session_id"]) for row in records}):
        session_rows = [row for row in records if row["session_id"] == session_id]
        add(
            f"highest_quality_session_{session_id}",
            session_rows,
            lambda row: float(row["quality_label"]),
            True,
        )

    if len(selected) < maximum_items:
        remaining = [row for row in records if record_key(row) not in selected]
        remaining.sort(key=lambda row: (-float(row["quality_label"]), record_key(row)))
        for row in remaining:
            key = record_key(row)
            selected[key] = row
            reasons[key].append("quality_rank_fill")
            if len(selected) >= maximum_items:
                break

    result = []
    for key in list(selected)[:maximum_items]:
        row = dict(selected[key])
        row["selection_reasons"] = reasons[key]
        result.append(row)
    return result


def normalize_ultrasound(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.shape != (200, 150) or not np.all(np.isfinite(image)):
        raise ValueError("ultrasound must be a finite 200x150 array")
    low, high = np.percentile(image, [1.0, 99.0])
    if high <= low:
        raise ValueError("ultrasound has no usable intensity range")
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def clean_rgb(image: np.ndarray) -> np.ndarray:
    gray = np.rint(normalize_ultrasound(image) * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def evaluator_overlay(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rgb = clean_rgb(image).astype(np.float32)
    labels = np.asarray(labels)
    if labels.shape != (200, 150):
        raise ValueError("raw labels must have shape 200x150")
    for label, color in MAJOR_LABELS.items():
        mask = labels == label
        if np.any(mask):
            rgb[mask] = 0.55 * rgb[mask] + 0.45 * np.asarray(color, dtype=np.float32)
    return np.rint(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)


def _load_records(capture_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for annotation_path in sorted(capture_root.glob("sessions/*/privileged/annotations.jsonl")):
        session_root = annotation_path.parents[1]
        public_by_key = {}
        with (session_root / "public" / "frames.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                public_by_key[(row["session_id"], int(row["frame_sequence"]))] = row
        with annotation_path.open(encoding="utf-8") as handle:
            for line in handle:
                annotation = json.loads(line)
                key = (annotation["session_id"], int(annotation["frame_sequence"]))
                public = public_by_key[key]
                if public["frame_sha256"] != annotation["frame_sha256"]:
                    raise RuntimeError(f"frame hash mismatch for {key}")
                records.append({**annotation, "split": public["split"], "_session_root": str(session_root)})
    if not records:
        raise RuntimeError("no complete Stage 2 session annotations found")
    return records


def _write_png(array: np.ndarray, path: Path, scale: int = 3) -> None:
    image = Image.fromarray(array, mode="RGB")
    image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    image.save(path)


def _contact_sheet(items: list[dict[str, Any]], output: Path) -> None:
    tile_width, tile_height = 900, 700
    columns = 2
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_width * columns, tile_height * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(items):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        clean = Image.open(item["clean_path"]).convert("RGB")
        overlay = Image.open(item["overlay_path"]).convert("RGB")
        clean.thumbnail((420, 560))
        overlay.thumbnail((420, 560))
        sheet.paste(clean, (x + 20, y + 70))
        sheet.paste(overlay, (x + 460, y + 70))
        caption = (
            f"{item['id']} valid={item['validity_label']} quality={item['quality_label']:.3f} "
            f"pos={item['position_error_mm']:.2f}mm angle={item['orientation_error_deg']:.2f}deg "
            f"L4={100.0 * item['l4_visible_fraction']:.1f}%"
        )
        draw.text((x + 20, y + 15), caption, fill="black")
        draw.text((x + 20, y + 45), "LEFT: clean actor image     RIGHT: PRIVILEGED evaluator overlay", fill=(120, 0, 0))
    sheet.save(output)


def _write_reproducible_zip(directory: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(str(Path(directory.name) / path.relative_to(directory)))
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--zip-output", required=True, type=Path)
    parser.add_argument("--maximum-items", type=int, default=18)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    clean_dir = args.output_dir / "clean_actor_ultrasound"
    overlay_dir = args.output_dir / "privileged_evaluator_overlays"
    clean_dir.mkdir()
    overlay_dir.mkdir()

    records = _load_records(args.capture_root)
    selected = select_review_records(records, args.maximum_items)
    archive_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    exported: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        session_root = Path(row.pop("_session_root"))
        if session_root not in archive_cache:
            ultrasound = np.load(session_root / "public" / "ultrasound.npz")["ultrasound"]
            labels = np.load(session_root / "privileged" / "raw_labels.npz")["raw_labels"]
            archive_cache[session_root] = ultrasound, labels
        ultrasound, labels = archive_cache[session_root]
        index = int(row["raw_label_archive_index"])
        item_id = f"{rank:02d}_{row['session_id']}_frame{int(row['frame_sequence']):03d}"
        clean_path = clean_dir / f"{item_id}_clean.png"
        overlay_path = overlay_dir / f"{item_id}_PRIVILEGED_overlay.png"
        _write_png(clean_rgb(ultrasound[index]), clean_path)
        _write_png(evaluator_overlay(ultrasound[index], labels[index]), overlay_path)
        exported.append(
            {
                "id": item_id,
                **{key: value for key, value in row.items() if key not in {"goal_cmd_pose", "command_pose_index", "controller_target_pose_index", "requested_waypoint_pose_index"}},
                "clean_file": str(clean_path.relative_to(args.output_dir)),
                "overlay_file": str(overlay_path.relative_to(args.output_dir)),
                "clean_path": str(clean_path),
                "overlay_path": str(overlay_path),
            }
        )

    manifest_rows = [
        {key: value for key, value in row.items() if key not in {"clean_path", "overlay_path"}}
        for row in exported
    ]
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_fields = (
        "id", "session_id", "split", "trajectory_id", "frame_sequence", "frame_sha256",
        "validity_label", "quality_label", "position_error_mm", "orientation_error_deg",
        "l4_visible_fraction", "technical_image_valid", "selection_reasons", "clean_file", "overlay_file",
    )
    with (args.output_dir / "selection_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({**row, "selection_reasons": ";".join(row["selection_reasons"])})

    _contact_sheet(exported, args.output_dir / "contact_sheet_clean_vs_privileged.png")
    protocol = next((args.capture_root / "context").glob("stage2_dataset_protocol_v*.yaml"))
    shutil.copy2(protocol, args.output_dir / protocol.name)
    shutil.copy2(args.capture_root / "results" / "capture_summary.json", args.output_dir / "capture_summary.json")
    readme = """# SonoGym Stage 2 image review package

- `clean_actor_ultrasound/`: clean ultrasound pixels available to the future actor.
- `privileged_evaluator_overlays/`: colored raw anatomical labels for reviewer interpretation only.
- `contact_sheet_clean_vs_privileged.png`: paired overview; left is clean, right is privileged.
- `selection_manifest.json/csv`: frozen selection reasons, labels, hashes and metrics.

Label colors: 25 sacrum red, 26 S1 orange, 27 L5 yellow, 28 L4 green,
29 L3 cyan, 30 L2 blue, 31 L1 magenta. Raw labels, target errors and overlays are
privileged offline annotation/evaluation data and must never enter actor inference.

Visible overlays and nonzero pixels establish geometric/image generation correspondence only;
they do not prove that clean ultrasound morphology is independently recognizable.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")

    checksum_lines = []
    for path in sorted(p for p in args.output_dir.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {path.relative_to(args.output_dir)}")
    (args.output_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    _write_reproducible_zip(args.output_dir, args.zip_output)
    print(json.dumps({"selected": len(exported), "zip": str(args.zip_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
