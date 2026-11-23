#!/usr/bin/env python3
"""Build weak visible-interface labels from synchronized fixed-s0010 captures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import yaml

from two_stage_navigation.contracts import ultrasound_sha256
from two_stage_navigation.sonoanatomy_labels import (
    IGNORE,
    OTHER_BONE_INTERFACE,
    SAP_INTERFACE,
    TP_INTERFACE,
    VisibleInterfaceResult,
    right_level_masks,
    visible_interface_labels,
)


LABEL_NAME = {
    0: "BACKGROUND_OR_NON_BONE",
    1: "TP_CORTICAL_INTERFACE",
    2: "SAP_CORTICAL_INTERFACE",
    3: "OTHER_VISIBLE_BONE_INTERFACE",
    255: "IGNORE_AMBIGUOUS_OR_OCCLUDED",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def expanded_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).resolve()


def bone_source_ids(path: Path) -> set[int]:
    mapping = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {int(source) for source, acoustic in mapping.items() if int(acoustic) == 13}


def session_directories(root: Path) -> list[Path]:
    sessions = root / "sessions"
    if not sessions.is_dir():
        raise FileNotFoundError(f"capture has no sessions directory: {root}")
    result = [path for path in sessions.iterdir() if path.is_dir()]
    if not result:
        raise ValueError(f"capture has no sessions: {root}")
    return sorted(result)


def selected_session_directories(
    root: Path,
    session_ids: list[str] | None,
    *,
    layout: str = "sessions_subdirectory",
) -> list[Path]:
    if layout == "sessions_subdirectory":
        candidates = session_directories(root)
    elif layout == "direct_children":
        candidates = sorted(path for path in root.iterdir() if path.is_dir())
        if not candidates:
            raise ValueError(f"capture has no direct session directories: {root}")
    else:
        raise ValueError(f"unsupported session layout: {layout}")
    available = {path.name: path for path in candidates}
    if session_ids is None:
        return sorted(available.values())
    if not session_ids or len(set(session_ids)) != len(session_ids):
        raise ValueError("source session_ids must be nonempty and unique")
    missing = sorted(set(session_ids) - set(available))
    if missing:
        raise ValueError(f"capture source is missing sessions: {missing}")
    return [available[session_id] for session_id in session_ids]


def capture_verification_passed(session: Path, mode: str) -> bool:
    if mode == "verification_report":
        verification = json.loads(
            (session / "verification_report.json").read_text(encoding="utf-8")
        )
        return bool(
            verification.get("gates")
            and all(bool(value) for value in verification["gates"].values())
            and int(verification.get("passed_frame_count", -1))
            == int(
                verification.get(
                    "frame_count", len(verification.get("frames", []))
                )
            )
        )
    if mode == "completed_stage2_summary":
        summary = json.loads((session / "summary.json").read_text(encoding="utf-8"))
        return bool(
            summary.get("status") == "COMPLETED"
            and int(summary.get("frame_count", -1)) > 0
            and (session / "results/actor_result.json.sha256").is_file()
            and (session / "results/evaluation_report.json.sha256").is_file()
        )
    raise ValueError(f"unsupported capture verification mode: {mode}")


def label_overlay(
    output: Path,
    image: np.ndarray,
    result: VisibleInterfaceResult,
    *,
    title: str,
) -> None:
    colors = {
        int(TP_INTERFACE): "#00b4d8",
        int(SAP_INTERFACE): "#f72585",
        int(OTHER_BONE_INTERFACE): "#ffd166",
        int(IGNORE): "#adb5bd",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), constrained_layout=True)
    for axis in axes:
        axis.imshow(image, cmap="gray", origin="upper")
        axis.set_xlabel("column")
        axis.set_ylabel("row")
    axes[0].set_title("clean ultrasound")
    for value, color in colors.items():
        rows, columns = np.where(result.labels == value)
        axes[1].scatter(columns, rows, s=8, c=color, label=LABEL_NAME[value])
    axes[1].set_title("weak visible-interface labels")
    axes[1].legend(loc="upper right", fontsize=6)
    rows, columns = np.where(result.first_bone)
    axes[2].scatter(columns, rows, s=7, c="#ff9f1c", label="first acoustic bone")
    rows, columns = np.where(result.image_supported & result.first_bone)
    axes[2].scatter(columns, rows, s=11, c="#80ff72", label="image-supported")
    axes[2].set_title("visibility and clean-image support")
    axes[2].legend(loc="upper right", fontsize=6)
    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def summarize(level: str, rows: list[dict[str, Any]], readiness: dict[str, Any]) -> dict[str, Any]:
    selected = [row for row in rows if row["level"] == level]
    sap = [row for row in selected if row["has_sap_interface"]]
    tp = [row for row in selected if row["has_tp_interface"]]
    other = [row for row in selected if row["other_bone_interface_pixels"] > 0]
    sap_sessions = {row["session_id"] for row in sap}
    gates = {
        "sap_positive_frames": len(sap)
        >= int(readiness["minimum_sap_positive_frames_per_level"]),
        "tp_positive_frames": len(tp)
        >= int(readiness["minimum_tp_positive_frames_per_level"]),
        "sap_positive_sessions": len(sap_sessions)
        >= int(readiness["minimum_sap_positive_sessions_per_level"]),
        "other_bone_frames": len(other)
        >= int(readiness["minimum_clear_other_bone_frames_per_level"]),
    }
    return {
        "level": level,
        "frame_count": len(selected),
        "session_count": len({row["session_id"] for row in selected}),
        "sap_positive_frame_count": len(sap),
        "sap_positive_session_count": len(sap_sessions),
        "tp_positive_frame_count": len(tp),
        "tp_positive_session_count": len({row["session_id"] for row in tp}),
        "other_bone_frame_count": len(other),
        "sap_interface_pixel_count": sum(row["sap_interface_pixels"] for row in selected),
        "tp_interface_pixel_count": sum(row["tp_interface_pixels"] for row in selected),
        "other_bone_interface_pixel_count": sum(
            row["other_bone_interface_pixels"] for row in selected
        ),
        "gate_results": gates,
        "ready_for_segmentation_training": all(gates.values()),
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    protocol_id = str(config.get("protocol_id", ""))
    if protocol_id not in {
        "s0010_sonoanatomy_dataset_v1",
        "s0010_sonoanatomy_dataset_v2",
        "s0010_sonoanatomy_dataset_v3",
        "s0010_sonoanatomy_dataset_v4",
        "s0010_sonoanatomy_dataset_v5",
    }:
        raise ValueError("unexpected dataset protocol")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    prediction_image = nib.load(str(args.prediction))
    segmentation_image = nib.load(str(args.segmentation))
    if prediction_image.shape != segmentation_image.shape or not np.allclose(
        prediction_image.affine, segmentation_image.affine
    ):
        raise ValueError("prediction and source segmentation grids differ")
    prediction = np.asarray(prediction_image.dataobj, dtype=np.uint8)
    segmentation = np.asarray(segmentation_image.dataobj, dtype=np.uint8)
    level_masks = {
        level: right_level_masks(
            prediction, segmentation, prediction_image.affine, level
        )
        for level in ("L4", "L5")
    }
    bone_ids = bone_source_ids(args.label_conversion)

    images_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    review_candidates: dict[str, list[tuple[int, np.ndarray, VisibleInterfaceResult, str]]] = {
        "L4": [],
        "L5": [],
    }

    for source in config["sources"]:
        level = str(source["level"])
        root = expanded_path(str(source["root"]))
        requested_sessions = source.get("session_ids")
        if requested_sessions is not None:
            requested_sessions = [str(value) for value in requested_sessions]
        layout = str(source.get("session_layout", "sessions_subdirectory"))
        verification_mode = str(source.get("verification", "verification_report"))
        frame_target_level = source.get("frame_target_level")
        session_namespace = str(source.get("session_id_namespace", ""))
        for session in selected_session_directories(
            root, requested_sessions, layout=layout
        ):
            verification_passed = capture_verification_passed(
                session, verification_mode
            )
            if not verification_passed:
                raise ValueError(f"capture verification did not pass: {session}")
            frames = read_jsonl(session / "public/frames.jsonl")
            replay = read_jsonl(session / "privileged/full_pose_replay.jsonl")
            replay_by_sequence = {
                int(row["frame_sequence"]): row for row in replay
            }
            images = np.load(session / "public/ultrasound.npz")["ultrasound"]
            coordinates = np.load(
                session / "privileged/pixel_to_volume_coordinates.npz"
            )["human_img_coords"]
            raw_labels = np.load(session / "privileged/raw_labels.npz")["raw_labels"]
            if not (len(frames) == len(images) == len(coordinates) == len(raw_labels)):
                raise ValueError(f"session arrays differ in length: {session}")
            if verification_mode == "completed_stage2_summary":
                summary = json.loads(
                    (session / "summary.json").read_text(encoding="utf-8")
                )
                if int(summary["frame_count"]) != len(frames):
                    raise ValueError(f"stage2 summary frame count differs: {session}")
            for index, (frame, image, coordinate_map, raw) in enumerate(
                zip(frames, images, coordinates, raw_labels)
            ):
                if (
                    frame_target_level is not None
                    and not str(frame.get("target_name", "")).startswith(
                        f"{frame_target_level}_"
                    )
                ):
                    continue
                image = np.asarray(image, dtype=np.float32)
                frame_hash = str(frame["frame_sha256"])
                if ultrasound_sha256(image) != frame_hash:
                    raise ValueError(f"image hash mismatch: {session.name}:{index}")
                if frame_hash in seen_hashes:
                    raise ValueError(f"duplicate image across sources: {frame_hash}")
                seen_hashes.add(frame_hash)
                result = visible_interface_labels(
                    image=image,
                    raw_labels=raw,
                    coordinate_map_index=coordinate_map,
                    masks=level_masks[level],
                    bone_source_ids=bone_ids,
                    maximum_track_distance_pixels=float(
                        config["image_support"]["maximum_track_distance_pixels"]
                    ),
                )
                archive_index = len(images_out)
                images_out.append(image)
                labels_out.append(result.labels)
                record = {
                    "archive_index": archive_index,
                    "level": level,
                    "source_role": source["role"],
                    "source_root": str(root),
                    "session_id": (
                        f"{session_namespace}/{session.name}"
                        if session_namespace
                        else session.name
                    ),
                    "acquisition_id": str(frame["acquisition_id"]),
                    "frame_sequence": int(frame["frame_sequence"]),
                    "trajectory_id": str(frame.get("trajectory_id", "")),
                    "frame_sha256": frame_hash,
                    "split_pose_position_m": json.dumps(
                        replay_by_sequence[index]["human_to_ee_position_m"]
                    ),
                    "split_pose_quaternion_wxyz": json.dumps(
                        replay_by_sequence[index]["human_to_ee_quaternion_wxyz"]
                    ),
                    "split_pose_source": "privileged_full_pose_replay_not_model_input",
                    "label_provenance": config["label_provenance"],
                    **result.metrics,
                }
                rows.append(record)
                priority = (
                    4 * int(record["sap_interface_pixels"])
                    + 2 * int(record["tp_interface_pixels"])
                    + min(int(record["other_bone_interface_pixels"]), 10)
                )
                review_candidates[level].append(
                    (priority, image, result, f"{session.name} frame {index}")
                )

    np.savez_compressed(
        args.output_dir / "dataset.npz",
        ultrasound=np.stack(images_out).astype(np.float32),
        labels=np.stack(labels_out).astype(np.uint8),
    )
    write_csv(args.output_dir / "manifest.csv", rows)
    summaries = {
        level: summarize(level, rows, config["readiness"]) for level in ("L4", "L5")
    }
    result = {
        "schema": "s0010-sonoanatomy-dataset-result-v1",
        "anatomy_id": "s0010",
        "label_provenance": config["label_provenance"],
        "semantic_status": "provisional_model_derived_simulator_labels",
        "frame_count": len(rows),
        "unique_frame_hash_count": len(seen_hashes),
        "levels": summaries,
        "input_hashes": {
            "config": sha256(args.config),
            "prediction": sha256(args.prediction),
            "segmentation": sha256(args.segmentation),
            "label_conversion": sha256(args.label_conversion),
        },
        "dataset_sha256": None,
        "training_gate_passed": all(
            summary["ready_for_segmentation_training"] for summary in summaries.values()
        ),
    }
    result["dataset_sha256"] = sha256(args.output_dir / "dataset.npz")
    write_json(args.output_dir / "result.json", result)
    for level, candidates in review_candidates.items():
        selected = sorted(candidates, key=lambda item: (-item[0], item[3]))[:6]
        for rank, (_, image, interface_result, title) in enumerate(selected):
            label_overlay(
                args.output_dir / "figures" / f"{level.lower()}_{rank:02d}.png",
                image,
                interface_result,
                title=f"{level}: {title}",
            )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
