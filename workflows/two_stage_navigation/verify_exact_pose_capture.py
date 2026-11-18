#!/usr/bin/env python3
"""Verify one bounded exact-pose SonoGym ultrasound capture on CPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml

GEOMETRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/ultrasound_plane_geometry.py"
)
GEOMETRY_SPEC = importlib.util.spec_from_file_location(
    "exact_pose_ultrasound_plane_geometry", GEOMETRY_PATH
)
if GEOMETRY_SPEC is None or GEOMETRY_SPEC.loader is None:
    raise RuntimeError("cannot load CPU ultrasound-plane geometry module")
GEOMETRY = importlib.util.module_from_spec(GEOMETRY_SPEC)
sys.modules[GEOMETRY_SPEC.name] = GEOMETRY
GEOMETRY_SPEC.loader.exec_module(GEOMETRY)
label_agreement = GEOMETRY.label_agreement
plane_from_full_pose = GEOMETRY.plane_from_full_pose
sample_label_image = GEOMETRY.sample_label_image

ACQUISITION_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/acquisition_integrity.py"
)
ACQUISITION_SPEC = importlib.util.spec_from_file_location(
    "exact_pose_acquisition_integrity", ACQUISITION_PATH
)
if ACQUISITION_SPEC is None or ACQUISITION_SPEC.loader is None:
    raise RuntimeError("cannot load CPU acquisition-integrity module")
ACQUISITION = importlib.util.module_from_spec(ACQUISITION_SPEC)
sys.modules[ACQUISITION_SPEC.name] = ACQUISITION
ACQUISITION_SPEC.loader.exec_module(ACQUISITION)
recorded_view_is_fresh = ACQUISITION.recorded_view_is_fresh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--assets-data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-known-job-51793371-coordinate-hash-defect",
        action="store_true",
        help="Historical post-hoc audit only; forbidden for new captures.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def ultrasound_sha256(image: np.ndarray) -> str:
    array = np.asarray(image, dtype="<f4")
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("ultrasound must be a finite two-dimensional array")
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") not in {
        "sonogym-s0010-exact-pose-replay-v1",
        "sonogym-s0010-exact-pose-replay-v2",
        "sonogym-s0010-tp-surface-core-dataset-v1",
        "sonogym-s0010-tp-surface-core-dataset-v2",
        "sonogym-s0010-tp-surface-core-dataset-v3",
        "sonogym-s0010-tp-surface-core-dataset-v4",
        "sonogym-s0010-tp-surface-core-dataset-v5",
        "sonogym-s0010-tp-surface-core-dataset-v6",
        "sonogym-s0010-tp-surface-core-dataset-v7",
    }:
        raise RuntimeError("unexpected protocol identity")
    seal = protocol_path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if file_sha256(protocol_path) != seal:
        raise RuntimeError("protocol seal mismatch")

    session = args.session_dir.resolve()
    public_rows = read_jsonl(session / "public/frames.jsonl")
    pose_rows = read_jsonl(session / "privileged/full_pose_replay.jsonl")
    ultrasound = np.load(session / "public/ultrasound.npz")["ultrasound"]
    raw_labels = np.load(session / "privileged/raw_labels.npz")["raw_labels"]
    coordinate_archive = np.load(
        session / "privileged/pixel_to_volume_coordinates.npz"
    )
    coordinates = coordinate_archive["human_img_coords"]
    local_coordinates = coordinate_archive["native_local_image_coordinates_m"]
    session_id = str(public_rows[0]["session_id"])
    session_spec = next(
        item for item in protocol["sessions"] if str(item["session_id"]) == session_id
    )
    expected_frames = int(
        session_spec.get(
            "expected_frame_count",
            protocol["sampling"].get("expected_frames_per_session"),
        )
    )
    lengths = {
        len(public_rows),
        len(pose_rows),
        int(ultrasound.shape[0]),
        int(raw_labels.shape[0]),
        int(coordinates.shape[0]),
    }
    if lengths != {expected_frames}:
        raise RuntimeError(f"capture arrays do not contain {expected_frames} frames: {lengths}")

    raw_path = (
        args.assets_data_dir.resolve()
        / "HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
    )
    raw_volume = np.asarray(nib.load(raw_path).dataobj, dtype=np.uint8)
    coordinate_tolerance = float(
        protocol["exact_pose_replay"]["maximum_coordinate_error_index"]
    )
    width = int(protocol["exact_pose_replay"]["image_width_pixels"])
    depth = int(protocol["exact_pose_replay"]["image_depth_pixels"])
    frame_results: list[dict[str, Any]] = []
    for index in range(expected_frames):
        public = public_rows[index]
        pose = pose_rows[index]
        if int(public["frame_sequence"]) != index or int(pose["frame_sequence"]) != index:
            raise RuntimeError("frame order is not contiguous")
        if (
            protocol["protocol_id"].startswith("sonogym-s0010-tp-surface-core-dataset-v")
            and public.get("acquisition_id") != pose.get("acquisition_id")
        ):
            raise RuntimeError(f"acquisition identity mismatch at index {index}")
        actual_hash = ultrasound_sha256(ultrasound[index])
        if actual_hash != public["frame_sha256"] or actual_hash != pose["frame_sha256"]:
            raise RuntimeError(f"frame hash mismatch at index {index}")
        coordinate_hash_matches = True
        if protocol["protocol_id"].startswith("sonogym-s0010-tp-surface-core-dataset-v"):
            if (
                array_sha256(ultrasound[index]) != pose.get("ultrasound_array_sha256")
                or array_sha256(raw_labels[index]) != pose.get("raw_label_array_sha256")
            ):
                raise RuntimeError(f"saved image or label hash mismatch at index {index}")
            coordinate_hash_matches = bool(
                array_sha256(coordinates[index])
                == pose.get("native_coordinate_array_sha256")
            )

        direct_raw = sample_label_image(
            volume=raw_volume,
            coordinates_index=coordinates[index],
            width=width,
            depth=depth,
        )
        direct_metrics = label_agreement(raw_labels[index], direct_raw)
        replay_plane = plane_from_full_pose(
            end_effector_position_m=np.asarray(pose["human_to_ee_position_m"]),
            rotation_columns=np.asarray(pose["human_to_ee_rotation_columns"]),
            image_origin_height_m=float(pose["image_origin_height_m"]),
            local_coordinates_m=local_coordinates,
            label_resolution_m=float(pose["label_resolution_m"]),
            volume_shape=tuple(int(value) for value in raw_volume.shape),
        )
        replay_raw = sample_label_image(
            volume=raw_volume,
            coordinates_index=replay_plane.coordinates_index,
            width=width,
            depth=depth,
        )
        replay_metrics = label_agreement(raw_labels[index], replay_raw)
        maximum_coordinate_error = float(
            np.max(np.abs(coordinates[index] - replay_plane.coordinates_index))
        )
        constants_match = bool(
            np.isclose(
                float(pose["image_origin_height_m"]),
                float(protocol["exact_pose_replay"]["image_origin_height_m"]),
                atol=1e-12,
            )
            and np.isclose(
                float(pose["label_resolution_m"]),
                float(protocol["exact_pose_replay"]["label_resolution_m"]),
                atol=1e-12,
            )
            and pose["image_size_width_depth"] == [width, depth]
        )
        frame_passes = bool(
            direct_metrics["exact_raw_label_agreement"] == 1.0
            and replay_metrics["passes_unchanged_gate"]
            and maximum_coordinate_error <= coordinate_tolerance
            and constants_match
        )
        frame_results.append(
            {
                "frame_sequence": index,
                "producer_sequence": int(public["producer_sequence"]),
                "frame_sha256": actual_hash,
                "direct_coordinate_metrics": direct_metrics,
                "full_pose_replay_metrics": replay_metrics,
                "maximum_coordinate_error_index": maximum_coordinate_error,
                "clipped_sample_count": replay_plane.clipped_sample_count,
                "runtime_constants_match_protocol": constants_match,
                "saved_coordinate_hash_matches_record": coordinate_hash_matches,
                "saved_coordinate_array_sha256": array_sha256(coordinates[index]),
                "frame_passes": frame_passes,
                "acquisition_id": public.get("acquisition_id"),
                "fresh_viewpoint": recorded_view_is_fresh(public),
            }
        )

    passed = sum(int(row["frame_passes"]) for row in frame_results)
    distinct_ultrasound_hashes = len({row["frame_sha256"] for row in frame_results})
    minimum_distinct_hashes = int(
        protocol["dataset_gate"]["minimum_distinct_frame_hashes_per_session"]
    )
    native_map_replay = all(
        row["direct_coordinate_metrics"]["exact_raw_label_agreement"] == 1.0
        for row in frame_results
    )
    synchronized_full_pose_replay = all(row["frame_passes"] for row in frame_results)
    tp_surface_core_protocol = protocol["protocol_id"].startswith(
        "sonogym-s0010-tp-surface-core-dataset-v"
    )
    coordinate_hash_integrity = all(
        row["saved_coordinate_hash_matches_record"] for row in frame_results
    )
    historical_hash_exception = bool(
        args.allow_known_job_51793371_coordinate_hash_defect
        and protocol["protocol_id"] == "sonogym-s0010-tp-surface-core-dataset-v1"
        and "51793371" in session.parts
    )
    if args.allow_known_job_51793371_coordinate_hash_defect and not historical_hash_exception:
        raise RuntimeError("coordinate-hash exception is restricted to historical job 51793371")
    if tp_surface_core_protocol:
        acquisition_freshness = all(row["fresh_viewpoint"] for row in frame_results)
        viewpoint_diversity = bool(
            distinct_ultrasound_hashes >= minimum_distinct_hashes
            and len({row["acquisition_id"] for row in frame_results}) == expected_frames
        )
    else:
        acquisition_freshness = True
        viewpoint_diversity = distinct_ultrasound_hashes >= minimum_distinct_hashes
    capture_passes = bool(
        native_map_replay
        and synchronized_full_pose_replay
        and acquisition_freshness
        and viewpoint_diversity
        and (coordinate_hash_integrity or historical_hash_exception)
    )
    report = {
        "schema": "sonogym-exact-pose-replay-report-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": seal,
        "source_commit": protocol["source"]["implementation_commit"],
        "raw_volume_path": str(raw_path),
        "raw_volume_sha256": file_sha256(raw_path),
        "frame_count": expected_frames,
        "distinct_ultrasound_hashes": distinct_ultrasound_hashes,
        "minimum_distinct_ultrasound_hashes": minimum_distinct_hashes,
        "passed_frame_count": passed,
        "coordinate_hash_mismatch_count": sum(
            not row["saved_coordinate_hash_matches_record"] for row in frame_results
        ),
        "coordinate_hash_note": (
            "A mismatch is a hard failure for new captures. The only permitted exception is an "
            "explicit post-hoc audit of historical job 51793371."
        ),
        "historical_coordinate_hash_exception_applied": historical_hash_exception,
        "gates": {
            "NATIVE_MAP_REPLAY": native_map_replay,
            "SYNCHRONIZED_FULL_POSE_REPLAY": synchronized_full_pose_replay,
            "ACQUISITION_FRESHNESS": acquisition_freshness,
            "VIEWPOINT_DIVERSITY": viewpoint_diversity,
            "ARRAY_HASH_INTEGRITY": coordinate_hash_integrity,
        },
        "verdict": "PROJECTION_PARITY_PASS" if capture_passes else "PROJECTION_REPAIR_BLOCKED",
        "frames": frame_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in ("frame_count", "passed_frame_count", "distinct_ultrasound_hashes", "verdict")}, sort_keys=True))
    return 0 if capture_passes else 3


if __name__ == "__main__":
    raise SystemExit(main())
