"""Stage only recorded B-mode, public poses and visible-interface pixel annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from two_stage_navigation.acquisition_integrity import validate_actor_6dof_record
from two_stage_navigation.contracts import ultrasound_sha256
from two_stage_navigation.simple_landmark_mapping import LEVELS

from .actor_pixels import visible_pixels

POSE_FIELDS = (
    "frame_sequence", "waypoint_index", "frame_sha256", "phase", "stage1_session_id", "acquisition_id",
    "actor_base_to_ee_position_m", "actor_base_to_ee_rotation_columns",
    "requested_relative_pose_index", "image_valid",
)
PIXEL_FIELDS = ("frame_sequence", "pixel_row_col", "identity", "confidence")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        sha = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def verify(path: Path, expected: str) -> str:
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"checksum mismatch: {path}")
    return actual


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text())


def probability_maps(images, cached, models, loaded):
    from two_stage_navigation.sonoanatomy_segmentation import (
        load_segmentation_checkpoint, segmentation_probabilities,
    )

    result = {}
    for level in LEVELS:
        base = cached[level]
        if len(base) == len(images):
            result[level] = base
            continue
        if level not in loaded:
            loaded[level], _ = load_segmentation_checkpoint(models[level]["path"], expected_level=level)
        tail = np.concatenate([segmentation_probabilities(loaded[level], images[i:i + 1])
                               for i in range(len(base), len(images))])
        result[level] = np.concatenate([base, tail])
    return result


def stage(source: Path, cache: Path, config: Path, output: Path) -> None:
    rules = read_json(config)
    output.mkdir(parents=True, exist_ok=False)
    record_manifest = {}
    checked_models = set()
    loaded_models = {}
    actor_regions = []
    prefix_hashes = {}
    for session, relative in rules["recordings"].items():
        root = source / relative
        receipt_path = root / "receipt.json"
        sha = verify(receipt_path, (root / "receipt.json.sha256").read_text().split()[0])
        receipt = read_json(receipt_path)
        for level, checkpoint in receipt["model_receipt"]["models"].items():
            if checkpoint["sha256"] != rules["checkpoint_sha256"][level]:
                raise ValueError(f"wrong {level} model: {session}")
            if checkpoint["path"] not in checked_models:
                verify(Path(checkpoint["path"]), checkpoint["sha256"])
                checked_models.add(checkpoint["path"])
        recorded_paths = {}
        for field in ("ultrasound", "actor_pose_source"):
            file = receipt[field]
            recorded_paths[field] = Path(file["path"])
            verify(recorded_paths[field], file["sha256"])
        ledger = [json.loads(line) for line in recorded_paths["actor_pose_source"].read_text().splitlines()]
        sequences = receipt["valid_sequences"]
        if sequences != list(range(len(sequences))):
            raise ValueError("this protocol requires a contiguous validated acquisition prefix")
        with np.load(recorded_paths["ultrasound"], allow_pickle=False) as volume:
            images = volume["ultrasound"][sequences]
        if images.shape != (len(sequences), *rules["image_shape"]):
            raise ValueError(f"unexpected ultrasound dimensions: {session}")
        frames = []
        last_producer = -1
        acquisition_ids = set()
        for i, image in enumerate(images):
            frame = ledger[i]
            validate_actor_6dof_record(frame)
            if (frame["frame_sequence"] != i or frame["waypoint_index"] != i
                    or frame["frame_sha256"] != ultrasound_sha256(image)
                    or frame["stage1_session_id"] != rules["stage1_session_id"]
                    or any(frame[key] is not True for key in (
                        "image_valid", "contact_ok", "producer_advanced", "transition_valid",
                    ))):
                raise ValueError(f"invalid synchronized frame: {session}:{i}")
            if frame["producer_sequence"] <= last_producer or frame["acquisition_id"] in acquisition_ids:
                raise ValueError(f"capture is stale or reused within session: {session}:{i}")
            last_producer = frame["producer_sequence"]
            acquisition_ids.add(frame["acquisition_id"])
            frames.append({key: frame[key] for key in POSE_FIELDS})
        out = output / session
        out.mkdir()
        np.savez_compressed(out / "bmode.npz", ultrasound=images)
        write_json(out / "public_poses.json", frames)
        raw_actor = recorded_paths["actor_pose_source"].parent.parent / "actor_result.json"
        stage1_sha = verify(raw_actor, raw_actor.with_suffix(".json.sha256").read_text().split()[0])
        stage1 = read_json(raw_actor)["stage1_region"]
        if set(stage1) != {"home_x", "l4_z", "l5_z", "ambiguous_z"}:
            raise ValueError("unexpected Stage 1 region fields")
        actor_regions.append(stage1)
        write_json(out / "stage1_region.json", stage1)

        common_seed = session.split("-")[0]
        cache_root = cache / common_seed
        cached_receipt = read_json(cache_root / "receipt.json")
        hashes = [frame["frame_sha256"] for frame in frames]
        cached_sequences = cached_receipt["valid_sequences"]
        if session == common_seed:
            prefix_hashes[session] = hashes
            if receipt["model_receipt"]["arrays"] != cached_receipt["arrays"]:
                raise ValueError("probability receipt differs from the paired acquisition")
        elif prefix_hashes[common_seed] != hashes[:len(cached_sequences)]:
            raise ValueError("supplemental acquisition differs from cached prefix")
        if (cached_sequences != list(range(len(cached_sequences)))
                or cached_receipt["models"] != receipt["model_receipt"]["models"]
                or cached_receipt["thresholds"] != receipt["thresholds"]):
            raise ValueError("cached probability provenance mismatch")
        cached = {}
        for level, record in cached_receipt["arrays"].items():
            path = cache_root / record["file"]
            verify(path, record["sha256"])
            cached[level] = np.load(path, allow_pickle=False, mmap_mode="r")
            if cached[level].shape != (len(cached_sequences), 4, *rules["image_shape"]):
                raise ValueError("cached U-Net probability dimensions changed")
        started = perf_counter()
        models = receipt["model_receipt"]["models"]
        probability = probability_maps(images, cached, models, loaded_models)
        inference_time = perf_counter() - started
        started = perf_counter()
        actor = visible_pixels(frames, images, probability, stage1, receipt["thresholds"], rules)
        routing_time = perf_counter() - started
        np.savez_compressed(out / "learned_pixels.npz", **actor)

        src = root / "oracle.npz"
        verify(src, receipt["oracle"]["output"]["npz"]["sha256"])
        with np.load(src, allow_pickle=False) as cloud:
            # Oracle diagnostic access is restricted to recorded visible 2D evidence.
            values = {key: cloud[key] for key in PIXEL_FIELDS}
        index = values["frame_sequence"]
        pixels = values["pixel_row_col"]
        confidences = values["confidence"]
        if (np.any((index < 0) | (index >= len(images))) or pixels.shape != (len(index), 2)
                or np.any((pixels < 0) | (pixels >= rules["image_shape"]))
                or np.any(~np.isfinite(confidences)) or np.any((confidences < 0) | (confidences > 1))):
            raise ValueError(f"invalid visible Oracle evidence: {session}")
        np.savez_compressed(out / "oracle_pixels.npz", **values)
        record_manifest[session] = {
            "source_receipt_sha256": sha, "source_receipt": str(receipt_path),
            "source_ultrasound_sha256": receipt["ultrasound"]["sha256"],
            "source_public_pose_sha256": receipt["actor_pose_source"]["sha256"],
            "source_stage1_region_receipt_sha256": stage1_sha,
            "source_oracle_visible_pixel_cache_sha256": receipt["oracle"]["output"]["npz"]["sha256"],
            "source_probability_cache": cached_receipt["arrays"],
            "cached_probability_frames": len(cached_sequences),
            "supplemental_cpu_inference_frames": len(images) - len(cached_sequences),
            "supplemental_cpu_inference_seconds": inference_time if len(images) > len(cached_sequences) else 0.,
            "public_learned_pixel_extraction_seconds": routing_time,
            "source_calibration_parity_mm": receipt["calibration_parity"]["maximum_error_mm"],
            "source_thresholds": receipt["thresholds"],
            "frame_count": len(images),
            "staged_sha256": {item.name: digest(item) for item in sorted(out.iterdir())},
        }
    if any(region != actor_regions[0] for region in actor_regions):
        raise ValueError("the Stage 1 reference differs across sessions")
    left, right, stop = rules["shared_prefix"]
    lframes = read_json(output / left / "public_poses.json")
    rframes = read_json(output / right / "public_poses.json")
    prefix_fields = tuple(key for key in POSE_FIELDS if key != "acquisition_id")
    if len(lframes) != stop or any(
        a[key] != b[key] for a, b in zip(lframes, rframes[:stop]) for key in prefix_fields
    ):
        raise ValueError("the declared shared image and executed pose prefix did not verify")
    write_json(output / "manifest.json", {
        "configuration_sha256": digest(config),
        "base_commit": rules["base_commit"],
        "recordings": record_manifest,
        "checkpoint_sha256": rules["checkpoint_sha256"],
        "shared_prefix_verified": rules["shared_prefix"],
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", required=True, type=Path)
    cli.add_argument("--cache", required=True, type=Path)
    cli.add_argument("--config", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    stage(args.source, args.cache, args.config, args.output)
