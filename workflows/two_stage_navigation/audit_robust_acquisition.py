"""Audit paired pose variation, simulator speckle, and actual sweep video."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from evaluate_robust_mapping import statistics
from run_simple_landmark_mapping import read_jsonl, sha256, write_json


def paired_tracking(root):
    result = {}
    for arm in ("E1", "E2"):
        poses = []
        for seed in (2026121700, 2026121702):
            session = root / f"{arm}_{seed}"
            frames = read_jsonl(session / "public/frames.jsonl")
            native = read_jsonl(session / "privileged/full_pose_replay.jsonl")
            assert len(frames) == len(native)
            poses.append({frame["waypoint_index"]: pose for frame, pose in zip(frames, native)
                          if frame["waypoint_index"] < 54 and frame["image_valid"]
                          and frame["frame_sha256"] == pose["frame_sha256"]})
        matching = sorted(set(poses[0]) & set(poses[1]))
        first, second = [[source[index] for index in matching] for source in poses]
        translation = 1000 * np.linalg.norm(
            np.array([row["human_to_ee_position_m"] for row in first]) -
            np.array([row["human_to_ee_position_m"] for row in second]), axis=1,
        )
        rotations = [Rotation.from_matrix([row["human_to_ee_rotation_columns"] for row in pose])
                     for pose in (first, second)]
        degrees = (rotations[0].inv() * rotations[1]).magnitude() * 180 / np.pi
        result[arm] = {"matched_waypoints": matching, "translation_mm": statistics(translation),
                       "rotation_deg": statistics(degrees),
                       "errors_by_waypoint": {str(waypoint): [float(pos), float(angle)]
                                              for waypoint, pos, angle in zip(matching, translation, degrees)}}
    return result


def paired_renderer(dataset):
    metadata = json.loads((dataset / "split_manifest.json").read_text())
    rows = metadata["records"]
    with np.load(dataset / "dataset.npz") as archive:
        images, labels = archive["ultrasound"], archive["labels"]
    by_split = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        group = (row["source_session"], row["waypoint"])
        by_split[row["split"]][group][(row["perturbed"], row["speckle_field"])] = row
    comparison = {
        "same_pose_different_fields": [],
        "different_pose_same_field": [],
        "different_pose_and_field": [],
    }
    for split, groups in by_split.items():
        fields = sorted({r["speckle_field"] for r in rows if r["split"] == split})
        first, other = fields[:2]
        for group in groups.values():
            if (False, first) not in group:
                continue
            base = group[(False, first)]
            for name, key in zip(comparison, ((False, other), (True, first), (True, other))):
                if key not in group:
                    continue
                candidate = group[key]
                same_geometry = base["geometry_sha256"] == candidate["geometry_sha256"]
                if same_geometry != (name == "same_pose_different_fields"):
                    raise ValueError("saved render geometry contradicts the comparison")
                a, b = base["archive_index"], candidate["archive_index"]
                comparison[name].append({
                    "split": split, "mae": float(np.abs(images[a] - images[b]).mean()),
                    "tp_visibility_changed": bool((labels[a] == 1).any() != (labels[b] == 1).any()),
                    "sap_visibility_changed": bool((labels[a] == 2).any() != (labels[b] == 2).any()),
                })
    grouped = []
    for name, rows in comparison.items():
        for split in ("train", "validation"):
            subset = [row for row in rows if row["split"] == split]
            if subset:
                grouped.append({"comparison": name, "split": split, "pairs": len(subset),
                                "image_mae": statistics([r["mae"] for r in subset]),
                                "tp_visibility_changed": sum(r["tp_visibility_changed"] for r in subset),
                                "sap_visibility_changed": sum(r["sap_visibility_changed"] for r in subset)})
    return grouped


def verify_video(path, source):
    import cv2

    cv2.setNumThreads(1)
    receipt = json.loads((path.parent / "video_manifest.json").read_text())
    frames = read_jsonl(source / "public/frames.jsonl")
    count_valid = 0
    for row in frames:
        if row["image_valid"]:
            camera = source / "public/robot" / f"{row['frame_sequence']:03d}.jpg"
            if not camera.is_file() or camera.stat().st_size == 0:
                raise ValueError(f"missing synchronized camera image {camera}")
            count_valid += 1
    if count_valid != receipt["verified_acquisition_frames"] or count_valid != receipt["robot_camera_frames"]:
        raise ValueError("video and acquisition manifests disagree")
    video = cv2.VideoCapture(str(path))
    fps = video.get(cv2.CAP_PROP_FPS)
    wanted = (receipt["dimensions"][1], receipt["dimensions"][0], 3)
    count = 0
    while True:
        read, frame = video.read()
        if not read:
            break
        if frame.shape != wanted:
            raise ValueError("decoded video dimensions disagree with the renderer manifest")
        count += 1
    video.release()
    if count != receipt["decoded_frames"] or fps != receipt["fps"] or count <= 0:
        raise ValueError("video decode did not reproduce the renderer manifest")
    return {"sha256": sha256(path), "decoded_frames": count, "fps": fps, "duration_seconds": count / fps,
            "dimensions": receipt["dimensions"], "synchronized_frame_pairs": count_valid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("acquisition", "review", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    write_json(args.output, {
        "between_seed_executed_pose": paired_tracking(args.acquisition),
        "paired_speckle_and_pose": paired_renderer(args.review / "E3-dataset"),
        "video": verify_video(args.review / "demo/revised_systematic_sweep.mp4",
                              args.acquisition / "E1_2026121700"),
    })


if __name__ == "__main__":
    main()
