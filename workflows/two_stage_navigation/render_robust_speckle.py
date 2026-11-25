"""Render the same synchronized probe geometry under independently seeded speckle fields."""

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "protocol", "reference", "acquisition", "output", "label-conversion"):
        parser.add_argument("--" + name, required=True, type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    app = launcher.app

    import gymnasium as gym
    import numpy as np
    from scipy.spatial.distance import cdist
    from scipy.spatial.transform import Rotation
    import torch
    import yaml
    import nibabel as nib
    import isaaclab_tasks  # noqa: F401
    import spinal_surgery  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from run_simple_landmark_mapping import load_evaluator, read_jsonl, write_json
    from two_stage_navigation.sonoanatomy_labels import visible_interface_labels
    from two_stage_navigation.ultrasound_plane_geometry import sample_label_image

    profile = yaml.safe_load(args.profile.read_text())["E3"]
    protocol = yaml.safe_load(args.protocol.read_text())
    output = args.output
    output.mkdir(exist_ok=False, parents=True)
    labels_for_level, bone_ids, _, volume_seals = load_evaluator(protocol, args.label_conversion)
    source_labels = np.asarray(nib.load(protocol["evaluator"]["source_segmentation_path"]).dataobj, dtype=np.uint8)

    def pose_rows(path):
        frames = read_jsonl(path / "public/frames.jsonl")
        poses = read_jsonl(path / "privileged/full_pose_replay.jsonl")
        assert len(frames) == len(poses)
        rows = []
        for f, p in zip(frames, poses):
            if not f["image_valid"]:
                continue
            assert f["frame_sha256"] == p["frame_sha256"] and f["acquisition_id"] == p["acquisition_id"]
            rows.append({
                "source_session": str(path),
                "waypoint": f["waypoint_index"],
                "source_frame_sequence": f["frame_sequence"],
                "lateral": f["requested_relative_pose_index"][0],
                "phase": f["phase"],
                "position_m": p["human_to_ee_position_m"],
                "rotation": p["human_to_ee_rotation_columns"],
                "perturbed": False,
            })
        return rows

    acquired = args.acquisition / "E1_2026121700"
    if not (acquired / "actor_result.json").exists():
        acquired = args.reference / "session_00"
    nominal_train = [p for p in pose_rows(acquired) if p["lateral"] <= 144.0]
    validation = [p for i in (1, 3) for p in pose_rows(args.reference / f"session_{i:02d}")
                  if p["lateral"] >= 147.0]
    validation_xyz = np.asarray([row["position_m"] for row in validation])
    buffer_m = profile["validation_xz_neighborhood_buffer_mm"] / 1000

    # The two validation trajectories and fields are absent from the pilot training split.
    def far_from_validation(row):
        return float(cdist(np.array([row["position_m"]]), validation_xyz).min()) >= buffer_m

    nominal_train = [p for p in nominal_train if far_from_validation(p)]
    train = list(nominal_train)
    for i, row in enumerate(nominal_train):
        rotation = Rotation.from_matrix(row["rotation"])
        direction = 1 if i % 2 else -1
        translation = rotation.apply([direction * profile["perturbation_mm"] / 1000, 0, 0])
        changed = rotation * Rotation.from_euler("z", direction * profile["perturbation_deg"], degrees=True)
        sample = dict(row, position_m=(np.array(row["position_m"]) + translation).tolist(),
                      rotation=changed.as_matrix().tolist(), perturbed=True)
        if far_from_validation(sample):
            train.append(sample)

    cfg = parse_env_cfg(protocol["scope"]["task"], device=args.device or "cuda:0", num_envs=1)
    cfg.seed = 2026121700
    env = gym.make(protocol["scope"]["task"], cfg=cfg)
    sensor = env.unwrapped.US_slicer
    assert sensor.sim_mode == "conv"
    device = sensor.T0_T1_map.device
    origin = torch.zeros((1, 3), device=device)
    identity = torch.tensor([[1., 0., 0., 0.]], device=device)
    result_rows, images, labels = [], [], []
    attempts = []

    def set_speckle(field):
        generator = torch.Generator(device=device).manual_seed(field)
        sensor.T0_T1_map.normal_(generator=generator)
        sensor.Vl_map.normal_(generator=generator)

    def render(row):
        pos = torch.tensor([row["position_m"]], dtype=torch.float32, device=device)
        xyzw = Rotation.from_matrix(row["rotation"]).as_quat()
        quat = torch.tensor(xyzw[[3, 0, 1, 2]][None], dtype=torch.float32, device=device)
        sensor.slice_US(origin, identity, pos, quat)
        image = sensor.us_img_tensor[0, :, :, 0].detach().cpu().numpy().copy()
        coordinates = sensor.human_img_coords[0].detach().cpu().numpy().copy()
        return image, coordinates, not bool(sensor.no_collide[0].item())

    def hash_array(a):
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

    @torch.no_grad()
    def acquire():
        # Use a known L5 TP waypoint whose recorded starts led to different maps.
        pair = [next(row for row in pose_rows(args.reference / f"session_{i:02d}") if row["waypoint"] == 5)
                for i in (2, 3)]
        diagnostic = {}
        states = ((0, 393931, "A0"), (0, 393932, "A1"), (1, 393931, "B1"),
                  (1, 393932, "C1"), (0, 393931, "D0"))
        for pose, field, name in states:
            set_speckle(field)
            image, native, contact = render(pair[pose])
            raw = sample_label_image(volume=source_labels, coordinates_index=native, width=image.shape[1],
                                     depth=image.shape[0])
            interface = visible_interface_labels(image=image, raw_labels=raw, coordinate_map_index=native,
                                                 masks=labels_for_level["L5"], bone_source_ids=bone_ids)
            diagnostic[name] = (image, native, contact, field, pair[pose], interface.labels)
        control = {}
        a, coords, _, _, _, _ = diagnostic["A0"]
        for name, (image, native, contact, field, row, mask) in diagnostic.items():
            control[name] = {
                **row, "speckle_field": field, "contact_proxy": contact,
                "image_sha256": hash_array(image), "geometry_sha256": hash_array(native),
                "same_geometry_as_A0": bool(np.array_equal(native, coords)),
                "same_image_as_A0": bool(np.array_equal(image, a)),
                "image_mae_from_A0": float(np.abs(a - image).mean()),
                "visible_l5_tp_pixels": int((mask == 1).sum()),
            }
        write_json(output / "factorial.json", control)
        np.savez_compressed(output / "factorial.npz",
                            **{f"{name}_image": row[0] for name, row in diagnostic.items()},
                            **{f"{name}_labels": row[5] for name, row in diagnostic.items()})
        for split, rows in (("train", train), ("validation", validation)):
            for field in profile[split + "_speckle_fields"]:
                set_speckle(field)
                for row in rows:
                    image, native, contact = render(row)
                    attempts.append(dict(row, speckle_field=field, split=split, contact_proxy=contact))
                    if not contact or not np.isfinite(image).all() or image.std() <= 1e-8:
                        continue
                    raw = sample_label_image(volume=source_labels, coordinates_index=native,
                                             width=image.shape[1], depth=image.shape[0])
                    result = visible_interface_labels(image=image, raw_labels=raw, coordinate_map_index=native,
                                                       masks=labels_for_level["L5"], bone_source_ids=bone_ids)
                    result_rows.append(dict(row, split=split, speckle_field=field,
                                            archive_index=len(images), image_sha256=hash_array(image),
                                            geometry_sha256=hash_array(native), label_sha256=hash_array(result.labels),
                                            **result.metrics))
                    images.append(image)
                    labels.append(result.labels)
                print(split, field, "accepted cumulative", len(images), flush=True)

    try:
        acquire()
        np.savez_compressed(output / "dataset.npz", ultrasound=images, labels=labels)
        write_json(output / "split_manifest.json", {
            "records": result_rows,
            "attempts": attempts,
            "evaluator_inputs_sha256": volume_seals,
            "minimum_requested_train_validation_distance_m": buffer_m,
            "actual_minimum_accepted_train_validation_distance_m": float(cdist(
                [p["position_m"] for p in result_rows if p["split"] == "train"],
                [p["position_m"] for p in result_rows if p["split"] == "validation"],
            ).min()),
        })
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()
