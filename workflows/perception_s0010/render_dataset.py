"""Render physically constrained s0010 poses with independent spatial speckle fields."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import resource
import time

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol", "profile", "previous", "historical", "legacy-training", "output"):
        parser.add_argument("--"+key, required=True, type=Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-receipt", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    app = launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import spinal_surgery  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import label
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    import torch
    import yaml
    from run_simple_landmark_mapping import load_evaluator, read_jsonl, sha256, write_json
    from two_stage_navigation.sonoanatomy_labels import visible_interface_labels
    from two_stage_navigation.ultrasound_plane_geometry import sample_label_image
    from replay import CONVERSION

    campaign = yaml.safe_load(args.profile.read_text())
    plan = campaign["data"]
    protocol = yaml.safe_load(args.protocol.read_text())
    output = args.output
    output.mkdir(exist_ok=False, parents=True)
    rng = np.random.default_rng(plan["seed"])
    evaluator = load_evaluator(protocol, CONVERSION)
    masks, bone_ids, _, label_hashes = evaluator
    raw_volume = np.asarray(nib.load(protocol["evaluator"]["source_segmentation_path"]).dataobj, dtype=np.uint8)
    radius = plan["legacy_and_split_buffer_mm"] / 1000
    legacy = []
    for level in ("l4", "l5"):
        with (args.legacy_training/f"{level}_split_manifest.csv").open() as handle:
            legacy.extend(json.loads(row["split_pose_position_m"]) for row in csv.DictReader(handle)
                          if row["split"] == "train" and row["included"] == "True")
    pilot = json.loads((args.previous/"review-52711211/E3-training/training_manifest.json").read_text())
    legacy.extend(row["position_m"] for row in pilot["training_records"])
    legacy_tree = cKDTree(legacy)
    if not args.benchmark:
        receipt = json.loads(args.benchmark_receipt.read_text())
        if not receipt["benchmark"] or receipt["profile_sha256"] != sha256(args.profile):
            raise ValueError("scaled generation needs the matching benchmark")
        planned_limit = min(plan["target_max_images"], int(
            receipt["images_per_second"] * plan["generation_seconds_budget"] * .85
        ))
    else:
        planned_limit = 0

    def rows(session, stop):
        f = read_jsonl(session/"public/frames.jsonl")
        poses = read_jsonl(session/"privileged/full_pose_replay.jsonl")
        assert len(f) == len(poses)
        actor = json.loads((session/"actor_result.json").read_text())
        assert actor["seed"] not in campaign["reserved_confirmation_seeds"]
        result = []
        for frame, pose in zip(f, poses):
            assert all(frame[k] == pose[k] for k in ("acquisition_id", "frame_sha256", "frame_sequence"))
            if frame["image_valid"] and frame["waypoint_index"] < stop:
                result.append({
                    "source": str(session), "source_seed": actor["seed"], "source_acquisition": frame["acquisition_id"],
                    "waypoint": frame["waypoint_index"], "phase": frame["phase"],
                    "requested_relative_pose_index": frame["requested_relative_pose_index"],
                    "level_hypotheses_at_source": frame["level_hypotheses"],
                    "position_m": pose["human_to_ee_position_m"], "rotation": pose["human_to_ee_rotation_columns"],
                    "pose_group": frame["acquisition_id"], "perturbation": 0,
                })
        return result

    def perturbed(row, repetitions):
        yield row
        initial = Rotation.from_matrix(row["rotation"])
        axes = initial.as_matrix()
        normal = axes[:, 2] * np.sign(axes[1, 2])
        origin = np.asarray(row["position_m"]) + .133 * normal
        for index in range(1, repetitions+1):
            displacement = rng.normal(0, [plan["position_sd_mm"]]*2+[plan["local_out_of_plane_sd_mm"]])
            if np.linalg.norm(displacement) > plan["max_translation_mm"]:
                displacement *= plan["max_translation_mm"] / np.linalg.norm(displacement)
            angles = rng.normal(0, plan["rotation_sd_degrees"], 3)
            if np.linalg.norm(angles) > plan["max_rotation_degrees"]:
                angles *= plan["max_rotation_degrees"] / np.linalg.norm(angles)
            changed = initial * Rotation.from_rotvec(np.deg2rad(angles))
            rotation = changed.as_matrix()
            new_normal = rotation[:, 2] * np.sign(rotation[1, 2])
            position = origin + axes @ (displacement / 1000) - .133 * new_normal
            yield dict(row, position_m=position.tolist(), rotation=rotation.tolist(), perturbation=index,
                       local_image_origin_displacement_mm=displacement.tolist(), rotation_vector_degrees=angles.tolist())

    def separated(items, other):
        if not other:
            return items
        tree = cKDTree([r["position_m"] for r in other])
        return [r for r in items if tree.query(r["position_m"])[0] >= radius]

    def spaced(items, minimum, limit):
        selected, coords = [], []
        for index in rng.permutation(len(items)):
            row = items[index]
            pos = np.asarray(row["position_m"])
            if not coords or min(np.linalg.norm(np.asarray(coords)-pos, axis=1)) >= minimum:
                coords.append(pos)
                selected.append(row)
            if len(selected) >= limit:
                break
        return selected

    old = args.previous/"acquisition-52710240"
    train_bases = rows(old/"E1_2026121700", 102)
    geo_bases = rows(old/"E1_2026121702", 149)
    combined_bases = [row for i in (1, 3) for row in rows(args.historical/f"session_{i:02d}", 54)]
    train_candidates = [pose for row in train_bases for pose in perturbed(row, plan["perturbations_per_base"])]
    combined_candidates = [pose for row in combined_bases for pose in perturbed(row, 4)]
    combined_candidates = [pose for pose in combined_candidates if legacy_tree.query(pose["position_m"])[0] >= radius]
    combined = spaced(combined_candidates, .00075, 96)
    geo_candidates = separated([pose for row in geo_bases for pose in perturbed(row, 1)], combined)
    # Reserve disjoint position neighborhoods from a different executed sweep.
    geo_band = [r for r in geo_candidates if r["requested_relative_pose_index"][0] in (136.5, 139.5, 148.5, 151.5)]
    geo = spaced(geo_band, .0010, 55)
    if len(geo) < 36:
        used = {id(row) for row in geo}
        candidates = [r for r in geo_candidates if id(r) not in used and
                      (r["waypoint"] % 11 == 0 or r["waypoint"] >= 102)]
        geo.extend(spaced(candidates, .0015, 36-len(geo)))
    candidates = separated(train_candidates, geo+combined)
    selected = spaced(candidates, .00045, int(max(plan["target_min_images"], planned_limit)*.88/5))
    if not selected or not geo or not combined:
        raise ValueError("split geometry lacks support")
    speckle = [selected[x] for x in rng.choice(len(selected), min(170, len(selected)), replace=False)]
    work = [
        ("train", selected, plan["train_fields"]),
        ("unseen_speckle", speckle, plan["unseen_speckle_fields"]),
        ("unseen_geometry", geo, plan["development_geometry_fields"]),
        ("combined", combined, plan["combined_test_fields"]),
    ]
    if args.benchmark:
        work = [("benchmark", work[0][1][:plan["benchmark_poses"]] + geo[:8] + combined[:8], plan["benchmark_fields"])]
    profile_offsets = {}
    if not args.benchmark:
        for name, entries, fields in work:
            profile_offsets[name] = {"poses": len(entries), "fields": fields}
        write_json(output/"planned_splits.json", profile_offsets)

    cfg = parse_env_cfg(protocol["scope"]["task"], device=args.device or "cuda:0", num_envs=1)
    cfg.seed = plan["data_seed"] if "data_seed" in plan else plan["seed"]
    env = gym.make(protocol["scope"]["task"], cfg=cfg)
    sensor = env.unwrapped.US_slicer
    assert sensor.sim_mode == "conv"
    device = sensor.T0_T1_map.device
    origin = torch.zeros((1, 3), device=device)
    identity = torch.tensor([[1., 0., 0., 0.]], device=device)

    def hash_array(x):
        return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

    timing = Counter()
    output_counts = {}
    sample = []
    rng_unused = hashlib.sha256()
    rendered_start = time.perf_counter()
    with (output/"manifest.jsonl").open("w") as manifest, (output/"dropped.jsonl").open("w") as dropped, torch.no_grad():
        for split, entries, fields in work:
            images, target = [], {"L4": [], "L5": []}
            rejected = 0
            counts = Counter()
            split_start = time.perf_counter()
            for field in fields:
                start = time.perf_counter()
                generator = torch.Generator(device=device).manual_seed(field)
                sensor.T0_T1_map.normal_(generator=generator)
                sensor.Vl_map.normal_(generator=generator)
                torch.cuda.synchronize()
                timing["rerandomization_seconds"] += time.perf_counter()-start
                for row in entries:
                    start = time.perf_counter()
                    pos = torch.tensor([row["position_m"]], dtype=torch.float32, device=device)
                    quat = Rotation.from_matrix(row["rotation"]).as_quat()[[3,0,1,2]]
                    quat = torch.tensor(quat[None], dtype=torch.float32, device=device)
                    sensor.slice_US(origin, identity, pos, quat)
                    image = sensor.us_img_tensor[0,:,:,0].detach().cpu().numpy().copy()
                    native = sensor.human_img_coords[0].detach().cpu().numpy()
                    contact = not bool(sensor.no_collide[0].item())
                    timing["render_seconds"] += time.perf_counter()-start
                    if not contact or not np.isfinite(image).all() or image.std() <= 1e-8:
                        dropped.write(json.dumps(dict(row, split=split, field=field, contact_proxy=contact))+"\n")
                        rejected += 1
                        continue
                    start = time.perf_counter()
                    raw = sample_label_image(volume=raw_volume, coordinates_index=native,
                                             width=image.shape[1], depth=image.shape[0])
                    record = dict(row, anatomy="s0010", split=split, field=field, contact_proxy=contact,
                                  archive_index=len(images), image_sha256=hash_array(image),
                                  geometry_sha256=hash_array(native), origin_offset_m=.133,
                                  image_resolution_mm=.5, anatomy_resolution_mm=1.5, labels={})
                    masks_for_row = {}
                    for level in target:
                        derived = visible_interface_labels(image=image, raw_labels=raw, coordinate_map_index=native,
                                                           masks=masks[level], bone_source_ids=bone_ids)
                        target[level].append(derived.labels)
                        masks_for_row[level] = derived.labels
                        fg_pixels = [int((derived.labels == code).sum()) for code in (0,1,2,3,255)]
                        for code, pixels in zip(("background","tp","sap","other","ignored"), fg_pixels):
                            counts[f"{level}_{code}_pixels"] += pixels
                            counts[f"{level}_{code}_frames"] += int(pixels > 0)
                        components = {name:int(label(derived.labels == code, np.ones((3,3)))[1])
                                      for name, code in (("tp",1),("sap",2),("other",3))}
                        for name, count in components.items():
                            counts[f"{level}_{name}_regions"] += count
                        record["labels"][level] = {**derived.metrics, "label_sha256":hash_array(derived.labels),
                                                   "component_counts":components}
                    # Draw positive and negative label examples selected without choosing a model.
                    if len(sample) < 30 and (len(images) % max(1,len(entries)//12) == 0):
                        sample.append((image, masks_for_row, split, field, row["waypoint"]))
                    timing["label_seconds"] += time.perf_counter()-start
                    images.append(image)
                    manifest.write(json.dumps(record, separators=(",",":"))+"\n")
                    rng_unused.update(record["image_sha256"].encode())
                manifest.flush()
                print(split, field, "accepted cumulative", len(images), flush=True)
            archive = output/f"{split}.npz"
            np.savez_compressed(archive, images=np.asarray(images, dtype=np.float32), **target)
            output_counts[split] = {"valid": len(images), "dropped": rejected, "attempted": len(entries)*len(fields),
                                    "distinct_planned_poses": len(entries), "fields": fields,
                                    "class_distribution": dict(counts),"elapsed_seconds": time.perf_counter()-split_start,
                                    "archive_bytes":archive.stat().st_size,"archive_sha256":sha256(archive)}

    elapsed = time.perf_counter()-rendered_start
    write_json(output/"receipt.json", {
        "benchmark": args.benchmark, "profile_sha256":sha256(args.profile), "source_protocol_sha256":sha256(args.protocol),
        "label_provenance": label_hashes, "splits":output_counts, "elapsed_seconds":elapsed,
        "images_per_second":sum(z["valid"] for z in output_counts.values())/elapsed,
        "timing":dict(timing), "gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
        "cpu_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "synchronous_image_sequence_sha256":rng_unused.hexdigest(),
        "combined_minimum_distance_from_original_and_pilot_train_mm":
            float(1000*legacy_tree.query([r["position_m"] for r in combined])[0].min()),
        "development_geometry_to_training_minimum_mm":
            float(1000*cKDTree([r["position_m"] for r in selected]).query([r["position_m"] for r in geo])[0].min()),
        "combined_to_pilot_train_minimum_mm":
            float(1000*cKDTree([r["position_m"] for r in selected]).query([r["position_m"] for r in combined])[0].min()),
    })
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chosen = []
    for level in ("L4","L5"):
        for class_id in (1,2,3):
            candidates = [pair for pair in sample if np.any(pair[1][level] == class_id)]
            chosen.append((level, class_id, candidates[0] if candidates else None))
    fig, axs = plt.subplots(2,3,figsize=(8,6),layout="constrained")
    for ax, (level,code,pair) in zip(axs.flat,chosen):
        ax.axis("off")
        if pair is None:
            continue
        im, lab, split, field, waypoint = pair
        ax.imshow(im,cmap="gray",vmin=0,vmax=np.quantile(im,.99))
        for c,col in ((1,"cyan"),(2,"magenta"),(3,"#9aff6a")):
            if np.any(lab[level] == c):
                ax.contour(lab[level] == c, levels=[.5],colors=[col],linewidths=.8)
        ax.set_title(f"{level} WP {waypoint} | {split}\nfield {field}",fontsize=8)
    fig.suptitle("Provisional visible interfaces | cyan TP · magenta SAP · green other bone")
    fig.savefig(output/"label_audit.png",dpi=160)
    plt.close(fig)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
