"""Pilot fine-tune the L5 U-Net on segregated simulator-rendered appearances."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from scipy.spatial.distance import cdist
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from run_simple_landmark_mapping import sha256, write_json
from train_s0010_sonoanatomy_segmentation import (
    NEGATIVE_PRESENCE_TOP_K_PIXELS, NEGATIVE_PRESENCE_WEIGHT, negative_presence_loss, pixel_metrics,
)
from two_stage_navigation.sonoanatomy_segmentation import (
    interface_loss, load_segmentation_checkpoint, normalized_ultrasound,
)


def accepted_mask(probabilities, thresholds):
    winner = np.argmax(probabilities, axis=0)
    accepted = np.zeros(winner.shape, dtype=np.uint8)
    for value, semantic in ((1, "TP"), (2, "SAP")):
        rule = thresholds[semantic]
        components, _ = label(
            (winner == value) & (probabilities[value] >= rule["threshold"]), structure=np.ones((3, 3)),
        )
        keep = np.bincount(components.ravel()) >= rule["minimum_pixels"]
        keep[0] = False
        accepted[keep[components]] = value
    return accepted


@torch.no_grad()
def evaluate(model, images, labels, records, thresholds, device, batch_size):
    model.eval()
    probability = []
    for batch in torch.from_numpy(images).split(batch_size):
        data = normalized_ultrasound(batch.to(device))[:, None]
        probability.extend(torch.softmax(model(data), dim=1).cpu().numpy())
    probability = np.stack(probability)
    masks = np.stack([accepted_mask(p, thresholds) for p in probability])
    metrics = pixel_metrics(probability, labels)
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[(row["source_session"], row["waypoint"])].append(i)
    for value, semantic in ((1, "TP"), (2, "SAP")):
        truth = (labels == value).any(axis=(1, 2))
        predicted = (masks == value).any(axis=(1, 2))
        tp = int((truth & predicted).sum())
        fp = int((~truth & predicted).sum())
        fn = int((truth & ~predicted).sum())
        competing = (~truth) & (labels == 3).any(axis=(1, 2))
        stability = [tuple(predicted[i] for i in indices) for indices in groups.values()
                     if len(indices) == 2]
        same = sum(a == b for a, b in stability)
        positive_pair = [(a, b) for (a, b), indices in zip(stability, [
            idx for idx in groups.values() if len(idx) == 2
        ]) if truth[indices].all()]
        metrics[semantic].update({
            "frame_tp": tp, "frame_fp": fp, "frame_fn": fn,
            "frame_precision": tp / (tp + fp) if tp + fp else None,
            "frame_recall": tp / (tp + fn) if tp + fn else None,
            "false_accepts_on_competing_bone_frames": int((competing & predicted).sum()),
            "competing_bone_frames": int(competing.sum()),
            "cross_field_presence_agreement": [int(same), len(stability)],
            "cross_field_retained_both_on_visible_pairs": [
                int(sum(a and b for a, b in positive_pair)), len(positive_pair),
            ],
        })
    return metrics, masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "protocol", "dataset", "original-dataset", "original-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--recover-completed-pilot", type=Path)
    args = parser.parse_args()
    profile = yaml.safe_load(args.profile.read_text())["E3"]
    specification = yaml.safe_load(args.protocol.read_text())["models"]["sonoanatomy"]["L5"]
    checkpoint = Path(specification["path"])
    if sha256(checkpoint) != specification["sha256"]:
        raise ValueError("L5 source checkpoint checksum mismatch")
    args.output.mkdir(exist_ok=False, parents=True)
    torch.manual_seed(profile["optimizer_seed"])
    device = torch.device(args.device)
    model, original = load_segmentation_checkpoint(checkpoint, expected_level="L5")
    model.to(device)
    thresholds = original["thresholds"]
    metadata = json.loads((args.dataset / "split_manifest.json").read_text())
    records = metadata["records"]
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "validation"]
    archive = np.load(args.dataset / "dataset.npz")
    images, labels = archive["ultrasound"].astype(np.float32), archive["labels"].astype(np.uint8)
    triplets = lambda rows: [row["archive_index"] for row in rows]
    train_images, train_labels = images[triplets(train)], labels[triplets(train)]
    val_images, val_labels = images[triplets(val)], labels[triplets(val)]
    if not len(train) or not len(val) or any(not (val_labels == k).any() for k in (1, 2)):
        write_json(args.output / "training_status.json", {"status": "INSUFFICIENT_SPLIT_SUPPORT"})
        return

    if args.recover_completed_pilot:
        source = args.recover_completed_pilot
        receipt = json.loads((source / "training_manifest.json").read_text())
        if (receipt["original_checkpoint_sha256"] != sha256(checkpoint)
                or receipt["validation_records"] != val or receipt["training_records"] != train):
            raise ValueError("completed pilot does not match the dataset and original checkpoint")
        target = source / "s0010_l5_sonoanatomy_unet_robust_pilot.pt"
        tuned, bundle = load_segmentation_checkpoint(target, expected_level="L5")
        if bundle["thresholds"] != thresholds:
            raise ValueError("recovered checkpoint has different semantic thresholds")
        before, before_masks = evaluate(model, val_images, val_labels, val, thresholds, device, profile["batch_size"])
        after, after_masks = evaluate(tuned.to(device), val_images, val_labels, val, thresholds, device,
                                      profile["batch_size"])
        with np.load(source / "validation_predictions.npz") as old:
            exact = {
                "baseline": int(np.all(before_masks == old["baseline"], axis=(1, 2)).sum()),
                "fine_tuned": int(np.all(after_masks == old["fine_tuned"], axis=(1, 2)).sum()),
            }
        count = len(receipt["retained_original_train_examples"])
        log = source.parent / "training.log"
        history = [float(line.split()[-1]) for line in log.read_text().splitlines() if line.startswith("epoch ")]
        if len(history) != profile["max_epochs"]:
            raise ValueError("training log did not record the fixed number of epochs")
        write_json(args.output / "training_result.json", {
            "status": "COMPLETED_METRICS_RECOVERED", "checkpoint": str(target),
            "checkpoint_sha256": sha256(target), "training_count": len(train) + count,
            "new_training_count": len(train), "original_training_count": count,
            "validation_count": len(val), "history": history, "original": before, "fine_tuned": after,
            "selection": "last fixed epoch; no selection on validation",
            "saved_gpu_masks_reproduced_by_cpu_per_checkpoint": exact,
        })
        return

    with args.original_manifest.open() as f:
        candidates = [row for row in csv.DictReader(f) if row["included"] == "True" and row["split"] == "train"]
    distances = cdist([json.loads(r["split_pose_position_m"]) for r in candidates],
                      [r["position_m"] for r in val]).min(axis=1)
    by_type = defaultdict(list)
    for row, distance in zip(candidates, distances):
        if distance < metadata["minimum_requested_train_validation_distance_m"]:
            continue
        key = (row["has_tp_interface"], row["has_sap_interface"])
        by_type[key].append(row)
    rng = np.random.default_rng(profile["optimizer_seed"])
    retained = []
    for group in by_type.values():
        retained.extend(group[i] for i in rng.choice(len(group), min(24, len(group)), replace=False))
    original_indices = [int(row["archive_index"]) for row in retained]
    with np.load(args.original_dataset) as prior:
        train_images = np.concatenate([train_images, prior["ultrasound"][original_indices].astype(np.float32)])
        train_labels = np.concatenate([train_labels, prior["labels"][original_indices].astype(np.uint8)])
    write_json(args.output / "training_manifest.json", {
        "original_checkpoint": str(checkpoint), "original_checkpoint_sha256": sha256(checkpoint),
        "training_records": train, "validation_records": val, "retained_original_train_examples": retained,
        "thresholds": thresholds, "optimizer_profile": profile,
    })
    batch_size = profile["batch_size"]
    metrics_before, mask_before = evaluate(model, val_images, val_labels, val, thresholds, device, batch_size)
    dataset = TensorDataset(torch.from_numpy(train_images), torch.from_numpy(train_labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(profile["optimizer_seed"]))
    label_counts = np.bincount(train_labels[train_labels < 4], minlength=4).astype(float)
    weights = torch.as_tensor(np.sqrt((label_counts.mean() + 1) / (label_counts + 1)).clip(.2, 5),
                              device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=profile["learning_rate"], weight_decay=1e-4)
    history = []
    for epoch in range(profile["max_epochs"]):
        model.train()
        losses = []
        for image, target in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(normalized_ultrasound(image.to(device))[:, None])
            target = target.long().to(device)
            loss = interface_loss(logits, target, weights) + NEGATIVE_PRESENCE_WEIGHT * negative_presence_loss(
                logits, target, top_k_pixels=NEGATIVE_PRESENCE_TOP_K_PIXELS,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
        print("epoch", epoch + 1, "loss", history[-1], flush=True)

    target = args.output / "s0010_l5_sonoanatomy_unet_robust_pilot.pt"
    torch.save(dict(original, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}), target)
    metrics_after, mask_after = evaluate(model, val_images, val_labels, val, thresholds, device, batch_size)
    np.savez_compressed(args.output / "validation_predictions.npz", baseline=mask_before,
                        fine_tuned=mask_after, ultrasound=val_images, labels=val_labels)
    write_json(args.output / "training_result.json", {
        "status": "COMPLETED", "checkpoint": str(target), "checkpoint_sha256": sha256(target),
        "training_count": len(dataset), "new_training_count": len(train), "original_training_count": len(retained),
        "validation_count": len(val), "history": history, "original": metrics_before, "fine_tuned": metrics_after,
        "selection": "last fixed epoch; no selection on validation",
    })


if __name__ == "__main__":
    main()
