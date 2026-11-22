#!/usr/bin/env python3
"""Train session-separated image-only TP/SAP interface segmentation baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
from scipy import ndimage
import torch
from torch.utils.data import DataLoader, Dataset

from two_stage_navigation.sonoanatomy_segmentation import (
    SmallUNet,
    interface_loss,
    normalized_ultrasound,
)


CLASS_NAME = {0: "BACKGROUND", 1: "TP", 2: "SAP", 3: "OTHER_BONE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def quaternion_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    return float(2.0 * math.degrees(math.acos(np.clip(abs(np.dot(first, second)), 0.0, 1.0))))


def parse_pose(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(json.loads(row["actor_base_to_ee_position_m"]), dtype=np.float64),
        np.asarray(json.loads(row["actor_base_to_ee_quaternion_wxyz"]), dtype=np.float64),
    )


def near_pose(row: dict[str, str], references: list[dict[str, str]]) -> bool:
    position, quaternion = parse_pose(row)
    for reference in references:
        other_position, other_quaternion = parse_pose(reference)
        if (
            np.linalg.norm(position - other_position) * 1000.0 <= 1.5
            and quaternion_angle_deg(quaternion, other_quaternion) <= 1.2
        ):
            return True
    return False


def assign_splits(rows: list[dict[str, str]], level: str) -> list[dict[str, str]]:
    selected = [dict(row) for row in rows if row["level"] == level]
    sessions = sorted({row["session_id"] for row in selected})
    if len(sessions) < 5:
        raise ValueError(f"{level} needs at least five sessions")
    stats = []
    for session in sessions:
        current = [row for row in selected if row["session_id"] == session]
        stats.append(
            (
                sum(row["has_sap_interface"] == "True" for row in current),
                sum(row["has_tp_interface"] == "True" for row in current),
                session,
            )
        )
    ranked = [item[2] for item in sorted(stats, key=lambda item: (-item[0], -item[1], item[2]))]
    development_session = ranked[0]
    calibration_session = ranked[1]
    for row in selected:
        row["split"] = (
            "development"
            if row["session_id"] == development_session
            else "calibration"
            if row["session_id"] == calibration_session
            else "train"
        )
        row["included"] = "True"
        row["exclusion_reason"] = ""

    train = [row for row in selected if row["split"] == "train"]
    calibration = [row for row in selected if row["split"] == "calibration"]
    for row in calibration:
        if near_pose(row, train):
            row["included"] = "False"
            row["exclusion_reason"] = "NEAR_DUPLICATE_OF_TRAIN_POSE"
    admitted_calibration = [row for row in calibration if row["included"] == "True"]
    references = train + admitted_calibration
    for row in selected:
        if row["split"] == "development" and near_pose(row, references):
            row["included"] = "False"
            row["exclusion_reason"] = "NEAR_DUPLICATE_OF_EARLIER_SPLIT_POSE"
    return selected


class InterfaceDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        rows: list[dict[str, str]],
        *,
        split: str,
        augment: bool,
    ) -> None:
        self.images = images
        self.labels = labels
        self.rows = [
            row for row in rows if row["split"] == split and row["included"] == "True"
        ]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.rows[index]
        archive_index = int(row["archive_index"])
        image = torch.from_numpy(self.images[archive_index].copy()).float()
        label = torch.from_numpy(self.labels[archive_index].copy()).long()
        if self.augment:
            scale = 0.9 + 0.2 * torch.rand(())
            offset = 0.04 * (torch.rand(()) - 0.5)
            noise = 0.01 * torch.randn_like(image)
            image = image * scale + offset + noise
        return image, label, archive_index


def class_weights(labels: np.ndarray, rows: list[dict[str, str]]) -> torch.Tensor:
    indices = [int(row["archive_index"]) for row in rows if row["split"] == "train"]
    values = labels[indices]
    valid = values != 255
    counts = np.bincount(values[valid], minlength=4).astype(np.float64)
    if np.any(counts[1:] == 0):
        raise ValueError(f"training split lacks a foreground class: {counts.tolist()}")
    reference = float(np.median(counts[1:]))
    weights = np.sqrt(reference / np.maximum(counts, 1.0))
    weights = np.clip(weights, 0.25, 8.0)
    weights[0] = 0.05
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict(
    model: SmallUNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    model.eval()
    archive_indices: list[int] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for images, labels, indices in loader:
        normalized = normalized_ultrasound(images.to(device))[:, None]
        current = torch.softmax(model(normalized), dim=1).cpu().numpy()
        probabilities.extend(current)
        targets.extend(labels.numpy())
        archive_indices.extend(int(value) for value in indices)
    return archive_indices, np.asarray(probabilities), np.asarray(targets)


def frame_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    class_id: int,
    threshold: float,
    minimum_pixels: int,
) -> dict[str, int | float]:
    truth = np.any(targets == class_id, axis=(1, 2))
    predicted = np.sum(probabilities[:, class_id] >= threshold, axis=(1, 2)) >= minimum_pixels
    tp = int(np.count_nonzero(truth & predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    tn = int(np.count_nonzero(~truth & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def calibrate(probabilities: np.ndarray, targets: np.ndarray, class_id: int) -> dict[str, Any]:
    candidates = []
    for threshold in (0.30, 0.40, 0.50, 0.60, 0.70):
        for minimum_pixels in (1, 2, 3, 5):
            metrics = frame_metrics(
                probabilities,
                targets,
                class_id=class_id,
                threshold=threshold,
                minimum_pixels=minimum_pixels,
            )
            candidates.append(
                {"threshold": threshold, "minimum_pixels": minimum_pixels, **metrics}
            )
    return max(
        candidates,
        key=lambda row: (
            row["f1"],
            row["precision"],
            row["recall"],
            row["threshold"],
            row["minimum_pixels"],
        ),
    )


def pixel_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    predicted = np.argmax(probabilities, axis=1)
    valid = targets != 255
    result: dict[str, Any] = {}
    for class_id, name in CLASS_NAME.items():
        truth = (targets == class_id) & valid
        estimate = (predicted == class_id) & valid
        tp = int(np.count_nonzero(truth & estimate))
        fp = int(np.count_nonzero(~truth & estimate))
        fn = int(np.count_nonzero(truth & ~estimate))
        result[name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
    return result


def localization_error_pixels(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    class_id: int,
    threshold: float,
) -> dict[str, float | int | None]:
    distances: list[float] = []
    supported_frames = 0
    for probability, target in zip(probabilities[:, class_id], targets):
        truth = target == class_id
        prediction = probability >= threshold
        if not np.any(truth) or not np.any(prediction):
            continue
        supported_frames += 1
        distance = ndimage.distance_transform_edt(~truth)
        distances.extend(float(value) for value in distance[prediction])
    return {
        "supported_frame_count": supported_frames,
        "median_pixels": float(np.median(distances)) if distances else None,
        "p90_pixels": float(np.percentile(distances, 90)) if distances else None,
    }


def train_level(
    *,
    level: str,
    images: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, str]],
    output: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    level_rows = assign_splits(rows, level)
    included = [row for row in level_rows if row["included"] == "True"]
    split_counts = {
        split: len([row for row in included if row["split"] == split])
        for split in ("train", "calibration", "development")
    }
    split_positive = {
        split: {
            "sap": sum(
                row["has_sap_interface"] == "True"
                for row in included
                if row["split"] == split
            ),
            "tp": sum(
                row["has_tp_interface"] == "True"
                for row in included
                if row["split"] == split
            ),
        }
        for split in split_counts
    }
    if any(split_counts[split] == 0 for split in split_counts) or any(
        split_positive[split][kind] == 0
        for split in split_positive
        for kind in ("sap", "tp")
    ):
        return {
            "level": level,
            "status": "TRAINING_BLOCKED_SPLIT_SUPPORT",
            "split_counts": split_counts,
            "split_positive_frames": split_positive,
            "excluded_near_duplicate_count": sum(
                row["included"] == "False" for row in level_rows
            ),
        }
    write_csv(output / f"{level.lower()}_split_manifest.csv", level_rows)
    train_data = InterfaceDataset(images, labels, level_rows, split="train", augment=True)
    calibration_data = InterfaceDataset(
        images, labels, level_rows, split="calibration", augment=False
    )
    development_data = InterfaceDataset(
        images, labels, level_rows, split="development", augment=False
    )
    generator = torch.Generator().manual_seed(args.seed + (4 if level == "L4" else 5))
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    calibration_loader = DataLoader(calibration_data, batch_size=args.batch_size)
    development_loader = DataLoader(development_data, batch_size=args.batch_size)
    weights = class_weights(labels, level_rows).to(device)
    model = SmallUNet(base_channels=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_images, batch_labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            normalized = normalized_ultrasound(batch_images.to(device))[:, None]
            loss = interface_loss(
                model(normalized), batch_labels.to(device), weights
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        _, calibration_probabilities, calibration_targets = predict(
            model, calibration_loader, device
        )
        calibration_pixel = pixel_metrics(calibration_probabilities, calibration_targets)
        calibration_loss = 1.0 - 0.5 * (
            calibration_pixel["TP"]["dice"] + calibration_pixel["SAP"]["dice"]
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "calibration_objective": calibration_loss,
                "calibration_tp_dice": calibration_pixel["TP"]["dice"],
                "calibration_sap_dice": calibration_pixel["SAP"]["dice"],
            }
        )
        if calibration_loss + 1.0e-6 < best_loss:
            best_loss = calibration_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 10:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    calibration_indices, calibration_probabilities, calibration_targets = predict(
        model, calibration_loader, device
    )
    thresholds = {
        CLASS_NAME[class_id]: calibrate(
            calibration_probabilities, calibration_targets, class_id
        )
        for class_id in (1, 2, 3)
    }
    checkpoint = output / f"s0010_{level.lower()}_sonoanatomy_unet_v1.pt"
    torch.save(
        {
            "schema": "s0010-sonoanatomy-unet-v1",
            "level": level,
            "state_dict": best_state,
            "base_channels": 16,
            "class_weights": weights.cpu(),
            "thresholds": thresholds,
            "input_contract": "clean_ultrasound_only",
            "seed": args.seed,
        },
        checkpoint,
    )
    reloaded_bundle = torch.load(checkpoint, map_location=device, weights_only=False)
    reloaded = SmallUNet(base_channels=int(reloaded_bundle["base_channels"])).to(device)
    reloaded.load_state_dict(reloaded_bundle["state_dict"])
    reloaded.eval()
    development_indices, development_probabilities, development_targets = predict(
        reloaded, development_loader, device
    )
    development_frame = {
        CLASS_NAME[class_id]: frame_metrics(
            development_probabilities,
            development_targets,
            class_id=class_id,
            threshold=float(thresholds[CLASS_NAME[class_id]]["threshold"]),
            minimum_pixels=int(thresholds[CLASS_NAME[class_id]]["minimum_pixels"]),
        )
        for class_id in (1, 2, 3)
    }
    result = {
        "level": level,
        "status": "TRAINED_OPENED_DEVELOPMENT",
        "split_counts": split_counts,
        "split_positive_frames": split_positive,
        "excluded_near_duplicate_count": sum(row["included"] == "False" for row in level_rows),
        "training_session_count": len(
            {row["session_id"] for row in included if row["split"] == "train"}
        ),
        "calibration_session_count": len(
            {row["session_id"] for row in included if row["split"] == "calibration"}
        ),
        "development_session_count": len(
            {row["session_id"] for row in included if row["split"] == "development"}
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "epochs_completed": len(history),
        "class_weights": weights.cpu().tolist(),
        "calibration_thresholds": thresholds,
        "development_frame_metrics": development_frame,
        "development_pixel_metrics": pixel_metrics(
            development_probabilities, development_targets
        ),
        "development_localization_error_pixels": {
            CLASS_NAME[class_id]: localization_error_pixels(
                development_probabilities,
                development_targets,
                class_id=class_id,
                threshold=float(thresholds[CLASS_NAME[class_id]]["threshold"]),
            )
            for class_id in (1, 2)
        },
        "development_archive_indices": development_indices,
        "calibration_archive_indices": calibration_indices,
        "scientific_status": "opened_development_not_independent_confirmation",
    }
    write_csv(output / f"{level.lower()}_training_history.csv", history)
    write_json(output / f"{level.lower()}_result.json", result)
    return result


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    archive = np.load(args.dataset_root / "dataset.npz")
    images = np.asarray(archive["ultrasound"], dtype=np.float32)
    labels = np.asarray(archive["labels"], dtype=np.uint8)
    rows = read_csv(args.dataset_root / "manifest.csv")
    if len(rows) != len(images) or len(images) != len(labels):
        raise ValueError("dataset arrays and manifest differ in length")
    if len({row["frame_sha256"] for row in rows}) != len(rows):
        raise ValueError("dataset contains duplicate images")
    results = {
        level: train_level(
            level=level,
            images=images,
            labels=labels,
            rows=rows,
            output=args.output_dir,
            args=args,
            device=device,
        )
        for level in ("L4", "L5")
    }
    summary = {
        "schema": "s0010-sonoanatomy-segmentation-result-v1",
        "dataset_sha256": sha256(args.dataset_root / "dataset.npz"),
        "manifest_sha256": sha256(args.dataset_root / "manifest.csv"),
        "device": str(device),
        "seed": args.seed,
        "levels": results,
        "independent_confirmation": "NOT_RUN",
    }
    write_json(args.output_dir / "result.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
