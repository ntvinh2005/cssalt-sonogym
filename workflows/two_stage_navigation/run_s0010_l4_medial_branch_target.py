#!/usr/bin/env python3
"""Evaluate RIGHT L4 medial-branch bony-target localization at RIGHT L5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any
import zipfile

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import yaml

import run_s0010_sap_observability as sap_projection
from two_stage_navigation.l4_medial_branch_target import (
    TargetContractGeometry,
    TargetViewHypothesis,
    build_target_contract_geometry,
    distance_to_target_region_mm,
    fixed_tp_relative_hypothesis,
    image_target_hypothesis,
    target_evidence_label,
)
from two_stage_navigation.stage1_interfaces import track_bright_interfaces


IMAGE_SHAPE = (200, 150)
L5_LABEL = 27
SAP_LABEL = 6
TP_LABEL = 5
TARGET_NAME = "L5_RIGHT_TP_SURFACE_CORE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--tp-revision", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
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


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    seal_path = path.with_suffix(".sha256")
    expected = seal_path.read_text(encoding="utf-8").split()[0]
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"protocol seal mismatch: {actual} != {expected}")
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != "s0010_l4_medial_branch_target_v1":
        raise ValueError("unexpected protocol identifier")
    return protocol, expected


def load_target_geometry(
    prediction_path: Path, segmentation_path: Path
) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray, TargetContractGeometry]:
    prediction_image = nib.load(str(prediction_path))
    segmentation_image = nib.load(str(segmentation_path))
    if prediction_image.shape != segmentation_image.shape or not np.allclose(
        prediction_image.affine, segmentation_image.affine
    ):
        raise ValueError("prediction and segmentation grids differ")
    prediction = np.asanyarray(prediction_image.dataobj).astype(np.uint8)
    segmentation = np.asanyarray(segmentation_image.dataobj).astype(np.uint8)
    x_world = nib.affines.apply_affine(
        prediction_image.affine,
        np.column_stack(
            (
                np.arange(prediction.shape[0]),
                np.zeros(prediction.shape[0]),
                np.zeros(prediction.shape[0]),
            )
        ),
    )[:, 0]
    level_points = np.argwhere(segmentation == L5_LABEL)
    if not len(level_points):
        raise ValueError("L5 source segmentation is empty")
    midline = float(np.median(x_world[level_points[:, 0]]))
    right = x_world[:, None, None] > midline
    sap_mask = (prediction == SAP_LABEL) & (segmentation == L5_LABEL) & right
    tp_mask = (prediction == TP_LABEL) & (segmentation == L5_LABEL) & right
    contract = build_target_contract_geometry(
        sap_mask=sap_mask, tp_mask=tp_mask, affine=prediction_image.affine
    )
    return prediction_image, sap_mask, tp_mask, contract


def first_bone_mask(raw_labels: np.ndarray, bone_ids: set[int]) -> np.ndarray:
    bone = np.isin(raw_labels, list(bone_ids))
    first = np.zeros_like(bone, dtype=bool)
    for column in range(bone.shape[1]):
        rows = np.flatnonzero(bone[:, column])
        if len(rows):
            first[int(rows[0]), column] = True
    return first


def projected_masks(
    coordinates: np.ndarray,
    contract: TargetContractGeometry,
    sap_mask: np.ndarray,
    tp_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        name: sap_projection.project_mask(coordinates, mask)[0]
        for name, mask in {
            "sap": sap_mask,
            "tp": tp_mask,
            "sap_core": contract.sap_core,
            "tp_core": contract.tp_core,
            "sap_uncertain": contract.sap_uncertain,
            "tp_uncertain": contract.tp_uncertain,
        }.items()
    }


def evaluate_hypothesis(
    hypothesis: TargetViewHypothesis,
    *,
    label: str,
    coordinates: np.ndarray,
    affine: np.ndarray,
    contract: TargetContractGeometry,
    tolerance_mm: float,
) -> dict[str, Any]:
    distance = None
    if hypothesis.row is not None and hypothesis.column is not None:
        distance = distance_to_target_region_mm(
            coordinate_map_index=coordinates,
            row=hypothesis.row,
            column=hypothesis.column,
            affine=affine,
            midpoint_world_mm=contract.midpoint_world_mm,
        )
    accepted = hypothesis.status == "ACCEPT"
    return {
        "status": hypothesis.status,
        "row": hypothesis.row,
        "column": hypothesis.column,
        "evidence_score": hypothesis.evidence_score,
        "proposal_track_distance_pixels": hypothesis.proposal_track_distance_pixels,
        "reason": hypothesis.reason,
        "distance_to_target_region_mm": distance,
        "correct_accept": bool(
            accepted and label == "PROXY_POSITIVE" and distance is not None and distance <= tolerance_mm
        ),
        "wrong_accept": bool(accepted and label == "PROXY_NEGATIVE"),
        "ambiguous_accept": bool(accepted and label == "PROXY_AMBIGUOUS"),
    }


def summarize(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    accepted = [row for row in rows if row[f"{prefix}_status"] == "ACCEPT"]
    distances = [
        row[f"{prefix}_distance_to_target_region_mm"]
        for row in accepted
        if row[f"{prefix}_distance_to_target_region_mm"] is not None
    ]
    sessions = sorted({row["session_id"] for row in rows})
    return {
        "frame_count": len(rows),
        "session_count": len(sessions),
        "accepted_frame_count": len(accepted),
        "correct_accepted_frame_count": sum(row[f"{prefix}_correct_accept"] for row in rows),
        "wrong_accepted_frame_count": sum(row[f"{prefix}_wrong_accept"] for row in rows),
        "ambiguous_accepted_frame_count": sum(row[f"{prefix}_ambiguous_accept"] for row in rows),
        "abstained_frame_count": len(rows) - len(accepted),
        "median_distance_to_target_region_mm": float(np.median(distances)) if distances else None,
        "minimum_distance_to_target_region_mm": float(np.min(distances)) if distances else None,
        "maximum_distance_to_target_region_mm": float(np.max(distances)) if distances else None,
    }


def save_anatomy_figure(
    path: Path,
    prediction_image: nib.Nifti1Image,
    sap_mask: np.ndarray,
    tp_mask: np.ndarray,
    contract: TargetContractGeometry,
) -> None:
    center_world = np.mean(contract.midpoint_world_mm, axis=0)
    center = np.rint(nib.affines.apply_affine(np.linalg.inv(prediction_image.affine), center_world)).astype(int)
    arrays = {
        "RIGHT L5 SAP": sap_mask,
        "RIGHT L5 TP": tp_mask,
        "target-region endpoints": contract.sap_core | contract.tp_core,
        "uncertainty endpoints": contract.sap_uncertain | contract.tp_uncertain,
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    planes = ((0, center[0], "sagittal"), (1, center[1], "coronal"), (2, center[2], "axial"))
    colors = {"RIGHT L5 SAP": "#00b4d8", "RIGHT L5 TP": "#ff9f1c", "target-region endpoints": "#f72585", "uncertainty endpoints": "#ffd166"}
    for ax, (axis, index, title) in zip(axes, planes):
        base = np.zeros(np.take(sap_mask, index, axis=axis).shape)
        ax.imshow(base.T, cmap="gray", origin="lower")
        for name, mask in arrays.items():
            plane = np.take(mask, index, axis=axis).T
            rows, columns = np.where(plane)
            ax.scatter(columns, rows, s=8 if "target" in name else 3, c=colors[name], label=name)
        ax.set_title(f"{title} index {index}")
        ax.set_aspect("equal")
    axes[-1].legend(loc="upper left", fontsize=7)
    fig.suptitle("Evaluator-only RIGHT L5 SAP/TP evidence; not a nerve annotation")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_demo(
    path: Path,
    *,
    session_id: str,
    images: np.ndarray,
    frames: list[dict[str, Any]],
    session_rows: list[dict[str, Any]],
    coordinates: np.ndarray,
    contract: TargetContractGeometry,
    sap_mask: np.ndarray,
    tp_mask: np.ndarray,
) -> None:
    by_sequence = {int(row["frame_sequence"]): row for row in session_rows}
    l5_sequences = sorted(by_sequence)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 writer")
    trajectory = np.asarray([frames[index]["executed_relative_pose_index"][:2] for index in l5_sequences])
    for sequence in l5_sequences:
        row = by_sequence[sequence]
        frame = frames[sequence]
        image = images[sequence]
        projected = {
            "sap": sap_projection.project_mask(coordinates[sequence], sap_mask)[0],
            "tp": sap_projection.project_mask(coordinates[sequence], tp_mask)[0],
            "sap_core": sap_projection.project_mask(coordinates[sequence], contract.sap_core)[0],
            "tp_core": sap_projection.project_mask(coordinates[sequence], contract.tp_core)[0],
        }
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), dpi=100, constrained_layout=True)
        axes[0, 0].imshow(image, cmap="gray", origin="upper")
        for track in track_bright_interfaces(image):
            axes[0, 0].plot(track.columns, track.rows, linewidth=1.2, label=track.track_id)
        axes[0, 0].scatter([frame["proposal_column"]], [frame["proposal_row"]], c="#00ff88", s=35, label="L5 TP proposal")
        if row["image_column"] is not None:
            axes[0, 0].scatter([row["image_column"]], [row["image_row"]], c="#ff2d95", s=45, marker="x", label="image-only target hypothesis")
        axes[0, 0].set_title("Actor: clean ultrasound and cortical evidence")
        axes[0, 0].legend(fontsize=6, loc="upper left")
        axes[0, 0].set_xlim(0, 149); axes[0, 0].set_ylim(199, 0)

        axes[0, 1].axis("off")
        lines = [
            "Actor decision",
            f"frame: {session_id}:{sequence}",
            f"phase: {frame['phase'].split(':')[-1]}",
            f"TP probability: {float(frame['target_probability']):.3f}",
            f"image evidence: {float(row['image_evidence_score']):.3f}",
            f"candidate status: {row['image_status']}",
            f"final MBB target decision: ABSTAIN",
            "reason: target contract/visibility gate not passed",
        ]
        axes[0, 1].text(0.02, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=11)
        axes[0, 1].set_title("Target localization")

        axes[1, 0].plot(trajectory[:, 0], trajectory[:, 1], color="#888888", linewidth=1)
        offset = l5_sequences.index(sequence)
        axes[1, 0].scatter(trajectory[: offset + 1, 0], trajectory[: offset + 1, 1], c=np.arange(offset + 1), cmap="viridis", s=16)
        axes[1, 0].scatter([trajectory[offset, 0]], [trajectory[offset, 1]], c="#ff2d95", s=60)
        axes[1, 0].set_xlabel("executed surface x (index)")
        axes[1, 0].set_ylabel("executed surface z (index)")
        axes[1, 0].set_title("Historical executed Stage 2 trajectory")

        axes[1, 1].imshow(image, cmap="gray", origin="upper")
        for name, color in (("tp", "#ff9f1c"), ("sap", "#00b4d8"), ("tp_core", "#f72585"), ("sap_core", "#f72585")):
            rr, cc = np.where(projected[name])
            axes[1, 1].scatter(cc, rr, s=5 if "core" in name else 2, c=color, label=name)
        axes[1, 1].set_xlim(0, 149); axes[1, 1].set_ylim(199, 0)
        axes[1, 1].set_title(f"Evaluator only: {row['label']} ({row['label_reason']})")
        axes[1, 1].legend(fontsize=6, loc="upper left")
        fig.suptitle("Offline replay; no new closed-loop MBB target navigation")
        fig.canvas.draw()
        canvas = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(720, 1280, 4)[:, :, :3]
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        plt.close(fig)
    writer.release()


def write_reports(output: Path, result: dict[str, Any], protocol: dict[str, Any]) -> None:
    contract = result["target_contract"]
    labels = result["frame_label_counts"]
    summaries = result["method_summaries"]
    (output / "TARGET_CONTRACT.md").write_text(
        f"""# RIGHT L4 medial-branch target contract at RIGHT L5

This contract is limited to fixed anatomy `s0010`. It defines evaluator-only research geometry and does not annotate a nerve or intervention endpoint.

## Result

**`ANATOMICAL_TARGET_CONTRACT = PARTIAL`: the independent CT model provides provisional RIGHT L5 SAP and TP masks, but their closest surfaces are {contract['minimum_surface_separation_mm']:.3f} mm apart and neither the nerve nor the junction is semantically labelled.**

| question | result |
|---|---|
| A1 RIGHT L5 TP identifiable | yes as model TP plus historical TP surface-core evidence; not one verified whole-TP mask |
| A2 RIGHT L5 SAP identifiable | provisional model-generated proposal only |
| A3 junction directly identified | no; only the reproducible closest-apposition relationship is available |
| A4 nerve or validated target available | no; bony surrogate only |
| A5 physical evaluator available | yes; RAS millimetres to {contract['midpoint_count']} patient-specific midpoint samples |
| A6 meaningful localization experiment | exploratory evaluation only; not clinical or intervention validation |

## 1. Geometry — can the simulator define a bounded evaluator region

`l4_medial_branch_target.py:build_target_contract_geometry`, run by `run_s0010_l4_medial_branch_target.py`.

**Criterion.** The masks must be aligned, right-sided, nonempty and disjoint; uncertainty must be derived from the 1.5 mm CT grid; no generic nerve offset is allowed.

### 1.0 Quantities

| quantity | value | source |
|---|---:|---|
| RIGHT L5 SAP voxels | {contract['right_l5_sap_voxels']} | frozen V3 prediction |
| RIGHT L5 TP voxels | {contract['right_l5_tp_voxels']} | frozen V3 prediction |
| minimum surface separation | {contract['minimum_surface_separation_mm']:.6f} mm | physical RAS distance |
| core endpoint voxels, SAP / TP | {contract['sap_core_surface_voxels']} / {contract['tp_core_surface_voxels']} | minimum plus one CT voxel |
| uncertainty endpoint voxels, SAP / TP | {contract['sap_uncertainty_surface_voxels']} / {contract['tp_uncertainty_surface_voxels']} | minimum plus one CT voxel diagonal |

### 1.1 Method

- Retain the frozen V3 RIGHT L5 model SAP and TP masks inside source L5 and the RAS right half.
- Find the closest surface relationship in physical millimetres.
- Use one CT voxel for the core separation band and one CT voxel diagonal for uncertainty.
- Use midpoint samples only as an evaluator distance reference, never as actor input or a nerve coordinate.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| run | `{output.parent.name}` | §1 |
| prediction SHA-256 | `{contract['geometry_provenance']['prediction_sha256']}` | §1 |
| segmentation SHA-256 | `{contract['geometry_provenance']['segmentation_sha256']}` | §1 |
| TP revision SHA-256 | `{contract['geometry_provenance']['tp_revision_sha256']}` | §1 |

```bash
sbatch hpg/jobs/s0010_l4_medial_branch_target_v1.sbatch
```

Nothing was dropped.

### 1.3 Results

**Verdict: partial**, because the geometry is measurable but the SAP semantics, junction boundary and nerve course are not independently validated.

### 1.4 Reading it

- The midpoint set is not a needle endpoint.
- The 3.0 mm evaluator gate is a development-resolution gate, not a clinical tolerance.

### 1.5 Files

| file | contents |
|---|---|
| `target_contract.json` | exact masks counts, separations, hashes and verdict |
| `anatomy_target_contract.png` | evaluator-only geometry overview |
""",
        encoding="utf-8",
    )
    (output / "REPORT.md").write_text(
        f"""# s0010 RIGHT L4 medial-branch target localization at RIGHT L5

This CPU-only experiment evaluates a bony surrogate from historical Stage 2 observations. It does not run new robot control or an MBB intervention.

## Result

**`ULTRASOUND_TARGET_LOCALIZATION = {result['ultrasound_target_localization']}`: zero of {sum(labels.values())} search frames showed both target-region endpoint surfaces as first-visible bone, so closed-loop execution was not eligible.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | target contract | bounded, measurable, nonclinical surrogate | `{result['anatomical_target_contract']}` | partial |
| 2 | ultrasound visibility | at least {protocol['evaluation']['minimum_positive_frames_per_session']} positives in {protocol['evaluation']['minimum_qualifying_sessions']} sessions | 0 positives; {labels['PROXY_AMBIGUOUS']} ambiguous; {labels['PROXY_NEGATIVE']} negative | fail |
| 3 | image-only localization | target evidence with zero wrong and ambiguous accepts | {summaries['image_interface']['wrong_accepted_frame_count']} wrong; {summaries['image_interface']['ambiguous_accepted_frame_count']} ambiguous accepts | fail |
| 4 | robot execution | only after steps 1–3 pass | historical replay only | not run |

## 1. Ultrasound evidence — is the target region observable

`run_s0010_l4_medial_branch_target.py`, run by the sealed CPU Slurm job.

**Criterion.** Both core endpoint surfaces must be first-visible bone in at least {protocol['evaluation']['minimum_positive_frames_per_session']} distinct frames from each of {protocol['evaluation']['minimum_qualifying_sessions']} sessions. Projected-but-occluded SAP remains ambiguous.

### 1.0 Quantities

| quantity | value | source |
|---|---:|---|
| attempted frames | {sum(labels.values())} | four historical sessions |
| proxy-positive | {labels['PROXY_POSITIVE']} | both endpoints first visible |
| proxy-ambiguous | {labels['PROXY_AMBIGUOUS']} | projected, occluded or noncore evidence |
| proxy-negative | {labels['PROXY_NEGATIVE']} | TP-only or other bone without SAP in plane |

### 1.1 Method

- Freeze RIGHT L5 TP proposals from the historical actor.
- Detect cortical interfaces from clean ultrasound.
- Compare the fixed TP-relative control with image intensity and posterior-shadow evidence along the interface.
- Use CT geometry, raw labels and native coordinates only after the actor hypothesis is frozen.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| source run | job `52039283` | §1 |
| CPU run | `{output.parent.name}` | §1 |
| protocol SHA-256 | `{result['protocol_sha256']}` | §1 |
| sessions | `{', '.join(protocol['source_sessions'])}` | §1 |

```bash
SONOGYM_CONFIG_FILE="$PWD/.sonogym.env" sbatch hpg/jobs/s0010_l4_medial_branch_target_v1.sbatch
```

Nothing was dropped after the declared phase filter; fresh confirmations were excluded by protocol.

### 1.3 Results

| method | accepted | wrong accepted | ambiguous accepted | median distance |
|---|---:|---:|---:|---:|
| existing L5 TP proposal | {summaries['tp_only']['accepted_frame_count']} | {summaries['tp_only']['wrong_accepted_frame_count']} | {summaries['tp_only']['ambiguous_accepted_frame_count']} | {summaries['tp_only']['median_distance_to_target_region_mm']:.3f} mm |
| fixed TP-relative control | {summaries['fixed_tp_relative']['accepted_frame_count']} | {summaries['fixed_tp_relative']['wrong_accepted_frame_count']} | {summaries['fixed_tp_relative']['ambiguous_accepted_frame_count']} | {summaries['fixed_tp_relative']['median_distance_to_target_region_mm']:.3f} mm |
| image-interface evidence | {summaries['image_interface']['accepted_frame_count']} | {summaries['image_interface']['wrong_accepted_frame_count']} | {summaries['image_interface']['ambiguous_accepted_frame_count']} | {summaries['image_interface']['median_distance_to_target_region_mm']:.3f} mm |

**Verdict: fail**, because no positive frame exists and both candidate methods accept unsupported evidence.

### 1.4 Reading it

- The historical L5 TP workflow remains valid for TP view acquisition.
- The image-only extension does not establish the TP/SAP relationship.
- No GPU job or new closed-loop execution is justified by this result.

### 1.5 Files

| file | contents |
|---|---|
| `frame_results.csv` | per-frame labels, actor hypotheses and evaluator distances |
| `session_results.csv` | session-level counts |
| `demo.mp4` | actual historical observations with separated actor/evaluator panels |
""",
        encoding="utf-8",
    )


def make_package(output: Path, result: dict[str, Any], protocol_path: Path) -> Path:
    package = output / "s0010_right_l4_medial_branch_target_review"
    package.mkdir(parents=True, exist_ok=True)
    for name in (
        "result.json",
        "frame_results.csv",
        "session_results.csv",
        "target_contract.json",
        "TARGET_CONTRACT.md",
        "REPORT.md",
    ):
        shutil.copy2(output / name, package / name)
    shutil.copy2(protocol_path, package / protocol_path.name)
    shutil.copy2(protocol_path.with_suffix(".sha256"), package / protocol_path.with_suffix(".sha256").name)
    shutil.copy2(output / "figures/anatomy_target_contract.png", package / "anatomy_target_contract.png")
    shutil.copy2(output / "demo.mp4", package / "demo.mp4")
    (package / "START_HERE.md").write_text(
        "# START HERE\n\n"
        "This package tests a simulator-only RIGHT L4 medial-branch bony target at RIGHT L5.\n\n"
        "- The video is an offline replay of historical Stage 2 job 52039283, not a new closed-loop run.\n"
        "- The upper-left, upper-right, and lower-left panels use actor-visible ultrasound, TP predictions, and executed motion.\n"
        "- The lower-right panel is evaluator-only and uses provisional CT-model SAP/TP geometry.\n"
        "- RIGHT L5 TP navigation remains historically verified; the SAP/TP target region did not pass the visibility gate.\n"
        "- No nerve, needle, injection, or clinical target is claimed.\n",
        encoding="utf-8",
    )
    checksums = []
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksums.append(f"{sha256(path)}  {path.relative_to(package)}")
    (package / "checksums.sha256").write_text("\n".join(checksums) + "\n")
    zip_path = output / "s0010_right_l4_medial_branch_target_review.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, Path(package.name) / path.relative_to(package))
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP validation failed: {bad}")
    (zip_path.with_suffix(".zip.sha256")).write_text(f"{sha256(zip_path)}  {zip_path.name}\n")
    result["review_zip"] = str(zip_path)
    result["review_zip_sha256"] = sha256(zip_path)
    return zip_path


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    protocol, protocol_sha = load_protocol(args.protocol)
    for key, path in {
        "prediction_sha256": args.prediction,
        "segmentation_sha256": args.segmentation,
        "tp_revision_sha256": args.tp_revision,
    }.items():
        actual = sha256(path)
        if actual != str(protocol["inputs"][key]):
            raise ValueError(f"input hash mismatch for {path}: {actual}")
    output = args.output_dir
    (output / "figures").mkdir(parents=True, exist_ok=True)
    prediction_image, sap_mask, tp_mask, contract = load_target_geometry(
        args.prediction, args.segmentation
    )
    bone_ids = {
        int(source)
        for source, acoustic in yaml.safe_load(args.label_conversion.read_text()).items()
        if int(acoustic) == 13
    }
    target_contract = {
        "anatomy_id": "s0010",
        "target": "RIGHT L4 medial branch at RIGHT L5",
        "endpoint": "bony surrogate only",
        "coordinate_system": "NIfTI RAS millimetres",
        "right_l5_sap_voxels": int(np.count_nonzero(sap_mask)),
        "right_l5_tp_voxels": int(np.count_nonzero(tp_mask)),
        "sap_core_surface_voxels": int(np.count_nonzero(contract.sap_core)),
        "tp_core_surface_voxels": int(np.count_nonzero(contract.tp_core)),
        "sap_uncertainty_surface_voxels": int(np.count_nonzero(contract.sap_uncertain)),
        "tp_uncertainty_surface_voxels": int(np.count_nonzero(contract.tp_uncertain)),
        "minimum_surface_separation_mm": contract.minimum_surface_separation_mm,
        "core_separation_limit_mm": contract.core_separation_limit_mm,
        "uncertainty_separation_limit_mm": contract.uncertainty_separation_limit_mm,
        "midpoint_count": int(len(contract.midpoint_world_mm)),
        "evaluator_tolerance_mm": float(protocol["evaluation"]["target_distance_tolerance_mm"]),
        "geometry_provenance": {
            "prediction": str(args.prediction.resolve()),
            "prediction_sha256": sha256(args.prediction),
            "segmentation": str(args.segmentation.resolve()),
            "segmentation_sha256": sha256(args.segmentation),
            "tp_revision": str(args.tp_revision.resolve()),
            "tp_revision_sha256": sha256(args.tp_revision),
        },
        "semantic_status": "model-generated SAP/TP research proposal",
        "nerve_model_available": False,
        "clinical_target_validated": False,
        "verdict": "PARTIAL",
    }
    write_json(output / "target_contract.json", target_contract)
    save_anatomy_figure(
        output / "figures/anatomy_target_contract.png",
        prediction_image,
        sap_mask,
        tp_mask,
        contract,
    )

    rows: list[dict[str, Any]] = []
    session_cache: dict[str, tuple[np.ndarray, list[dict[str, Any]], np.ndarray]] = {}
    for session_id in protocol["source_sessions"]:
        session = args.run_root / "stage2" / session_id
        frames = read_jsonl(session / "public/frames.jsonl")
        images = np.load(session / "public/ultrasound.npz")["ultrasound"]
        coordinates = np.load(session / "privileged/pixel_to_volume_coordinates.npz")[
            "human_img_coords"
        ]
        raw_labels = np.load(session / "privileged/raw_labels.npz")["raw_labels"]
        if not (len(frames) == len(images) == len(coordinates) == len(raw_labels)):
            raise ValueError(f"session array mismatch: {session_id}")
        session_cache[session_id] = (images, frames, coordinates)
        for sequence, frame in enumerate(frames):
            if frame["target_name"] != TARGET_NAME:
                continue
            phase = str(frame["phase"]).split(":")[-1]
            if phase not in set(protocol["include_phases"]):
                continue
            projected = projected_masks(coordinates[sequence], contract, sap_mask, tp_mask)
            first = first_bone_mask(raw_labels[sequence], bone_ids)
            label, label_reason = target_evidence_label(projected, first)
            tp_only = TargetViewHypothesis(
                status="ACCEPT",
                row=float(frame["proposal_row"]),
                column=float(frame["proposal_column"]),
                evidence_score=float(frame["target_probability"]),
                proposal_track_distance_pixels=0.0,
                reason="EXISTING_L5_TP_PROPOSAL",
            )
            control = fixed_tp_relative_hypothesis(
                images[sequence],
                proposal_row=float(frame["proposal_row"]),
                proposal_column=float(frame["proposal_column"]),
                proposal_track_id=str(frame["proposal_track_id"]),
                column_offset_pixels=float(protocol["fixed_tp_relative_control"]["column_offset_pixels"]),
                maximum_proposal_track_distance_pixels=float(protocol["actor_method"]["maximum_proposal_track_distance_pixels"]),
            )
            image_hypothesis = image_target_hypothesis(
                images[sequence],
                proposal_row=float(frame["proposal_row"]),
                proposal_column=float(frame["proposal_column"]),
                proposal_track_id=str(frame["proposal_track_id"]),
                minimum_column_offset_pixels=int(protocol["actor_method"]["minimum_column_offset_pixels"]),
                maximum_column_offset_pixels=int(protocol["actor_method"]["maximum_column_offset_pixels"]),
                smoothing_width_pixels=int(protocol["actor_method"]["smoothing_width_pixels"]),
                minimum_evidence_score=float(protocol["actor_method"]["minimum_evidence_score"]),
                maximum_proposal_track_distance_pixels=float(protocol["actor_method"]["maximum_proposal_track_distance_pixels"]),
                minimum_image_margin_pixels=int(protocol["actor_method"]["minimum_image_margin_pixels"]),
            )
            record: dict[str, Any] = {
                "session_id": session_id,
                "split": "method_selection" if session_id in protocol["method_selection_sessions"] else "final_comparison_opened_development",
                "frame_sequence": sequence,
                "frame_sha256": frame["frame_sha256"],
                "phase": phase,
                "executed_relative_pose_index": json.dumps(frame["executed_relative_pose_index"]),
                "tp_probability": float(frame["target_probability"]),
                "label": label,
                "label_reason": label_reason,
                "sap_projected_pixels": int(np.count_nonzero(projected["sap"])),
                "tp_projected_pixels": int(np.count_nonzero(projected["tp"])),
                "sap_core_first_visible_pixels": int(np.count_nonzero(projected["sap_core"] & first)),
                "tp_core_first_visible_pixels": int(np.count_nonzero(projected["tp_core"] & first)),
            }
            for prefix, hypothesis in (("tp_only", tp_only), ("fixed", control), ("image", image_hypothesis)):
                evaluated = evaluate_hypothesis(
                    hypothesis,
                    label=label,
                    coordinates=coordinates[sequence],
                    affine=prediction_image.affine,
                    contract=contract,
                    tolerance_mm=float(protocol["evaluation"]["target_distance_tolerance_mm"]),
                )
                record.update({f"{prefix}_{key}": value for key, value in evaluated.items()})
            rows.append(record)

    labels = {name: sum(row["label"] == name for row in rows) for name in ("PROXY_POSITIVE", "PROXY_AMBIGUOUS", "PROXY_NEGATIVE", "TECHNICALLY_INVALID")}
    method_rows = [row for row in rows if row["split"] == "method_selection"]
    comparison_rows = [row for row in rows if row["split"] != "method_selection"]
    summaries = {
        "tp_only": summarize(rows, "tp_only"),
        "fixed_tp_relative": summarize(rows, "fixed"),
        "image_interface": summarize(rows, "image"),
    }
    session_rows = []
    for session_id in protocol["source_sessions"]:
        selected = [row for row in rows if row["session_id"] == session_id]
        session_rows.append(
            {
                "session_id": session_id,
                "split": selected[0]["split"],
                "frame_count": len(selected),
                "positive_count": sum(row["label"] == "PROXY_POSITIVE" for row in selected),
                "ambiguous_count": sum(row["label"] == "PROXY_AMBIGUOUS" for row in selected),
                "negative_count": sum(row["label"] == "PROXY_NEGATIVE" for row in selected),
                "image_accept_count": sum(row["image_status"] == "ACCEPT" for row in selected),
                "image_wrong_accept_count": sum(row["image_wrong_accept"] for row in selected),
                "image_ambiguous_accept_count": sum(row["image_ambiguous_accept"] for row in selected),
            }
        )
    minimum_positive = int(protocol["evaluation"]["minimum_positive_frames_per_session"])
    qualifying_sessions = sum(row["positive_count"] >= minimum_positive for row in session_rows)
    target_visibility_pass = qualifying_sessions >= int(protocol["evaluation"]["minimum_qualifying_sessions"])
    localization_pass = bool(
        target_visibility_pass
        and summaries["image_interface"]["wrong_accepted_frame_count"] <= int(protocol["evaluation"]["maximum_wrong_accepted_frames"])
        and summaries["image_interface"]["ambiguous_accepted_frame_count"] <= int(protocol["evaluation"]["maximum_ambiguous_accepted_frames"])
    )
    result = {
        "schema": "s0010-l4-medial-branch-target-result-v1",
        "source_commit": args.source_commit,
        "protocol_sha256": protocol_sha,
        "source_run": str(args.run_root.resolve()),
        "source_evidence_status": "historically_opened_development",
        "target_contract": target_contract,
        "frame_label_counts": labels,
        "method_selection_frame_count": len(method_rows),
        "final_comparison_frame_count": len(comparison_rows),
        "method_summaries": summaries,
        "qualifying_target_visible_sessions": qualifying_sessions,
        "target_visibility_gate_passed": target_visibility_pass,
        "anatomical_target_contract": "PARTIAL",
        "ultrasound_target_localization": "DEVELOPMENT_ONLY" if localization_pass else "FAIL",
        "robot_navigation": "HISTORICAL_REPLAY_ONLY",
        "full_mbb_intervention": "BLOCKED",
        "new_closed_loop_execution": False,
        "runtime_seconds": time.monotonic() - started,
    }
    write_csv(output / "frame_results.csv", rows)
    write_csv(output / "session_results.csv", session_rows)
    write_json(output / "result.json", result)
    write_reports(output, result, protocol)
    demo_session = str(protocol["demo_session"])
    render_demo(
        output / "demo.mp4",
        session_id=demo_session,
        images=session_cache[demo_session][0],
        frames=session_cache[demo_session][1],
        session_rows=[row for row in rows if row["session_id"] == demo_session],
        coordinates=session_cache[demo_session][2],
        contract=contract,
        sap_mask=sap_mask,
        tp_mask=tp_mask,
    )
    make_package(output, result, args.protocol)
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
