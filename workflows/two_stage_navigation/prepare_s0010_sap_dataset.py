#!/usr/bin/env python3
"""Build fail-closed level-conditioned SAP patch data from synchronized captures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml

import audit_s0010_sap_correspondence as correspondence
import correct_s0010_sap_observability_v1 as correction
import run_s0010_sap_observability as observability

EVALUATION_ROOT = (
    Path(__file__).resolve().parents[2]
    / "source/spinal_surgery/spinal_surgery/evaluation"
)
sys.path.insert(0, str(EVALUATION_ROOT))
from two_stage_navigation.stage1_features import (  # noqa: E402
    UNITY_REFERENCE_PATCH_SPEC,
    extract_interface_patch,
    normalized_patch_intensity,
    patch_feature,
)
from two_stage_navigation.tp_surface_core_dataset import (  # noqa: E402
    PatchProposal,
    actor_visible_patch_proposals,
    count_points_in_patch,
)

LEVELS = ("L4", "L5")
POSITIVE = "SAP_RELATED_CORTICAL_INTERFACE"
NEGATIVE = "OTHER_OR_BACKGROUND"
AMBIGUOUS = "BOUNDARY_OR_IDENTITY_AMBIGUOUS"
TECHNICALLY_INVALID = "TECHNICALLY_INVALID"
PROPOSAL_STRIDE_PIXELS = 12
MINIMUM_LOCAL_SURFACE_PIXELS = 5
MINIMUM_UNIQUE = {
    "train": {POSITIVE: 30, NEGATIVE: 30},
    "calibration": {POSITIVE: 10, NEGATIVE: 10},
    "evaluation": {POSITIVE: 10, NEGATIVE: 10},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def intended_level(
    protocol: dict[str, Any], frame_sequence: int, session_id: str | None = None
) -> str:
    sampling = protocol["sampling"]
    order = sampling.get("waypoint_level_order")
    if order is None and session_id is not None:
        order = sampling.get("waypoint_level_order_by_session", {}).get(session_id)
    if order is None:
        raise ValueError("protocol lacks a frozen waypoint level order")
    for item in order:
        if int(item["start_index"]) <= frame_sequence < int(item["stop_index_exclusive"]):
            return str(item["level"])
    raise ValueError(f"frame {frame_sequence} has no frozen target level")


def session_directory(session: dict[str, Any], run_root: Path) -> Path:
    source = session.get("source_session_path")
    if source is None:
        return run_root / "sessions" / session["session_id"]
    path = Path(str(source))
    if not path.is_absolute():
        raise ValueError("source_session_path must be absolute")
    return path


def validate_session_records(
    session: dict[str, Any],
    session_dir: Path,
    public: list[dict[str, Any]],
    frame_count: int,
) -> None:
    expected_count = session.get("frame_count")
    if expected_count is not None and int(expected_count) != frame_count:
        raise ValueError(f"frozen frame count changed: {session_dir.name}")
    expected_verification = session.get("verification_sha256")
    verification_path = session_dir / "verification_report.json"
    if expected_verification is not None and sha256(verification_path) != str(
        expected_verification
    ):
        raise ValueError(f"verification report hash changed: {session_dir.name}")
    if len(public) != frame_count:
        raise ValueError(f"public manifest length changed: {session_dir.name}")
    for index, row in enumerate(public):
        if row.get("session_id") != session["session_id"]:
            raise ValueError(f"public session identity mismatch: {session_dir.name}")
        if int(row.get("frame_sequence", -1)) != index:
            raise ValueError(f"public frame sequence mismatch: {session_dir.name}")


def patch_bounds(proposal: PatchProposal) -> tuple[int, int, int, int]:
    spec = UNITY_REFERENCE_PATCH_SPEC
    column_start = int(round(proposal.column - spec.width_pixels / 2.0))
    row_start = int(round(proposal.row - spec.interface_offset_pixels))
    return row_start, row_start + spec.depth_pixels, column_start, column_start + spec.width_pixels


def mask_count(proposal: PatchProposal, mask: np.ndarray) -> int:
    rows, columns = np.where(mask)
    return count_points_in_patch(
        proposal,
        rows=rows,
        columns=columns,
        patch_spec=UNITY_REFERENCE_PATCH_SPEC,
    )


def alignment_for_track(
    image: np.ndarray,
    proposal: PatchProposal,
    surface: np.ndarray,
    tracks_by_id: dict[str, Any],
) -> correspondence.Alignment:
    track = tracks_by_id[proposal.track_id]
    return correspondence.track_alignment(image, track, surface)


def best_alignment_from_tracks(
    image: np.ndarray, tracks: list[Any], surface: np.ndarray
) -> correspondence.Alignment:
    if not tracks:
        return correspondence.best_alignment(image, surface)
    rows = [correspondence.track_alignment(image, track, surface) for track in tracks]
    return max(
        rows,
        key=lambda row: (
            row.passes,
            row.longest_aligned_run_columns,
            row.aligned_fraction,
            -row.median_distance_pixels,
        ),
    )


def correspondence_target_keys(
    intended: dict[tuple[str, int], str],
    surfaces: dict[tuple[str, int, str], np.ndarray],
    level: str,
    minimum_surface_columns: int,
    require_projected_surface: bool,
) -> tuple[list[tuple[str, int]], int]:
    intended_keys = [key for key, value in intended.items() if value == level]
    if not require_projected_surface:
        return intended_keys, 0
    retained = [
        key
        for key in intended_keys
        if np.count_nonzero(np.any(surfaces[(*key, level)], axis=0))
        >= minimum_surface_columns
    ]
    return retained, len(intended_keys) - len(retained)


def choose_candidate(rows: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    eligible = [row for row in rows if row["provisional_label"] == label]
    if not eligible:
        return None
    if label == POSITIVE:
        return max(
            eligible,
            key=lambda row: (
                int(row["surface_pixels_in_patch"]),
                int(row["longest_aligned_run_columns"]),
                float(row["aligned_fraction"]),
                float(row["feature_gradient_p90"]),
                -int(row["proposal_index"]),
            ),
        )
    return max(
        eligible,
        key=lambda row: (
            int(row["other_level_surface_pixels_in_patch"]),
            float(row["feature_gradient_p90"]),
            float(row["feature_intensity_std"]),
            -int(row["proposal_index"]),
        ),
    )


def choose_image_candidate(
    rows: list[dict[str, Any]],
    label: str,
    require_all_clear_for_negative: bool,
) -> dict[str, Any] | None:
    if (
        label == NEGATIVE
        and require_all_clear_for_negative
        and any(row["provisional_label"] != NEGATIVE for row in rows)
    ):
        return None
    return choose_candidate(rows, label)


def require_clear_negative_image(
    protocol: dict[str, Any], command_line_required: bool
) -> bool:
    return bool(
        command_line_required
        or protocol["label_gate"].get(
            "require_all_proposals_clear_for_negative_image", False
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--label-conversion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("development", "evaluation"), default="development"
    )
    parser.add_argument(
        "--require-all-proposals-clear-for-negative-image", action="store_true"
    )
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if sha256(args.protocol) != args.protocol.with_suffix(".sha256").read_text().split()[0]:
        raise ValueError("protocol seal mismatch")
    prediction_image = nib.load(args.prediction)
    segmentation_image = nib.load(args.segmentation)
    if not np.array_equal(prediction_image.affine, segmentation_image.affine):
        raise ValueError("prediction and segmentation affines differ")
    masks = observability.pseudo_sap_masks(
        np.asarray(prediction_image.dataobj),
        np.asarray(segmentation_image.dataobj),
        prediction_image.affine,
    )
    bone_ids = correction.bone_source_ids(args.label_conversion)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    proposal_rows: list[dict[str, Any]] = []
    feature_rows: list[np.ndarray] = []
    proposal_features: list[np.ndarray] = []
    raw_patches: list[np.ndarray] = []
    session_surfaces: dict[tuple[str, int, str], np.ndarray] = {}
    frame_images: dict[tuple[str, int], np.ndarray] = {}
    frame_tracks: dict[tuple[str, int], list[Any]] = {}
    intended: dict[tuple[str, int], str] = {}
    incomplete_sessions: list[dict[str, str]] = []
    require_clear_negative = require_clear_negative_image(
        protocol, args.require_all_proposals_clear_for_negative_image
    )

    phase_splits = (
        {"train", "calibration"} if args.phase == "development" else {"evaluation"}
    )
    for session in protocol["sessions"]:
        if session["split"] not in phase_splits:
            continue
        session_dir = session_directory(session, args.run_root)
        required_paths = (
            session_dir / "verification_report.json",
            session_dir / "public/ultrasound.npz",
            session_dir / "public/frames.jsonl",
            session_dir / "privileged/pixel_to_volume_coordinates.npz",
            session_dir / "privileged/raw_labels.npz",
        )
        if not session_dir.is_dir() or any(not path.exists() for path in required_paths):
            summary_path = session_dir / "summary.json"
            summary = (
                json.loads(summary_path.read_text()) if summary_path.exists() else {}
            )
            incomplete_sessions.append(
                {
                    "session_id": session["session_id"],
                    "split": session["split"],
                    "reason": str(summary.get("error", "capture artifacts missing")),
                }
            )
            continue
        verification = json.loads((session_dir / "verification_report.json").read_text())
        if verification.get("verdict") != "PROJECTION_PARITY_PASS":
            raise ValueError(f"exact-pose replay failed for {session_dir.name}")
        images = np.load(session_dir / "public/ultrasound.npz")["ultrasound"]
        coordinates = np.load(
            session_dir / "privileged/pixel_to_volume_coordinates.npz"
        )["human_img_coords"]
        raw_labels = np.load(session_dir / "privileged/raw_labels.npz")["raw_labels"]
        public = read_jsonl(session_dir / "public/frames.jsonl")
        if not (len(images) == len(coordinates) == len(raw_labels) == len(public)):
            raise ValueError(f"misaligned capture arrays in {session_dir.name}")
        validate_session_records(session, session_dir, public, len(images))
        for frame_sequence, image in enumerate(images):
            frame_key = (session_dir.name, frame_sequence)
            frame_images[frame_key] = image
            tracks = correspondence.track_bright_interfaces(image)
            frame_tracks[frame_key] = tracks
            tracks_by_id = {track.track_id: track for track in tracks}
            intended[frame_key] = intended_level(
                protocol, frame_sequence, session_dir.name
            )
            projections: dict[str, dict[str, np.ndarray]] = {}
            for level in LEVELS:
                diagnostics = observability.projection_diagnostics(
                    coordinates[frame_sequence], masks[level]
                )
                surface = correction.first_bone_surface(
                    raw_labels=raw_labels[frame_sequence],
                    corrected_projection=diagnostics["projected"],
                    bone_ids=bone_ids,
                )["visible_pseudo_surface"]
                session_surfaces[(*frame_key, level)] = surface
                projections[level] = {
                    "surface": surface,
                    "pseudo": diagnostics["projected"]["pseudo"],
                    "uncertainty": diagnostics["projected"]["uncertainty"],
                }

            proposals = actor_visible_patch_proposals(
                image,
                patch_spec=UNITY_REFERENCE_PATCH_SPEC,
                column_stride_pixels=PROPOSAL_STRIDE_PIXELS,
            )
            for level in LEVELS:
                other_level = "L5" if level == "L4" else "L4"
                level_rows: list[dict[str, Any]] = []
                for proposal_index, proposal in enumerate(proposals):
                    raw_patch = extract_interface_patch(
                        image, proposal.row, proposal.column, UNITY_REFERENCE_PATCH_SPEC
                    )
                    if raw_patch is None:
                        raise AssertionError("production proposal returned an invalid patch")
                    feature = patch_feature(raw_patch, UNITY_REFERENCE_PATCH_SPEC)
                    if not np.array_equal(feature, proposal.feature):
                        raise AssertionError("production feature path mismatch")
                    alignment = alignment_for_track(
                        image,
                        proposal,
                        projections[level]["surface"],
                        tracks_by_id,
                    )
                    surface_count = mask_count(proposal, projections[level]["surface"])
                    pseudo_count = mask_count(proposal, projections[level]["pseudo"])
                    uncertainty_count = mask_count(
                        proposal, projections[level]["uncertainty"]
                    )
                    other_surface_count = mask_count(
                        proposal, projections[other_level]["surface"]
                    )
                    if (
                        intended[frame_key] == level
                        and alignment.passes
                        and surface_count >= MINIMUM_LOCAL_SURFACE_PIXELS
                    ):
                        provisional_label = POSITIVE
                        negative_type = ""
                    elif (
                        intended[frame_key] != level
                        and pseudo_count == 0
                        and uncertainty_count == 0
                    ):
                        provisional_label = NEGATIVE
                        negative_type = (
                            "COMPETING_LEVEL_SAP"
                            if other_surface_count >= MINIMUM_LOCAL_SURFACE_PIXELS
                            else "OTHER_BONE_OR_BACKGROUND"
                        )
                    else:
                        provisional_label = AMBIGUOUS
                        negative_type = ""
                    row_start, row_end, column_start, column_end = patch_bounds(proposal)
                    row = {
                        "sample_id": f"{session_dir.name}:{frame_sequence}:{level}:{proposal.track_id}:{proposal_index}",
                        "session_id": session_dir.name,
                        "split": session["split"],
                        "trajectory_group": f"{session_dir.name}:segment_{frame_sequence // 2:02d}",
                        "frame_sequence": frame_sequence,
                        "acquisition_id": public[frame_sequence]["acquisition_id"],
                        "frame_sha256": public[frame_sequence]["frame_sha256"],
                        "image_archive": str((session_dir / "public/ultrasound.npz").resolve()),
                        "image_archive_index": frame_sequence,
                        "intended_capture_level": intended[frame_key],
                        "conditioned_level": level,
                        "proposal_index": proposal_index,
                        "proposal_feature_index": len(proposal_features),
                        "track_id": proposal.track_id,
                        "patch_row": float(proposal.row),
                        "patch_column": float(proposal.column),
                        "patch_row_start": row_start,
                        "patch_row_end": row_end,
                        "patch_column_start": column_start,
                        "patch_column_end": column_end,
                        "surface_pixels_in_patch": surface_count,
                        "pseudo_pixels_in_patch": pseudo_count,
                        "uncertainty_pixels_in_patch": uncertainty_count,
                        "other_level_surface_pixels_in_patch": other_surface_count,
                        **asdict(alignment),
                        "feature_intensity_std": float(feature[-5]),
                        "feature_gradient_p90": float(feature[-1]),
                        "provisional_label": provisional_label,
                        "negative_type": negative_type,
                        "selected_for_dataset": False,
                        "training_eligible": False,
                        "feature_index": -1,
                        "raw_patch_sha256": array_sha256(raw_patch.astype("<f4")),
                    }
                    proposal_features.append(feature)
                    level_rows.append(row)
                selected = choose_image_candidate(
                    level_rows,
                    POSITIVE if intended[frame_key] == level else NEGATIVE,
                    require_clear_negative,
                )
                if selected is not None:
                    selected["selected_for_dataset"] = True
                    selected["training_eligible"] = True
                    selected["feature_index"] = len(feature_rows)
                    proposal = proposals[int(selected["proposal_index"])]
                    patch = extract_interface_patch(
                        image, proposal.row, proposal.column, UNITY_REFERENCE_PATCH_SPEC
                    )
                    assert patch is not None
                    feature_rows.append(patch_feature(patch, UNITY_REFERENCE_PATCH_SPEC))
                    raw_patches.append(patch.astype(np.float32))
                proposal_rows.extend(level_rows)

    selected_rows = [row for row in proposal_rows if row["selected_for_dataset"]]
    write_csv(output / "proposal_manifest.csv", proposal_rows)
    write_csv(output / "sample_manifest.csv", selected_rows)
    np.savez_compressed(
        output / "dataset_arrays.npz",
        features=np.asarray(feature_rows, dtype=np.float64),
        proposal_features=np.asarray(proposal_features, dtype=np.float64),
        raw_patches=np.asarray(raw_patches, dtype=np.float32),
    )

    gate = protocol["label_gate"]
    correspondence_summary: dict[str, Any] = {}
    correspondence_rows: list[dict[str, Any]] = []
    for level in LEVELS:
        target_keys, excluded_without_surface = correspondence_target_keys(
            intended,
            session_surfaces,
            level,
            int(gate["minimum_surface_columns"]),
            bool(gate.get("require_projected_surface_for_matched_control", False)),
        )
        matched = [
            best_alignment_from_tracks(
                frame_images[key], frame_tracks[key], session_surfaces[(*key, level)]
            )
            for key in target_keys
        ]
        mismatch: list[correspondence.Alignment] = []
        ranking_wins = 0
        for key, matched_alignment in zip(target_keys, matched):
            correspondence_rows.append(
                {
                    "session_id": key[0],
                    "frame_sequence": key[1],
                    "level": level,
                    "comparison": "MATCHED",
                    "projection_source": f"{key[0]}:{key[1]}:{level}",
                    **asdict(matched_alignment),
                }
            )
            other_level = "L5" if level == "L4" else "L4"
            other_alignment = best_alignment_from_tracks(
                frame_images[key],
                frame_tracks[key],
                session_surfaces[(*key, other_level)],
            )
            controls = [other_alignment]
            correspondence_rows.append(
                {
                    "session_id": key[0],
                    "frame_sequence": key[1],
                    "level": level,
                    "comparison": "OTHER_LEVEL",
                    "projection_source": f"{key[0]}:{key[1]}:{other_level}",
                    **asdict(other_alignment),
                }
            )
            distinct = [
                other
                for other in target_keys
                if other[0] != key[0]
            ][:5]
            distinct_alignments = [
                best_alignment_from_tracks(
                    frame_images[key], frame_tracks[key], session_surfaces[(*other, level)]
                )
                for other in distinct
            ]
            controls.extend(distinct_alignments)
            for other, alignment in zip(distinct, distinct_alignments):
                correspondence_rows.append(
                    {
                        "session_id": key[0],
                        "frame_sequence": key[1],
                        "level": level,
                        "comparison": "DISTINCT_POSE",
                        "projection_source": f"{other[0]}:{other[1]}:{level}",
                        **asdict(alignment),
                    }
                )
            mismatch.extend(controls)
            ranking_wins += int(
                all(
                    matched_alignment.longest_aligned_run_columns
                    > control.longest_aligned_run_columns
                    for control in controls
                )
            )
        matched_rate = float(np.mean([row.passes for row in matched])) if matched else 0.0
        mismatch_rate = float(np.mean([row.passes for row in mismatch])) if mismatch else 1.0
        ranking_rate = ranking_wins / len(matched) if matched else 0.0
        correspondence_summary[level] = {
            "matched_pass": int(sum(row.passes for row in matched)),
            "matched_total": len(matched),
            "matched_acceptance_rate": matched_rate,
            "mismatched_pass": int(sum(row.passes for row in mismatch)),
            "mismatched_total": len(mismatch),
            "mismatched_false_alignment_rate": mismatch_rate,
            "ranking_wins": ranking_wins,
            "ranking_total": len(matched),
            "matched_without_projected_surface_excluded": excluded_without_surface,
            "correct_projection_ranking_rate": ranking_rate,
            "label_validation_pass": bool(
                matched_rate >= float(gate["minimum_matched_acceptance_rate"])
                and mismatch_rate <= float(gate["maximum_mismatched_false_alignment_rate"])
                and ranking_rate >= float(gate["minimum_correct_projection_ranking_rate"])
            ),
        }
    write_csv(output / "correspondence_controls.csv", correspondence_rows)

    counts: dict[str, dict[str, dict[str, int]]] = {}
    for level in LEVELS:
        counts[level] = {}
        for split in sorted({row["split"] for row in selected_rows}):
            counts[level][split] = {}
            for label in (POSITIVE, NEGATIVE):
                images = {
                    row["frame_sha256"]
                    for row in selected_rows
                    if row["conditioned_level"] == level
                    and row["split"] == split
                    and row["provisional_label"] == label
                    and row["training_eligible"]
                }
                counts[level][split][label] = len(images)

    readiness: dict[str, Any] = {}
    for level in LEVELS:
        deficits: list[str] = []
        required_splits = (
            ("train", "calibration") if args.phase == "development" else ("evaluation",)
        )
        for split in required_splits:
            requirements = MINIMUM_UNIQUE[split]
            if split not in counts[level]:
                deficits.append(f"missing split {split}")
                continue
            for label, minimum in requirements.items():
                actual = counts[level][split].get(label, 0)
                if actual < minimum:
                    deficits.append(f"{split}:{label}={actual}<{minimum}")
        if not correspondence_summary[level]["label_validation_pass"]:
            deficits.append("matched/mismatched correspondence gate failed")
        readiness[level] = {
            "annotation_status": (
                "SIMULATOR_PROXY_SAP_LABEL_READY" if not deficits else "SIMULATOR_PROXY_SAP_LABEL_NOT_READY"
            ),
            "dataset_status": "READY" if not deficits else "BLOCKED",
            "deficits": deficits,
        }

    report = {
        "schema": "s0010-sap-level-conditioned-dataset-v1",
        "phase": args.phase,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(args.protocol),
        "prediction_sha256": sha256(args.prediction),
        "segmentation_sha256": sha256(args.segmentation),
        "preprocessing": {
            "patch_width": UNITY_REFERENCE_PATCH_SPEC.width_pixels,
            "patch_depth": UNITY_REFERENCE_PATCH_SPEC.depth_pixels,
            "interface_offset": UNITY_REFERENCE_PATCH_SPEC.interface_offset_pixels,
            "feature_width": UNITY_REFERENCE_PATCH_SPEC.feature_width,
            "proposal_stride_pixels": PROPOSAL_STRIDE_PIXELS,
        },
        "counts": counts,
        "correspondence": correspondence_summary,
        "readiness": readiness,
        "incomplete_sessions": incomplete_sessions,
        "actor_feature_privileged_fields": [],
        "require_all_proposals_clear_for_negative_image": require_clear_negative,
    }
    (output / "readiness_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if all(item["dataset_status"] == "READY" for item in readiness.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
