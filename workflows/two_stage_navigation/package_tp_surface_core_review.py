#!/usr/bin/env python3
"""Package a stratified, checksum-verifiable TP surface-core review handoff."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


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
        raise RuntimeError("review decision template would be empty")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    return parser.parse_args()


def copy_one(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"missing package input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    dataset = args.dataset_dir.resolve()
    preflight = args.preflight_dir.resolve()
    geometry_root = args.geometry_root.resolve()
    output_zip = args.output_zip.resolve()
    staging = output_zip.with_suffix("")
    if staging.exists() or output_zip.exists():
        raise RuntimeError("refusing to overwrite an existing review package")
    staging.mkdir(parents=True)

    for name in (
        "candidate_sample_manifest.csv",
        "dataset_summary.json",
        "label_schema.json",
        "session_split_manifest.csv",
    ):
        copy_one(dataset / name, staging / name)
    copy_one(preflight / "preflight_result.json", staging / "preflight_result.json")
    review_source = dataset / "review"
    if not review_source.is_dir():
        raise RuntimeError("review overlays are missing")
    shutil.copytree(review_source, staging / "verified_overlays")

    smoke = run_dir / "sessions" / "s0010tpscv6_smoke"
    copy_one(smoke / "verification_report.json", staging / "smoke_verification_report.json")
    copy_one(smoke / "public" / "ultrasound.npz", staging / "smoke_ultrasound.npz")
    copy_one(smoke / "privileged" / "raw_labels.npz", staging / "smoke_raw_labels.npz")
    copy_one(
        smoke / "privileged" / "pixel_to_volume_coordinates.npz",
        staging / "smoke_pixel_to_volume_coordinates.npz",
    )

    geometry_files: list[Path] = []
    for landmark in ("L4_RIGHT_transverse_process", "L5_RIGHT_transverse_process"):
        for suffix in ("_revision.npz", "_before_after.png"):
            matches = list(geometry_root.rglob(f"{landmark}{suffix}"))
            if len(matches) != 1:
                raise RuntimeError(f"expected one geometry asset for {landmark}{suffix}")
            geometry_files.append(matches[0])
    manifests = list(geometry_root.rglob("correction_manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError("expected one correction manifest")
    geometry_files.append(manifests[0])
    for source in geometry_files:
        copy_one(source, staging / "geometry" / source.name)

    rows = read_csv(dataset / "candidate_sample_manifest.csv")
    decisions = []
    for row in rows:
        if not row.get("review_asset"):
            continue
        decisions.append(
            {
                "sample_id": row["sample_id"],
                "session_id": row["session_id"],
                "split": row["split"],
                "frame_sha256": row["frame_sha256"],
                "class_name": row["candidate_id"],
                "review_target": row["review_target"],
                "review_asset": f"verified_overlays/{Path(row['review_asset']).name}",
                "geometry_version_sha256": row["geometry_version_sha256"],
                "anatomical_boundary_decision": "PENDING",
                "acoustic_visibility": "NOT_REVIEWED",
                "semantic_identity": "NOT_REVIEWED",
                "reviewer_identity": "",
                "reviewer_qualification": "",
                "notes": "",
            }
        )
    write_csv(staging / "review_decisions_template.csv", decisions)

    summary = json.loads((dataset / "dataset_summary.json").read_text(encoding="utf-8"))
    preflight_report = json.loads(
        (preflight / "preflight_result.json").read_text(encoding="utf-8")
    )
    readme = f"""# s0010 RIGHT L4/L5 TP surface-core review

This simulator-only package contains synchronized ultrasound correspondence for the proposed
distal TP surface cores. It does not annotate whole TP, SAP, a medial branch nerve, an MBB target,
or injection readiness.

- Capture job: 51811811
- Dataset status: `{summary['status']}`
- Candidate rows: {summary['rows']}
- Verified geometric intersections: {summary['verified_intersections']}
- Stratified review assets: {len(decisions)}
- Training-eligible rows before review: {summary['training_eligible_rows']}
- No-fit status: `{preflight_report['status']}`; `fit_called={str(preflight_report['fit_called']).lower()}`

## Required decisions

For every row in `review_decisions_template.csv`, inspect the clean ultrasound and the separate
reviewer-only projection. Record acoustic visibility and semantic identity. For L4/L5 positive
candidates, a qualified anatomical reviewer must also approve or reject the exact geometry checksum.
Background candidates require semantic rejection as TP but never anatomical approval as TP.
Ambiguous or occluded examples remain excluded. Do not convert unknown values into negatives.

The source-image coverage deficits recorded in `dataset_summary.json` are acquisition deficits;
review cannot manufacture missing independent L4 images. Model fitting must remain blocked until
the launcher reports `TRAINING_READY_FOR_APPROVAL`.
"""
    (staging / "README.md").write_text(readme, encoding="utf-8")

    checksum_path = staging / "checksums.sha256"
    package_files = sorted(
        path for path in staging.rglob("*") if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(staging).as_posix()}\n" for path in package_files),
        encoding="utf-8",
    )
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(path for path in staging.rglob("*") if path.is_file()):
            archive.write(path, path.relative_to(staging).as_posix())

    result = {
        "schema": "sonogym-tp-surface-core-review-package-v1",
        "zip": str(output_zip),
        "zip_sha256": sha256(output_zip),
        "file_count": sum(path.is_file() for path in staging.rglob("*")),
        "review_asset_count": len(decisions),
        "dataset_status": summary["status"],
        "preflight_status": preflight_report["status"],
        "fit_called": preflight_report["fit_called"],
    }
    result_path = output_zip.with_suffix(".result.json")
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
