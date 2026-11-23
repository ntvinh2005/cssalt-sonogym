#!/usr/bin/env python3
"""Package a completed fixed-s0010 Stage 2 run and its review evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any
import zipfile


PACKAGE_NAME = "s0010_mbb_stage2_review"
ARCHIVE_NAME = f"{PACKAGE_NAME}.zip"
FULL_VIDEO_NAME = "s0010_mbb_stage2_actual_execution_full.mp4"
BLINDED_VIDEO_NAME = "s0010_mbb_stage2_actual_execution_blinded.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination)


def verify_seal(path: Path, seal: Path | None = None) -> None:
    seal = seal or path.with_name(f"{path.name}.sha256")
    if not seal.is_file():
        raise FileNotFoundError(seal)
    fields = seal.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name or fields[0] != sha256(path):
        raise ValueError(f"invalid checksum seal: {seal}")


def find_protocol(context: Path) -> Path:
    protocols = sorted(context.glob("s0010_mbb_single_anatomy_poc_protocol_v*.yaml"))
    if len(protocols) != 1:
        raise ValueError(f"expected one Stage 2 protocol, found {len(protocols)}")
    verify_seal(protocols[0], protocols[0].with_suffix(".sha256"))
    return protocols[0]


def find_review_videos(review_root: Path) -> tuple[Path, Path]:
    full = review_root / FULL_VIDEO_NAME
    blinded = review_root / BLINDED_VIDEO_NAME
    manifest = read_json(review_root / "manifest.json")
    if manifest.get("schema") != "sonogym-s0010-mbb-stage2-review-v1":
        raise ValueError("unexpected review manifest schema")
    if not full.is_file() or not blinded.is_file():
        raise FileNotFoundError("review MP4 is missing")
    if full.stat().st_size == 0 or blinded.stat().st_size == 0:
        raise ValueError("review MP4 is empty")
    if manifest.get("full_video", {}).get("sha256") != sha256(full):
        raise ValueError("full review MP4 hash differs from manifest")
    if manifest.get("blinded_video", {}).get("sha256") != sha256(blinded):
        raise ValueError("blinded review MP4 hash differs from manifest")
    return full, blinded


def derive_verdicts(report: dict[str, Any]) -> dict[str, str]:
    stage2 = report.get("stage2", {})
    stage2_verdict = (
        "STAGE2_COMPLETE=PASS"
        if bool(stage2.get("stage2_complete", False))
        else "STAGE2_COMPLETE=FAIL"
    )
    intervention = report.get("mbb_intervention", {})
    if bool(intervention.get("executed", False)):
        raise ValueError("full MBB intervention must remain unexecuted")
    return {
        "stage2": stage2_verdict,
        "full_mbb_intervention": "FULL_MBB_INTERVENTION=BLOCKED",
    }


def write_reproducible_zip(directory: Path, destination: Path) -> None:
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            info = zipfile.ZipInfo(str(Path(directory.name) / path.relative_to(directory)))
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def copy_run_evidence(run_root: Path, package: Path) -> list[str]:
    context = run_root / "context"
    find_protocol(context)
    copy_tree(context, package / "run/context")
    copy_tree(run_root / "results", package / "run/results")
    stage1_results = run_root / "stage1/results"
    verify_seal(stage1_results / "actor_results.json", stage1_results / "actor_results.sha256")
    verify_seal(
        stage1_results / "evaluation_report.json",
        stage1_results / "evaluation_report.sha256",
    )
    for name in (
        "actor_results.json",
        "actor_results.sha256",
        "evaluation_report.json",
        "evaluation_report.sha256",
    ):
        copy_file(
            run_root / "stage1/results" / name,
            package / "run/stage1/results" / name,
        )

    sessions = sorted(path for path in (run_root / "stage2").iterdir() if path.is_dir())
    if not sessions:
        raise ValueError("run has no Stage 2 sessions")
    for session in sessions:
        for result_name in ("actor_result.json", "evaluation_report.json"):
            result = session / "results" / result_name
            verify_seal(result)
            copy_file(result, package / "run/stage2" / session.name / "results" / result_name)
            copy_file(
                result.with_name(f"{result.name}.sha256"),
                package / "run/stage2" / session.name / "results" / f"{result.name}.sha256",
            )
        for relative in (
            Path("summary.json"),
            Path("application-exit-codes.env"),
            Path("public/frames.jsonl"),
            Path("public/ultrasound.npz"),
            Path("video/robot_probe_patient.mp4"),
            Path("video/ultrasound_search.mp4"),
        ):
            copy_file(session / relative, package / "run/stage2" / session.name / relative)
        copy_tree(
            session / "visual_evidence",
            package / "run/stage2" / session.name / "visual_evidence",
        )
    return [session.name for session in sessions]


def copy_model_evidence(dataset_root: Path, training_root: Path, package: Path) -> None:
    dataset = read_json(dataset_root / "result.json")
    training = read_json(training_root / "result.json")
    if training.get("dataset_sha256") != dataset.get("dataset_sha256"):
        raise ValueError("training and dataset hashes differ")
    for name in ("result.json", "manifest.csv"):
        copy_file(dataset_root / name, package / "dataset" / name)
    for name in (
        "result.json",
        "l4_result.json",
        "l5_result.json",
        "l4_split_manifest.csv",
        "l5_split_manifest.csv",
        "l4_training_history.csv",
        "l5_training_history.csv",
        "s0010_l4_sonoanatomy_unet_v1.pt",
        "s0010_l5_sonoanatomy_unet_v1.pt",
    ):
        copy_file(training_root / name, package / "training" / name)


def build_package(
    *,
    run_root: Path,
    review_root: Path,
    dataset_root: Path,
    training_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)
    package = output_dir / PACKAGE_NAME
    package.mkdir()

    report = read_json(run_root / "results/end_to_end_report.json")
    verdicts = derive_verdicts(report)
    sessions = copy_run_evidence(run_root, package)
    expected_sessions = int(report.get("session_count", 0))
    if expected_sessions != len(sessions):
        raise ValueError("aggregate report and Stage 2 session counts differ")

    full_video, blinded_video = find_review_videos(review_root)
    copy_tree(review_root, package / "review")
    copy_model_evidence(dataset_root, training_root, package)
    manifest = {
        "schema": "s0010-mbb-stage2-review-package-v1",
        "protocol": find_protocol(run_root / "context").name,
        "sessions": sessions,
        "review_videos": [full_video.name, blinded_video.name],
        "verdicts": verdicts,
    }
    (package / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    files = sorted(path for path in package.rglob("*") if path.is_file())
    (package / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(package)}\n" for path in files),
        encoding="utf-8",
    )
    archive = output_dir / ARCHIVE_NAME
    write_reproducible_zip(package, archive)
    outer = {
        **manifest,
        "archive": archive.name,
        "archive_sha256": sha256(archive),
        "file_count": sum(1 for path in package.rglob("*") if path.is_file()),
    }
    (output_dir / "package_manifest.json").write_text(
        json.dumps(outer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outer


def main() -> int:
    args = parse_args()
    result = build_package(
        run_root=args.run_root.resolve(),
        review_root=args.review_root.resolve(),
        dataset_root=args.dataset_root.resolve(),
        training_root=args.training_root.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
