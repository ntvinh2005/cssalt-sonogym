#!/usr/bin/env python3
"""Package the clinician-scan Stage 2 report and reproducibility evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--perception-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, package: Path, relative: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target = package / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source: Path, package: Path, relative: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, package / relative)


def git_output(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    package = args.output_dir / "s0010_stage2_clinician_scan_review"
    package.mkdir(parents=True)

    repo_files = [
        "docs/research/sonogym_s0010_l4_medial_branch_target.md",
        "docs/notation.md",
        "configs/experiments/s0010_sonoanatomy_dataset_v2.yaml",
        "workflows/two_stage_navigation/prepare_s0010_clinician_scan.py",
        "workflows/two_stage_navigation/evaluate_s0010_clinician_scan.py",
        "workflows/two_stage_navigation/evaluate_s0010_clinician_scan_perception.py",
        "workflows/two_stage_navigation/build_s0010_clinician_scan_review.py",
        "hpg/jobs/s0010_clinician_scan_capture_v1.sbatch",
        "hpg/jobs/s0010_clinician_scan_evaluate_v1.sbatch",
        "hpg/jobs/s0010_clinician_scan_perception_v1.sbatch",
        "hpg/jobs/s0010_clinician_scan_review_v2.sbatch",
    ]
    for value in repo_files:
        copy_file(args.repo_root / value, package, Path("repository") / value)

    copy_file(
        args.capture_root / "context/s0010_clinician_scan_feasibility_v2.yaml",
        package,
        Path("capture/protocol.yaml"),
    )
    for session in sorted((args.capture_root / "sessions").iterdir()):
        if not session.is_dir():
            continue
        for name in ("summary.json", "verification_report.json", "actor_6dof_validation.json"):
            copy_file(session / name, package, Path("capture/sessions") / session.name / name)
        copy_file(
            session / "public/frames.jsonl",
            package,
            Path("capture/sessions") / session.name / "frames.jsonl",
        )

    copy_tree(args.motion_root / "evaluation", package, Path("motion_evaluation"))
    copy_file(args.dataset_root / "result.json", package, Path("dataset/result.json"))
    copy_file(args.dataset_root / "manifest.csv", package, Path("dataset/manifest.csv"))
    copy_tree(args.dataset_root / "figures", package, Path("dataset/figures"))
    for name in (
        "result.json",
        "l4_result.json",
        "l5_result.json",
        "l4_split_manifest.csv",
        "l5_split_manifest.csv",
        "l4_training_history.csv",
        "l5_training_history.csv",
    ):
        copy_file(args.training_root / name, package, Path("training") / name)
    copy_tree(args.perception_root / "evaluation", package, Path("perception_evaluation"))
    copy_tree(args.review_root / "review", package, Path("videos"))

    provenance = package / "provenance"
    provenance.mkdir()
    (provenance / "git-commit.txt").write_text(git_output(args.repo_root, "rev-parse", "HEAD"))
    (provenance / "git-status.txt").write_text(git_output(args.repo_root, "status", "--short"))
    (provenance / "uncommitted.patch").write_text(git_output(args.repo_root, "diff", "--binary"))
    job_roots = {
        "capture": args.capture_root,
        "motion": args.motion_root,
        "training": args.training_root.parent,
        "perception": args.perception_root,
        "review": args.review_root,
    }
    for label, root in job_roots.items():
        context = root / "context"
        if context.is_dir():
            copy_tree(context, package, Path("provenance/jobs") / label)

    files = sorted(path for path in package.rglob("*") if path.is_file())
    checksum_path = package / "SHA256SUMS"
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(package)}\n" for path in files)
    )
    archive = args.output_dir / "s0010_stage2_clinician_scan_review.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                handle.write(path, Path(package.name) / path.relative_to(package))
    manifest = {
        "schema": "s0010-stage2-clinician-scan-review-package-v1",
        "package_directory": str(package.resolve()),
        "archive": str(archive.resolve()),
        "archive_sha256": sha256(archive),
        "file_count": len([path for path in package.rglob("*") if path.is_file()]),
        "scientific_verdict": "STAGE2_COMPLETE=FAIL",
        "clinical_intervention_verdict": "FULL_MBB_INTERVENTION=BLOCKED",
    }
    (args.output_dir / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
