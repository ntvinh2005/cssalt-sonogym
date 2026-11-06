#!/usr/bin/env python3
"""Resumably download, verify, and safely extract the pinned SonoGym assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


REVISION = "b37b080a8673f856266a2306724e48d5e034521a"
FILES = {
    "assets.tar.gz": (754262884, "0d5840da3af2eb97e1ca9ebc7c2a969d32b24eef3f3107bc0c9f7db41e4a8f77"),
    "models.tar.gz": (754887112, "545113c94b641eda199c6edee1e3586f8cd3a844f0b600470a828fec6210b14f"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(path: Path, expected_size: int) -> None:
    if path.exists() and path.stat().st_size == expected_size:
        return
    url = (
        "https://huggingface.co/datasets/yunkao/SonoGym_assets_models/resolve/"
        f"{REVISION}/{path.name}?download=true"
    )
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for resumable downloads")
    subprocess.run(
        [curl, "-L", "--fail", "--retry", "5", "--retry-delay", "5", "-C", "-", "-o", str(path), url],
        check=True,
    )
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"wrong size for {path}: {path.stat().st_size} != {expected_size}")


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"archive links are not allowed: {member.name}")
        tar.extractall(destination)


def locate(root: Path, kind: str) -> Path:
    if kind == "assets":
        candidates = [root / "assets" / "data", root / "data"]
    else:
        candidates = [root / "models", root / "models" / "models"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    matches = [path for path in root.rglob(kind) if path.is_dir()]
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(f"could not uniquely locate extracted {kind} directory in {root}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    runtime = Path(args.runtime_dir).resolve()
    downloads = runtime / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, dict[str, object]] = {}
    for name, (size, expected_hash) in FILES.items():
        path = downloads / name
        download(path, size)
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"SHA-256 mismatch for {path}: {actual_hash}")
        resolved[name] = {"bytes": size, "sha256": actual_hash}

    if not args.download_only:
        for name, kind, destination in (
            ("assets.tar.gz", "assets", runtime / "assets" / "data"),
            ("models.tar.gz", "models", runtime / "models"),
        ):
            if destination.is_dir() and any(destination.iterdir()):
                continue
            staging = runtime / f".extract-{kind}"
            if staging.exists():
                raise RuntimeError(f"stale extraction directory requires inspection: {staging}")
            safe_extract(downloads / name, staging)
            source = locate(staging, kind)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            shutil.rmtree(staging)

    manifest = {
        "dataset": "yunkao/SonoGym_assets_models",
        "revision": REVISION,
        "files": resolved,
        "assets_data_dir": str(runtime / "assets" / "data"),
        "models_dir": str(runtime / "models"),
        "extracted": not args.download_only,
    }
    (runtime / "assets-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
