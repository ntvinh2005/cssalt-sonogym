#!/usr/bin/env python3
"""Build a small, deterministic readiness manifest for MBB-oriented Stage 2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED = (
    ("L4", "superior_articular_process", "L3_MEDIAL_BRANCH"),
    ("L4", "transverse_process", "L3_MEDIAL_BRANCH"),
    ("L5", "superior_articular_process", "L4_MEDIAL_BRANCH"),
    ("L5", "transverse_process", "L4_MEDIAL_BRANCH"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_for(
    annotations: list[dict[str, Any]], level: str, name: str, side: str
) -> dict[str, Any]:
    rows = [
        row
        for row in annotations
        if row["anatomy_level"] == level
        and row["landmark_name"] == name
        and row["laterality"] == side
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one {level} {side} {name} annotation, found {len(rows)}")
    return rows[0]


def build_manifest(source: Path, repository_root: Path) -> dict[str, Any]:
    document = json.loads(source.read_text(encoding="utf-8"))
    if document["patient_id"] != "s0010":
        raise RuntimeError("this audit is restricted to patient s0010")
    fine = document["annotations"]["group_B_future_MBB_bony_geometry"]
    matrix: list[dict[str, Any]] = []
    for side in ("LEFT", "RIGHT"):
        for level, landmark, branch in REQUIRED:
            row = record_for(fine, level, landmark, side)
            matrix.append(
                {
                    "facet_level": "L4-L5",
                    "site_branch": branch,
                    "vertebral_level": level,
                    "laterality": side,
                    "landmark": landmark,
                    "readiness": row["status"],
                    "confidence": row["confidence"],
                    "positional_uncertainty_mm": row["positional_uncertainty_mm"],
                    "generation_method": row["generation_method"],
                    "evidence": row["supporting_visualizations"],
                    "blocker": "; ".join(row["limitations"]),
                    "minimum_next_requirement": (
                        "independent anatomical delineation and review"
                        if landmark == "superior_articular_process"
                        else (
                            "independent anatomical review of process boundaries "
                            "before supervised use"
                        )
                    ),
                }
            )

    return {
        "schema": "sonogym-stage2-mbb-landmark-readiness-v1",
        "source_annotation_generation_commit": document["provenance"][
            "cssalt_sonogym_commit"
        ],
        "patient_id": "s0010",
        "facet_level": "L4-L5",
        "laterality_configuration": "REQUIRED_EXPLICIT_LEFT_OR_RIGHT",
        "source_annotation": {
            "path": source.relative_to(repository_root).as_posix(),
            "sha256": sha256(source),
            "status": document["status"],
        },
        "landmark_readiness": matrix,
        "nerve_reference_readiness": [
            {
                "branch": branch,
                "readiness": "MISSING",
                "blocker": (
                    "no medial-branch nerve segmentation or independently reviewed "
                    "patient-specific nerve course"
                ),
            }
            for branch in ("L3_MEDIAL_BRANCH", "L4_MEDIAL_BRANCH")
        ],
        "ultrasound_observability": {
            "readiness": "UNRESOLVED",
            "evidence": (
                "existing captures establish raw-label occupancy, not SAP/TP semantic "
                "discriminability"
            ),
            "renderer_limitation": (
                "raw vertebral labels 25-31 map to the common acoustic bone class 13"
            ),
        },
        "target_region": {
            "readiness": "UNRESOLVED",
            "kind": "bony-landmark-derived anatomical surrogate region",
            "localized": False,
            "blocker": (
                "required SAP annotations and branch-specific patient geometry are not "
                "validated; nerve segmentation is absent"
            ),
        },
        "readiness_decision": "MBB_STAGE2_BLOCKED_GROUND_TRUTH",
        "scientific_scope": {
            "patient_count": 1,
            "clinical_validation": False,
            "detector_trained": False,
            "target_region_implemented": False,
            "gpu_work": "NOT_RUN",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("experiments/unity_two_stage_transfer/s0010_anatomical_annotations_v2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/unity_two_stage_transfer/stage2_mbb_landmark_readiness_v1.json"),
    )
    args = parser.parse_args()
    root = args.repository_root.resolve()
    source = args.source if args.source.is_absolute() else root / args.source
    output = args.output if args.output.is_absolute() else root / args.output
    result = build_manifest(source, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "decision": result["readiness_decision"]}))


if __name__ == "__main__":
    main()
