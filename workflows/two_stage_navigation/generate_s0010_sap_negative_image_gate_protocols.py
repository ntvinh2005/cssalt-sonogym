#!/usr/bin/env python3
"""Clone SAP development protocols with the complete-negative-image gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProtocolClone:
    source_name: str
    output_name: str
    protocol_id: str


PROTOCOL_CLONES = (
    ProtocolClone(
        "s0010_sap_validation_development_protocol_v1.yaml",
        "s0010_sap_validation_development_protocol_v2.yaml",
        "sonogym-s0010-sap-validation-development-v2",
    ),
    ProtocolClone(
        "s0010_sap_validation_followup_protocol_v1.yaml",
        "s0010_sap_validation_followup_protocol_v2.yaml",
        "sonogym-s0010-sap-validation-followup-v2",
    ),
    ProtocolClone(
        "s0010_sap_existing_capture_development_protocol_v2.yaml",
        "s0010_sap_existing_capture_development_protocol_v3.yaml",
        "sonogym-s0010-sap-validation-development-v3-existing-captures",
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clone_protocol(
    source: dict[str, Any],
    *,
    source_path: Path,
    protocol_id: str,
    implementation_revision: str,
) -> dict[str, Any]:
    document = deepcopy(source)
    prior_id = str(document["protocol_id"])
    document["protocol_id"] = protocol_id
    document["source"] = {
        **document["source"],
        "prior_protocol_id": prior_id,
        "prior_protocol_path": source_path.name,
        "prior_protocol_sha256": file_sha256(source_path),
        "negative_image_gate_implementation_revision": implementation_revision,
    }
    document["label_gate"][
        "require_all_proposals_clear_for_negative_image"
    ] = True
    return document


def write_protocol(path: Path, document: dict[str, Any]) -> str:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    digest = file_sha256(path)
    path.with_suffix(".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    args = parser.parse_args()
    outputs = []
    for spec in PROTOCOL_CLONES:
        source_path = args.protocol_dir / spec.source_name
        output_path = args.protocol_dir / spec.output_name
        source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        document = clone_protocol(
            source,
            source_path=source_path,
            protocol_id=spec.protocol_id,
            implementation_revision=args.implementation_revision,
        )
        digest = write_protocol(output_path, document)
        outputs.append(
            {
                "path": str(output_path),
                "prior_sha256": document["source"]["prior_protocol_sha256"],
                "sha256": digest,
            }
        )
    print(json.dumps({"protocols": outputs}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
