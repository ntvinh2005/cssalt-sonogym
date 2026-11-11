"""Dataset provenance and public/privileged annotation separation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import FORBIDDEN_ACTOR_FIELDS, assert_no_privileged_fields


JOIN_FIELDS = ("session_id", "frame_sequence", "frame_sha256")
VALID_SPLITS = frozenset({"train_dev", "calibration_dev", "untouched_audit"})


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def join_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    try:
        key = (str(row["session_id"]), int(row["frame_sequence"]), str(row["frame_sha256"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid frame join key: {error}") from error
    if not key[0] or key[1] < 0 or len(key[2]) != 64:
        raise ValueError(f"invalid frame join key {key!r}")
    return key


def validate_split_assignments(assignments: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Return session-to-split mapping after proving whole-session independence."""
    session_to_split: dict[str, str] = {}
    for assignment in assignments:
        session_id = str(assignment.get("session_id", ""))
        split = str(assignment.get("split", ""))
        if not session_id or split not in VALID_SPLITS:
            raise ValueError(f"invalid split assignment {assignment!r}")
        previous = session_to_split.setdefault(session_id, split)
        if previous != split:
            raise ValueError(f"session {session_id!r} appears in multiple splits")
    if not session_to_split:
        raise ValueError("no split assignments supplied")
    return session_to_split


def validate_public_rows(
    rows: Iterable[Mapping[str, Any]], session_to_split: Mapping[str, str]
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    sequence_by_session: dict[str, int] = {}
    for source in rows:
        row = dict(source)
        assert_no_privileged_fields(row)
        key = join_key(row)
        if key in seen:
            raise ValueError(f"duplicate public frame {key!r}")
        seen.add(key)
        if key[0] not in session_to_split:
            raise ValueError(f"unassigned public session {key[0]!r}")
        if row.get("split") != session_to_split[key[0]]:
            raise ValueError(f"incorrect split for session {key[0]!r}")
        previous = sequence_by_session.get(key[0], -1)
        if key[1] <= previous:
            raise ValueError(f"nonmonotonic public sequence in session {key[0]!r}")
        sequence_by_session[key[0]] = key[1]
        validated.append(row)
    return validated


def validate_privileged_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for source in rows:
        row = dict(source)
        key = join_key(row)
        if key in seen:
            raise ValueError(f"duplicate privileged frame {key!r}")
        seen.add(key)
        if not FORBIDDEN_ACTOR_FIELDS.intersection(row):
            raise ValueError("privileged row lacks an explicitly privileged field")
        validated.append(row)
    return validated


def validate_one_to_one_join(
    public_rows: Iterable[Mapping[str, Any]], privileged_rows: Iterable[Mapping[str, Any]]
) -> None:
    public_keys = Counter(join_key(row) for row in public_rows)
    privileged_keys = Counter(join_key(row) for row in privileged_rows)
    if any(count != 1 for count in public_keys.values()) or any(
        count != 1 for count in privileged_keys.values()
    ):
        raise ValueError("frame keys must be unique on both sides")
    missing_privileged = sorted(public_keys.keys() - privileged_keys.keys())
    orphan_privileged = sorted(privileged_keys.keys() - public_keys.keys())
    if missing_privileged or orphan_privileged:
        raise ValueError(
            f"public/privileged join mismatch; missing={missing_privileged[:3]}, "
            f"orphan={orphan_privileged[:3]}"
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_summary(
    public_rows: list[Mapping[str, Any]],
    privileged_rows: list[Mapping[str, Any]],
    protocol_sha256: str,
) -> dict[str, Any]:
    validate_one_to_one_join(public_rows, privileged_rows)
    sessions = sorted({str(row["session_id"]) for row in public_rows})
    split_counts = Counter(str(row["split"]) for row in public_rows)
    return {
        "schema": "sonogym-two-stage-dataset-summary-v1",
        "protocol_sha256": protocol_sha256,
        "session_count": len(sessions),
        "frame_count": len(public_rows),
        "split_frame_counts": dict(sorted(split_counts.items())),
        "public_manifest_sha256": canonical_json_sha256(public_rows),
        "privileged_manifest_sha256": canonical_json_sha256(privileged_rows),
        "one_to_one_join": True,
        "actor_privileged_leakage": 0,
    }
