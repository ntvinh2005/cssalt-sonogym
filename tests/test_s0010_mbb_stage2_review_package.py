#!/usr/bin/env python3
"""CPU checks for the fixed-s0010 Stage 2 review archive."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "workflows/two_stage_navigation/package_s0010_mbb_stage2_review.py"
)
SPEC = importlib.util.spec_from_file_location("package_s0010_mbb_stage2_review", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def seal(path: Path, seal_path: Path | None = None) -> None:
    target = seal_path or path.with_name(f"{path.name}.sha256")
    target.write_text(f"{MODULE.sha256(path)}  {path.name}\n", encoding="utf-8")


def make_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    run = root / "run"
    review = root / "review"
    dataset = root / "dataset"
    training = root / "training"

    protocol = run / "context/s0010_mbb_single_anatomy_poc_protocol_v6.yaml"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("protocol_id: test\n", encoding="utf-8")
    seal(protocol, protocol.with_suffix(".sha256"))
    (run / "context/git-commit.txt").write_text("abc123\n", encoding="utf-8")
    write_json(
        run / "results/end_to_end_report.json",
        {
            "session_count": 1,
            "stage2": {"stage2_complete": True},
            "mbb_intervention": {"executed": False},
        },
    )

    stage1 = run / "stage1/results"
    actor = stage1 / "actor_results.json"
    evaluator = stage1 / "evaluation_report.json"
    write_json(actor, {"decision": "PASS"})
    write_json(evaluator, {"decision": "PASS"})
    seal(actor, stage1 / "actor_results.sha256")
    seal(evaluator, stage1 / "evaluation_report.sha256")

    session = run / "stage2/session_00"
    write_json(session / "summary.json", {"status": "COMPLETED"})
    (session / "application-exit-codes.env").write_text("actor=0\n", encoding="utf-8")
    (session / "public").mkdir(parents=True)
    (session / "public/frames.jsonl").write_text("{}\n", encoding="utf-8")
    (session / "public/ultrasound.npz").write_bytes(b"ultrasound")
    (session / "video").mkdir()
    (session / "video/robot_probe_patient.mp4").write_bytes(b"robot")
    (session / "video/ultrasound_search.mp4").write_bytes(b"ultrasound-video")
    (session / "visual_evidence").mkdir()
    (session / "visual_evidence/failure.png").write_bytes(b"failure")
    for name in ("actor_result.json", "evaluation_report.json"):
        path = session / "results" / name
        write_json(path, {"status": "sealed"})
        seal(path)

    review.mkdir()
    full = review / MODULE.FULL_VIDEO_NAME
    blinded = review / MODULE.BLINDED_VIDEO_NAME
    full.write_bytes(b"full-video")
    blinded.write_bytes(b"blinded-video")
    write_json(
        review / "manifest.json",
        {
            "schema": "sonogym-s0010-mbb-stage2-review-v1",
            "full_video": {"sha256": MODULE.sha256(full)},
            "blinded_video": {"sha256": MODULE.sha256(blinded)},
        },
    )

    write_json(dataset / "result.json", {"dataset_sha256": "dataset-hash"})
    (dataset / "manifest.csv").write_text("frame\n0\n", encoding="utf-8")
    write_json(training / "result.json", {"dataset_sha256": "dataset-hash"})
    for name in ("l4_result.json", "l5_result.json"):
        write_json(training / name, {"status": "TRAINED_OPENED_DEVELOPMENT"})
    for name in (
        "l4_split_manifest.csv",
        "l5_split_manifest.csv",
        "l4_training_history.csv",
        "l5_training_history.csv",
    ):
        (training / name).write_text("value\n1\n", encoding="utf-8")
    (training / "s0010_l4_sonoanatomy_unet_v1.pt").write_bytes(b"l4-model")
    (training / "s0010_l5_sonoanatomy_unet_v1.pt").write_bytes(b"l5-model")
    return run, review, dataset, training


class MbbStage2ReviewPackageTests(unittest.TestCase):
    def test_package_is_sealed_and_zip_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, review, dataset, training = make_inputs(root)
            first = root / "first"
            second = root / "second"
            result = MODULE.build_package(
                run_root=run,
                review_root=review,
                dataset_root=dataset,
                training_root=training,
                output_dir=first,
            )
            MODULE.build_package(
                run_root=run,
                review_root=review,
                dataset_root=dataset,
                training_root=training,
                output_dir=second,
            )
            self.assertEqual(result["verdicts"]["stage2"], "STAGE2_COMPLETE=PASS")
            self.assertEqual(
                result["verdicts"]["full_mbb_intervention"],
                "FULL_MBB_INTERVENTION=BLOCKED",
            )
            self.assertEqual(
                MODULE.sha256(first / MODULE.ARCHIVE_NAME),
                MODULE.sha256(second / MODULE.ARCHIVE_NAME),
            )
            package = first / MODULE.PACKAGE_NAME
            checksum_text = (package / "SHA256SUMS").read_text(encoding="utf-8")
            self.assertIn("run/stage2/session_00/results/actor_result.json", checksum_text)
            self.assertIn("visual_evidence/failure.png", checksum_text)
            with zipfile.ZipFile(first / MODULE.ARCHIVE_NAME) as archive:
                self.assertTrue(
                    all(item.date_time == (1980, 1, 1, 0, 0, 0) for item in archive.infolist())
                )

    def test_invalid_actor_seal_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, review, dataset, training = make_inputs(root)
            actor = run / "stage2/session_00/results/actor_result.json"
            actor.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum seal"):
                MODULE.build_package(
                    run_root=run,
                    review_root=review,
                    dataset_root=dataset,
                    training_root=training,
                    output_dir=root / "output",
                )

    def test_executed_intervention_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must remain unexecuted"):
            MODULE.derive_verdicts(
                {
                    "stage2": {"stage2_complete": True},
                    "mbb_intervention": {"executed": True},
                }
            )


if __name__ == "__main__":
    unittest.main()
