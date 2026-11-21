#!/usr/bin/env python3
"""Render calibration behavior for the frozen multi-view ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from two_stage_navigation.stage1_interfaces import track_bright_interfaces

import run_s0010_stage2_multiview_evidence as multiview
import run_s0010_stage2_multiview_ranker as ranker_workflow
import run_s0010_ultrasound_representation_experiment as representation


def figure_array(figure: plt.Figure) -> np.ndarray:
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()


def plot_interfaces(axis: plt.Axes, image: np.ndarray) -> None:
    axis.imshow(image, cmap="gray", vmin=float(np.min(image)), vmax=float(np.max(image)))
    colors = ("#00e5ff", "#ffca28", "#ab47bc", "#66bb6a")
    for track, color in zip(track_bright_interfaces(image), colors):
        axis.plot(track.columns, track.rows, color=color, linewidth=1.5)
    axis.set_axis_off()


def render_frame(
    *,
    session: multiview.SessionData,
    index: int,
    scores: np.ndarray,
    support_indices: list[list[int]],
    selected_sequence: int | None,
) -> plt.Figure:
    support = support_indices[index]
    peer = (
        max(support, key=lambda value: session.correspondences[index, value])
        if support
        else None
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), dpi=100)
    axes[0, 0].imshow(session.images[index], cmap="gray")
    axes[0, 0].set_title("Clean ultrasound")
    axes[0, 0].set_axis_off()

    plot_interfaces(axes[0, 1], session.images[index])
    axes[0, 1].set_title(
        "Actor-visible cortical interfaces\n"
        f"base={session.base_scores[index]:.3f}; rank={scores[index]:.3f}"
    )

    if peer is None:
        axes[1, 0].text(
            0.5,
            0.5,
            "No distinct corresponding view",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set_axis_off()
    else:
        plot_interfaces(axes[1, 0], session.images[peer])
        axes[1, 0].set_title(
            "Strongest supporting view\n"
            f"frame={session.frame_sequences[peer]}; "
            f"correspondence={session.correspondences[index, peer]:.3f}"
        )

    axis = axes[1, 1]
    pose = session.poses
    axis.plot(pose[:, 0], pose[:, 1], color="#b0bec5", linewidth=1.0)
    axis.scatter(
        pose[:, 0],
        pose[:, 1],
        c=session.base_scores,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=22,
        label="observed views",
    )
    axis.scatter(
        pose[index, 0], pose[index, 1], marker="*", s=180, c="#ef5350", label="current"
    )
    if support:
        axis.scatter(
            pose[support, 0],
            pose[support, 1],
            facecolors="none",
            edgecolors="#29b6f6",
            s=90,
            linewidths=1.5,
            label="support",
        )
    axis.set_xlabel("cumulative relative x (mm)")
    axis.set_ylabel("cumulative relative z (mm)")
    axis.set_title("Executed-motion history")
    axis.legend(loc="best", fontsize=7)
    axis.set_aspect("equal", adjustable="datalim")

    sequence = int(session.frame_sequences[index])
    selected = selected_sequence is not None and sequence == selected_sequence
    evaluator = session.statuses[index].lower().replace("_", " ")
    figure.suptitle(
        f"{session.session_id}; frame={sequence}; support={len(support)}; "
        f"selected={selected}\nEvaluator-only label: {evaluator}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return figure


def select_demo_sessions(records: list[dict[str, Any]]) -> list[str]:
    predicates = (
        lambda row: row["accepted_positive"],
        lambda row: row["wrong_accepted_positive_session"],
        lambda row: row["positive_present"] and not row["accepted"],
    )
    selected: list[str] = []
    for predicate in predicates:
        match = next(
            (row["session_id"] for row in records if predicate(row) and row["session_id"] not in selected),
            None,
        )
        if match is not None:
            selected.append(str(match))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dataset", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--level", choices=representation.LEVELS, default="L5")
    parser.add_argument("--fps", type=float, default=3.0)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = joblib.load(
        args.checkpoint_dir / f"s0010_{args.level.lower()}_multiview_ranker.joblib"
    )
    samples = representation.read_csv(args.development_dataset / "sample_manifest.csv")
    proposals = representation.read_csv(args.development_dataset / "proposal_manifest.csv")
    sessions = multiview.build_sessions(
        sample_rows=samples,
        proposal_rows=proposals,
        level=args.level,
        split="calibration",
        bundle=checkpoint["base_model"],
        recorded=representation.RecordedImages(),
    )
    selection = checkpoint["selection"]
    summary, records, frames = ranker_workflow.evaluate_ranker(
        sessions,
        checkpoint["ranker"],
        checkpoint["evidence_config"],
        threshold=float(selection["threshold"]),
        margin_threshold=float(selection["margin_threshold"]),
    )
    multiview.write_csv(output / "calibration_sessions.csv", records)
    multiview.write_csv(output / "calibration_frames.csv", frames)
    selected_ids = select_demo_sessions(records)
    if not selected_ids:
        raise ValueError("no calibration session is available for the demo")
    record_by_session = {row["session_id"]: row for row in records}
    writer = imageio.get_writer(
        output / f"s0010_{args.level.lower()}_multiview_correspondence.mp4",
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    )
    rendered = 0
    try:
        for session in sessions:
            if session.session_id not in selected_ids:
                continue
            scores, _, support_indices = ranker_workflow.ranker_session_scores(
                session, checkpoint["ranker"], checkpoint["evidence_config"]
            )
            selected_sequence = record_by_session[session.session_id][
                "selected_frame_sequence"
            ]
            for index in range(len(session.images)):
                figure = render_frame(
                    session=session,
                    index=index,
                    scores=scores,
                    support_indices=support_indices,
                    selected_sequence=selected_sequence,
                )
                writer.append_data(figure_array(figure))
                plt.close(figure)
                rendered += 1
    finally:
        writer.close()
    result = {
        "schema": "s0010-multiview-ranker-demo-v1",
        "level": args.level,
        "split": "calibration",
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "selected_sessions": selected_ids,
        "rendered_frames": rendered,
        "selection": selection,
        "calibration_summary": summary,
        "evaluation_opened_by_demo": False,
        "actor_panels_use_privileged_geometry": False,
        "evaluator_label_panel_is_privileged": True,
    }
    (output / "demo_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
