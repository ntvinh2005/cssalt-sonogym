"""Actor-visible patch proposals and offline TP surface-core review targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stage1_features import PatchSpec, extract_patch_feature
from .stage1_interfaces import InterfaceTrackerConfig, track_bright_interfaces


BACKGROUND_CLASS = "BACKGROUND_OR_OTHER_STRUCTURE"
L4_CLASS = "L4_RIGHT_TP_SURFACE_CORE"
L5_CLASS = "L5_RIGHT_TP_SURFACE_CORE"
CLASS_IDS = {BACKGROUND_CLASS: 0, L4_CLASS: 1, L5_CLASS: 2}


@dataclass(frozen=True)
class PatchProposal:
    track_id: str
    row: float
    column: float
    feature: np.ndarray


def actor_visible_patch_proposals(
    image: np.ndarray,
    *,
    patch_spec: PatchSpec,
    column_stride_pixels: int,
    tracker_config: InterfaceTrackerConfig = InterfaceTrackerConfig(),
) -> list[PatchProposal]:
    """Generate patches from clean-image morphology without privileged coordinates."""
    array = np.asarray(image, dtype=np.float64)
    if column_stride_pixels <= 0:
        raise ValueError("column stride must be positive")
    half_width = patch_spec.width_pixels / 2.0
    columns = np.arange(
        int(np.ceil(half_width)),
        int(np.floor(array.shape[1] - half_width)) + 1,
        column_stride_pixels,
        dtype=np.float64,
    )
    proposals: list[PatchProposal] = []
    for track in track_bright_interfaces(array, tracker_config):
        for column in columns:
            if column < track.columns[0] or column > track.columns[-1]:
                continue
            row = track.row_at(float(column))
            feature = extract_patch_feature(array, row, float(column), patch_spec)
            if feature is not None:
                proposals.append(PatchProposal(track.track_id, row, float(column), feature))
    return proposals


def count_points_in_patch(
    proposal: PatchProposal,
    *,
    rows: np.ndarray,
    columns: np.ndarray,
    patch_spec: PatchSpec,
) -> int:
    column_start = int(round(proposal.column - patch_spec.width_pixels / 2.0))
    row_start = int(round(proposal.row - patch_spec.interface_offset_pixels))
    return int(
        np.count_nonzero(
            (rows >= row_start)
            & (rows < row_start + patch_spec.depth_pixels)
            & (columns >= column_start)
            & (columns < column_start + patch_spec.width_pixels)
        )
    )


def proposed_review_target(
    core_counts: dict[str, int], uncertain_counts: dict[str, int]
) -> tuple[str, int | None, str]:
    """Return a review target, never a semantic training label."""
    active = [name for name, count in core_counts.items() if count > 0]
    uncertain = any(count > 0 for count in uncertain_counts.values())
    if len(active) > 1 or uncertain:
        return "AMBIGUOUS", None, "core overlap conflict or uncertain-boundary overlap"
    if active == [L4_CLASS]:
        return "REVIEW_POSITIVE", CLASS_IDS[L4_CLASS], "L4 surface-core geometry in patch"
    if active == [L5_CLASS]:
        return "REVIEW_POSITIVE", CLASS_IDS[L5_CLASS], "L5 surface-core geometry in patch"
    return "REVIEW_HARD_NEGATIVE", CLASS_IDS[BACKGROUND_CLASS], "no TP core or uncertain geometry in patch"
