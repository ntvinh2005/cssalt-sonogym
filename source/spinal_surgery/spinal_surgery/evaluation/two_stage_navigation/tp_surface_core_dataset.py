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


def projected_pixel_diagnostics(
    proposal: PatchProposal,
    *,
    core_rows: np.ndarray,
    core_columns: np.ndarray,
    uncertain_rows: np.ndarray,
    uncertain_columns: np.ndarray,
    patch_spec: PatchSpec,
) -> dict[str, float | int | bool]:
    """Measure deduplicated projected support inside one proposed source patch."""

    column_start = int(round(proposal.column - patch_spec.width_pixels / 2.0))
    row_start = int(round(proposal.row - patch_spec.interface_offset_pixels))

    def inside_unique(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        rows_array = np.asarray(rows, dtype=np.int64)
        columns_array = np.asarray(columns, dtype=np.int64)
        inside = (
            (rows_array >= row_start)
            & (rows_array < row_start + patch_spec.depth_pixels)
            & (columns_array >= column_start)
            & (columns_array < column_start + patch_spec.width_pixels)
        )
        if not np.any(inside):
            return np.empty((0, 2), dtype=np.int64)
        return np.unique(np.column_stack((rows_array[inside], columns_array[inside])), axis=0)

    core = inside_unique(core_rows, core_columns)
    uncertain = inside_unique(uncertain_rows, uncertain_columns)
    size = patch_spec.feature_size_pixels
    if core.size:
        relative_rows = core[:, 0] - row_start
        relative_columns = core[:, 1] - column_start
        cells = np.unique(
            np.column_stack(
                (
                    np.minimum(relative_rows * size // patch_spec.depth_pixels, size - 1),
                    np.minimum(relative_columns * size // patch_spec.width_pixels, size - 1),
                )
            ),
            axis=0,
        )
        border = (
            (cells[:, 0] == 0)
            | (cells[:, 0] == size - 1)
            | (cells[:, 1] == 0)
            | (cells[:, 1] == size - 1)
        )
        interface_cell_row = patch_spec.interface_offset_pixels * size / patch_spec.depth_pixels
        centre_cell_column = size / 2.0
        scaled_distance = np.hypot(
            cells[:, 0] - interface_cell_row,
            cells[:, 1] - centre_cell_column,
        )
        near_interface = np.abs(cells[:, 0] - interface_cell_row) <= 1.0
        cell_count = int(len(cells))
        interior_count = int(np.count_nonzero(~border))
        nearest = float(np.min(scaled_distance))
        near_interface_count = int(np.count_nonzero(near_interface))
        border_fraction = float(np.mean(border))
    else:
        cell_count = 0
        interior_count = 0
        nearest = float("inf")
        near_interface_count = 0
        border_fraction = 1.0
    adequate = bool(
        cell_count >= 2
        and interior_count >= 1
        and nearest <= 3.0
        and near_interface_count >= 1
        and len(uncertain) == 0
    )
    return {
        "unique_core_pixels": int(len(core)),
        "unique_uncertain_pixels": int(len(uncertain)),
        "core_feature_cells": cell_count,
        "interior_core_feature_cells": interior_count,
        "near_interface_core_feature_cells": near_interface_count,
        "nearest_core_distance_feature_cells": nearest,
        "border_core_feature_cell_fraction": border_fraction,
        "adequate_projected_support": adequate,
        "patch_row_start": row_start,
        "patch_column_start": column_start,
    }


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
