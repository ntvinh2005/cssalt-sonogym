"""Actor evidence and privileged evaluation for the RIGHT L5 bony surrogate."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .stage1_interfaces import InterfaceTrack, track_bright_interfaces
from .ultrasound_surface_features import normalized_image


@dataclass(frozen=True)
class TargetContractGeometry:
    sap_surface: np.ndarray
    tp_surface: np.ndarray
    sap_core: np.ndarray
    tp_core: np.ndarray
    sap_uncertain: np.ndarray
    tp_uncertain: np.ndarray
    midpoint_world_mm: np.ndarray
    minimum_surface_separation_mm: float
    core_separation_limit_mm: float
    uncertainty_separation_limit_mm: float


@dataclass(frozen=True)
class TargetViewHypothesis:
    status: str
    row: float | None
    column: float | None
    evidence_score: float
    proposal_track_distance_pixels: float | None
    reason: str


def target_evidence_label(
    projected: dict[str, np.ndarray], first_bone: np.ndarray
) -> tuple[str, str]:
    """Keep projected-but-occluded or boundary evidence out of hard negatives."""

    required = {
        "sap",
        "tp",
        "sap_core",
        "tp_core",
        "sap_uncertain",
        "tp_uncertain",
    }
    if set(projected) != required:
        raise ValueError("projected masks do not match the target contract")
    first = np.asarray(first_bone, dtype=bool)
    if any(np.asarray(mask).shape != first.shape for mask in projected.values()):
        raise ValueError("projected masks and first-bone mask differ in shape")
    visible = {name: np.asarray(mask, dtype=bool) & first for name, mask in projected.items()}
    core_pair = bool(np.any(visible["sap_core"]) and np.any(visible["tp_core"]))
    uncertain_pair = bool(
        np.any(visible["sap_uncertain"]) and np.any(visible["tp_uncertain"])
    )
    full_pair = bool(np.any(visible["sap"]) and np.any(visible["tp"]))
    if core_pair:
        return "PROXY_POSITIVE", "BOTH_CORE_ENDPOINTS_FIRST_VISIBLE"
    if uncertain_pair:
        return "PROXY_AMBIGUOUS", "UNCERTAINTY_ENDPOINTS_FIRST_VISIBLE"
    if full_pair:
        return "PROXY_AMBIGUOUS", "TP_AND_SAP_VISIBLE_OUTSIDE_TARGET_CORE"
    if any(np.any(projected[name]) for name in ("sap", "sap_core", "sap_uncertain")):
        return "PROXY_AMBIGUOUS", "SAP_IN_PLANE_BUT_NOT_FIRST_VISIBLE"
    if np.any(visible["tp"]):
        return "PROXY_NEGATIVE", "TP_ONLY_WITHOUT_SAP_IN_PLANE"
    if np.any(first):
        return "PROXY_NEGATIVE", "OTHER_BONE_OR_BACKGROUND"
    return "TECHNICALLY_INVALID", "NO_FIRST_BONE_SURFACE"


def _surface(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(3, 1)
    return np.asarray(mask, dtype=bool) & ~ndimage.binary_erosion(
        np.asarray(mask, dtype=bool), structure=structure
    )


def build_target_contract_geometry(
    *, sap_mask: np.ndarray, tp_mask: np.ndarray, affine: np.ndarray
) -> TargetContractGeometry:
    """Build the evaluator-only two-surface region with CT-derived uncertainty."""

    sap = np.asarray(sap_mask, dtype=bool)
    tp = np.asarray(tp_mask, dtype=bool)
    matrix = np.asarray(affine, dtype=np.float64)
    if sap.shape != tp.shape or sap.ndim != 3 or matrix.shape != (4, 4):
        raise ValueError("target masks or affine have incompatible shapes")
    if not np.any(sap) or not np.any(tp) or np.any(sap & tp):
        raise ValueError("target masks must be nonempty and disjoint")
    spacing = np.sqrt(np.sum(matrix[:3, :3] ** 2, axis=0))
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0.0):
        raise ValueError("affine spacing must be finite and positive")

    sap_surface = _surface(sap)
    tp_surface = _surface(tp)
    distance_to_tp = ndimage.distance_transform_edt(~tp_surface, sampling=spacing)
    distance_to_sap = ndimage.distance_transform_edt(~sap_surface, sampling=spacing)
    minimum = float(min(distance_to_tp[sap_surface].min(), distance_to_sap[tp_surface].min()))
    core_limit = minimum + float(np.min(spacing))
    uncertainty_limit = minimum + float(np.linalg.norm(spacing))
    sap_core = sap_surface & (distance_to_tp <= core_limit + 1e-9)
    tp_core = tp_surface & (distance_to_sap <= core_limit + 1e-9)
    sap_uncertain = sap_surface & (distance_to_tp <= uncertainty_limit + 1e-9)
    tp_uncertain = tp_surface & (distance_to_sap <= uncertainty_limit + 1e-9)
    if min(np.count_nonzero(sap_core), np.count_nonzero(tp_core)) == 0:
        raise ValueError("two-surface core is empty")

    sap_index = np.argwhere(sap_core)
    tp_index = np.argwhere(tp_core)
    sap_world = nib.affines.apply_affine(matrix, sap_index)
    tp_world = nib.affines.apply_affine(matrix, tp_index)
    tp_tree = cKDTree(tp_world)
    sap_tree = cKDTree(sap_world)
    midpoints = [
        0.5 * (point + tp_world[index])
        for point, index in zip(sap_world, tp_tree.query(sap_world)[1])
    ]
    midpoints.extend(
        0.5 * (point + sap_world[index])
        for point, index in zip(tp_world, sap_tree.query(tp_world)[1])
    )
    midpoint_world = np.unique(np.round(np.asarray(midpoints), 6), axis=0)
    return TargetContractGeometry(
        sap_surface=sap_surface,
        tp_surface=tp_surface,
        sap_core=sap_core,
        tp_core=tp_core,
        sap_uncertain=sap_uncertain,
        tp_uncertain=tp_uncertain,
        midpoint_world_mm=midpoint_world,
        minimum_surface_separation_mm=minimum,
        core_separation_limit_mm=core_limit,
        uncertainty_separation_limit_mm=uncertainty_limit,
    )


def _proposal_track(
    tracks: Sequence[InterfaceTrack], row: float, column: float, track_id: str
) -> tuple[InterfaceTrack | None, float | None]:
    eligible = [track for track in tracks if track.columns[0] <= column <= track.columns[-1]]
    if not eligible:
        return None, None
    named = [track for track in eligible if track.track_id == track_id]
    pool = named or eligible
    track = min(pool, key=lambda item: abs(item.row_at(column) - row))
    return track, float(abs(track.row_at(column) - row))


def fixed_tp_relative_hypothesis(
    image: np.ndarray,
    *,
    proposal_row: float,
    proposal_column: float,
    proposal_track_id: str,
    column_offset_pixels: float,
    maximum_proposal_track_distance_pixels: float,
) -> TargetViewHypothesis:
    """Place a control point at a fixed offset along the detected TP interface."""

    tracks = track_bright_interfaces(image)
    track, distance = _proposal_track(
        tracks, proposal_row, proposal_column, proposal_track_id
    )
    if track is None or distance is None:
        return TargetViewHypothesis("ABSTAIN", None, None, 0.0, None, "NO_INTERFACE_TRACK")
    if distance > maximum_proposal_track_distance_pixels:
        return TargetViewHypothesis(
            "ABSTAIN", None, None, 0.0, distance, "TP_PROPOSAL_NOT_ON_INTERFACE"
        )
    column = float(proposal_column + column_offset_pixels)
    if column < track.columns[0] or column > track.columns[-1]:
        return TargetViewHypothesis(
            "ABSTAIN", None, None, 0.0, distance, "OFFSET_OUTSIDE_TRACK"
        )
    return TargetViewHypothesis(
        "ACCEPT",
        track.row_at(column),
        column,
        float(track.mean_ridge_score),
        distance,
        "FIXED_TP_RELATIVE_CONTROL",
    )


def image_target_hypothesis(
    image: np.ndarray,
    *,
    proposal_row: float,
    proposal_column: float,
    proposal_track_id: str,
    minimum_column_offset_pixels: int,
    maximum_column_offset_pixels: int,
    smoothing_width_pixels: int,
    minimum_evidence_score: float,
    maximum_proposal_track_distance_pixels: float,
    minimum_image_margin_pixels: int,
) -> TargetViewHypothesis:
    """Use clean-image interface intensity and posterior shadow beyond the TP proposal."""

    array = normalized_image(image)
    tracks = track_bright_interfaces(array)
    track, distance = _proposal_track(
        tracks, proposal_row, proposal_column, proposal_track_id
    )
    if track is None or distance is None:
        return TargetViewHypothesis("ABSTAIN", None, None, 0.0, None, "NO_INTERFACE_TRACK")
    if distance > maximum_proposal_track_distance_pixels:
        return TargetViewHypothesis(
            "ABSTAIN", None, None, 0.0, distance, "TP_PROPOSAL_NOT_ON_INTERFACE"
        )
    lower = int(math.ceil(proposal_column + minimum_column_offset_pixels))
    upper = int(math.floor(proposal_column + maximum_column_offset_pixels))
    lower = max(lower, int(track.columns[0]), minimum_image_margin_pixels)
    upper = min(upper, int(track.columns[-1]), array.shape[1] - 1 - minimum_image_margin_pixels)
    if upper < lower:
        return TargetViewHypothesis(
            "ABSTAIN", None, None, 0.0, distance, "NO_MEDIAL_INTERFACE_SEARCH_BAND"
        )
    columns = np.arange(lower, upper + 1, dtype=np.int64)
    rows = np.rint([track.row_at(float(column)) for column in columns]).astype(np.int64)
    rows = np.clip(rows, 0, array.shape[0] - 1)
    intensity = array[rows, columns]
    shadow = np.zeros(len(columns), dtype=np.float64)
    for index, (row, column) in enumerate(zip(rows, columns)):
        proximal = array[max(0, row - 2) : min(array.shape[0], row + 2), column]
        distal = array[min(array.shape[0], row + 3) : min(array.shape[0], row + 11), column]
        proximal_mean = float(np.mean(proximal)) if proximal.size else 0.0
        distal_mean = float(np.mean(distal)) if distal.size else proximal_mean
        shadow[index] = np.clip(proximal_mean - distal_mean, 0.0, 1.0)
    evidence = 0.55 * intensity + 0.45 * shadow
    width = max(1, int(smoothing_width_pixels))
    kernel = np.ones(width, dtype=np.float64) / width
    smoothed = np.convolve(np.pad(evidence, (width // 2, width - 1 - width // 2), mode="edge"), kernel, mode="valid")
    selected = int(np.argmax(smoothed))
    score = float(smoothed[selected])
    column = float(columns[selected])
    row = float(track.row_at(column))
    if score < minimum_evidence_score:
        return TargetViewHypothesis(
            "ABSTAIN", row, column, score, distance, "INSUFFICIENT_IMAGE_EVIDENCE"
        )
    return TargetViewHypothesis(
        "ACCEPT", row, column, score, distance, "IMAGE_INTERFACE_EVIDENCE"
    )


def projected_coordinate_index(row: float, column: float, *, depth: int, width: int) -> int:
    rounded_row = int(np.clip(round(row), 0, depth - 1))
    rounded_column = int(np.clip(round(column), 0, width - 1))
    return rounded_column * depth + rounded_row


def distance_to_target_region_mm(
    *,
    coordinate_map_index: np.ndarray,
    row: float,
    column: float,
    affine: np.ndarray,
    midpoint_world_mm: np.ndarray,
    depth: int = 200,
    width: int = 150,
) -> float:
    coordinates = np.asarray(coordinate_map_index, dtype=np.float64)
    midpoints = np.asarray(midpoint_world_mm, dtype=np.float64)
    if coordinates.shape != (depth * width, 3) or midpoints.ndim != 2 or midpoints.shape[1] != 3:
        raise ValueError("coordinate map or target region has an invalid shape")
    index = projected_coordinate_index(row, column, depth=depth, width=width)
    point_world = nib.affines.apply_affine(np.asarray(affine, dtype=np.float64), coordinates[index])
    return float(np.min(np.linalg.norm(midpoints - point_world[None, :], axis=1)))
