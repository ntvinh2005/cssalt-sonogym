"""Simulator-supervised visible cortical-interface labels for fixed-s0010 research.

The functions in this module use privileged simulator anatomy only while
constructing offline labels.  They are deliberately separate from actor-side
inference code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import nibabel as nib
import numpy as np
from scipy import ndimage

from .stage1_interfaces import track_bright_interfaces


BACKGROUND = np.uint8(0)
TP_INTERFACE = np.uint8(1)
SAP_INTERFACE = np.uint8(2)
OTHER_BONE_INTERFACE = np.uint8(3)
IGNORE = np.uint8(255)

LEVEL_SOURCE_LABEL = {"L4": 28, "L5": 27}
TP_SUBREGION_LABEL = 5
SAP_SUBREGION_LABEL = 6


@dataclass(frozen=True)
class VisibleInterfaceResult:
    labels: np.ndarray
    first_bone: np.ndarray
    image_supported: np.ndarray
    projected_tp: np.ndarray
    projected_sap: np.ndarray
    projected_uncertainty: np.ndarray
    metrics: Mapping[str, int | float | bool]


def right_level_masks(
    prediction: np.ndarray,
    segmentation: np.ndarray,
    affine: np.ndarray,
    level: str,
) -> dict[str, np.ndarray]:
    """Return right-sided model TP/SAP masks and one-voxel uncertainty.

    These masks remain provisional simulator labels.  The uncertainty is the
    one-voxel shell around either model class, plus their shared shell.
    """

    if level not in LEVEL_SOURCE_LABEL:
        raise ValueError(f"unsupported level: {level}")
    if prediction.shape != segmentation.shape or prediction.ndim != 3:
        raise ValueError("prediction and source segmentation grids differ")
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("affine must be 4 x 4")
    source_label = LEVEL_SOURCE_LABEL[level]
    level_points = np.argwhere(segmentation == source_label)
    if not len(level_points):
        raise ValueError(f"source segmentation contains no {level}")
    x_world = nib.affines.apply_affine(
        matrix,
        np.column_stack(
            (
                np.arange(prediction.shape[0]),
                np.zeros(prediction.shape[0]),
                np.zeros(prediction.shape[0]),
            )
        ),
    )[:, 0]
    midline = float(np.median(x_world[level_points[:, 0]]))
    right = x_world[:, None, None] > midline
    level_mask = segmentation == source_label
    tp = (prediction == TP_SUBREGION_LABEL) & level_mask & right
    sap = (prediction == SAP_SUBREGION_LABEL) & level_mask & right
    if not np.any(tp) or not np.any(sap) or np.any(tp & sap):
        raise ValueError(f"{level} right TP/SAP masks are empty or overlap")
    structure = ndimage.generate_binary_structure(3, 1)
    shell_tp = ndimage.binary_dilation(tp, structure=structure) & ~ndimage.binary_erosion(
        tp, structure=structure
    )
    shell_sap = ndimage.binary_dilation(sap, structure=structure) & ~ndimage.binary_erosion(
        sap, structure=structure
    )
    return {
        "tp": tp,
        "sap": sap,
        "uncertainty": shell_tp | shell_sap,
    }


def project_mask(
    coordinate_map_index: np.ndarray,
    volume_mask: np.ndarray,
    *,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Project a volume mask using the native width-major coordinate ordering."""

    depth, width = image_shape
    coordinates = np.asarray(coordinate_map_index, dtype=np.float64)
    if coordinates.shape != (depth * width, 3):
        raise ValueError(
            f"coordinate map must have shape {(depth * width, 3)}, got {coordinates.shape}"
        )
    indices = np.trunc(coordinates).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < np.asarray(volume_mask.shape)), axis=1)
    hits = np.zeros(len(indices), dtype=bool)
    valid_indices = indices[valid]
    hits[valid] = volume_mask[
        valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
    ]
    return hits.reshape(width, depth).T


def first_bone_surface(raw_labels: np.ndarray, bone_source_ids: set[int]) -> np.ndarray:
    """Return the first acoustic-bone pixel in every image column."""

    raw = np.asarray(raw_labels)
    if raw.ndim != 2:
        raise ValueError("raw label image must be two-dimensional")
    bone = np.isin(raw, sorted(bone_source_ids))
    first = np.zeros_like(bone, dtype=bool)
    columns = np.flatnonzero(np.any(bone, axis=0))
    if len(columns):
        rows = np.argmax(bone[:, columns], axis=0)
        first[rows, columns] = True
    return first


def image_cortical_support(
    image: np.ndarray,
    *,
    maximum_distance_pixels: float = 3.0,
) -> np.ndarray:
    """Rasterize actor-visible bright-interface tracks into a support band."""

    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("ultrasound image must be two-dimensional")
    tracks = track_bright_interfaces(array)
    track_pixels = np.zeros_like(array, dtype=bool)
    for track in tracks:
        columns = np.asarray(track.columns, dtype=np.int64)
        rows = np.rint(track.rows).astype(np.int64)
        valid = (
            (rows >= 0)
            & (rows < array.shape[0])
            & (columns >= 0)
            & (columns < array.shape[1])
        )
        track_pixels[rows[valid], columns[valid]] = True
    if not np.any(track_pixels):
        return track_pixels
    distance = ndimage.distance_transform_edt(~track_pixels)
    return distance <= float(maximum_distance_pixels)


def visible_interface_labels(
    *,
    image: np.ndarray,
    raw_labels: np.ndarray,
    coordinate_map_index: np.ndarray,
    masks: Mapping[str, np.ndarray],
    bone_source_ids: set[int],
    maximum_track_distance_pixels: float = 3.0,
) -> VisibleInterfaceResult:
    """Construct weak TP/SAP/other visible-interface labels for one frame.

    Geometry establishes the offline anatomical proposal.  A pixel becomes a
    hard interface label only when it is both the first bone hit and supported
    by an interface detected from the clean image.  Occluded or boundary-shell
    evidence is ignored instead of silently becoming background.
    """

    array = np.asarray(image, dtype=np.float32)
    raw = np.asarray(raw_labels)
    if array.shape != raw.shape:
        raise ValueError("ultrasound and raw-label images differ in shape")
    required = {"tp", "sap", "uncertainty"}
    if set(masks) != required:
        raise ValueError(f"expected masks {sorted(required)}, got {sorted(masks)}")
    projected = {
        name: project_mask(coordinate_map_index, mask, image_shape=array.shape)
        for name, mask in masks.items()
    }
    first = first_bone_surface(raw, bone_source_ids)
    supported = image_cortical_support(
        array, maximum_distance_pixels=maximum_track_distance_pixels
    )
    labels = np.full(array.shape, BACKGROUND, dtype=np.uint8)

    projected_identity = projected["tp"] | projected["sap"]
    projected_conflict = projected["tp"] & projected["sap"]
    occluded_identity = projected_identity & ~first
    labels[occluded_identity | projected["uncertainty"]] = IGNORE

    unsupported_first = first & ~supported
    labels[unsupported_first] = IGNORE
    confident_first = first & supported & ~projected["uncertainty"] & ~projected_conflict
    labels[confident_first & projected["tp"]] = TP_INTERFACE
    labels[confident_first & projected["sap"]] = SAP_INTERFACE
    labels[confident_first & ~projected_identity] = OTHER_BONE_INTERFACE
    labels[first & projected_conflict] = IGNORE

    counts = {
        "tp_interface_pixels": int(np.count_nonzero(labels == TP_INTERFACE)),
        "sap_interface_pixels": int(np.count_nonzero(labels == SAP_INTERFACE)),
        "other_bone_interface_pixels": int(
            np.count_nonzero(labels == OTHER_BONE_INTERFACE)
        ),
        "ignored_pixels": int(np.count_nonzero(labels == IGNORE)),
        "first_bone_pixels": int(np.count_nonzero(first)),
        "image_supported_first_bone_pixels": int(np.count_nonzero(first & supported)),
        "projected_tp_pixels": int(np.count_nonzero(projected["tp"])),
        "projected_sap_pixels": int(np.count_nonzero(projected["sap"])),
        "occluded_identity_pixels": int(np.count_nonzero(occluded_identity)),
        "has_tp_interface": bool(np.any(labels == TP_INTERFACE)),
        "has_sap_interface": bool(np.any(labels == SAP_INTERFACE)),
    }
    return VisibleInterfaceResult(
        labels=labels,
        first_bone=first,
        image_supported=supported,
        projected_tp=projected["tp"],
        projected_sap=projected["sap"],
        projected_uncertainty=projected["uncertainty"],
        metrics=counts,
    )
