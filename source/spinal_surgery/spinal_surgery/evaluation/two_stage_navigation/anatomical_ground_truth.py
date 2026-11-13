"""CPU-only geometry helpers for privileged s0010 anatomical annotations."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


S0010_STRUCTURES = {
    25: ("sacrum", "sacrum.stl"),
    26: ("S1", "vertebrae_S1.stl"),
    27: ("L5", "vertebrae_L5.stl"),
    28: ("L4", "vertebrae_L4.stl"),
    29: ("L3", "vertebrae_L3.stl"),
    30: ("L2", "vertebrae_L2.stl"),
    31: ("L1", "vertebrae_L1.stl"),
}
ANNOTATION_STATUSES = frozenset({"PROVISIONAL", "UNRESOLVED", "REJECTED"})
ANNOTATION_TYPES = frozenset({"point", "region", "surface"})
LATERALITIES = frozenset({"LEFT", "RIGHT", "MIDLINE", "BILATERAL", "NONE"})
FORBIDDEN_ACTOR_FIELDS = frozenset(
    {
        "raw_label",
        "segmentation",
        "mesh_identity",
        "target_coordinate",
        "evaluator_score",
        "mbb_region",
    }
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_affine(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(affine, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("affine must be a finite 4x4 matrix")
    homogeneous = np.concatenate([array, np.ones((len(array), 1))], axis=1)
    return (homogeneous @ matrix.T)[:, :3]


def index_to_patient_local_m(points: np.ndarray, spacing_m: float) -> np.ndarray:
    if not math.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be finite and positive")
    return np.asarray(points, dtype=np.float64) * spacing_m


def patient_local_to_world_m(
    points: np.ndarray, rotation: np.ndarray, translation_m: Iterable[float]
) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(tuple(translation_m), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if matrix.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("invalid patient-to-world transform")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-7):
        raise ValueError("rotation must be orthonormal")
    return array @ matrix.T + translation


def point_summary(points: np.ndarray) -> dict[str, Any]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ValueError("points must be a nonempty (N, 3) array")
    return {
        "point_count": int(len(array)),
        "centroid": np.mean(array, axis=0).tolist(),
        "bounds": [np.min(array, axis=0).tolist(), np.max(array, axis=0).tolist()],
    }


def quantile_region(
    vertices: np.ndarray,
    *,
    x_fraction: tuple[float, float] = (0.0, 1.0),
    y_fraction: tuple[float, float] = (0.0, 1.0),
    z_fraction: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(vertices, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ValueError("vertices must be a nonempty (N, 3) array")
    intervals = (x_fraction, y_fraction, z_fraction)
    mask = np.ones(len(array), dtype=bool)
    for axis, interval in enumerate(intervals):
        low, high = interval
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("quantile intervals must lie in [0, 1]")
        values = array[:, axis]
        lower, upper = np.quantile(values, [low, high])
        mask &= (values >= lower) & (values <= upper)
    if not np.any(mask):
        raise ValueError("quantile region selected no vertices")
    return np.flatnonzero(mask), array[mask]


def count_exact_shared_points(first: np.ndarray, second: np.ndarray) -> int:
    """Count unique 3-D points that occur exactly in both point clouds."""

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    for name, array in (("first", first_array), ("second", second_array)):
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"{name} points must have shape (N, 3)")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} points must be finite")
    first_points = {tuple(point) for point in first_array.tolist()}
    second_points = {tuple(point) for point in second_array.tolist()}
    return len(first_points & second_points)


def validate_annotation(annotation: dict[str, Any]) -> None:
    required = {
        "patient_id",
        "anatomy_level",
        "raw_source_label",
        "landmark_name",
        "laterality",
        "annotation_type",
        "coordinate_frame",
        "units",
        "geometry",
        "source_files",
        "generation_method",
        "confidence",
        "positional_uncertainty_mm",
        "supporting_visualizations",
        "limitations",
        "status",
    }
    missing = required - set(annotation)
    if missing:
        raise ValueError(f"annotation is missing fields: {sorted(missing)}")
    if annotation["status"] not in ANNOTATION_STATUSES:
        raise ValueError("invalid annotation status")
    if annotation["annotation_type"] not in ANNOTATION_TYPES:
        raise ValueError("invalid annotation type")
    if annotation["laterality"] not in LATERALITIES:
        raise ValueError("invalid annotation laterality")
    uncertainty = annotation["positional_uncertainty_mm"]
    if uncertainty is not None and (
        not math.isfinite(float(uncertainty)) or float(uncertainty) < 0.0
    ):
        raise ValueError("invalid positional uncertainty")
    if annotation["status"] == "PROVISIONAL" and not annotation["geometry"]:
        raise ValueError("provisional annotations require geometry")
    if (
        annotation["landmark_name"] == "facet_joint_region"
        and annotation["status"] == "PROVISIONAL"
        and annotation["generation_method"] == "closest_candidate_articular_regions"
    ):
        raise ValueError(
            "closest mesh regions alone cannot support a provisional facet joint"
        )


def reject_privileged_actor_payload(payload: dict[str, Any]) -> None:
    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                lowered = str(key).lower()
                if any(token in lowered for token in FORBIDDEN_ACTOR_FIELDS):
                    raise ValueError(
                        "privileged anatomical field in actor payload: "
                        + ".".join((*path, str(key)))
                    )
                visit(child, (*path, str(key)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))

    visit(payload, ())
