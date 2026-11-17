"""CPU reference geometry for SonoGym ultrasound image planes.

The native :class:`LabelImgSlicer` stores image-local coordinates as
``(width, elevation, depth)``.  This module mirrors that convention without
importing Isaac Sim so that review and annotation tools can be tested on CPU.
It does not change the renderer or the task.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImagePlane:
    """A sampled image plane in patient-local volume-index coordinates."""

    coordinates_index: np.ndarray
    origin_index: np.ndarray
    rotation_columns: np.ndarray
    clipped_sample_count: int
    total_sample_count: int


def _rotation(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-7):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(result), 1.0, atol=1e-7):
        raise ValueError("rotation must be right-handed")
    return result


def native_local_coordinates(
    *, width: int, depth: int, resolution_m: float, elevation_m: float = 0.0
) -> np.ndarray:
    """Return ``(depth, width, 3)`` local coordinates in native axis order.

    The last dimension is ``(width x, elevation y, depth z)``.  Array row is
    depth and array column is width.  This is an independently executable CPU
    transcription of the coordinate construction in ``LabelImgSlicer``.
    """

    if width <= 0 or depth <= 0 or not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("width, depth, and resolution must be positive")
    columns = np.arange(width, dtype=np.float64) - width // 2
    rows = np.arange(depth, dtype=np.float64)
    x_grid, z_grid = np.meshgrid(columns, rows, indexing="xy")
    return np.stack(
        [
            x_grid * resolution_m,
            np.full_like(x_grid, float(elevation_m)),
            z_grid * resolution_m,
        ],
        axis=-1,
    )


def plane_from_full_pose(
    *,
    end_effector_position_m: np.ndarray,
    rotation_columns: np.ndarray,
    image_origin_height_m: float,
    local_coordinates_m: np.ndarray,
    label_resolution_m: float,
    volume_shape: tuple[int, int, int] | None = None,
) -> ImagePlane:
    """Mirror the native full-pose image-origin and coordinate transform.

    As in the native slicer, a downward-facing probe flips only the direction
    used to place the image origin.  The supplied rotation is otherwise kept
    unchanged when transforming image-local coordinates.
    """

    position = np.asarray(end_effector_position_m, dtype=np.float64)
    rotation = _rotation(rotation_columns)
    local = np.asarray(local_coordinates_m, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("end-effector position must be a finite length-three vector")
    if local.ndim < 2 or local.shape[-1] != 3 or not np.all(np.isfinite(local)):
        raise ValueError("local coordinates must be finite with final dimension three")
    if not np.isfinite(image_origin_height_m) or not np.isfinite(label_resolution_m):
        raise ValueError("image and label resolutions must be finite")
    if label_resolution_m <= 0:
        raise ValueError("label resolution must be positive")

    normal = rotation[:, 2]
    origin_normal = normal if float(np.dot(normal, [0.0, 1.0, 0.0])) >= 0.0 else -normal
    origin_m = position + float(image_origin_height_m) * origin_normal
    coordinates_index = (origin_m + local @ rotation.T) / label_resolution_m
    clipped = 0
    if volume_shape is not None:
        shape = np.asarray(volume_shape, dtype=np.int64)
        if shape.shape != (3,) or np.any(shape <= 0):
            raise ValueError("volume shape must contain three positive dimensions")
        outside = np.any(
            (coordinates_index < 0.0) | (coordinates_index > (shape - 1)), axis=-1
        )
        clipped = int(np.count_nonzero(outside))
        coordinates_index = np.clip(coordinates_index, 0.0, shape - 1)
    return ImagePlane(
        coordinates_index=coordinates_index,
        origin_index=origin_m / label_resolution_m,
        rotation_columns=rotation,
        clipped_sample_count=clipped,
        total_sample_count=int(np.prod(local.shape[:-1])),
    )


def plane_from_reduced_pose(
    *,
    realized_pose_index: np.ndarray,
    surface_y_index: np.ndarray,
    surface_normals: np.ndarray,
    width: int,
    depth: int,
    image_resolution_m: float,
    label_resolution_m: float,
    probe_height_m: float,
    image_origin_height_m: float,
    volume_shape: tuple[int, int, int] | None = None,
) -> ImagePlane:
    """Approximate the native plane from recorded ``(surface x, z, angle)``.

    This is explicitly reduced-pose reconstruction, not exact pose replay.
    Surface lookup follows the native integer truncation convention.
    """

    pose = np.asarray(realized_pose_index, dtype=np.float64)
    surface_y = np.asarray(surface_y_index, dtype=np.float64)
    normals = np.asarray(surface_normals, dtype=np.float64)
    if pose.shape != (3,) or not np.all(np.isfinite(pose)):
        raise ValueError("realized pose must be a finite length-three vector")
    if surface_y.ndim != 2 or normals.shape != (*surface_y.shape, 3):
        raise ValueError("surface arrays have incompatible shapes")
    ix = int(np.clip(np.trunc(pose[0]), 0, surface_y.shape[0] - 1))
    iz = int(np.clip(np.trunc(pose[1]), 0, surface_y.shape[1] - 1))
    normal = normals[ix, iz]
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm <= 1e-12:
        raise ValueError("surface normal is zero or nonfinite")
    normal = normal / normal_norm
    x_projection = np.asarray([np.cos(pose[2]), 0.0, np.sin(pose[2])])
    y_axis = np.cross(normal, x_projection)
    y_norm = float(np.linalg.norm(y_axis))
    if not np.isfinite(y_norm) or y_norm <= 1e-12:
        raise ValueError("surface normal and in-plane direction are degenerate")
    y_axis /= y_norm
    x_axis = np.cross(y_axis, normal)
    x_axis /= np.linalg.norm(x_axis)
    rotation = np.column_stack([x_axis, y_axis, normal])
    surface_point_m = np.asarray(
        [pose[0], float(surface_y[ix, iz]), pose[1]], dtype=np.float64
    ) * label_resolution_m
    end_effector_m = surface_point_m - normal * float(probe_height_m)
    local = native_local_coordinates(
        width=width, depth=depth, resolution_m=image_resolution_m
    )
    return plane_from_full_pose(
        end_effector_position_m=end_effector_m,
        rotation_columns=rotation,
        image_origin_height_m=image_origin_height_m,
        local_coordinates_m=local,
        label_resolution_m=label_resolution_m,
        volume_shape=volume_shape,
    )


def project_points_to_pixels(
    *,
    points_index: np.ndarray,
    image_origin_index: np.ndarray,
    rotation_columns: np.ndarray,
    width: int,
    depth: int,
    image_resolution_m: float,
    label_resolution_m: float,
    plane_tolerance_mm: float,
) -> dict[str, np.ndarray]:
    """Project patient-local mesh points into native image row/column space."""

    points = np.asarray(points_index, dtype=np.float64)
    origin = np.asarray(image_origin_index, dtype=np.float64)
    rotation = _rotation(rotation_columns)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points must be a finite Nx3 array")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("image origin must be finite and length three")
    local_m = ((points - origin) * label_resolution_m) @ rotation
    columns = np.rint(local_m[:, 0] / image_resolution_m + width // 2).astype(int)
    rows = np.rint(local_m[:, 2] / image_resolution_m).astype(int)
    elevation_distance_mm = np.abs(local_m[:, 1]) * 1000.0
    inside = (
        (columns >= 0)
        & (columns < width)
        & (rows >= 0)
        & (rows < depth)
        & (elevation_distance_mm <= plane_tolerance_mm)
    )
    return {
        "rows": rows,
        "columns": columns,
        "elevation_distance_mm": elevation_distance_mm,
        "inside": inside,
    }
