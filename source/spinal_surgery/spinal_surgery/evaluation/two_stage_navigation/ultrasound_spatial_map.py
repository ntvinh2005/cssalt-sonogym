"""Actor-side motion-compensated spatial cortical mapping.

The map uses clean ultrasound, public image calibration, and the executed
``(surface x, surface z, probe angle)`` feedback.  It deliberately does not
recover the hidden surface normal or use native pixel-to-volume coordinates;
those records belong to the offline evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .stage1_interfaces import InterfaceTrackerConfig, track_bright_interfaces


@dataclass(frozen=True)
class SpatialMapConfig:
    pixel_resolution_mm: float = 0.5
    label_resolution_mm: float = 1.5
    sample_stride_pixels: int = 6
    association_radius_mm: float = 4.0
    minimum_view_translation_mm: float = 0.75
    minimum_view_angle_deg: float = 0.75
    include_depth: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.pixel_resolution_mm,
            self.label_resolution_mm,
            self.sample_stride_pixels,
            self.association_radius_mm,
            self.minimum_view_translation_mm,
            self.minimum_view_angle_deg,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in positive):
            raise ValueError("spatial-map settings must be finite and positive")


@dataclass(frozen=True)
class CorticalMapPoint:
    frame_sequence: int
    frame_sha256: str
    track_id: str
    row: int
    column: int
    coordinate_mm: tuple[float, float, float]
    confidence: float


@dataclass(frozen=True)
class SpatialAssociation:
    left_index: int
    right_index: int
    actor_distance_mm: float


def _wrap_degrees(value: float) -> float:
    return abs(math.degrees((value + math.pi) % (2.0 * math.pi) - math.pi))


def pose_is_distinct(
    left: Sequence[float], right: Sequence[float], config: SpatialMapConfig
) -> bool:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != (3,) or second.shape != (3,) or not np.all(np.isfinite([first, second])):
        raise ValueError("poses must be finite x/z/angle triples")
    translation_mm = float(
        np.linalg.norm(first[:2] - second[:2]) * config.label_resolution_mm
    )
    return bool(
        translation_mm >= config.minimum_view_translation_mm
        or _wrap_degrees(float(first[2] - second[2])) >= config.minimum_view_angle_deg
    )


def cortical_map_points(
    image: np.ndarray,
    *,
    executed_pose_index: Sequence[float],
    frame_sequence: int,
    frame_sha256: str,
    config: SpatialMapConfig = SpatialMapConfig(),
    tracker_config: InterfaceTrackerConfig = InterfaceTrackerConfig(),
) -> tuple[CorticalMapPoint, ...]:
    """Map image-derived interface samples into an actor-owned 2.5D frame."""

    array = np.asarray(image, dtype=np.float64)
    pose = np.asarray(executed_pose_index, dtype=np.float64)
    if array.ndim != 2 or pose.shape != (3,) or not np.all(np.isfinite(pose)):
        raise ValueError("image and executed pose must be finite")
    if frame_sequence < 0 or not frame_sha256:
        raise ValueError("frame identity is incomplete")
    origin = pose[:2] * config.label_resolution_mm
    lateral = np.asarray([math.cos(pose[2]), math.sin(pose[2])], dtype=np.float64)
    center_column = array.shape[1] // 2
    result: list[CorticalMapPoint] = []
    for track in track_bright_interfaces(array, tracker_config):
        for column in range(
            int(track.columns[0]), int(track.columns[-1]) + 1, config.sample_stride_pixels
        ):
            row = int(np.clip(round(track.row_at(float(column))), 0, array.shape[0] - 1))
            lateral_mm = (column - center_column) * config.pixel_resolution_mm
            surface = origin + lateral * lateral_mm
            depth_mm = row * config.pixel_resolution_mm if config.include_depth else 0.0
            result.append(
                CorticalMapPoint(
                    frame_sequence=frame_sequence,
                    frame_sha256=frame_sha256,
                    track_id=track.track_id,
                    row=row,
                    column=column,
                    coordinate_mm=(float(surface[0]), float(surface[1]), float(depth_mm)),
                    confidence=float(track.mean_ridge_score),
                )
            )
    return tuple(result)


def associate_spatial_points(
    points: Sequence[CorticalMapPoint],
    *,
    poses_by_frame: dict[int, Sequence[float]],
    config: SpatialMapConfig = SpatialMapConfig(),
) -> tuple[SpatialAssociation, ...]:
    """Return mutual-nearest associations from physically distinct frames."""

    hashes_by_frame: dict[int, str] = {}
    grouped: dict[int, list[int]] = {}
    for index, point in enumerate(points):
        prior = hashes_by_frame.setdefault(point.frame_sequence, point.frame_sha256)
        if prior != point.frame_sha256:
            raise ValueError("one frame sequence has multiple image hashes")
        grouped.setdefault(point.frame_sequence, []).append(index)
    if len(set(hashes_by_frame.values())) != len(hashes_by_frame):
        raise ValueError("duplicate image hash cannot create independent map support")

    result: list[SpatialAssociation] = []
    frames = sorted(grouped)
    for frame_offset, left_frame in enumerate(frames):
        for right_frame in frames[frame_offset + 1 :]:
            if not pose_is_distinct(
                poses_by_frame[left_frame], poses_by_frame[right_frame], config
            ):
                continue
            left_indices = np.asarray(grouped[left_frame], dtype=np.int64)
            right_indices = np.asarray(grouped[right_frame], dtype=np.int64)
            left = np.asarray([points[index].coordinate_mm for index in left_indices])
            right = np.asarray([points[index].coordinate_mm for index in right_indices])
            distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
            left_best = np.argmin(distances, axis=1)
            right_best = np.argmin(distances, axis=0)
            for local_left, local_right in enumerate(left_best):
                if right_best[local_right] != local_left:
                    continue
                distance = float(distances[local_left, local_right])
                if distance <= config.association_radius_mm:
                    result.append(
                        SpatialAssociation(
                            left_index=int(left_indices[local_left]),
                            right_index=int(right_indices[local_right]),
                            actor_distance_mm=distance,
                        )
                    )
    return tuple(result)


def frame_support_counts(
    points: Sequence[CorticalMapPoint], associations: Sequence[SpatialAssociation]
) -> dict[int, int]:
    """Count distinct supporting frames for each source frame."""

    support: dict[int, set[int]] = {}
    for edge in associations:
        left = points[edge.left_index].frame_sequence
        right = points[edge.right_index].frame_sequence
        support.setdefault(left, set()).add(right)
        support.setdefault(right, set()).add(left)
    return {frame: len(neighbors) for frame, neighbors in support.items()}
