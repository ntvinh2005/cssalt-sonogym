"""Actor-side motion-compensated spatial cortical mapping.

The map uses clean ultrasound, public image calibration, and the executed
``(surface x, surface z, probe angle)`` feedback.  It deliberately does not
recover the hidden surface normal or use native pixel-to-volume coordinates;
those records belong to the offline evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

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
    maximum_tangent_angle_deg: float = 60.0
    include_depth: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.pixel_resolution_mm,
            self.label_resolution_mm,
            self.sample_stride_pixels,
            self.association_radius_mm,
            self.minimum_view_translation_mm,
            self.minimum_view_angle_deg,
            self.maximum_tangent_angle_deg,
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
    tangent: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class SpatialAssociation:
    left_index: int
    right_index: int
    actor_distance_mm: float


@dataclass(frozen=True)
class SpatialViewCandidate:
    frame_sequence: int
    frame_sha256: str
    target_probability: float
    proposal_row: float
    proposal_column: float


@dataclass(frozen=True)
class SpatialLocalizerConfig:
    hypothesis: Literal["support", "landmark_relative"]
    support_radius_mm: float = 4.0
    minimum_supporting_frames: int = 3
    maximum_support_distance_mm: float = 4.0
    minimum_target_probability: float = 0.5
    target_probability_weight: float = 0.45
    support_weight: float = 0.55
    support_saturation_count: int = 8
    landmark_prior_scale_mm: float = 18.0
    minimum_score: float = 0.45
    minimum_margin: float = 0.02
    top_k: int = 3

    def __post_init__(self) -> None:
        if self.hypothesis not in {"support", "landmark_relative"}:
            raise ValueError("unsupported localizer hypothesis")
        positive = (
            self.support_radius_mm,
            self.minimum_supporting_frames,
            self.maximum_support_distance_mm,
            self.support_saturation_count,
            self.landmark_prior_scale_mm,
            self.top_k,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in positive):
            raise ValueError("localizer settings must be finite and positive")
        if not 0.0 <= self.minimum_target_probability <= 1.0:
            raise ValueError("minimum probability must lie in [0, 1]")
        if not 0.0 <= self.minimum_score <= 1.0 or self.minimum_margin < 0.0:
            raise ValueError("invalid score gates")
        if not math.isclose(
            self.target_probability_weight + self.support_weight, 1.0, abs_tol=1e-9
        ):
            raise ValueError("localizer score weights must sum to one")


@dataclass(frozen=True)
class SpatialRankedView:
    frame_sequence: int
    frame_sha256: str
    score: float
    target_probability: float
    supporting_frames: tuple[int, ...]
    median_support_distance_mm: float
    landmark_prior_distance_mm: float
    eligible: bool


@dataclass(frozen=True)
class SpatialTargetDecision:
    status: Literal["ACCEPT", "ABSTAIN"]
    selected: SpatialRankedView | None
    ranking: tuple[SpatialRankedView, ...]
    reason: str


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


def map_image_pixel(
    *,
    row: float,
    column: float,
    image_shape: tuple[int, int],
    executed_pose_index: Sequence[float],
    config: SpatialMapConfig = SpatialMapConfig(),
) -> np.ndarray:
    """Map one image pixel with the same actor-side approximation as the map."""

    pose = np.asarray(executed_pose_index, dtype=np.float64)
    if pose.shape != (3,) or not np.all(np.isfinite(pose)):
        raise ValueError("executed pose must be a finite x/z/angle triple")
    height, width = image_shape
    if not (0.0 <= row < height and 0.0 <= column < width):
        raise ValueError("pixel lies outside the image")
    origin = pose[:2] * config.label_resolution_mm
    lateral = np.asarray([math.cos(pose[2]), math.sin(pose[2])], dtype=np.float64)
    lateral_mm = (column - width // 2) * config.pixel_resolution_mm
    surface = origin + lateral * lateral_mm
    depth = row * config.pixel_resolution_mm if config.include_depth else 0.0
    return np.asarray([surface[0], surface[1], depth], dtype=np.float64)


def calibrated_image_coordinates_mm(
    *,
    end_effector_position_m: Sequence[float],
    rotation_columns: np.ndarray,
    width: int,
    depth: int,
    pixel_resolution_mm: float,
    image_origin_height_mm: float,
    home_up_axis: Sequence[float] = (0.0, 1.0, 0.0),
) -> np.ndarray:
    """Transform native image pixels with a public probe-pose calibration.

    This function is suitable for robot forward-kinematics pose feedback.  It
    does not read patient anatomy, labels, or native simulator coordinates.
    """

    position_mm = np.asarray(end_effector_position_m, dtype=np.float64) * 1000.0
    rotation = np.asarray(rotation_columns, dtype=np.float64)
    up = np.asarray(home_up_axis, dtype=np.float64)
    if position_mm.shape != (3,) or rotation.shape != (3, 3) or up.shape != (3,):
        raise ValueError("calibrated transform inputs have invalid shape")
    if not (
        np.all(np.isfinite(position_mm))
        and np.all(np.isfinite(rotation))
        and np.all(np.isfinite(up))
    ):
        raise ValueError("calibrated transform inputs must be finite")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("rotation columns must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("rotation columns must be right-handed")
    if width <= 0 or depth <= 0 or pixel_resolution_mm <= 0.0:
        raise ValueError("image dimensions and spacing must be positive")
    normal = rotation[:, 2]
    origin_direction = normal if float(np.dot(normal, up)) >= 0.0 else -normal
    origin = position_mm + image_origin_height_mm * origin_direction
    columns = (np.arange(width, dtype=np.float64) - width // 2) * pixel_resolution_mm
    rows = np.arange(depth, dtype=np.float64) * pixel_resolution_mm
    lateral_grid, depth_grid = np.meshgrid(columns, rows, indexing="xy")
    return (
        origin[None, None, :]
        + lateral_grid[:, :, None] * rotation[:, 0]
        + depth_grid[:, :, None] * rotation[:, 2]
    )


def coordinates_in_target_frame_mm(
    coordinates_mm: np.ndarray,
    *,
    world_to_source_position_m: Sequence[float],
    world_to_source_rotation: np.ndarray,
    world_to_target_position_m: Sequence[float],
    world_to_target_rotation: np.ndarray,
) -> np.ndarray:
    coordinates = np.asarray(coordinates_mm, dtype=np.float64)
    source_position = np.asarray(world_to_source_position_m, dtype=np.float64)
    source_rotation = np.asarray(world_to_source_rotation, dtype=np.float64)
    target_position = np.asarray(world_to_target_position_m, dtype=np.float64)
    target_rotation = np.asarray(world_to_target_rotation, dtype=np.float64)
    if coordinates.shape[-1] != 3:
        raise ValueError("coordinates must end in three spatial values")
    if source_position.shape != (3,) or target_position.shape != (3,):
        raise ValueError("frame positions must have shape (3,)")
    if source_rotation.shape != (3, 3) or target_rotation.shape != (3, 3):
        raise ValueError("frame rotations must have shape (3, 3)")
    world = coordinates @ source_rotation.T + source_position * 1000.0
    return (world - target_position * 1000.0) @ target_rotation


def calibrated_cortical_map_points(
    image: np.ndarray,
    *,
    end_effector_position_m: Sequence[float],
    rotation_columns: np.ndarray,
    frame_sequence: int,
    frame_sha256: str,
    config: SpatialMapConfig = SpatialMapConfig(),
    tracker_config: InterfaceTrackerConfig = InterfaceTrackerConfig(),
    image_origin_height_mm: float = 133.0,
) -> tuple[CorticalMapPoint, ...]:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("image must be a finite two-dimensional array")
    if frame_sequence < 0 or not frame_sha256:
        raise ValueError("frame identity is incomplete")
    coordinates = calibrated_image_coordinates_mm(
        end_effector_position_m=end_effector_position_m,
        rotation_columns=rotation_columns,
        width=array.shape[1],
        depth=array.shape[0],
        pixel_resolution_mm=config.pixel_resolution_mm,
        image_origin_height_mm=image_origin_height_mm,
        home_up_axis=np.asarray(rotation_columns, dtype=np.float64)[:, 2],
    )
    rotation = np.asarray(rotation_columns, dtype=np.float64)
    result: list[CorticalMapPoint] = []
    for track in track_bright_interfaces(array, tracker_config):
        for column in range(
            int(track.columns[0]),
            int(track.columns[-1]) + 1,
            config.sample_stride_pixels,
        ):
            row = int(np.clip(round(track.row_at(float(column))), 0, array.shape[0] - 1))
            lower_column = max(int(track.columns[0]), column - 1)
            upper_column = min(int(track.columns[-1]), column + 1)
            lower_row = float(track.row_at(float(lower_column)))
            upper_row = float(track.row_at(float(upper_column)))
            delta_column = max(upper_column - lower_column, 1)
            row_slope = (upper_row - lower_row) / delta_column
            tangent = rotation[:, 0] + row_slope * rotation[:, 2]
            tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
            coordinate = coordinates[row, column]
            result.append(
                CorticalMapPoint(
                    frame_sequence=frame_sequence,
                    frame_sha256=frame_sha256,
                    track_id=track.track_id,
                    row=row,
                    column=column,
                    coordinate_mm=tuple(float(value) for value in coordinate),
                    confidence=float(track.mean_ridge_score),
                    tangent=tuple(float(value) for value in tangent),
                )
            )
    return tuple(result)


def calibrated_pose_is_distinct(
    left_position_m: Sequence[float],
    left_rotation: np.ndarray,
    right_position_m: Sequence[float],
    right_rotation: np.ndarray,
    config: SpatialMapConfig = SpatialMapConfig(),
) -> bool:
    left_position = np.asarray(left_position_m, dtype=np.float64)
    right_position = np.asarray(right_position_m, dtype=np.float64)
    first_rotation = np.asarray(left_rotation, dtype=np.float64)
    second_rotation = np.asarray(right_rotation, dtype=np.float64)
    if left_position.shape != (3,) or right_position.shape != (3,):
        raise ValueError("positions must have shape (3,)")
    if first_rotation.shape != (3, 3) or second_rotation.shape != (3, 3):
        raise ValueError("rotations must have shape (3, 3)")
    if not all(
        np.all(np.isfinite(value))
        for value in (left_position, right_position, first_rotation, second_rotation)
    ):
        raise ValueError("poses must be finite")
    translation_mm = float(np.linalg.norm(right_position - left_position) * 1000.0)
    relative = first_rotation.T @ second_rotation
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cosine))
    return bool(
        translation_mm >= config.minimum_view_translation_mm
        or angle_deg >= config.minimum_view_angle_deg
    )


def associate_calibrated_spatial_points(
    points: Sequence[CorticalMapPoint],
    *,
    poses_by_frame: dict[int, tuple[Sequence[float], np.ndarray]],
    config: SpatialMapConfig = SpatialMapConfig(),
) -> tuple[SpatialAssociation, ...]:
    hashes_by_frame: dict[int, str] = {}
    grouped: dict[int, list[int]] = {}
    for index, point in enumerate(points):
        prior = hashes_by_frame.setdefault(point.frame_sequence, point.frame_sha256)
        if prior != point.frame_sha256:
            raise ValueError("one frame sequence has multiple image hashes")
        grouped.setdefault(point.frame_sequence, []).append(index)
    if len(set(hashes_by_frame.values())) != len(hashes_by_frame):
        raise ValueError("duplicate image hash cannot create independent map support")
    tangent_cosine = math.cos(math.radians(config.maximum_tangent_angle_deg))
    result: list[SpatialAssociation] = []
    frames = sorted(grouped)
    for frame_offset, left_frame in enumerate(frames):
        for right_frame in frames[frame_offset + 1 :]:
            left_pose = poses_by_frame[left_frame]
            right_pose = poses_by_frame[right_frame]
            if not calibrated_pose_is_distinct(
                left_pose[0], left_pose[1], right_pose[0], right_pose[1], config
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
                if distance > config.association_radius_mm:
                    continue
                first = points[int(left_indices[local_left])]
                second = points[int(right_indices[local_right])]
                if first.tangent is not None and second.tangent is not None:
                    alignment = abs(
                        float(np.dot(np.asarray(first.tangent), np.asarray(second.tangent)))
                    )
                    if alignment < tangent_cosine:
                        continue
                result.append(
                    SpatialAssociation(
                        left_index=int(left_indices[local_left]),
                        right_index=int(right_indices[local_right]),
                        actor_distance_mm=distance,
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


def rank_spatial_target_views(
    points: Sequence[CorticalMapPoint],
    candidates: Sequence[SpatialViewCandidate],
    *,
    poses_by_frame: dict[int, Sequence[float]],
    stage1_prior_pose_index: Sequence[float],
    map_config: SpatialMapConfig = SpatialMapConfig(),
    localizer_config: SpatialLocalizerConfig,
    image_shape: tuple[int, int] = (200, 150),
) -> SpatialTargetDecision:
    """Rank target views from map support and the Stage 1 actor prior."""

    prior = np.asarray(stage1_prior_pose_index, dtype=np.float64)
    if prior.shape != (3,) or not np.all(np.isfinite(prior)):
        raise ValueError("Stage 1 prior must be a finite x/z/angle triple")
    grouped: dict[int, list[CorticalMapPoint]] = {}
    for point in points:
        grouped.setdefault(point.frame_sequence, []).append(point)
    ranking: list[SpatialRankedView] = []
    for candidate in candidates:
        coordinate = map_image_pixel(
            row=candidate.proposal_row,
            column=candidate.proposal_column,
            image_shape=image_shape,
            executed_pose_index=poses_by_frame[candidate.frame_sequence],
            config=map_config,
        )
        nearest_by_frame: list[tuple[float, int]] = []
        for frame_sequence, frame_points in grouped.items():
            if frame_sequence == candidate.frame_sequence:
                continue
            distances = [
                float(np.linalg.norm(coordinate - np.asarray(point.coordinate_mm)))
                for point in frame_points
            ]
            if distances:
                nearest = min(distances)
                if nearest <= localizer_config.support_radius_mm:
                    nearest_by_frame.append((nearest, frame_sequence))
        nearest_by_frame.sort()
        support_distances = np.asarray(
            [value for value, _ in nearest_by_frame], dtype=np.float64
        )
        supporting_frames = tuple(frame for _, frame in nearest_by_frame)
        median_distance = (
            float(np.median(support_distances)) if support_distances.size else math.inf
        )
        support_fraction = min(
            len(supporting_frames) / localizer_config.support_saturation_count, 1.0
        )
        spatial_quality = (
            math.exp(-median_distance / localizer_config.maximum_support_distance_mm)
            if math.isfinite(median_distance)
            else 0.0
        )
        support_score = support_fraction * spatial_quality
        score = (
            localizer_config.target_probability_weight * candidate.target_probability
            + localizer_config.support_weight * support_score
        )
        prior_coordinate = prior[:2] * map_config.label_resolution_mm
        prior_distance = float(np.linalg.norm(coordinate[:2] - prior_coordinate))
        if localizer_config.hypothesis == "landmark_relative":
            score *= math.exp(-prior_distance / localizer_config.landmark_prior_scale_mm)
        eligible = bool(
            candidate.target_probability >= localizer_config.minimum_target_probability
            and len(supporting_frames) >= localizer_config.minimum_supporting_frames
            and median_distance <= localizer_config.maximum_support_distance_mm
        )
        ranking.append(
            SpatialRankedView(
                frame_sequence=candidate.frame_sequence,
                frame_sha256=candidate.frame_sha256,
                score=float(score),
                target_probability=candidate.target_probability,
                supporting_frames=supporting_frames,
                median_support_distance_mm=median_distance,
                landmark_prior_distance_mm=prior_distance,
                eligible=eligible,
            )
        )
    ordered = tuple(
        sorted(
            ranking,
            key=lambda row: (
                not row.eligible,
                -row.score,
                -len(row.supporting_frames),
                row.frame_sequence,
            ),
        )
    )
    eligible = [row for row in ordered if row.eligible]
    if not eligible:
        return SpatialTargetDecision("ABSTAIN", None, ordered, "NO_ELIGIBLE_SPATIAL_VIEW")
    selected = eligible[0]
    second_score = eligible[1].score if len(eligible) > 1 else 0.0
    if selected.score < localizer_config.minimum_score:
        return SpatialTargetDecision("ABSTAIN", None, ordered, "SCORE_BELOW_GATE")
    if selected.score - second_score < localizer_config.minimum_margin:
        return SpatialTargetDecision("ABSTAIN", None, ordered, "INSUFFICIENT_SCORE_MARGIN")
    return SpatialTargetDecision("ACCEPT", selected, ordered, "SUPPORTED_SPATIAL_VIEW")
